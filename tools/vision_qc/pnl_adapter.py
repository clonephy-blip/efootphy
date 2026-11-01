"""Optional offline PnLCalib teacher adapter, using a pinned external checkout.

Weights/code remain external. This module never imports their detector into Android.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

# Explicit semantic map: source32 -> PnLCalib's 1-based 57-point grid.
# Circle IDs 30/31 are provisional: ellipse extrema and ground-axis points differ.
SOURCE_TO_PNL=(1,4,8,20,24,28,9,21,45,5,31,34,25,2,32,51,29,6,33,36,26,57,10,22,3,7,11,23,27,30,50,52)


def map_source_points(raw57,width,height):
    raw=np.asarray(raw57,dtype=float).reshape(57,3)
    result=raw[np.asarray(SOURCE_TO_PNL)-1].copy();result[:,:2]*=[width/960,height/540]
    return result


class PnlTeacher:
    def __init__(self,repository,keypoint_weights,line_weights=None,threads=4):
        import torch
        import yaml
        root=Path(repository).resolve();sys.path.insert(0,str(root))
        from model.cls_hrnet import get_cls_net
        from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool,get_keypoints_from_heatmap_batch_maxpool_l
        from utils.utils_heatmap import coords_to_dict,complete_keypoints
        from utils.utils_calib import FramebyFrameCalib
        torch.set_num_threads(threads);torch.backends.nnpack.enabled=False;torch.backends.mkldnn.enabled=True
        self.torch=torch;self.decode=get_keypoints_from_heatmap_batch_maxpool;self.decode_lines=get_keypoints_from_heatmap_batch_maxpool_l
        self.to_dict=coords_to_dict;self.complete=complete_keypoints;self.calibrator=FramebyFrameCalib
        self.net=get_cls_net(yaml.safe_load((root/'config/hrnetv2_w48.yaml').read_text()))
        self.net.load_state_dict(torch.load(keypoint_weights,map_location='cpu',weights_only=True));self.net.float().eval()
        self.line_net=None
        if line_weights:
            from model.cls_hrnet_l import get_cls_net as make_lines
            self.line_net=make_lines(yaml.safe_load((root/'config/hrnetv2_w48_l.yaml').read_text()))
            self.line_net.load_state_dict(torch.load(line_weights,map_location='cpu',weights_only=True));self.line_net.float().eval()
        self.metadata={'keypoint_weights_sha256':hashlib.sha256(Path(keypoint_weights).read_bytes()).hexdigest(),
                       'line_weights_sha256':hashlib.sha256(Path(line_weights).read_bytes()).hexdigest() if line_weights else None,
                       'code_source':'https://github.com/mguti97/PnLCalib','scope':'CPU offline teacher; no device real-time claim'}

    def predict(self,frame,calibrate=True):
        import cv2
        height,width=frame.shape[:2];image=cv2.resize(frame,(960,540))
        tensor=self.torch.from_numpy(np.ascontiguousarray(image[:,:,::-1].transpose(2,0,1)))[None].float()/255
        started=time.perf_counter()
        with self.torch.no_grad():heat=self.net(tensor);line_heat=self.line_net(tensor) if self.line_net else None
        forward_ms=(time.perf_counter()-started)*1000;points=self.decode(heat[:,:-1]);raw57=points[0,:,0].numpy()
        predicted=map_source_points(raw57,width,height);result={'raw57':raw57.tolist(),'prediction32':predicted.tolist(),
                    'forward_ms':forward_ms,'source_mapping_provisional_ids':[30,31],'H_world_to_image':None,
                    'metric_scale_verified':False,'models_validated':False}
        if calibrate and line_heat is not None:
            lines=self.decode_lines(line_heat[:,:-1]);kp=self.to_dict(points,threshold=.3434)[0]
            ln=self.to_dict(lines,threshold=.7867)[0]
            completed,completed_lines=self.complete(kp,ln,w=960,h=540,normalize=True)
            camera=self.calibrator(iwidth=width,iheight=height,denormalize=True);camera.update(completed,completed_lines)
            # Upstream ground-plane voting; the returned matrix maps image -> centred field coordinates.
            chosen=camera.heuristic_voting_ground(refine_lines=True)
            result['detected_keypoints']=kp;result['detected_lines']=ln
            if chosen and chosen.get('homography') is not None:
                image_to_centred=np.asarray(chosen['homography'],float)
                translation=np.array([[1,0,52.5],[0,1,34],[0,0,1.]])
                image_to_world=translation@image_to_centred
                H=np.linalg.inv(image_to_world)
                if np.isfinite(H).all():result['H_world_to_image']=(H/H[2,2]).tolist()
                result['upstream_reprojection_error']=float(chosen['rep_err'])
        result['total_ms']=(time.perf_counter()-started)*1000
        return result


def main():
    import argparse
    import os
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--line-weights',type=Path);p.add_argument('--images',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();os.environ['ATEN_CPU_CAPABILITY']='default'
    import cv2
    cv2.setNumThreads(1)
    if a.out.exists():raise ValueError('Use a new output directory')
    a.out.mkdir(parents=True);teacher=PnlTeacher(a.repository,a.weights,a.line_weights);records=[]
    for image in a.images:
        frame=cv2.imread(str(image))
        if frame is None:raise ValueError('Cannot decode input')
        pred=teacher.predict(frame);records.append({'image':str(image),'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),**pred})
        (a.out/'report.json').write_text(json.dumps({'teacher':teacher.metadata,'records':records},indent=2))
        print(json.dumps({'image':image.name,'auto_calibration':pred['H_world_to_image'] is not None,'forward_ms':pred['forward_ms'],'total_ms':pred['total_ms']}),flush=True)


if __name__=='__main__':main()
