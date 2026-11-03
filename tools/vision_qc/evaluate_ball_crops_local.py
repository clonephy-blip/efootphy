"""Local ball AP measurement without Validator callbacks or dataset downloads."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import cv2,numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['weights','data','out']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--padding',type=int,choices=[0,16],default=0,help='Optional legacy square-validation border; deployed contract uses zero')
    a=p.parse_args()
    if a.out.exists() or not a.weights.is_file():raise ValueError('Local weights and new report required')
    os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
    def no_network(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('No network in local crop evaluation')
    sys.addaudithook(no_network)
    import torch
    from ultralytics import YOLO,settings
    from ultralytics.utils.nms import non_max_suppression
    from ultralytics.utils.metrics import ap_per_class,box_iou
    from .train_ball_local import ball_labels
    settings.update({'sync':False});torch.set_num_threads(2);cv2.setNumThreads(1)
    model=YOLO(str(a.weights)).model.float().eval().fuse(verbose=False)
    images=sorted((a.data/'valid/images').glob('*.jpg'));iou_levels=np.linspace(.5,.95,10);statistics=[];targets=0
    if not images or model.model[-1].nc!=7 or model.names[0]!='ball':raise ValueError('Expected source crop data and seven-channel ball model')
    for start in range(0,len(images),16):
        paths=images[start:start+16];tensors=[];labels=[]
        for path in paths:
            f=cv2.imread(str(path));assert f.shape[:2]==(320,320)
            if a.padding:f=cv2.copyMakeBorder(f,a.padding,a.padding,a.padding,a.padding,cv2.BORDER_CONSTANT,value=(114,114,114))
            tensors.append(np.ascontiguousarray(f[:,:,::-1].transpose(2,0,1),dtype=np.float32)/255.)
            labels.append(ball_labels(a.data/'valid/labels'/(path.stem+'.txt')))
        with torch.inference_mode():raw=model(torch.from_numpy(np.stack(tensors)))[0]
        predictions=non_max_suppression(raw,conf_thres=.001,iou_thres=.7,classes=[0],multi_label=True,agnostic=False,max_det=300,nc=7)
        for truth,pred in zip(labels,predictions):
            pred[:,:4]-=a.padding;pred[:,:4].clamp_(0,320)
            targets+=len(truth);correct=np.zeros((len(pred),10),bool)
            if len(truth) and len(pred):
                xywh=torch.from_numpy(truth[:,1:])*320
                gt=torch.cat([xywh[:,:2]-xywh[:,2:]/2,xywh[:,:2]+xywh[:,2:]/2],1)
                iou=box_iou(gt,pred[:,:4]).cpu().numpy()
                for j,level in enumerate(iou_levels):
                    matches=np.array(np.nonzero(iou>=level)).T
                    if len(matches):
                        if len(matches)>1:
                            matches=matches[np.argsort(iou[matches[:,0],matches[:,1]])[::-1]]
                            matches=matches[np.unique(matches[:,1],return_index=True)[1]]
                            matches=matches[np.unique(matches[:,0],return_index=True)[1]]
                        correct[matches[:,1].astype(int),j]=True
            statistics.append((correct,pred[:,4].cpu().numpy(),pred[:,5].cpu().numpy(),truth[:,0]))
    if targets==0:raise ValueError('AP requires reviewed positive targets')
    tp,conf,cls,target=[np.concatenate([s[i] for s in statistics]) for i in range(4)]
    metrics=ap_per_class(tp,conf,cls,target,plot=False,names={0:'ball'});ap=metrics[5]
    report={'weights_sha256':hashlib.sha256(a.weights.read_bytes()).hexdigest(),'images':len(images),'ball_instances':targets,
            'ap50':float(ap[:,0].mean()),'ap50_95':float(ap.mean()),'precision':float(metrics[2].mean()),'recall':float(metrics[3].mean()),
            'confidence_floor':.001,'nms_iou':.7,'iou_thresholds':iou_levels.tolist(),'batch':16,
            'source_crop_side':320,'input_side':320+2*a.padding,'padding':a.padding,
            'runner':'Local PyTorch inference and AP functions; no Validator callbacks','onnxruntime_imported':'onnxruntime' in sys.modules,
            'python_network_operations_blocked':True,'independent_match_evaluation':False}
    a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
