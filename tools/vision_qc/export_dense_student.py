"""Export the retained dense student, omitting its unused original YOLO pose head."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def make_student(encoder,head):
    from torch import nn
    from .heatmap_pose import decode
    class Student(nn.Module):
        def __init__(self):
            super().__init__();self.layers=nn.ModuleList(list(encoder.model)[:-1]);self.head=head
            self.save=set(encoder.save);self.outputs=tuple(encoder.model[-1].f)

        def features(self,rgb):
            x=rgb;values=[]
            for layer in self.layers:
                if layer.f!=-1:x=values[layer.f] if isinstance(layer.f,int) else [x if j==-1 else values[j] for j in layer.f]
                x=layer(x);values.append(x if layer.i in self.save else None)
            return [values[i] for i in self.outputs]

        def forward(self,rgb):return decode(*self.head(self.features(rgb),rgb),original_size=640)
    return Student().float().eval()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoder',type=Path,required=True);p.add_argument('--head',type=Path,required=True)
    p.add_argument('--validation-images',type=Path,required=True);p.add_argument('--mobile-images',type=Path,required=True)
    p.add_argument('--geometry',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    import cv2
    import torch
    import onnxruntime as ort
    from ultralytics import YOLO
    from .infer_heatmap import load_head
    from .heatmap_pose import decode
    from .evaluate_landmarks import score_points,aggregate
    if a.out.exists():raise ValueError('Use a new output directory')
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.backends.nnpack.enabled=False;torch.backends.mkldnn.enabled=True
    digest=hashlib.sha256(a.encoder.read_bytes()).hexdigest();encoder=YOLO(a.encoder).model.float().eval()
    head=load_head(a.head,digest).float().eval();student=make_student(encoder,head)
    a.out.mkdir(parents=True);model=a.out/'dense-landmarks32.onnx';example=torch.zeros(1,3,320,320)
    with torch.inference_mode():torch.onnx.export(student,example,str(model),input_names=['rgb'],output_names=['keypoints640'],opset_version=17,dynamo=False)
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    cases=[]
    for path in sorted(a.validation_images.glob('*.jpg')):
        row=np.array(list(map(float,(path.parent.parent/'labels'/path.with_suffix('.txt').name).read_text().split())))
        if len(row)!=101:raise ValueError('Expected 32-point validation labels')
        cases.append((path,row[5:].reshape(32,3)*[640,640,1],'source_validation'))
    for frame in json.loads(a.geometry.read_text())['frames']:
        target=np.zeros((32,3))
        for point in frame['points']:
            if point['status']=='manual_reference':target[point['id']]=[*point['manual_xy_px'],2]
        cases.append((a.mobile_images/frame['source_file'],target,'sparse_mobile_three'))
    captured=[];hook=encoder.model[-1].register_forward_pre_hook(lambda _,args:captured.append(list(args[0])))
    records=[]
    try:
        for path,target,scope in cases:
            frame=cv2.imread(str(path))
            if frame is None:raise ValueError('Image failed to decode')
            height,width=frame.shape[:2];rgb=cv2.resize(frame,(320,320))[:,:,::-1]
            array=np.ascontiguousarray(rgb.transpose(2,0,1)[None],dtype=np.float32)/255;tensor=torch.from_numpy(array)
            with torch.inference_mode():
                encoder(tensor);original_features=captured.pop();expected=decode(*head(original_features,tensor)).numpy()
                cut_features=student.features(tensor);cut=student(tensor).numpy()
            feature_error=max(float((x-y).abs().max()) for x,y in zip(original_features,cut_features))
            session.run(None,{'rgb':array});times=[]
            for _ in range(5):
                started=time.perf_counter();actual=session.run(None,{'rgb':array})[0];times.append((time.perf_counter()-started)*1000)
            difference=np.abs(actual-expected);scale=np.array([width/640,height/640,1])
            legacy_points=score_points(target,expected[0]*scale,True,width,height)
            onnx_points=score_points(target,actual[0]*scale,True,width,height)
            records.append({'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'scope':scope,
                            'feature_cut_max_abs':feature_error,'torch_cut_max_abs':float(np.max(np.abs(cut-expected))),
                            'onnx_coordinate_max_abs_640px':float(difference[...,:2].max()),'onnx_score_max_abs':float(difference[...,2].max()),
                            'ort_cpu_forward_ms_median':float(np.median(times)),'legacy_points':legacy_points,'onnx_points':onnx_points})
    finally:hook.remove()
    summaries={scope:{backend:aggregate([q for r in records if r['scope']==scope for q in r[backend+'_points']])
                      for backend in ['legacy','onnx']} for scope in ['source_validation','sparse_mobile_three']}
    parity=all(r['feature_cut_max_abs']<=1e-6 and r['torch_cut_max_abs']<=1e-6 and r['onnx_coordinate_max_abs_640px']<=.02 and r['onnx_score_max_abs']<=1e-5 for r in records)
    pck_equal=all(summaries[s]['legacy']['pck']==summaries[s]['onnx']['pck'] for s in summaries)
    report={'schema_version':1,'profile':'dense_landmarks32_research','encoder_sha256':digest,
            'head_sha256':hashlib.sha256(a.head.read_bytes()).hexdigest(),'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
            'bytes':model.stat().st_size,'input':{'name':'rgb','shape':[1,3,320,320],'dtype':'float32','range':[0,1],'color':'RGB','resize':'direct'},
            'output':{'name':'keypoints640','shape':[1,32,3],'coordinate_space':'640x640 reference pixels; scale x by original_width/640, y by original_height/640'},
            'omits_original_pose_head':True,'includes_local_soft_argmax':True,'parity_passed':parity,'pck_unchanged':pck_equal,
            'summaries':summaries,'records':records,'ort_threads':1,'torch_threads':2,
            'timing_scope':'CPU forward including local soft argmax; excludes image decode/resize/capture/render; not phone FPS',
            'pitch_scene_gate':False,'android_import_supported':False,'models_validated':False,'release_eligible':False,
            'warning':'faithful export of the retained research baseline; limited source accuracy; not a deployment approval'}
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'bytes':report['bytes'],'parity_passed':parity,'pck_unchanged':pck_equal,'summaries':summaries},indent=2))
    if not parity or not pck_equal:raise RuntimeError('Dense student parity/metric preservation failed')


if __name__=='__main__':main()
