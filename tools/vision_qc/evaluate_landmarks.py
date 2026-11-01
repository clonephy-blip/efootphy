"""Pixel-space landmark QC; missing detections remain misses in PCK denominators."""
import argparse
import hashlib
import json
import math
import time
import os
from pathlib import Path

from .dataset_qc import EXTENSIONS, split_directory
from .inspect_frames import iou, letterbox


def score_points(target, predicted, detected, width, height, point_threshold=.5):
    if len(target) != 32 or len(predicted) != 32:
        raise ValueError('Expected 32 keypoints')
    records=[]
    for index,(gt,pred) in enumerate(zip(target,predicted)):
        if gt[2] != 2:
            continue
        finite=all(math.isfinite(x) for x in pred)
        inside=finite and 0<=pred[0]<=width and 0<=pred[1]<=height
        confident=bool(detected and inside and point_threshold<=pred[2]<=1)
        error=math.hypot(gt[0]-pred[0],gt[1]-pred[1]) if detected and finite else None
        records.append({'id':index,'error_px':error,'confident_in_image':confident,
            'pck':{str(r):bool(confident and error<=r) for r in [3,5,10]},
            'unfiltered_pck':{str(r):bool(error is not None and error<=r) for r in [3,5,10]}})
    return records


def aggregate(records):
    values=[x['error_px'] for x in records if x['error_px'] is not None]
    def percentile(q):
        if not values:return None
        ordered=sorted(values);p=(len(ordered)-1)*q;lo=math.floor(p);hi=math.ceil(p)
        return ordered[lo]+(ordered[hi]-ordered[lo])*(p-lo)
    n=len(records)
    return {'visible_ground_truth_points':n,
        'confident_in_image_points':sum(x['confident_in_image'] for x in records),
        'coverage':sum(x['confident_in_image'] for x in records)/n if n else None,
        'pck':{str(r):sum(x['pck'][str(r)] for x in records)/n if n else None for r in [3,5,10]},
        'unfiltered_pck':{str(r):sum(x['unfiltered_pck'][str(r)] for x in records)/n if n else None for r in [3,5,10]},
        'coordinate_error_on_detected_objects':{'count':len(values),'median_px':percentile(.5),'p95_px':percentile(.95)}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--bundle',type=Path)
    source.add_argument('--checkpoint',type=Path,help='FP32 pose checkpoint; same metrics as ONNX evaluation')
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--split',choices=['train','val','test'],default='val')
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    import cv2
    import numpy as np
    import onnxruntime as ort
    import yaml
    from PIL import Image,ImageDraw
    cv2.setNumThreads(1)
    if args.out.exists():raise ValueError('Use a new output directory')
    if args.bundle:
        manifest=json.loads((args.bundle/'manifest.json').read_text());model=args.bundle/'model.onnx'
        digest=hashlib.sha256(model.read_bytes()).hexdigest()
        if digest!=manifest['model_sha256'] or manifest['profile']!='landmarks' or manifest['kpt_shape']!=[32,3]:
            raise ValueError('Model hash/profile/keypoint schema mismatch')
        size=manifest['input_size'];backend='ONNX Runtime CPU 1 thread'
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
        input_name=session.get_inputs()[0].name
        def forward(tensor):return session.run(None,{input_name:tensor})[0]
    else:
        os.environ['ATEN_CPU_CAPABILITY']='default'
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(1);torch.backends.nnpack.enabled=False;torch.backends.mkldnn.enabled=True
        loaded=YOLO(args.checkpoint);net=loaded.model.eval()
        if loaded.task!='pose' or getattr(net.model[-1],'kpt_shape',None)!=[32,3] or list(loaded.names.values())!=['pitch']:
            raise ValueError('Expected one-class 32-point pitch checkpoint')
        digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest();size=640;backend='PyTorch CPU 1 thread'
        def forward(tensor):
            with torch.inference_mode():raw=net(torch.from_numpy(tensor))
            if isinstance(raw,(tuple,list)):raw=raw[0]
            return raw.detach().cpu().numpy()
    config=yaml.safe_load(args.data.read_text())
    folder=split_directory(args.data.resolve().parent,config[args.split])
    files=sorted(p for p in folder.rglob('*') if p.suffix.lower() in EXTENSIONS)
    if not files:raise ValueError('No evaluation images')
    forward(np.zeros((1,3,size,size),np.float32))
    args.out.mkdir(parents=True)
    all_points=[];images=[];times=[]
    for path in files:
        frame=cv2.imread(str(path))
        if frame is None:raise ValueError('Image failed to decode')
        height,width=frame.shape[:2]
        label=folder.parent/'labels'/path.relative_to(folder).with_suffix('.txt')
        rows=[x for x in label.read_text().splitlines() if x.strip()]
        if len(rows)!=1:raise ValueError('Expected exactly one pitch per image')
        row=list(map(float,rows[0].split()))
        if len(row)!=101:raise ValueError('Bad keypoint label')
        target=np.asarray(row[5:]).reshape(32,3)*[width,height,1]
        cx,cy,bw,bh=np.asarray(row[1:5])*[width,height,width,height]
        target_box=[cx-bw/2,cy-bh/2,cx+bw/2,cy+bh/2]
        tensor,transform=letterbox(frame,size)
        started=time.perf_counter();raw=forward(tensor);times.append((time.perf_counter()-started)*1000)
        expected_anchors=sum((size//s)**2 for s in [8,16,32])
        if raw.shape!=(1,101,expected_anchors) or not np.isfinite(raw).all():
            raise ValueError('Unexpected/non-finite landmark output')
        prediction=raw[0,:,int(raw[0,4].argmax())]
        scale,left,top,_,_=transform
        cx,cy,bw,bh=map(float,prediction[:4]);score=float(prediction[4])
        box=[(cx-bw/2-left)/scale,(cy-bh/2-top)/scale,(cx+bw/2-left)/scale,(cy+bh/2-top)/scale]
        overlap=float(iou(box,target_box));detected=bool(.25<=score<=1 and overlap>=.5)
        predicted=prediction[5:].reshape(32,3).copy()
        predicted[:,0]=(predicted[:,0]-left)/scale;predicted[:,1]=(predicted[:,1]-top)/scale
        points=score_points(target,predicted,detected,width,height)
        all_points.extend(points)
        preview=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(preview)
        for k in points:
            index=k['id'];gx,gy,_=map(float,target[index]);px,py,_=map(float,predicted[index])
            draw.ellipse((gx-3,gy-3,gx+3,gy+3),outline='yellow',width=2)
            draw.text((gx+3,gy+3),str(index),fill='yellow')
            if detected and -width<=px<=2*width and -height<=py<=2*height:
                draw.line((gx,gy,px,py),fill=(230,80,80),width=1)
                draw.ellipse((px-3,py-3,px+3,py+3),outline='cyan',width=2)
        filename=f'frame-{len(images):03d}.jpg';preview.save(args.out/filename)
        images.append({'image':path.relative_to(args.data.resolve().parent).as_posix(),
            'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'object_score':score,
            'box_iou':overlap,'object_matched':detected,'points':points,'summary':aggregate(points),'preview':filename})
    report={'schema_version':1,'model_sha256':digest,'backend':backend,'split':args.split,'images':len(images),
        'scope':'source split only; not independent Mobile/match evaluation',
        'object_selection':'highest pitch-score anchor, score>=0.25 and GT bbox IoU>=0.5',
        'point_threshold':.5,'visibility_scored':2,'coordinate_space':'original source image pixels',
        'summary':aggregate(all_points),'per_keypoint':{str(i):aggregate([x for x in all_points if x['id']==i]) for i in range(32)},
        'cpu_forward_ms':{'median':float(np.median(times)),'p95':float(np.percentile(times,95)),
                          'scope':backend+', forward only; not Android latency'},
        'records':images,'models_validated':False,'camera_calibration_enabled':False}
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    canvas=Image.new('RGB',(1280,665*math.ceil(len(images)/2)),(15,20,25));draw=ImageDraw.Draw(canvas)
    for i,item in enumerate(images):
        x=(i%2)*640;y=(i//2)*665
        with Image.open(args.out/item['preview']) as im:canvas.paste(im.resize((640,640)),(x,y+25))
        draw.text((x+5,y+5),f"{i} | GT yellow / prediction cyan | PCK5 {item['summary']['pck']['5']:.3f}",fill='white')
    canvas.save(args.out/'contact.jpg')
    print(json.dumps({k:report[k] for k in ['images','summary','cpu_forward_ms']},indent=2))


if __name__=='__main__':main()
