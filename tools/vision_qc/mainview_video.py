"""Bounded ONNX full-view video diagnostic, not an accuracy or Android benchmark."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np


FIELD_CLASSES=['ball','cursor','cursor_secondary','goalkeeper','player','powerbar','referee']


def prepare_frame(frame,size=640):
    if frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3:raise ValueError('uint8 BGR frame required')
    h,w=frame.shape[:2];r=min(size/w,size/h);rw,rh=round(w*r),round(h*r)
    left,top=(size-rw)//2,(size-rh)//2
    canvas=np.full((size,size,3),114,np.uint8);canvas[top:top+rh,left:left+rw]=cv2.resize(frame,(rw,rh))
    tensor=np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.
    return tensor,(left,top,r,w,h)


def decode_field(output,transform,threshold=.35,iou_threshold=.45):
    raw=np.asarray(output)
    if raw.shape!=(1,11,8400) or not np.isfinite(raw).all():raise ValueError('Unexpected/nonfinite field output')
    if not 0<threshold<=1 or not 0<iou_threshold<1:raise ValueError('Invalid thresholds')
    values=raw[0].T;labels=values[:,4:].argmax(1);scores=values[np.arange(len(values)),labels+4]
    keep=scores>=threshold;values,labels,scores=values[keep],labels[keep],scores[keep]
    boxes=np.c_[values[:,:2]-values[:,2:4]/2,values[:,:2]+values[:,2:4]/2]
    left,top,r,width,height=transform
    boxes=(boxes-np.array([left,top,left,top]))/r
    boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,width);boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,height)
    detections=[]
    for label in range(7):
        order=np.flatnonzero((labels==label)&(boxes[:,2]>boxes[:,0])&(boxes[:,3]>boxes[:,1]))
        order=order[np.argsort(-scores[order],kind='stable')]
        while len(order):
            i=int(order[0]);box=boxes[i]
            detections.append({'class_id':label,'class_name':FIELD_CLASSES[label],'score':float(scores[i]),
                               'box_xyxy_px':box.tolist(),'player_identity':None})
            rest=order[1:]
            if not len(rest):break
            a=boxes[rest];intersection=np.maximum(0,np.minimum(a[:,2:],box[2:])-np.maximum(a[:,:2],box[:2])).prod(1)
            union=(a[:,2:]-a[:,:2]).prod(1)+(box[2:]-box[:2]).prod()-intersection
            order=rest[intersection/np.maximum(union,1e-12)<=iou_threshold]
    return detections


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--times-ms',type=Path,required=True,help='JSON array of diagnostic sample PTS targets')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--max-samples',type=int,default=48)
    args=p.parse_args()
    if args.out.exists():raise ValueError('Use a new output directory')
    requested=json.loads(args.times_ms.read_text())
    if not isinstance(requested,list) or not 1<=len(requested)<=args.max_samples or args.max_samples>120:
        raise ValueError('Sample budget exceeded')
    if any(not isinstance(t,(float,int)) or not np.isfinite(t) or t<0 for t in requested):raise ValueError('Invalid sample timestamps')
    if any(b<=a for a,b in zip(requested,requested[1:])):raise ValueError('Samples must increase')
    manifest=json.loads((args.bundle/'manifest.json').read_text());model=args.bundle/'model.onnx'
    digest=hashlib.sha256(model.read_bytes()).hexdigest()
    if manifest['profile']!='field' or manifest['classes']!=FIELD_CLASSES or digest!=manifest['model_sha256']:
        raise ValueError('Field model provenance mismatch')
    import onnxruntime as ort
    cv2.setNumThreads(1);options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
    if session.get_inputs()[0].shape!=[1,3,640,640] or session.get_outputs()[0].shape!=[1,11,8400]:
        raise ValueError('Incompatible field tensor contract')
    args.out.mkdir(parents=True);capture=cv2.VideoCapture(str(args.video));records=[];panels=[];previous=-1.
    try:
        for i,timestamp in enumerate(requested):
            capture.set(cv2.CAP_PROP_POS_MSEC,float(timestamp));started=time.perf_counter();ok,frame=capture.read()
            decode_ms=(time.perf_counter()-started)*1000
            if not ok:raise ValueError('Cannot decode requested frame')
            pts=float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if not np.isfinite(pts) or pts<=previous:raise ValueError('Decoder did not provide increasing PTS')
            previous=pts;started=time.perf_counter();tensor,transform=prepare_frame(frame)
            preprocess_ms=(time.perf_counter()-started)*1000;started=time.perf_counter()
            raw=session.run(None,{session.get_inputs()[0].name:tensor})[0]
            forward_ms=(time.perf_counter()-started)*1000;started=time.perf_counter();detections=decode_field(raw,transform)
            postprocess_ms=(time.perf_counter()-started)*1000
            # Overlay always labels output as candidates, never real roster identity.
            panel=frame.copy()
            for d in detections:
                x0,y0,x1,y1=np.rint(d['box_xyxy_px']).astype(int)
                color=(30,220,255) if d['class_id']==0 else (255,220,0)
                cv2.rectangle(panel,(x0,y0),(x1,y1),color,2)
                cv2.putText(panel,f"{d['class_name']} {d['score']:.2f}",(x0,max(15,y0-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
            ww=1170;panel=cv2.resize(panel,(ww,round(frame.shape[0]*ww/frame.shape[1])));banner=np.full((44,ww,3),20,np.uint8)
            counts=dict(Counter(d['class_name'] for d in detections))
            cv2.putText(banner,f"OFFLINE full-view candidates | source PTS {pts/1000:.3f}s | ball={counts.get('ball',0)} player={counts.get('player',0)}",(10,29),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1,cv2.LINE_AA)
            panel=np.vstack([banner,panel]);cv2.imwrite(str(args.out/f'frame-{i:03d}.jpg'),panel,[cv2.IMWRITE_JPEG_QUALITY,92])
            if i in {0,len(requested)//4,len(requested)//2,len(requested)-1}:panels.append(cv2.resize(panel,(780,round(panel.shape[0]*780/panel.shape[1]))))
            records.append({'sample_index':i,'requested_ms':timestamp,'decoded_pts_ms':pts,'counts':counts,
                'decode_ms':decode_ms,'preprocess_ms':preprocess_ms,'forward_ms':forward_ms,
                'postprocess_ms':postprocess_ms,'detections':detections,'scene_verified':False})
    finally:capture.release()
    while len(panels)<4:panels.append(panels[-1].copy())
    cv2.imwrite(str(args.out/'contact-sheet.jpg'),np.vstack([np.hstack(panels[:2]),np.hstack(panels[2:4])]),[cv2.IMWRITE_JPEG_QUALITY,94])
    elapsed=[r['forward_ms'] for r in records[1:]]
    report={'source_video':args.video.name,'source_sha256':hashlib.sha256(args.video.read_bytes()).hexdigest(),
        'model_sha256':digest,'threshold':.35,'records':records,'samples':len(records),
        'forward_ms_median_excluding_first':float(np.median(elapsed)) if elapsed else None,
        'forward_ms_p95_excluding_first':float(np.percentile(elapsed,95)) if elapsed else None,
        'scope':'Offline sampled frames, no labelled accuracy, no scene/action/identity ground truth, not Android FPS',
        'models_validated':False,'automatic_training_import_allowed':False,'new_weights_trained':False}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['samples','forward_ms_median_excluding_first','forward_ms_p95_excluding_first','scope']}))


if __name__=='__main__':main()
