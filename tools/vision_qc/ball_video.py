"""Causal, every-decoded-frame ball replay; not an Android real-time claim."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .ball_vision import BallObservation, CausalBallTrack, ball_flow_observation, control_marker_candidates
from .mainview_video import prepare_frame


def decode_ball(raw, transform, origin=(0,0), threshold=.15):
    raw=np.asarray(raw)
    if raw.shape!=(1,11,2100) or not np.isfinite(raw).all():
        raise ValueError('Expected finite seven-class 320px YOLO output')
    if not 0<threshold<=1:raise ValueError('Invalid threshold')
    if len(transform)==5:
        left,top,ratio,w,h=transform;rx=ry=ratio
    elif len(transform)==6:
        left,top,rx,ry,w,h=transform
    else:raise ValueError('Wrong pixel transform')
    values=raw[0].T
    selected=values[values[:,4]>=threshold];out=[]
    for row in selected:
        cx,cy=(row[:2]-[left,top])/[rx,ry]
        if not (0<=cx<w and 0<=cy<h):continue
        bw,bh=row[2:4]/[rx,ry]
        if min(bw,bh)<=0:continue
        out.append({'xy_px':[float(cx+origin[0]),float(cy+origin[1])],
                    'score':float(row[4]),'box_wh_px':[float(bw),float(bh)]})
    # Suppress repeated anchors; preserve the strongest physical-ball candidates.
    kept=[]
    for d in sorted(out,key=lambda x:-x['score']):
        if all(np.linalg.norm(np.array(d['xy_px'])-x['xy_px'])>max(5.,min(d['box_wh_px'])/2) for x in kept):
            kept.append(d)
    return kept[:16]


class NativeBallDetector:
    def __init__(self, model, source_stretch=False, backend='onnxruntime'):
        self.backend=backend
        if backend=='opencv':
            # This path never imports ONNX Runtime. It reads a local graph and
            # uses the CPU engine, with the same preprocessing/decoder contract.
            if hasattr(cv2.dnn,'ENGINE_CLASSIC'):
                self.net=cv2.dnn.readNetFromONNX(str(model),cv2.dnn.ENGINE_CLASSIC)
            else:self.net=cv2.dnn.readNetFromONNX(str(model))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            if self.forward(np.zeros((1,3,320,320),np.float32)).shape!=(1,11,2100):raise ValueError('Wrong ball output contract')
        elif backend=='onnxruntime':
            import onnxruntime as ort
            ort.disable_telemetry_events()
            options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
            self.session=ort.InferenceSession(str(model),sess_options=options,providers=['CPUExecutionProvider'])
            if self.session.get_inputs()[0].shape!=[1,3,320,320] or self.session.get_outputs()[0].shape!=[1,11,2100]:raise ValueError('Wrong ball model tensor contract')
            self.input_name=self.session.get_inputs()[0].name
        else:raise ValueError('Unknown inference backend')
        self.source_stretch=source_stretch

    def forward(self,tensor):
        if tensor.shape!=(1,3,320,320) or tensor.dtype!=np.float32 or not np.isfinite(tensor).all():raise ValueError('Finite RGB FP32 320px tensor required')
        if self.backend=='opencv':
            self.net.setInput(tensor);output=self.net.forward()
        else:output=self.session.run(None,{self.input_name:tensor})[0]
        if output.shape!=(1,11,2100) or not np.isfinite(output).all():raise ValueError('Wrong ball output contract')
        return output

    def detect_crop(self, frame, rect, threshold=.15):
        x0,y0,x1,y1=rect;h,w=frame.shape[:2]
        if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Bad crop')
        crop=frame[y0:y1,x0:x1]
        if self.source_stretch:
            tensor=np.ascontiguousarray(cv2.resize(crop,(320,320))[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.
            tr=(0,0,320/(x1-x0),320/(y1-y0),x1-x0,y1-y0)
        else:
            tensor,tr=prepare_frame(crop,320)
        return decode_ball(self.forward(tensor),tr,(x0,y0),threshold)

    def scan(self, frame, predicted=None, excluded_rects=()):
        h,w=frame.shape[:2]
        if self.source_stretch:
            # Explicit experimental adapter for a stretched training export.
            # 16:9 is an assumption, not a recovered original camera calibration.
            tw,th=min(640,w),min(360,h)
            if predicted is None:
                xs=np.linspace(0,w-tw,max(1,int(np.ceil(w/tw))))
                ys=np.linspace(0,h-th,max(1,int(np.ceil(h/th))))
                rects=[(int(x),int(y),int(x)+tw,int(y)+th) for y in ys for x in xs]
            else:
                x=int(np.clip(round(predicted[0]-tw/2),0,w-tw));y=int(np.clip(round(predicted[1]-th/2),0,h-th))
                rects=[(x,y,x+tw,y+th)]
        elif predicted is None:
            # Coarse reacquisition. Runtime still processes the current source frame.
            side=min(640,w,h)
            xs=np.linspace(0,w-side,max(1,int(np.ceil(w/side))))
            ys=np.linspace(0,h-side,max(1,int(np.ceil(h/side))))
            rects=[(int(x),int(y),int(x)+side,int(y)+side) for y in ys for x in xs]
        else:
            side=min(320,w,h);x,y=np.rint(np.array(predicted)-side/2).astype(int)
            x=int(np.clip(x,0,w-side));y=int(np.clip(y,0,h-side));rects=[(x,y,x+side,y+side)]
        detections=[]
        for rect in rects:
            detections.extend(self.detect_crop(frame,rect))
        detections=[d for d in detections if not any(x0<=d['xy_px'][0]<x1 and y0<=d['xy_px'][1]<y1 for x0,y0,x1,y1 in excluded_rects)]
        return detections,len(rects)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--start-ms',type=float,default=0)
    p.add_argument('--end-ms',type=float,default=11500);p.add_argument('--max-frames',type=int,default=800)
    p.add_argument('--excluded-rects',type=Path);p.add_argument('--acquire',type=float,default=.25)
    p.add_argument('--maintain',type=float,default=.15)
    p.add_argument('--flow',action='store_true',help='Current-image LK candidates between detector refreshes')
    p.add_argument('--detector-interval-ms',type=float,default=0.,help='Only ball flow still runs on each source frame')
    p.add_argument('--source-stretch',action='store_true',help='Experimental 16:9 crop-to-square source adapter')
    p.add_argument('--backend',choices=['onnxruntime','opencv'],default='onnxruntime')
    p.add_argument('--refine-reacquire',action='store_true',help='Experimental same-frame native confirmation of weak coarse candidates')
    a=p.parse_args()
    if a.out.exists() or not 0<=a.start_ms<a.end_ms or not 1<=a.max_frames<=2000 or not 0<=a.detector_interval_ms<=250:
        raise ValueError('New output and bounded replay required')
    rects=[] if a.excluded_rects is None else json.loads(a.excluded_rects.read_text())
    if a.detector_interval_ms and not a.flow:raise ValueError('Skipped neural updates require the per-frame flow branch')
    if a.refine_reacquire:
        if a.source_stretch:raise ValueError('Refinement requires the native geometry adapter')
        from .reacquire_ball import CoarseToFineBallDetector
        detector=CoarseToFineBallDetector(a.model,backend=a.backend,acquire=a.acquire)
    else:detector=NativeBallDetector(a.model,a.source_stretch,a.backend)
    track=CausalBallTrack(a.acquire,a.maintain)
    cv2.setNumThreads(1);capture=cv2.VideoCapture(str(a.video));a.out.mkdir(parents=True)
    records=[];previous=-1.;predicted=None;last_thumb=None;index=-1
    previous_gray=None;flow_xy=None;last_detector_pts=-float('inf')
    journal=(a.out/'frames.jsonl').open('x')
    try:
        while True:
            started=time.perf_counter();ok,frame=capture.read();index+=1
            decode_ms=(time.perf_counter()-started)*1000
            if not ok:break
            pts=float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if pts<a.start_ms:continue
            if pts>a.end_ms:break
            if len(records)>=a.max_frames:raise ValueError('Source frame budget exceeded')
            if pts<=previous:raise ValueError('Non-increasing source PTS')
            previous=pts;started=time.perf_counter()
            thumb=cv2.resize(frame,(64,36)).astype(np.float32)
            cut=bool(last_thumb is not None and np.abs(thumb-last_thumb).mean()>40.)
            if cut:predicted=None
            last_thumb=thumb
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            flow=ball_flow_observation(previous_gray,gray,flow_xy) if a.flow and not cut else None
            refresh=(pts-last_detector_pts>=a.detector_interval_ms or predicted is None or (a.flow and flow is None))
            if refresh:
                detections,tiles=detector.scan(frame,predicted,rects);last_detector_pts=pts
            else:detections,tiles=[],0
            observations=[BallObservation(*d['xy_px'],d['score']) for d in detections]
            # Prefer a detector observation that is spatially consistent with image flow.
            if flow is not None and not any(np.linalg.norm(np.array([o.x,o.y])-[flow.x,flow.y])<20 for o in observations):
                observations.append(flow)
            state=track.update(pts,observations,scene_cut=cut)
            predicted=state['xy_px']
            flow_xy=state['xy_px'] if state['state']=='OBSERVED' else None
            previous_gray=gray
            markers=control_marker_candidates(frame)
            record={'source_frame_index':index,'pts_ms':pts,'decode_ms':decode_ms,
                    'ball_pipeline_ms':(time.perf_counter()-started)*1000,'tiles':tiles,
                    'ball':state,'ball_candidates':detections,'control_marker_candidates':markers,
                    'detector_refreshed':refresh,'flow_candidate':None if flow is None else {'xy_px':[flow.x,flow.y],'quality':flow.score},
                    'scene_cut_candidate':cut,'ground_truth_reviewed':False}
            records.append(record)
            journal.write(json.dumps(record)+'\n');journal.flush()
            panel=cv2.resize(frame,(1170,540))
            xy=state['xy_px'];colour=(0,255,0) if state['state']=='OBSERVED' else (0,180,255)
            if xy is not None:
                pt=tuple(np.rint(np.array(xy)/2).astype(int));cv2.circle(panel,pt,12,colour,2)
                cv2.putText(panel,state['state'],(pt[0]+14,pt[1]),0,.45,colour,1)
                x,y=np.rint(xy).astype(int);h,w=frame.shape[:2]
                x=int(np.clip(x-48,0,w-96));y=int(np.clip(y-48,0,h-96))
                inset=cv2.resize(frame[y:y+96,x:x+96],(192,192));panel[65:257,960:1152]=inset
            cv2.putText(panel,f"OFFLINE every source frame | {pts/1000:.3f}s | {state['state']} | unverified",(12,25),0,.58,(255,255,255),1,cv2.LINE_AA)
            cv2.imwrite(str(a.out/f'frame-{len(records)-1:04d}.jpg'),panel,[cv2.IMWRITE_JPEG_QUALITY,90])
    finally:
        capture.release();journal.close()
    if not records:raise ValueError('No frames in requested interval')
    times=np.array([x['ball_pipeline_ms'] for x in records[1:]])
    counts={s:sum(r['ball']['state']==s for r in records) for s in ['OBSERVED','PREDICTED','LOST']}
    report={'source_video_sha256':hashlib.sha256(a.video.read_bytes()).hexdigest(),
            'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'frames':len(records),
            'source_start_pts_ms':records[0]['pts_ms'],'source_end_pts_ms':records[-1]['pts_ms'],
            'ball_pipeline_ms_median':float(np.median(times)) if len(times) else None,
            'ball_pipeline_ms_p95':float(np.percentile(times,95)) if len(times) else None,
            'status_counts_NOT_accuracy':counts,'records':records,'uses_future_frames':False,
            'source_frames_skipped_within_interval':0,'ground_truth_reviewed':False,
            'flow_enabled':a.flow,'detector_interval_ms':a.detector_interval_ms,
            'source_stretch_adapter':a.source_stretch,
            'inference_backend':a.backend,
            'refined_reacquisition_enabled':a.refine_reacquire,
            'android_real_time_verified':False,'model_is_temporal':False,'models_validated':False}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    # VFR replay preserves original source intervals instead of inventing a 2Hz rate.
    lines=[]
    for i,r in enumerate(records):
        duration=(records[i+1]['pts_ms']-r['pts_ms'])/1000 if i+1<len(records) else (records[-1]['pts_ms']-records[-2]['pts_ms'])/1000 if len(records)>1 else 1/60
        lines.extend([f"file 'frame-{i:04d}.jpg'",'option framerate 1000',f'duration {duration:.9f}'])
    (a.out/'frames.ffconcat').write_text('ffconcat version 1.0\n'+'\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))


if __name__=='__main__':main()
