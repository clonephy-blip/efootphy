"""Controlled replay from reviewed stills; this is NOT a gameplay-video test."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .pitch_tracking import PitchTracker,SiftMatcher,XFeatMatcher,project


def replay_frame(base,hud_mask,index):
    """Known image warp + fixed HUD + moving occluders + two deliberate losses."""
    import cv2
    height,width=base.shape[:2]
    angle=.65*math.sin(index/17);zoom=1+.022*math.sin(index/22)
    A=cv2.getRotationMatrix2D((width/2,height/2),angle,zoom)
    A[:,2]+=[width*.025*math.sin(index/16),height*.016*math.sin(index/13)]
    warp=np.vstack([A,[.000003*math.sin(index/23),.000004*math.sin(index/19),1.]])
    frame=cv2.warpPerspective(base,warp,(width,height),borderMode=cv2.BORDER_REFLECT101)
    for j in range(5):
        x=int(width*(.25+.55*((index*.009+j*.17)%1)));y=int(height*(.35+.065*j))
        cv2.rectangle(frame,(x,y),(x+int(width*.009),y+int(height*.045)),(15,30,220),-1)
    valid=index not in {20,21,22,60,61,62,63,64}
    if 20<=index<=22:frame[:]=(35,115,40)
    if 60<=index<=64:
        frame[:]=(50,25,15);cv2.putText(frame,'CONTROLLED MENU CUT',(width//5,height//2),cv2.FONT_HERSHEY_SIMPLEX,2,(240,240,240),3)
    frame[hud_mask==0]=base[hud_mask==0]
    return frame,warp,valid


def hud_pixels(base,spec):
    import cv2
    mask=np.full(base.shape[:2],255,np.uint8)
    for x,y,w,h in spec.get('rects',[]):cv2.rectangle(mask,(int(x),int(y)),(int(x+w),int(y+h)),0,-1)
    for x,y,r in spec.get('circles',[]):cv2.circle(mask,(int(x),int(y)),int(r),0,-1)
    return mask


def summarize(records):
    good=[r for r in records if r['index']>0 and r['valid_input']]
    bad=[r for r in records if not r['valid_input']]
    accepted=[r for r in good if r['status']=='TRACKING']
    times=[r['elapsed_ms'] for r in records if r['index']>0]
    errors=[r['registration_median_px'] for r in accepted]
    recovery=[]
    for start in [23,65]:
        found=next((r['index'] for r in records if r['index']>=start and r['status']=='TRACKING'),None)
        recovery.append({'return_frame':start,'reacquired_frame':found,'delay_frames':None if found is None else found-start})
    return {'eligible_frames':len(good),'accepted_frames':len(accepted),'coverage':len(accepted)/len(good),
            'false_tracking_on_bad_frames':sum(r['status']=='TRACKING' for r in bad),'bad_frames':len(bad),
            'accepted_registration_median_px':float(np.median(errors)) if errors else None,
            'accepted_registration_p95_px':float(np.percentile(errors,95)) if errors else None,
            'cpu_ms_median':float(np.median(times)),'cpu_ms_p95':float(np.percentile(times,95)),
            'recovery':recovery}


def draw_panel(frame,observation,lines,label,error,width=768,hud_spec=None):
    import cv2
    height=int(round(frame.shape[0]*width/frame.shape[1]/2))*2
    panel=cv2.resize(frame,(width,height));clean=panel.copy();T=np.diag([width/frame.shape[1],height/frame.shape[0],1.])
    if observation.homography is not None:
        for line in lines:
            try:xy=project(T@observation.homography,line)
            except ValueError:continue
            for a,b in zip(xy,xy[1:]):
                if np.max(np.abs([a,b]))>1e6:continue
                ok,a,b=cv2.clipLine((0,0,width,height),tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)))
                if ok:cv2.line(panel,a,b,(255,220,15),1,cv2.LINE_AA)
    if hud_spec:
        hud=cv2.resize(hud_pixels(frame,hud_spec),(width,height),interpolation=cv2.INTER_NEAREST)
        panel[hud==0]=clean[hud==0]
    cv2.rectangle(panel,(0,0),(width,40),(15,20,25),-1)
    text=f'{label} | {observation.status} | {observation.source}'
    if error is not None:text+=f' | drift {error:.2f}px'
    cv2.putText(panel,text,(8,26),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,235,235),1,cv2.LINE_AA)
    return panel


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--geometry',required=True,type=Path);p.add_argument('--images',required=True,type=Path)
    p.add_argument('--seed-report',type=Path,help='Optional cached PnL report; calibration cost is recorded separately')
    p.add_argument('--xfeat-repository',required=True,type=Path);p.add_argument('--xfeat-weights',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path);p.add_argument('--frames',type=int,default=80)
    a=p.parse_args()
    if a.frames<80:raise ValueError('At least 80 frames are required for both recovery checks')
    if a.out.exists():raise ValueError('Use a new output directory')
    import cv2
    import torch
    cv2.setNumThreads(1);torch.set_num_threads(2);torch.backends.nnpack.enabled=False;torch.backends.mkldnn.enabled=True
    geometry=json.loads(a.geometry.read_text());lines=[x['world_xy_m_assumed'] for x in geometry['topology']]
    seeds={Path(x['image']).name:x for x in json.loads(a.seed_report.read_text())['records']} if a.seed_report else {}
    a.out.mkdir(parents=True);results=[]
    xfeat=XFeatMatcher(a.xfeat_repository,a.xfeat_weights)
    for f in geometry['frames']:
        image=a.images/f['source_file'];base=cv2.imread(str(image))
        if base is None:raise ValueError('Image failed to decode')
        height,width=base.shape[:2];hud=hud_pixels(base,f['hud_mask'])
        seed=seeds.get(image.name);H=np.asarray(seed['H_world_to_image'] if seed else f['H_world_to_image'])
        seed_source='pnl_teacher' if seed else 'manual_reference'
        grid=np.float32([[x*width,y*height] for x in np.linspace(.15,.85,7) for y in np.linspace(.25,.8,5)])
        methods={'lk':None,'sift_lk':SiftMatcher(),'xfeat_lk':xfeat};frame_panels={};method_results={}
        for name,matcher in methods.items():
            tracker=PitchTracker(matcher=matcher,world_lines=lines);records=[];calls_before=matcher.calls if matcher else 0
            for index in range(a.frames):
                # Keep the seed frame identical to the initializer input.
                frame,warp,valid=(base.copy(),np.eye(3),True) if index==0 else replay_frame(base,hud,index)
                started=time.perf_counter()
                obs=tracker.seed(frame,H,0,f['hud_mask'],seed_source) if index==0 else tracker.update(frame,index*1000/30)
                elapsed=(time.perf_counter()-started)*1000
                error=None;tail=None
                if obs.homography is not None:
                    distances=np.linalg.norm(project(obs.homography@np.linalg.inv(H),grid)-project(warp,grid),axis=1)
                    error=float(np.median(distances));tail=float(np.percentile(distances,95))
                records.append({'index':index,'valid_input':valid,'status':obs.status,'source':obs.source,
                                'H_world_to_image':obs.homography.tolist() if obs.homography is not None else None,
                                'registration_median_px':error,'registration_p95_px':tail,'elapsed_ms':elapsed,'metrics':obs.metrics})
                frame_panels.setdefault(index,[]).append(draw_panel(frame,obs,lines,name,error,hud_spec=f['hud_mask']))
            method_results[name]={'summary':summarize(records),'feature_extractions':(matcher.calls-calls_before) if matcher else 0,'frames':records}
            print(json.dumps({'camera':f['index'],'method':name,**method_results[name]['summary']}),flush=True)
        # Encoding/rendering is excluded from per-frame timing. The runner measures these separately.
        sample=np.vstack(frame_panels[0]);video=a.out/f"controlled-camera-{f['index']}.mp4"
        writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(sample.shape[1],sample.shape[0]))
        if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
        for index in range(a.frames):writer.write(np.vstack(frame_panels[index]))
        writer.release();cv2.imwrite(str(a.out/f"camera-{f['index']}-frame-35.jpg"),np.vstack(frame_panels[35]))
        results.append({'camera':f['index'],'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                        'seed_source':seed_source,'seed_teacher_ms':seed.get('total_ms') if seed else None,'methods':method_results,'preview':video.name})
        report={'schema_version':1,'scope':'controlled projective replay from three stills; NOT continuous gameplay or Android FPS',
                'fps_timestamps':30,'tracking_width':640,'torch_threads':2,'opencv_threads':1,
                'metric_definition':'drift against known image warp; excludes absolute seed calibration error',
                'hud':'manually supplied masks, fixed pixels retained during image warps',
                'teacher_latency_excluded_from_tracker':True,'models_validated':False,'results':results}
        (a.out/'report.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
