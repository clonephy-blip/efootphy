"""Offline video runner: timestamped pitch tracking, LOST suppression and QC log.

The supplied calibration must refer to --seed-frame in this exact video.
This synchronous research runner is not the Android capture worker.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .benchmark_tracking import draw_panel
from .pitch_tracking import PitchTracker,SiftMatcher,XFeatMatcher


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',required=True,type=Path);p.add_argument('--calibration',required=True,type=Path)
    p.add_argument('--seed-frame',type=int,default=0);p.add_argument('--max-frames',type=int,default=3000)
    p.add_argument('--matcher',choices=['lk','sift','xfeat'],default='sift')
    p.add_argument('--xfeat-repository',type=Path);p.add_argument('--xfeat-weights',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.seed_frame<0 or a.max_frames<1:raise ValueError('Invalid frame range')
    if a.matcher=='xfeat' and not (a.xfeat_repository and a.xfeat_weights):raise ValueError('XFeat requires reviewed local code and weights')
    if a.out.exists():raise ValueError('Use a new output directory')
    import cv2
    cv2.setNumThreads(1)
    if a.matcher=='xfeat':
        import torch
        torch.set_num_threads(2);torch.backends.nnpack.enabled=False;torch.backends.mkldnn.enabled=True
    matcher=XFeatMatcher(a.xfeat_repository,a.xfeat_weights) if a.matcher=='xfeat' else SiftMatcher() if a.matcher=='sift' else None
    calibration=json.loads(a.calibration.read_text());lines=calibration.get('world_lines',[])
    tracker=PitchTracker(matcher=matcher,world_lines=lines)
    capture=cv2.VideoCapture(str(a.video));fps=capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not np.isfinite(fps) or fps<=0:raise ValueError('Cannot read a video with valid FPS')
    a.out.mkdir(parents=True);writer=None;times=[];counts={'TRACKING':0,'LOST':0};index=0
    try:
        with (a.out/'frames.jsonl').open('w') as log:
            while index<a.seed_frame+a.max_frames:
                ok,frame=capture.read()
                if not ok:break
                if index<a.seed_frame:index+=1;continue
                timestamp=index*1000/fps;started=time.perf_counter()
                obs=tracker.seed(frame,calibration['H_world_to_image'],timestamp,calibration.get('hud_mask'),
                                 calibration.get('source','external_calibration')) if index==a.seed_frame else tracker.update(frame,timestamp)
                elapsed=(time.perf_counter()-started)*1000;times.append(elapsed);counts[obs.status]+=1
                log.write(json.dumps({'frame':index,'timestamp_ms':timestamp,'status':obs.status,'source':obs.source,
                                      'H_world_to_image':obs.homography.tolist() if obs.homography is not None else None,
                                      'tracking_ms':elapsed,'metrics':obs.metrics})+'\n')
                panel=draw_panel(frame,obs,lines,a.matcher,None,width=960,hud_spec=calibration.get('hud_mask'))
                if writer is None:
                    writer=cv2.VideoWriter(str(a.out/'tracked.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(panel.shape[1],panel.shape[0]))
                    if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
                writer.write(panel);index+=1
    finally:
        capture.release()
        if writer is not None:writer.release()
    if not times:raise ValueError('No frames read at the requested seed position')
    report={'scope':'offline file replay; not device FPS','input_fps':fps,'frames':len(times),'status_counts':counts,
            'cpu_tracking_ms':{'median':float(np.median(times)),'p95':float(np.percentile(times,95))},
            'timing_excludes_decode_render_encode':True,'metric_scale_verified':False,'models_validated':False}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
