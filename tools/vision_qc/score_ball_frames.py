"""Frozen-threshold full-frame scoring; missing balls remain in denominators."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .ball_clip_data import validate_ball_review
from .ball_video import NativeBallDetector


def score_predictions(records,threshold=.25,tolerance_px=10.):
    if not 0<threshold<=1 or tolerance_px<=0:raise ValueError('Invalid scoring limits')
    visible=accepted=correct=oracle=absent=negative_fp=ignored=0;errors=[]
    for r in records:
        ds=sorted([d for d in r['detections'] if d['score']>=threshold],key=lambda d:-d['score'])
        if r['visibility'] not in ('visible','absent'):ignored+=1;continue
        if r['visibility']=='absent':absent+=1;negative_fp+=int(bool(ds));continue
        visible+=1;distances=[float(np.linalg.norm(np.array(d['xy_px'])-r['reference_xy_px'])) for d in ds];oracle+=int(bool(distances) and min(distances)<=tolerance_px)
        if distances:accepted+=1;errors.append(distances[0]);correct+=int(distances[0]<=tolerance_px)
    return {'reviewed_visible_frames':visible,'reviewed_absent_frames':absent,'ignored_frames':ignored,'frames_with_top1_detection':accepted,'top1_correct_within_tolerance':correct,'any_candidate_correct_within_tolerance':oracle,'negative_frames_with_false_detection':negative_fp,'top1_recall_on_reviewed_visible_frames':correct/visible if visible else None,'top1_precision_on_reviewed_visible_frames':correct/accepted if accepted else None,'median_top1_error_px_including_wrong_objects':float(np.median(errors)) if errors else None,'threshold':threshold,'tolerance_px':tolerance_px,'match_independent':False}


def score_tracking_records(records,tolerance_px=10.):
    if tolerance_px<=0:raise ValueError('Positive tolerance required')
    previous=None;previous_end=None;run=None;gap=0.;visible=observed=correct=wrong=0
    for r in records:
        t,end=r['pts_ms'],r['frame_end_pts_ms']
        if not np.isfinite([t,end]).all() or end<=t or(previous is not None and t<=previous):raise ValueError('True source intervals required')
        if previous_end is not None and abs(t-previous_end)>1e-4:run=None
        previous,previous_end=t,end
        if r['visibility']!='visible':run=None;continue
        visible+=1;p=r['ball']['xy_px'];reference=r['reference_xy_px']
        if not np.isfinite(reference).all():raise ValueError('Finite reference required')
        error=None if p is None else float(np.linalg.norm(np.array(p)-reference));hit=error is not None and np.isfinite(error) and error<=tolerance_px;obs=r['ball']['state']=='OBSERVED'
        observed+=int(hit and obs);correct+=int(hit);wrong+=int(obs and not hit)
        if hit:run=None
        else:
            if run is None:run=t
            gap=max(gap,end-run)
    return {'source_frames':len(records),'reviewed_visible_frames':visible,'uncertain_reference_frames_excluded':len(records)-visible,'correct_observed_within_tolerance':observed,'correct_position_including_short_prediction':correct,'wrong_observed_on_visible_frames':wrong,'longest_visible_missing_position_interval_ms':gap,'tolerance_px':tolerance_px,'gap_method':'True source intervals; split at unknown references or missing source intervals','independent_match_evaluation':False}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['plan','reviews','model','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--excluded-rects',type=Path);p.add_argument('--split',choices=['train','val'],default='val');p.add_argument('--threshold',type=float,default=.25);p.add_argument('--backend',choices=['onnxruntime','opencv'],default='onnxruntime');p.add_argument('--refine-reacquire',action='store_true');a=p.parse_args()
    if a.out.exists():raise ValueError('New score output required')
    plan=json.loads(a.plan.read_text());rows=json.loads(a.reviews.read_text());splits={s['video_id']:s['split'] for s in plan['sources']}
    cv2.setNumThreads(1)
    if a.refine_reacquire:
        from .reacquire_ball import CoarseToFineBallDetector
        d=CoarseToFineBallDetector(a.model,backend=a.backend,acquire=a.threshold)
    else:d=NativeBallDetector(a.model,backend=a.backend)
    masks=[] if not a.excluded_rects else json.loads(a.excluded_rects.read_text());records=[]
    for r in rows:
        if splits[r['video_id']]!=a.split:continue
        if r.get('reviewed') is not True or r['visibility'] not in ('visible','absent'):records.append({'id':r['id'],'visibility':'uncertain','detections':[]});continue
        validate_ball_review(r,Path(r['image_path']).read_bytes(),plan);f=cv2.imread(r['image_path']);start=time.perf_counter();ds,tiles=d.scan(f,None,masks);box=r['ball_box_xyxy_px'];reference=None if box is None else((np.array(box)[:2]+box[2:])/2).tolist();records.append({'id':r['id'],'visibility':r['visibility'],'reference_xy_px':reference,'detections':ds,'tiles':tiles,'forward_scan_ms':(time.perf_counter()-start)*1000,'source_frame_index':r['source_frame_index'],'pts_ms':r['pts_ms']})
    report={'model_sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),'inference_backend':a.backend,'refined_reacquisition_enabled':a.refine_reacquire,'reviews_sha256':hashlib.sha256(a.reviews.read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256(a.plan.read_bytes()).hexdigest(),'scoring':score_predictions(records,a.threshold),'records':records,'crop_selection_uses_ground_truth':False,'predictions_are_new_labels':False,'android_fps_validated':False,'independent_match_evaluation':False};a.out.write_text(json.dumps(report,indent=2));print(json.dumps(report['scoring']))
