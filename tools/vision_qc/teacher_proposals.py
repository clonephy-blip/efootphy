"""Train-only, two-view teacher proposals. Automatic proposals are never GT."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .pitch_tracking import project

# Physical straight-line junctions only. Circle/spot semantics remain uncertain
# between the source annotations, manual Mobile template and external teacher.
JUNCTION_IDS=(0,1,2,3,4,5,6,7,9,12,13,16,17,20,22,23,24,25,26,27,28,29)


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gate_points(frame,known,prediction,aligned_second,world,reference_ids=None):
    """All pixel thresholds are fixed in 640-source space, independent of PCK."""
    import cv2
    cv2.setRNGSeed(20261101)
    known=np.asarray(known,float);first=np.asarray(prediction,float);second=np.asarray(aligned_second,float);world=np.asarray(world,float)
    if known.shape!=(32,3) or first.shape!=(32,3) or second.shape!=(32,3) or world.shape!=(32,2):raise ValueError('Expected the 32-point schema')
    if frame.shape[:2]!=(640,640):raise ValueError('Gate thresholds require 640-source pixels')
    reference_ids=tuple(range(30)) if reference_ids is None else tuple(reference_ids)
    selected=np.array([i for i in reference_ids if known[i,2]==2],int);result=np.zeros((32,3));details=[]
    if len(selected)<6:return result,{'reason':'too_few_reference_points','points':details}
    H,mask=cv2.findHomography(world[selected],known[selected,:2],cv2.RANSAC,3.,maxIters=2000,confidence=.999)
    if H is None or mask is None:return result,{'reason':'no_reference_homography','points':details}
    inliers=mask.ravel().astype(bool);reference_errors=np.linalg.norm(project(H,world[selected])-known[selected,:2],axis=1)
    coverage=cv2.contourArea(cv2.convexHull(known[selected,:2].astype(np.float32)))/(640*640)
    if inliers.sum()<6 or inliers.mean()<.85 or np.percentile(reference_errors,95)>3 or coverage<.01:
        return result,{'reason':'weak_reference_geometry','points':details,'inliers':int(inliers.sum()),
                       'reference_p95_px':float(np.percentile(reference_errors,95)),'coverage':float(coverage)}
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);white=cv2.inRange(hsv,np.array([0,0,135]),np.array([179,95,255]))
    distance=cv2.distanceTransform(255-white,cv2.DIST_L2,3);green=cv2.inRange(hsv,np.array([25,30,25]),np.array([100,255,255]))
    expected=project(H,world);mean=(first[:,:2]+second[:,:2])/2
    for i,(a,b,xy,expected_xy) in enumerate(zip(first,second,mean,expected)):
        reason=None
        if i not in reference_ids:reason='provisional_semantics'
        elif not np.isfinite([*a,*b]).all():reason='nonfinite'
        elif min(a[2],b[2])<.6 or max(a[2],b[2])>1:reason='low_confidence'
        elif np.linalg.norm(a[:2]-b[:2])>3:reason='two_view_disagreement'
        elif not (4<=xy[0]<636 and 76<=xy[1]<548):reason='border_or_hud_band'
        elif np.linalg.norm(xy-expected_xy)>4:reason='reference_projection_disagreement'
        elif known[i,2]==2 and np.linalg.norm(xy-known[i,:2])>3:reason='known_label_disagreement'
        else:
            x,y=np.rint(xy).astype(int)
            if distance[y,x]>2:reason='no_white_line_support'
            elif np.mean(green[y-4:y+5,x-4:x+5]>0)<.25:reason='no_surrounding_pitch'
        if reason is None:result[i]=[*xy,2]
        details.append({'id':i,'accepted':reason is None,'reason':reason,'new_point':bool(known[i,2]!=2),
                        'min_confidence':float(min(a[2],b[2])) if np.isfinite([a[2],b[2]]).all() else None})
    return result,{'reason':'geometry_checked','inliers':int(inliers.sum()),'reference_p95_px':float(np.percentile(reference_errors,95)),
                  'coverage':float(coverage),'H_world_to_640':H.tolist(),'points':details}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',required=True,type=Path);p.add_argument('--field-root',required=True,type=Path)
    p.add_argument('--pose-root',required=True,type=Path);p.add_argument('--geometry',required=True,type=Path)
    p.add_argument('--repository',required=True,type=Path);p.add_argument('--weights',required=True,type=Path)
    p.add_argument('--junctions-only',action='store_true',help='Exclude unresolved circle/spot semantics from reference and target IDs')
    p.add_argument('--reuse-predictions',type=Path,help='Completed two-view cache from this exact review, geometry and teacher weight')
    p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    import cv2
    from .pnl_adapter import PnlTeacher
    cv2.setNumThreads(1);cv2.setRNGSeed(20261101)
    review=json.loads(a.review.read_text());geo=json.loads(a.geometry.read_text());world=np.array([q['world_xy_m_assumed'] for q in geo['keypoints']])
    images=a.field_root.resolve();pose=a.pose_root.resolve();records=[];reasons=Counter()
    def pixels(path):
        frame=cv2.imread(str(path))
        if frame is None:raise ValueError('Image failed to decode')
        return hashlib.sha256(cv2.resize(frame,(640,640)).tobytes()).hexdigest()
    holdout={pixels(p) for p in (pose/'valid/images').glob('*.jpg')}
    for item in review['accepted']:
        image=(images/item['image']).resolve();source=(pose/item['source_image']).resolve()
        if not image.is_relative_to(images/'train/images') or not source.is_relative_to(pose/'train/images') or item['target_split']!='train':raise ValueError('Only train-to-train proposals are allowed')
        if item.get('review_status')!='approved_geometry_review':raise ValueError('Expected reviewed reference proposals')
        if file_hash(image)!=item['image_sha256'] or file_hash(source)!=item['source_image_sha256']:raise ValueError('Reviewed source bytes changed')
        if pixels(image) in holdout:raise ValueError('Holdout pixels cannot enter teacher training proposals')
    if a.out.exists():raise ValueError('Use a new output directory')
    cached={};teacher=None
    if a.reuse_predictions:
        previous=json.loads(a.reuse_predictions.read_text())
        if previous.get('completed') is not True or previous['review_sha256']!=file_hash(a.review) or previous['geometry_sha256']!=file_hash(a.geometry) or previous['teacher']['keypoint_weights_sha256']!=file_hash(a.weights):raise ValueError('Two-view prediction cache provenance mismatch')
        if previous['transform_640']!=[[1.,0.,8.],[0.,1.,-4.],[0.,0.,1.]] or previous['photometric']!={'gain':.92,'bias':3}:raise ValueError('Prediction cache transform changed')
        cached={r['image_sha256']:r for r in previous['records']};metadata=previous['teacher']
        if len(cached)!=len(review['accepted']):raise ValueError('Incomplete/duplicate prediction cache')
    else:
        teacher=PnlTeacher(a.repository,a.weights);metadata=teacher.metadata
    a.out.mkdir(parents=True);started=time.perf_counter()
    M=np.array([[1,0,8],[0,1,-4],[0,0,1.]],float)
    for index,item in enumerate(review['accepted']):
        path=images/item['image'];frame=cv2.resize(cv2.imread(str(path)),(640,640))
        if cached:
            saved=cached[item['image_sha256']];first={'prediction32':saved['prediction_first']}
            aligned=np.asarray(saved['prediction_second_aligned']);forward_ms=saved['teacher_forward_ms']
        else:
            first=teacher.predict(frame,calibrate=False)
            alternate=cv2.warpPerspective(frame,M,(640,640),borderMode=cv2.BORDER_REFLECT101)
            alternate=np.clip(alternate.astype(np.float32)*.92+3,0,255).astype(np.uint8)
            second=teacher.predict(alternate,calibrate=False);aligned=np.array(second['prediction32']);aligned[:,:2]=project(np.linalg.inv(M),aligned[:,:2])
            forward_ms=first['forward_ms']+second['forward_ms']
        points,gate=gate_points(frame,item['points'],first['prediction32'],aligned,world,reference_ids=JUNCTION_IDS if a.junctions_only else None)
        reasons.update([q['reason'] or 'accepted' for q in gate['points']] if gate['points'] else [gate['reason']])
        record={'image':item['image'],'target_split':'train','image_sha256':item['image_sha256'],
                'reference_source_image':item['source_image'],'reference_source_image_sha256':item['source_image_sha256'],
                'prediction_first':first['prediction32'],'prediction_second_aligned':aligned.tolist(),
                'teacher_points_640':points.tolist(),'gate':gate,'teacher_forward_ms':forward_ms}
        records.append(record)
        report={'schema_version':1,'scope':'automatic train-only proposals, NOT manual GT','teacher':metadata,
                'reference_ids':list(JUNCTION_IDS) if a.junctions_only else list(range(30)),
                'teacher_predictions_reused':bool(cached),'prediction_cache_sha256':file_hash(a.reuse_predictions) if a.reuse_predictions else None,
                'review_sha256':file_hash(a.review),'geometry_sha256':file_hash(a.geometry),'transform_640':M.tolist(),
                'photometric':{'gain':.92,'bias':3},'records':records,'rejection_counts':dict(reasons),
                'completed':len(records)==len(review['accepted']),'expected_images':len(review['accepted']),
                'accepted_points':sum(sum(q['accepted'] for q in r['gate']['points']) for r in records),
                'new_points':sum(sum(q['accepted'] and q['new_point'] for q in r['gate']['points']) for r in records),
                'validation_labels_used':False,'manual_ground_truth':False,'models_validated':False,'release_eligible':False,
                'elapsed_seconds':time.perf_counter()-started}
        (a.out/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'completed':index+1,'total':len(review['accepted']),'accepted_points':report['accepted_points'],'new_points':report['new_points']}),flush=True)


if __name__=='__main__':main()
