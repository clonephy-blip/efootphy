"""Score the external teacher on the same source labels and sparse Mobile refs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .evaluate_landmarks import aggregate,score_points
from .pnl_adapter import PnlTeacher,SOURCE_TO_PNL
from .pitch_tracking import project


def score_calibrations(geometry,predictions):
    """Sparse, human-observed ground references only; no inferred points scored."""
    by_name={Path(r['image']).name:r for r in predictions['records']};rows=[];all_errors=[]
    for frame in geometry['frames']:
        pred=by_name.get(frame['source_file'],{});H=pred.get('H_world_to_image')
        points=[q for q in frame['points'] if q['status']=='manual_reference'];errors=[]
        if H is not None:
            xy=project(np.asarray(H),[q['world_xy_m_assumed'] for q in points])
            errors=np.linalg.norm(xy-np.array([q['manual_xy_px'] for q in points]),axis=1).tolist()
        rows.append({'image':frame['source_file'],'manual_reference_count':len(points),
                     'predicted_count':len(errors),'correct_within_5px':sum(e<=5 for e in errors),'errors_px':errors})
        all_errors.extend(errors)
    return {'scope':'three selected Mobile stills with sparse manual references, NOT independent full-field accuracy',
            'manual_reference_count':sum(r['manual_reference_count'] for r in rows),'correct_within_5px':sum(e<=5 for e in all_errors),
            'median_px':float(np.median(all_errors)) if all_errors else None,
            'p95_px':float(np.percentile(all_errors,95)) if all_errors else None,
            'max_px':float(max(all_errors)) if all_errors else None,'rows':rows,'metric_scale_verified':False,'models_validated':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository',type=Path,required=True);p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--validation-images',type=Path,required=True);p.add_argument('--geometry',type=Path,required=True)
    p.add_argument('--mobile-images',type=Path,required=True);p.add_argument('--baseline-report',type=Path)
    p.add_argument('--calibration-report',type=Path,help='Optional dual-network PnL output for sparse geometry QC')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    import cv2
    cv2.setNumThreads(1)
    if a.out.exists():raise ValueError('Use a new output directory')
    a.out.mkdir(parents=True);teacher=PnlTeacher(a.repository,a.weights);cases=[]
    for path in sorted(a.validation_images.glob('*.jpg')):
        image=cv2.imread(str(path));h,w=image.shape[:2]
        row=np.asarray(list(map(float,(path.parent.parent/'labels'/path.with_suffix('.txt').name).read_text().split())))
        if len(row)!=101:raise ValueError('Expected one 32-keypoint pitch label')
        cases.append((path,row[5:].reshape(32,3)*[w,h,1],'source_validation'))
    geometry=json.loads(a.geometry.read_text())
    for f in geometry['frames']:
        target=np.zeros((32,3))
        for point in f['points']:
            if point['status']=='manual_reference':target[point['id']]=[*point['manual_xy_px'],2]
        cases.append((a.mobile_images/f['source_file'],target,'reviewed_mobile_three'))
    records=[]
    for path,target,scope in cases:
        frame=cv2.imread(str(path))
        if frame is None:raise ValueError('Image failed to decode')
        pred=teacher.predict(frame,calibrate=False);h,w=frame.shape[:2]
        points=score_points(target,pred['prediction32'],True,w,h)
        records.append({'image':path.name,'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'scope':scope,'prediction32':pred['prediction32'],'points':points,'summary':aggregate(points),
                        'forward_ms':pred['forward_ms'],'total_ms':pred['total_ms']})
        report={'teacher':teacher.metadata,'mapping_source_to_pnl_1based':SOURCE_TO_PNL,
                'provisional_point_ids':[30,31],'coordinate_space':'original image pixels',
                'point_threshold':.5,'records':records,'models_validated':False,
                'warning':'source validation is not independent Mobile testing; three Mobile images have only sparse manual references'}
        for key,core in [('summary_by_scope',False),('core30_summary_by_scope',True)]:
            report[key]={s:aggregate([q for r in records if r['scope']==s for q in r['points'] if not core or q['id']<30])
                         for s in ['source_validation','reviewed_mobile_three']}
        if a.baseline_report:
            baseline=json.loads(a.baseline_report.read_text());validation=baseline.get('validation',baseline)
            points=[q for r in validation['records'] for q in r['points']]
            report['baseline_same_source']={'full32':aggregate(points),'core30':aggregate([q for q in points if q['id']<30])}
        if a.calibration_report:report['calibration_on_sparse_mobile']=score_calibrations(geometry,json.loads(a.calibration_report.read_text()))
        (a.out/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'scope':scope,'image':path.name,'pck5':records[-1]['summary']['pck']['5']}),flush=True)


if __name__=='__main__':main()
