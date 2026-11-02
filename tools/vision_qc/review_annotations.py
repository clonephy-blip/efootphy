"""Validate external annotation proposals and score sparse reviewed references.

This gate never edits labels or treats an external AI's uncertainty as truth.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


COORDINATE_SPACE='original_image_pixels_top_left_y_down'
VISIBILITY={'visible','occluded','outside','uncertain'}


def finite_number(value):return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def validate(payload,manifest,image_root=None):
    if type(payload.get('schema_version')) is not int or payload['schema_version']!=1 or payload.get('status')!='automatic_review_proposals_not_ground_truth' or payload.get('coordinate_space')!=COORDINATE_SPACE:
        raise ValueError('External annotation schema/status/coordinate space mismatch')
    expected={f['frame_id']:f for f in manifest['frames']}
    if len(expected)!=len(manifest['frames']):raise ValueError('Duplicate manifest frame')
    found={};warnings=[]
    for frame in payload['frames']:
        key=frame['frame_id']
        if key not in expected or key in found:raise ValueError('Unknown/duplicate annotation frame')
        item=expected[key];w=item['width_px'];h=item['height_px']
        if frame['width_px']!=w or frame['height_px']!=h:raise ValueError('Image dimensions changed')
        if image_root is not None:
            root=Path(image_root).resolve();image=(root/item['file']).resolve()
            if not image.is_relative_to(root) or hashlib.sha256(image.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Input image path/hash mismatch')
        seen=set();visible=0
        for point in frame['points']:
            identifier=point['candidate_id'];state=point['visibility'];xy=point['xy_px'];uncertainty=point.get('estimated_uncertainty_px')
            if not isinstance(point.get('physical_name_vi'),str) or not point['physical_name_vi'].strip():raise ValueError('Physical point name missing')
            if identifier is not None:
                if not isinstance(identifier,int) or isinstance(identifier,bool) or identifier not in range(32) or identifier in seen:raise ValueError('Invalid/duplicate candidate ID')
                seen.add(identifier)
            if state not in VISIBILITY:raise ValueError('Unknown visibility')
            if state=='visible':
                if not isinstance(xy,list) or len(xy)!=2 or not all(finite_number(v) for v in xy) or not (0<=xy[0]<w and 0<=xy[1]<h):raise ValueError('Visible coordinates must be finite original-image pixels inside the frame')
                visible+=1
            elif xy is not None:raise ValueError('Unobserved points cannot have coordinates')
            if uncertainty is not None and (not finite_number(uncertainty) or uncertainty<0):raise ValueError('Invalid declared uncertainty')
        if not visible:warnings.append({'frame_id':key,'reason':'no_visible_points_supplied; no visual coordinate review established'})
        found[key]=frame
    if set(found)!=set(expected):raise ValueError('Incomplete annotation handback')
    return found,warnings


def percentile(values,q):
    if not values:return None
    values=sorted(values);index=(len(values)-1)*q;lo=math.floor(index);hi=math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def summarize(records):
    errors=[r['error_px'] for r in records if r['error_px'] is not None];n=len(records)
    return {'reviewed_reference_points':n,'visible_coordinate_comparisons':len(errors),'missing_or_unobserved':n-len(errors),
            'correct_within_px':{str(t):sum(r['error_px'] is not None and r['error_px']<=t for r in records) for t in [5,10]},
            'pck':{str(t):sum(r['error_px'] is not None and r['error_px']<=t for r in records)/n if n else None for t in [5,10]},
            'coordinate_errors_on_comparable_points':{'count':len(errors),'median_px':percentile(errors,.5),'p95_px':percentile(errors,.95),'max_px':max(errors) if errors else None}}


def review(payload,manifest,reference,image_root=None):
    frames,warnings=validate(payload,manifest,image_root);expected={f['frame_id']:f for f in manifest['frames']};records=[]
    per_frame={}
    for reference_frame in reference['frames']:
        key=f"mobile-{reference_frame['index']:02d}"
        if key not in frames:raise ValueError('Reference frame not in manifest')
        item=expected[key]
        if item['sha256']!=reference_frame['source_sha256'] or [item['width_px'],item['height_px']]!=reference_frame['image_size']:
            raise ValueError('Reference refers to different original image bytes or dimensions')
        proposed={p['candidate_id']:p for p in frames[key]['points'] if p['candidate_id'] is not None};rows=[]
        reference_ids=set()
        for point in reference_frame['points']:
            if point['status']!='manual_reference':continue
            identifier=point['id'];xy_reference=point['manual_xy_px']
            if identifier in reference_ids or not isinstance(xy_reference,list) or len(xy_reference)!=2 or not all(finite_number(v) for v in xy_reference) or not (0<=xy_reference[0]<item['width_px'] and 0<=xy_reference[1]<item['height_px']):
                raise ValueError('Invalid/duplicate reviewed reference point')
            reference_ids.add(identifier)
            p=proposed.get(point['id']);xy=p['xy_px'] if p and p['visibility']=='visible' else None
            error=math.dist(xy,point['manual_xy_px']) if xy is not None else None
            rows.append({'frame_id':key,'candidate_id':point['id'],'reference_xy_px':point['manual_xy_px'],
                         'proposal_xy_px':xy,'proposal_visibility':p['visibility'] if p else 'missing',
                         'error_px':error,'declared_uncertainty_px':p.get('estimated_uncertainty_px') if p else None})
        records.extend(rows);per_frame[key]=summarize(rows)
    uncertainty=[r for r in records if r['error_px'] is not None and r['declared_uncertainty_px'] is not None]
    return {'schema_version':1,'scope':'external review against sparse previously reviewed manual image points; not full-field or video accuracy',
            'schema_valid':True,'warnings':warnings,'summary':summarize(records),'per_frame':per_frame,'records':records,
            'declared_uncertainty_diagnostic':{'comparisons':len(uncertainty),'within_declared_radius':sum(r['error_px']<=r['declared_uncertainty_px'] for r in uncertainty),
                                                'calibrated_probability':False},
            'automatic_training_import_allowed':False,'training_labels_modified':False,'models_validated':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['annotations','manifest','reference','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--images',type=Path,required=True);a=parser.parse_args()
    if a.out.exists():raise ValueError('Use a new output file')
    result=review(json.loads(a.annotations.read_text(encoding='utf-8-sig')),json.loads(a.manifest.read_text()),json.loads(a.reference.read_text()),a.images)
    result['annotation_sha256']=hashlib.sha256(a.annotations.read_bytes()).hexdigest();result['manifest_sha256']=hashlib.sha256(a.manifest.read_bytes()).hexdigest();result['reference_sha256']=hashlib.sha256(a.reference.read_bytes()).hexdigest()
    a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result['summary'],indent=2))


if __name__=='__main__':main()
