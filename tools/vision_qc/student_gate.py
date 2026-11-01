"""Source-only no-regression decision; never authorizes an Android release."""
import argparse
import json
import math
from pathlib import Path


def measurements(report):
    summary=report['validation']['summary'];errors=summary['coordinate_error_on_detected_objects']
    return {'points':summary['visible_ground_truth_points'],'pck5':summary['pck']['5'],
            'median_px':errors['median_px'],'p95_px':errors['p95_px'],'coverage':summary['coverage'],'coordinate_points':errors['count']}


def no_regression(reference,candidate):
    a=measurements(reference);b=measurements(candidate)
    if a['points']!=b['points'] or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in b.values()):raise ValueError('Incomparable/non-finite source metrics')
    if a['coordinate_points']!=a['points'] or b['coordinate_points']!=b['points']:raise ValueError('Do not lower tail error by dropping coordinate predictions')
    checks={'pck5':b['pck5']>=a['pck5'],'median':b['median_px']<=a['median_px'],
            'p95':b['p95_px']<=a['p95_px'],'coverage':b['coverage']>=a['coverage']}
    return {'passed':all(checks.values()),'improved':any(a[k]!=b[k] for k in ['pck5','median_px','p95_px','coverage']),
            'checks':checks,'reference':a,'candidate':b}


def decide(baseline,control,distilled):
    for report in [control,distilled]:
        for key in ['encoder_sha256','source_dataset_fingerprint','point_threshold','coordinate_space']:
            if report[key]!=baseline[key]:raise ValueError('Source benchmark definition changed: '+key)
    for key in ['initial_head_sha256','optimizer_updates','batch','gradient_accumulation','training_views',
                'source_train_images','pseudo_train_images','views_per_source','seed','teacher_proposals_sha256','pseudo_policy']:
        if control[key]!=distilled[key]:raise ValueError('Unmatched control/student exposure: '+key)
    if control['teacher_weight']!=0 or not 0<distilled['teacher_weight']<=.5:raise ValueError('Expected a zero-teacher control and bounded treatment')
    if control['validation_used_for_optimization_or_checkpoint_selection'] or distilled['validation_used_for_optimization_or_checkpoint_selection']:raise ValueError('Validation must not drive optimizer or checkpoint stopping')
    inputs=[[x['source'],x['view'],x['matrix_320'],x['image_sha256']] for x in control.get('augmentation_records',[])]
    if inputs and inputs!=[[x['source'],x['view'],x['matrix_320'],x['image_sha256']] for x in distilled.get('augmentation_records',[])]:raise ValueError('Augmented inputs differ')
    b_control=no_regression(baseline,control);b_distilled=no_regression(baseline,distilled);c_distilled=no_regression(control,distilled)
    selected='baseline'
    if b_control['passed'] and b_control['improved']:selected='control'
    if b_distilled['passed'] and b_distilled['improved'] and c_distilled['passed']:selected='distilled'
    return {'scope':'reused source validation only; exploratory research choice, NOT independent Mobile acceptance',
            'selected_research_checkpoint':selected,'control_vs_baseline':b_control,
            'distilled_vs_baseline':b_distilled,'distilled_vs_control':c_distilled,
            'policy':'PCK5 and coverage cannot decrease; coordinate median and P95 cannot increase',
            'models_validated':False,'release_eligible':False,'android_model_changed':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['baseline','control','distilled','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();reports=[]
    for path in [a.baseline,a.control,a.distilled]:
        report=json.loads(path.read_text());generated=path.parent/'generated-labels.json'
        if generated.is_file():report['augmentation_records']=json.loads(generated.read_text())
        reports.append(report)
    result=decide(*reports)
    if a.out.exists():raise ValueError('Use a new output file')
    a.out.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
