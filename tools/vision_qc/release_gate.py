"""Fail closed on incomplete independent evaluation. Never enables an APK feature."""
import argparse
import json
import math
import re
from pathlib import Path

HASH = re.compile(r'[a-f0-9]{64}')


def evaluate(report, model_sha256, testset_sha256):
    reasons = []
    derived = {}

    def fail(code):
        if code not in reasons:
            reasons.append(code)

    def get(path):
        value = report
        for key in path.split('.'):
            if not isinstance(value, dict) or key not in value:
                fail('missing:' + path)
                return None
            value = value[key]
        return value

    def flag(path):
        if get(path) is not True:
            fail('unverified:' + path)

    def number(path, integer=False):
        value = get(path)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            fail('invalid:' + path)
            return None
        if integer and not isinstance(value, int):
            fail('invalid:' + path)
            return None
        return value

    def minimum(path, threshold):
        value = number(path)
        if value is not None and value < threshold:
            fail('below_target:' + path)

    def maximum(path, threshold):
        value = number(path)
        if value is not None and value > threshold:
            fail('above_target:' + path)

    def ratio(name, numerator, denominator, threshold=None):
        if numerator is None or denominator is None or denominator <= 0:
            fail('no_support:' + name)
            return
        result = numerator / denominator
        derived[name] = result
        if threshold is not None and result < threshold:
            fail('below_target:' + name)

    if get('schema_version') != 1:
        fail('schema_version')
    for key, expected in [('model_sha256', model_sha256), ('testset_sha256', testset_sha256)]:
        if not isinstance(expected, str) or HASH.fullmatch(expected) is None or get(key) != expected:
            fail('mismatch:' + key)
    evidence = get('evidence_sha256')
    if not isinstance(evidence, str) or HASH.fullmatch(evidence) is None:
        fail('missing_evidence_digest')
    for path in ['evaluation.independent', 'evaluation.frozen_before_tuning', 'evaluation.annotations_reviewed',
                 'camera.held_out_points', 'lifecycle.completed']:
        flag(path)
    if get('evaluation.split_unit') != 'match':
        fail('split_is_not_match')
    minimum('evaluation.matches', 10)
    minimum('evaluation.devices', 3)

    for key, target in [('radar', .97), ('association', .99)]:
        tp, fp, fn = [number(key + '.' + x, integer=True) for x in ['tp', 'fp', 'fn']]
        if None not in [tp, fp, fn]:
            ratio(key + '_precision', tp, tp + fp, target)
            ratio(key + '_recall', tp, tp + fn, .97 if key == 'radar' else None)
    correct, incorrect, unknown = [number('team.' + x, integer=True) for x in ['correct', 'incorrect', 'unknown']]
    if None not in [correct, incorrect, unknown]:
        ratio('team_accuracy_on_assigned', correct, correct + incorrect, .99)
        ratio('team_coverage', correct + incorrect, correct + incorrect + unknown)
    idtp, idfp, idfn = [number('tracking.' + x, integer=True) for x in ['idtp', 'idfp', 'idfn']]
    if None not in [idtp, idfp, idfn]:
        ratio('idf1', 2 * idtp, 2 * idtp + idfp + idfn, .90)

    maximum('camera.p95_original_px', 3)
    minimum('camera.held_out_point_count', 100)
    maximum('negative.false_guidance_frames', 0)
    minimum('negative.frames', 1000)
    minimum('negative.scene_categories', 6)
    minimum('performance.runs_per_device', 3)
    minimum('performance.devices', 3)
    minimum('performance.soak_minutes_per_run', 20)
    maximum('performance.latency_p95_ms', 150)
    maximum('performance.missed_frame_delta_pp', 2)
    # A negative delta must be reported as zero; keep the raw paired baseline traces as evidence.
    maximum('lifecycle.failures', 0)
    apis = get('lifecycle.api_levels')
    if not isinstance(apis, list) or not {26, 29, 31, 34, 35}.issubset(set(x for x in apis if isinstance(x, int))):
        fail('lifecycle_coverage_incomplete')
    ci = get('confidence_intervals')
    if not isinstance(ci, dict) or ci.get('method') != 'match_bootstrap':
        fail('match_bootstrap_missing')
    else:
        minimum('confidence_intervals.resamples', 2000)
        for key in ['radar_precision', 'radar_recall', 'association_precision', 'team_accuracy_on_assigned', 'idf1']:
            bounds = ci.get(key)
            if not isinstance(bounds, list) or len(bounds) != 2 or any(
                    isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1
                    for x in bounds) or bounds[0] > bounds[1]:
                fail('invalid_confidence_interval:' + key)
    return {'schema_version': 1, 'passed': not reasons, 'reasons': reasons, 'derived': derived,
            'scope': 'reported evidence completeness and numeric targets; independent human review is still required'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--testset-sha256', required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.report.read_text()), args.model_sha256, args.testset_sha256)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 2)


if __name__ == '__main__':
    main()
