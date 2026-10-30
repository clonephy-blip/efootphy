import copy
import unittest

from tools.vision_qc.release_gate import evaluate


def complete_report():
    # Synthetic evidence for testing the gate, never an accuracy report for the application.
    return {'schema_version': 1, 'model_sha256': 'a'*64, 'testset_sha256': 'b'*64,
            'evidence_sha256': 'c'*64,
            'evaluation': {'independent': True, 'frozen_before_tuning': True, 'annotations_reviewed': True,
                           'split_unit': 'match', 'matches': 10, 'devices': 3},
            'radar': {'tp': 9900, 'fp': 100, 'fn': 100},
            'association': {'tp': 9990, 'fp': 10, 'fn': 100},
            'team': {'correct': 9990, 'incorrect': 10, 'unknown': 100},
            'tracking': {'idtp': 9500, 'idfp': 500, 'idfn': 500},
            'camera': {'held_out_points': True, 'p95_original_px': 2.5, 'held_out_point_count': 100},
            'negative': {'false_guidance_frames': 0, 'frames': 1000, 'scene_categories': 6},
            'performance': {'runs_per_device': 3, 'devices': 3, 'soak_minutes_per_run': 20,
                            'latency_p95_ms': 100, 'missed_frame_delta_pp': 1},
            'lifecycle': {'completed': True, 'failures': 0, 'api_levels': [26, 29, 31, 34, 35]},
            'confidence_intervals': {'method': 'match_bootstrap', 'resamples': 2000,
                **{x: [.90, .999] for x in ['radar_precision', 'radar_recall', 'association_precision',
                                            'team_accuracy_on_assigned', 'idf1']}}}


class ReleaseGateTest(unittest.TestCase):
    def score(self, report):
        return evaluate(report, 'a'*64, 'b'*64)

    def test_complete_synthetic_report(self):
        report = complete_report(); before = copy.deepcopy(report)
        result = self.score(report)
        self.assertTrue(result['passed'], result)
        self.assertAlmostEqual(.95, result['derived']['idf1'])
        self.assertEqual(before, report)

    def test_missing_evidence_is_blocked(self):
        self.assertFalse(self.score({})['passed'])

    def test_different_model_or_testset_is_blocked(self):
        for key in ['model_sha256', 'testset_sha256']:
            report = complete_report(); report[key] = 'd'*64
            self.assertIn('mismatch:'+key, self.score(report)['reasons'])

    def test_no_support_cannot_claim_perfect_accuracy(self):
        report = complete_report(); report['radar'] = {'tp': 0, 'fp': 0, 'fn': 0}
        self.assertIn('no_support:radar_precision', self.score(report)['reasons'])

    def test_nan_infinity_negative_boolean_are_rejected(self):
        for value in [float('nan'), float('inf'), -1, True]:
            report = complete_report(); report['performance']['latency_p95_ms'] = value
            self.assertFalse(self.score(report)['passed'])

    def test_bad_association_and_ids_fail_even_with_perfect_radar(self):
        report = complete_report(); report['association']['fp'] = 500
        report['tracking']['idfn'] = 10000
        result = self.score(report)
        self.assertIn('below_target:association_precision', result['reasons'])
        self.assertIn('below_target:idf1', result['reasons'])

    def test_one_false_tactical_frame_blocks_negative_gate(self):
        report = complete_report(); report['negative']['false_guidance_frames'] = 1
        self.assertFalse(self.score(report)['passed'])

    def test_frames_from_one_match_do_not_become_ten_matches(self):
        report = complete_report(); report['evaluation']['matches'] = 1
        self.assertIn('below_target:evaluation.matches', self.score(report)['reasons'])

    def test_runtime_missing_device_or_lifecycle_blocks(self):
        report = complete_report(); report['lifecycle']['api_levels'].remove(34)
        self.assertIn('lifecycle_coverage_incomplete', self.score(report)['reasons'])

    def test_wrong_bootstrap_unit_is_blocked(self):
        report = complete_report(); report['confidence_intervals']['method'] = 'random_frame_bootstrap'
        self.assertFalse(self.score(report)['passed'])


if __name__ == '__main__':
    unittest.main()
