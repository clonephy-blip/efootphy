import unittest

from .evaluate_landmarks import aggregate,score_points


class LandmarkMetricTest(unittest.TestCase):
    def targets(self):return [[10,10,2]]+[[0,0,0] for _ in range(31)]
    def predictions(self,x=10,y=10,score=.9):return [[x,y,score] for _ in range(32)]

    def test_missing_object_is_not_removed_from_denominator(self):
        s=aggregate(score_points(self.targets(),self.predictions(),False,640,640))
        self.assertEqual(s['visible_ground_truth_points'],1)
        self.assertEqual(s['pck']['10'],0);self.assertIsNone(s['coordinate_error_on_detected_objects']['median_px'])

    def test_low_confidence_correct_point_is_a_miss_but_is_reported_unfiltered(self):
        s=aggregate(score_points(self.targets(),self.predictions(score=.1),True,640,640))
        self.assertEqual(s['pck']['3'],0);self.assertEqual(s['unfiltered_pck']['3'],1)
        self.assertEqual(s['coverage'],0)

    def test_pixel_error_and_thresholds_use_euclidean_distance(self):
        s=aggregate(score_points(self.targets(),self.predictions(x=13,y=14),True,640,640))
        self.assertEqual(s['pck']['3'],0);self.assertEqual(s['pck']['5'],1)
        self.assertEqual(s['coordinate_error_on_detected_objects']['median_px'],5)

    def test_unsupported_keypoint_has_no_fabricated_zero_accuracy(self):
        s=aggregate([]);self.assertEqual(s['visible_ground_truth_points'],0);self.assertIsNone(s['pck']['5'])
