"""Deterministic regression tests for field postprocess (no video or weights)."""
import unittest
import numpy as np

from tools.pc_brain.field_postprocess import (
    decode_field_detections, cluster_kit_colors, CLASS_BALL, CLASS_PLAYER,
    CLASS_REFEREE, CLASS_CURSOR,
)


def prediction(n):
    return np.zeros((1, 11, n), dtype=np.float32)


def place(p, i, cx, cy, w, h, scores):
    p[0, :4, i] = [cx, cy, w, h]
    for cls, score in scores.items():
        p[0, 4 + cls, i] = score


class ClassLocalNmsTests(unittest.TestCase):
    def test_player_and_referee_can_overlap(self):
        p = prediction(2)
        place(p, 0, 100, 100, 20, 40, {CLASS_PLAYER: .94})
        place(p, 1, 100, 100, 20, 40, {CLASS_REFEREE: .91})
        d = decode_field_detections(p, 640, 360)
        self.assertEqual(len(d[CLASS_PLAYER]), 1)
        self.assertEqual(len(d[CLASS_REFEREE]), 1)

    def test_duplicates_same_class_suppressed(self):
        p = prediction(3)
        place(p, 0, 100, 100, 20, 40, {CLASS_PLAYER: .91})
        place(p, 1, 101, 101, 20, 40, {CLASS_PLAYER: .87})
        place(p, 2, 220, 100, 20, 40, {CLASS_PLAYER: .75})
        d = decode_field_detections(p, 640, 360)
        self.assertEqual(len(d[CLASS_PLAYER]), 2)
        self.assertAlmostEqual(d[CLASS_PLAYER][0][1], .91, places=5)

    def test_ball_class_channel_independent_of_argmax(self):
        p = prediction(1)
        place(p, 0, 70, 60, 3, 3, {CLASS_BALL: .23, CLASS_CURSOR: .60})
        d = decode_field_detections(p, 640, 360)
        self.assertEqual(len(d[CLASS_BALL]), 1)
        self.assertEqual(len(d[CLASS_CURSOR]), 1)

    def test_stretch_coordinates_original_frame(self):
        p = prediction(1)
        place(p, 0, 160, 160, 32, 32, {CLASS_BALL: .9})
        d = decode_field_detections(p, 1280, 720)
        b = d[CLASS_BALL][0][0]
        np.testing.assert_allclose(b, [576, 324, 704, 396])

    def test_invalid_and_outside_boxes_excluded(self):
        p = prediction(4)
        place(p, 0, 100, 100, -10, 20, {CLASS_PLAYER: .95})
        place(p, 1, 200, 100, 5, 5, {CLASS_PLAYER: np.nan})
        place(p, 2, 500, 500, 4, 4, {CLASS_PLAYER: .99})
        place(p, 3, 10, 20, 8, 8, {CLASS_PLAYER: .83})
        d = decode_field_detections(p, 640, 360)
        self.assertEqual(len(d[CLASS_PLAYER]), 1)

    def test_model_shape_checked(self):
        with self.assertRaisesRegex(ValueError, "Expected field model"):
            decode_field_detections(np.zeros((1, 7, 2100), dtype=np.float32), 640, 360)

    def test_no_candidates(self):
        self.assertTrue(all(not v for v in
                            decode_field_detections(prediction(0), 640, 360).values()))


class JerseyClassificationTests(unittest.TestCase):
    def test_one_kit_only_remains_unknown(self):
        tags, centers = cluster_kit_colors([[25, 50, 100], [26, 50, 102]])
        self.assertEqual(tags, [-1, -1])
        self.assertIsNone(centers)

    def test_distinct_kits_do_not_swap_when_detection_order_reverses(self):
        colors = [[220, 10, 10], [10, 10, 220], [210, 10, 20], [20, 10, 210]]
        teams, centers = cluster_kit_colors(colors)
        self.assertEqual(teams[0], teams[2])
        self.assertEqual(teams[1], teams[3])
        self.assertNotEqual(teams[0], teams[1])
        reversed_teams, _ = cluster_kit_colors(list(reversed(colors)), prior=centers)
        self.assertEqual(reversed_teams, list(reversed(teams)))

    def test_missing_and_ambiguous_jerseys_remain_unknown(self):
        colors = [[200, 20, 20], None, [20, 20, 200]]
        _, centers = cluster_kit_colors(colors)
        tags, _ = cluster_kit_colors([None, [110, 20, 110]], prior=centers)
        self.assertEqual(tags, [-1, -1])

    def test_prior_centers_are_used_even_when_only_one_team_visible(self):
        first, centers = cluster_kit_colors([[200, 20, 20], [20, 20, 200]])
        second, _ = cluster_kit_colors([[200, 20, 20], None], prior=centers)
        self.assertEqual(second, [first[0], -1])


if __name__ == "__main__":
    unittest.main()
