"""Smoke tests for paired field decode, independent of private model and clips."""
import unittest
import numpy as np

from tools.pc_brain.benchmark_field_video import (
    _old_pc_decoder, _score_ball, _summarize
)
from tools.pc_brain.field_postprocess import decode_field_detections


def frame_proposals():
    pred = np.zeros((1, 11, 3), dtype=np.float32)
    pred[0, :4, 0] = [100, 100, 20, 40]
    pred[0, 4+4, 0] = .90  # player
    pred[0, :4, 1] = [100, 100, 20, 40]
    pred[0, 4+6, 1] = .88  # referee, same box
    pred[0, :4, 2] = [65, 50, 4, 4]
    pred[0, 4+0, 2] = .20
    pred[0, 4+1, 2] = .75  # cursor scores higher than ball
    return pred


class BenchmarkDecoderTests(unittest.TestCase):
    def test_same_frame_shows_legacy_and_new_semantic_difference(self):
        pred = frame_proposals()
        old_ball, old_counts = _old_pc_decoder(pred, 640, 360, 320)
        new = decode_field_detections(pred, 640, 360, input_size=320)
        self.assertEqual(old_ball, [])  # ball suppressed by argmax in original
        self.assertEqual(len(new[0]), 1)
        self.assertEqual(old_counts.get(4, 0) + old_counts.get(6, 0), 1)
        self.assertEqual(len(new[4]) + len(new[6]), 2)

    def test_ball_pixel_error_and_missing_proposals(self):
        hit, error = _score_ball([90, 90, 110, 110], [100, 100])
        self.assertTrue(hit)
        self.assertEqual(error, 0.0)
        missing, error = _score_ball(None, [100, 100])
        self.assertFalse(missing)
        self.assertIsNone(error)

    def test_summary_counts_but_does_not_invent_accuracy(self):
        row = {"new": {"ball_top1_hit_10px": True,
                       "ball_any_hit_10px": True, "ball_proposals": 1,
                       "player_proposals": 3, "referee_proposals": 1,
                       "cursor_proposals": 2}}
        got = _summarize([row], "new")
        self.assertEqual(got["ball_top1_hit_10px"], 1)
        self.assertEqual(got["player_proposals"], 3)
        self.assertNotIn("player_recall", got)


if __name__ == "__main__":
    unittest.main()
