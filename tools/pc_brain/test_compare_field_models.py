"""No private video/checkpoints required for strict 320-vs-640 comparison tests."""
import copy
import unittest

from tools.pc_brain.compare_field_models import compare_reports


def fixture(size, hits):
    frames = []
    for i, hit in enumerate(hits):
        metrics = {
            "ball_top1_hit_10px": hit,
            "ball_any_hit_10px": hit,
            "ball_proposals": 1,
            "player_proposals": 2,
            "referee_proposals": 0,
            "cursor_proposals": 1,
        }
        frames.append({
            "frame_id": f"val-00-{i:02}",
            "source_frame_index": 30 + i * 40,
            "original_pts_ms": float(500 + i * 500),
            "old": copy.deepcopy(metrics),
            "new": copy.deepcopy(metrics),
        })
    return {
        "status": "research_only_not_model_validation",
        "model_sha256": str(size) * 16,
        "model_input_size": size,
        "validation_source": "MATCH_01_PART_010, assistant-reviewed frozen labels",
        "independent_match": False,
        "inference_scope": "Local OpenCV DNN CPU, not Windows/Android runtime",
        "thresholds": {"ball": 0.08, "other": 0.22, "nms_iou": 0.5, "ball_hit_tolerance_px": 10},
        "inference_ms_median": float(size / 10),
        "inference_ms_p95": float(size / 8),
        "old": {"evaluated_visible_frames": len(frames)},
        "new": {"evaluated_visible_frames": len(frames)},
        "auxiliary_negatives": {"frames": 10},
        "records": frames,
    }


class PairingTests(unittest.TestCase):
    def test_reports_gain_and_loss_on_same_frozen_frames(self):
        small = fixture(320, [True, False, True])
        large = fixture(640, [False, True, True])
        report = compare_reports(small, large)
        self.assertEqual(report["status"], "PAIRED_RESEARCH_ONLY")
        self.assertFalse(report["eligible_for_auto_promotion"])
        a = report["new_decoder_paired"]["top1_ball_10px"]
        self.assertEqual(a["640_gains_over_320"], 1)
        self.assertEqual(a["640_losses_vs_320"], 1)
        self.assertEqual(a["both_hit"], 1)
        self.assertEqual(a["paired_n"], 3)
        self.assertEqual(len(report["top1_disagreement_frames"]), 2)
        self.assertEqual(report["per_clip"][0]["clip_id"], "val-00")

    def test_wrong_checkpoint_order_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "320x320"):
            compare_reports(fixture(640, [True]), fixture(320, [True]))

    def test_source_timestamps_must_match(self):
        small, large = fixture(320, [True]), fixture(640, [False])
        large["records"][0]["original_pts_ms"] += 100
        with self.assertRaisesRegex(ValueError, "PTS changed"):
            compare_reports(small, large)

    def test_frame_ids_must_match(self):
        small, large = fixture(320, [True]), fixture(640, [True])
        large["records"][0]["frame_id"] = "val-03-00"
        with self.assertRaisesRegex(ValueError, "frame IDs differ"):
            compare_reports(small, large)

    def test_mismatched_thresholds_are_rejected(self):
        small, large = fixture(320, [True]), fixture(640, [False])
        large["thresholds"]["ball"] = .10
        with self.assertRaisesRegex(ValueError, "Thresholds differ"):
            compare_reports(small, large)

    def test_duplicate_source_frames_are_rejected(self):
        small, large = fixture(320, [True]), fixture(640, [False])
        small["records"].append(copy.deepcopy(small["records"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate frame ID"):
            compare_reports(small, large)

    def test_exploratory_old_multi_proposal_scores_rejected(self):
        small, large = fixture(320, [True]), fixture(640, [True])
        small["records"][0]["old"]["ball_any_hit_10px"] = False
        with self.assertRaisesRegex(ValueError, "Legacy top-1 invariant"):
            compare_reports(small, large)

    def test_top1_cannot_be_correct_if_no_proposal_is_correct(self):
        small, large = fixture(320, [True]), fixture(640, [True])
        large["records"][0]["new"]["ball_any_hit_10px"] = False
        with self.assertRaisesRegex(ValueError, "Impossible top-1"):
            compare_reports(small, large)

    def test_nan_latency_rejected(self):
        small, large = fixture(320, [True]), fixture(640, [False])
        large["inference_ms_median"] = float("nan")
        with self.assertRaisesRegex(ValueError, "missing finite"):
            compare_reports(small, large)

    def test_cannot_use_same_checkpoint_for_both_sizes(self):
        small, large = fixture(320, [True]), fixture(640, [False])
        large["model_sha256"] = small["model_sha256"]
        with self.assertRaisesRegex(ValueError, "Same ONNX hash"):
            compare_reports(small, large)

    def test_missing_evidence_rejected(self):
        small, large = fixture(320, [True]), fixture(640, [False])
        large["records"] = []
        with self.assertRaisesRegex(ValueError, "No paired"):
            compare_reports(small, large)

    def test_non_boolean_scoring_rejected(self):
        small, large = fixture(320, [True]), fixture(640, [False])
        large["records"][0]["new"]["ball_top1_hit_10px"] = 1
        with self.assertRaisesRegex(ValueError, "Invalid boolean"):
            compare_reports(small, large)


if __name__ == "__main__":
    unittest.main()
