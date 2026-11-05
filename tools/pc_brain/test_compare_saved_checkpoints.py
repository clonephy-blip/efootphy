"""Synthetic QC tests; no user footage or heavyweight ONNX checkpoints in CI."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from tools.pc_brain.compare_saved_checkpoints import (
    extract_known_checkpoint, paired_320s,
)


def fixture_archive(root, manifest, model=b"test-once", member="run/weights/best.onnx"):
    path = Path(root) / "checkpoint.zip"
    with ZipFile(path, "w") as z:
        z.writestr(member, model)
        z.writestr("manifest.json", json.dumps(manifest))
    return path


class FrozenArchiveIdentityTests(unittest.TestCase):
    def test_ball_only_manifest_and_sha_checked(self):
        with tempfile.TemporaryDirectory() as td:
            blob = b"synthetic model, not ONNX"
            sha = hashlib.sha256(blob).hexdigest()
            p = fixture_archive(
                td, {"training_class_ids": [0], "onnx_sha256": sha}, blob
            )
            dest = Path(td) / "selected.onnx"
            config = {
                "member": "run/weights/best.onnx", "manifest": "manifest.json",
                "size": 320, "role": "ball_only_finetuned", "sha256": sha
            }
            result = extract_known_checkpoint(p, config, dest)
            self.assertEqual(result["role"], "ball_only_finetuned")
            self.assertEqual(dest.read_bytes(), blob)

    def test_reject_modified_checkpoint_even_with_same_name(self):
        with tempfile.TemporaryDirectory() as td:
            sha = hashlib.sha256(b"original").hexdigest()
            p = fixture_archive(td, {"training_class_ids": [0], "onnx_sha256": sha},
                                b"unverified replacement")
            c = {"member": "run/weights/best.onnx", "manifest": "manifest.json",
                 "size": 320, "role": "ball_only_finetuned", "sha256": sha}
            with self.assertRaisesRegex(ValueError, "SHA256"):
                extract_known_checkpoint(p, c, Path(td) / "out.onnx")

    def test_reject_non_ball_only_training_profile(self):
        with tempfile.TemporaryDirectory() as td:
            blob = b"fake model"
            sha = hashlib.sha256(blob).hexdigest()
            p = fixture_archive(td, {"training_class_ids": [0, 4],
                                     "onnx_sha256": sha}, blob)
            c = {"member": "run/weights/best.onnx", "manifest": "manifest.json",
                 "size": 320, "role": "ball_only_finetuned", "sha256": sha}
            with self.assertRaisesRegex(ValueError, "ball-only"):
                extract_known_checkpoint(p, c, Path(td) / "out.onnx")


def mock_report(ids, scores, pts=None):
    rows = []
    for index, (name, hit) in enumerate(zip(ids, scores)):
        rows.append({"frame_id": name, "source_frame_index": 100 + index,
                     "original_pts_ms": float(index) if pts is None else pts[index],
                     "new": {"ball_top1_hit_10px": hit}})
    return {"records": rows}


class ArchivedPairingTests(unittest.TestCase):
    def test_round2_round3_disagreements(self):
        first = mock_report(["val-01-00", "val-01-01", "val-01-02"],
                            [True, True, False])
        second = mock_report(["val-01-00", "val-01-01", "val-01-02"],
                             [False, True, True])
        summary = paired_320s(first, second)
        self.assertEqual(summary["round2_correct"], 2)
        self.assertEqual(summary["round3_correct"], 2)
        self.assertEqual(summary["round3_gains"], 1)
        self.assertEqual(summary["round3_losses"], 1)

    def test_source_pts_mismatch_fails_closed(self):
        a = mock_report(["val-01-00"], [True], [100.0])
        b = mock_report(["val-01-00"], [False], [120.0])
        with self.assertRaisesRegex(ValueError, "timing"):
            paired_320s(a, b)


if __name__ == "__main__":
    unittest.main()
