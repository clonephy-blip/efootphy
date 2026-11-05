"""Unit tests do not require ONNX, screen capture, or a Windows runner."""
import tempfile
import unittest
from pathlib import Path

from tools.pc_brain.runtime_safety import is_known_team, resolve_field_model


class ModelPathTests(unittest.TestCase):
    def test_repo_relative_model_without_cwd_assumption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "tools" / "pc_brain" / "efootball_pc_brain.py"
            model = root / "tools" / "models" / "field_yolo11n_320.onnx"
            model.parent.mkdir(parents=True)
            model.write_bytes(b"fake-model-test-only")
            self.assertEqual(resolve_field_model(script, home=root / "home"), model.resolve())

    def test_existing_antigravity_model_for_backward_compatibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            legacy = (root / ".gemini" / "antigravity" / "scratch" / "efootphy"
                      / "tools" / "models" / "field_yolo11n_320.onnx")
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(b"fake-model-test-only")
            script = root / "workspace" / "tools" / "pc_brain" / "efootball_pc_brain.py"
            self.assertEqual(resolve_field_model(script, home=root), legacy.resolve())

    def test_explicit_override_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "another-model.onnx"
            model.write_bytes(b"fake-model-test-only")
            script = root / "tools" / "pc_brain" / "efootball_pc_brain.py"
            self.assertEqual(resolve_field_model(script, override=str(model)), model.resolve())

    def test_bad_override_fails_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fallback = root / "tools" / "models" / "field_yolo11n_320.onnx"
            fallback.parent.mkdir(parents=True)
            fallback.write_bytes(b"fake-model-test-only")
            script = root / "tools" / "pc_brain" / "efootball_pc_brain.py"
            with self.assertRaises(FileNotFoundError):
                resolve_field_model(script, override=str(root / "missing.onnx"), home=root)

    def test_missing_model_error_is_actionable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "tools" / "pc_brain" / "efootball_pc_brain.py"
            with self.assertRaisesRegex(FileNotFoundError, "EFOOTPHY_FIELD_MODEL"):
                resolve_field_model(script, home=root)


class TeamHintTests(unittest.TestCase):
    def test_only_classified_outfield_teams(self):
        self.assertTrue(is_known_team(0))
        self.assertTrue(is_known_team(1))
        for value in (-1, 2, 3, None, True, "0", 1.0):
            with self.subTest(value=value):
                self.assertFalse(is_known_team(value))


if __name__ == "__main__":
    unittest.main()
