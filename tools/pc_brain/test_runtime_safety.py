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



class FieldModelContractTests(unittest.TestCase):
    def test_accepts_320_and_640_field_models(self):
        from types import SimpleNamespace
        from tools.pc_brain.runtime_safety import validate_field_onnx_io
        def node(shape):
            return SimpleNamespace(shape=shape)
        self.assertEqual(
            validate_field_onnx_io([node([1, 3, 320, 320])], [node([1, 11, 2100])]), 320
        )
        self.assertEqual(
            validate_field_onnx_io([node([1, 3, 640, 640])], [node([1, 11, 8400])]), 640
        )

    def test_rejects_wrong_model_or_dynamic_shape(self):
        from types import SimpleNamespace
        from tools.pc_brain.runtime_safety import validate_field_onnx_io
        def node(shape):
            return SimpleNamespace(shape=shape)
        for input_shape, output_shape in (
            ([1, 3, 640, 640], [1, 7, 8400]),
            ([1, 3, 640, 640], [1, 11, 2100]),
            ([1, 3, "height", "width"], [1, 11, 8400]),
            ([1, 3, 416, 416], [1, 11, 3549]),
            ([1, 1, 640, 640], [1, 11, 8400]),
        ):
            with self.subTest(input_shape=input_shape, output_shape=output_shape):
                with self.assertRaises(ValueError):
                    validate_field_onnx_io([node(input_shape)], [node(output_shape)])

    def test_rejects_multiple_outputs(self):
        from types import SimpleNamespace
        from tools.pc_brain.runtime_safety import validate_field_onnx_io
        a = SimpleNamespace(shape=[1, 3, 320, 320])
        b = SimpleNamespace(shape=[1, 11, 2100])
        with self.assertRaises(ValueError):
            validate_field_onnx_io([a], [b, b])


if __name__ == "__main__":
    unittest.main()
