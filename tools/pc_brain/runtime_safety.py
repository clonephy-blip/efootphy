"""Small, dependency-free safety rules for the Windows PC co-processor.

A missing or ambiguous model must stop startup rather than silently loading the
wrong file. Never infer a football team from a missing classifier result.
"""
from pathlib import Path


def resolve_field_model(script_file, override=None, home=None):
    """Use an explicit model override or a known local efootphy model location."""
    script_dir = Path(script_file).resolve().parent
    root_candidate = script_dir.parent / "models" / "field_yolo11n_320.onnx"
    user_home = Path(home).expanduser() if home is not None else Path.home()
    legacy_candidate = (
        user_home / ".gemini" / "antigravity" / "scratch" / "efootphy"
        / "tools" / "models" / "field_yolo11n_320.onnx"
    )
    if override is not None and str(override).strip():
        candidates = [Path(str(override)).expanduser()]
    else:
        candidates = [root_candidate, legacy_candidate]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Khong tim thay model field_yolo11n_320.onnx. "
        "Dat model tai tools/models/ hoac set EFOOTPHY_FIELD_MODEL. "
        f"Duong dan da kiem tra: {checked}"
    )


def is_known_team(value):
    """Only explicitly classified HOME/AWAY actors can create pass hints."""
    return type(value) is int and value in (0, 1)


def validate_field_onnx_io(inputs, outputs):
    """Fail closed unless this is a static YOLO11 field model for seven classes.

    Accept 320 and 640 square model input sizes; never silently resize into a
    different graph or reinterpret a minimap/pose ONNX as a field detector.
    Inputs/outputs are ONNX Runtime NodeArg objects with .shape attributes.
    """
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Field ONNX must have exactly one input and one output")
    source = list(inputs[0].shape)
    result = list(outputs[0].shape)
    if len(source) != 4 or source[:2] != [1, 3] or source[2] != source[3]:
        raise ValueError(f"Unsupported field model input shape: {source}")
    side = source[2]
    if type(side) is not int or side not in (320, 640):
        raise ValueError(f"Field model input must be static 320x320 or 640x640: {source}")
    cells = sum((side // stride) ** 2 for stride in (8, 16, 32))
    if result != [1, 11, cells]:
        raise ValueError(
            f"Expected field ONNX [1,11,{cells}] for {side}px, got {result}; "
            "check model class schema (ball/cursor/cursor_secondary/"
            "goalkeeper/player/powerbar/referee)"
        )
    return side
