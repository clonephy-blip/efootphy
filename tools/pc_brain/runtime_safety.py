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
