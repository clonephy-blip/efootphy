# Lượt 4 — benchmark video eFootball thật (05/11/2026)

## Trạng thái

**Research only; NO model promotion, NO independent-match certification.**
This document records a preliminary paired comparison performed against
previously stored efootphy research artifacts. It does **not** claim that the
current PC Brain's actual 320x320 model has been tested.

## Inputs and provenance

- Video: retained `efootphy-round2-source-clips.zip`, **12 validation clips**
  from `MATCH_01_PART_010`. Frames are a **lossy video re-encode** at
  original 2340x1080 resolution. Mapping uses `source_first_frame` and
  original frame IDs/PTS from `source-clips/provenance.json`.
- Reference: the **38 ball-visible assistant-reviewed points** from
  `efootphy-round3-qc-2026-11-03.zip`, file
  `r2-coarse-fullframe.json`. The other 10 uncertain frames are NOT negatives
  and are omitted.
- Model: `efootphy-field-research-2026-11-01.zip/bundle/model.onnx`,
  profile `field`, 7 classes, input **640×640**, SHA-256
  `3d4119b1247849a9ecc92ee31498b78b0f9e6ddd6815fc8485db2c2c1abc47c6`.
  This is a one-epoch research checkpoint; it is NOT the same as the
  `tools/models/field_yolo11n_320.onnx` expected by PC Brain on the user's PC.
- Auxiliary negatives: 10 stored menu/transition/black-scene screenshots in
  `negative-qc/`. These do NOT establish false ball alarms during gameplay.
- Preprocessing: BGR original frame, resize **stretch** to 640x640, RGB/255.
- Backend: OpenCV DNN CPU on this research runner, **not Windows PC hardware
  or Android**. Review tolerance: 10px on source frame.

## Preliminary observed comparison

The numbers below came from a local reimplementation of the legacy
argmax/global-NMS path and iteration-3 independent-ball/class-NMS path using
**the same ONNX outputs**. A repeat with the exact committed
`benchmark_field_video.py` should confirm these before the result is
used as a release gate.

| Metric | Legacy PC postprocessing | Iteration 3 postprocessing |
|---|---:|---:|
| Ball top-1 correct within 10px, 38 visible frames | 25/38 | **25/38** |
| Any ball proposal correct within 10px | 26/38 | 27/38 |
| Frames with ball proposals | 33/38 | 33/38 |
| Total ball proposals (after top-8 selection/NMS) | 128 | 77 |
| Player proposals (not labeled/scored) | 495 | 503 |
| Referee proposals (not labeled/scored) | 11 | 13 |
| Cursor proposals (not labeled/scored) | 47 | 47 |
| Auxiliary menu/transition frames with a ball candidate | 0/10 | 0/10 |

A single added correct non-top proposal appeared on `val-02-03`.
No top-1 hit was gained. More player/referee proposals are **not** evidence
of improved player/referee recall because no corresponding ground-truth
annotation was scored. Ball candidate count differs partly due to NMS.
This sample cannot determine general false-positive rate.

A local experimental CPU run gave field inference approximately **85ms median**;
this is a host-specific measurement and excludes scrcpy, rendering, PC→Android
transmission, device thermals, and UI frame time. It does NOT prove or disprove
60 FPS on the user's PC.

**Decision:** do not promote iteration 3 as an accuracy improvement based on
this small paired sample. Keep changes in draft PR for device and video QC.
Do not fine-tune on validation video `PART_010` or regression `PART_005`.

## Critical contract fix

The available field artifact is **640×640**, but the PC Brain formerly assumed
a **320×320** ONNX model without inspecting the file. Lượt 4 adds
`validate_field_onnx_io` to reject incompatible ONNX files and support static
320×320 / 640×640 models with exactly 7 field classes. This prevents running
a 640 model with a 320 input silently.

To choose the **correct local model** before starting the Windows launcher,
set the environment variable `EFOOTPHY_FIELD_MODEL` to its exact ONNX path.
Do not assume the model in a research ZIP is identical to a private 320 model.

## Reproduce with the EXACT committed decoder

Extract the `bundle/model.onnx` from the private field research archive.
Keep the source ZIP and QC ZIP private; do NOT upload footage/checkpoints into
the GitHub repository.

```powershell
python -m tools.pc_brain.benchmark_field_video `
  --model "C:\path\to\model.onnx" `
  --input-size 640 `
  --source-clips "C:\path\to\efootphy-round2-source-clips.zip" `
  --qc "C:\path\to\efootphy-round3-qc-2026-11-03.zip" `
  --expected-model-sha256 3d4119b1247849a9ecc92ee31498b78b0f9e6ddd6815fc8485db2c2c1abc47c6 `
  --out "round4-field-benchmark.json"
```

The new benchmark runner imports
`tools.pc_brain.field_postprocess.decode_field_detections` directly and
uses frozen reviewed samples. If its numbers differ from this exploratory
comparison, prefer its reproducible output and investigate the discrepancy
before making deployment claims.

## Still needed

1. Benchmark **the actual 320 model** that PC Brain loads on Windows.
2. Review player/referee/cursor and ball-negative gameplay labels on separate
   matches; report per-class TP/FP/FN and UNKNOWN-team coverage.
3. Benchmark true ball tracking/trajectory, aerial case `PART_005`, and
   phone/USB RTT and frame-time under game load.
4. Only then accept/reject accuracy change and connect to tactical inference.

No field/ball model or data is uploaded in this branch.
