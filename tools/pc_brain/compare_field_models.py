"""Strict paired field-model research comparison; no automatic model promotion.

Requires two *actual different ONNX files* (320 and 640) and the same frozen
clips/QC. Reuses iteration-4's decoder, so image selection and scoring are
identical across model sizes. Never infers scores from an absent checkpoint.
"""
import argparse
import json
import math
from pathlib import Path

from tools.pc_brain.benchmark_field_video import run_benchmark


POLICIES = ("old", "new")
FIELDS = ("ball_top1_hit_10px", "ball_any_hit_10px")


def _records_by_frame(report):
    rows = report.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("No paired scored frames; refuse empty comparison")
    result = {}
    for row in rows:
        frame_id = row.get("frame_id")
        if not isinstance(frame_id, str) or not frame_id.startswith("val-"):
            raise ValueError("Scored frame lacks frozen val-* ID")
        if frame_id in result:
            raise ValueError(f"Duplicate frame ID: {frame_id}")
        if "source_frame_index" not in row or "original_pts_ms" not in row:
            raise ValueError(f"Missing source timing/provenance for {frame_id}")
        for policy in POLICIES:
            data = row.get(policy)
            if not isinstance(data, dict) or not all(k in data for k in FIELDS):
                raise ValueError(f"Missing {policy} ball metrics for {frame_id}")
            if any(type(data[key]) is not bool for key in FIELDS):
                raise ValueError(f"Invalid boolean scoring for {frame_id}")
        # Actual legacy PC detector only returns one selected ball, so an
        # old any-hit that differs from old top-1 means the benchmark compared
        # a different exploratory candidate protocol instead of PC runtime.
        if row["old"]["ball_any_hit_10px"] != row["old"]["ball_top1_hit_10px"]:
            raise ValueError(f"Legacy top-1 invariant violated at {frame_id}")
        for policy in POLICIES:
            if row[policy]["ball_top1_hit_10px"] and not row[policy]["ball_any_hit_10px"]:
                raise ValueError(f"Impossible top-1/any-hit mismatch at {frame_id}")
        result[frame_id] = row
    return result


def _paired_counts(a, b, metric):
    hit_a = hit_b = gains = losses = both_hit = both_miss = 0
    for key in sorted(a):
        v0 = a[key]["new"][metric]
        v1 = b[key]["new"][metric]
        hit_a += int(v0)
        hit_b += int(v1)
        gains += int(not v0 and v1)
        losses += int(v0 and not v1)
        both_hit += int(v0 and v1)
        both_miss += int(not v0 and not v1)
    return {
        "model_320_hits": hit_a,
        "model_640_hits": hit_b,
        "640_gains_over_320": gains,
        "640_losses_vs_320": losses,
        "both_hit": both_hit,
        "both_miss": both_miss,
        "paired_n": len(a),
    }


def compare_reports(model_320, model_640):
    """Compare real same-frame data; reject any mismatched split/provenance."""
    if model_320.get("model_input_size") != 320:
        raise ValueError("Left report is not an actual 320x320 model run")
    if model_640.get("model_input_size") != 640:
        raise ValueError("Right report is not an actual 640x640 model run")
    if model_320.get("model_sha256") == model_640.get("model_sha256"):
        raise ValueError("Same ONNX hash cannot represent independent input-size comparison")
    for key in ("validation_source", "inference_scope"):
        if model_320.get(key) != model_640.get(key):
            raise ValueError(f"Not comparable: {key}")
    if model_320.get("thresholds") != model_640.get("thresholds"):
        raise ValueError("Thresholds differ; paired result cannot be compared")
    if model_320.get("independent_match") is not False or model_640.get("independent_match") is not False:
        raise ValueError("Missing explicit independent-match limitation")
    a = _records_by_frame(model_320)
    b = _records_by_frame(model_640)
    if a.keys() != b.keys():
        raise ValueError("Frozen video frame IDs differ across model runs")
    clips = {}
    differences = []
    for frame_id in sorted(a):
        x, y = a[frame_id], b[frame_id]
        if x["source_frame_index"] != y["source_frame_index"]:
            raise ValueError(f"Frame index changed at {frame_id}")
        if abs(float(x["original_pts_ms"]) - float(y["original_pts_ms"])) > 1e-3:
            raise ValueError(f"PTS changed at {frame_id}")
        clip_id = frame_id.rsplit("-", 1)[0]
        clips.setdefault(clip_id, [[], []])
        clips[clip_id][0].append(x)
        clips[clip_id][1].append(y)
        if x["new"]["ball_top1_hit_10px"] != y["new"]["ball_top1_hit_10px"]:
            differences.append({
                "frame_id": frame_id,
                "source_frame_index": x["source_frame_index"],
                "320_top1_hit": x["new"]["ball_top1_hit_10px"],
                "640_top1_hit": y["new"]["ball_top1_hit_10px"],
            })
    per_clip = []
    for clip_id, (left, right) in sorted(clips.items()):
        la = {x["frame_id"]: x for x in left}
        lb = {x["frame_id"]: x for x in right}
        per_clip.append({
            "clip_id": clip_id,
            "new_top1": _paired_counts(la, lb, "ball_top1_hit_10px"),
            "new_any_candidate": _paired_counts(la, lb, "ball_any_hit_10px"),
        })
    for label, report in (("320", model_320), ("640", model_640)):
        for key in ("inference_ms_median", "inference_ms_p95"):
            val = report.get(key)
            if type(val) not in (float, int) or not math.isfinite(val) or val < 0:
                raise ValueError(f"{label} missing finite {key}")
    return {
        "status": "PAIRED_RESEARCH_ONLY",
        "eligible_for_auto_promotion": False,
        "why_not_promoted": [
            "Only 38 assistant-reviewed visible ball frames from MATCH_01_PART_010",
            "No independent match; same match family and potentially reused validation",
            "No in-game absent-ball negatives or independently scored player/referee/cursor labels",
            "OpenCV host latency is not scrcpy + PC Brain + USB + phone latency",
            "Review aerial-ball regression PART_005 and per-device sustained load first",
        ],
        "model_320_sha256": model_320["model_sha256"],
        "model_640_sha256": model_640["model_sha256"],
        "source": model_320["validation_source"],
        "thresholds": model_320["thresholds"],
        "new_decoder_paired": {
            "top1_ball_10px": _paired_counts(a, b, "ball_top1_hit_10px"),
            "any_ball_candidate_10px": _paired_counts(a, b, "ball_any_hit_10px"),
        },
        "per_clip": per_clip,
        "top1_disagreement_frames": differences,
        "runtime_opencv_cpu_ms": {
            "model_320_median": model_320["inference_ms_median"],
            "model_320_p95": model_320["inference_ms_p95"],
            "model_640_median": model_640["inference_ms_median"],
            "model_640_p95": model_640["inference_ms_p95"],
        },
        "model_320": {k: model_320[k] for k in ("old", "new", "auxiliary_negatives")},
        "model_640": {k: model_640[k] for k in ("old", "new", "auxiliary_negatives")},
    }


def main():
    ap = argparse.ArgumentParser(description="Pair REAL 320 vs 640 field ONNX models")
    ap.add_argument("--model-320", type=Path, required=True)
    ap.add_argument("--model-640", type=Path, required=True)
    ap.add_argument("--source-clips", type=Path, required=True)
    ap.add_argument("--qc", type=Path, required=True)
    ap.add_argument("--sha-320", help="Expected SHA-256 of private 320 model")
    ap.add_argument("--sha-640", help="Expected SHA-256 of 640 research model")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    for file in (args.model_320, args.model_640, args.source_clips, args.qc):
        if not file.is_file():
            ap.error(f"Required file is missing: {file}")

    left = run_benchmark(args.model_320, args.source_clips, args.qc,
                         input_size=320, expected_model_sha256=args.sha_320)
    right = run_benchmark(args.model_640, args.source_clips, args.qc,
                          input_size=640, expected_model_sha256=args.sha_640)
    comparison = compare_reports(left, right)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(comparison, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(json.dumps({key: value for key, value in comparison.items()
                      if key not in ("per_clip", "top1_disagreement_frames",
                                     "model_320", "model_640")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
