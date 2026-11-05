"""Repeatable research comparison for the three retained eFootPhy ONNX archives.

This intentionally distinguishes ball-focused Round2/Round3 320 checkpoints
from the field-source 640 checkpoint. It DOES NOT identify the user's private
Windows PC Brain model and does not choose a deployment checkpoint.

Usage (from repository root):
  python -m tools.pc_brain.compare_saved_checkpoints \
    --round2-zip round2-trained-checkpoint.zip \
    --round3-zip round3-trained-checkpoint.zip \
    --field-zip field-research.zip \
    --source-clips round2-source-clips.zip --qc round3-qc.zip \
    --out round6-research.json
"""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from tools.pc_brain.benchmark_field_video import run_benchmark
from tools.pc_brain.compare_field_models import compare_reports


ARCHIVES = {
    "round2_ball_320": {
        "member": "run/weights/best.onnx",
        "manifest": "manifest.json",
        "size": 320,
        "sha256": "a05e725de2bc6e043733f6a2c84709b384f661b00d2d7aabadda4ed5029afbd8",
        "role": "ball_only_finetuned",
    },
    "round3_ball_320": {
        "member": "model/best.onnx",
        "manifest": "manifest.json",
        "size": 320,
        "sha256": "ba108ac54736c41e6cdbef1db25e8e06ba05d97e5195566d2b67349bccdc2fd6",
        "role": "ball_only_finetuned",
    },
    "field_source_640": {
        "member": "bundle/model.onnx",
        "manifest": "bundle/manifest.json",
        "size": 640,
        "sha256": "3d4119b1247849a9ecc92ee31498b78b0f9e6ddd6815fc8485db2c2c1abc47c6",
        "role": "field_source_seven_classes",
    },
}


def extract_known_checkpoint(archive_path, config, destination):
    """Reject changed weights; model equivalence cannot be inferred from names."""
    with ZipFile(archive_path) as archive:
        try:
            blob = archive.read(config["member"])
            manifest = json.loads(archive.read(config["manifest"]))
        except KeyError as exc:
            raise ValueError("Expected checkpoint/manifest missing") from exc
    digest = hashlib.sha256(blob).hexdigest()
    if digest != config["sha256"]:
        raise ValueError(f"Unexpected checkpoint SHA256: {digest}")
    if config["role"] == "ball_only_finetuned":
        if manifest.get("training_class_ids") != [0]:
            raise ValueError("Expected a ball-only fine-tuned checkpoint")
        if manifest.get("onnx_sha256") != digest:
            raise ValueError("Model SHA256 does not match archive manifest")
    else:
        profile = manifest.get("profile")
        if profile != "field":
            raise ValueError("Expected field-profile research model")
    destination = Path(destination)
    destination.write_bytes(blob)
    return {"sha256": digest, "role": config["role"], "input": config["size"]}


def summarize_one(result):
    return {
        "model_sha256": result["model_sha256"],
        "model_input_size": result["model_input_size"],
        "legacy_top1": result["old"],
        "classwise_ball": result["new"],
        "opencv_cpu_inference_ms_p50": result["inference_ms_median"],
        "opencv_cpu_inference_ms_p95": result["inference_ms_p95"],
        "menu_transition_negatives": result["auxiliary_negatives"],
    }


def paired_320s(a, b):
    """Compare ball top1 per frame; 320 checkpoints differ in training weights."""
    first = {x["frame_id"]: x for x in a["records"]}
    second = {x["frame_id"]: x for x in b["records"]}
    if first.keys() != second.keys() or len(first) != len(a["records"]):
        raise ValueError("Frozen frames do not align")
    out = {"round2_correct": 0, "round3_correct": 0, "round3_gains": 0,
           "round3_losses": 0, "compared_frames": len(first)}
    for frame_id in first:
        x, y = first[frame_id], second[frame_id]
        if (x["source_frame_index"], x["original_pts_ms"]) != (
                y["source_frame_index"], y["original_pts_ms"]):
            raise ValueError(f"Source timing mismatch {frame_id}")
        hit2 = x["new"]["ball_top1_hit_10px"]
        hit3 = y["new"]["ball_top1_hit_10px"]
        out["round2_correct"] += int(hit2)
        out["round3_correct"] += int(hit3)
        out["round3_gains"] += int(not hit2 and hit3)
        out["round3_losses"] += int(hit2 and not hit3)
    return out


def run_archived_comparison(archives, source_clips, qc):
    """Run same exact frozen benchmark for three authentic stored checkpoints."""
    with TemporaryDirectory() as scratch:
        data = {}
        identities = {}
        for name, config in ARCHIVES.items():
            path = Path(scratch) / f"{name}.onnx"
            identities[name] = extract_known_checkpoint(archives[name], config, path)
            data[name] = run_benchmark(path, source_clips, qc,
                                       config["size"], config["sha256"])
        comparisons = {
            "round2_vs_field640": compare_reports(data["round2_ball_320"],
                                                  data["field_source_640"]),
            "round3_vs_field640": compare_reports(data["round3_ball_320"],
                                                  data["field_source_640"]),
            "round2_vs_round3": paired_320s(data["round2_ball_320"],
                                            data["round3_ball_320"]),
        }
        return {
            "status": "RESEARCH_ONLY_NO_DEPLOYMENT_CHANGE",
            "eligible_for_auto_promotion": False,
            "checkpoint_identities": identities,
            "models": {name: summarize_one(report) for name, report in data.items()},
            "comparisons": comparisons,
            "warnings": [
                "Neither 320 checkpoint is proven identical to the private Windows PC Brain model",
                "Both 320 checkpoints fine-tuned class-id 0 (ball) and used crop/reacquisition",
                "Different checkpoint training and 320/640 input sizes confound any resolution-only claim",
                "Same-match previously reviewed validation is NOT a blind independent benchmark",
                "OpenCV CPU timing here is NOT scrcpy/USB/HUD end-to-end latency",
                "No model should be automatically promoted from these exploratory scores",
            ],
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round2-zip", type=Path, required=True)
    ap.add_argument("--round3-zip", type=Path, required=True)
    ap.add_argument("--field-zip", type=Path, required=True)
    ap.add_argument("--source-clips", type=Path, required=True)
    ap.add_argument("--qc", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    requested = [args.round2_zip, args.round3_zip, args.field_zip,
                 args.source_clips, args.qc]
    for path in requested:
        if not path.is_file():
            ap.error(f"Required research artifact missing: {path}")
    data = run_archived_comparison({
        "round2_ball_320": args.round2_zip,
        "round3_ball_320": args.round3_zip,
        "field_source_640": args.field_zip,
    }, args.source_clips, args.qc)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(json.dumps({"status": data["status"], "models": data["models"],
                      "round2_vs_round3": data["comparisons"]["round2_vs_round3"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
