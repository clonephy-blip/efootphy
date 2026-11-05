"""Paired research benchmark of original PC field decoder versus class-local decoder.

Prerequisites: numpy + opencv-python-headless. Data/weights are local private
artifacts, never uploaded by this script. It reads frozen PART_010 reviewed
ball-visible labels; does not train or modify them.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import cv2
import numpy as np
from tools.pc_brain.field_postprocess import decode_field_detections


def _sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def _score_ball(box, reference, tolerance=10.0):
    if box is None:
        return False, None
    x1, y1, x2, y2 = box
    err = float(np.hypot((x1 + x2) / 2 - reference[0],
                         (y1 + y2) / 2 - reference[1]))
    return err <= tolerance, err


def _old_pc_decoder(output, frame_w, frame_h, size):
    """Mirror previous PC script: argmax, global object NMS, ball top-score."""
    y = np.asarray(output, dtype=np.float32)[0]
    if y.shape[0] != 11:
        raise ValueError(f"Field output must have 11 channels, not {y.shape}")
    trans = y.T
    ids = np.argmax(trans[:, 4:], axis=1)
    confs = np.max(trans[:, 4:], axis=1)
    boxes = []
    rects = []
    scores = []
    ball = None
    ball_conf = -1.0
    for i in range(len(confs)):
        cls, conf = int(ids[i]), float(confs[i])
        cx, cy, bw, bh = [float(x) for x in trans[i, :4]]
        x1, y1 = (cx - bw / 2) * frame_w / size, (cy - bh / 2) * frame_h / size
        x2, y2 = (cx + bw / 2) * frame_w / size, (cy + bh / 2) * frame_h / size
        if cls == 0 and conf >= 0.08:
            if conf > ball_conf:
                ball = [x1, y1, x2, y2]
                ball_conf = conf
        elif conf >= 0.22:
            boxes.append((cls, [x1, y1, x2, y2]))
            rects.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
            scores.append(conf)
    counts = {}
    if rects:
        keep = cv2.dnn.NMSBoxes(rects, scores, score_threshold=0.22,
                               nms_threshold=0.50)
        for i in np.asarray(keep).flatten().tolist() if len(keep) else []:
            cls = boxes[int(i)][0]
            counts[cls] = counts.get(cls, 0) + 1
    return [(ball, ball_conf)] if ball is not None else [], counts


def _summarize(records, mode):
    a = [row[mode] for row in records]
    return {
        "evaluated_visible_frames": len(a),
        "ball_top1_hit_10px": sum(r["ball_top1_hit_10px"] for r in a),
        "ball_any_hit_10px": sum(r["ball_any_hit_10px"] for r in a),
        "frames_with_ball_proposal": sum(r["ball_proposals"] > 0 for r in a),
        "ball_proposals_total": sum(r["ball_proposals"] for r in a),
        "player_proposals": sum(r["player_proposals"] for r in a),
        "referee_proposals": sum(r["referee_proposals"] for r in a),
        "cursor_proposals": sum(r["cursor_proposals"] for r in a),
    }


def run_benchmark(model, source_zip, qc_zip, input_size, expected_model_sha256=None):
    model_path = Path(model)
    model_bytes = model_path.read_bytes()
    model_sha256 = _sha256(model_bytes)
    if expected_model_sha256 is not None and model_sha256 != expected_model_sha256:
        raise ValueError("Model SHA256 mismatch; refusing incomparable weights")
    if input_size not in (320, 640):
        raise ValueError("Only supported 320 or 640 field ONNX is accepted")
    net = cv2.dnn.readNetFromONNX(str(model_path))
    reports, negatives, inference_ms = [], [], []
    with zipfile.ZipFile(source_zip) as sources, zipfile.ZipFile(qc_zip) as qc:
        provenance = {p["clip_id"]: p for p in json.loads(
            sources.read("source-clips/provenance.json"))}
        references = json.loads(qc.read("r2-coarse-fullframe.json"))["records"]
        visible = [r for r in references if r["visibility"] == "visible"]
        if len(visible) != 38 or any(not r["id"].startswith("val-") for r in visible):
            raise ValueError("Frozen 38 visible PART_010 references were changed")
        with tempfile.TemporaryDirectory() as tmp:
            for clip in sorted({r["id"].rsplit("-", 1)[0] for r in visible}):
                p = provenance[clip]
                if p["split"] != "val" or p["video_id"] != "MATCH_01_PART_010":
                    raise ValueError("Validation source provenance mismatch")
                raw = sources.read(f"source-clips/{clip}.mp4")
                if _sha256(raw) != p["clip_sha256"]:
                    raise ValueError("Validation clip bytes changed")
                local = Path(tmp) / f"{clip}.mp4"
                local.write_bytes(raw)
                cap = cv2.VideoCapture(str(local))
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot open {clip}")
                try:
                    subset = [r for r in visible if r["id"].startswith(clip + "-")]
                    for ref in subset:
                        index = ref["source_frame_index"] - p["source_first_frame"]
                        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                        ok, frame = cap.read()
                        if not ok:
                            raise RuntimeError(f"Cannot decode {clip} frame {index}")
                        height, width = frame.shape[:2]
                        blob = cv2.dnn.blobFromImage(
                            frame, scalefactor=1 / 255, size=(input_size, input_size),
                            swapRB=True, crop=False
                        )
                        net.setInput(blob)
                        start = cv2.getTickCount()
                        preds = net.forward()
                        ms = (cv2.getTickCount() - start) * 1000 / cv2.getTickFrequency()
                        inference_ms.append(float(ms))
                        cells = sum((input_size // k)**2 for k in (8, 16, 32))
                        if list(preds.shape) != [1, 11, cells]:
                            raise ValueError(f"Field model shape mismatch: {preds.shape}")
                        old_ball, old_counts = _old_pc_decoder(
                            preds, width, height, input_size)
                        new = decode_field_detections(
                            preds, width, height, input_size=input_size,
                            object_threshold=.22, ball_threshold=.08, iou_threshold=.50)
                        new_ball = new[0]
                        new_counts = {k: len(v) for k, v in new.items()}
                        record = {"frame_id": ref["id"],
                                  "source_frame_index": ref["source_frame_index"],
                                  "original_pts_ms": ref["pts_ms"]}
                        for name, balls, counts in (
                            ("old", old_ball, old_counts), ("new", new_ball, new_counts)):
                            top = balls[0][0] if balls else None
                            hit, error = _score_ball(top, ref["reference_xy_px"])
                            record[name] = {
                                "ball_top1_hit_10px": bool(hit),
                                "ball_top1_error_px": error,
                                "ball_any_hit_10px": any(
                                    _score_ball(b[0], ref["reference_xy_px"])[0]
                                    for b in balls),
                                "ball_proposals": len(balls),
                                "player_proposals": counts.get(4, 0),
                                "referee_proposals": counts.get(6, 0),
                                "cursor_proposals": counts.get(1, 0) + counts.get(2, 0),
                            }
                        reports.append(record)
                finally:
                    cap.release()
        # These are menu/black/transition hard negatives only.
        for name in sorted(n for n in qc.namelist()
                           if n.startswith("negative-qc/") and n.endswith(".jpg")):
            img = cv2.imdecode(np.frombuffer(qc.read(name), np.uint8),
                               cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Invalid negative: {name}")
            h, w = img.shape[:2]
            net.setInput(cv2.dnn.blobFromImage(
                img, 1 / 255, (input_size, input_size), swapRB=True, crop=False))
            preds = net.forward()
            old, _ = _old_pc_decoder(preds, w, h, input_size)
            new = decode_field_detections(preds, w, h, input_size=input_size)
            negatives.append({"name": name, "old_ball": bool(old),
                              "new_ball": bool(new[0])})

    return {
        "status": "research_only_not_model_validation",
        "model_sha256": model_sha256,
        "model_input_size": input_size,
        "validation_source": "MATCH_01_PART_010, assistant-reviewed frozen labels",
        "independent_match": False,
        "labels_independently_verified": False,
        "video_source_clips_are_lossy": True,
        "thresholds": {"ball": 0.08, "other": 0.22, "nms_iou": 0.50,
                       "ball_hit_tolerance_px": 10},
        "inference_scope": "Local OpenCV DNN CPU, not Windows/Android runtime",
        "inference_ms_median": float(np.median(inference_ms)),
        "inference_ms_p95": float(np.percentile(inference_ms, 95)),
        "old": _summarize(reports, "old"),
        "new": _summarize(reports, "new"),
        "auxiliary_negatives": {
            "frames": len(negatives),
            "old_ball_proposal_frames": sum(x["old_ball"] for x in negatives),
            "new_ball_proposal_frames": sum(x["new_ball"] for x in negatives),
        },
        "records": reports,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source-clips", required=True)
    ap.add_argument("--qc", required=True)
    ap.add_argument("--input-size", type=int, required=True, choices=(320, 640))
    ap.add_argument("--expected-model-sha256")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    report = run_benchmark(args.model, args.source_clips, args.qc,
                           args.input_size, args.expected_model_sha256)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
