"""Field YOLO11 output decoding with explicit ball-channel recall and per-class NMS.

The field ONNX contract is (1, 11, N): 4 xywh coordinates in 320x320 stretched
input pixels, then seven class confidence channels. This is NOT the minimap
model (different class count / meaning). No training labels or future frames
are used in inference.
"""
import numpy as np

NUM_CLASSES = 7
CLASS_BALL = 0
CLASS_CURSOR = 1
CLASS_CURSOR_SEC = 2
CLASS_GOALKEEPER = 3
CLASS_PLAYER = 4
CLASS_POWERBAR = 5
CLASS_REFEREE = 6


def _iou(one, others):
    if not len(others):
        return np.empty(0, dtype=np.float32)
    x1 = np.maximum(one[0], others[:, 0])
    y1 = np.maximum(one[1], others[:, 1])
    x2 = np.minimum(one[2], others[:, 2])
    y2 = np.minimum(one[3], others[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    one_area = (one[2] - one[0]) * (one[3] - one[1])
    other_area = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / np.maximum(1e-8, one_area + other_area - inter)


def _class_nms(boxes, confidences, max_candidates=250, max_kept=50, iou_threshold=0.50):
    """Stable confidence-ranked greedy NMS, solely within one class."""
    order = np.argsort(-confidences, kind="stable")[:max_candidates]
    keep = []
    while len(order) and len(keep) < max_kept:
        first = int(order[0])
        keep.append(first)
        if len(order) == 1:
            break
        following = order[1:]
        order = following[_iou(boxes[first], boxes[following]) <= iou_threshold]
    return keep


def decode_field_detections(model_output, frame_width, frame_height,
                            input_size=320, object_threshold=0.22,
                            ball_threshold=0.08, iou_threshold=0.50):
    """Return class id -> [(bbox xyxy in original pixels, confidence), ...].

    Balls are considered from their *own channel*, even where another class
    has a higher score at the same anchor. Only the best non-ball class at a
    position is considered, to avoid duplicating every player as 6 classes.
    """
    if frame_width <= 0 or frame_height <= 0 or input_size <= 0:
        raise ValueError("Invalid frame or model input dimensions")
    output = np.asarray(model_output, dtype=np.float32)
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    if output.ndim != 2 or output.shape[0] != 4 + NUM_CLASSES:
        raise ValueError(f"Expected field model (1, 11, N), got {output.shape}")
    if output.shape[1] == 0:
        return {cid: [] for cid in range(NUM_CLASSES)}

    # Stretch to square matches field dataset preprocessing, NOT letterbox.
    xywh = output[:4].T
    scores = output[4:].T
    sx, sy = frame_width / float(input_size), frame_height / float(input_size)
    cx, cy, bw, bh = xywh.T
    boxes = np.column_stack(((cx - bw * 0.5) * sx, (cy - bh * 0.5) * sy,
                             (cx + bw * 0.5) * sx, (cy + bh * 0.5) * sy))
    valid = np.isfinite(xywh).all(axis=1) & np.isfinite(scores).all(axis=1)
    valid &= (bw > 0) & (bh > 0)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0.0, float(frame_width))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0.0, float(frame_height))
    valid &= (boxes[:, 2] - boxes[:, 0] >= 0.5) & (boxes[:, 3] - boxes[:, 1] >= 0.5)

    # Keep ball proposals independent of non-ball argmax. Ball NMS is separate
    # and top-1 ball selection happens *after* class-local suppression.
    ball_candidates = valid & (scores[:, CLASS_BALL] >= ball_threshold)
    other_classes = np.argmax(scores[:, 1:], axis=1) + 1
    other_scores = scores[np.arange(len(scores)), other_classes]

    detected = {}
    for cid in range(NUM_CLASSES):
        if cid == CLASS_BALL:
            mask = ball_candidates
            conf = scores[:, CLASS_BALL]
            max_kept = 8
        else:
            mask = valid & (other_classes == cid) & (other_scores >= object_threshold)
            conf = other_scores
            max_kept = 50
        ids = np.flatnonzero(mask)
        if not len(ids):
            detected[cid] = []
            continue
        local = _class_nms(boxes[ids], conf[ids], max_kept=max_kept,
                           iou_threshold=iou_threshold)
        detected[cid] = [(boxes[ids[k]].tolist(), float(conf[ids[k]])) for k in local]
    return detected


def cluster_kit_colors(colors, prior=None, min_separation=28.0, ambiguous_margin=8.0):
    """Classify sampled torso BGR colors; -1 means unavailable/ambiguous.

    If no two reliably different kit colors appear, do NOT manufacture teams.
    Return (team ids, optional pair of stable BGR centers). This is a heuristic;
    stable roster identity and game-level team accuracy remain unverified.
    """
    assigned = [-1] * len(colors)
    valid_ids, valid = [], []
    for index, color in enumerate(colors):
        if color is None:
            continue
        a = np.asarray(color, dtype=np.float32)
        if a.shape != (3,) or not np.isfinite(a).all():
            continue
        valid_ids.append(index)
        valid.append(a)
    if not valid:
        return assigned, prior
    data = np.stack(valid)
    if prior is None:
        if len(data) < 2:
            return assigned, None
        d2 = np.sum((data[:, None, :] - data[None, :, :]) ** 2, axis=2)
        a, b = np.unravel_index(int(d2.argmax()), d2.shape)
        if float(np.sqrt(d2[a, b])) < min_separation:
            return assigned, None
        centers = np.stack((data[a], data[b]))
    else:
        centers = np.asarray(prior, dtype=np.float32).copy()
        if centers.shape != (2, 3) or not np.isfinite(centers).all():
            return assigned, None
        if float(np.linalg.norm(centers[0] - centers[1])) < min_separation:
            return assigned, None
    for _ in range(3):
        distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        groups = np.argmin(distances, axis=1)
        for k in range(2):
            matches = data[groups == k]
            if len(matches):
                # Smooth color changes, do not flip team IDs with detection order.
                centers[k] = centers[k] * 0.8 + np.mean(matches, axis=0) * 0.2
    if float(np.linalg.norm(centers[0] - centers[1])) < min_separation:
        return assigned, prior
    distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
    for i, row in enumerate(distances):
        if abs(float(row[0] - row[1])) >= ambiguous_margin:
            assigned[valid_ids[i]] = int(np.argmin(row))
    return assigned, (centers[0], centers[1])
