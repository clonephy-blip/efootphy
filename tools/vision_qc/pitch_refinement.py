"""Bounded paint alignment of an existing semantic pitch calibration.

Image evidence is not ground truth. This offline experiment never certifies a
calibration, metre scale, Android performance, or training-label eligibility.
No reference coordinates or manually fitted homography enter the optimizer.
"""
from dataclasses import asdict, dataclass
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

from .pitch_tracking import field_mask, project


@dataclass(frozen=True)
class RefinementConfig:
    max_width: int = 960
    spacing_px: float = 4.0
    capture_px: float = 16.0
    min_samples: int = 80
    max_evaluations: int = 80
    max_displacement_px: float = 32.0
    prior_sigma_px: float = 12.0

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(v) and v > 0 for v in values.values()):
            raise ValueError('Refinement budgets must be finite and positive')
        for name in ('max_width', 'min_samples', 'max_evaluations'):
            if not isinstance(values[name], int):
                raise ValueError('Integer refinement budget required')
        if self.max_width < 64 or self.min_samples < 16:
            raise ValueError('Insufficient refinement budget')


def _sample_template(H, lines, width, height, spacing):
    """Fixed image samples from the seed; horizon-crossing pieces are omitted."""
    samples, families = [], []
    for family, line in enumerate(lines):
        world = np.asarray(line, float)
        if world.ndim != 2 or world.shape[1] != 2 or not np.isfinite(world).all():
            raise ValueError('Expected finite world polylines')
        for a, b in zip(world, world[1:]):
            dense = np.linspace(a, b, max(2, int(np.linalg.norm(a-b)/.5)+2))
            homogeneous = np.c_[dense, np.ones(len(dense))] @ H.T
            for first, second in zip(homogeneous, homogeneous[1:]):
                if first[2]*second[2] <= 1e-10:
                    continue
                pair = np.asarray([first[:2]/first[2], second[:2]/second[2]])
                if not np.isfinite(pair).all() or np.max(np.abs(pair)) > 1e6:
                    continue
                ok, start, end = cv2.clipLine((0, 0, width, height),
                    tuple(np.rint(pair[0]).astype(int)), tuple(np.rint(pair[1]).astype(int)))
                if not ok:
                    continue
                length = np.linalg.norm(np.subtract(start, end))
                if length < .5:
                    continue
                points = np.linspace(start, end, max(2, int(length/spacing)+2))
                samples.extend(points); families.extend([family]*len(points))
    return np.asarray(samples, float).reshape(-1, 2), np.asarray(families, int)


def _bilinear(array, points, outside):
    h, w = array.shape
    x, y = points.T
    valid = np.isfinite(points).all(1) & (x >= 0) & (x < w-1) & (y >= 0) & (y < h-1)
    out = np.full(len(points), float(outside))
    xx, yy = x[valid], y[valid]
    ix, iy = np.floor(xx).astype(int), np.floor(yy).astype(int)
    dx, dy = xx-ix, yy-iy
    out[valid] = ((1-dx)*(1-dy)*array[iy, ix] + dx*(1-dy)*array[iy, ix+1]
        + (1-dx)*dy*array[iy+1, ix] + dx*dy*array[iy+1, ix+1])
    return out


def refine_calibration(frame, seed_H, world_lines, hud_spec=None, config=None):
    """Return (selected H, diagnostics); unsupported candidates retain the seed.

    frame: original-resolution uint8 BGR; H: world XY -> original image pixels.
    hud_spec uses original pixels: rects[x,y,w,h] and circles[x,y,r].
    All visibility decisions and residual denominators are fixed before fitting.
    """
    started = time.perf_counter(); config = config or RefinementConfig()
    if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
            or frame.shape[2] != 3 or min(frame.shape[:2]) < 32):
        raise ValueError('Expected a uint8 BGR image at least 32x32')
    H = np.asarray(seed_H, float)
    if H.shape != (3, 3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H) != 3 or abs(H[2, 2]) < 1e-10:
        raise ValueError('Invalid seed homography')
    H = H/H[2, 2]
    metrics = {'accepted': False, 'config': asdict(config), 'models_validated': False,
               'automatic_training_import_allowed': False, 'metric_scale_verified': False}

    def finish(reason, selected=H):
        metrics.update(reason=reason, elapsed_ms=(time.perf_counter()-started)*1000)
        return selected.copy(), metrics

    h, w = frame.shape[:2]; scale = min(1., config.max_width/w)
    ww, hh = int(round(w*scale)), int(round(h*scale))
    T = np.diag([ww/w, hh/h, 1.]); work = cv2.resize(frame, (ww, hh))
    hud = np.full((h, w), 255, np.uint8)
    for x, y, rw, rh in (hud_spec or {}).get('rects', []):
        cv2.rectangle(hud, (int(x), int(y)), (int(x+rw), int(y+rh)), 0, -1)
    for x, y, r in (hud_spec or {}).get('circles', []):
        cv2.circle(hud, (int(x), int(y)), int(r), 0, -1)
    hud = cv2.resize(hud, (ww, hh), interpolation=cv2.INTER_NEAREST)
    mask = field_mask(work, hud)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, np.array([0, 0, 125]), np.array([179, 100, 255]))
    white = cv2.bitwise_and(white, mask)
    if np.count_nonzero(white) < 40:
        return finish('insufficient_paint')
    distance = cv2.distanceTransform(255-white, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    xy, families = _sample_template(T@H, world_lines, ww, hh, config.spacing_px)
    if not len(xy):
        return finish('no_visible_template')
    valid = (_bilinear(mask, xy, 0) > 250) & (_bilinear(distance, xy, 999) <= config.capture_px)
    xy, families = xy[valid], families[valid]
    counts = np.bincount(families) if len(families) else np.array([])
    metrics.update(fixed_samples=len(xy), visible_families=int(np.count_nonzero(counts >= 12)))
    if len(xy) < config.min_samples or np.count_nonzero(counts >= 12) < 3:
        return finish('insufficient_distributed_evidence')
    # Equalize long/short line families without allowing tiny fragments to dominate.
    weights = np.sqrt(np.median(counts[counts >= 12])/np.maximum(12, counts[families]))
    weights /= np.sqrt(np.mean(weights**2))
    N = np.array([[1/ww, 0, -.5], [0, 1/ww, -hh/(2*ww)], [0, 0, 1.]])
    invN = np.linalg.inv(N)
    anchors = np.asarray([[x, y] for x in np.linspace(0, ww-1, 3) for y in np.linspace(0, hh-1, 3)])

    def correction(p):
        return invN @ np.array([[1+p[0], p[1], p[2]], [p[3], 1+p[4], p[5]], [p[6], p[7], 1.]]) @ N

    def distances(D):
        moved = project(D, xy)
        values = _bilinear(distance, moved, config.capture_px*2)
        # Samples cannot disappear into HUD or outside the image to improve score.
        return np.maximum(values, np.where(_bilinear(mask, moved, 0) > 250, 0., config.capture_px*2))

    def residual(p):
        D = correction(p)
        return np.r_[distances(D)*weights, ((project(D, anchors)-anchors)/config.prior_sigma_px).ravel()]

    bounds = np.array([.08, .08, .04, .08, .08, .04, .08, .08])
    fit = least_squares(residual, np.zeros(8), bounds=(-bounds, bounds), loss='soft_l1',
                        f_scale=1.5, diff_step=1e-4, max_nfev=config.max_evaluations)
    D = correction(fit.x); before = distances(np.eye(3)); after = distances(D)
    moved = project(D, anchors); delta = np.linalg.norm(moved-anchors, axis=1)
    corners = np.array([[0, 0], [ww-1, 0], [ww-1, hh-1], [0, hh-1]], np.float32)
    warped = project(D, corners).astype(np.float32)
    ratio = float(cv2.contourArea(warped, oriented=True)/cv2.contourArea(corners, oriented=True))
    metrics.update(evaluations=int(fit.nfev), optimizer_converged=bool(fit.success),
        mean_distance_before=float(np.mean(before)), mean_distance_after=float(np.mean(after)),
        p95_distance_before=float(np.percentile(before, 95)), p95_distance_after=float(np.percentile(after, 95)),
        support_before=float(np.mean(before <= 3)), support_after=float(np.mean(after <= 3)),
        max_displacement_work_px=float(delta.max()), image_area_ratio=ratio,
        candidate_H_world_to_image=(np.linalg.inv(T)@D@T@H).tolist())
    denominators = np.c_[corners, np.ones(4)]@D[2]
    # Reaching a fixed iteration budget is not a numerical failure. Every finite
    # proposal still has to pass the unchanged geometric and paint gates below.
    # Optimizer convergence is diagnostic; it never substitutes for external QC.
    if fit.status < 0 or not np.isfinite(fit.x).all() or not np.isfinite(fit.cost):
        return finish('optimizer_failure')
    if (denominators.min()*denominators.max() <= 0 or not cv2.isContourConvex(warped)
            or not .85 < ratio < 1.18 or delta.max() > config.max_displacement_px):
        return finish('unsafe_geometric_change')
    if (np.mean(after) >= .92*np.mean(before) or np.percentile(after, 95) > np.percentile(before, 95)+.25
            or np.mean(after <= 3) < np.mean(before <= 3)):
        return finish('paint_evidence_not_improved')
    selected = np.linalg.inv(T)@D@T@H
    metrics['accepted'] = True
    return finish('image_evidence_improved_requires_external_qc', selected/selected[2, 2])
