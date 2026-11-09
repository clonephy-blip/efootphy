"""Optional shadow evidence and calibrated geometry; never a substitute ball detection.

Image positions are original-frame pixels. Camera world units must be metres,
with the ground at Z=0. Light direction is the direction light travels (Z<0).
No camera/light calibration => no metric 3D estimate.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ShadowCandidate:
    xy_px: tuple
    quality: float  # heuristic ranking, NOT a calibrated detection probability
    area_px: float


@dataclass(frozen=True)
class CameraCalibration:
    intrinsic: object
    rotation_world_to_camera: object
    translation_world_to_camera: object
    timestamp_s: float
    camera_epoch: int


@dataclass(frozen=True)
class Ball3DEstimate:
    xyz_m: tuple
    ray_gap_m: float
    reprojection_error_px: float
    source: str = 'calibrated_ball_and_shadow'


def triangulate_ball_shadow(ball_px, shadow_px, calibration, light_direction,
                            now_s, camera_epoch, max_age_s=.25,
                            max_ray_gap_m=.15, max_reprojection_px=5.):
    """Intersect a ball camera ray and the known light ray through its shadow.

    The shadow must already have been associated with THIS ball. Association is
    not solved or certified by this function. Unknown/degenerate geometry returns
    None rather than a guessed height. This is a directional-light model; stadium
    point lights, multiple shadows and game-specific lighting require calibration.
    """
    if calibration is None or light_direction is None:
        return None
    if not all(math.isfinite(float(v)) for v in
               (now_s, calibration.timestamp_s, max_age_s, max_ray_gap_m, max_reprojection_px)):
        return None
    age = now_s - calibration.timestamp_s
    if age < 0 or age > max_age_s or camera_epoch != calibration.camera_epoch:
        return None
    k = np.asarray(calibration.intrinsic, dtype=float)
    r = np.asarray(calibration.rotation_world_to_camera, dtype=float)
    t = np.asarray(calibration.translation_world_to_camera, dtype=float)
    light = np.asarray(light_direction, dtype=float)
    bp, sp = np.asarray(ball_px, dtype=float), np.asarray(shadow_px, dtype=float)
    if k.shape != (3, 3) or r.shape != (3, 3) or t.shape != (3,) or light.shape != (3,) or bp.shape != (2,) or sp.shape != (2,):
        return None
    if not all(np.isfinite(a).all() for a in (k, r, t, light, bp, sp)):
        return None
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-7):
        return None
    if not np.allclose(r @ r.T, np.eye(3), atol=1e-5) or abs(np.linalg.det(r)-1) > 1e-5:
        return None
    norm = float(np.linalg.norm(light))
    if norm < 1e-9:
        return None
    light = light / norm
    if light[2] >= -1e-4:
        return None
    try:
        homography = k @ np.column_stack((r[:, 0], r[:, 1], t))
        ground = np.linalg.solve(homography, [sp[0], sp[1], 1.])
        if abs(ground[2]) < 1e-9:
            return None
        shadow = np.array([ground[0]/ground[2], ground[1]/ground[2], 0.])
        center = -r.T @ t
        ray = r.T @ np.linalg.solve(k, [bp[0], bp[1], 1.])
        ray_norm = float(np.linalg.norm(ray))
        if not math.isfinite(ray_norm) or ray_norm < 1e-9:
            return None
        ray /= ray_norm
        a = np.column_stack((ray, -light))
        singular = np.linalg.svd(a, compute_uv=False)
        if singular[-1] < singular[0] * 1e-3:
            return None
        distance, light_parameter = np.linalg.lstsq(a, shadow-center, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    camera_point = center + distance*ray
    light_point = shadow + light_parameter*light
    point = (camera_point+light_point)*.5
    gap = float(np.linalg.norm(camera_point-light_point))
    if not np.isfinite(point).all() or not math.isfinite(gap) or distance <= 0 or point[2] < 0 or gap > max_ray_gap_m:
        return None
    projected = k @ (r @ point+t)
    if projected[2] <= 0:
        return None
    error = float(np.linalg.norm(projected[:2]/projected[2]-bp))
    if error > max_reprojection_px:
        return None
    return Ball3DEstimate(tuple(float(v) for v in point), gap, error)


def find_shadow_candidates(frame_bgr, ball_xy_px, ball_radius_px=4.,
                           player_boxes=(), search_radius_px=256., max_candidates=4):
    """Bounded local dark-on-grass proposals. No ball hypothesis => no search.

    This is an untrained proposal stage. Player shadows, mowing stripes, HUD
    pixels and compression can still fool it. It must not promote a shadow to an
    observed ball, possession event, ground projection, height or landing point.
    """
    import cv2
    frame = np.asarray(frame_bgr)
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError('Expected uint8 BGR image')
    if ball_xy_px is None:
        return []
    point = np.asarray(ball_xy_px, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        return []
    if not (1 <= ball_radius_px <= 40 and 8 <= search_radius_px <= 512 and 0 < max_candidates <= 8):
        raise ValueError('Unbounded shadow search configuration')
    height, width = frame.shape[:2]
    bx, by = point
    if not (0 <= bx < width and 0 <= by < height):
        return []
    x0, y0 = max(0, int(bx-search_radius_px)), max(0, int(by-search_radius_px))
    x1, y1 = min(width, int(bx+search_radius_px)+1), min(height, int(by+search_radius_px)+1)
    patch = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (15, 15), 0).astype(np.float32)
    contrast = background-gray.astype(np.float32)
    grass = (hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 85) & (hsv[:, :, 1] >= 30)
    # Text outlines and cyan control rings generated many false proposals in
    # the first real-frame pilot. Exclude their immediate neighborhood. This
    # also sacrifices recall near white pitch lines; it is not a learned mask.
    bright_neutral = (hsv[:, :, 1] <= 70) & (hsv[:, :, 2] >= 150)
    bright_cyan = ((hsv[:, :, 0] >= 80) & (hsv[:, :, 0] <= 110) &
                   (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] >= 140))
    overlay_edges = cv2.dilate((bright_neutral | bright_cyan).astype(np.uint8), np.ones((9,9),np.uint8))
    dark = grass & (contrast >= 16) & (gray < background*.75)
    mask = (dark.astype(np.uint8)*255)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    proposals = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 3 or area > min(700., math.pi*ball_radius_px**2*12):
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if x == 0 or y == 0 or x+w >= patch.shape[1] or y+h >= patch.shape[0]:
            continue
        if max(w,h)/max(1,min(w,h)) > 6:
            continue
        moment = cv2.moments(contour)
        if moment['m00'] <= 0:
            continue
        cx, cy = moment['m10']/moment['m00']+x0, moment['m01']/moment['m00']+y0
        if overlay_edges[int(cy-y0),int(cx-x0)]:
            continue
        distance = math.hypot(cx-bx,cy-by)
        if distance < 1.5*ball_radius_px or distance > search_radius_px:
            continue
        if any(box[0]-2 <= cx <= box[2]+2 and box[1]-2 <= cy <= box[3]+2 for box in player_boxes):
            continue
        hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
        solidity = area/max(hull_area,1.)
        if solidity < .5:
            continue
        component = np.zeros((h,w),np.uint8)
        cv2.drawContours(component,[contour],-1,255,-1,offset=(-x,-y))
        mean_contrast = float(contrast[y:y+h,x:x+w][component>0].mean())
        quality = min(1.,max(0.,min(1.,mean_contrast/50.)*solidity*math.exp(-distance/search_radius_px)))
        proposals.append(ShadowCandidate((cx,cy),quality,area))
    return sorted(proposals,key=lambda p:p.quality,reverse=True)[:max_candidates]
