"""Full-resolution ball proposals and an optional learned patch scorer.

Candidates are appearance evidence, never proof of possession or 3-D position.
No ground-truth crop, player box, future frame, or minimap is used at inference.
"""
from dataclasses import dataclass

import cv2
import numpy as np


def ball_proposals(frame, max_candidates=256, excluded_rects=()):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError('Expected uint8 BGR image')
    if not 1 <= max_candidates <= 1024:
        raise ValueError('Invalid candidate budget')
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2].astype(np.float32)
    # Multi-scale bright, compact features: retain the original pixel resolution.
    response = np.maximum.reduce([
        value - cv2.GaussianBlur(value, (0, 0), sigma)
        for sigma in (2., 4., 7.)])
    white = (hsv[:, :, 1] < 145) & (value > 130)
    green = cv2.inRange(hsv, (28, 35, 25), (90, 255, 255))
    # Nearby grass includes airborne balls; the ball itself need not be green.
    field = cv2.dilate(green, np.ones((41, 41), np.uint8)) > 0
    response[~(white & field)] = 0
    for x0, y0, x1, y1 in excluded_rects:
        if not (0 <= x0 < x1 <= frame.shape[1] and 0 <= y0 < y1 <= frame.shape[0]):
            raise ValueError('Invalid excluded rectangle')
        response[int(y0):int(y1), int(x0):int(x1)] = 0
    maxima = (response >= cv2.dilate(response, np.ones((9, 9), np.uint8))) & (response >= 12.)
    ys, xs = np.nonzero(maxima)
    scores = response[ys, xs]
    order = np.argsort(-scores, kind='stable')
    chosen = []
    for i in order:
        x, y = int(xs[i]), int(ys[i])
        if all((x-a[0])**2+(y-a[1])**2 >= 100 for a in chosen):
            chosen.append((x, y, float(scores[i])))
        if len(chosen) == max_candidates:
            break
    return chosen


def patch_batch(frame, points, side=48):
    if not points:
        return np.empty((0, 3, side, side), np.float32)
    if side < 16 or side % 2:
        raise ValueError('Even patch side >=16 required')
    border = side//2
    padded = cv2.copyMakeBorder(frame, border, border, border, border, cv2.BORDER_REFLECT_101)
    patches = []
    for p in points:
        x, y = np.rint(p[:2]).astype(int)
        if not (0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]):
            raise ValueError('Patch centre outside image')
        patches.append(padded[y:y+side, x:x+side, ::-1].transpose(2, 0, 1))
    return np.ascontiguousarray(patches, dtype=np.float32)/255.


def make_patch_model():
    import torch.nn as nn
    # A small appearance baseline, not a pretrained temporal model.
    return nn.Sequential(
        nn.Conv2d(3, 12, 5, stride=2, padding=2), nn.ReLU(),
        nn.Conv2d(12, 24, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.ReLU(),
        nn.Flatten(), nn.Linear(32*6*6, 32), nn.ReLU(), nn.Linear(32, 1))


@dataclass(frozen=True)
class BallObservation:
    x: float
    y: float
    score: float
    source: str = 'detector'


class CausalBallTrack:
    """Time-based single-ball association with bounded, explicit prediction.

    Screen-space extrapolation is deliberately short. A scene cut must reset it.
    It cannot recover an unobserved trajectory or guarantee the correct identity.
    """
    def __init__(self, acquire=.9, maintain=.65, max_prediction_ms=100., gate_px=75., max_flow_without_detector_ms=80.):
        if not 0 < maintain <= acquire <= 1 or max_prediction_ms <= 0 or gate_px <= 0 or max_flow_without_detector_ms<=0:
            raise ValueError('Invalid tracking parameters')
        self.acquire, self.maintain = acquire, maintain
        self.max_prediction_ms, self.gate_px = max_prediction_ms, gate_px
        self.max_flow_without_detector_ms=max_flow_without_detector_ms
        self.reset()

    def reset(self):
        self.last_pts = None
        self.observed_pts = None
        self.detector_pts = None
        self.position = None
        self.velocity = np.zeros(2, np.float64)

    def update(self, pts_ms, observations, scene_cut=False):
        if not np.isfinite(pts_ms) or pts_ms < 0:
            raise ValueError('Invalid PTS')
        if self.last_pts is not None and pts_ms <= self.last_pts:
            raise ValueError('PTS must increase')
        for o in observations:
            if not np.isfinite([o.x, o.y, o.score]).all() or not 0 <= o.score <= 1 or o.source not in ('detector','optical_flow'):
                raise ValueError('Invalid observation')
        if scene_cut:
            self.reset()
        self.last_pts = float(pts_ms)
        age = None if self.observed_pts is None else pts_ms-self.observed_pts
        if age is not None and age > self.max_prediction_ms:
            self.position = None
            self.velocity[:] = 0
            self.observed_pts = None
            age = None
        predicted = None if self.position is None else self.position+self.velocity*(age/1000.)
        eligible = []
        for o in observations:
            if o.source=='optical_flow' and (self.detector_pts is None or pts_ms-self.detector_pts>self.max_flow_without_detector_ms):
                continue
            distance = 0. if predicted is None else float(np.linalg.norm(np.array([o.x, o.y])-predicted))
            threshold = self.acquire if predicted is None else self.maintain
            # Grow the association gate with elapsed wall-clock time, not frame count.
            gate = self.gate_px*(1.+(age or 0.)/100.)
            if o.score >= threshold and (predicted is None or distance <= gate):
                eligible.append((o.score-.15*distance/gate, o))
        if eligible:
            o = max(eligible, key=lambda item: item[0])[1]
            position = np.array([o.x, o.y], dtype=np.float64)
            if self.position is not None and age > 0:
                measured = (position-self.position)/(age/1000.)
                self.velocity = .5*self.velocity+.5*measured
            self.position = position
            self.observed_pts = float(pts_ms)
            if o.source=='detector':self.detector_pts=float(pts_ms)
            return {'state':'OBSERVED', 'xy_px':position.tolist(), 'observation_score':o.score,
                    'observation_source':o.source,
                    'last_detector_pts_ms':self.detector_pts,
                    'last_observed_pts_ms':float(pts_ms), 'prediction_age_ms':0.}
        if predicted is not None:
            return {'state':'PREDICTED', 'xy_px':predicted.tolist(), 'observation_score':None,
                    'observation_source':None,
                    'last_observed_pts_ms':self.observed_pts, 'prediction_age_ms':float(age)}
        return {'state':'LOST', 'xy_px':None, 'observation_score':None,
                'observation_source':None,
                'last_observed_pts_ms':None, 'prediction_age_ms':None}


def ball_flow_observation(previous_gray, current_gray, previous_xy):
    """A current-image flow candidate with forward/backward and patch checks.

    These checks can still accept the wrong object. They are image evidence,
    not a probability of correct identity, and require separate video QC.
    """
    if previous_gray is None or previous_xy is None:
        return None
    if previous_gray.shape!=current_gray.shape or previous_gray.dtype!=np.uint8 or current_gray.dtype!=np.uint8:
        raise ValueError('Matching uint8 gray frames required')
    p=np.array(previous_xy,dtype=np.float32).reshape(1,1,2)
    options=dict(winSize=(25,25),maxLevel=3,criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,25,.01))
    nxt,status,error=cv2.calcOpticalFlowPyrLK(previous_gray,current_gray,p,None,**options)
    if nxt is None or not status[0,0] or error[0,0]>25:
        return None
    back,valid,_=cv2.calcOpticalFlowPyrLK(current_gray,previous_gray,nxt,None,**options)
    if back is None or not valid[0,0] or np.linalg.norm(back-p)>1.5:
        return None
    x,y=nxt[0,0];h,w=current_gray.shape
    if not (14<=x<w-14 and 14<=y<h-14):
        return None
    old=cv2.getRectSubPix(previous_gray,(25,25),tuple(map(float,p[0,0])))
    new=cv2.getRectSubPix(current_gray,(25,25),(float(x),float(y)))
    if old.std()<8 or new.std()<8:
        return None
    correlation=float(cv2.matchTemplate(new,old,cv2.TM_CCOEFF_NORMED)[0,0])
    if not np.isfinite(correlation) or correlation<.7:
        return None
    return BallObservation(float(x),float(y),float(min(.85,correlation)),'optical_flow')


def control_marker_candidates(frame):
    """Cyan triangle candidates; not roster identity or possession labels."""
    mask = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), (78, 110, 140), (104, 255, 255))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    markers = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not 8 <= area <= frame.shape[0]*frame.shape[1]*.001:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        approx = cv2.approxPolyDP(contour, .06*cv2.arcLength(contour, True), True)
        if .5 <= w/h <= 1.8 and len(approx) == 3 and .3 <= area/(w*h) <= .7:
            tip = max(approx[:, 0].tolist(), key=lambda p:p[1])
            markers.append({'tip_xy_px':tip, 'box_xyxy_px':[x,y,x+w,y+h],
                            'state':'UNASSOCIATED', 'player_identity':None, 'possession':None})
    return markers


def associate_control_markers(markers, detections, image_height):
    """Require one nearby body candidate; keep ambiguous markers unassociated."""
    scale=image_height/1080.
    output=[]
    for marker in markers:
        copy=dict(marker);x,y=marker['tip_xy_px'];matches=[]
        for i,d in enumerate(detections):
            if d['class_name'] not in ('player','goalkeeper'):
                continue
            x0,y0,x1,y1=d['box_xyxy_px']
            if x0-6*scale<=x<=x1+6*scale and -8*scale<=y0-y<=50*scale:
                matches.append(i)
        if len(matches)==1:
            copy.update(state='ASSOCIATED_UI_CONTROL_CANDIDATE',detection_index=matches[0])
        elif len(matches)>1:
            copy['state']='AMBIGUOUS'
        output.append(copy)
    return output
