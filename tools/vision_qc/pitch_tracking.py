"""Image-based pitch motion with gated LK and optional learned XFeat relocalization.

The initializer owns semantic calibration. This tracker never certifies metre scale.
"""
from dataclasses import dataclass
import math
from pathlib import Path
import sys
import time

import numpy as np


def project(H,points):
    p=np.asarray(points,dtype=float);q=np.c_[p,np.ones(len(p))]@np.asarray(H).T
    if not np.isfinite(q).all() or np.any(np.abs(q[:,2])<1e-8):raise ValueError('Invalid projective horizon')
    return q[:,:2]/q[:,2,None]


def estimate_motion(source,target,size,min_inliers=12):
    import cv2
    a=np.asarray(source,np.float32).reshape(-1,2);b=np.asarray(target,np.float32).reshape(-1,2)
    if len(a)!=len(b) or len(a)<min_inliers or not np.isfinite(a).all() or not np.isfinite(b).all():return None,{'reason':'too_few_finite_matches'}
    H,mask=cv2.findHomography(a,b,cv2.RANSAC,2.,maxIters=1000,confidence=.995)
    if H is None or mask is None or not np.isfinite(H).all():return None,{'reason':'no_homography'}
    keep=mask.ravel().astype(bool);n=int(keep.sum());ratio=n/len(a);w,h=size
    coverage=min(abs(cv2.contourArea(cv2.convexHull(p[keep]))) for p in [a,b])/(w*h) if n>=3 else 0
    try:error=np.linalg.norm(project(H,a[keep])-b[keep],axis=1)
    except ValueError:return None,{'reason':'projective_horizon'}
    metrics={'matches':len(a),'inliers':n,'inlier_ratio':ratio,'coverage':coverage,'median_reprojection_px':float(np.median(error)) if n else None}
    if n<min_inliers or ratio<.65 or coverage<.025 or not n or np.median(error)>1.5:
        return None,{**metrics,'reason':'weak_geometric_consensus'}
    corners=np.float32([[0,0],[w,0],[w,h],[0,h]])
    denominators=np.c_[corners,np.ones(4)]@H[2]
    if np.min(np.abs(denominators))<1e-8 or np.min(denominators)*np.max(denominators)<=0 or abs(H[2,2])<1e-8:
        return None,{**metrics,'reason':'horizon_crosses_image'}
    try:warped=project(H,corners)
    except ValueError:return None,{**metrics,'reason':'invalid_corners'}
    area=float(cv2.contourArea(warped.astype(np.float32),oriented=True))/(w*h)
    if not cv2.isContourConvex(warped.astype(np.float32)) or not .45<area<2.2:return None,{**metrics,'reason':'flip_or_extreme_scale'}
    return H/H[2,2],{**metrics,'reason':'accepted'}


def field_mask(frame,hud=None):
    import cv2
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    green=cv2.inRange(hsv,np.array([25,35,22]),np.array([100,255,255]))
    green=cv2.morphologyEx(green,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(green)
    if count<2:return np.zeros(frame.shape[:2],np.uint8)
    largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));green=np.uint8(labels==largest)*255
    green=cv2.dilate(green,np.ones((7,7),np.uint8))
    if hud is not None:green=cv2.bitwise_and(green,hud)
    return green


class XFeatMatcher:
    """Loads reviewed local upstream code/weights. No torch.hub or network access."""
    def __init__(self,repository,weights,top_k=1400):
        import torch
        root=Path(repository).resolve()
        if not (root/'modules/xfeat.py').is_file():raise ValueError('Expected the pinned XFeat checkout')
        sys.path.insert(0,str(root))
        from modules.xfeat import XFeat
        state=torch.load(weights,map_location='cpu',weights_only=True)
        self.model=XFeat(weights=state,top_k=top_k);self.calls=0

    def extract(self,frame,mask):
        import torch
        rgb=np.ascontiguousarray(frame[:,:,::-1]);features=self.model.detectAndCompute(rgb)[0]
        xy=features['keypoints'].cpu().numpy();desc=features['descriptors'].cpu().numpy()
        ij=np.rint(xy).astype(int);valid=(ij[:,0]>=0)&(ij[:,0]<mask.shape[1])&(ij[:,1]>=0)&(ij[:,1]<mask.shape[0])
        valid[valid]&=mask[ij[valid,1],ij[valid,0]]>0;self.calls+=1
        return xy[valid],desc[valid]

    def match(self,reference,target):
        a,da=reference;b,db=target
        if min(len(a),len(b))<16:return np.empty((0,2)),np.empty((0,2))
        similarity=np.clip(da@db.T,-1.,1.)
        nearest=similarity.argmax(1);reverse=similarity.argmax(0)
        best=similarity[np.arange(len(a)),nearest];second=np.partition(similarity,-2,axis=1)[:,-2]
        # MNN plus ambiguity margin prevents repetitive pitch stripes dominating.
        valid=(reverse[nearest]==np.arange(len(a)))&(best>.8)&(best-second>1e-5)&((1-best)<.8*(1-second))
        return a[valid],b[nearest[valid]]


class SiftMatcher:
    """Classical control with the same mask and geometric acceptance gates."""
    def __init__(self):
        import cv2
        self.detector=cv2.SIFT_create(nfeatures=1400);self.calls=0

    def extract(self,frame,mask):
        import cv2
        keypoints,descriptors=self.detector.detectAndCompute(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),mask);self.calls+=1
        return np.asarray([k.pt for k in keypoints],np.float32).reshape(-1,2),descriptors

    def match(self,reference,target):
        import cv2
        a,da=reference;b,db=target
        if da is None or db is None or min(len(a),len(b))<16:return np.empty((0,2)),np.empty((0,2))
        matcher=cv2.BFMatcher();pairs=matcher.knnMatch(da,db,k=2);reverse=matcher.match(db,da)
        accepted=[x for x,y in pairs if x.distance<.75*y.distance and reverse[x.trainIdx].trainIdx==x.queryIdx]
        return a[[m.queryIdx for m in accepted]],b[[m.trainIdx for m in accepted]]


@dataclass
class Observation:
    status:str
    homography:object
    source:str
    timestamp_ms:float
    metrics:dict


class PitchTracker:
    def __init__(self,matcher=None,max_width=640,reanchor_every=20,max_gap_ms=250,world_lines=None):
        if max_width<64 or reanchor_every<1 or max_gap_ms<=0:raise ValueError('Invalid tracker budget')
        self.matcher=matcher;self.max_width=max_width;self.reanchor_every=reanchor_every;self.max_gap_ms=max_gap_ms
        self.world_lines=world_lines or [];self.reference=None;self.H=None;self.prev_gray=None;self.points=None
        self.last_time=None;self.last_input_time=None;self.frame_index=0;self.reference_features=None

    def _prepare(self,frame,hud_spec):
        import cv2
        if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3 or min(frame.shape[:2])<32:
            raise ValueError('Expected a uint8 BGR frame of at least 32x32')
        h,w=frame.shape[:2];scale=min(1,self.max_width/w);ww=max(32,int(round(w*scale/32))*32);hh=max(32,int(round(h*scale/32))*32)
        work=cv2.resize(frame,(ww,hh));T=np.diag([ww/w,hh/h,1.]);hud=np.full((h,w),255,np.uint8)
        for x,y,bw,bh in hud_spec.get('rects',[]):cv2.rectangle(hud,(int(x),int(y)),(int(x+bw),int(y+bh)),0,-1)
        for x,y,r in hud_spec.get('circles',[]):cv2.circle(hud,(int(x),int(y)),int(r),0,-1)
        hud=cv2.resize(hud,(ww,hh),interpolation=cv2.INTER_NEAREST)
        return work,cv2.cvtColor(work,cv2.COLOR_BGR2GRAY),field_mask(work,hud),T

    def seed(self,frame,H_world_to_image,timestamp_ms,hud_spec=None,source='manual_reference'):
        import cv2
        H=np.asarray(H_world_to_image,float)
        if H.shape!=(3,3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H)!=3 or abs(H[2,2])<1e-8:raise ValueError('Invalid seed calibration')
        if not math.isfinite(timestamp_ms):raise ValueError('Invalid timestamp')
        self.hud_spec=hud_spec or {};work,gray,mask,T=self._prepare(frame,self.hud_spec)
        points=cv2.goodFeaturesToTrack(gray,500,.01,5,mask=mask)
        line_metrics,consistent=self._white_support(work,mask,H,T)
        if np.mean(mask>0)<.08 or points is None or len(points)<16 or not consistent:
            self.reference=None;self.H=None;self.prev_gray=None;self.points=None
            return Observation('LOST',None,'none',timestamp_ms,{'reason':'seed_quality_failed',**line_metrics})
        self.shape=frame.shape[:2];self.T=T;self.H=H/H[2,2];self.seed_H=self.H.copy();self.seed_source=source
        self.reference=work;self.reference_gray=gray;self.reference_mask=mask;self.prev_gray=gray
        self.reference_features=self.matcher.extract(work,mask) if self.matcher else None
        self.points=points;self.last_time=timestamp_ms;self.last_input_time=timestamp_ms;self.frame_index=0
        return Observation('TRACKING',self.H.copy(),'seed',timestamp_ms,{**line_metrics,'metric_scale_verified':False,'seed_source':source})

    def _white_support(self,frame,mask,H,T):
        if not self.world_lines:return {'white_support':None,'white_samples':0},True
        import cv2
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV);white=cv2.inRange(hsv,np.array([0,0,125]),np.array([179,100,255]))
        distances=cv2.distanceTransform(255-white,cv2.DIST_L2,3);samples=[]
        for line in self.world_lines:
            try:
                p=project(T@H,line);p=p[np.isfinite(p).all(1)]
            except ValueError:continue
            # Dense image samples, not only distant world endpoints.
            for a,b in zip(p,p[1:]):
                ok,a,b=cv2.clipLine((0,0,frame.shape[1],frame.shape[0]),tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)))
                if not ok:continue
                for x,y in np.linspace(a,b,max(2,min(100,int(np.linalg.norm(np.asarray(a)-b)/4)+1))):
                    ix=int(x);iy=int(y)
                    if mask[iy,ix]:samples.append(float(distances[iy,ix]))
        support=float(np.mean(np.asarray(samples)<=3)) if samples else 0.
        return {'white_support':support,'white_samples':len(samples)},len(samples)>=40 and support>=.55

    def update(self,frame,timestamp_ms):
        import cv2
        started=time.perf_counter();self.frame_index+=1
        def reject(reason):
            self.H=None;self.prev_gray=None;self.points=None
            return Observation('LOST',None,'none',timestamp_ms,{'reason':reason,'cpu_ms':(time.perf_counter()-started)*1000})
        if self.reference is None:return reject('needs_seed')
        if frame.shape[:2]!=self.shape:
            self.reference=None;return reject('resolution_changed_needs_seed')
        if not math.isfinite(timestamp_ms) or timestamp_ms<=self.last_input_time:
            return reject('timestamp_not_monotonic')
        gap=timestamp_ms-self.last_input_time;self.last_input_time=timestamp_ms
        work,gray,mask,T=self._prepare(frame,self.hud_spec)
        if gap>self.max_gap_ms:self.H=None;self.prev_gray=None;self.points=None
        candidate=None;metrics={};source='lk'
        if self.H is not None and self.points is not None and len(self.points)>=16:
            forward,status,_=cv2.calcOpticalFlowPyrLK(self.prev_gray,gray,self.points,None,winSize=(21,21),maxLevel=3)
            if forward is not None and status is not None:
                backward,back_status,_=cv2.calcOpticalFlowPyrLK(gray,self.prev_gray,forward,None,winSize=(21,21),maxLevel=3)
                if backward is not None and back_status is not None:
                    a=self.points.reshape(-1,2);b=forward.reshape(-1,2)
                    good=status.ravel().astype(bool)&back_status.ravel().astype(bool)&(np.linalg.norm(backward.reshape(-1,2)-a,axis=1)<=1.)
                    ij=np.rint(b).astype(int);good&=(ij[:,0]>=0)&(ij[:,0]<mask.shape[1])&(ij[:,1]>=0)&(ij[:,1]<mask.shape[0])
                    good[good]&=mask[ij[good,1],ij[good,0]]>0
                    delta,metrics=estimate_motion(a[good],b[good],gray.shape[::-1])
                    if delta is not None:candidate=np.linalg.inv(T)@delta@self.T@self.H
        must_reanchor=candidate is None or self.frame_index%self.reanchor_every==0
        if must_reanchor and self.matcher:
            query=self.matcher.extract(work,mask);a,b=self.matcher.match(self.reference_features,query)
            delta,learned_metrics=estimate_motion(a,b,gray.shape[::-1])
            if delta is not None:
                anchored=np.linalg.inv(T)@delta@self.T@self.seed_H
                if candidate is not None:
                    # A confident relocalization must agree with the current local motion.
                    grid=np.float32([[x,y] for x in [.2,.5,.8] for y in [.2,.5,.8]])*[frame.shape[1],frame.shape[0]]
                    try:
                        world=project(np.linalg.inv(self.seed_H),grid)
                        disagreement=float(np.median(np.linalg.norm(project(T@candidate,world)-project(T@anchored,world),axis=1)))
                    except (ValueError,np.linalg.LinAlgError):return reject('invalid_reanchor_projection')
                    if disagreement>6:return reject('reanchor_disagrees_with_local_motion')
                candidate=anchored;metrics=learned_metrics;source='xfeat' if isinstance(self.matcher,XFeatMatcher) else 'sift'
            elif candidate is None:return reject('learned_relocalization_failed')
        if candidate is None:return reject('motion_estimation_failed')
        line_metrics,consistent=self._white_support(work,mask,candidate,T)
        if not consistent:return reject('field_line_consistency_failed')
        self.H=candidate/candidate[2,2];self.prev_gray=gray;self.T=T
        self.points=cv2.goodFeaturesToTrack(gray,500,.01,5,mask=mask);self.last_time=timestamp_ms
        return Observation('TRACKING',self.H.copy(),source,timestamp_ms,
                           {**metrics,**line_metrics,'metric_scale_verified':False,'seed_source':self.seed_source,'cpu_ms':(time.perf_counter()-started)*1000})
