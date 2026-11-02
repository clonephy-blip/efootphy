"""Bounded same-frame coarse-to-native ball confirmation; no temporal lookahead."""
import math
import numpy as np
from .ball_video import NativeBallDetector


class CoarseToFineBallDetector(NativeBallDetector):
    def __init__(self,model,backend='opencv',acquire=.25,proposal=.03,max_refinements=2,max_shift_px=24.):
        if not all(math.isfinite(v) for v in [acquire,proposal,max_shift_px]) or not 0<proposal<acquire<=1 or not isinstance(max_refinements,int) or not 1<=max_refinements<=4 or not 0<max_shift_px<=64:raise ValueError('Invalid bounded confirmation policy')
        super().__init__(model,backend=backend)
        self.acquire=acquire;self.proposal=proposal;self.max_refinements=max_refinements;self.max_shift_px=max_shift_px

    def scan(self,frame,predicted=None,excluded_rects=()):
        if predicted is not None:return super().scan(frame,predicted,excluded_rects)
        h,w=frame.shape[:2];side=min(640,w,h)
        xs=np.linspace(0,w-side,max(1,int(np.ceil(w/side))));ys=np.linspace(0,h-side,max(1,int(np.ceil(h/side))))
        rects=[(int(x),int(y),int(x)+side,int(y)+side) for y in ys for x in xs]
        def allowed(d):return not any(x0<=d['xy_px'][0]<x1 and y0<=d['xy_px'][1]<y1 for x0,y0,x1,y1 in excluded_rects)
        proposals=[]
        for rect in rects:proposals.extend(d for d in self.detect_crop(frame,rect,self.proposal) if allowed(d))
        output=[d for d in proposals if d['score']>=.15]
        if any(d['score']>=self.acquire for d in proposals):return output,len(rects)
        chosen=[]
        for d in sorted(proposals,key=lambda d:-d['score']):
            if all(np.linalg.norm(np.array(d['xy_px'])-p['xy_px'])>48 for p in chosen):chosen.append(d)
            if len(chosen)>=self.max_refinements:break
        count=len(rects)
        for coarse in chosen:
            native=min(320,w,h);x,y=np.rint(np.array(coarse['xy_px'])-native/2).astype(int)
            x=int(np.clip(x,0,w-native));y=int(np.clip(y,0,h-native));count+=1
            for d in self.detect_crop(frame,(x,y,x+native,y+native),self.acquire):
                if not allowed(d) or d['score']<self.acquire or np.linalg.norm(np.array(d['xy_px'])-coarse['xy_px'])>self.max_shift_px:continue
                confirmed=dict(d);confirmed.update(reacquisition_evidence='same_frame_coarse_to_native',coarse_score=coarse['score'])
                output.append(confirmed)
        return output,count
