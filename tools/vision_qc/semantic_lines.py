"""Offline semantic straight-line calibration; conics remain independent checks.

An automatic seed/proposals assign line identities. Paint evidence can support
or reject those identities, but cannot certify them or create training labels.
"""
from dataclasses import asdict, dataclass
import time

import cv2
import numpy as np
from scipy.optimize import least_squares

from .pitch_tracking import project


@dataclass(frozen=True)
class SemanticLineConfig:
    band_px: float = 40.
    max_proposal_shift_px: float = 100.
    proposal_confidence: float = .5
    paint_contrast: float = 35.
    ransac_trials: int = 180
    inlier_px: float = 1.8
    angle_degrees: float = 6.
    minimum_span_px: float = 80.
    minimum_bins: int = 4
    minimum_pixels: int = 40
    max_fit_evaluations: int = 100
    max_correction_px: float = 128.
    max_joint_p95_px: float = 6.

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError('Finite positive budgets required')
        for name in ('ransac_trials','minimum_bins','minimum_pixels','max_fit_evaluations'):
            if not isinstance(getattr(self,name),int):raise ValueError('Integer budgets required')
        if not 0 < self.proposal_confidence <= 1 or not 1 <= self.minimum_bins <= 12:
            raise ValueError('Invalid confidence or coverage')


def _normal_line(a,b):
    line=np.cross([*a,1.],[*b,1.]);norm=np.linalg.norm(line[:2])
    if not np.isfinite(line).all() or norm < 1e-8:raise ValueError('Degenerate line')
    return line/norm


def _weighted_tls(points,weights):
    centre=np.average(points,axis=0,weights=weights);q=points-centre
    _,vectors=np.linalg.eigh((q*weights[:,None]).T@q/weights.sum())
    normal=vectors[:,0];return np.r_[normal,-normal@centre]


def semantic_field_mask(frame,hud=None):
    """Grass adjacency for paint; yellow boards cannot seed the pitch component."""
    rgb=frame.astype(np.float32)
    green=np.uint8((rgb[:,:,1]>35)&(rgb[:,:,1]>1.12*rgb[:,:,0])&(rgb[:,:,1]>1.12*rgb[:,:,2]))*255
    # Bridge the two grass sides of a painted line before selecting a component;
    # otherwise only one paint edge survives and biases its centre estimate.
    green=cv2.morphologyEx(green,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    count,labels,stats,_=cv2.connectedComponentsWithStats(green)
    if count<2:return np.zeros(frame.shape[:2],np.uint8)
    largest=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]));green=np.uint8(labels==largest)*255
    green=cv2.dilate(green,np.ones((5,5),np.uint8))
    return cv2.bitwise_and(green,hud) if hud is not None else green


def _fit_paint(points,weights,seed_line,direction,length,config,rng):
    if len(points)<config.minimum_pixels:return None
    if len(points)>5000:
        ids=rng.choice(len(points),5000,replace=False);points=points[ids];weights=weights[ids]
    best=None;best_score=-1.
    cosine=np.cos(np.deg2rad(config.angle_degrees))
    for _ in range(config.ransac_trials):
        a,b=points[rng.choice(len(points),2,replace=False)]
        if np.linalg.norm(a-b)<config.minimum_span_px/2:continue
        line=_normal_line(a,b)
        if abs(line[:2]@seed_line[:2])<cosine:continue
        keep=np.abs(points@line[:2]+line[2])<=config.inlier_px
        if keep.sum()<config.minimum_pixels:continue
        along=points[keep]@direction
        span=float(np.ptp(along));bins=len(np.unique(np.clip(((along-along.min())/max(length,1)*12).astype(int),0,11)))
        if span<config.minimum_span_px or bins<config.minimum_bins:continue
        score=float(weights[keep].sum())*bins
        if score>best_score:best,best_score=line,score
    if best is None:return None
    for _ in range(5):
        keep=np.abs(points@best[:2]+best[2])<=config.inlier_px
        if keep.sum()<config.minimum_pixels:return None
        best=_weighted_tls(points[keep],weights[keep])
    error=np.abs(points@best[:2]+best[2]);keep=error<=config.inlier_px
    along=points[keep]@direction
    if (keep.sum()<config.minimum_pixels or np.ptp(along)<config.minimum_span_px
            or abs(best[:2]@seed_line[:2])<cosine):return None
    return best,points[keep],{'inlier_pixels':int(keep.sum()),'span_px':float(np.ptp(along)),
        'paint_residual_p95_px':float(np.percentile(error[keep],95))}


def _solve_lines(observations,size):
    """Point-to-line DLT: no correspondence between arbitrary segment endpoints."""
    w,h=size;N=np.array([[1/w,0,-.5],[0,1/w,-h/(2*w)],[0,0,1.]])
    world=np.concatenate([o['world_xy'] for o in observations]);centre=world.mean(0)
    radius=np.mean(np.linalg.norm(world-centre,axis=1))
    if radius<1e-6:raise ValueError('Degenerate world geometry')
    W=np.array([[1/radius,0,-centre[0]/radius],[0,1/radius,-centre[1]/radius],[0,0,1.]])
    rows=[]
    for o in observations:
        line=np.linalg.inv(N).T@o['line_abc'];line/=np.linalg.norm(line[:2])
        for point in o['world_xy']:
            p=W@np.r_[point,1.];rows.append(np.outer(line,p).ravel())
    A=np.asarray(rows);_,singular,vh=np.linalg.svd(A,full_matrices=True)
    if len(singular)<8 or singular[7]/singular[0]<1e-5:raise ValueError('Line constraints do not determine a homography')
    H=np.linalg.inv(N)@vh[-1].reshape(3,3)@W
    if abs(H[2,2])<1e-10 or np.linalg.matrix_rank(H)!=3:raise ValueError('Invalid fitted homography')
    return H/H[2,2],float(singular[7]/singular[0])


def calibrate_semantic_lines(frame,seed_H,segments,point_proposals=None,hud_spec=None,config=None):
    """Return (selected H, diagnostics) using image evidence, never references.

    segments: [{id, world_xy:[[x0,y0],[x1,y1]]}]. point_proposals optionally
    contains {world_xy,image_xy,confidence} from an automatic semantic detector.
    A proposal moves a search band only; pixels outside HUD must support its fit.
    """
    started=time.perf_counter();config=config or SemanticLineConfig()
    if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim!=3 or frame.shape[2]!=3 or min(frame.shape[:2])<32:
        raise ValueError('Expected an original-resolution uint8 BGR image')
    H=np.asarray(seed_H,float)
    if H.shape!=(3,3) or not np.isfinite(H).all() or np.linalg.matrix_rank(H)!=3 or abs(H[2,2])<1e-10:raise ValueError('Invalid automatic seed')
    H=H/H[2,2];height,width=frame.shape[:2]
    report={'accepted':False,'config':asdict(config),'lines':[], 'models_validated':False,
            'metric_scale_verified':False,'automatic_training_import_allowed':False}

    def finish(reason,selected=H):
        report.update(reason=reason,elapsed_ms=(time.perf_counter()-started)*1000)
        return selected.copy(),report

    hud=np.full((height,width),255,np.uint8)
    for x,y,w,h in (hud_spec or {}).get('rects',[]):cv2.rectangle(hud,(int(x),int(y)),(int(x+w),int(y+h)),0,-1)
    for x,y,r in (hud_spec or {}).get('circles',[]):cv2.circle(hud,(int(x),int(y)),int(r),0,-1)
    field=semantic_field_mask(frame,hud)
    f=frame.astype(np.float32);contrast=f.min(2)-.5*f.max(2)
    yy,xx=np.nonzero((contrast>=config.paint_contrast)&(field>0))
    paint=np.c_[xx,yy].astype(float);strength=contrast[yy,xx]
    if len(paint)<config.minimum_pixels:return finish('insufficient_unmasked_paint')
    observations=[];rng=np.random.default_rng(2026);ids=set()
    for segment in segments:
        sid=segment['id'];world=np.asarray(segment['world_xy'],float)
        if not isinstance(sid,str) or sid in ids or world.shape!=(2,2) or not np.isfinite(world).all():raise ValueError('Invalid or duplicate semantic segment')
        ids.add(sid);world_line=_normal_line(*world)
        dense=np.linspace(*world,101);homogeneous=np.c_[dense,np.ones(len(dense))]@H.T
        visible=(np.abs(homogeneous[:,2])>1e-8)
        xy=np.full((len(dense),2),np.nan);xy[visible]=homogeneous[visible,:2]/homogeneous[visible,2,None]
        visible&=(xy[:,0]>=0)&(xy[:,0]<width)&(xy[:,1]>=0)&(xy[:,1]<height)
        if visible.sum()<2:continue
        indices=np.flatnonzero(visible);wa,wb=dense[indices[[0,-1]]]
        a,b=xy[indices[[0,-1]]];length=float(np.linalg.norm(a-b))
        if length<config.minimum_span_px:continue
        expected=_normal_line(a,b);direction=(b-a)/length
        offsets=[]
        for proposal in point_proposals or []:
            wp=np.asarray(proposal['world_xy'],float);ip=np.asarray(proposal['image_xy'],float);conf=proposal['confidence']
            if wp.shape!=(2,) or ip.shape!=(2,) or not np.isfinite(np.r_[wp,ip,conf]).all() or not 0<=conf<=1:raise ValueError('Invalid automatic point proposal')
            if conf<config.proposal_confidence or abs(world_line@np.r_[wp,1.])>1e-5:continue
            t=float((wp-world[0])@(world[1]-world[0])/np.sum((world[1]-world[0])**2))
            offset=float(expected@np.r_[ip,1.])
            if -.01<=t<=1.01 and abs(offset)<=config.max_proposal_shift_px:offsets.append((offset,conf))
        shifted=expected.copy()
        if offsets:
            offsets.sort();total=sum(p for _,p in offsets);cumulative=0
            for offset,confidence in offsets:
                cumulative+=confidence
                if cumulative>=total/2:shifted[2]-=offset;break
        perpendicular=np.abs(paint@shifted[:2]+shifted[2]);along=(paint-a)@direction
        in_band=(perpendicular<=config.band_px)&(along>=-20)&(along<=length+20)
        fitted=_fit_paint(paint[in_band],strength[in_band],shifted,direction,length,config,rng)
        if fitted is None:
            report['lines'].append({'id':sid,'accepted':False,'reason':'no_distributed_paint_consensus'});continue
        line,pixels,metrics=fitted
        evidence={'id':sid,'accepted':True,'world_xy':[wa.tolist(),wb.tolist()],
            'line_abc':line.tolist(),'proposal_search_shift_px':float(expected[2]-shifted[2]),
            'paint_endpoints_px':pixels[[np.argmin(pixels@direction),np.argmax(pixels@direction)]].tolist(),**metrics}
        report['lines'].append(evidence);observations.append(evidence)
    report['accepted_line_count']=len(observations)
    if len(observations)<4:return finish('too_few_named_lines')
    try:initial,rank=_solve_lines(observations,(width,height))
    except ValueError:return finish('insufficient_geometric_rank')
    norm=np.array([[1/width,0,-.5],[0,1/width,-height/(2*width)],[0,0,1.]])
    invnorm=np.linalg.inv(norm);I=np.eye(3)
    seed_inverse=np.linalg.inv(H)

    def matrix(p):return invnorm@(I+np.r_[p,0.].reshape(3,3))@norm@H

    D=norm@initial@seed_inverse@invnorm
    if abs(D[2,2])<1e-10:return finish('invalid_initial_correction')
    parameters=(D/D[2,2]-I).ravel()[:8]
    if np.max(np.abs(parameters))>.4:return finish('initial_correction_out_of_bounds')

    def residual(matrix_H):
        return np.concatenate([project(matrix_H,np.linspace(*o['world_xy'],8))@np.asarray(o['line_abc'])[:2]+o['line_abc'][2] for o in observations])

    try:
        fit=least_squares(lambda p:residual(matrix(p)),parameters,bounds=(-.4,.4),loss='soft_l1',f_scale=1.,max_nfev=config.max_fit_evaluations)
        candidate=matrix(fit.x);candidate/=candidate[2,2]
        before=np.abs(residual(H));after=np.abs(residual(candidate))
    except (ValueError,np.linalg.LinAlgError):return finish('joint_fit_failed')
    anchors=np.concatenate([project(H,o['world_xy']) for o in observations]);delta=candidate@seed_inverse
    displacement=np.linalg.norm(project(delta,anchors)-anchors,axis=1)
    corners=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    warped=project(delta,corners).astype(np.float32);denominators=np.c_[corners,np.ones(4)]@delta[2]
    area=float(cv2.contourArea(warped,oriented=True)/cv2.contourArea(corners,oriented=True))
    report.update(candidate_H_world_to_image=candidate.tolist(),geometric_rank_ratio=rank,
        fit_evaluations=int(fit.nfev),optimizer_converged=bool(fit.success),
        fitted_line_mean_before_px=float(np.mean(before)),fitted_line_mean_after_px=float(np.mean(after)),
        fitted_line_p95_before_px=float(np.percentile(before,95)),fitted_line_p95_after_px=float(np.percentile(after,95)),
        max_observed_correction_px=float(displacement.max()),image_area_ratio=area)
    if (fit.status<0 or not np.isfinite(candidate).all() or denominators.min()*denominators.max()<=0
            or not cv2.isContourConvex(warped) or not .65<area<1.5 or displacement.max()>config.max_correction_px):
        return finish('unsafe_correction')
    if np.percentile(after,95)>config.max_joint_p95_px or np.mean(after)>=.8*np.mean(before):
        return finish('named_lines_disagree_or_do_not_improve')
    report['accepted']=True
    return finish('named_paint_supported_requires_external_qc',candidate)


def observe_line_junctions(keypoints,segments,line_evidence,image_size,hud_spec=None):
    """Separate image observations; never imply a usable global homography.

    Only intersections of two supported, nonparallel semantic straight segments
    are returned. Curves/spots have no observation. Points outside support spans
    or behind the configured HUD remain unknown. This is not a ground-truth API.
    """
    width,height=image_size
    if not isinstance(width,int) or not isinstance(height,int) or min(width,height)<32:
        raise ValueError('Invalid image dimensions')
    by_id={s['id']:s for s in segments};evidence=[]
    for e in line_evidence:
        if not e.get('accepted') or e.get('id') not in by_id:continue
        if 'line_abc' not in e or 'paint_endpoints_px' not in e:continue
        line=np.asarray(e['line_abc'],float);ends=np.asarray(e['paint_endpoints_px'],float)
        if line.shape!=(3,) or ends.shape!=(2,2) or not np.isfinite(np.r_[line,ends.ravel()]).all():
            raise ValueError('Invalid line evidence')
        norm=np.linalg.norm(line[:2])
        if norm<1e-8:raise ValueError('Degenerate image line')
        evidence.append((e,by_id[e['id']],line/norm,ends))
    outputs=[]
    for point in keypoints:
        world=np.asarray(point['world_xy'],float)
        if world.shape!=(2,) or not np.isfinite(world).all():raise ValueError('Invalid semantic world point')
        eligible=[]
        for e,segment,line,ends in evidence:
            a,b=np.asarray(segment['world_xy'],float);world_line=_normal_line(a,b)
            if abs(world_line@np.r_[world,1.])>1e-5:continue
            t=float((world-a)@(b-a)/np.sum((b-a)**2))
            if -.0001<=t<=1.0001:eligible.append((e,line,ends))
        choices=[]
        for i,(e1,l1,ends1) in enumerate(eligible):
            for e2,l2,ends2 in eligible[i+1:]:
                sine=float(abs(np.cross(l1,l2)[2]))
                if sine<np.sin(np.deg2rad(15)):continue
                p=np.cross(l1,l2);xy=p[:2]/p[2]
                if not np.isfinite(xy).all() or not (0<=xy[0]<width and 0<=xy[1]<height):continue
                supported=True
                for ends in [ends1,ends2]:
                    direction=ends[1]-ends[0];length=np.linalg.norm(direction)
                    along=float((xy-ends[0])@direction/max(length,1e-8))
                    if length<1e-8 or not -20<=along<=length+20:supported=False
                if not supported:continue
                if any(x<=xy[0]<=x+w and y<=xy[1]<=y+h for x,y,w,h in (hud_spec or {}).get('rects',[])):continue
                if any(np.linalg.norm(xy-[x,y])<=r for x,y,r in (hud_spec or {}).get('circles',[])):continue
                uncertainty=max(1.,float(e1['paint_residual_p95_px']),float(e2['paint_residual_p95_px']))/sine
                choices.append((uncertainty,xy,[e1['id'],e2['id']]))
        if choices:
            uncertainty,xy,sources=min(choices,key=lambda c:c[0])
            outputs.append({'id':point['id'],'image_xy':xy.tolist(),'source_line_ids':sources,
                'estimated_uncertainty_px':float(uncertainty),'global_calibration_usable':False,
                'automatic_training_import_allowed':False})
    return outputs
