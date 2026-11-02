"""Source-bound video review gates; model proposals are not training labels."""
from bisect import bisect_left
import hashlib
import math


def frame_span(pts, start_ms, end_ms):
    if not pts or any(not math.isfinite(t) for t in pts) or any(b<=a for a,b in zip(pts,pts[1:])):
        raise ValueError('Strict source PTS required')
    if not 0<=start_ms<end_ms:raise ValueError('Invalid clip interval')
    a,b=bisect_left(pts,start_ms),bisect_left(pts,end_ms)
    if a==b or a>=len(pts):raise ValueError('No source frames')
    return a,b


def validate_clip_plan(plan):
    sources={};hashes={};clips={};intervals={}
    for s in plan['sources']:
        identity,digest,split=s['video_id'],s['sha256'],s['split']
        if identity in sources or split not in ('train','val','regression'):raise ValueError('Duplicate source or invalid split')
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('Source SHA256 required')
        if digest in hashes and hashes[digest]!=split:raise ValueError('Source hash crosses splits')
        sources[identity]=s;hashes[digest]=split
    for c in plan['clips']:
        name,source,a,b=c['clip_id'],c['video_id'],c['start_ms'],c['end_ms']
        if name in clips or source not in sources or not 0<=a<b or b-a>10000:raise ValueError('Invalid clip or budget')
        if c.get('split',sources[source]['split'])!=sources[source]['split']:raise ValueError('A video cannot cross splits')
        if any(max(a,x)<min(b,y) for x,y in intervals.get(source,[])):raise ValueError('Overlapping clips')
        intervals.setdefault(source,[]).append((a,b));clips[name]=c
    return {'sources':sources,'clips':clips,'independent_match_evaluation':False,'automatic_annotation_import_allowed':False}


def validate_ball_review(row, data, plan):
    ref=validate_clip_plan(plan)
    if row.get('reviewed') is not True or row.get('reviewed_by') not in ('assistant','user'):raise ValueError('Unreviewed label')
    if hashlib.sha256(data).hexdigest()!=row.get('image_sha256'):raise ValueError('Image hash changed')
    clip=ref['clips'].get(row.get('clip_id'))
    if clip is None or row['video_id']!=clip['video_id']:raise ValueError('Annotation source mismatch')
    if row.get('source_video_sha256')!=ref['sources'][row['video_id']]['sha256']:raise ValueError('Video hash changed')
    if not clip['start_ms']<=row['pts_ms']<clip['end_ms']:raise ValueError('Label outside clip')
    visibility=row.get('visibility');box=row.get('ball_box_xyxy_px')
    if visibility not in ('visible','absent','occluded','uncertain'):raise ValueError('Explicit visibility required')
    if visibility=='visible':
        w,h=row['image_size_wh']
        if row.get('geometry_reviewed') is not True:raise ValueError('Final geometry review required')
        if not isinstance(box,list) or len(box)!=4 or not all(math.isfinite(x) for x in box) or not 0<=box[0]<box[2]<=w or not 0<=box[1]<box[3]<=h:raise ValueError('Invalid visible box')
    elif box is not None:raise ValueError('No invented hidden-ball coordinate')
    indices=row['causal_context_frame_indices'];pts=row.get('causal_context_pts_ms',[])
    if len(indices)!=3 or len(set(indices))!=3 or indices!=sorted(indices) or indices[-1]!=row['source_frame_index']:raise ValueError('Distinct causal frames required')
    if len(pts)!=3 or not all(math.isfinite(t) for t in pts) or any(b<=a for a,b in zip(pts,pts[1:])) or pts[-1]!=row['pts_ms'] or pts[0]<clip['start_ms']:raise ValueError('Causal context crosses boundary')
    return visibility in ('visible','absent')
