"""Causal timestamp windows and match-level split checks for video research.

Box labels on independent stills never establish action, identity or 3D labels.
"""
from bisect import bisect_right
import math


def causal_window(timestamps_ms,end_index,count=8,step_ms=50.,max_gap_ms=150.,cut_indices=()):
    """Use only frames at/before requested times; never cross a known scene cut.

    Returns None on warmup, duplicate sampling or gaps. There is no silent frame
    repetition, future padding or invented observation across missing video.
    """
    times=list(timestamps_ms)
    if (not times or any(not math.isfinite(t) for t in times)
            or any(b<=a for a,b in zip(times,times[1:]))):raise ValueError('Strictly increasing finite PTS required')
    if (not isinstance(end_index,int) or not 0<=end_index<len(times) or not isinstance(count,int)
            or count<2 or not math.isfinite(step_ms) or step_ms<=0
            or not math.isfinite(max_gap_ms) or max_gap_ms<=0):raise ValueError('Invalid causal window budget')
    cuts=list(cut_indices)
    if any(not isinstance(c,int) or not 0<=c<len(times) for c in cuts):raise ValueError('Invalid scene cut')
    start=max([c for c in cuts if c<=end_index]+[0]);last=times[end_index]
    indices=[bisect_right(times,last-(count-1-i)*step_ms,lo=start,hi=end_index+1)-1 for i in range(count)]
    if min(indices)<start or len(set(indices))!=count:return None
    # Include skipped intermediate frames so a missing interval cannot hide in resampling.
    if any(b-a>max_gap_ms for a,b in zip(times[indices[0]:end_index],times[indices[0]+1:end_index+1])):return None
    if any(last-(count-1-i)*step_ms-times[j]>max_gap_ms for i,j in enumerate(indices)):return None
    return indices


def validate_match_splits(videos):
    """Require verified match IDs; filenames do not establish independent splits."""
    matches={};hashes={};seen=set();errors=[]
    for video in videos:
        identity=video.get('video_id');split=video.get('split');match=video.get('match_id');digest=video.get('sha256')
        if not isinstance(identity,str) or not identity or identity in seen:errors.append('invalid_or_duplicate_video_id')
        seen.add(identity)
        if split not in {'train','val','test'}:errors.append('unassigned_split')
        if not isinstance(match,str) or not match or video.get('match_id_verified') is not True:
            errors.append('match_identity_unverified')
        elif match in matches and matches[match]!=split:errors.append('match_leakage')
        else:matches[match]=split
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            errors.append('invalid_source_hash')
        elif digest in hashes and hashes[digest]!=split:errors.append('duplicate_video_across_splits')
        else:hashes[digest]=split
    present={v.get('split') for v in videos}
    if not {'train','val','test'}<=present:errors.append('missing_evaluation_split')
    return {'independent_evaluation_ready':not errors,'errors':sorted(set(errors)),
            'automatic_training_import_allowed':False}
