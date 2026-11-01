"""Conservative train-to-train geometry proposals; never infer labels from the holdout."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .heatmap_pose import transform_points


def geometry_gate(source, target, inliers, matrix):
    """Reject degenerate/local/poorly explained projective matches."""
    mask=np.asarray(inliers).reshape(-1).astype(bool);s=source[mask];t=target[mask]
    if len(s)<14 or mask.mean()<.65:return False
    def hull_area(points):
        values=sorted(set(map(tuple,points.tolist())))
        if len(values)<3:return 0.
        def cross(o,a,b):return (a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0])
        lower=[];upper=[]
        for point in values:
            while len(lower)>=2 and cross(lower[-2],lower[-1],point)<=0:lower.pop()
            lower.append(point)
        for point in reversed(values):
            while len(upper)>=2 and cross(upper[-2],upper[-1],point)<=0:upper.pop()
            upper.append(point)
        hull=lower[:-1]+upper[:-1]
        return abs(sum(a[0]*b[1]-a[1]*b[0] for a,b in zip(hull,hull[1:]+hull[:1])))/2
    if hull_area(s)<640*640*.04 or hull_area(t)<640*640*.04:return False
    if not np.isfinite(matrix).all() or np.linalg.cond(matrix)>1e6:return False
    homogeneous=np.c_[s,np.ones(len(s))]@matrix.T
    if np.any(np.abs(homogeneous[:,2])<1e-8):return False
    projected=homogeneous[:,:2]/homogeneous[:,2,None]
    if np.median(np.linalg.norm(projected-t,axis=1))>1.5:return False
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pose-root',type=Path,required=True);p.add_argument('--field-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import cv2
    from PIL import Image,ImageDraw
    cv2.setNumThreads(1);cv2.setRNGSeed(20261101)
    sift=cv2.SIFT_create(nfeatures=600,contrastThreshold=.025)
    def features(path):
        frame=cv2.resize(cv2.imread(str(path)),(640,640));hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        grass=cv2.inRange(hsv,np.array([25,30,25]),np.array([100,255,255]))
        mask=cv2.dilate(grass,np.ones((13,13),np.uint8));mask[:72]=0;mask[550:]=0
        key,desc=sift.detectAndCompute(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),mask)
        return frame,np.float32([x.pt for x in key]),desc
    sources=[]
    for path in sorted((a.pose_root/'train/images').glob('*.jpg')):
        frame,xy,desc=features(path)
        row=np.asarray(list(map(float,(path.parent.parent/'labels'/path.with_suffix('.txt').name).read_text().split())))
        sources.append({'path':path,'frame':frame,'xy':xy,'desc':desc,'points':row[5:].reshape(32,3)*[640,640,1]})
    validation=[]
    for path in sorted((a.pose_root/'valid/images').glob('*.jpg')):
        frame,xy,desc=features(path);validation.append({'xy':xy,'desc':desc})
    train_desc=np.concatenate([x['desc'] for x in sources if x['desc'] is not None])
    owner=np.concatenate([np.full(len(x['desc']),i) for i,x in enumerate(sources) if x['desc'] is not None])
    index=cv2.FlannBasedMatcher(dict(algorithm=1,trees=4),dict(checks=32));index.add([train_desc]);index.train()
    matcher=cv2.BFMatcher()
    def homography(xy,desc,source):
        if desc is None or len(desc)<14 or source['desc'] is None:return None
        pairs=matcher.knnMatch(source['desc'],desc,k=2)
        good=[first for first,second in pairs if first.distance<.65*second.distance]
        if len(good)<14:return None
        s=np.float32([source['xy'][m.queryIdx] for m in good]);t=np.float32([xy[m.trainIdx] for m in good])
        matrix,inliers=cv2.findHomography(s,t,cv2.RANSAC,2.5)
        if matrix is None or not geometry_gate(s,t,inliers,matrix):return None
        return matrix,int(inliers.sum()),len(good)
    # Hash excludes identical image bytes; geometry-only comparisons also exclude near-duplicate holdout views.
    holdout_hashes={hashlib.sha256(p.read_bytes()).hexdigest() for p in (a.pose_root/'valid/images').glob('*.jpg')}
    paths=sorted((a.field_root/'train/images').glob('*.jpg'))
    if a.limit:paths=paths[:a.limit]
    accepted=[];rejected=[];started=time.monotonic();preview=0
    for i,path in enumerate(paths):
        frame,xy,desc=features(path);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        reason='no_reliable_source_match';match=None;source=None
        if digest in holdout_hashes or any(homography(xy,desc,v) is not None for v in validation):
            reason='holdout_duplicate_or_matching_view'
        elif desc is not None:
            matches=index.knnMatch(desc,k=2)
            votes=np.zeros(len(sources),np.int32)
            for first,second in matches:
                if first.distance<.8*second.distance:votes[owner[first.trainIdx]]+=1
            for j in np.argsort(votes)[-4:][::-1]:
                if votes[j]<4:continue
                candidate=homography(xy,desc,sources[j])
                if candidate and (match is None or candidate[1]>match[1]):match=candidate;source=sources[j]
            if match:
                points=transform_points(source['points'],match[0],640)
                visible=points[:,2]==2
                # Independent white-line evidence near each transferred label; no model score substitutes for this.
                hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
                white=cv2.inRange(hsv,np.array([0,0,135]),np.array([179,85,255]))
                distance=cv2.distanceTransform(255-white,cv2.DIST_L2,3)
                evidence=[]
                for k,(x,y,v) in enumerate(points):
                    if v==2:
                        d=float(distance[int(min(639,y)),int(min(639,x))]);evidence.append(d)
                        if d>4:points[k]=0
                remaining=int((points[:,2]==2).sum())
                if remaining>=6 and remaining>=.8*int(visible.sum()):
                    record={'path':str(path.resolve()),'image':str(path.relative_to(a.field_root)),
                            'target_split':'train','image_sha256':digest,
                            'source_image':str(source['path'].relative_to(a.pose_root)),
                            'source_image_sha256':hashlib.sha256(source['path'].read_bytes()).hexdigest(),
                            'matrix_640':match[0].tolist(),'inliers':match[1],'matches':match[2],
                            'points':points.tolist(),'white_line_distance_px':evidence,
                            'status':'automatic_geometric_proposal_not_manual_ground_truth'}
                    accepted.append(record);reason=None
                    if preview<24:
                        image=Image.fromarray(frame[:,:,::-1]);draw=ImageDraw.Draw(image)
                        for k,(x,y,v) in enumerate(points):
                            if v==2:
                                draw.ellipse((x-4,y-4,x+4,y+4),outline='yellow',width=2);draw.text((x+4,y),str(k),fill='yellow')
                        image.save(a.out/f'proposal-{preview:03d}.jpg');preview+=1
                else:reason='insufficient_white_line_support'
        if reason:rejected.append({'image':str(path.relative_to(a.field_root)),'reason':reason})
        if (i+1)%100==0:print('TRANSFER',i+1,len(paths),'accepted',len(accepted),round(time.monotonic()-started,1),flush=True)
    from collections import Counter
    report={'scope':'train-to-train geometric proposals only; source identities and Mobile generalization unverified',
            'source_pose_train_images':len(sources),'field_train_scanned':len(paths),'accepted':accepted,'rejected':rejected,
            'rejection_counts':dict(Counter(x['reason'] for x in rejected)),'elapsed_seconds':time.monotonic()-started,
            'validation_labels_used':False,'manual_ground_truth':False,'release_eligible':False}
    (a.out/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['source_pose_train_images','field_train_scanned','rejection_counts','elapsed_seconds']}))
    print('ACCEPTED',len(accepted))


if __name__=='__main__':main()
