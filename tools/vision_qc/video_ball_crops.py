"""Export reviewed appearance crops while preserving causal source references."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
import yaml

from .ball_clip_data import validate_ball_review,validate_clip_plan
from .ball_crops import crop_labels,grid_origins

CLASSES=['ball','cursor','cursor_secondary','goalkeeper','player','powerbar','referee']


def build_video_crops(plan_path,reviews_path,out,replay_data=None,replay_train_limit=240,replay_val_limit=192):
    plan=json.loads(Path(plan_path).read_text());refs=validate_clip_plan(plan);rows=json.loads(Path(reviews_path).read_text());out=Path(out)
    if out.exists():raise ValueError('New output required')
    accepted=[];seen=set();frames=set();ignored=[]
    for r in rows:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',r['id']):raise ValueError('Unsafe review identifier')
        key=(r['source_video_sha256'],r['source_frame_index'])
        if r['id'] in seen or key in frames:raise ValueError('Duplicate source frame')
        seen.add(r['id']);frames.add(key)
        if r.get('reviewed') is not True:ignored.append(r['id']);continue
        if validate_ball_review(r,Path(r['image_path']).read_bytes(),plan):accepted.append(r)
        else:ignored.append(r['id'])
    if not {'train','val'}<={refs['sources'][r['video_id']]['split'] for r in accepted}:raise ValueError('Reviewed train and val required')
    rng=np.random.default_rng(20261102);records=[];counts={'train':0,'valid':0};source_counts=dict(counts);out.mkdir(parents=True)
    for row in accepted:
        split='valid' if refs['sources'][row['video_id']]['split']=='val' else refs['sources'][row['video_id']]['split']
        if split not in counts:continue
        frame=cv2.imread(row['image_path']);h,w=frame.shape[:2]
        if [w,h]!=row['image_size_wh']:raise ValueError('Image dimensions changed')
        b=row['ball_box_xyxy_px'];boxes=[] if b is None else [[0,(b[0]+b[2])/2,(b[1]+b[3])/2,b[2]-b[0],b[3]-b[1]]];crops=[]
        if split=='train':
            if b is not None:
                for side in [320,640]:
                    c=np.array(boxes[0][1:3])+rng.uniform(-side*.3,side*.3,2)
                    crops.append((int(np.clip(round(c[0]-side/2),0,w-side)),int(np.clip(round(c[1]-side/2),0,h-side)),side,'reviewed positive with position jitter'))
            for _ in range(2):
                crops.append((int(rng.integers(0,w-320+1)),int(rng.integers(0,h-320+1)),320,'uniform random field/HUD crop; retain any ball label'))
            if b is not None:
                for d in row.get('candidates',[]):
                    cx,cy=d['xy_px']
                    if np.linalg.norm(np.array([cx,cy])-boxes[0][1:3])>80:
                        crops.append((int(np.clip(cx-160,0,w-320)),int(np.clip(cy-160,0,h-320)),320,'non-ball model candidate; reviewed one-ball frame'));break
        else:
            for side in [320,640]:crops.extend((x,y,side,'fixed full-frame validation grid') for y in grid_origins(h,side) for x in grid_origins(w,side))
        images=out/split/'images';labels=out/split/'labels';images.mkdir(parents=True,exist_ok=True);labels.mkdir(exist_ok=True)
        for j,(x,y,side,policy) in enumerate(crops):
            stem=f"video-{row['id']}-{j:02d}";patch=cv2.resize(frame[y:y+side,x:x+side],(320,320))
            if not cv2.imwrite(str(images/(stem+'.jpg')),patch,[cv2.IMWRITE_JPEG_QUALITY,97]):raise ValueError('Write failed')
            (labels/(stem+'.txt')).write_text('\n'.join(crop_labels(boxes,x,y,side))+'\n')
            records.append({'crop':stem,'split':split,'source_review_id':row['id'],'source_video_sha256':row['source_video_sha256'],'source_image_sha256':row['image_sha256'],'source_frame_index':row['source_frame_index'],'pts_ms':row['pts_ms'],'source_side':side,'origin_xy':[x,y],'policy':policy,'causal_context_frame_indices':row['causal_context_frame_indices']});counts[split]+=1
        source_counts[split]+=1
    if replay_data is not None:
        replay_data=Path(replay_data)
        for split,limit in [('train',replay_train_limit),('valid',replay_val_limit)]:
            for image in sorted((replay_data/split/'images').glob('*.jpg'),key=lambda p:hashlib.sha256(p.name.encode()).hexdigest())[:limit]:
                stem='replay-'+image.stem;shutil.copy2(image,out/split/'images'/(stem+'.jpg'));shutil.copy2(replay_data/split/'labels'/(image.stem+'.txt'),out/split/'labels'/(stem+'.txt'));records.append({'crop':stem,'split':split,'policy':'existing-source replay in original split','source_image_sha256':hashlib.sha256(image.read_bytes()).hexdigest()});counts[split]+=1
    (out/'data.yaml').write_text(yaml.safe_dump({'train':'train/images','val':'valid/images','test':None,'nc':7,'names':CLASSES},sort_keys=False))
    report={'input_side':320,'source_sides':[320,640],'tile_counts':counts,'reviewed_video_frames':source_counts,'validation_crop_selection_uses_labels':False,'independent_match_split_verified':False,'ignored_or_unreviewed_frames':ignored,'crops':records,'plan_sha256':hashlib.sha256(Path(plan_path).read_bytes()).hexdigest(),'review_sha256':hashlib.sha256(Path(reviews_path).read_bytes()).hexdigest(),'temporal_model_trained':False,'automatic_training_import_allowed':False}
    (out/'crop-provenance.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--reviews',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--replay-data',type=Path);a=p.parse_args();r=build_video_crops(a.plan,a.reviews,a.out,a.replay_data);print(json.dumps({k:r[k] for k in ['tile_counts','reviewed_video_frames']}))
