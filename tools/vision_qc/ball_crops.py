"""Native-resolution crop preparation; validation tiles do not use labels."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from .normalize import normalize_row


def grid_origins(length, side):
    if length < side:
        raise ValueError('Image smaller than native crop')
    return sorted(set(list(range(0, length-side+1, side))+[length-side]))


def crop_labels(boxes, x, y, side):
    rows=[]
    for cid, cx, cy, bw, bh in boxes:
        left, top = max(cx-bw/2,x), max(cy-bh/2,y)
        right, bottom = min(cx+bw/2,x+side), min(cy+bh/2,y+side)
        cw, ch = right-left, bottom-top
        if min(cw,ch) >= 2 and cw*ch >= .25*bw*bh:
            rows.append(f'{int(cid)} {(left+right-2*x)/(2*side):.9g} {(top+bottom-2*y)/(2*side):.9g} {cw/side:.9g} {ch/side:.9g}')
    return rows


def prepare_native_crops(source, out, side=320):
    source,out=Path(source),Path(out)
    if out.exists() or side not in (320,640):
        raise ValueError('New output and supported native side required')
    config=yaml.safe_load((source/'data.yaml').read_text())
    if config.get('kpt_shape') or config.get('names') != ['ball','cursor','cursor_secondary','goalkeeper','player','powerbar','referee']:
        raise ValueError('Wrong source schema')
    rng=np.random.default_rng(20261102);records=[];groups={};hashes={};counts={}
    for split in ['train','valid']:
        images=sorted((source/split/'images').glob('*'))
        if not images or len(images)>256:raise ValueError('Source image budget exceeded')
        count=0;groups[split]=set();hashes[split]=set()
        for image in images:
            frame=cv2.imread(str(image))
            if frame is None:raise ValueError('Bad image')
            h,w=frame.shape[:2];boxes=[]
            for line in (source/split/'labels'/(image.stem+'.txt')).read_text().splitlines():
                if not line.strip():continue
                row,_,_=normalize_row(line,7);a=list(map(float,row.split()));boxes.append(np.array(a)*[1,w,h,w,h])
            if split=='train':
                origins=[]
                for b in boxes:
                    if b[0]!=0:continue
                    for _ in range(2):
                        cx,cy=b[1:3]+rng.uniform(-side*.34,side*.34,2)
                        origins.append((int(np.clip(cx-side/2,0,w-side)),int(np.clip(cy-side/2,0,h-side))))
                for _ in range(2):
                    origins.append((int(rng.integers(0,w-side+1)),int(rng.integers(0,h-side+1))))
            else:
                origins=[(x,y) for y in grid_origins(h,side) for x in grid_origins(w,side)]
            group=image.name.split('.rf.')[0];digest=hashlib.sha256(image.read_bytes()).hexdigest()
            groups[split].add(group);hashes[split].add(digest)
            for j,(x,y) in enumerate(origins):
                stem=f'{image.stem}-crop{j:02d}';folder=out/split
                (folder/'images').mkdir(parents=True,exist_ok=True);(folder/'labels').mkdir(exist_ok=True)
                if not cv2.imwrite(str(folder/'images'/(stem+'.jpg')),frame[y:y+side,x:x+side],[cv2.IMWRITE_JPEG_QUALITY,96]):raise ValueError('Write failed')
                rows=crop_labels(boxes,x,y,side);(folder/'labels'/(stem+'.txt')).write_text('\n'.join(rows)+'\n')
                records.append({'split':split,'source':image.name,'source_sha256':digest,'crop':stem,'origin_xy':[x,y],
                                'origin_policy':'label-guided positives + random negatives' if split=='train' else 'full fixed grid, label independent'})
                count+=1
        counts[split]=count
    if groups['train']&groups['valid'] or hashes['train']&hashes['valid']:
        raise ValueError('Source group/hash leakage')
    copied=dict(config);copied.pop('path',None);copied.update(train=str((out/'train/images').resolve()),val=str((out/'valid/images').resolve()),test=None)
    (out/'data.yaml').write_text(yaml.safe_dump(copied,sort_keys=False))
    report={'native_side':side,'tile_counts':counts,'crops':records,'independent_match_split_verified':False,
            'validation_crop_selection_uses_labels':False,'input_selection':'Existing labelled source export only'}
    (out/'crop-provenance.json').write_text(json.dumps(report,indent=2));return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();v=prepare_native_crops(a.source,a.out);print(json.dumps(v['tile_counts']))
