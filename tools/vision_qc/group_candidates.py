"""Review filename-derived source groups. Hypotheses never certify independence."""
import argparse
import collections
import json
import re
from pathlib import Path

import yaml

from .dataset_qc import EXTENSIONS,split_directory


def infer_group(stem):
    stem=stem.split('.rf.')[0]
    for pattern in [r'(m\d+)_\d+p_f\d+(?:_(?:jpg|png))?',r'(yt_.+)_\d+s?(?:_(?:jpg|png))?']:
        match=re.fullmatch(pattern,stem)
        if match:return match[1]
    return None


def inspect(data):
    data=Path(data).resolve();config=yaml.safe_load(data.read_text())
    counts=collections.defaultdict(collections.Counter);unparsed=collections.Counter();frames=collections.defaultdict(set)
    records=[]
    for split,key in [('train','train'),('valid','val'),('test','test')]:
        if not config.get(key):continue
        try:folder=split_directory(data.parent,config[key])
        except ValueError:
            if split=='test':continue
            raise
        for image in sorted(p for p in folder.rglob('*') if p.suffix.lower() in EXTENSIONS):
            group=infer_group(image.stem)
            frames[image.stem.split('.rf.')[0]].add(split)
            if group:counts[group][split]+=1
            else:unparsed[split]+=1
            records.append({'image':image.relative_to(data.parent).as_posix(),'split':split,'inferred_group':group})
    cross=[g for g,c in counts.items() if len(c)>1]
    return {'schema_version':1,'scope':'Filename-prefix hypotheses; original recordings/group identities not verified',
        'recognized_group_count':len(counts),'recognized_image_count':sum(sum(c.values()) for c in counts.values()),
        'recognized_groups':{g:dict(c) for g,c in sorted(counts.items())},
        'groups_crossing_splits':sorted(cross),'source_frame_stems_crossing_splits':sum(len(s)>1 for s in frames.values()),
        'unparsed_images':dict(unparsed),'records':records,
        'verified':False,'eligible_for_independent_evaluation':False,
        'next_action':'Review identities and unparsed sources, then create a new group-disjoint split. Do not pass these hypotheses as certified groups.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();report=inspect(a.data);a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['recognized_group_count','recognized_image_count','groups_crossing_splits','unparsed_images','verified']},indent=2))
