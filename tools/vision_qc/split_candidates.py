"""Plan disjoint filename-group candidates from a frozen audit. No certification."""
import argparse
import collections
import hashlib
import json
from pathlib import Path,PurePosixPath

from .group_candidates import infer_group


def plan_split(audit, candidates, forbidden_weights=(), seed=20261101):
    if not audit.get('data_valid') or audit.get('task')!='detect':
        raise ValueError('A valid frozen detection audit is required')
    images={x['image']:x for x in audit['images']};rows=candidates['records']
    by_name={x['image']:x for x in rows}
    if len(images)!=len(audit['images']) or len(by_name)!=len(rows) or set(images)!=set(by_name):
        raise ValueError('Candidate records do not exactly match the audit')
    groups={};pixels=collections.defaultdict(list)
    for name,item in images.items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Unsafe image path')
        group=infer_group(path.stem)
        if by_name[name]['inferred_group']!=group:raise ValueError('Group hypotheses differ from filename inference')
        if not item.get('pixel_sha256') or not item.get('sha256') or not item.get('label_sha256'):
            raise ValueError('Frozen image/pixel/label hashes are required')
        groups[name]=group;pixels[item['pixel_sha256']].append(name)
    parents={g:g for g in groups.values() if g is not None}
    def find(g):
        while parents[g]!=g:
            parents[g]=parents[parents[g]];g=parents[g]
        return g
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    for names in pixels.values():
        known=sorted({groups[n] for n in names if groups[n] is not None})
        for group in known[1:]:union(known[0],group)
    tainted=set()
    for names in pixels.values():
        if any(groups[n] is None for n in names):
            tainted.update(find(groups[n]) for n in names if groups[n] is not None)
    clusters=collections.defaultdict(list);quarantine=[]
    for name in sorted(images):
        g=groups[name]
        if g is None or find(g) in tainted:quarantine.append(name)
        else:clusters[find(g)].append(name)
    if len(clusters)<3:raise ValueError('Need at least three untainted source clusters')
    cluster_records=[]
    for names in clusters.values():
        source_groups=sorted({groups[n] for n in names})
        cid='cluster-'+hashlib.sha256('|'.join(source_groups).encode()).hexdigest()[:12]
        cluster_records.append({'id':cid,'groups':source_groups,'images':names})
    cluster_records.sort(key=lambda x:(-len(x['images']),hashlib.sha256(f"{seed}|{x['id']}".encode()).hexdigest()))
    n=len(clusters);train=max(1,int(n*.7));valid=max(1,int(n*.15));test=n-train-valid
    if test<1:train-=1-test;test=1
    capacity={'train':train,'valid':valid,'test':test};fractions={'train':.7,'valid':.15,'test':.15}
    total=sum(len(x['images']) for x in cluster_records)
    counts=collections.Counter();group_counts=collections.Counter();assignments={}
    for item in cluster_records:
        size=len(item['images'])
        def cost(split):
            target=total*fractions[split]
            return ((counts[split]+size-target)**2-(counts[split]-target)**2)/target
        split=min((s for s in capacity if group_counts[s]<capacity[s]),key=lambda s:(cost(s),s))
        item['candidate_split']=split;counts[split]+=size;group_counts[split]+=1
        for name in item['images']:assignments[name]=split
    for name in quarantine:assignments[name]='quarantine'
    for names in pixels.values():
        used={assignments[n] for n in names if assignments[n]!='quarantine'}
        if len(used)>1:raise RuntimeError('Exact-pixel leakage in split plan')
    records=[{'image':name,'candidate_split':assignments[name],'source_split':images[name]['split'],
        'inferred_group':groups[name],'image_sha256':images[name]['sha256'],
        'label_sha256':images[name]['label_sha256'],'pixel_sha256':images[name]['pixel_sha256']} for name in sorted(images)]
    fingerprint=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()
    return {'schema_version':1,'kind':'candidate_plan_only','seed':seed,
        'source_dataset_fingerprint':audit['dataset_fingerprint'],'plan_fingerprint':fingerprint,
        'splits':{s:{'images':counts[s],'source_clusters':group_counts[s],
            'inferred_groups':sum(len(x['groups']) for x in cluster_records if x['candidate_split']==s)} for s in capacity},
        'quarantine_images':len(quarantine),'clusters':cluster_records,'records':records,
        'filename_groups_disjoint':True,'exact_pixels_disjoint':True,
        'source_identities_verified':False,'class_balance_verified':False,
        'eligible_for_independent_evaluation':False,'ready_for_training':False,
        'checkpoint_policy':{'requires_fresh_official_pretrained':True,
            'forbidden_checkpoint_sha256':sorted(set(forbidden_weights)),
            'reason':'The phase-2 model has already trained on images that may move into these candidate holdouts'},
        'next_checks':['Review inferred source identities and quarantined sources','Check class support after regrouping',
                       'Reverify bytes and near-duplicate links before materializing a training split']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',required=True,type=Path);p.add_argument('--groups',required=True,type=Path)
    p.add_argument('--forbid-checkpoints',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();blocked=[x['sha256'] for x in json.loads(a.forbid_checkpoints.read_text())]
    result=plan_split(json.loads(a.audit.read_text()),json.loads(a.groups.read_text()),blocked)
    if a.out.exists():raise ValueError('Use a new plan path')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ['splits','quarantine_images','ready_for_training','checkpoint_policy']},indent=2))
