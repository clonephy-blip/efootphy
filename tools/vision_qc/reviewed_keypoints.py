"""Apply explicit, hash-guarded human label corrections to a separate copy."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path


def apply_reviewed(source, destination, review):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents or destination.exists():
        raise ValueError('Destination must be new and outside source')
    if isinstance(review, (str, Path)):
        review = json.loads(Path(review).read_text())
    edits = []
    seen = set()
    for item in review['corrections']:
        relative = Path(item['label'])
        path = (source / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(source) or path.suffix != '.txt' or path in seen:
            raise ValueError('Unsafe or repeated label path')
        seen.add(path)
        original = path.read_bytes()
        if hashlib.sha256(original).hexdigest() != item['sha256']:
            raise ValueError('Source label hash differs from review')
        lines = [line for line in original.decode().splitlines() if line.strip()]
        if len(lines) != 1:
            raise ValueError('Expected one pitch instance')
        row = [float(x) for x in lines[0].split()]
        if len(row) != 101 or row[0] != 0 or not all(math.isfinite(x) for x in row):
            raise ValueError('Expected finite 32x3 pitch label')
        before = row.copy()
        for op in item['operations']:
            ids = op['ids']
            if not ids or any(type(i) is not int or not 0 <= i < 32 for i in ids) or len(set(ids)) != len(ids):
                raise ValueError('Invalid keypoint IDs')
            if op['kind'] == 'swap' and len(ids) == 2:
                a,b = [5 + 3*i for i in ids]
                row[a:a+3],row[b:b+3] = row[b:b+3],row[a:a+3]
            elif op['kind'] == 'missing' and len(ids) == 1:
                a = 5 + 3*ids[0]; row[a:a+3] = [0,0,0]
            else:
                raise ValueError('Unknown review operation')
        edited = (' '.join(format(x,'.12g') for x in row)+'\n').encode()
        edits.append((relative,original,edited,{'label':relative.as_posix(),'source_sha256':item['sha256'],
            'reviewed_sha256':hashlib.sha256(edited).hexdigest(),'operations':item['operations'],
            'reason':item['reason'],'before':before[5:],'after':row[5:]}))
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns('*.cache'))
    for relative, original, edited, _ in edits:
        if (source/relative).read_bytes() != original:
            raise RuntimeError('Source changed while applying review')
        (destination/relative).write_bytes(edited)
    report={'schema_version':1,'source_modified':False,'model_predictions_used':False,
            'review_scope':'explicit human review; not an independent annotation certification',
            'corrections':[x[3] for x in edits]}
    (destination/'reviewed-corrections.json').write_text(json.dumps(report,indent=2))
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--review',required=True,type=Path)
    a=p.parse_args();print(json.dumps(apply_reviewed(a.source,a.out,a.review),indent=2))
