"""Create a separate YOLO detection dataset, converting polygon rows to boxes.

Usage: python normalize_detection_labels.py SOURCE_DIRECTORY NEW_OUTPUT_DIRECTORY
The source must contain data.yaml. Pose datasets are rejected explicitly.
"""
import argparse
import collections
import json
import math
import shutil
from pathlib import Path

import yaml


def normalize_row(line, nc):
    values = [float(value) for value in line.split()]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError('Empty or non-finite row')
    cid = int(values[0])
    if values[0] != cid or not 0 <= cid < nc:
        raise ValueError('Class index out of range')
    converted = False
    if len(values) == 5:
        coords = values[1:]
    elif len(values) >= 7 and (len(values)-1) % 2 == 0:
        points = values[1:]
        if not all(0 <= value <= 1 for value in points):
            raise ValueError('Polygon coordinate out of range')
        xs, ys = points[0::2], points[1::2]
        xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        coords = [(xmin+xmax)/2, (ymin+ymax)/2, xmax-xmin, ymax-ymin]
        converted = True
    else:
        raise ValueError(f'Unrecognized detection/polygon row: {len(values)} fields')
    if not all(0 <= value <= 1 for value in coords) or min(coords[2:]) <= 0:
        raise ValueError('Invalid normalized bounding box')
    return f'{cid} ' + ' '.join(format(value, '.12g') for value in coords), converted, cid


def normalize_dataset(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError('Output must be outside the source dataset')
    config = yaml.safe_load((source/'data.yaml').read_text())
    if config.get('kpt_shape'):
        raise ValueError('Pose/keypoint data must keep its keypoint labels; not a detection conversion')
    names = config['names']
    nc = len(names)
    if config.get('nc', nc) != nc:
        raise ValueError('nc does not match class names')
    # Validate the available pairs before creating any output.
    outputs = []
    counts = collections.Counter()
    converted_total = 0
    rows_total = 0
    for split in ['train', 'valid', 'test']:
        image_index = {p.stem: p for p in (source/split/'images').glob('*') if p.is_file()}
        for label in sorted((source/split/'labels').glob('*.txt')):
            image = image_index.get(label.stem)
            if image is None:
                raise ValueError(f'Missing image for {label}')
            rows = []
            changed = []
            for line_no, line in enumerate(label.read_text().splitlines(), 1):
                if not line.strip():
                    continue
                row, converted, cid = normalize_row(line, nc)
                rows.append(row)
                counts[str(cid)] += 1
                rows_total += 1
                if converted:
                    converted_total += 1
                    changed.append({'line': line_no, 'source_fields': len(line.split()), 'output_row': row})
            outputs.append({'split': split, 'source_label': label, 'source_image': image,
                            'rows': rows, 'converted_rows': changed})
    if not outputs:
        raise ValueError('No matching image-label pairs found')
    destination.mkdir(parents=True, exist_ok=False)
    available_splits = set()
    for item in outputs:
        split = item['split'];available_splits.add(split)
        labels = destination/split/'labels';labels.mkdir(parents=True,exist_ok=True)
        images = destination/split/'images';images.mkdir(parents=True,exist_ok=True)
        (labels/item['source_label'].name).write_text('\n'.join(item['rows'])+'\n')
        shutil.copy2(item['source_image'], images/item['source_image'].name)
    copied = dict(config)
    copied.pop('path',None)
    copied['train'] = 'train/images'
    copied['val'] = 'valid/images' if 'valid' in available_splits else None
    copied['test'] = 'test/images' if 'test' in available_splits else None
    (destination/'data.yaml').write_text(yaml.safe_dump(copied,sort_keys=False))
    for name in ['README.dataset.txt','README.roboflow.txt']:
        if (source/name).is_file():
            shutil.copy2(source/name,destination/name)
    result = {'scope': 'Only the files present in the local source directory',
              'input_image_label_pairs': len(outputs), 'label_rows': rows_total,
              'polygon_rows_converted': converted_total, 'class_instances':dict(counts),
              'conversions': [{'split':x['split'],'file':x['source_label'].name,
                               'converted_rows':x['converted_rows']} for x in outputs if x['converted_rows']],
              'source_modified':False}
    (destination/'normalization.json').write_text(json.dumps(result,indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('destination',type=Path)
    args=parser.parse_args()
    print(json.dumps(normalize_dataset(args.source,args.destination),indent=2))
