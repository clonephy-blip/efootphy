"""Audit a complete local YOLO export. Counts are decoded files, not README totals."""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

import yaml
from PIL import Image

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def split_directory(root, configured):
    if not isinstance(configured, str):
        raise ValueError('Only directory-based exports are supported by this auditor')
    candidate = (root / configured).resolve()
    # Roboflow exports commonly include ../train/images next to a root data.yaml.
    if not candidate.is_dir() and configured.startswith('../'):
        candidate = (root / configured[3:]).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_dir():
        raise ValueError(f'Missing or external split directory: {configured}')
    return candidate


def audit(data_yaml, expected_images=None, groups_file=None, require_test=True):
    data_yaml = Path(data_yaml).resolve()
    root = data_yaml.parent
    config = yaml.safe_load(data_yaml.read_text())
    if not isinstance(config, dict):
        raise ValueError('Dataset YAML must be an object')
    names = config.get('names', [])
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError('Class ids must be consecutive from zero')
        names = [names[i] for i in range(len(names))]
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names) or not names or len(names) != len(set(names)) or config.get('nc', len(names)) != len(names):
        raise ValueError('Class names/nc must be nonempty, unique and consistent')
    shape = config.get('kpt_shape')
    if shape and (len(shape) != 2 or shape[0] < 1 or shape[1] not in (2, 3)):
        raise ValueError('Invalid kpt_shape')
    task = 'pose' if shape else 'detect'
    columns = 5 + (math.prod(shape) if shape else 0)
    errors, warnings, records = [], [], []
    counts = collections.Counter()
    sizes = collections.Counter()
    instances = collections.Counter()
    fingerprints = collections.defaultdict(list)
    file_fingerprints = collections.defaultdict(list)
    groups = json.loads(Path(groups_file).read_text()) if groups_file else {}
    group_splits = collections.defaultdict(set)
    perceptual = []
    polygon_rows = 0
    missing_group = []
    for split, key in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
        configured = config.get(key)
        if not configured:
            (warnings if split == 'test' else errors).append({'code': 'missing_split', 'split': split})
            continue
        try:
            folder = split_directory(root, configured)
        except ValueError as exc:
            (warnings if split == 'test' else errors).append({'code': 'missing_split', 'split': split, 'detail': str(exc)})
            continue
        images = sorted(p for p in folder.rglob('*') if p.suffix.lower() in EXTENSIONS and p.is_file())
        if not images:
            errors.append({'code': 'empty_split', 'split': split})
        stems = set()
        for image in images:
            relative = image.relative_to(root).as_posix()
            if not image.resolve().is_relative_to(root):
                errors.append({'code': 'external_image', 'image': relative})
                continue
            label = folder.parent / 'labels' / image.relative_to(folder).with_suffix('.txt')
            stem = image.relative_to(folder).with_suffix('').as_posix()
            if stem in stems:
                errors.append({'code': 'ambiguous_image_stem', 'image': relative})
            stems.add(stem)
            group = groups.get(relative)
            if not group:
                missing_group.append(relative)
            else:
                group_splits[group].add(split)
            try:
                with Image.open(image) as im:
                    im.verify()
                with Image.open(image) as im:
                    rgb = im.convert('RGB')
                    size = rgb.size
                    pixel_hash = hashlib.sha256(rgb.tobytes() + str(size).encode()).hexdigest()
                    gray = rgb.convert('L').resize((9, 8))
                    values = list(gray.get_flattened_data())
                    phash = sum(int(values[y*9+x] > values[y*9+x+1]) << (y*8+x) for y in range(8) for x in range(8))
                with image.open('rb') as handle:
                    file_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
            except Exception as exc:
                errors.append({'code': 'invalid_image', 'image': relative, 'detail': str(exc)})
                continue
            counts[split] += 1
            sizes[f'{size[0]}x{size[1]}'] += 1
            fingerprints[pixel_hash].append((split, relative))
            file_fingerprints[file_hash].append((split, relative))
            perceptual.append((split, relative, phash))
            records.append({'image': relative, 'split': split, 'group': group,
                            'sha256': file_hash, 'pixel_sha256': pixel_hash, 'size': size})
            if not label.is_file():
                errors.append({'code': 'missing_label', 'image': relative})
                continue
            if not label.resolve().is_relative_to(root):
                errors.append({'code': 'external_label', 'image': relative})
                continue
            for line_no, line in enumerate(label.read_text().splitlines(), 1):
                if not line.strip():
                    continue  # Explicit empty labels are allowed for reviewed negative frames.
                context = {'image': relative, 'line': line_no}
                try:
                    row = [float(v) for v in line.split()]
                except ValueError:
                    errors.append({'code': 'not_numeric', **context})
                    continue
                if not row or not all(math.isfinite(v) for v in row):
                    errors.append({'code': 'non_finite', **context})
                    continue
                cid = int(row[0])
                if row[0] != cid or not 0 <= cid < len(names):
                    errors.append({'code': 'class_out_of_range', **context})
                    continue
                instances[names[cid]] += 1
                if len(row) != columns:
                    polygon = task == 'detect' and len(row) >= 7 and (len(row)-1) % 2 == 0
                    polygon_rows += int(polygon)
                    errors.append({'code': 'polygon_in_detection' if polygon else 'wrong_columns',
                                   'actual': len(row), 'expected': columns, **context})
                    continue
                if not all(0 <= v <= 1 for v in row[1:5]) or min(row[3:5]) <= 0:
                    errors.append({'code': 'invalid_bbox', **context})
                if shape:
                    for i in range(shape[0]):
                        point = row[5+i*shape[1]:5+(i+1)*shape[1]]
                        if not all(0 <= v <= 1 for v in point[:2]) or (shape[1] == 3 and point[2] not in (0, 1, 2)):
                            errors.append({'code': 'invalid_keypoint', 'keypoint': i, **context})
            with label.open('rb') as handle:
                records[-1]['label_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
        labels = folder.parent / 'labels'
        if labels.is_dir():
            for label in labels.rglob('*.txt'):
                stem = label.relative_to(labels).with_suffix('').as_posix()
                if stem not in stems:
                    errors.append({'code': 'orphan_label', 'label': label.relative_to(root).as_posix()})
    for entries in fingerprints.values():
        if len(entries) > 1:
            cross_split = len({s for s, _ in entries}) > 1
            (errors if cross_split else warnings).append({'code': 'duplicate_pixels_across_splits' if cross_split else 'duplicate_pixels',
                                                         'images': [p for _, p in entries]})
    near_pairs = []
    for i, (split, path, phash) in enumerate(perceptual):
        for other_split, other_path, other_phash in perceptual[i+1:]:
            if split != other_split and (phash ^ other_phash).bit_count() <= 3:
                if len(near_pairs) < 100:
                    near_pairs.append([path, other_path])
    if near_pairs:
        warnings.append({'code': 'near_duplicates_need_review', 'examples': near_pairs})
    if missing_group:
        warnings.append({'code': 'video_groups_missing', 'count': len(missing_group)})
    for group, splits in group_splits.items():
        if len(splits) > 1:
            errors.append({'code': 'video_group_leakage', 'group': group, 'splits': sorted(splits)})
    total = sum(counts.values())
    if expected_images is not None and total != expected_images:
        errors.append({'code': 'incomplete_export', 'decoded': total, 'expected': expected_images})
    for name in names:
        if not instances[name]:
            warnings.append({'code': 'class_without_instances', 'class': name})
    if require_test and not counts['test']:
        warnings.append({'code': 'independent_test_absent'})
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {'task': task, 'names': names, 'kpt_shape': shape, 'decoded_images': total, 'split_counts': dict(counts),
            'dimensions': dict(sizes), 'instances': dict(instances), 'polygon_rows': polygon_rows,
            'data_errors': errors, 'review_warnings': warnings, 'data_valid': not errors,
            'eligible_for_independent_evaluation': not errors and not warnings,
            'dataset_fingerprint': digest, 'images': records,
            'scope': 'all local files reachable through the configured split directories'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--expected-images', type=int)
    parser.add_argument('--groups', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.data, args.expected_images, args.groups)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['decoded_images','split_counts','polygon_rows','data_valid','eligible_for_independent_evaluation']}))
    raise SystemExit(0 if report['data_valid'] else 2)


if __name__ == '__main__':
    main()
