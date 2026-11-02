"""Bounded CPU appearance-baseline training from existing ball annotations.

This does not invent temporal supervision from unrelated still images. The
output is a research bundle and must pass consecutive-video QC separately.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import yaml

from .ball_vision import ball_proposals, make_patch_model, patch_batch
from .normalize import normalize_row


def load_split(root, split, seed, max_images):
    rng = np.random.default_rng(seed)
    images = sorted((root/split/'images').glob('*'))
    if not images or len(images) > max_images:
        raise ValueError('Missing images or image budget exceeded')
    patches, targets, records, evaluation = [], [], [], []
    for image in images:
        if image.suffix.lower() not in ('.jpg', '.png', '.jpeg', '.webp'):
            continue
        frame = cv2.imread(str(image))
        if frame is None:
            raise ValueError('Unreadable source image')
        h, w = frame.shape[:2]
        balls = []
        label = root/split/'labels'/(image.stem+'.txt')
        for line in label.read_text().splitlines():
            if not line.strip():
                continue
            row, _, cid = normalize_row(line, 7)
            if cid == 0:
                box = np.array(list(map(float, row.split()[1:])))*[w,h,w,h]
                balls.append(box)
        proposals = ball_proposals(frame)
        for box in balls:
            centres = np.clip(box[:2]+rng.uniform(-5,5,(10,2)), [0,0], [w-1,h-1])
            patches.extend(patch_batch(frame, centres.tolist()))
            targets.extend([1.]*len(centres))
        negatives = [p for p in proposals if all(np.linalg.norm(np.array(p[:2])-b[:2]) > max(28.,max(b[2:])) for b in balls)]
        rng.shuffle(negatives)
        negatives = negatives[:24]
        for _ in range(12):
            point = rng.uniform([0,0],[w-1,h-1])
            if all(np.linalg.norm(point-b[:2]) > max(28.,max(b[2:])) for b in balls):
                negatives.append(point)
        patches.extend(patch_batch(frame, negatives))
        targets.extend([0.]*len(negatives))
        proposal_patches = patch_batch(frame, proposals)
        evaluation.append((proposal_patches, np.asarray([p[:2] for p in proposals]).reshape(-1,2), balls))
        records.append({'file':image.name, 'sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                        'label_sha256':hashlib.sha256(label.read_bytes()).hexdigest(), 'balls':len(balls),
                        'group_candidate':image.name.split('.rf.')[0]})
    return np.asarray(patches, np.float32), np.asarray(targets, np.float32), records, evaluation


def evaluate_proposals(model, cases, threshold=.9):
    import torch
    total, proposal_hits, top1_hits, accepted, correct = 0,0,0,0,0
    with torch.inference_mode():
        for patches, points, balls in cases:
            total += len(balls)
            for b in balls:
                if len(points) and np.min(np.linalg.norm(points-b[:2],axis=1)) <= 10:
                    proposal_hits += 1
            if not len(points):
                continue
            scores = torch.sigmoid(model(torch.from_numpy(patches))).numpy()[:,0]
            k = int(scores.argmax())
            hit = bool(balls) and min(np.linalg.norm(points[k]-b[:2]) for b in balls) <= 10
            top1_hits += int(hit)
            if scores[k] >= threshold:
                accepted += 1
                correct += int(hit)
    return {'balls':total, 'proposal_hits_within_10px':proposal_hits,
            'top1_hits_within_10px':top1_hits, 'accepted_frames':accepted,
            'accepted_correct_frames':correct, 'threshold':threshold}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--max-seconds', type=float, default=300.)
    p.add_argument('--max-images-per-split', type=int, default=256)
    a = p.parse_args()
    if a.out.exists() or not 1 <= a.epochs <= 100 or not 0 < a.max_seconds <= 900:
        raise ValueError('Use a new output path and bounded training budget')
    config = yaml.safe_load((a.data/'data.yaml').read_text())
    if config.get('kpt_shape') or config.get('names') != ['ball','cursor','cursor_secondary','goalkeeper','player','powerbar','referee']:
        raise ValueError('Wrong source class schema')
    import torch
    import onnx  # Fail before training if the export dependency is missing.
    torch.set_num_threads(2)
    torch.manual_seed(20261102); np.random.seed(20261102); random.seed(20261102)
    cv2.setNumThreads(1)
    started = time.perf_counter()
    tx, ty, tr, _ = load_split(a.data, 'train', 17, a.max_images_per_split)
    vx, vy, vr, ev = load_split(a.data, 'valid', 29, a.max_images_per_split)
    if {r['sha256'] for r in tr} & {r['sha256'] for r in vr}:
        raise ValueError('Duplicate source image across splits')
    if {r['group_candidate'] for r in tr} & {r['group_candidate'] for r in vr}:
        raise ValueError('Same source filename group across splits')
    if min(ty.sum(), vy.sum()) < 10:
        raise ValueError('Too few existing ball annotations')
    a.out.mkdir(parents=True)
    print(json.dumps({'prepared_seconds':time.perf_counter()-started,'train_patches':len(tx),'valid_patches':len(vx),'train_images':len(tr),'valid_images':len(vr)}),flush=True)
    model = make_patch_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(ty)-ty.sum())/ty.sum()]))
    x, y = torch.from_numpy(tx), torch.from_numpy(ty[:,None])
    valid_x, valid_y = torch.from_numpy(vx), torch.from_numpy(vy[:,None])
    (a.out/'source-records.json').write_text(json.dumps({'source_train':tr,'source_validation':vr},indent=2))
    history=[]; best=float('inf')
    for epoch in range(a.epochs):
        model.train(); losses=[]
        for indices in torch.randperm(len(x)).split(64):
            batch = x[indices].clone()
            if random.random() < .5: batch = batch.flip(3)
            if random.random() < .5: batch = batch.flip(2)
            batch = (batch*random.uniform(.8,1.2)).clamp(0,1)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch),y[indices]);loss.backward();optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            logits = torch.cat([model(b) for b in valid_x.split(128)])
            val_loss = float(loss_fn(logits,valid_y))
        record = {'epoch':epoch+1,'train_loss':float(np.mean(losses)),'valid_loss':val_loss,'elapsed_s':time.perf_counter()-started}
        history.append(record);print(json.dumps(record),flush=True)
        if val_loss < best:
            best=val_loss;torch.save(model.state_dict(),a.out/'best.pt')
        (a.out/'training-history.json').write_text(json.dumps(history,indent=2))
        if time.perf_counter()-started >= a.max_seconds:
            break
    model.load_state_dict(torch.load(a.out/'best.pt',map_location='cpu',weights_only=True));model.eval()
    metrics = evaluate_proposals(model,ev)
    (a.out/'source-proposal-metrics.json').write_text(json.dumps(metrics,indent=2))
    torch.onnx.export(model,torch.zeros(1,3,48,48),str(a.out/'ball-patch.onnx'),
                      input_names=['rgb_patches'],output_names=['ball_logits'],
                      dynamic_axes={'rgb_patches':{0:'batch'},'ball_logits':{0:'batch'}},opset_version=17,dynamo=False)
    digest=hashlib.sha256((a.out/'ball-patch.onnx').read_bytes()).hexdigest()
    report={'schema':'ball-patch-research-v1','seed':20261102,'input':'RGB float32 [N,3,48,48] / 255',
            'output':'Uncalibrated physical-ball centre logits [N,1]', 'onnx_sha256':digest,
            'source_train':tr,'source_validation':vr,'history':history,'source_proposal_metrics':metrics,
            'elapsed_seconds':time.perf_counter()-started,'independent_match_validation':False,
            'temporal_model_trained':False,'models_validated':False,'automatic_training_import_allowed':False}
    (a.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'source_proposal_metrics':metrics,'elapsed_seconds':report['elapsed_seconds'],'onnx_sha256':digest}),flush=True)


if __name__ == '__main__':
    main()
