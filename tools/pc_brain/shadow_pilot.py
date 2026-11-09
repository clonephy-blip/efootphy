"""Replay frozen detector proposals through the optional shadow stage.

No model inference, label-generated seeds or accuracy claim. Output contains
private gameplay crops: keep it outside the public repository.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import tempfile
import time
import zipfile

import cv2
import numpy as np

from tools.pc_brain.ball_shadow import find_shadow_candidates


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def run(clips_zip, detections, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source = json.loads(Path(detections).read_text())
    groups = defaultdict(list)
    for record in source['records']:
        match = re.fullmatch(r'(val-\d+)-\d+', record['frame_id'])
        if match is None:
            raise ValueError('Expected frozen validation frame IDs')
        groups[match[1]].append(record)
    results, preview_candidates, timings = [], [], []
    cv2.setNumThreads(2)
    with zipfile.ZipFile(clips_zip) as archive, tempfile.TemporaryDirectory() as tmp:
        provenance = {p['clip_id']: p for p in json.loads(archive.read('source-clips/provenance.json'))}
        for clip_id, records in sorted(groups.items()):
            prov = provenance[clip_id]
            path = Path(tmp) / (clip_id+'.mp4')
            path.write_bytes(archive.read('source-clips/'+clip_id+'.mp4'))
            if sha256(path) != prov['clip_sha256']:
                raise ValueError('Source clip hash mismatch')
            indexed = {r['source_frame_index']-prov['source_first_frame']: r for r in records}
            if len(indexed) != len(records) or min(indexed) < 0 or max(indexed) >= prov['frames']:
                raise ValueError('Duplicate or out-of-range source frame')
            cap = cv2.VideoCapture(str(path))
            try:
                for index in range(max(indexed)+1):
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError('Cannot decode expected source frame')
                    if index not in indexed:
                        continue
                    record = indexed[index]
                    boxes = record['ball_candidates_xyxy_score']
                    box = boxes[0] if boxes else None
                    seed = None if box is None else ((box[0]+box[2])/2, (box[1]+box[3])/2)
                    radius = 4. if box is None else max(2., min(20., min(box[2]-box[0], box[3]-box[1])*.5))
                    actors = [box[:4] for cls in ('3', '4', '6') for box in record['actor_detections'].get(cls, [])]
                    t0 = time.perf_counter()
                    proposals = find_shadow_candidates(frame, seed, radius, player_boxes=actors)
                    elapsed_ms = (time.perf_counter()-t0)*1000
                    if seed is not None:
                        timings.append(elapsed_ms)
                    results.append(dict(frame_id=record['frame_id'], seed_xy_px=seed,
                                        shadow_candidates=[asdict(p) for p in proposals],
                                        elapsed_ms=elapsed_ms, seed_changed=False))
                    if seed is not None:
                        x, y = map(int, seed)
                        x0, y0 = max(0,x-256), max(0,y-256)
                        crop = frame[y0:min(frame.shape[0],y+257),x0:min(frame.shape[1],x+257)].copy()
                        cv2.circle(crop,(x-x0,y-y0),9,(0,255,255),2)
                        for n, p in enumerate(proposals):
                            px, py = int(p.xy_px[0]-x0),int(p.xy_px[1]-y0)
                            cv2.drawMarker(crop,(px,py),(255,0,255),cv2.MARKER_CROSS,14,2)
                            cv2.putText(crop,str(n+1),(px+8,py),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,255),1)
                        tile = np.zeros((330,400,3),np.uint8)
                        scale = min(400/crop.shape[1],280/crop.shape[0])
                        resized = cv2.resize(crop,None,fx=scale,fy=scale)
                        tile[:resized.shape[0],:resized.shape[1]] = resized
                        cv2.putText(tile,record['frame_id']+f' | proposals: {len(proposals)}',(7,300),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
                        cv2.putText(tile,'Yellow: detector / Magenta: UNVERIFIED',(7,321),cv2.FONT_HERSHEY_SIMPLEX,.43,(210,210,210),1)
                        preview_candidates.append((len(proposals),record['frame_id'],tile))
            finally:
                cap.release()
            path.unlink()
    summary = dict(
        status='research_shadow_proposals_only', model_sha256=source['model_sha256'],
        source_detections_sha256=sha256(detections), source_clips_sha256=sha256(clips_zip),
        frames=len(results), frames_with_detector_seed=len(timings),
        frames_with_shadow_proposals=sum(bool(r['shadow_candidates']) for r in results),
        total_shadow_proposals=sum(len(r['shadow_candidates']) for r in results),
        seeds_changed=0, shadow_accuracy=None, tracking_improvement=None,
        independent_match=False, calibrated_3d=False,
        incremental_ms_median=float(np.median(timings)) if timings else None,
        incremental_ms_p95=float(np.percentile(timings,95)) if timings else None,
        timing_scope='Local OpenCV CPU, 2 threads, proposal stage only; no inference/USB/render',
        records=results)
    (output_dir/'shadow_pilot.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Deliberately show both high-proposal and empty cases; not a representative accuracy sample.
    ordered = sorted(preview_candidates,key=lambda item:(-item[0],item[1]))
    chosen = ordered[:4]+sorted(ordered[4:],key=lambda item:(item[0],item[1]))[:4]
    if chosen:
        tiles = [item[2] for item in chosen]
        while len(tiles) < 8:
            tiles.append(np.zeros_like(tiles[0]))
        montage = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,8,2)])
        cv2.imwrite(str(output_dir/'shadow_proposal_review.jpg'),montage)
    return {k:v for k,v in summary.items() if k!='records'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clips-zip',type=Path,required=True)
    parser.add_argument('--detections',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.clips_zip,args.detections,args.output_dir),indent=2))
