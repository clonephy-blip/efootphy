# Ball-first research workflow

The screen-space ball branch is independent of slower field/player inference.
It does not project airborne balls onto the pitch plane or infer possession from
a control marker. Android integration and sustained device performance remain
separate acceptance gates.

## Data and training

1. Preserve source annotations and checksums. Keep every augmented version of a
   source image in one split. A filename group is not a verified match identity.
2. Check source preprocessing. A square stretch export can distort ball and body
   geometry. `mainview_video --source-stretch` is an explicit research adapter;
   its separate x/y coordinate restoration is tested. Do not enable it for an
   unrelated model without inspecting training provenance.
3. `python -m tools.vision_qc.ball_crops --source SOURCE --out NEW_DIRECTORY`
   prepares native-resolution crops. Training uses labelled positive crops plus
   random negatives. Validation covers a fixed full-image grid without looking
   at ball coordinates when choosing crops. Parent image groups/hashes may not
   cross splits. Match independence still requires a verified grouping manifest.
4. `python -m tools.vision_qc.train_native_ball --data CROPS/data.yaml
   --weights FIELD_CHECKPOINT --out NEW_RUN` warm-starts a separate ball branch.
   It retains the seven-channel source class order and trains ball channel zero.
   Other channels are not suitable for player/referee inference after this run.
   The time budget is checked after saved epochs; a running epoch can exceed it.
5. `train_ball_patch` is an optional small appearance-baseline experiment, not the
   selected detector or a trained temporal model. Reject false-positive-heavy
   checkpoints even when their patch validation loss decreases.

The `half-pitch` training profile accepts `names: [half_pitch]` and
`kpt_shape: [18, 3]`. It is separate from the existing 32-point `landmarks`
profile. Preserve multiple half-pitch instances and visibility; do not convert
the rows into polygons or invent a mapping between the two landmark schemas.
Supply separate split directories before training. Keep geometric pose
augmentation disabled until landmark reflection semantics are verified.

## Consecutive video evaluation

Use `python -m tools.vision_qc.ball_video --video VIDEO --model BALL_ONNX
--out NEW_REPLAY --end-ms END --excluded-rects HUD_JSON`.

The runner decodes sequentially and processes every source frame in the selected
interval. It records original VFR timestamps, per-frame latency, raw candidates,
short predictions, losses, and an incremental JSONL journal. The preview concat
file uses original intervals, with millisecond time-base precision.

`OBSERVED` means an accepted appearance observation, not verified ground truth.
`PREDICTED` is short screen-space extrapolation and expires by elapsed time.
`LOST` has no drawable ball position. State counts are not precision or recall.
Score visible-ball centre error, false tracks, longest missed interval and
reacquisition delay against reviewed contiguous annotations, keeping hidden
frames distinct from visible misses. Check both flight and close contact.

`--flow` is experimental. Optical flow must pass forward/backward, texture and
patch checks and receives a distinct observation-source label. It cannot renew
an identity indefinitely without detector confirmation. Smooth player-following
tracks are failures. Keep flow disabled unless reviewed video QC improves
accuracy without hiding gaps. `--source-stretch` additionally assumes a 16:9
crop geometry; it is an experiment, not recovered camera calibration.

The cyan-triangle module produces UI-control candidates and requires a unique
nearby player body to associate them. Ambiguity stays explicit. Control is
distinct from roster identity, team, ball ownership and tactical intent.

## Export and deployment gates

Archive checkpoint, source/crop provenance, exact runtime versions, output
contract, ONNX checksum and numerical parity results. A successful export or
source AP improvement does not establish Mobile video accuracy. Keep private
images, predictions, weights and experiment measurements outside the public
repository. Do not claim 60 FPS from replay encoding or host CPU measurements.
