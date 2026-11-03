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

## Reviewed Mobile video crops

`video_ball_crops --plan CLIPS_JSON --reviews REVIEWS_JSON --out NEW_CROPS
--replay-data EXISTING_CROPS` builds an appearance dataset from source-bound
reviews. Keep each original video in one split; adjacent parts of a match do
not establish independent-match validation. Review the actual ball geometry,
not only a model score. Image/video hashes, source frame index, original PTS,
visibility and three distinct causal context references are required. Unknown
or occluded positions are withheld rather than exported as background labels.
Context references preserve provenance; this exporter does not train a temporal
model. Any use of future frames to assist offline review must be declared and
must never become runtime input.

Train crops use jittered ball positions, uniform background crops and optional
reviewed false-object candidates. Validation crops use the same fixed full-frame
grids regardless of the reference ball position. Original split replay can be
mixed in to measure forgetting. Keep the full-frame held-out benchmark as well:
crop AP alone can hide reacquisition failures and HUD false positives.

`train_native_ball` accepts `--learning-rate` and `--stop-after-epochs`. The
latter stops after saving an epoch while retaining the requested LR schedule.
Each completed epoch publishes fsynced best/last checkpoints and a `ready.json`
with hashes and provenance. Copy completed snapshots to durable storage while
long runs continue; a local snapshot alone does not survive workspace resets.

`score_ball_frames` scans each reviewed held-out full frame without choosing
crops from ground truth. Missing visible balls stay in the recall denominator;
top-one results and any-candidate results are separate. Continuous scoring also
separates genuine observations from short predictions. Missing-position duration
uses original frame intervals, including the final missed frame, and resets at
unknown references or missing source intervals. Freeze thresholds and masks
before comparing checkpoints.

`video_ball_crops --base-crops PREVIOUS_CROPS` appends reviewed training frames
while copying the existing validation payload byte for byte. It rejects any
validation-source extension, duplicated source frame, changed payload count or
crop identity collision. A fingerprint of every validation image and label is
recorded before and after the operation. Keep the original dataset and the
extension provenance available together.

`--backend opencv` on `ball_video` and `score_ball_frames` selects local OpenCV
DNN CPU inference. Where engine selection is available, it explicitly chooses
the classic engine; this code path does not import ONNX Runtime. Validate raw
output parity against PyTorch before changing runtime, and remeasure latency.
The optional ONNX Runtime path explicitly disables its telemetry before
creating a session. Training sets the upstream offline/autoinstall settings
before importing Ultralytics and disables synchronization.

`--refine-reacquire` is an experimental search-policy alternative. While lost,
it can confirm up to two weak coarse candidates using native-resolution crops
of the same current frame. A nearby native observation must meet the normal
acquisition threshold; weak coarse scores alone cannot create a track. An
active track still uses its ordinary single native crop. Compare both policies
on the same model, and both model versions under the same policy. Additional
forward passes cost time and may introduce false positives: keep the option
explicit and score no-ball states as well as visible-ball recall. No future
frames or ground-truth crop coordinates are used during inference.

For a strictly local continuation, `train_ball_local --data CROPS/data.yaml
--weights CHECKPOINT --out NEW_RUN` uses PyTorch, the existing network and its
detection loss directly. It does not construct an Ultralytics Trainer,
Validator or Exporter, and has no integration callbacks. A Python audit hook
refuses socket connect, DNS and send operations in that worker. The loop
records finite losses/gradients, optimizer updates, complete-epoch snapshots,
final EMA selection and frozen validation identity. It trains only channel
zero; all other checkpoint channels remain unsuitable for actor inference.
Its seed, augmentation and optimizer details differ from the upstream trainer
and must be reported rather than treated as an identical training recipe.

The selected model is the final EMA after a predefined epoch count, with no
validation-based checkpoint selection. ONNX export uses the local PyTorch
export API and ONNX graph checks. `evaluate_ball_crops_local` computes ball AP
from local model inference, NMS and metric functions without dataset download
or Validator callbacks. First reproduce a stored baseline on the same crop
payload; differing metric implementations cannot be compared silently.

Crop validation shape matters: an upstream rectangular validator can add a
stride-dependent border even to square source crops. The local AP runner uses
an exact 320px input by default, matching the deployed tensor contract.
`--padding 16` reproduces a legacy 352px padded diagnostic when needed; it is
not an alternative input for the fixed 320px ONNX model. Record the input side
and compare both checkpoints under the same preprocessing. Keep training
quality, full-frame reacquisition, false alarms and contiguous tracking as
separate acceptance gates; improved crop AP alone cannot select a model.

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
