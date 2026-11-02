# Full-view scene understanding: training upgrade design

The main video view supplies players, the physical ball, body motion and events.
The minimap is an auxiliary observation for off-camera coverage and consistency.
It cannot establish ball height, contact, player identity or referee decisions.
This document is a design, not a claim that the described models are trained or
integrated into Android.

## Selected research references

| Component | Reference | Intended use and adaptation |
|---|---|---|
| Athlete state | [SoccerNet GSR](https://github.com/SoccerNet/sn-gamestate) | Reference modular detection, calibration, identity and team workflow; not an Android dependency bundle |
| Identity continuity | [BoT-SORT](https://github.com/NirAharon/BoT-SORT), [ByteTrack](https://github.com/FoundationVision/ByteTrack) | Compare motion/appearance association under camera movement; track ID remains distinct from roster identity |
| Body motion | [RTMPose](https://github.com/open-mmlab/mmpose), [deployment](https://github.com/open-mmlab/mmpose/blob/main/docs/en/user_guides/how_to_deploy.md) | Candidate small body-pose model on selected player crops, exported for profiling |
| Tiny moving ball | [TrackNetV3](https://github.com/qaz812345/TrackNetV3) | Temporal heatmap idea; upstream targets badminton with background estimation and trajectory repair. Rework for moving camera, soccer and causal inference |
| Actor and event | [FOOTPASS](https://github.com/JeremieOchin/FOOTPASS) | Track-aware action and interaction reference; output actor/team/event/time. Broadcast training requires Mobile adaptation; game-level postprocessing is not automatically causal |
| Flight reconstruction | [ball3d](https://github.com/lukaszgrad/ball3d) | Offline physics/calibration reference. Upstream CC-BY-NC-SA and manually segmented benchmark inputs require separate consideration; not bundled into runtime |

Keep reference frameworks isolated: their Python/Torch requirements differ.
Source licences, checkpoint licences and dataset access are separate. No upstream
dataset access agreement is accepted or private footage uploaded by these tools.

## Trainable branches

1. **Player/official detector.** Fine-tune a small pretrained detector with the
   full reviewed image set. Compare a larger offline teacher and smaller runtime
   candidate only after a fixed evaluation protocol. Full-backbone fine-tuning
   must not reuse frozen-encoder feature caches as if they carried gradients.
2. **Physical-ball sequence model.** Preserve image detail using high-resolution
   crops and camera-aware search. Use past-frame heatmaps, visibility masks and
   contact-state annotations. Missing ball boxes are unknown unless absence was
   explicitly reviewed. Training crops may use labels; evaluation proposals must
   not use ground-truth ball centres. Whole-match median backgrounds and future
   trajectory repair cannot silently leak into an online benchmark.
3. **Identity and pose.** Associate players across frames, combine appearance,
   confirmed UI names/jersey evidence and roster metadata. Keep unknown identity
   when ambiguous. Train body pose/actions only with corresponding annotations;
   pitch keypoints are not human-body keypoints.
4. **Causal actor-action model.** Consume past player/ball features, pose, camera
   state and interactions. Start with pass, drive, shot, header, tackle and block;
   sprint, fall and foul need separate reviewed labels. Teacher proposals remain
   untrusted until reviewed. Predict possession probabilities rather than always
   assigning the nearest player.
5. **Ball-flight and outcomes.** Homography applies to the ground plane, not an
   airborne ball. Use a calibrated camera and a sequence model with uncertainty.
   Start with estimated flight state and a landing region; never claim an exact
   3D path from one ambiguous view. Game gravity/clock scaling is unverified.

## Runtime information contract

Every output carries source video/capture time, capture epoch, scene/cut ID,
confidence and provenance. Distinguish:

- `track_id` from optional `roster_player_id` and known game attributes;
- observed pitch-relative velocity from a hidden in-game speed rating;
- physical ball, ball shadow, cursor, control ring and minimap ball marker;
- observed contact/action from a tentative event and a confirmed game outcome;
- visible coordinates from forecasts during occlusion and unavailable state.

Use video PTS, not a nominal frame rate or the accelerated in-game match clock.
Reset temporal windows at menus, replay transitions and camera cuts. Feature
history can be retained cheaply; do not retain dozens of full-resolution RGBA
frames or queue stale detections behind the current game.

Goal/foul confirmation should incorporate independently observed game UI/state.
An animation alone does not establish a referee decision. All-player identity
and exact metric speeds remain unverified until their own tests pass.

## Work delivered by the current tools

`tools.vision_qc.mainview_video` runs a bounded, local, frozen field ONNX probe on
original video frames. It preserves source PTS and emits candidate boxes only;
counts are not accuracy, tracking or roster identification. Install
`requirements-video.txt` for its optional runtime dependency.

`tools.vision_qc.temporal_context` samples causal VFR windows without future
padding, gap bridging or known scene-cut crossing. Its split check requires
verified match IDs; matching filename prefixes do not establish independence.

`notebooks/gpu_field_pilot.ipynb` prepares a bounded GPU detector/landmark pilot
with the existing training entrypoint. It does not train action, identity or 3D
models and has not been GPU-validated in this environment. No silent CPU fallback
is allowed. Dataset preflight runs before training.

## Acceptance stages

Evaluate independently by match, with camera/theme/device variation. Keep task
metrics separate: person AP/recall, small-ball localization and visibility,
tracking HOTA/ID switches, identity attribution, per-action actor-aware precision
and temporal error, possession accuracy, and landing-region coverage/error.
Event predictions must be evaluated using only information available at their
reported decision time. State confidence and detection scores are not tactical
confidence. Profile capture, preparation, model, association and rendering on
the target device, including game frame time, thermal state and sustained memory.
