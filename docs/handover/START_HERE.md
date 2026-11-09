# Start here — eFootPhy PC + USB + Android HUD

Last requirements update: 2026-11-09, Asia/Saigon.

The delivery architecture is **Android game → USB/scrcpy → PC vision/tracking → USB JSON → Android HUD**. PC-side capture/CPU/GPU inference is the target. The phone only renders observations, uncertainty and trajectories. A 60 Hz HUD is not evidence of 60 AI updates/s or 60 FPS game performance.

## Read in order

1. `CURRENT_STATE.json`: working branch, checkpoints, gates and next actions.
2. `DATASETS.md`: which source folder belongs to which task; 32-point and 18-point schemas are different.
3. `JOURNAL.md`: changes, actual test commands, measured results and rejected ideas.
4. `RUN_PC.md`: beginner-friendly Windows/USB run instructions.
5. `../../AGENTS.md`: contributor rules, including a progress note for every meaningful step.

Working integration branch: `work/pc-hud-shadow-20261109`. Source bases: PC research `324b9e9dfa7f4a821e16e33c3f36d0838f234e5b` plus Colab packaging fix `70d57ce9c50e223ae14918be8a362bd0b720618d`. This is research, not a deployed release. `main` still contains older app code; a handover index on main does not mean the integration code has been released.

## Verified before this session

- Three user-trained checkpoints completed 30 epochs on 2026-11-09. Archive: `efootphy_train_results_20261109_065253.zip`.
- Field 320: source validation AP50 player ≈0.966, referee ≈0.952, ball ≈0.102. Same frozen 38-frame full-screen video test: ball top-1 correct within 10 original pixels = **11/38**, versus **25/38** for source Field 640. Different weights and resolution; not a size-only ablation or independent-match certification.
- Pitch 32: pose precision/recall/AP all zero across 30 epochs. A high whole-pitch bounding-box score does not validate landmarks, homography or distances.
- Minimap 320: class order `ball, controlled, dot`, input `[1,3,320,320]`, output `[1,7,2100]`; structurally verified trial model. Its inference belongs on PC in the target architecture. The earlier Android import ZIP is a legacy compatibility artifact.
- No physical Windows/phone end-to-end latency, game FPS, persistent player identity or full 3D accuracy has been certified.

Private QC deliverables: `efootphy_qc_20261109.md`, `efootphy_qc_evidence_20261109.zip`, `field320_vs_field640_offline.mp4`, `pitch_labels_vs_predictions.jpg`. Do not upload their private source assets to GitHub.

## Shadow direction

Use ball detections, a separately detected shadow, player occlusion, camera motion and temporal consistency as distinct evidence. Estimate height only with validated camera/light geometry. A dark spot alone must never become a confirmed ball detection. Start with an opt-in shadow evidence module and measure false associations before promoting it.

## Current concrete work

Read the current journal for progress on data audit, the PC-only mode, shadow research and tests. Do not restart broad training until the pitch annotation contract has been checked.

This session completed the raw schema audit, PC HUD-only implementation and 38-frame shadow pilot. Read Step 02/03 in the journal: the pitch checkpoint remains rejected and the shadow heuristic remains opt-in research.
