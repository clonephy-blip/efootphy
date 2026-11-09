# Progress journal

## 2026-11-09 — Step 01: establish the PC architecture and common handover

**User request:** PC reads Android gameplay over USB and performs CPU/GPU analysis; Android only draws HUD/ball paths. Keep notes after changes so other AIs can continue. Investigate the ball's shadow as additional evidence during aerial motion.

**Done:** fetched current branch heads; integrated PC research base `324b9e9` and Colab packaging fixes `70d57ce` in an isolated local branch without changing existing branches. Read the current private QC report. Verified access to the Drive source folders, and found 93 train-label files in the 32-point pitch set. Added contributor rules, architecture, dataset-role index and machine-readable state.

**Evidence:** all three prior checkpoints have 30 epochs; frozen full-screen ball check is 11/38 for new Field 320 versus 25/38 for source Field 640; pitch pose AP remains zero. These prior results are not new training performed in this step.

**Not yet verified:** raw YAML/annotation audit; new shadow algorithms; PC-only Android cleanup; new unit/video tests; physical USB/GPU/phone behavior. Nothing has been promoted to a production model.

**Next:** materialize only the current YAML and pitch annotations needed for audit, record actual schemas, then implement bounded shadow evidence and calibrated geometry with independent tests. Keep all private source content outside this public repository.
