# Progress journal

## 2026-11-09 — Step 01: establish the PC architecture and common handover

**User request:** PC reads Android gameplay over USB and performs CPU/GPU analysis; Android only draws HUD/ball paths. Keep notes after changes so other AIs can continue. Investigate the ball's shadow as additional evidence during aerial motion.

**Done:** fetched current branch heads; integrated PC research base `324b9e9` and Colab packaging fixes `70d57ce` in an isolated local branch without changing existing branches. Read the current private QC report. Verified access to the Drive source folders, and found 93 train-label files in the 32-point pitch set. Added contributor rules, architecture, dataset-role index and machine-readable state.

**Evidence:** all three prior checkpoints have 30 epochs; frozen full-screen ball check is 11/38 for new Field 320 versus 25/38 for source Field 640; pitch pose AP remains zero. These prior results are not new training performed in this step.

**Not yet verified:** raw YAML/annotation audit; new shadow algorithms; PC-only Android cleanup; new unit/video tests; physical USB/GPU/phone behavior. Nothing has been promoted to a production model.

**Next:** materialize only the current YAML and pitch annotations needed for audit, record actual schemas, then implement bounded shadow evidence and calibrated geometry with independent tests. Keep all private source content outside this public repository.

## 2026-11-09 — Step 02: identify source schemas and reject failed pitch deployment

**Goal:** make every dataset's role explicit before more training. Read all four current `data.yaml` files; recorded their exact class order, keypoint count and SHA-256 in `DATASETS.md`. Downloaded/read all 93 train and 10 valid pitch32 annotations, the 10 valid images, and a separate 12-file pitch18 sample. Private artifacts: `label_audit_summary.json`, `pitch32_pixel_qc.json`, source YAML/labels/images.

**Measured:** no invalid column counts, visibility flags or visible normalized coordinates in the checked files. This is a structural audit, not semantic certification. Pitch32 model SHA `56e988423891528f587e0cf5256b1d4406287f7823d997c370ebc1dcb3e9f3c1`, OpenCV DNN CLASSIC, direct 640×640 RGB input, top-scoring pitch box, numeric keypoint IDs matched, predicted keypoint confidence ignored for this XY diagnosis: **0/92 v2 points within 5 px, 0/92 within 10 px; median 191.8623 px, mean 201.9892 px**. Visibility-1 points excluded from this 92-point score. Same source validation, not independent-match QC. Do not compare this median with old 98-pixel figures without reproducing identical preprocessing.

**Decision:** failed checkpoint stays blocked from geometry deployment. A valid label format does not explain why learning failed. Next: verify a shared semantic diagram and test whether a tiny train subset learns the right coordinates before paying for another full run. Do not merge pitch18/pitch32 labels.

## 2026-11-09 — Step 03: implement PC analysis + Android HUD and bounded shadow evidence

**Changed:** `app` launcher/service now only receives and renders PC observations. Archived eight local-capture/model files under `legacy/android-local` outside the APK source set; removed Android ONNX and vision-core app dependencies, MediaProjection permission/entry points. HUD distinguishes observed yellow ball and short predicted dashed gray ball, expires stale results and checks aspect ratio. Socket shutdown closes the active client. Windows launcher validates USB/model paths, uses read-only scrcpy and environment overrides. PC captures only the exact scrcpy window, selects available ORT providers explicitly, uses monotonic ball timestamps, limits prediction to 150 ms, and disables unvalidated possession/pass hints.

**Shadow module:** optional local dark-on-grass proposals; candidates never change the chosen ball. Separate ray-geometry function requires a current camera matrix, camera epoch and directional-light vector. Unknown/stale/degenerate geometry returns no 3D estimate. Runtime emits `UNCALIBRATED` and no ball height. Default `EFOOTPHY_SHADOW_EVIDENCE=0`.

**Local validation commands:**

```bash
PYTHONPATH=/tmp/efootphy-qc-deps-20261109 python -m unittest discover -s tools/pc_brain -t . -p 'test_*.py' -v
PYTHONPATH=/tmp/efootphy-qc-deps-20261109 python -W error::ResourceWarning -m unittest discover -s tools/vision_qc -t . -p 'test_*.py' -v
python -m compileall -q tools/pc_brain
git diff --check
```

**Results:** 73 PC tests passed, including calibrated synthetic geometry, bad calibration, provider choice, short prediction/reacquisition, no shadow-only ball, and exact-window capture selection. 213 data-tool tests ran: 206 passed, 7 skipped because optional PyTorch training dependencies are absent. These do not test native ORT, GPU, Windows capture or a physical phone. Android lint/build CI pending at this commit.

**Offline shadow pilot command:** from repo root, using the existing private research inputs alongside this checkout:

```bash
PYTHONPATH=/tmp/efootphy-qc-deps-20261109 python -m tools.pc_brain.shadow_pilot --clips-zip ../post-train-qc-20261109/research-inputs/efootphy-round2-source-clips.zip --detections ../post-train-qc-20261109/outputs/field640_source_frozen38.json --output-dir ../pc-data-audit-20261109/shadow-pilot
```

The replay checks source clip hashes and source frame indices. Frozen Field640 outputs seed the shadow search; ground truth is not used for search. Model SHA `3d4119b1247849a9ecc92ee31498b78b0f9e6ddd6815fc8485db2c2c1abc47c6`. Same 38 visible reviewed research frames, 33 with detector seeds. Initial pilot: 92 proposals across 25 frames, p50 3.877 ms / p95 5.631 ms. Visual review exposed text/control-marker false positives. One bounded correction excludes bright neutral/cyan edges and tightens green hue; this can also reject real shadows near white lines.

**Re-run:** 8 proposals across 5 frames; p50 5.519 ms / p95 8.098 ms, local CPU 2 threads, shadow stage only. Seeds changed: 0. Remaining proposals still include player/HUD artifacts in the inspected montage. Fewer proposals is **not** measured higher accuracy or tracking improvement. No verified shadow labels, no precision/recall, no calibrated 3D, no independent match. Both v0 and final pilot JSON/crops retained privately. Do not promote this heuristic as a reliable shadow detector.

**Blockers and next reproducible work:** label separate ball/shadow pairs on timestamped aerial clips and add temporal association; verify camera/light assumptions before any metric trajectory. For end-to-end tests, record Windows CPU/GPU/provider, source/model hashes, capture size, USB RTT and HUD-off/on video. Current MSS capture can include letterboxing, occlusion and the phone's own HUD, and needs physical viewport/feedback tests. Minimap-PC integration and persistent identity remain pending. Runbook: `RUN_PC.md`.

## 2026-11-09 — Step 04: successful Android CI and preserved evidence

Implementation commit: `30a879d8fd4e3dd4e80cfc860b57ebf6d5498501`. [GitHub Actions run 42584545761](https://github.com/clonephy-blip/efootphy/actions/runs/42584545761) completed successfully for both data-tools and Android. Downloaded reports and checked their published SHA-256 before reading them: **48 JVM tests, 0 failures/errors/skips; Android lint 0 errors, 15 warnings; APK assembled successfully**. Warnings concern target/dependency updates, backup/icon configuration, bubble accessibility and text internationalization. Passing lint is not zero warnings or a physical-device certification.

APK: 2,189,116 bytes, SHA-256 `33e353700a73e6d05adcc8ef44a0f96ae7b8ffaa9140fda9b164eff1652f4fdf`, artifact `13658387379`. ZIP inspection found no native model/ONNX libraries. No Windows/phone install or USB performance test was run here.

Saved private audit + both shadow pilots as `efootphy_pc_shadow_qc_20261109.zip` in the user's ChatGPT files: 137 archive entries, SHA-256 `5242bd15c301bf0b139796a6848c40bce60dcb984b765bba684fd30d32fe7020`. Contains the checked labels/YAML, 10 validation images and private QC crops; do not copy into public GitHub.

An additional Drive handover folder creation was not executed: automatic approval review failed due to usage limit, not a finding that the content was unsafe. Do not retry through another route to bypass that check. Existing GitHub handover is the canonical record.

**Resume here:** annotate ball/shadow association on aerial clips with timestamps and held-out clip boundaries; verify pitch semantic IDs and a small learning canary; then measure the USB/HUD path on the real PC/phone. The current session improved architecture, bounded state handling, data traceability and QC. It did not train a new checkpoint, certify ball-shadow accuracy or complete 3D reconstruction.
