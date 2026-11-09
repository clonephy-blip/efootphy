# eFootPhy: instructions for every AI contributor

Read `docs/handover/START_HERE.md`, `CURRENT_STATE.json`, and `JOURNAL.md` before changing this project.

## User requirements confirmed 2026-11-09

- Android sends gameplay video over USB/scrcpy. All capture analysis, model inference, tracking, calibration, minimap fusion and tactics run on the PC CPU/GPU.
- Android is a HUD receiver/renderer only. Do not add on-device inference or require MediaProjection consent for PC mode. Archived local-minimap code is legacy, not the delivery target.
- Read screens only. Do not inject into the game or generate game inputs. Start scrcpy with `--no-control`.
- A shadow is an additional visual observation, not automatically the ball or the ball's vertical ground projection. Distinguish observed ball, shadow proposal, short prediction and lost state.
- Never invent metric height/3D positions without current camera and lighting calibration. Unknown quantities stay unknown.
- Keep the 32-point pitch and 18-point half-pitch datasets separate until a verified semantic correspondence exists. Check actual YAML, class order, labels and splits; do not infer a dataset's task from a filename alone.
- Do not replace an accepted checkpoint based on train loss or overall mAP. Require matched ball/video QC and explicit artifact hashes. Phone/PC end-to-end performance remains a separate gate.

## Required record after every meaningful change or experiment

Update `docs/handover/JOURNAL.md` and `CURRENT_STATE.json` in the same change. Record: goal; files/modules changed; dataset/schema and model hashes; exact command and measured outcome; what remains unverified; blockers; and the next reproducible action. Label synthetic tests, offline replay and physical-device results separately. Include failures and rejected experiments. Never describe an experiment as completed before it runs.

At session end, publish the notes with the code and give the user the handover link. Every session must produce a concrete result or identify the exact blocking input promptly. Do not imply that work keeps running after the session ends.

## Privacy and collaboration

This repository is public. Commit code, schemas, reproducible recipes and bounded summaries only. Keep private video, screenshots, annotations, model binaries, Drive file IDs and signed transfer URLs outside Git. Preserve other AIs' branches. Use an isolated branch and a reviewable PR; do not merge release changes automatically.
