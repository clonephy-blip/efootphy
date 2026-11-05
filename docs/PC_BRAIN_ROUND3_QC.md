# PC Brain — Round 3: classwise detection and conservative kits (2026-11-05)

## Scope
Branch: `fix/field-classwise-nms-ball-20261105`. This is stacked on
`feat/pc-brain-transport-rtt-20261105` (round 2). This is NOT merged.

Changes:
- Field YOLO11 output contract: **[1,11,N]**, class order
  `ball, cursor, cursor_secondary, goalkeeper, player, powerbar, referee`.
  Rescale from **320×320 stretched input** to original scrcpy frame.
- Ball candidates use their own class score, independently from non-ball
  argmax. The best-scoring ball comes from ball-specific proposals after NMS.
- Class-local greedy NMS preserves overlapping player/referee/cursor proposals
  and collapses duplicates only within the same class.
- Nonfinite/out-of-frame/degenerate boxes are ignored; candidate ranking and
  per-class output counts are bounded.
- Jersey clustering initializes from the **most distinct two sampled torso
  colors**, holds ambiguous or missing kit data as UNKNOWN, and preserves
  centers across frames. All-team-detected, known team identity, and roster
  continuity are **not** established.
- Do not render pass suggestions when the ball is only extrapolated.

## Deterministic tests

Run under the same pinned Python dependencies as GitHub Actions:

```sh
python -m pip install -r tools/vision_qc/requirements-qc.txt
python -m unittest tools.pc_brain.test_runtime_safety tools.pc_brain.test_transport_rtt tools.pc_brain.test_field_postprocess -v
```

The new tests cover:
1. Overlapping player/referee boxes both survive class-local NMS.
2. Duplicate player proposals collapse.
3. The ball is not dropped merely because a cursor scores higher at that anchor.
4. Coordinates correctly undo 320×320 **stretch** for a 1280×720 frame.
5. Malformed, NaN, or out-of-frame proposals are rejected.
6. A one-kit scene remains UNKNOWN rather than manufacturing two teams.
7. Color clusters retain IDs when player detection order reverses.
8. Partial/ambiguous torso colors remain UNKNOWN.
9. Existing color centers work when only one team is currently in view.

## What these tests do NOT establish

- No new Roboflow or eFootball video was run in this branch.
- No measured precision, recall, HOTA, IDF1, or ball trajectory continuity gain.
- Ball recall may increase while false positives also increase. The `0.08`
  ball threshold is a **retained experiment parameter**, not calibrated.
- No guarantee that camera movement, jersey patterns, lighting changes, or
  controlled-player marker handling are correct.
- No verified 60 FPS; current processing still depends on PC/ADB/scrcpy hardware.

## Real-video gate (must pass before promoting)

Use **MATCH_01_PART_007** for new examples/training only;
**MATCH_01_PART_010** remains frozen validation;
**MATCH_01_PART_005** remains regression including the third airborne-ball case.
Do not move frames across these splits or pick model proposals using validation
ball centers. Preserve source PTS and frame IDs, and annotate uncertainty.

On the *same original frames and output settings*, compare round 2 vs this
round at fixed thresholds:
- Class-wise TP/FP/FN and AP/recall for balls, players, referees and cursors.
- Small-ball and airborne-ball presence, false ball tracks and longest lost span.
- Team confusion/UNKNOWN coverage, per-frame ID consistency and cursor linkage.
- Detection latency p50/p95, transport RTT p50/p95, game frame-time impact.

Keep this change behind a draft PR until actual video and phone acceptance.
No change to Android model import contract or official tactical-advice gate.
