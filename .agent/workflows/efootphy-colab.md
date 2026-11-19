---
description: Train and evaluate the four Efootphy vision components through Colab CLI, with versioned results and bounded jobs.
---

Read AGENTS.md, the handover files, and docs/COLAB_AUTOMATION.vi.md.

1. Read the installed CLI guide with `colab skill`. Check its actual `--help` when a flag differs from the guide. Use Linux/macOS. With Windows, run the CLI and repository in WSL Linux.
2. Copy the pilot configuration to an ignored local configuration such as `.colab.private.json`. Verify the mounted project path, exact class order, pretrained weights, pose keypoint contract and source video. Keep private paths, identifiers and credentials out of Git.
3. Use `python -m tools.colab_pipeline plan --config .colab.private.json --run-id <unique-id>` to inspect the job. Authentication and Drive consent need the user at their terminal. Once credentials and Drive are ready, execute the requested experiment without asking again for already authorized actions.
4. Start with the existing-model baseline, then a one-epoch training canary. Run ball, actors, pitch and minimap independently so each experiment has a clear cause. Supply the verified dataset YAML for pitch and minimap before selecting these jobs. Use the existing preparation functions for ball/actors.
5. Keep the original working model until the candidate has matched video QC. Record independent-test status and original-resolution error, self-reported ball gaps, false ball associations, source FPS, actual processing FPS and unmeasured device latency separately.
6. Let Python scripts and the remote worker perform long loops. Read `summary.json` when complete. Use the normal fast agent for inventory, logs and repeatable runs. Escalate to a coding/reasoning model for timestamp, projection or identity bugs supported by evidence. After two failed fixes of the same problem, gather a new diagnostic before another attempt.
7. Check `controller.log` and per-step `execution.log` on failure. `--resume` verifies the same configuration and reuses completed steps, then resumes a saved training checkpoint when available. Collect reports and stop the owned session. If Drive is awaiting consent, show the exact mount command and session status promptly.
8. Update CURRENT_STATE.json and JOURNAL.md with exact commands and results. A completed train or ONNX export is a research artifact. Physical USB/HUD latency, live-game FPS and independent-match tracking require their own measurements.
