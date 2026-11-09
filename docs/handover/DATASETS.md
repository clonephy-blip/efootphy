# Dataset roles — verified source YAML on 2026-11-09

Root: the user's connected Drive folder `train ai`. Private IDs and source annotations stay outside this public repository. Class order is part of the model contract.

| Exact folder | Actual schema | Intended use / checked scope |
|---|---|---|
| `efootball_detection` | `nc: 7`; `ball, cursor, cursor_secondary, goalkeeper, player, powerbar, referee` | Main gameplay object detection on PC. YAML verified this session; not a new full label audit of this folder. |
| `UI Mini Map` | `nc: 3`; `ball, controlled, dot` | Cropped minimap markers; inference belongs on PC. Runtime integration pending. YAML declares test but no test folder was visible at this root. |
| `efootball-analytics-points.v2i.yolov8` | `nc: 1`, `pitch`, `kpt_shape: [32,3]` | Source used in the Nov 9 Colab pitch run. Checked all 93 train and 10 valid label files. |
| `efootball_pitch_calibration.v22-final_version.yolov8` | `nc: 1`, `half_pitch`, `kpt_shape: [18,3]` | Different landmark IDs/object definition. Only 12 label files sampled this session (18 objects); train exists but valid/test folders were not visible despite YAML paths. Not consumed by the last notebook run. |

Both pitch YAML files have identity `flip_idx`; this needs semantic review before enabling horizontal augmentation, but is not proof of corrupt labels. Never concatenate 18-point and 32-point rows or assume matching numeric IDs.

## Measured annotation checks

- Pitch32 train: 93 rows, 101 values per row, 972 landmarks with visibility 1 or 2 (931 with 2, 41 with 1), no checked column/visibility/visible-coordinate range error. Every ID appears, with only 8–46 visible examples per ID.
- Pitch32 validation: 10 rows, 92 landmarks with visibility 2 plus 2 with visibility 1; ID 5 absent. All 10 downloaded exported JPEGs are 640×640.
- No matching source-basename prefix across the two pitch32 splits. This does **not** prove independent matches or non-adjacent video frames.
- Pitch18 sample: 12 files / 18 objects, 59 values per row, 117 visibility-1-or-2 points, no checked format/range error. ID 1 absent from this sample only. Do not extrapolate to all pitch18 files.
- These checks do not establish physical landmark semantics, annotation pixel accuracy or calibration quality.

## YAML SHA-256 fingerprints

- `field`: `1dd3837edc77669700b85d6dda731df241eb0767f1576cefeb3674c8e5a5dead`
- `minimap`: `b00ecf226989c6274927a1a5566f4b86e85d23fc9797ab4ed8456f86c0feb203`
- `pitch32`: `7f3afcaa2502e128eca29c37c6a1ccef83a53562870a3c345f898af411b0f3e5`
- `pitch18`: `40443b639c0afbd22e809d37093ca3244ff93ca8885d79501a3beddc375cd20a`

## Other source types

| Name | Role |
|---|---|
| `MATCH_01_PART_005.mp4`, `...007.mp4`, `...010.mp4` | Temporal footage: 005 contains aerial examples; prior research used 007 for train crops and 010 for reviewed validation. These are not independently held-out matches. |
| `efootball_ai_colab_training.ipynb` | Code/instructions, not a dataset. Use the corrected GitHub notebook. |
| `efootball_ai_assistant_package.zip` | Earlier packaged project; inspect contents before treating anything as data. |
| `efootphy_train_results_20261109_065253.zip` | User-trained field/pitch/minimap weights and reports; does not contain original labels/data.yaml. |

No verified current source class represents `ball_shadow`. Keep ball pixels, shadow pixels, occlusion state and associations separate in new temporal labels. Never label a shadow as a ball to increase ball recall artificially.
