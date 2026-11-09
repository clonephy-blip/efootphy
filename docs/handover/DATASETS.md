# Dataset roles — do not mix these folders

Root source folder: `train ai` on the user's connected Drive. Private IDs/URLs are intentionally not published here.

| Exact folder/file name | Intended task | Schema / status |
|---|---|---|
| `efootball_detection` | Objects in the main gameplay image | Expected seven classes: ball, cursor, cursor_secondary, goalkeeper, player, powerbar, referee. Read current YAML to verify. Not minimap data. |
| `UI Mini Map` | Markers inside the cropped minimap | Expected ball, controlled, dot. These markers do not establish aerial ball height or player identity. Run the model on PC. |
| `efootball-analytics-points.v2i.yolov8` | 32 semantic pitch landmarks | Separate schema from half_pitch. Current Drive train/labels listing contains 93 files; YAML/label audit in progress. |
| `efootball_pitch_calibration.v22-final_version.yolov8` | 18-point half-pitch calibration source | Separate IDs and object definition. Do not concatenate its rows with the 32-point set or assume numeric ID equivalence. Split independence must be audited. |
| `MATCH_01_PART_005.mp4`, `...007.mp4`, `...010.mp4` | Temporal footage and source samples | Existing research used 007 for train crops and 010 for reviewed validation. Reuse is not an independent-match blind test. 005 includes difficult aerial scenes. |
| `efootball_ai_colab_training.ipynb` | Training instructions | Code, not training data or proof of a successful checkpoint. Use the corrected notebook revision. |
| `efootball_ai_assistant_package.zip` | Earlier packaged project/artifact | Inspect before using; not automatically a raw dataset. |
| `efootphy_train_results_20261109_065253.zip` | User-trained weights and reports | Contains field, pitch and minimap runs; does not contain original annotation labels/data.yaml. |

No current source annotation has a verified `ball_shadow` class. Shadow proposals must be marked heuristic until a reviewed shadow-label set is available. Keep shadow labels separate from ball labels.
