# Experimental paint alignment

`pitch_refinement.refine_calibration(frame, seed_H, world_lines, hud_spec)` takes
an existing semantic world-to-image calibration and proposes a small projective
correction using visible paint. It returns the selected matrix and diagnostics.
Images are uint8 BGR; HUD rectangles and circles use original image pixels.
Install `requirements-qc.txt` for the offline NumPy/OpenCV/SciPy dependencies.

The optimizer receives no reference points or manual calibration. It fixes its
sample set before optimization, masks the HUD, balances line families, and uses
a bounded robust distance objective. Samples moving outside valid pitch pixels
remain in the denominator with a penalty. Geometry and image-evidence gates
reject unsupported candidates and retain the seed. The fixed evaluation budget
is recorded separately from numerical convergence.

Paint agreement does not prove semantic accuracy: shirts, goals, line symmetry,
occlusion, incorrect pitch dimensions and incomplete visible geometry can all
mislead it. Compare every proposal against separately reviewed image references;
record rejected candidates as well. Never choose parameters or generate labels
from those evaluation references. Additional evaluation images are required
before making generalization claims.

This module is deliberately disconnected from Android capture and training.
`models_validated`, metric-scale verification, and automatic training import
remain false. Synthetic tests check implementation behavior only.
