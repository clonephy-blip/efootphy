# Named paint lines and image junctions

This offline module adds two separate outputs after an automatic pitch detector:

- `calibrate_semantic_lines(frame, seed_H, segments, point_proposals, hud_spec)`
  fits visible named straight lines, proposes a joint homography, and retains the
  seed when line rank, consensus, reprojection or geometric checks fail.
- `observe_line_junctions(keypoints, segments, line_evidence, image_size, hud_spec)`
  measures visible image intersections of supported nonparallel lines. A local
  junction can be observable while the global homography remains unusable.

Inputs use original image pixels and consistent world XY units. `frame` is uint8
BGR. A segment is `{id, world_xy: [[x0,y0],[x1,y1]]}`; an optional automatic point
proposal is `{world_xy, image_xy, confidence}`. HUD rectangles use `[x,y,w,h]` and
circles `[x,y,r]`. Keypoint requests use `{id,world_xy}`. Neither API accepts human
reference coordinates or a manual calibration as a quality target.

Automatic points only relocate search bands. Green-channel dominance rejects
yellow advertising as a grass seed; closing connects grass across painted lines.
Masked paint then needs distributed line consensus and orientation support.
Line DLT constrains world points to image lines without pretending arbitrary
segment endpoints correspond. Rank-deficient parallel evidence is withheld.

Junctions require two semantic segments, sufficient crossing angle, visible
paint support near the intersection, and a location outside HUD. Curves, spots,
hidden points and long extrapolations remain unknown. The returned uncertainty
is a geometric heuristic, not a calibrated confidence interval.

Evaluate image-keypoint outputs separately from homography projections. Keep
the same reference denominator and explicitly identify every unchanged baseline
fallback. Independently inspect projected conics; do not fit conics or choose
parameters from evaluation references. Better line-fit residuals alone do not
establish better landmark accuracy.

The module does not train models, generate trusted labels, or connect to Android
capture. Outputs keep model validation, metric verification and automatic label
import disabled. Existing detector identities and assumed field geometry can be
wrong. New video/device evaluation remains necessary before runtime integration.
