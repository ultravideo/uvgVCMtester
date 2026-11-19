# Metrics

Each task has one **primary metric**, the value plotted in the RD curves and used for BD-rate. Other values are
stored next to it in the score files and the results JSON. Where a config option selects between variants, both
variants are always computed and stored; the option only picks which one is plotted. Every option below can be set
globally in the config and per sequence (`sequence_overrides.<seq>`).

| Task | Primary metric | Variants (config option) | Score file |
|---|---|---|---|
| Object detection | COCO bbox AP@[.50:.95] | `ap_average: class \| object` | `QPxx.json` |
| Instance segmentation | COCO mask AP@[.50:.95] | `ap_average: class \| object` | `QPxx_instance_segmentation.json` |
| Tracking | HOTA | `hota_average: class \| object` | `QPxx_tracking.json` |
| Keypoints | COCO keypoint AP@[.50:.95] (OKS) | – | `QPxx_keypoints.json` |
| Panoptic segmentation | PQ | `pq_average: all \| gt` | `QPxx_panoptic.json` |
| Semantic segmentation | mIoU | – | `QPxx_panoptic.json` (shared) |
| Depth | AbsRel (lower is better) | `depth_eval: raw \| aligned`, `depth_min_mm` | `QPxx_depth.json` |
| Oriented object detection | rotated-box AP@[.50:.95] | `obb_score: dota \| vehicle`, `obb_classes` | `QPxx_oriented_detection.json` |
| Licence plate recognition | plate accuracy (end to end) | `lpr_iou` | `QPxx_plate_recognition.json` |
| Race bib recognition | bib accuracy (end to end) | `lpr_iou` | `QPxx_bib_recognition.json` |

Score files are in `data/<seq>/results/scores/<model_id>/`. The per-sequence `classes` option applies to detection,
instance segmentation, tracking, panoptic and semantic segmentation ([Class selection](#class-selection)).

## Object detection

- **Metric:** COCO-style bbox AP averaged over IoU 0.50:0.95 (pycocotools default).
- **Ground truth:** `iscrowd` objects are ignore regions. A ground-truth object's area is its box area at the decoded
  resolution. All 80 COCO classes are passed to pycocotools; the average covers the classes present.
- **Averaging over classes** (`ap_average`):
  - `class` (default): COCO standard, AP per class, then the mean over the classes.
  - `object`: object-weighted. Matching stays class-aware (a detection matches only ground truth of its own class),
    but the detections of all classes are pooled into one precision–recall curve, so every ground-truth object
    weighs equally and a class with one or two objects cannot dominate. As in the class average, only classes with
    ground truth enter the curve; detections of other classes are ignored.
  - With a single class both are identical.

## Instance segmentation

- **Metric:** COCO-style mask AP@[.50:.95], with the same `ap_average` option as detection. The bbox AP of the same
  predictions is stored too.
- **Ground truth:** masks from `<sequence_name>_masks/`, downscaled to the decoded size with nearest neighbour. An
  object's mask is the set of pixels with exactly its `mask_color`; other colours are background. Polygons
  (normalised like the boxes) are used only when there is no masks folder. An object without a mask is an ignore
  region.

## Tracking

- **Metric:** HOTA (primary), MOTA / MOTP and IDF1 with [TrackEval](https://github.com/kovalp/TrackEval), per class
  and averaged over the classes.
- **Averaging over classes** (`hota_average`):
  - `class` (default): class average (TrackEval `combine_classes_class_averaged`).
  - `object`: detection-weighted (TrackEval `combine_classes_det_averaged`). The classes' TP / FN / FP and
    association counts are summed, then HOTA, MOTA and IDF1 are computed once, so each class weighs by its
    detections; matching stays per class. FloorballGame uses it so that the single ball does not count as half.
- **Ground truth:** uses `track_id`. Crowd objects and objects without `track_id` are distractors (MOTChallenge
  procedure: tracker boxes matched to them at IoU ≥ 0.5 are removed). A file with no `track_id` at all skips the task.
- **Procedure:** frames are tracked in order, with a fresh tracker for every QP. A duplicate `track_id` in a frame
  keeps the first box (with a warning) when the boxes overlap; otherwise it is an error.

## Keypoints

- **Metric:** COCO keypoint AP@[.50:.95] (pycocotools OKS, standard COCO sigmas, maxDets 20, area ranges medium /
  large).
- **Ground truth:** `<sequence_name>_pose_annotations.json`: 17 keypoints (x, y, v) and the person's segmentation
  `area` as a fraction of the image area (scaled to the decoded frame: `area` × W × H). Only labelled keypoints
  (v > 0) count. Persons without labelled keypoints and `iscrowd` persons are ignore regions.
- `classes` does not apply (persons only).

## Panoptic segmentation

- **Metric:** PQ, with SQ and RQ, overall and for things and stuff separately (COCO panopticapi algorithm).
- **Ground truth:** `<sequence_name>_panoptic_annotations.json` + `<sequence_name>_panoptic_masks/`, nearest-neighbour
  downscale. Colours without a segment are void and ignored.
- **Matching:** segments match when they have the same class and IoU > 0.5 (void pixels of the prediction removed
  from the union). TP / FP / FN / IoU are summed over all frames per class. An unmatched prediction lying > 50 % on
  void or on the crowd region of its class is not a false positive.
- **Averaging over classes** (`pq_average`):
  - `all` (default, panopticapi standard): classes with TP + FP + FN > 0. A class that is only predicted, never in
    the ground truth, enters the average with PQ 0, so the number of averaged classes can change from QP to QP.
  - `gt`: only the classes with at least one ground-truth segment in the evaluated frames (with `classes`: the
    listed things + stuff). False positives of predicted-only classes still cost TP / IoU where they cover
    ground-truth segments, but those classes are not averaged in. HighwayView and HighwayDrive use it.

## Semantic segmentation

- **Metric:** mIoU. Per-class IoU accumulated over all frames on non-void ground-truth pixels, then the mean over the
  classes that have ground-truth pixels. Pixel accuracy is stored too.
- **Predictions:** derived from the panoptic predictions (segments merged by class), so one inference and one score
  file (`QPxx_panoptic.json`) serve both tasks.

## Depth

- **Metric:** AbsRel = mean |d − g| / g (primary, lower is better). Also SqRel, RMSE [m], RMSE_log, log10 and
  δ1 / δ2 / δ3 (share of pixels with max(d/g, g/d) < 1.25, 1.25², 1.25³).
- **Procedure:** computed per frame over the valid ground-truth pixels, then averaged over the frames. Valid = depth
  > 0 and ≥ `depth_min_mm`.
- **Ground truth:** `<sequence_name>_depth/depth_NNNN.png`, 16-bit, millimetres, nearest-neighbour resized to the
  decoded frame size. Metric predictions are in mm (≥ 1 mm).
- **Variants:**
  - `stats` (metric models): the predicted depth as is.
  - `stats_median_scaled` (metric models): prediction × median(GT) / median(prediction) per frame.
  - `stats_aligned` (all models): per-frame least-squares scale and shift on the valid pixels, in depth space for
    metric models and in inverse-depth space (fit to 1/GT) for relative models (affine-invariant inverse depth,
    MiDaS protocol); aligned depth clamped to [1 mm, 100 m].
- **Plotted** (`depth_eval`): `raw` (default) plots `stats` (relative models have no point); `aligned` plots
  `stats_aligned`.

## Oriented object detection

- **Metric:** rotated-box AP@[.50:.95]: pycocotools COCOeval with the IoU replaced by the exact IoU of the two
  rotated boxes (convex quadrilaterals; crowd ground truth: intersection / detection area). Everything else is as
  COCO (101-point AP, class average over classes with ground truth, small / medium / large by polygon area), except
  maxDets [1, 10, 300] (frames with ~100 objects; Ultralytics OBB max_det 300).
- **Also stored:** object-weighted AP, AR, and the DOTA task1 mAP@0.5 (DOTA_devkit: VOC07 11-point AP per class,
  mean over classes with ground truth).
- **Ground truth:** `<sequence_name>_obb_annotations.json`, 4 corners, DOTA v1.0 0-based class ids. Polygons are used
  as given (corners outside the frame are not clipped).
- **Options:**
  - `obb_classes`: DOTA class names scored on that sequence.
  - `obb_score`: `dota` (default) plots `stats`. `vehicle` merges large vehicle and small vehicle into one class
    "vehicle" (id 10) in ground truth and predictions, evaluated the same way (`stats_vehicle`, full result under
    "vehicle"); `obb_classes` containing either keeps the merged class. SyntheticAerial uses `vehicle`.

## Licence plate recognition

- **Metric:** plate accuracy (end to end) = ground-truth plates found **and** read exactly / all ground-truth plates.
- **Matching:** per frame, predictions in descending detector score each take the unmatched ground-truth plate with
  the highest box IoU, if IoU ≥ `lpr_iou` (one-to-one). Default 0.5 as in UFPR-ALPR; CarPark 0.3 because its ground-
  truth boxes are tighter vertically than the detector's. `lpr_iou` is part of the score cache key.
- **Text:** compared after normalisation (upper case; spaces, hyphens and dots removed).
- **Also stored:** character accuracy = mean over ground-truth plates of 1 − Levenshtein(pred, gt) / max(len)
  (missed = 0), accuracy when detected, detection recall, false plates (unmatched predictions), plate-box AP / AP50 /
  AP75 (pycocotools, one class), plate accuracy per track.

## Race bib recognition

- **Metric:** bib accuracy = the plate metric above on `<sequence_name>_bib_annotations.json` ("bib" key, box IoU ≥
  `lpr_iou`).
- **Difference:** predicted texts are reduced to their digits first, and predictions without digits are dropped
  (scene-text OCR also reads sponsor text).
- **Stats:** `bib_accuracy`, `char_accuracy`, `accuracy_detected`, `detection_recall`, `false_bibs`, `det_AP*`.

## Object size ranges (APs / APm / APl)

Detection, instance segmentation, oriented detection and keypoints also report AP per object size, with the standard
COCO area ranges (pycocotools default `areaRng`, not changed here):

| Range | Area (px²) | e.g. a square of |
|---|---|---|
| small (APs, ARs) | < 32² = 1,024 | up to 32 × 32 |
| medium (APm, ARm) | 32² to 96² = 1,024 to 9,216 | 32 × 32 to 96 × 96 |
| large (APl, ARl) | > 96² = 9,216 | above 96 × 96 |

- The area is in pixels at the decoded frame size: the ground-truth box area (detection), the mask pixel count
  (instance segmentation), the rotated polygon area (oriented detection), or the annotated person area × W × H
  (keypoints; COCO keypoints has only medium and large).
- At 1920×1080, small is below about 0.05 % of the frame area and large above about 0.44 %.
- The bounds are inclusive in pycocotools (an object of exactly 1,024 or 9,216 px² counts in both ranges).
- A range without ground truth gives −1 in pycocotools, stored as `null` and left out of the extended analysis.
- Who computes them: in `stats` (class average) pycocotools `COCOeval.summarize()` (oriented detection: from
  `COCOeval`'s accumulated results, with maxDets 300); in `stats_object_weighted` vcmtest, from pycocotools'
  per-image matches with the same area ranges, pooling all classes into one curve.
- Because the ranges are absolute, the same object can change range when the frame size changes (half the
  resolution = a quarter of the area). Size sub-metrics of encodings with different frame sizes are not directly
  comparable ([BD_RATE.md](BD_RATE.md#comparability-checks)).

## Class selection

`sequence_overrides.<seq>.classes`:
- Detection, instance segmentation and tracking score only the listed classes; ground truth and predictions of
  other classes are left out, as if only these categories existed.
- Panoptic and semantic segmentation: only the listed thing classes plus all stuff classes enter the PQ / mIoU
  averages (per-class values unchanged; pixel accuracy covers all classes).
- A mapping `{<task>: [names]}` restricts only the tasks it names.

## BD-rate and BD-accuracy

Test encoding against an anchor encoding of the same source, per task and model (`vcmtest/bd.py`).

- **Curve:** the plotted metric (%) against bitrate, for one model.
- **Pareto front:** each curve is reduced to its Pareto front first. A point is removed when another point of the
  curve has bitrate ≤ and accuracy ≥ (≤ for lower-is-better metrics), at least one strictly. The removed QPs are
  recorded and drawn hollow.
- **Interpolation:** the Pareto points are interpolated with PCHIP (piecewise cubic Hermite; scipy
  `PchipInterpolator`, identical to the `bjontegaard` package's pchip method) and integrated exactly over the
  overlapping interval.
- **BD-rate (%)** = 100 × (10^D − 1), where D = mean of log10(bitrate_test) − log10(bitrate_anchor) as functions of
  accuracy, over the overlapping accuracy range. Negative = the test needs less bitrate at equal accuracy (for any
  metric direction).
- **BD-accuracy (pp)** = mean of accuracy_test − accuracy_anchor as functions of log10 bitrate, over the overlapping
  log-bitrate range. Positive = higher value (better for accuracies, worse for lower-is-better metrics).
- **n/a** when a curve has fewer than `bd.min_points` (default 4) Pareto points, or the ranges do not overlap.
- **Skipped:** a task when the two results differ in metric key, metric options, classes, `lpr_iou`,
  `depth_min_mm`, annotation version / fingerprint or bitrate unit; a model when its weights or frame counts differ.
- **Mean:** per task, the arithmetic mean over the models with a value.

## Extended BD analysis

`bd.extended` (default true; `vcmtest/bd_extended.py`). Writes `extended.json` and `<test>__vs__<anchor>_extended.md`
next to `bd.json`, from the same results (no rescoring).

1. **Breakdown:** BD-rate / BD-accuracy, exactly as above (Pareto, PCHIP, overlap, `bd.min_points`), of
   sub-metrics. These are the components of the plotted stats field (the field the metric option selects, e.g.
   `stats_object_weighted`):
   - detection / instance segmentation / oriented: AP50, AP75, APs, APm, APl ([size ranges](#object-size-ranges-aps--apm--apl);
     COCO −1 = no ground truth in that size range: left out); the extended Markdown notes the ranges above the table;
   - keypoints: AP50, AP75, APm, APl;
   - tracking: DetA, AssA, LocA, MOTA, IDF1;
   - panoptic: SQ, RQ, PQ_th, PQ_st; semantic: pixel_accuracy; depth: delta1;
   - plate / bib: detection_recall, accuracy_detected, char_accuracy, det_AP.

   And per class: per-class AP (detection, instance segmentation, oriented), per-class value of the plotted key
   (tracking), per-class PQ of the classes with ground truth (panoptic; things limited to `classes`), per-class IoU
   (semantic). Per sub-metric: per-model values, the mean over the models with a value, and the mean of the models'
   best anchor value (a BD of a sub-metric near 0 on the anchor is not meaningful).
2. **Model agreement** per task (from `bd.json`):
   - BD-rate and BD-accuracy over the models: n, sign counts, mean, median, range, standard deviation.
   - Verdict: consistent (all the same sign) / mostly (≥ 75 %) / mixed / single model; flagged when the standard
     deviation exceeds |mean|.
   - QP steps per side (points by decreasing bitrate): mean and worst change over the models (pp, positive = better
     at the lower bitrate); mean change per halving of the bitrate (change / log2(rate ratio)); the models with an
     inversion (better at the lower bitrate: one of more than two models = model noise, all = content or codec
     effect); and the models whose largest drop per halving is at that step (all at one step = the encoding
     collapses there).
