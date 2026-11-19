# Adding a metric

How to add an evaluation metric to a task that already exists, with a tested example: **detection F1@0.5 at the
best score threshold** (`docs/examples/det_f1.py`).

There are two levels:

1. **Store an extra metric** next to the existing ones, in the scores files (sections 1–2): a metric function and
   two lines in `vcmtest/score.py`.
2. **Make it selectable as the plotted metric**, per run or per sequence, like `ap_average`, `pq_average` or
   `depth_eval` (section 3): one entry in the registry `vcmtest/metric_options.py`.

A new *task* (new annotations, a new prediction format) is a larger change; section 5 lists the parts involved.

---

## 1. How scoring works

```
predictions/<model_id>/QPxx.json ──score_<task>()──▶ scores/<model_id>/QPxx[_<task>].json ──▶ <seq>_results.json ──▶ RD plot
     (vcmtest/infer.py)             (vcmtest/score.py,      {"key": {...}, "result": {...}}     (vcmtest/results.py)
                                     vcmtest/metrics/*.py)
```

- **`vcmtest/metrics/<name>.py`** holds the metric code. It is a pure function of the annotations and the
  predictions of all evaluated frames; there are no file paths or caching inside. Examples: `coco_det.py`
  (pycocotools AP), `mot.py` (TrackEval HOTA), `depth.py`, `panoptic.py`, `lpr.py`.
- **`vcmtest/score.py`** has one `score_<task>()` per task. It selects the frames and classes, builds the cache
  **`key`**, calls the metric functions and writes the scores file. The result is a dict whose **`stats*`
  fields** hold the metric values: `stats` is the main one, and variants include `stats_object_weighted`,
  `stats_gt_classes` and `stats_aligned`.
- **Caching:** the scores file is reused when its `key` is unchanged. The key holds the hashes of the predictions,
  annotations and frame set, the score version and the settings that change the values. **Anything new that
  changes the result must change the key**, or old cached scores are reused.
- **`vcmtest/results.py`** builds one RD point per model and QP from the scores files. Which
  `(key, label, stats field)` is plotted for a task in a sequence comes from the registry
  **`vcmtest/metric_options.py`** (section 3).
- **Values are fractions in [0, 1].** Plots and logs show them × 100.

---

## 2. Store an extra metric (example: F1@0.5)

### Step 1: the metric function, `vcmtest/metrics/det_f1.py`
The full file is in [examples/det_f1.py](examples/det_f1.py). Its interface is the same as `coco_det.evaluate`:

```python
def evaluate_f1(ann, pred_frames: dict, frames: list[int], width: int, height: int) -> dict:
    """ann.frames[fn] = GT objects (normalised x_min, x_max, y_min, y_max, class_id, iscrowd);
    pred_frames[str(fn)] = [[class_id, score, x1, x2, y1, y2], ...] (normalised); returns values in [0, 1]."""
    ...
    return {"F1": ..., "precision": ..., "recall": ..., "score_threshold": ..., "tp": ..., "fp": ..., "fn": ...}
```

Rules:
- **Follow the conventions of the task's existing metric:**
  - crowd regions (`iscrowd = 1`) are ignored;
  - only classes with ground truth count;
  - coordinates are scaled to the decoded frame size (`width`, `height`).
- **Test the function on the ground truth used as predictions:** it must give the perfect score. For the example:
  `F1 = 1.0, tp = 1028, fp = 0, fn = 0` on 100 WinterDrive frames.
- **Use no randomness and no GPU,** so the result depends only on the inputs.

### Step 2: compute it in `vcmtest/score.py` (`score_detection`)
```python
    key = {"version": SCORE_VERSION, "task": "detection", ...,
           "pycocotools": metadata.version("pycocotools"), "f1_version": 1}   # new: invalidates old scores
    ...
    result = evaluate(ann_c, pf, frames, w, h)
    from .metrics.det_f1 import evaluate_f1
    result["stats_f1"] = evaluate_f1(ann_c, pf, frames, w, h)                 # new result field
```
- **Pass the same filtered inputs as the existing metric** (here `ann_c` and `pf`, after the per-sequence class
  selection).
- **Add an entry to the key** (`"f1_version": 1`): existing scores files then miss it, and the next run rescores
  this task only. Increase it whenever you change the metric code.
- **Bump the global `SCORE_VERSION` only if you change existing values.** It invalidates the scores of all tasks.
- **Expensive metric:** make it optional with a config setting, as `obb_score: vehicle` computes `stats_vehicle`
  only when selected. Put the setting in the key.

After a run, the value is in `scores/<model_id>/QPxx.json` (`result.stats_f1`). If it is not made selectable
(section 3) and you also want it in every RD point of `<seq>_results.json`, add the field name to
`EXTRA_POINT_FIELDS` in `vcmtest/metric_options.py`.

---

## 3. Make it the plotted metric (selectable): one registry entry

Which value is plotted is defined in **`vcmtest/metric_options.py`**:
- `TASK_METRICS`: each task's default plotted metric (key, axis label, lower is better).
- `METRIC_OPTIONS`: the config options that select another variant. These are the existing `ap_average`,
  `hota_average`, `pq_average`, `depth_eval` and `obb_score`.

### Step 3: add the option to `METRIC_OPTIONS`
```python
METRIC_OPTIONS = {
    "det_metric": MetricOption(
        ("detection",), "ap",                                    # tasks it applies to, default value
        {"ap": Variant(),                                        # default: the task's metric (result["stats"])
         "f1": Variant("stats_f1", key="F1",                     # result field, key in it
                       label="detection F1@0.5 (best score threshold, object-weighted)")},   # axis label
        "detection: ap (COCO AP, averaged as ap_average) | f1 (F1@0.5 at the best score threshold)"),
    "ap_average": ...,
}
```
That is all. With this entry the pipeline:
- **Config:** adds `det_metric: ap` to the defaults, allows it per sequence in `sequence_overrides.<seq>`, and
  rejects values other than `ap` / `f1` in both places.
- **Plotted metric:** plots `result["stats_f1"]["F1"]` with that label when `f1` is selected (`primary_metric()`),
  and uses it in the log table, the sanity check and `summary.json`.
- **Results JSON:** records the value per task (`tasks.detection.det_metric`) and copies the variant's field
  (`stats_f1`) into every RD point.

Notes:
- **Order matters when a task has several options.** The first option, in registry order, whose value selects a
  non-default variant decides. Put `det_metric` before `ap_average`, so `f1` wins and `ap_average` applies when
  `det_metric: ap`.
- **Lower-is-better metrics** (errors, like depth AbsRel): set `lower_is_better=True` on the variant (or on the
  `TaskMetric`). The sanity check then expects the value to *rise* with QP.
- **`label`** may also be a function of the task's default label, e.g. `ap_average: object`:
  `lambda l: l.replace("COCO ", "") + ", object-weighted"`.
- **Extra result fields** that are not a variant's stats (e.g. depth's `scale`) are copied into the points when
  listed in `EXTRA_POINT_FIELDS`.
- **Documentation:** in `configs/example_config.yaml`, add the option with a comment (it works without).

### Step 4: use it
```yaml
det_metric: f1                       # all sequences
sequence_overrides:
  WinterDrive_1920x1080_444_10b_VTM_RA:
    det_metric: f1                   # or only this sequence
```
Result of the example on WinterDrive (car, truck), F1 in %:

| Model | QP29 (2044 kbps) | QP47 (107 kbps) |
|---|---|---|
| RF-DETR-M 1152 | 84.0 | 60.4 |
| Faster R-CNN v2 | 79.6 | 53.7 |

The best score thresholds were 0.47–0.52 at QP29 and 0.33–0.34 at QP47.

---

## 4. Checks and documentation

1. **Unit test** (pytest), with no data needed: the ground truth used as predictions gives the perfect score; a
   few hand-made cases (one missed object, one false positive) give the expected values.
2. **A run on two QPs:**
   - `<seq>_results.json` → `tasks.<task>.metric` / `metric_key` show the new metric;
   - the RD plot's y axis is labelled with it;
   - no `SANITY` warnings that the existing metric does not also show.
3. **Re-run without changes:** the scores must be cached (`scores cached` in the log). If they are recomputed every
   time, something in the key changes between runs (e.g. a float or a dict order).
4. **Documentation:**
   - `docs/METRICS.md`: definition, matching rules, averaging;
   - README: the config option;
   - the comment of the option in `configs/example_config.yaml` (and in your `configs/config.yaml`).

---

## 5. A new task (outline)

A new task needs more than a metric:
- **Annotations:** a file type in `vcmtest/annotations.py`, with discovery in `paths.py`.
- **The task name:** in `config.py` (`IMPLEMENTED_TASKS`, its annotation type).
- **Wrappers that output its prediction format** (`docs/ADDING_MODELS.md`), with their checks in `run_inference`.
- **Scoring:** a `score_<task>()` and a scores file name in `score.py`, the call in `pipeline.py`, and
  its `TASK_METRICS` entry in `vcmtest/metric_options.py`.

Use a recent task as the template; `bib_recognition` reuses the plate metric with another annotation file.
