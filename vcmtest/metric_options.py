"""Registry of plotted metrics: each task's default metric and the config options that select another variant.

Everything about *which value is plotted* for a task lives here. vcmtest.config derives the defaults, the
per-sequence override keys and the validation from it, and vcmtest.results derives primary_metric(), the
recorded option values and the result fields copied into each RD point. Adding a selectable metric variant =
compute its result field in vcmtest/score.py + one entry here (see docs/ADDING_METRICS.md).

This module must stay light (imported by vcmtest.config): no scoring libraries.

TASK_METRICS[task] = TaskMetric(key, label, lower_is_better): the default plotted value is
    scores result["stats"][key]; values are fractions in [0, 1] (shown x 100).
METRIC_OPTIONS[name] = MetricOption: a config key (global, and per sequence in sequence_overrides.<seq>) with the
    tasks it applies to, its default value and one Variant per allowed value. A Variant names the result field
    holding its stats ("stats" = the default) and may change the key and the axis label. For a task with
    several options (depth: depth_eval + depth_metric) they combine in registry order: the stats field of the
    first variant that names one, the key of the last variant that sets one, and the labels applied in turn (a
    callable label maps the label so far).
DERIVED_KEYS[key] = f(stats) -> value: plotted values computed from a stats dict instead of stored in it (e.g.
    one_minus_AbsRel); stat_value() reads either kind, so no rescoring is needed to plot them.
RESOLVED_FIELDS[field] = f(result or point) -> real result field: a per-model stats field (e.g. stats_median_or_aligned
    = stats_median_scaled for metric depth models, stats_aligned for relative ones); resolve_stats_field() maps it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskMetric:
    """Default plotted metric of a task: stats key, axis label and direction."""
    key: str
    label: str
    lower_is_better: bool = False


@dataclass(frozen=True)
class Variant:
    """One allowed value of a metric option: the result field holding its stats and optional key / label changes."""
    stats: str = "stats"                       # result field with the stats of this variant
    key: str | None = None                     # None = the task's default key
    label: str | Callable | None = None        # None = the task's label; str; or f(task label) -> str
    lower_is_better: bool | None = None        # None = as the task's metric (only read together with key)


@dataclass(frozen=True)
class MetricOption:
    """A config option selecting the plotted metric variant for its tasks."""
    tasks: tuple
    default: str
    variants: dict = field(default_factory=dict)
    doc: str = ""


TASK_METRICS: dict[str, TaskMetric] = {
    "detection": TaskMetric("AP", "COCO bbox AP@[.50:.95]"),
    "instance_segmentation": TaskMetric("AP", "COCO mask AP@[.50:.95]"),
    "tracking": TaskMetric("HOTA", "HOTA (class average)"),
    "panoptic_segmentation": TaskMetric("PQ", "PQ (COCO panoptic, class average)"),
    "semantic_segmentation": TaskMetric("mIoU", "mIoU (classes in the ground truth)"),
    "keypoints": TaskMetric("AP", "COCO keypoint AP@[.50:.95] (OKS)"),
    "depth": TaskMetric("AbsRel", "depth AbsRel, lower is better", lower_is_better=True),
    "oriented_detection": TaskMetric("AP", "rotated-box AP@[.50:.95] (polygon IoU)"),
    "plate_recognition": TaskMetric("plate_accuracy", "end-to-end plate accuracy (IoU >= lpr_iou, exact text)"),
    "bib_recognition": TaskMetric("bib_accuracy", "end-to-end race bib accuracy (IoU >= lpr_iou, exact digits)"),
}

METRIC_OPTIONS: dict[str, MetricOption] = {
    "ap_average": MetricOption(
        ("detection", "instance_segmentation"), "class",
        {"class": Variant(),
         "object": Variant("stats_object_weighted",
                           label=lambda task_label: task_label.replace("COCO ", "") + ", object-weighted")},
        "detection / instance seg AP: class (COCO mean of per-class AP) | object (object-weighted)"),
    "hota_average": MetricOption(
        ("tracking",), "class",
        {"class": Variant(),
         "object": Variant("stats_object_weighted", label="HOTA (object-weighted)")},
        "tracking HOTA / MOTA / IDF1: class (class average) | object (TrackEval detection-averaged over the classes)"),
    "pq_average": MetricOption(
        ("panoptic_segmentation",), "all",
        {"all": Variant(),
         "gt": Variant("stats_gt_classes", label="PQ (COCO panoptic, average over the GT classes)")},
        "panoptic PQ: all (panopticapi, classes with TP+FP+FN > 0) | gt (classes in the ground truth)"),
    "depth_eval": MetricOption(
        ("depth",), "raw",
        {"raw": Variant(),
         "aligned": Variant("stats_aligned", label="depth AbsRel, per-frame scale+shift aligned, lower is better"),
         "median_scaled": Variant("stats_median_or_aligned",
                                  label="depth AbsRel, median-scaled (relative: aligned), lower is better")},
        "depth: raw (metric models as is) | aligned (per-frame scale+shift fit, all models) | median_scaled "
        "(metric models: per-frame median scaling; relative models: scale+shift fit, as aligned)"),
    "depth_metric": MetricOption(
        ("depth",), "abs_rel",
        {"abs_rel": Variant(),
         "one_minus_abs_rel": Variant(key="one_minus_AbsRel", lower_is_better=False,
                                      label=lambda lab: lab.replace("depth AbsRel", "depth 1 - AbsRel")
                                      .replace("lower is better", "higher is better"))},
        "depth plotted value: abs_rel (AbsRel, lower is better) | one_minus_abs_rel (1 - AbsRel, higher is better; "
        "same BD-rate, BD-accuracy sign as the other tasks; can be negative when AbsRel > 1)"),
    "obb_score": MetricOption(
        ("oriented_detection",), "dota",
        {"dota": Variant(),
         "vehicle": Variant("stats_vehicle", label="rotated-box AP@[.50:.95], vehicle (small + large merged)")},
        "oriented detection: dota (DOTA classes) | vehicle (small + large vehicle merged; computed when selected)"),
}

# plotted values derived from a stats dict (not stored in the scores files)
DERIVED_KEYS: dict[str, Callable] = {
    "one_minus_AbsRel": lambda st: None if st.get("AbsRel") is None else 1.0 - st["AbsRel"],
}

# per-model stats fields: virtual field -> f(result or RD point) -> the real result field
RESOLVED_FIELDS: dict[str, Callable] = {
    "stats_median_or_aligned": lambda r: ("stats_median_scaled" if r.get("depth_output", "metric") == "metric"
                                          else "stats_aligned"),
}

# result fields copied into each RD point besides the variant fields (depth details)
EXTRA_POINT_FIELDS = ("stats_median_scaled", "scale", "valid_pixel_share", "depth_output")


def option_value(cfg: dict, seq: str | None, name: str) -> str:
    """The option's value for a sequence: sequence_overrides.<seq>.<name>, else the global value, else default."""
    ov = ((cfg.get("sequence_overrides") or {}).get(seq) or {}) if seq is not None else {}
    return ov.get(name, cfg.get(name, METRIC_OPTIONS[name].default))


def options_for(task: str) -> list[str]:
    """Names of the metric options that apply to a task, in registry order."""
    return [n for n, o in METRIC_OPTIONS.items() if task in o.tasks]


def plotted_metric(cfg: dict, seq: str, task: str) -> tuple[str, str, str]:
    """(key in the stats field, axis label, result field) of the plotted metric for a task in a sequence.
    The task's options combine in registry order (module docstring)."""
    tm = TASK_METRICS[task]
    key, label, stats = tm.key, tm.label, "stats"
    for name in options_for(task):
        v = METRIC_OPTIONS[name].variants[option_value(cfg, seq, name)]
        if v.stats != "stats" and stats == "stats":
            stats = v.stats
        if v.key:
            key = v.key
        if v.label:
            label = v.label(label) if callable(v.label) else v.label
    return key, label, stats


def stat_value(stats: dict, key: str):
    """The plotted value `key` of a stats dict: stored, or derived (DERIVED_KEYS); None if missing."""
    if key in DERIVED_KEYS:
        return DERIVED_KEYS[key](stats)
    return stats.get(key)


def resolve_stats_field(field: str, r: dict) -> str:
    """The real result field of a stats field for one result or RD point (RESOLVED_FIELDS), else field itself."""
    return RESOLVED_FIELDS[field](r) if field in RESOLVED_FIELDS else field


def lower_is_better_keys() -> set[str]:
    """Metric keys where a smaller value is better (task defaults and variant keys)."""
    keys = {tm.key for tm in TASK_METRICS.values() if tm.lower_is_better}
    for o in METRIC_OPTIONS.values():
        for v in o.variants.values():
            if v.key and v.lower_is_better:
                keys.add(v.key)
    return keys


def point_fields() -> tuple[str, ...]:
    """Result fields copied into each RD point: every variant's stats field (except "stats") + the extras."""
    fields = []
    for o in METRIC_OPTIONS.values():
        for v in o.variants.values():
            if v.stats != "stats" and v.stats not in fields and v.stats not in RESOLVED_FIELDS:
                fields.append(v.stats)
    return tuple(fields) + EXTRA_POINT_FIELDS
