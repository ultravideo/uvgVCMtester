"""Stage 4: per-sequence results JSON + RD-curve PNGs, and results/summary.json.

data/<seq>/results/<seq>_results.json   bitrates, accuracies, models, versions, warnings
data/<seq>/results/<seq>_<task>_rd.png  rate-accuracy curve(s)
<results_root>/summary.json             all sequences' results collected into one file

The per-sequence file includes every valid, up-to-date data point on disk for the configured models
(all QP folders, not only the QPs of the latest run). Points whose predictions or scores no longer
match the current config or annotations are listed under 'not_plotted' with the reason.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from . import metric_options as _mo
from .bitrate import bitrate_info
from .config import model_id, project_relative, resolve_path, sequence_depth_min_mm, sequence_lpr_iou, task_class_ids
from .infer import predictions_path
from .score import PANOPTIC_SCORE_VERSION, SCORE_VERSION, scores_path

log = logging.getLogger("vcmtest")

RESULTS_VERSION = 1
# plotted metrics: registry in vcmtest/metric_options.py (task defaults + options selecting variants)
PRIMARY_METRIC = {t: (m.key, m.label) for t, m in _mo.TASK_METRICS.items()}   # task default (key, label); [0, 1]
LOWER_IS_BETTER = _mo.lower_is_better_keys()   # metric keys where a smaller value is better (sanity check direction)


# sequence_overrides.<seq>.classes applies to config.CLASS_TASKS only (keypoints: persons only; oriented_detection:
# DOTA classes, selected with obb_classes instead); in these two tasks it selects the thing classes only
PANOPTIC_RESULT_TASKS = {"panoptic_segmentation", "semantic_segmentation"}


def results_json_path(sp) -> Path:
    """Path of the sequence's <seq>_results.json."""
    return sp.results_dir / f"{sp.name}_results.json"


def _read_json(p: Path) -> dict | None:
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _all_qp_items(sp, allowed=None) -> list:
    """QPs of this run plus every other QPxx folder of the sequence (for points from earlier runs).

    allowed: sequence_overrides.<seq>.qps list (config.sequence_result_qps) = only these QPs; None = all."""
    from .paths import DiscoveryError, qp_dirs, scan_qp_dir
    items = {q.qp: q for q in sp.qps}
    for qp, d in qp_dirs(sp.dir).items():
        if qp not in items:
            try:
                items[qp] = scan_qp_dir(qp, d)
            except DiscoveryError:
                continue
    return [items[k] for k in sorted(items) if allowed is None or k in allowed]


def _disk_decode_info(q) -> dict:
    """Decode info of a QP not processed in this run (from its decode marker), else {}."""
    if q.bitstream is None:
        return {"source": "provided_pngs", "frames": len(q.pngs) or None}
    m = _read_json(q.dir / f"{q.bitstream.stem}.decode.json")
    if m:
        return m
    if q.pngs:
        return {"source": "provided_pngs", "frames": len(q.pngs)}
    return {}


def _frame_size(qp_dir: Path, dec: dict) -> list[int] | None:
    """[width, height] of the frames the task networks saw: from a frame PNG in the QP folder, else from the decoded
    stream (frames removed), else None."""
    from .paths import PNG_RE
    from .score import image_size
    png = next((f for f in qp_dir.glob("*.png") if PNG_RE.match(f.name)), None) if qp_dir.is_dir() else None
    if png is not None:
        try:
            return list(image_size(png))
        except OSError:
            pass
    st = dec.get("stream") or {}
    return [st["width"], st["height"]] if st.get("width") and st.get("height") else None


def _expected_decode_settings(dec: dict, cfg: dict):
    from .decode import decode_settings
    return None if dec.get("source") == "provided_pngs" else decode_settings(cfg["decode"])


def primary_metric(cfg: dict, seq: str, task: str) -> tuple[str, str, str]:
    """(stat key, label, result field holding the stats) of the plotted metric for a task in a sequence."""
    return _mo.plotted_metric(cfg, seq, task)


def _point(sp, q, m, mid, dec, cfg, ann_sha, qp_entry, metric_key, run_status, task="detection",
           stats_field="stats") -> dict:
    """One RD point, or {"qp", "error"} when no valid, up-to-date score exists."""
    from .annotations import file_sha256
    from .infer import cached_predictions
    pred_p, sc_p = predictions_path(sp.results_dir, mid, q.qp), scores_path(sp.results_dir, mid, q.qp, task)
    if run_status is not None and run_status != "ok":
        return {"qp": q.qp, "error": run_status}
    sc = _read_json(sc_p)
    if not sc or not pred_p.is_file():
        return {"qp": q.qp, "error": "not run"}
    n_pred = (_read_json(pred_p) or {}).get("meta", {}).get("frames", -1)
    if cached_predictions(pred_p, m, _expected_decode_settings(dec, cfg), n_pred, sp.fps,
                          require_depth_maps=False) is None:   # the depth maps may be removed (scored already)
        return {"qp": q.qp, "error": "stale: predictions do not match current model/threshold/decode settings"}
    key = sc.get("key", {})
    if key.get("version") != SCORE_VERSION:
        return {"qp": q.qp, "error": "stale: scores from an older scoring version (rerun this QP)"}
    if task == "depth" and key.get("depth_min_mm") != sequence_depth_min_mm(cfg, sp.name):
        return {"qp": q.qp, "error": "stale: scores were computed with another depth_min_mm (rerun this QP)"}
    if task in ("plate_recognition", "bib_recognition") and key.get("lpr_iou") != sequence_lpr_iou(cfg, sp.name):
        return {"qp": q.qp, "error": "stale: scores were computed with another lpr_iou (rerun this QP)"}
    if task in PANOPTIC_RESULT_TASKS and key.get("panoptic_version") != PANOPTIC_SCORE_VERSION:
        return {"qp": q.qp, "error": "stale: panoptic scores from an older version (rerun this QP)"}
    if key.get("annotations_sha256") != ann_sha or key.get("predictions_sha256") != file_sha256(pred_p):
        return {"qp": q.qp, "error": "stale: scores do not match current annotations/predictions (rerun this QP)"}
    cls = task_class_ids(getattr(sp, "classes", None), task)
    expected = list(cls) if cls is not None else None
    if task == "oriented_detection" and getattr(sp, "obb_classes", None) is not None:
        expected = list(sp.obb_classes)
    if key.get("classes") != expected:
        return {"qp": q.qp, "error": "stale: scores were computed for another class selection (rerun this QP)"}
    if qp_entry.get("bitrate_error") or qp_entry.get("bitrate") is None:
        return {"qp": q.qp, "error": qp_entry.get("bitrate_error") or "bitrate unknown"}
    r = sc["result"]
    if stats_field not in r:
        return {"qp": q.qp, "error": f"stale: scores have no {stats_field} (rerun this QP)"}
    if r[stats_field] is None:
        why = (f"{r.get('depth_output')} depth model: no raw metric result (use depth_eval: aligned)"
               if task == "depth" and r.get("depth_output", "metric") != "metric" else f"no {stats_field} result")
        return {"qp": q.qp, "error": why}
    return {"qp": q.qp, "bitrate": qp_entry["bitrate"], "bitrate_unit": qp_entry.get("bitrate_unit"),
            **({"bitrate_kbps": qp_entry["bitrate_kbps"]} if qp_entry.get("bitrate_kbps") is not None else {}),
            metric_key: r[stats_field][metric_key], "stats": r["stats"],
            **({k: r[k] for k in _mo.point_fields() if k in r}),
            "per_class_AP": r.get("per_class_AP"), "frames": r.get("frames"),
            **({"per_class": r["per_class"]} if "per_class" in r else {}),
            **({k: r[k] for k in ("per_class_PQ", "per_class_IoU") if k in r}),
            "gt_objects": r.get("gt_objects"), "detections": r.get("detections"),
            **({"bbox_stats": r["bbox_stats"]} if "bbox_stats" in r else {}),
            "in_this_run": run_status is not None, "scores_file": str(sc_p)}


def _task_annotations_sha(task: str, inv) -> str:
    """The annotations fingerprint the task's score files are keyed on."""
    from .annotations import file_sha256, load_detection_annotations, load_panoptic_annotations
    if task in ("panoptic_segmentation", "semantic_segmentation"):
        from .metrics.panoptic import gt_fingerprint
        return gt_fingerprint(load_panoptic_annotations(inv.panoptic_json), inv.panoptic_masks_dir)
    if task == "keypoints":
        return file_sha256(inv.pose_json)
    if task == "oriented_detection":
        return file_sha256(inv.obb_json)
    if task == "plate_recognition":
        return file_sha256(inv.lpr_json)
    if task == "bib_recognition":
        return file_sha256(inv.bib_json)
    if task == "depth":
        from .metrics.depth import gt_fingerprint
        from .paths import scan_pngs
        return gt_fingerprint(scan_pngs(inv.depth_dir))
    if task == "instance_segmentation":
        from .metrics.coco_segm import segm_annotations_sha256
        return segm_annotations_sha256(load_detection_annotations(inv.detection_json), inv.masks_dir)
    return file_sha256(inv.detection_json)


def _sequence_bitrate_unit(qps: dict) -> str | None:
    """The bitrate unit of the sequence's curves: the most common unit over the QPs with a bitrate (bitstream "kbps"
    wins ties). QPs with another unit get a bitrate_error (not plotted: one x axis per sequence)."""
    from collections import Counter
    units = Counter(e.get("bitrate_unit") for e in qps.values() if e.get("bitrate") is not None)
    if not units:
        return None
    top = max(units.items(), key=lambda kv: (kv[1], kv[0] == "kbps"))[0]
    for qp, e in qps.items():
        if e.get("bitrate") is not None and e.get("bitrate_unit") != top:
            e["bitrate_error"] = (f"bitrate unit {e.get('bitrate_unit')!r} differs from the sequence's unit {top!r} "
                                  f"(one bitrate axis per sequence)")
            log.warning(f"QP{qp}: {e['bitrate_error']}; not plotted")
    return top


def sanity_check(points: list[dict], key: str) -> dict:
    """Accuracy should not drop when bitrate rises (i.e. should fall as QP rises). For error metrics
    (LOWER_IS_BETTER, e.g. depth AbsRel) the error should not rise with bitrate."""
    pts = [p for p in points if p.get("bitrate") is not None and p.get(key) is not None]
    if key in LOWER_IS_BETTER:
        drops = [f"QP{b['qp']} ({100 * b[key]:.2f}) > QP{a['qp']} ({100 * a[key]:.2f}) error at higher bitrate"
                 for a, b in zip(pts, pts[1:]) if b[key] > a[key]]
    else:
        drops = [f"QP{b['qp']} ({100 * b[key]:.2f}) < QP{a['qp']} ({100 * a[key]:.2f}) at higher bitrate"
                 for a, b in zip(pts, pts[1:]) if b[key] < a[key]]
    qp_order = [p["qp"] for p in pts]
    return {"accuracy_increases_with_bitrate": not drops, "violations": drops,
            "qp_order_matches_bitrate": qp_order == sorted(qp_order, reverse=True)}


def build_sequence_results(sp, cfg: dict, status: dict, env: dict) -> dict:
    """All valid data points on disk for the configured models, over all QP folders of the sequence."""
    from .annotations import annotation_version
    from .config import sequence_result_qps
    from .paths import qp_dirs
    allowed = sequence_result_qps(cfg, sp.name)
    items = _all_qp_items(sp, allowed)
    run_qps = {q.qp for q in items} & {q.qp for q in sp.qps}
    qps_excluded = sorted(set(qp_dirs(sp.dir)) - set(allowed)) if allowed is not None else []
    qps, decs = {}, {}
    for q in items:
        qs = status["qps"].get(q.qp, {})
        dec = (qs.get("decode") if q.qp in run_qps else _disk_decode_info(q)) or {}
        decs[q.qp] = dec
        frames = dec.get("frames")
        if not frames:  # e.g. frames removed with --remove-frames: use the predictions' frame count
            for tp in sp.tasks:
                for m in tp.models:
                    pr = _read_json(predictions_path(sp.results_dir, model_id(m), q.qp))
                    frames = frames or (pr or {}).get("meta", {}).get("frames")
        entry = {"dir": str(q.dir), "in_this_run": q.qp in run_qps}
        entry.update(bitrate_info(q.bitstream, sp.fps, frames, q.log_file, q.dir))
        if q.qp in run_qps:
            for w in entry.get("warnings", []):
                log.warning(f"QP{q.qp}: {w}")
        entry["decode"] = {k: dec.get(k) for k in ("source", "settings", "md5_check", "ffmpeg", "stream")
                           if dec.get(k) is not None}
        size = _frame_size(q.dir, dec)
        if size:
            entry["frame_size"] = size      # [width, height] of the evaluated frames
        if qs.get("error"):
            entry["error"] = qs["error"]
        qps[str(q.qp)] = entry
    bitrate_unit = _sequence_bitrate_unit(qps)
    # "version" of each annotation JSON present; "" when the key or the JSON does not exist
    inv = sp.annotations
    ann_versions = {t: annotation_version(inv.json_for(t)) for t in ("detection", "pose", "panoptic")} if inv else {}
    annotations = None
    if inv:
        annotations = inv.as_dict() | {"versions": {t: v for t, v in ann_versions.items() if inv.json_for(t)}}

    tasks = {}
    for tp in sp.tasks:
        t = {"status": tp.status, "annotation_type": tp.annotation_type,
             "annotation_version": ann_versions.get(tp.annotation_type, "")}
        if tp.reason:
            t["reason"] = tp.reason
        if tp.status != "run":
            tasks[tp.task] = t
            continue
        metric_key, metric_label, stats_field = primary_metric(cfg, sp.name, tp.task)
        t["metric"] = metric_label
        t["metric_key"] = metric_key          # key of the plotted value in each point (may differ per sequence)
        for name in _mo.options_for(tp.task):           # the plotted-metric options' values for this sequence
            t[name] = _mo.option_value(cfg, sp.name, name)
        if tp.task == "oriented_detection" and getattr(sp, "obb_classes", None) is not None:
            from .annotations import dota_label_map
            names = {c: n for n, c in dota_label_map().items()}
            t["classes"] = [names[c] for c in sp.obb_classes]
        if tp.task in ("plate_recognition", "bib_recognition"):
            t["lpr_iou"] = sequence_lpr_iou(cfg, sp.name)
        if tp.task == "depth":
            t["depth_min_mm"] = sequence_depth_min_mm(cfg, sp.name)
        t["metric_scale"] = "fraction in [0, 1]"
        cls = task_class_ids(getattr(sp, "classes", None), tp.task)
        if cls is not None:
            from .score import class_names
            if tp.task in PANOPTIC_RESULT_TASKS:
                t["thing_classes"] = class_names(cls)   # these things + all stuff classes are evaluated
            else:
                t["classes"] = class_names(cls)    # only these classes are evaluated
        ann_sha = _task_annotations_sha(tp.task, sp.annotations)
        t["annotations_sha256"] = ann_sha      # fingerprint the scores are keyed on (BD checks anchor = test)
        models = {}
        for slot, m in enumerate(tp.models):
            mid = model_id(m)
            mres = {"config": m, "slot": slot, "points": [], "not_plotted": []}
            for q in items:
                pr = _read_json(predictions_path(sp.results_dir, mid, q.qp))
                if pr and "model" not in mres:
                    mres["model"] = pr["meta"]["model"]
                run_status = (status["qps"].get(q.qp, {}).get("inference", {}).get(mid, "not run")
                              if q.qp in run_qps else None)
                pt = _point(sp, q, m, mid, decs[q.qp], cfg, ann_sha, qps[str(q.qp)], metric_key, run_status,
                            tp.task, stats_field)
                if "error" not in pt:
                    mres["points"].append(pt)
                elif pt["error"] != "not run" or q.qp in run_qps:
                    mres["not_plotted"].append(pt)
            mres["points"].sort(key=lambda p: (p.get("bitrate") is None, p.get("bitrate") or 0, p["qp"]))
            mres["sanity"] = sanity_check(mres["points"], metric_key)
            models[mid] = mres
        t["models"] = models
        if tp.skipped_models:
            t["models_not_run"] = dict(tp.skipped_models)   # e.g. input_size not implemented for the wrapper
        tasks[tp.task] = t

    return {
        "results_version": RESULTS_VERSION,
        "sequence": sp.name,
        "sequence_name": sp.sequence_name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "framerate": sp.fps,
        "framerate_source": sp.fps_source,
        "annotations": annotations,
        **({"overrides_inherited": cfg["_inherited_overrides"][sp.name]}
           if sp.name in cfg.get("_inherited_overrides", {}) else {}),   # from the BD anchor (configs/anchors.yaml)
        "decode_settings": cfg["decode"] | {},
        "qps": qps,
        "bitrate_unit": bitrate_unit,     # x-axis unit of all curves (bitstream: kbps; QPxx/bitrate.json: as given)
        **({"qps_excluded": {"qps": qps_excluded, "reason": f"not in sequence_overrides.{sp.name}.qps "
                                                         "(files kept on disk)"}} if qps_excluded else {}),
        "tasks": tasks,
        "warnings": sp.warnings,
        "errors": status.get("errors", []),
        "environment": env,
        "config_file": cfg.get("_config_path"),
        "seed": cfg["seed"],
    }


def write_sequence_results(sp, cfg: dict, status: dict, env: dict) -> dict:
    """Build the sequence results and write <seq>_results.json.

    Also draws the combined and per-model RD plots of each task and logs the result table.

    Returns:
        The results dict as written.
    """
    from .plot import plot_rd

    res = build_sequence_results(sp, cfg, status, env)
    for task, t in res["tasks"].items():
        if t.get("status") != "run" or not t.get("models"):
            continue
        key, label = t.get("metric_key", PRIMARY_METRIC[task][0]), t["metric"]
        series = [{"label": mid, "slot": m["slot"],
                   "points": [{"qp": p["qp"], "bitrate": p.get("bitrate"),
                               "value": None if p.get(key) is None else 100 * p[key]} for p in m["points"]]}
                  for mid, m in t["models"].items()]
        title = f"{sp.name} - {task}" + (f" ({', '.join(t['classes'])})" if t.get("classes") else
                                        f" (things: {', '.join(t['thing_classes'])})" if t.get("thing_classes") else "")
        if len(series) == 1:
            title += f"\n{series[0]['label']}"
        png = plot_rd(series, title, f"{label} (%)", sp.results_dir / f"{sp.name}_{task}_rd.png",
                      log_x=cfg["plot"]["log_x"], bitrate_unit=res.get("bitrate_unit"))
        t["rd_plot"] = str(png)
        _write_per_model_plots(sp, task, t, series, f"{label} (%)", cfg["plot"]["log_x"], res.get("bitrate_unit"))
    out = results_json_path(sp)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    tmp.replace(out)
    log.info(f"Results: {out}")
    _log_table(res)
    return res


def per_model_plot_path(sp, task: str, mid: str) -> Path:
    """Path of one model's RD plot: results/rd_per_model/<seq>_<task>_<model_id>_rd.png."""
    return sp.results_dir / "rd_per_model" / f"{sp.name}_{task}_{mid}_rd.png"


def _write_per_model_plots(sp, task: str, t: dict, series: list[dict], ylabel: str, log_x: bool,
                           bitrate_unit: str | None = "kbps") -> None:
    """One RD plot per model with valid points (same colour/marker as in the combined plot), written or overwritten.
    Plots are never deleted: those of models no longer configured (or without valid points now) stay on disk as
    they were last drawn; the results file references only the current ones (rd_plot)."""
    from .plot import plot_rd
    keep = set()
    for s in series:
        m = t["models"][s["label"]]
        if not any(p.get("value") is not None for p in s["points"]):
            continue
        png = plot_rd([s], f"{sp.name} - {task}\n{s['label']}", ylabel,
                      per_model_plot_path(sp, task, s["label"]), log_x=log_x, bitrate_unit=bitrate_unit)
        m["rd_plot"] = str(png)
        keep.add(png.name)


def _log_table(res: dict) -> None:
    for task, t in res["tasks"].items():
        if t.get("status") != "run":
            log.info(f"  {task}: {t['status']} ({t.get('reason', '')})")
            continue
        key = t.get("metric_key", PRIMARY_METRIC[task][0])
        for mid, m in t["models"].items():
            log.info(f"  {task}/{mid}: {len(m['points'])} point(s)")
            for v in m.get("sanity", {}).get("violations", []):
                log.warning(f"    SANITY: accuracy drops with more bitrate: {v}")
            if not m.get("sanity", {}).get("qp_order_matches_bitrate", True):
                log.warning("    SANITY: bitrate does not decrease monotonically with QP")
            for p in m["points"]:
                br = "n/a" if p.get("bitrate") is None else f"{p['bitrate']:9.5g} {p.get('bitrate_unit') or ''}"
                log.info(f"    QP{p['qp']:<3} {br}  {key}={100 * p[key]:6.2f}"
                         + ("" if p.get("in_this_run") else "  (earlier run)"))
            for p in m.get("not_plotted", []):
                log.warning(f"    QP{p['qp']:<3} not plotted: {p['error']}")


def _summary_data_root(path: Path) -> str | None:
    """Data root an existing summary.json was collected from (recorded, or derived from its first results file for
    summaries written before it was recorded); None if unknown."""
    old = _read_json(path) if path.is_file() else None
    if not old:
        return None
    if old.get("data_root"):
        return old["data_root"]
    files = [s.get("results_file") for s in (old.get("sequences") or {}).values() if s.get("results_file")]
    return project_relative(Path(files[0]).parents[2]) if files else None


def write_summary(cfg: dict) -> Path:
    """Collect all data/<seq>/results/<seq>_results.json into <results_root>/summary.json."""
    data_root = resolve_path(cfg["data_root"])
    seqs = {}
    for p in sorted(data_root.glob("*/results/*_results.json")):
        r = _read_json(p)
        if not r or r.get("sequence") + "_results.json" != p.name:
            continue
        tasks = {}
        for task, t in r["tasks"].items():
            if t.get("status") != "run":
                tasks[task] = {"status": t.get("status"), "reason": t.get("reason"),
                               "annotation_version": t.get("annotation_version", "")}
                continue
            key = t.get("metric_key") or PRIMARY_METRIC.get(task, (None,))[0]
            tasks[task] = {"metric": t.get("metric"), "annotation_version": t.get("annotation_version", ""),
                           "models": {
                mid: {"score_threshold": (m.get("model") or {}).get("score_threshold"),
                      "weights_sha256": (m.get("model") or {}).get("weights_sha256"),
                      "points": [{"qp": pt["qp"], "bitrate": pt.get("bitrate"), "bitrate_unit": pt.get("bitrate_unit"),
                                  **({"bitrate_kbps": pt["bitrate_kbps"]} if "bitrate_kbps" in pt else {}),
                                  key: pt.get(key),
                                  **({"error": pt["error"]} if "error" in pt else {})} for pt in m["points"]]}
                for mid, m in t.get("models", {}).items()}}
        root = resolve_path(".")
        rel = p.relative_to(root) if p.is_relative_to(root) else p      # data_root may be outside the project
        seqs[r["sequence"]] = {"results_file": str(rel), "generated": r.get("generated"),
                               "framerate": r.get("framerate"), "tasks": tasks,
                               "warnings": r.get("warnings", []), "errors": r.get("errors", [])}
    dr = project_relative(data_root)
    out = resolve_path(cfg["results_root"]) / "summary.json"
    prev = _summary_data_root(out)
    if prev is not None and prev != dr:
        log.warning(f"{out} was collected from data root {prev}; now rewritten from {dr} (the sequences of {prev} "
                    f"are no longer in it; their results files are kept): use a separate results_root per data root")
    summary = {"generated": datetime.now().isoformat(timespec="seconds"), "data_root": dr, "sequences": seqs}
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=1)
    log.info(f"Summary: {out} ({len(seqs)} sequence(s))")
    return out
