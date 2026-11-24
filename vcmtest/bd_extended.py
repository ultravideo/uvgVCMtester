"""Extended BD analysis of one anchor/test pair (docs/METRICS.md, "Extended BD analysis").

Written next to bd.json as extended.json and <test>__vs__<anchor>_extended.md, from the same two results JSONs and the
compare() output (no rescoring). Two parts:

1. Breakdown: BD-rate / BD-accuracy of the result's sub-metrics per model, with the same method as the plotted metric
   (Pareto front, PCHIP, overlap, bd.min_points):
   - components of the plotted stats field (e.g. AP50 / AP75 / APs / APm / APl for detection, DetA / AssA for tracking,
     detection recall / char accuracy for plate recognition), so a loss can be placed on object size, localisation,
     association or recognition;
   - per class (detection / instance seg / oriented: per-class AP; tracking: per-class value of the plotted key;
     panoptic: per-class PQ of the classes with ground truth; semantic: per-class IoU).
   Every sub-metric is scored on the same QPs as the plotted metric. A COCO value of -1 (no ground truth in that size
   range) is left out. The anchor's best value is listed: a BD of a curve near 0 says little.
2. Model agreement, per task: do the models agree on the codec's effect?
   - BD-rate (and BD-accuracy) over the models: sign counts, mean, median, min, max, standard deviation; verdict
     "consistent" (all models the same sign), "mostly" (>= 75 % the same sign) or "mixed", and a flag when the spread
     (standard deviation) exceeds |mean|.
   - QP steps of each side (points by decreasing bitrate): per step the mean accuracy change over the models (also
     per halving of the bitrate, so steps of different size compare), the models whose accuracy rises although the
     bitrate falls (inversion; one model = model noise, every model = an effect of the content or the codec) and the
     models whose largest drop per halving is at that step (all at one step = a collapse of the encoding there).
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from pathlib import Path

from . import metric_options as _mo
from .bd import _compare_model, _fmt

log = logging.getLogger("vcmtest")

_AP_PARTS = ("AP50", "AP75", "APs", "APm", "APl")
# task -> components of the plotted stats field (fractions; all higher is better)
COMPONENTS = {
    "detection": _AP_PARTS,
    "instance_segmentation": _AP_PARTS,
    "oriented_detection": _AP_PARTS,
    "keypoints": ("AP50", "AP75", "APm", "APl"),
    "tracking": ("DetA", "AssA", "LocA", "MOTA", "IDF1"),
    "panoptic_segmentation": ("SQ", "RQ", "PQ_th", "PQ_st"),
    "semantic_segmentation": ("pixel_accuracy",),
    "depth": ("delta1",),
    "plate_recognition": ("detection_recall", "accuracy_detected", "char_accuracy", "det_AP"),
    "bib_recognition": ("detection_recall", "accuracy_detected", "char_accuracy", "det_AP"),
}
# COCO-style values where -1 means "no ground truth in this range"
_NEG_IS_MISSING = set(_AP_PARTS) | {"det_AP"}
_COMPONENT_LIB = set()   # components where lower is better (none at present)
SAME_SIGN_MOSTLY = 0.75


def _stats_field(task: str, trec: dict) -> str:
    """Result field of the plotted stats, from the option values recorded in the task's results."""
    for name in _mo.options_for(task):
        val = trec.get(name)
        v = _mo.METRIC_OPTIONS[name].variants.get(val) if val is not None else None
        if v is not None and v.stats != "stats":
            return v.stats
    return "stats"


def _class_getter(task: str, key: str, trec: dict):
    """(field description, f(point, class) -> fraction | None, classes(point) -> names) for the per-class breakdown,
    or None if the task has none."""
    if task in ("detection", "instance_segmentation", "oriented_detection"):
        return ("per-class AP", lambda p, c: (p.get("per_class_AP") or {}).get(c),
                lambda p: list(p.get("per_class_AP") or {}))
    if task == "tracking":
        return (f"per-class {key}", lambda p, c: ((p.get("per_class") or {}).get(c) or {}).get(key),
                lambda p: list(p.get("per_class") or {}))
    if task == "panoptic_segmentation":
        things = trec.get("thing_classes")

        def classes(p):
            out = []
            for c, v in (p.get("per_class_PQ") or {}).items():
                if not isinstance(v, dict) or v.get("TP", 0) + v.get("FN", 0) <= 0:
                    continue        # no ground truth in this class
                if things is not None and v.get("isthing") and c not in things:
                    continue
                out.append(c)
            return out
        return ("per-class PQ (classes with ground truth)",
                lambda p, c: ((p.get("per_class_PQ") or {}).get(c) or {}).get("PQ"), classes)
    if task == "semantic_segmentation":
        return ("per-class IoU", lambda p, c: (p.get("per_class_IoU") or {}).get(c),
                lambda p: list(p.get("per_class_IoU") or {}))
    return None


def _sub_model(m: dict, get) -> dict:
    """A copy of a results model entry whose points carry one sub-metric value under '_v'."""
    pts = []
    for p in m.get("points", []):
        v = get(p)
        if v is None:
            continue
        pts.append({"qp": p["qp"], "bitrate": p.get("bitrate"), "frames": p.get("frames"), "_v": float(v)})
    return {"slot": m.get("slot", 0), "model": m.get("model"), "points": pts}


def _bd_sub(ma: dict, mt: dict, get, lib: bool, sel: dict, min_pts: int) -> dict:
    sa, st = _sub_model(ma, get), _sub_model(mt, get)
    r = _compare_model(sa, st, "_v", lib, sel, min_pts)
    best = (min if lib else max)
    va = [p["value"] for p in r["anchor"]["points"]]
    vt = [p["value"] for p in r["test"]["points"]]
    out = {"bd_rate_pct": r.get("bd_rate_pct"), "bd_accuracy_pp": r.get("bd_accuracy_pp"),
           "anchor_best": best(va) if va else None, "test_best": best(vt) if vt else None,
           "removed_qps": {s: r[s]["removed_qps"] for s in ("anchor", "test") if r[s]["removed_qps"]}}
    reason = r.get("reason") or "; ".join(v for k, v in r.items() if k.endswith("_reason"))
    if reason:
        out["reason"] = reason
    return out


def _mean_rec(models: dict) -> dict:
    br = [m["bd_rate_pct"] for m in models.values() if m.get("bd_rate_pct") is not None]
    ba = [m["bd_accuracy_pp"] for m in models.values() if m.get("bd_accuracy_pp") is not None]
    ab = [m["anchor_best"] for m in models.values() if m.get("anchor_best") is not None]
    return {"bd_rate_pct": sum(br) / len(br) if br else None, "bd_rate_models": len(br),
            "bd_accuracy_pp": sum(ba) / len(ba) if ba else None, "bd_accuracy_models": len(ba),
            "anchor_best_mean": sum(ab) / len(ab) if ab else None, "models": len(models)}


def breakdown(task: str, t: dict, ta: dict, tt: dict, sel: dict, min_pts: int) -> dict:
    """Sub-metric BD per model for one compared task. ta / tt = the task records of the anchor / test results."""
    field = _stats_field(task, tt)
    key = t["metric_key"]
    mids = list(t["models"])
    ma, mt = ta.get("models", {}), tt.get("models", {})
    out = {"stats_field": field, "components": {}, "classes": {}}

    def comp_get(name):
        def get(p):
            v = (p.get(_mo.resolve_stats_field(field, p)) or {}).get(name)
            return None if v is None or (name in _NEG_IS_MISSING and v < 0) else v
        return get

    for name in COMPONENTS.get(task, ()):
        models = {mid: _bd_sub(ma[mid], mt[mid], comp_get(name), name in _COMPONENT_LIB, sel, min_pts)
                  for mid in mids}
        models = {k: v for k, v in models.items() if v["anchor_best"] is not None or v["test_best"] is not None}
        if models:
            out["components"][name] = {"models": models, "mean": _mean_rec(models)}
    cg = _class_getter(task, key, tt)
    if cg is not None:
        desc, get, classes = cg
        out["class_field"] = desc
        names: list = []
        for mid in mids:
            for p in ma[mid].get("points", []) + mt[mid].get("points", []):
                names += [c for c in classes(p) if c not in names]
        for c in names:
            def g(p, c=c):
                if c not in classes(p):
                    return None
                v = get(p, c)
                return None if v is None or v < 0 else v
            models = {mid: _bd_sub(ma[mid], mt[mid], g, False, sel, min_pts) for mid in mids}
            models = {k: v for k, v in models.items() if v["anchor_best"] is not None or v["test_best"] is not None}
            if models:
                out["classes"][c] = {"models": models, "mean": _mean_rec(models)}
    return out


def _spread(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    neg, pos = sum(v < 0 for v in vals), sum(v > 0 for v in vals)
    mean = sum(vals) / len(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    if len(vals) < 2:
        verdict = "single model"
    elif neg == len(vals) or pos == len(vals):
        verdict = "consistent"
    elif max(neg, pos) / len(vals) >= SAME_SIGN_MOSTLY:
        verdict = "mostly"
    else:
        verdict = "mixed"
    return {"n": len(vals), "negative": neg, "positive": pos, "mean": mean, "median": statistics.median(vals),
            "min": min(vals), "max": max(vals), "stdev": sd, "verdict": verdict,
            "spread_exceeds_mean": len(vals) > 1 and sd > abs(mean)}


def _steps(t: dict, side: str) -> list[dict]:
    """Per QP step of one side (points by decreasing bitrate): mean change over the models (pp, sign = better/worse
    for the metric; also per halving of the bitrate), models with an inversion (better at the lower bitrate) and
    models whose largest drop per halving of the bitrate is here."""
    lib = t["lower_is_better"]
    sgn = -1 if lib else 1
    steps: dict = {}
    largest: dict = {}
    for mid, m in t["models"].items():
        pts = sorted(m[side]["points"], key=lambda p: -p["bitrate"])
        worst = None
        for a, b in zip(pts, pts[1:]):
            k = (a["qp"], b["qp"])
            d = sgn * (b["value"] - a["value"])       # > 0: better at the lower bitrate
            h = math.log2(a["bitrate"] / b["bitrate"]) if a["bitrate"] > b["bitrate"] else None
            s = steps.setdefault(k, {"changes": {}, "per_halving": {}, "inverted": []})
            s["changes"][mid] = d
            if h:
                s["per_halving"][mid] = d / h
            if d > 0:
                s["inverted"].append(mid)
            dn = d / h if h else d
            if worst is None or dn < worst[1]:
                worst = (k, dn)
        if worst is not None and worst[1] < 0:
            largest.setdefault(worst[0], []).append(mid)
    out = []
    for (qa, qb), s in sorted(steps.items(), key=lambda kv: kv[0]):
        ch, ph = list(s["changes"].values()), list(s["per_halving"].values())
        out.append({"from_qp": qa, "to_qp": qb, "models": len(ch), "mean_change_pp": sum(ch) / len(ch),
                    "min_change_pp": min(ch), "mean_change_per_halving_pp": sum(ph) / len(ph) if ph else None,
                    "inverted": s["inverted"], "largest_drop": largest.get((qa, qb), [])})
    out.sort(key=lambda s: -max(p["bitrate"] for m in t["models"].values() for p in m[side]["points"]
                                if p["qp"] == s["from_qp"]))
    return out


def _step_note(s: dict) -> str:
    n, k = s["models"], len(s["inverted"])
    if k == 0:
        return ""
    if n > 1 and k == n:
        return "all models better at the lower bitrate: content / codec effect"
    if k == 1 and n > 2:
        return "one model: model noise"
    return f"{k}/{n} models"


def agreement(t: dict) -> dict:
    """Model agreement of one compared task.

    Spread and sign verdict of the BD-rate and BD-accuracy values over the models, and the per QP step changes
    of each side (anchor, test) with a note.
    """
    vals_r = [m["bd_rate_pct"] for m in t["models"].values() if m.get("bd_rate_pct") is not None]
    vals_a = [m["bd_accuracy_pp"] for m in t["models"].values() if m.get("bd_accuracy_pp") is not None]
    out = {"bd_rate": _spread(vals_r), "bd_accuracy": _spread(vals_a), "steps": {}}
    for side in ("anchor", "test"):
        st = _steps(t, side)
        for s in st:
            s["note"] = _step_note(s)
        out["steps"][side] = st
    return out


def analyse(bd_res: dict, anchor_res: dict, test_res: dict, cfg: dict) -> dict:
    """Extended analysis of one BD pair.

    Per compared task: the sub-metric breakdown, the model agreement and each model's BD-rate / BD-accuracy.
    """
    sel = (cfg["bd"].get("qps") or {}).get(bd_res["test"]) or {}
    min_pts = cfg["bd"]["min_points"]
    out = {"anchor": bd_res["anchor"], "test": bd_res["test"], "generated": bd_res["generated"],
           "vcmtest_version": bd_res["vcmtest_version"], "method": bd_res["method"], "min_points": min_pts,
           "tasks": {}}
    for task, t in bd_res["tasks"].items():
        if t["status"] != "compared" or not t["models"]:
            continue
        ta, tt = anchor_res["tasks"][task], test_res["tasks"][task]
        out["tasks"][task] = {"metric": t.get("metric"), "metric_key": t["metric_key"],
                              "lower_is_better": t["lower_is_better"],
                              "breakdown": breakdown(task, t, ta, tt, sel, min_pts),
                              "agreement": agreement(t),
                              "models": {mid: {"bd_rate_pct": m.get("bd_rate_pct"),
                                               "bd_accuracy_pp": m.get("bd_accuracy_pp")}
                                         for mid, m in t["models"].items()}}
    return out


# ----------------------------------------------------------------------------- markdown

def _short(mid: str) -> str:
    return mid.replace("ultralytics-", "").replace("torchvision-", "").replace("rfdetr-rfdetr-", "rfdetr-")


def _bd_cell(m: dict | None) -> str:
    if m is None:
        return ""
    if m.get("bd_rate_pct") is None:
        return "n/a"
    return _fmt(m["bd_rate_pct"], 1)


# What a size-range object's area is, per task (COCO area ranges, in pixels at the decoded frame size).
_AREA_OF = {"detection": "box area", "instance_segmentation": "mask pixel count",
            "oriented_detection": "rotated polygon area", "keypoints": "annotated person area"}


def size_note(task: str, components: dict) -> str | None:
    """Meaning of the APs / APm / APl rows of a task's breakdown (None if it has none)."""
    sizes = [k for k in ("APs", "APm", "APl") if k in components]
    if not sizes:
        return None
    rng = {"APs": "APs = small (area < 32² = 1,024 px²)", "APm": "APm = medium (32² to 96² = 1,024 to 9,216 px²)",
           "APl": "APl = large (> 96² = 9,216 px²)"}
    return (f"Object sizes: {'; '.join(rng[k] for k in sizes)}. COCO area ranges; area = "
            f"{_AREA_OF.get(task, 'object area')} in pixels at the decoded frame size. A range without ground truth "
            "is left out.")


def _sub_table(title: str, rows: dict, mids: list[str], overall: dict, note: str | None = None) -> list[str]:
    L = [f"**{title}** (BD-rate %, per model; mean over the models with a value; anchor best = mean of the "
         "models' best anchor value, %)", "",
         *([note, ""] if note else []),
         "| | Anchor best | Mean BD-rate | Mean BD-accuracy | " + " | ".join(_short(m) for m in mids) + " |",
         "|---|---:|---:|---:|" + "---:|" * len(mids)]
    for name, r in [("(plotted)", overall)] + list(rows.items()):
        mn = r["mean"]
        ab = "" if mn.get("anchor_best_mean") is None else f"{mn['anchor_best_mean']:.1f}"
        L.append(f"| {name} | {ab} | {_fmt(mn['bd_rate_pct'], 1)} ({mn['bd_rate_models']}/{mn['models']}) | "
                 f"{_fmt(mn['bd_accuracy_pp'], 2)} | " + " | ".join(_bd_cell(r["models"].get(m)) for m in mids)
                 + " |")
    notes = sorted({f"{name} / {_short(mid)}: {m['reason']}" for name, r in rows.items()
                    for mid, m in r["models"].items() if m.get("reason")})
    if notes:
        L += ["", "<details><summary>n/a reasons</summary>", ""] + [f"- {n}" for n in notes] + ["", "</details>"]
    return L + [""]


def markdown(ext: dict) -> str:
    """Markdown report of the extended analysis (breakdown tables and model agreement per task)."""
    L = [f"# Extended BD analysis: {ext['test']} vs anchor {ext['anchor']}", "",
         f"Generated {ext['generated']} (vcmtest {ext['vcmtest_version']}) from the same results as bd.json "
         "(no rescoring). Method for every sub-metric: as bd.json (Pareto front, PCHIP over log10 bitrate, "
         f"overlap, at least {ext['min_points']} Pareto points). Negative BD-rate = the test needs less bitrate.", "",
         "1. **Breakdown:** BD of the sub-metrics (components of the plotted value, per class), to see where the "
         "test encoding gains or loses (object size, localisation, association, recognition, class). A sub-metric "
         "near 0 on the anchor (anchor best) gives an unreliable BD.",
         "2. **Model agreement:** do the models agree? Verdict on the BD-rate signs (consistent / mostly >= 75 % / "
         "mixed; spread flag = standard deviation > |mean|), and per QP step the models that get better at the "
         "lower bitrate (inversion: one model = model noise, all models = content or codec effect) and where each "
         "model's largest drop is.", ""]
    for task, e in ext["tasks"].items():
        mids = list(e["models"])
        bdn = e["breakdown"]
        L += [f"## {task}: {e.get('metric')}" + (" (lower is better)" if e["lower_is_better"] else ""), ""]
        a = e["agreement"]
        for lab, k, unit in (("BD-rate", "bd_rate", "%"), ("BD-accuracy", "bd_accuracy", "pp")):
            s = a[k]
            if not s["n"]:
                L.append(f"- **{lab}:** no values")
                continue
            L.append(f"- **{lab} agreement: {s['verdict']}** ({s['negative']} negative, {s['positive']} positive "
                     f"of {s['n']}); mean {_fmt(s['mean'], 1)} {unit}, median {_fmt(s['median'], 1)}, range "
                     f"{_fmt(s['min'], 1)} .. {_fmt(s['max'], 1)}, sd {s['stdev']:.1f}"
                     + ("; **spread exceeds |mean|**" if s["spread_exceeds_mean"] else ""))
        L.append("")
        plotted = {"models": {m: {"bd_rate_pct": v["bd_rate_pct"]} for m, v in e["models"].items()},
                   "mean": _mean_rec({m: {"bd_rate_pct": v["bd_rate_pct"], "bd_accuracy_pp": v["bd_accuracy_pp"]}
                                      for m, v in e["models"].items()})}
        if bdn["components"]:
            L += _sub_table(f"Components ({bdn['stats_field']})", bdn["components"], mids, plotted,
                            size_note(task, bdn["components"]))
        if bdn["classes"]:
            L += _sub_table(f"Classes ({bdn['class_field']})", bdn["classes"], mids, plotted)
        for side in ("anchor", "test"):
            st = a["steps"][side]
            if not st:
                continue
            L += [f"**QP steps, {side}** (change = mean over the models, pp; positive = better at the lower "
                  "bitrate; per halving = divided by log2 of the bitrate ratio)", "",
                  "| Step | Models | Mean change | Per halving | Worst | Inverted "
                  "| Largest drop (per halving) of | Note |",
                  "|---|---:|---:|---:|---:|---|---|---|"]
            for s in st:
                L.append(f"| QP{s['from_qp']}→{s['to_qp']} | {s['models']} | {_fmt(s['mean_change_pp'], 1)} | "
                         f"{_fmt(s['mean_change_per_halving_pp'], 1)} | "
                         f"{_fmt(s['min_change_pp'], 1)} | {', '.join(_short(m) for m in s['inverted'])} | "
                         f"{', '.join(_short(m) for m in s['largest_drop'])} | {s['note']} |")
            L.append("")
    return "\n".join(L) + "\n"


def write(bd_res: dict, anchor_res: dict, test_res: dict, cfg: dict, d: Path) -> Path:
    """Write extended.json and <test>__vs__<anchor>_extended.md into d.

    Tasks whose BD-rate signs or spread disagree over the models are logged.

    Returns:
        Path of the Markdown file.
    """
    ext = analyse(bd_res, anchor_res, test_res, cfg)
    with open(d / "extended.json", "w") as f:
        json.dump(ext, f, indent=1)
    md = d / f"{bd_res['test']}__vs__{bd_res['anchor']}_extended.md"
    md.write_text(markdown(ext))
    for task, e in ext["tasks"].items():
        s = e["agreement"]["bd_rate"]
        if s["n"] > 1 and (s["verdict"] != "consistent" or s["spread_exceeds_mean"]):
            log.info(f"  {task}: BD-rate sign over the models {s['verdict']} ({s['negative']} negative, "
                     f"{s['positive']} positive), sd {s['stdev']:.1f} vs mean {s['mean']:+.1f} %: see {md.name}")
    log.info(f"  extended analysis: {md}")
    return md
