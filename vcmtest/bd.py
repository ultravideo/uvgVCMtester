"""Bjontegaard comparison of a test encoding against an anchor encoding of the same source (docs/METRICS.md).

Anchors: configs/anchors.yaml {sequence_name: anchor folder} (+ config bd.anchors). Every other data folder of that
sequence name is a test. The comparison reads only the two results JSONs (data/<seq>/results/<seq>_results.json),
per task and model:
  BD-rate (%)      = mean log10-bitrate difference test - anchor over the overlapping accuracy range,
                     as 100 * (10^mean - 1); negative = the test needs less bitrate for the same accuracy
  BD-accuracy (pp) = mean accuracy difference test - anchor (plotted metric, %) over the overlapping log10-bitrate
                     range; positive = higher value (for lower-is-better metrics, e.g. depth AbsRel: worse)
Curves are first reduced to their Pareto front (dominated points removed: another point with lower or equal
bitrate and better or equal accuracy, one strictly), then interpolated with PCHIP (piecewise cubic Hermite) and
integrated exactly over the overlap. Fewer than bd.min_points Pareto points or no overlap -> n/a with the reason.

Outputs in bd.out_dir (BD-results/): <test>__vs__<anchor>/bd.json, <task>_rd.png, rd_per_model/, a Markdown
table, the extended analysis (bd.extended: extended.json + <test>__vs__<anchor>_extended.md, vcmtest/bd_extended.py),
and summary.json over all pairs.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path

from . import __version__
from . import metric_options as _mo
from .config import ConfigError, resolve_path

log = logging.getLogger("vcmtest")

METHOD = ("Bjontegaard with PCHIP (piecewise cubic Hermite) interpolation, log10 bitrate, Pareto-front points "
          "only, integrated over the overlapping range")
LOWER_IS_BETTER = _mo.lower_is_better_keys()
# per-task result fields that change the plotted value: must be equal in anchor and test
_SETTING_FIELDS = ("metric_key", "classes", "thing_classes", "lpr_iou", "depth_min_mm")


# ----------------------------------------------------------------------------- math

def pareto(points: list[tuple], higher_is_better: bool = True) -> tuple[list[tuple], list[tuple]]:
    """points [(qp, bitrate, value)] -> (kept, removed), kept sorted by bitrate. A point is removed when another
    point has bitrate <= and value >= (<= if lower is better), at least one strictly. Exact duplicates in
    (bitrate, value): the first is kept."""
    sgn = 1 if higher_is_better else -1
    kept, removed = [], []
    for i, (q, r, v) in enumerate(points):
        dominated = any((r2 <= r and sgn * v2 >= sgn * v and (r2 < r or sgn * v2 > sgn * v))
                        or (r2 == r and v2 == v and j < i)
                        for j, (_, r2, v2) in enumerate(points) if j != i)
        (removed if dominated else kept).append((q, r, v))
    return sorted(kept, key=lambda p: p[1]), sorted(removed, key=lambda p: p[1])


def _pchip_mean(x: list[float], y: list[float], lo: float, hi: float) -> float:
    from scipy.interpolate import PchipInterpolator
    order = sorted(range(len(x)), key=lambda i: x[i])
    f = PchipInterpolator([x[i] for i in order], [y[i] for i in order])
    return float(f.integrate(lo, hi)) / (hi - lo)


def bd_rate(anchor: list[tuple], test: list[tuple]) -> tuple[float | None, dict]:
    """anchor/test: Pareto points [(qp, bitrate, value)], values strictly monotonic in bitrate.
    Returns (BD-rate in %, info with the accuracy overlap) or (None, info with the reason)."""
    va, vt = [p[2] for p in anchor], [p[2] for p in test]
    lo, hi = max(min(va), min(vt)), min(max(va), max(vt))
    if not hi > lo:
        return None, {"reason": f"no overlap in accuracy (anchor {min(va):.3f}..{max(va):.3f}, "
                                f"test {min(vt):.3f}..{max(vt):.3f})"}
    ma = _pchip_mean(va, [math.log10(p[1]) for p in anchor], lo, hi)
    mt = _pchip_mean(vt, [math.log10(p[1]) for p in test], lo, hi)
    return 100.0 * (10 ** (mt - ma) - 1), {"accuracy_overlap": [lo, hi]}


def bd_accuracy(anchor: list[tuple], test: list[tuple]) -> tuple[float | None, dict]:
    """Mean value difference test - anchor (in the values' unit) over the overlapping log10-bitrate range."""
    ra, rt = [math.log10(p[1]) for p in anchor], [math.log10(p[1]) for p in test]
    lo, hi = max(min(ra), min(rt)), min(max(ra), max(rt))
    if not hi > lo:
        return None, {"reason": f"no overlap in bitrate (anchor {10 ** min(ra):.4g}..{10 ** max(ra):.4g}, "
                                f"test {10 ** min(rt):.4g}..{10 ** max(rt):.4g})"}
    ma = _pchip_mean(ra, [p[2] for p in anchor], lo, hi)
    mt = _pchip_mean(rt, [p[2] for p in test], lo, hi)
    return mt - ma, {"bitrate_overlap": [10 ** lo, 10 ** hi]}


# ----------------------------------------------------------------------------- pairs

def results_path(cfg: dict, seq: str) -> Path:
    """Path of a sequence's <seq>_results.json under data_root."""
    return resolve_path(cfg["data_root"]) / seq / "results" / f"{seq}_results.json"


def configured_pairs(cfg: dict, seqs: list[str] | None = None) -> list[tuple[str, str]]:
    """(anchor, test) pairs of the anchors (configs/anchors.yaml + bd.anchors) over the data folders; with seqs,
    only the pairs in which one of them is the anchor or the test."""
    from .config import anchor_of
    data_root = resolve_path(cfg["data_root"])
    folders = sorted(p.name for p in data_root.iterdir() if p.is_dir()) if data_root.is_dir() else []
    pairs = [(anchor_of(cfg, f), f) for f in folders if anchor_of(cfg, f)]
    if seqs is not None:
        s = set(seqs)
        pairs = [(a, t) for a, t in pairs if a in s or t in s]
    return pairs


def _encoding(res: dict) -> dict:
    """Codec / profile / pixel format of the decoded bitstreams (as probed when decoding)."""
    out = {}
    for k in ("codec_name", "profile", "pix_fmt"):
        vals = sorted({str(((q.get("decode") or {}).get("stream") or {}).get(k))
                       for q in res.get("qps", {}).values() if ((q.get("decode") or {}).get("stream") or {}).get(k)})
        if vals:
            out[k] = vals[0] if len(vals) == 1 else vals
    sizes = sorted({_size_str(q) for q in res.get("qps", {}).values()} - {None})
    if sizes:
        out["frame_size"] = sizes[0] if len(sizes) == 1 else sizes
    return out


def _size_str(q: dict) -> str | None:
    """'<W>x<H>' of one QP's evaluated frames (results "frame_size", else the decoded stream), None if unknown."""
    fs = q.get("frame_size")
    if fs:
        return f"{fs[0]}x{fs[1]}"
    st = (q.get("decode") or {}).get("stream") or {}
    return f"{st['width']}x{st['height']}" if st.get("width") and st.get("height") else None


def frame_size_note(enc: dict) -> str | None:
    """Note on the frame sizes of the two sides (encodings of compare()): differ, or unknown; None if equal.
    Not a reason to skip: each side is scored against the same annotations at its own size."""
    a, t = enc["anchor"].get("frame_size"), enc["test"].get("frame_size")
    if a is None or t is None:
        miss = [s for s, v in (("anchor", a), ("test", t)) if v is None]
        return (f"frame size not checked: not recorded for {' and '.join(miss)} (results file older than the "
                "frame_size field; rerun the sequence to record it)")
    if a != t:
        return (f"frame sizes differ (anchor {a}, test {t}): BD computed; each side is scored against the same "
                "annotations at its own frame size; size-range sub-metrics (APs / APm / APl) use absolute pixel "
                "areas and are not directly comparable")
    return None


def _curve(m: dict, key: str, qps: list[int] | None) -> list[tuple]:
    return [(p["qp"], float(p["bitrate"]), 100.0 * float(p[key])) for p in m.get("points", [])
            if p.get("bitrate") is not None and p.get(key) is not None and p["bitrate"] > 0
            and (qps is None or p["qp"] in qps)]


def _frames(m: dict) -> set:
    return {p.get("frames") for p in m.get("points", []) if p.get("frames") is not None}


def compare(anchor_res: dict, test_res: dict, cfg: dict) -> dict:
    """BD comparison of two results JSONs, per task and model (see module docstring)."""
    an, tn = anchor_res["sequence"], test_res["sequence"]
    sel = (cfg["bd"].get("qps") or {}).get(tn) or {}
    min_pts = cfg["bd"]["min_points"]
    out = {"anchor": an, "test": tn, "sequence_name": test_res.get("sequence_name"),
           "generated": datetime.now().isoformat(timespec="seconds"), "vcmtest_version": __version__,
           "method": METHOD, "min_points": min_pts,
           "anchor_results": {"file": str(results_path(cfg, an)), "generated": anchor_res.get("generated")},
           "test_results": {"file": str(results_path(cfg, tn)), "generated": test_res.get("generated")},
           "encodings": {"anchor": _encoding(anchor_res), "test": _encoding(test_res)},
           "bitrate_unit": test_res.get("bitrate_unit"), "tasks": {}}
    if sel:
        out["qp_selection"] = sel
    note = frame_size_note(out["encodings"])
    if note:
        out["frame_size_note"] = note
    if anchor_res.get("sequence_name") != test_res.get("sequence_name"):
        out["warning"] = (f"different sequence names: anchor {anchor_res.get("sequence_name")}, "
                          f"test {test_res.get("sequence_name")}")
    unit_problem = None
    if anchor_res.get("bitrate_unit") != test_res.get("bitrate_unit"):
        unit_problem = (f"bitrate units differ (anchor {anchor_res.get('bitrate_unit')}, "
                        f"test {test_res.get('bitrate_unit')}); not converted")
    elif not test_res.get("bitrate_unit"):
        unit_problem = "no bitrate unit (no bitrates)"
    test_order = list(test_res.get("tasks", {}))
    for task in sorted(set(anchor_res.get("tasks", {})) | set(test_order),
                       key=lambda t: test_order.index(t) if t in test_order else 99):
        ta, tt = anchor_res.get("tasks", {}).get(task), test_res.get("tasks", {}).get(task)
        rec = {"annotation_version": {"anchor": (ta or {}).get("annotation_version"),
                                      "test": (tt or {}).get("annotation_version")}}
        reason = None
        ran = [bool(x and x.get("status") == "run") for x in (ta, tt)]
        if not any(ran):
            out["tasks"][task] = rec | {"status": "not_run"}
            continue
        if not all(ran):
            reason = (f"not run on both (anchor: {(ta or {}).get('status', 'absent')}, "
                      f"test: {(tt or {}).get('status', 'absent')})")
        elif unit_problem:
            reason = unit_problem
        else:
            diffs = [f"{k}: anchor {ta.get(k)!r}, test {tt.get(k)!r}"
                     for k in _SETTING_FIELDS + tuple(_mo.options_for(task)) if ta.get(k) != tt.get(k)]
            if ta.get("annotation_version") != tt.get("annotation_version"):
                diffs.append(f"annotation_version: anchor {ta.get('annotation_version')!r}, "
                             f"test {tt.get('annotation_version')!r}")
            sa, st = ta.get("annotations_sha256"), tt.get("annotations_sha256")
            if sa and st and sa != st:
                diffs.append("annotations_sha256 differs (annotation files or masks not identical)")
            elif not (sa and st):
                rec["note"] = "annotations_sha256 not recorded in an older results file: compared by version only"
            if diffs:
                reason = "settings differ: " + "; ".join(diffs)
        if reason:
            out["tasks"][task] = rec | {"status": "skipped", "reason": reason}
            continue
        key = tt["metric_key"]
        lib = key in LOWER_IS_BETTER
        rec |= {"status": "compared", "metric": tt.get("metric"), "metric_key": key, "lower_is_better": lib,
                "settings": {k: tt[k] for k in _SETTING_FIELDS + tuple(_mo.options_for(task)) if k in tt},
                "models": {}}
        ma, mt = ta.get("models", {}), tt.get("models", {})
        if set(ma) - set(mt):
            rec["only_in_anchor"] = sorted(set(ma) - set(mt))
        if set(mt) - set(ma):
            rec["only_in_test"] = sorted(set(mt) - set(ma))
        for mid in [m for m in ma if m in mt]:
            rec["models"][mid] = _compare_model(ma[mid], mt[mid], key, lib, sel, min_pts)
        vals = [m for m in rec["models"].values()]
        br = [m["bd_rate_pct"] for m in vals if m.get("bd_rate_pct") is not None]
        ba = [m["bd_accuracy_pp"] for m in vals if m.get("bd_accuracy_pp") is not None]
        rec["mean"] = {"bd_rate_pct": sum(br) / len(br) if br else None, "bd_rate_models": len(br),
                       "bd_accuracy_pp": sum(ba) / len(ba) if ba else None, "bd_accuracy_models": len(ba),
                       "models": len(vals)}
        out["tasks"][task] = rec
    return out


def _compare_model(ma: dict, mt: dict, key: str, lib: bool, sel: dict, min_pts: int) -> dict:
    rec = {"slot": ma.get("slot", 0)}
    wa, wt = (ma.get("model") or {}).get("weights_sha256"), (mt.get("model") or {}).get("weights_sha256")
    fa, ft = _frames(ma), _frames(mt)
    sides = {}
    for side, m in (("anchor", ma), ("test", mt)):
        pts = _curve(m, key, sel.get(side))
        kept, removed = pareto(pts, higher_is_better=not lib)
        sides[side] = (kept, removed)
        rec[side] = {"points": [{"qp": q, "bitrate": r, "value": v} for q, r, v in sorted(pts, key=lambda p: p[1])],
                     "pareto_qps": [p[0] for p in kept], "removed_qps": [p[0] for p in removed]}
    reason = None
    if wa and wt and wa != wt:
        reason = "model weights differ (weights_sha256)"
    elif len(fa | ft) > 1:
        reason = f"frame counts differ (anchor {sorted(fa)}, test {sorted(ft)})"
    else:
        short = [f"{s} {len(sides[s][0])}" for s in ("anchor", "test") if len(sides[s][0]) < min_pts]
        if short:
            reason = f"fewer than {min_pts} Pareto points ({', '.join(short)})"
    if reason:
        rec["reason"] = reason
        rec["bd_rate_pct"] = rec["bd_accuracy_pp"] = None
        return rec
    (ka, _), (kt, _) = sides["anchor"], sides["test"]
    rec["bd_rate_pct"], info_r = bd_rate(ka, kt)
    rec["bd_accuracy_pp"], info_a = bd_accuracy(ka, kt)
    for k, v in info_r.items():
        rec["bd_rate_reason" if k == "reason" else k] = v
    for k, v in info_a.items():
        rec["bd_accuracy_reason" if k == "reason" else k] = v
    return rec


# ----------------------------------------------------------------------------- outputs

def pair_dir(cfg: dict, anchor: str, test: str) -> Path:
    """Output folder of one BD pair: <bd.out_dir>/<test>__vs__<anchor>."""
    return resolve_path(cfg["bd"]["out_dir"]) / f"{test}__vs__{anchor}"


def _fmt(v, nd=2, sign=True) -> str:
    return "n/a" if v is None else (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}")


def _write_plots(cfg: dict, res: dict, d: Path) -> None:
    from .plot import plot_bd
    log_x = cfg["bd"]["log_x"]
    unit = res.get("bitrate_unit")
    for task, t in res["tasks"].items():
        if t.get("status") != "compared" or not t["models"]:
            continue
        ylabel = f"{t.get('metric') or t['metric_key']} (%)"
        series = []
        for mid, m in t["models"].items():
            s = {"label": f"{mid}  BD-rate {_fmt(m.get('bd_rate_pct'))} %", "slot": m["slot"]}
            for side in ("anchor", "test"):
                byq = {p["qp"]: (p["bitrate"], p["value"]) for p in m[side]["points"]}
                s[side] = {"kept": [byq[q] for q in m[side]["pareto_qps"]],
                           "removed": [byq[q] for q in m[side]["removed_qps"]]}
            series.append(s)
            m["rd_plot"] = str(plot_bd([s], f"{task} - {mid}\ntest {res['test']}\nanchor {res['anchor']}", ylabel,
                                       d / "rd_per_model" / f"{task}_{mid}_rd.png", log_x, unit))
        mean = t["mean"]["bd_rate_pct"]
        t["rd_plot"] = str(plot_bd(series, f"{task} - mean BD-rate {_fmt(mean)} %\ntest {res['test']}\n"
                                           f"anchor {res['anchor']}", ylabel, d / f"{task}_rd.png",
                                   log_x, unit))


def _qps_cell(m: dict, side: str) -> str:
    """'<Pareto QPs> / <removed QPs>' of one side of a model's comparison."""
    return (",".join(map(str, m[side]["pareto_qps"])) or "-") + " / " + (",".join(map(str, m[side]["removed_qps"]))
                                                                         or "-")


def _markdown(res: dict) -> str:
    L = [f"# BD comparison: {res['test']} vs anchor {res['anchor']}", "",
         f"Generated {res['generated']} (vcmtest {res['vcmtest_version']}). {res['method']}.", "",
         "BD-rate: bitrate change of the test at equal accuracy (negative = test needs less bitrate). "
         "BD-accuracy: change of the plotted metric at equal bitrate, percentage points (for lower-is-better "
         "metrics a negative value is better).", "",
         f"Anchor encoding: {res['encodings']['anchor']}  ", f"Test encoding: {res['encodings']['test']}", ""]
    if res.get("qp_selection"):
        L += [f"QP selection: {res['qp_selection']}", ""]
    if res.get("warning"):
        L += [f"**Warning:** {res['warning']}", ""]
    if res.get("frame_size_note"):
        L += [f"**Note:** {res['frame_size_note']}", ""]
    for task, t in res["tasks"].items():
        if t["status"] != "compared":
            if t["status"] == "skipped":
                L += [f"## {task}", "", f"Skipped: {t['reason']}", ""]
            continue
        L += [f"## {task}: {t.get('metric')}" + (" (lower is better)" if t["lower_is_better"] else ""), "",
              "| Model | BD-rate (%) | BD-accuracy (pp) | Anchor QPs (Pareto / removed) | Test QPs (Pareto / removed) "
              "| Note |", "|---|---:|---:|---|---|---|"]
        for mid, m in t["models"].items():
            note = m.get("reason") or "; ".join(v for k, v in m.items() if k.endswith("_reason"))
            L.append(f"| {mid} | {_fmt(m.get('bd_rate_pct'))} | {_fmt(m.get('bd_accuracy_pp'))} | "
                     f"{_qps_cell(m, 'anchor')} | {_qps_cell(m, 'test')} | {note} |")
        mn = t["mean"]
        L.append(f"| **mean** | {_fmt(mn['bd_rate_pct'])} ({mn['bd_rate_models']}/{mn['models']}) | "
                 f"{_fmt(mn['bd_accuracy_pp'])} ({mn['bd_accuracy_models']}/{mn['models']}) | | | |")
        for k in ("only_in_anchor", "only_in_test"):
            if t.get(k):
                L.append(f"\n{k.replace('_', ' ')}: {', '.join(t[k])}")
        if t.get("note"):
            L.append(f"\n{t['note']}")
        L.append("")
    return "\n".join(L) + "\n"


def run_pair(cfg: dict, anchor: str, test: str) -> dict:
    """Compare one pair from the results JSONs on disk and write its outputs. Raises ConfigError if a results
    JSON is missing."""
    res = {}
    for name in (anchor, test):
        p = results_path(cfg, name)
        if not p.is_file():
            raise ConfigError(f"no results for {name} ({p}); run the sequence first")
        with open(p) as f:
            res[name] = json.load(f)
    out = compare(res[anchor], res[test], cfg)
    d = pair_dir(cfg, anchor, test)
    d.mkdir(parents=True, exist_ok=True)
    _write_plots(cfg, out, d)
    with open(d / "bd.json", "w") as f:
        json.dump(out, f, indent=1)
    (d / f"{test}__vs__{anchor}.md").write_text(_markdown(out))
    if cfg["bd"].get("extended", True):
        from . import bd_extended
        bd_extended.write(out, res[anchor], res[test], cfg, d)
    log.info(f"BD {test} vs anchor {anchor}: {d}")
    if out.get("frame_size_note"):
        log.warning(f"  {out['frame_size_note']}")
    for task, t in out["tasks"].items():
        if t["status"] != "compared":
            if t["status"] == "skipped":
                log.warning(f"  {task}: skipped ({t['reason']})")
            continue
        for mid, m in t["models"].items():
            extra = m.get("reason") or "; ".join(v for k, v in m.items() if k.endswith("_reason"))
            rem = {s: m[s]["removed_qps"] for s in ("anchor", "test") if m[s]["removed_qps"]}
            log.info(f"  {task}/{mid}: BD-rate {_fmt(m.get('bd_rate_pct'))} %, BD-accuracy "
                     f"{_fmt(m.get('bd_accuracy_pp'))} pp" + (f"  [{extra}]" if extra else "")
                     + (f"  removed (not Pareto): {rem}" if rem else ""))
        log.info(f"  {task}: mean BD-rate {_fmt(t['mean']['bd_rate_pct'])} % over {t['mean']['bd_rate_models']} "
                 f"model(s), mean BD-accuracy {_fmt(t['mean']['bd_accuracy_pp'])} pp")
    return out


def write_bd_summary(cfg: dict) -> Path:
    """Collect all <out_dir>/*/bd.json into <out_dir>/summary.json (committed to git)."""
    root = resolve_path(cfg["bd"]["out_dir"])
    pairs = {}
    for p in sorted(root.glob("*/bd.json")):
        with open(p) as f:
            r = json.load(f)
        tasks = {}
        for task, t in r["tasks"].items():
            if t["status"] != "compared":
                if t["status"] == "skipped":
                    tasks[task] = {"status": "skipped", "reason": t["reason"]}
                continue
            tasks[task] = {"metric": t.get("metric"), "lower_is_better": t["lower_is_better"], "mean": t["mean"],
                           "models": {mid: {k: m.get(k) for k in ("bd_rate_pct", "bd_accuracy_pp")}
                                      | {"pareto_removed": {s: m[s]["removed_qps"] for s in ("anchor", "test")
                                                            if m[s]["removed_qps"]}}
                                      | ({"reason": m["reason"]} if m.get("reason") else {})
                                      for mid, m in t["models"].items()}}
        pairs[p.parent.name] = {"anchor": r["anchor"], "test": r["test"], "generated": r["generated"],
                                "encodings": r["encodings"],
                                **({"frame_size_note": r["frame_size_note"]} if r.get("frame_size_note") else {}),
                                "tasks": tasks}
    out = root / "summary.json"
    root.mkdir(parents=True, exist_ok=True)
    ov = overview(pairs)
    with open(out, "w") as f:
        json.dump({"generated": datetime.now().isoformat(timespec="seconds"), "method": METHOD, "pairs": pairs,
                   "overview": ov}, f, indent=1)
    (root / "overview.md").write_text(_overview_markdown(ov))
    log.info(f"BD summary: {out} ({len(pairs)} pair(s), {len(ov)} comparison(s); overview.md)")
    return out


def _spec(folder: str) -> str:
    """Encoding spec of a data folder: the name after '<sequence_name>_' (CampusView_1920x1080_444_10b_VTM_RA ->
    1920x1080_444_10b_VTM_RA)."""
    return folder.split("_", 1)[1] if "_" in folder else folder


def _mean(vals: list) -> float | None:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def overview(pairs: dict) -> dict:
    """Cross-sequence overview: pairs grouped by comparison '<test spec> vs <anchor spec>' (folder names without the
    sequence name). Per task: each sequence's task mean (BD-rate %, BD-accuracy pp over its models), the overall mean
    over the sequences that have a value (each sequence weighs equally), and per model the mean over the sequences where
    that model id was compared. Tasks with a different metric or direction across sequences are kept apart by
    metric name."""
    groups: dict = {}
    for _name, p in sorted(pairs.items()):
        key = f"{_spec(p['test'])} vs {_spec(p['anchor'])}"
        seq = p["anchor"].split("_", 1)[0]
        g = groups.setdefault(key, {"test_spec": _spec(p["test"]), "anchor_spec": _spec(p["anchor"]),
                                    "sequences": [], "tasks": {}})
        g["sequences"].append(seq)
        for task, t in p["tasks"].items():
            if "mean" not in t:
                g["tasks"].setdefault(task, {"per_sequence": {}, "models": {}})["per_sequence"][seq] = {
                    "skipped": t.get("reason")}
                continue
            gt = g["tasks"].setdefault(task, {"per_sequence": {}, "models": {}})
            gt.setdefault("metric", t.get("metric"))
            gt.setdefault("lower_is_better", t.get("lower_is_better"))
            m = t["mean"]
            gt["per_sequence"][seq] = {"bd_rate_pct": m.get("bd_rate_pct"), "bd_accuracy_pp": m.get("bd_accuracy_pp"),
                                       "models": m.get("models"), "bd_rate_models": m.get("bd_rate_models")}
            for mid, mm in t["models"].items():
                e = gt["models"].setdefault(mid, {"per_sequence": {}})
                e["per_sequence"][seq] = {"bd_rate_pct": mm.get("bd_rate_pct"),
                                          "bd_accuracy_pp": mm.get("bd_accuracy_pp")}
    for g in groups.values():
        for gt in g["tasks"].values():
            ps = [v for v in gt["per_sequence"].values() if "skipped" not in v]
            gt["mean"] = {"bd_rate_pct": _mean([v["bd_rate_pct"] for v in ps]),
                          "bd_rate_sequences": sum(v["bd_rate_pct"] is not None for v in ps),
                          "bd_accuracy_pp": _mean([v["bd_accuracy_pp"] for v in ps]),
                          "bd_accuracy_sequences": sum(v["bd_accuracy_pp"] is not None for v in ps),
                          "sequences": len(ps)}
            for e in gt["models"].values():
                v = list(e["per_sequence"].values())
                e["bd_rate_pct"] = _mean([x["bd_rate_pct"] for x in v])
                e["bd_accuracy_pp"] = _mean([x["bd_accuracy_pp"] for x in v])
                e["sequences"] = sum(x["bd_rate_pct"] is not None for x in v)
    return groups


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:+.1f} %"


def _overview_markdown(ov: dict) -> str:
    L = ["# BD overview across sequences", "",
         "Generated by vcmtest from BD-results/summary.json. Pairs grouped by comparison (encoding spec of the test "
         "vs the anchor, sequence name removed). Per task: each sequence's mean over its models, and the overall mean "
         "over the sequences (each sequence weighs equally; n/a values left out, counts in brackets). Negative "
         "BD-rate = the test needs less bitrate for the same accuracy. Per-model values: mean over the sequences "
         "where the model was compared.", ""]
    for key, g in ov.items():
        seqs = sorted(set(g["sequences"]))
        L += [f"## {key}", "", f"Sequences: {', '.join(seqs)}", "",
              "| Task | " + " | ".join(seqs) + " | **Mean BD-rate** | **Mean BD-accuracy** |",
              "|---|" + "---:|" * len(seqs) + "---:|---:|"]
        for task, gt in g["tasks"].items():
            cells = []
            for s in seqs:
                v = gt["per_sequence"].get(s)
                if v is None:
                    cells.append("")
                elif "skipped" in v:
                    cells.append("skipped")
                else:
                    cells.append(f"{_pct(v['bd_rate_pct'])} ({v['bd_rate_models']}/{v['models']})")
            m = gt["mean"]
            L.append(f"| {task} | " + " | ".join(cells)
                     + f" | **{_pct(m['bd_rate_pct'])}** ({m['bd_rate_sequences']}/{m['sequences']})"
                     + f" | {_fmt(m['bd_accuracy_pp'], 2)} pp ({m['bd_accuracy_sequences']}/{m['sequences']}) |")
        L += ["", "Per model (BD-rate %, mean over sequences):", "",
              "| Task | Model | " + " | ".join(seqs) + " | Mean |", "|---|---|" + "---:|" * len(seqs) + "---:|"]
        for task, gt in g["tasks"].items():
            for mid, e in gt["models"].items():
                cells = [_fmt(e["per_sequence"][s]["bd_rate_pct"], 1) if s in e["per_sequence"] else ""
                         for s in seqs]
                L.append(f"| {task} | {mid} | " + " | ".join(cells)
                         + f" | {_fmt(e['bd_rate_pct'], 1)} ({e['sequences']}) |")
        L.append("")
    return "\n".join(L) + "\n"


def run_bd(cfg: dict, pairs: list[tuple[str, str]], strict: bool = False) -> int:
    """Run the given (anchor, test) pairs; pairs whose results are missing are skipped with a warning (an error if
    strict). Returns the number of failed pairs."""
    n_fail = 0
    for anchor, test in pairs:
        try:
            run_pair(cfg, anchor, test)
        except ConfigError as e:
            if strict:
                raise
            log.warning(f"BD {test} vs {anchor} skipped: {e}")
            n_fail += 1
    if pairs:
        write_bd_summary(cfg)
    else:
        log.info("BD: no anchor/test pairs")
    return n_fail
