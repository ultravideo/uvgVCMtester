"""Panoptic segmentation (PQ, COCO panopticapi algorithm) and semantic segmentation (mIoU).

Ground truth (<sequence_name>_panoptic_annotations.json + <sequence_name>_panoptic_masks/):
- One segment per JSON entry; its pixels are those with exactly its "mask_color" in the frame's mask PNG,
  downscaled to the decoded frame size with nearest neighbour. Pixels of any other colour are void (unlabelled).
- class_id = COCO panoptic category id (task_models/labels_panoptic.py; things 1..90, stuff 92..200).
- A segment whose colour has no pixels (at the decoded size) is left out (counted in "gt_counts").
- iscrowd=1 segments are crowd regions (panopticapi: never matched; an unmatched prediction of the same class
  lying mostly on void + that crowd region is not a false positive).

Predictions: per frame a list of segments [class_id, score, W, H, rle] (COCO panoptic category id, pycocotools
compressed RLE at the decoded size W x H). They are painted in list order (a later segment wins on overlap).

PQ (Kirillov et al., COCO panopticapi pq_compute): per frame, a predicted and a ground-truth segment of the same
class match when IoU > 0.5, with the prediction's void pixels removed from the union. Unmatched non-crowd GT
segments are false negatives. Unmatched predictions are false positives unless more than half of their pixels are
void or the crowd region of their class. TP / FP / FN / IoU sums are accumulated over all frames per class, then
PQ = sum IoU / (TP + FP/2 + FN/2), SQ = sum IoU / TP, RQ = TP / (TP + FP/2 + FN/2). The averages are over
classes with TP + FP + FN > 0 (panopticapi), overall and separately for things and stuff ("stats"). The same
averages over the GT classes only (classes with at least one ground-truth segment in the evaluated frames) are
in "stats_gt_classes": classes that are only predicted (false positives only) are left out of them. Their false
positives still count in the GT classes where they overlap (lost TP / FN there); config pq_average selects which
of the two is the plotted PQ.

mIoU: the per-pixel class maps (all segments of a class merged; crowd segments keep their class; prediction gaps =
no class). On non-void ground-truth pixels, intersection and union per class are accumulated over all frames;
IoU = TP / (TP + FP + FN). mIoU = mean over the classes that have ground-truth pixels in the evaluated frames;
pixels predicted as a class absent from the GT still count as errors for the true class.

Thing-class selection (sequence_overrides.<seq>.classes): evaluate(things=...) keeps only the listed thing classes
(plus all stuff classes) in the per-class results and in the PQ / mIoU averages. Matching and the per-class values
are unchanged (other things' predictions stay in the frame, as in panopticapi); pixel_accuracy covers all classes.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from ..annotations import AnnotationError, DetectionAnnotations, file_sha256, panoptic_categories
from .coco_segm import _load_mask_png, _rle_json, mask_files

log = logging.getLogger("vcmtest")

GT_VERSION = 1
OFFSET = 1 << 16          # segment ids per frame are < 65536
PQ_KEYS = ["PQ", "SQ", "RQ", "PQ_th", "SQ_th", "RQ_th", "PQ_st", "SQ_st", "RQ_st"]


# ----------------------------------------------------------------------------- ground truth

def gt_fingerprint(ann: DetectionAnnotations, masks_dir: Path) -> str:
    """Fingerprint of the panoptic ground truth (annotation JSON hash and the mask PNGs)."""
    h = hashlib.sha256(f"panoptic-gt-v{GT_VERSION}|{ann.sha256}|".encode())
    for fn, p in mask_files(masks_dir).items():
        h.update(f"{fn}|{p.name}|{file_sha256(p)}|".encode())
    return h.hexdigest()


def build_gt(ann: DetectionAnnotations, masks_dir: Path, width: int, height: int) -> dict:
    """{"frames": {fn: [{"id", "category_id", "iscrowd", "area", "rle"}]}, "counts": {...}}"""
    from pycocotools import mask as mu
    files = mask_files(masks_dir)
    frames, counts = {}, {"segments": 0, "crowd": 0, "empty": 0}
    for fn, segs in ann.frames.items():
        if not segs:
            frames[fn] = []
            continue
        if fn not in files:
            raise AnnotationError(f"frame {fn}: panoptic segments but no mask PNG in {masks_dir}")
        keys = _load_mask_png(files[fn], width, height)
        out = []
        for o in segs:
            r, g, b = (int(v) for v in o["mask_color"])
            m = keys == ((r << 16) | (g << 8) | b)
            area = int(m.sum())
            if not area:
                counts["empty"] += 1
                continue
            counts["crowd" if o["iscrowd"] else "segments"] += 1
            out.append({"id": len(out) + 1, "category_id": int(o["class_id"]), "iscrowd": int(o["iscrowd"]),
                        "area": area, "rle": _rle_json(mu.encode(np.asfortranarray(m.astype(np.uint8))))})
        frames[fn] = out
    return {"frames": frames, "counts": counts}


_GT_MEMO: dict[tuple, dict] = {}


def cached_gt(ann: DetectionAnnotations, masks_dir: Path, width: int, height: int, cache_dir: Path | None) -> dict:
    """build_gt, cached in memory and as <cache_dir>/panoptic_gt_<W>x<H>.json (keyed by the GT fingerprint)."""
    sha = gt_fingerprint(ann, masks_dir)
    memo_key = (sha, width, height)
    if memo_key in _GT_MEMO:
        return _GT_MEMO[memo_key]
    key = {"sha256": sha, "width": width, "height": height}
    path = cache_dir / f"panoptic_gt_{width}x{height}.json" if cache_dir else None
    gt = None
    if path and path.is_file():
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("key") == key:
                gt = data["gt"]
                gt["frames"] = {int(k): v for k, v in gt["frames"].items()}
        except (OSError, json.JSONDecodeError, KeyError):
            gt = None
    if gt is None:
        log.info(f"Building panoptic ground truth at {width}x{height} from {masks_dir.name}/ ...")
        gt = build_gt(ann, masks_dir, width, height)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump({"key": key, "gt": gt}, f)
            tmp.replace(path)
    _GT_MEMO[memo_key] = gt
    return gt


# ----------------------------------------------------------------------------- per-frame maps

def _decode(rle: dict) -> np.ndarray:
    from pycocotools import mask as mu
    r = {"size": rle["size"], "counts": rle["counts"].encode("ascii") if isinstance(rle["counts"], str)
         else rle["counts"]}
    return mu.decode(r).astype(bool)


def gt_maps(segs: list[dict], width: int, height: int) -> tuple[np.ndarray, dict]:
    """(segment id map, 0 = void; {id: segment})"""
    ids = np.zeros((height, width), np.int32)
    for s in segs:
        ids[_decode(s["rle"])] = s["id"]
    return ids, {s["id"]: s for s in segs}


def pred_maps(preds: list, width: int, height: int, valid: set[int]) -> tuple[np.ndarray, dict]:
    """(segment id map, 0 = no prediction; {id: {"category_id", "area"}}), segments painted in list order."""
    ids = np.zeros((height, width), np.int32)
    cats = {}
    for i, p in enumerate(preds, start=1):
        cid, _score, pw, ph, rle = p[:5]
        if (int(pw), int(ph)) != (width, height):
            raise AnnotationError(f"prediction mask size {pw}x{ph} differs from the frame size {width}x{height}")
        if int(cid) not in valid:
            raise AnnotationError(f"prediction class {cid} is not a COCO panoptic category")
        ids[_decode({"size": [height, width], "counts": rle})] = i
        cats[i] = int(cid)
    labels, cnt = np.unique(ids, return_counts=True)
    segs = {int(lab): {"category_id": cats[int(lab)], "area": int(c)} for lab, c in zip(labels, cnt) if lab}
    return ids, segs


# ----------------------------------------------------------------------------- metrics

def pq_frame(gt_ids, gt_segs, pr_ids, pr_segs, stat: dict) -> None:
    """Accumulate panopticapi TP/FP/FN/IoU for one frame into stat[category] = [tp, fp, fn, iou]."""
    def acc(c):
        return stat.setdefault(c, [0, 0, 0, 0.0])
    labels, cnt = np.unique(gt_ids.astype(np.int64) * OFFSET + pr_ids, return_counts=True)
    inter = {(int(lab // OFFSET), int(lab % OFFSET)): int(c) for lab, c in zip(labels, cnt)}
    gt_matched, pr_matched = set(), set()
    for (g, p), n in inter.items():
        if g not in gt_segs or p not in pr_segs:
            continue
        gs, ps = gt_segs[g], pr_segs[p]
        if gs["iscrowd"] or gs["category_id"] != ps["category_id"]:
            continue
        union = ps["area"] + gs["area"] - n - inter.get((0, p), 0)
        iou = n / union
        if iou > 0.5:
            a = acc(gs["category_id"])
            a[0] += 1
            a[3] += iou
            gt_matched.add(g)
            pr_matched.add(p)
    crowd = {}
    for g, gs in gt_segs.items():
        if g in gt_matched:
            continue
        if gs["iscrowd"]:
            crowd[gs["category_id"]] = g
            continue
        acc(gs["category_id"])[2] += 1
    for p, ps in pr_segs.items():
        if p in pr_matched:
            continue
        n = inter.get((0, p), 0)
        if ps["category_id"] in crowd:
            n += inter.get((crowd[ps["category_id"]], p), 0)
        if n / ps["area"] > 0.5:
            continue
        acc(ps["category_id"])[1] += 1


def pq_summary(stat: dict, isthing: dict[int, bool], names: dict[int, str],
               things: set[int] | None = None, only: set[int] | None = None) -> tuple[dict, dict]:
    """(averages, per-class values). only: average over these classes only (per-class values unchanged)."""
    per_class = {}
    for c, (tp, fp, fn, iou) in sorted(stat.items()):
        if tp + fp + fn == 0 or (things is not None and isthing[c] and c not in things):
            continue
        d = tp + 0.5 * fp + 0.5 * fn
        per_class[c] = {"PQ": iou / d, "SQ": iou / tp if tp else 0.0, "RQ": tp / d, "TP": tp, "FP": fp, "FN": fn,
                        "isthing": isthing[c]}

    def avg(sel):
        cs = [v for c, v in per_class.items() if sel(c) and (only is None or c in only)]
        if not cs:
            return None, None, None
        return tuple(float(np.mean([v[k] for v in cs])) for k in ("PQ", "SQ", "RQ"))

    stats = {}
    for suffix, sel in (("", lambda c: True), ("_th", lambda c: isthing[c]), ("_st", lambda c: not isthing[c])):
        stats["PQ" + suffix], stats["SQ" + suffix], stats["RQ" + suffix] = avg(sel)
    stats["n_classes"] = len(per_class) if only is None else len([c for c in per_class if c in only])
    return stats, {names[c]: v for c, v in per_class.items()}


def evaluate(gt: dict, pred_frames: dict, frames: list[int], width: int, height: int,
             things: set[int] | None = None) -> dict:
    """PQ (primary) and mIoU over the given frames. Values in [0, 1].
    things: COCO panoptic thing ids kept in the averages (None = all); stuff classes are always kept."""
    cats = panoptic_categories()
    names = {c: n for c, n, _ in cats}
    isthing = {c: bool(t) for c, _, t in cats}
    valid = set(names)
    n_bins = max(valid) + 1
    conf = np.zeros(n_bins * n_bins, np.int64)        # [gt class, pred class], class 0 = none
    stat: dict = {}
    n_pred = 0
    gt_classes: set[int] = set()
    for fn in frames:
        gsegs = gt["frames"].get(fn, [])
        gt_classes.update(int(s["category_id"]) for s in gsegs)
        g_ids, g_segs = gt_maps(gsegs, width, height)
        preds = pred_frames.get(str(fn), [])
        n_pred += len(preds)
        p_ids, p_segs = pred_maps(preds, width, height, valid)
        pq_frame(g_ids, g_segs, p_ids, p_segs, stat)
        g_cls = np.zeros(max(len(g_segs) + 1, 1), np.int64)
        for i, s in g_segs.items():
            g_cls[i] = s["category_id"]
        p_cls = np.zeros(len(preds) + 1, np.int64)
        for i, s in p_segs.items():
            p_cls[i] = s["category_id"]
        gc, pc = g_cls[g_ids], p_cls[p_ids]
        valid_px = gc > 0
        conf += np.bincount(gc[valid_px] * n_bins + pc[valid_px], minlength=n_bins * n_bins)
    stats, per_class = pq_summary(stat, isthing, names, things)
    stats_gt, _ = pq_summary(stat, isthing, names, things, only=gt_classes)

    conf = conf.reshape(n_bins, n_bins)
    tp = np.diag(conf)
    gt_px, pr_px = conf.sum(1), conf.sum(0)
    present = [c for c in range(1, n_bins)
               if gt_px[c] > 0 and (things is None or not isthing.get(c, False) or c in things)]
    ious = {names[c]: float(tp[c] / (gt_px[c] + pr_px[c] - tp[c])) for c in present}
    stats["mIoU"] = float(np.mean(list(ious.values()))) if ious else None
    stats["pixel_accuracy"] = float(tp[1:].sum() / gt_px[1:].sum()) if gt_px[1:].sum() else None
    stats_gt["mIoU"], stats_gt["pixel_accuracy"] = stats["mIoU"], stats["pixel_accuracy"]    # GT classes already
    out = {"metric": "PQ (COCO panopticapi), class average; mIoU over GT classes", "frames": len(frames),
            "image_size": [width, height], "stats": stats, "stats_gt_classes": stats_gt, "per_class_PQ": per_class,
            "per_class_IoU": ious,
            "gt_counts": gt.get("counts"), "predicted_segments": n_pred}
    if things is not None:
        out["things_evaluated"] = sorted(names[c] for c in things)
    return out
