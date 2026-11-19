"""End-to-end licence plate / race bib recognition: plate (bib) accuracy (primary) and character accuracy, plus
box AP. item = "plate" (plate_recognition, GT key "plate") or "bib" (bib_recognition, GT key "bib", digits_only:
predicted texts are reduced to their digits and predictions without digits are dropped before matching, so other
scene text such as sponsor logos is ignored). The stats keys use the item name (plate_accuracy / bib_accuracy).

Ground truth: <sequence_name>_lpr_annotations.json (annotations.load_lpr_annotations): per frame
{"track_id", "x_min", "x_max", "y_min", "y_max", "plate"}; "plate" = the visible characters (a plate cut by the
frame edge has only its visible part, e.g. "48"). Predictions: [score, x1, x2, y1, y2, text, ocr_confidence].

Matching, per frame: predictions in descending detector score; each takes the unmatched ground-truth plate with the
highest box IoU if that IoU >= iou_thr (config lpr_iou, default 0.5 as UFPR-ALPR: a detection is correct at IoU >= 0.5;
CarPark 0.3: its GT boxes are tighter vertically than the detector's), one-to-one.
Text comparison: normalise() both (upper case, spaces / hyphens / dots removed).

- plate_accuracy (primary): ground-truth plates whose matched prediction reads exactly the ground-truth text, divided
  by all ground-truth plates (a missed plate counts as wrong).
- char_accuracy: mean over all ground-truth plates of 1 - edit_distance(pred, gt) / max(len(pred), len(gt))
  (Levenshtein; a missed plate = 0).
- accuracy_detected: plate accuracy over the matched plates only (reading quality given detection).
- detection_recall: matched / ground-truth plates; false_plates: unmatched predictions (counted, not in the
  accuracies); detection AP / AP50 / AP75: pycocotools bbox AP of the plate boxes (one class, maxDets 100).
- per_track: plate accuracy per ground-truth track id.
"""

from __future__ import annotations

import contextlib
import io
import re

import numpy as np

from ..annotations import DetectionAnnotations

IOU_THR = 0.5
METRIC_VERSION = 1
_STRIP = re.compile(r"[\s\-.·]")


def normalise(text: str) -> str:
    """Text for comparison: upper case, spaces / hyphens / dots removed."""
    return _STRIP.sub("", str(text)).upper()


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance between two strings."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_score(pred: str, gt: str) -> float:
    """Character score 1 - edit_distance / max length (1.0 if both are empty)."""
    n = max(len(pred), len(gt))
    return 1.0 if n == 0 else 1.0 - edit_distance(pred, gt) / n


def _box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """[A, B] IoU of boxes [x1, x2, y1, y2] (any consistent unit)."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 1], b[None, :, 1]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 2], b[None, :, 2]), 0, None)
    inter = ix * iy
    aa = (a[:, 1] - a[:, 0]) * (a[:, 3] - a[:, 2])
    ab = (b[:, 1] - b[:, 0]) * (b[:, 3] - b[:, 2])
    union = aa[:, None] + ab[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def _detection_ap(ann: DetectionAnnotations, pred_frames: dict, frames: list[int], w: int, h: int) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images, anns, dets = [], [], []
    aid = 1
    for fn in frames:
        images.append({"id": fn, "width": w, "height": h})
        for o in ann.frames.get(fn, []):
            bw, bh = (o["x_max"] - o["x_min"]) * w, (o["y_max"] - o["y_min"]) * h
            anns.append({"id": aid, "image_id": fn, "category_id": 1, "bbox": [o["x_min"] * w, o["y_min"] * h, bw, bh],
                         "area": bw * bh, "iscrowd": 0})
            aid += 1
        for p in pred_frames.get(str(fn), []):
            dets.append({"image_id": fn, "category_id": 1, "score": float(p[0]),
                         "bbox": [p[1] * w, p[3] * h, (p[2] - p[1]) * w, (p[4] - p[3]) * h]})
    if not dets:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "licence plate"}]}
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.params.imgIds = frames
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {k: (float(v) if v > -1 else None) for k, v in zip(("AP", "AP50", "AP75"), ev.stats[:3])}


_NON_DIGIT = re.compile(r"\D")


def evaluate(ann: DetectionAnnotations, pred_frames: dict, frames: list[int], width: int, height: int,
             iou_thr: float = IOU_THR, item: str = "plate", digits_only: bool = False) -> dict:
    """Recognition accuracy (see module docstring) over the given frames; stats keys use the item name.

    Returns:
        Result dict with "stats" (<item>_accuracy, char_accuracy, accuracy_detected, detection_recall,
        det_AP / det_AP50 / det_AP75), counts and per-track accuracy.
    """
    n_gt = exact = matched = exact_matched = false = 0
    chars = []
    per_track: dict = {}
    n_pred = 0
    for fn in frames:
        gts = ann.frames.get(fn, [])
        preds = pred_frames.get(str(fn), [])
        if digits_only:
            preds = [list(p[:5]) + [_NON_DIGIT.sub("", str(p[5]))] + list(p[6:]) for p in preds]
            preds = [p for p in preds if p[5]]
        preds = sorted(preds, key=lambda p: -float(p[0]))
        n_pred += len(preds)
        gb = np.array([[o["x_min"], o["x_max"], o["y_min"], o["y_max"]] for o in gts], float).reshape(-1, 4) * \
            [width, width, height, height]
        pb = np.array([p[1:5] for p in preds], float).reshape(-1, 4) * [width, width, height, height]
        iou = _box_iou(pb, gb)
        taken = np.zeros(len(gts), bool)
        read = {}
        for i, p in enumerate(preds):
            if not len(gts):
                false += 1
                continue
            cand = np.where(taken, -1.0, iou[i])
            j = int(cand.argmax())
            if cand[j] >= iou_thr:
                taken[j] = True
                read[j] = normalise(p[5])
            else:
                false += 1
        for j, o in enumerate(gts):
            n_gt += 1
            g = normalise(o[item])
            ok = j in read and read[j] == g
            exact += ok
            chars.append(char_score(read[j], g) if j in read else 0.0)
            if j in read:
                matched += 1
                exact_matched += ok
            t = per_track.setdefault(str(o.get("track_id", "")), [0, 0])
            t[0] += ok
            t[1] += 1
    stats = {f"{item}_accuracy": exact / n_gt if n_gt else None,
             "char_accuracy": float(np.mean(chars)) if chars else None,
             "accuracy_detected": exact_matched / matched if matched else None,
             "detection_recall": matched / n_gt if n_gt else None}
    if digits_only:
        pred_frames = {f: [p for p in v if _NON_DIGIT.sub("", str(p[5]))] for f, v in pred_frames.items()}
    stats.update({f"det_{k}": v for k, v in _detection_ap(ann, pred_frames, frames, width, height).items()})
    return {"metric": f"end-to-end {item} accuracy (IoU >= {iou_thr} and exact text) over all ground-truth {item}s"
                      + ("; predicted texts reduced to digits, predictions without digits dropped" if digits_only
                         else ""),
            "iou_threshold": iou_thr, "digits_only": digits_only,
            "metric_version": METRIC_VERSION, "frames": len(frames), "image_size": [width, height],
            f"gt_{item}s": n_gt, "predictions": n_pred, "matched": matched, f"false_{item}s": false,
            "stats": stats,
            f"per_track_{item}_accuracy": {k: v[0] / v[1] for k, v in sorted(per_track.items())}}
