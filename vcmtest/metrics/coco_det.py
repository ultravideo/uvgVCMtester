"""Object detection: COCO-style AP averaged over IoU 0.50:0.95 (pycocotools defaults).

- Images = annotated frames; image size from the decoded frames (normalised -> pixel boxes).
- Ground-truth area = box area in pixels (no masks/polygons are used for detection), so the COCO
  small/medium/large ranges refer to box area at the decoded resolution.
- iscrowd=1 objects are COCO ignore regions (standard pycocotools behaviour).
- All 80 categories are passed; pycocotools averages only over categories that have ground truth.
- maxDets [1, 10, 100]; detections beyond 100 per image are ignored by COCOeval.
- Two averages over classes are reported:
  "stats": COCO standard, AP per class then the mean over classes (class average);
  "stats_object_weighted": object-weighted (micro) average: matching is unchanged (same IoU rules, a detection
  only matches ground truth of its own class, crowd = ignore), but the detections of all classes are pooled into
  one precision-recall curve per IoU threshold and area range, so every ground-truth object weighs equally.
  Only classes with (non-ignored) ground truth in that area range enter the curve, as in the COCO class average:
  detections of classes absent from the ground truth are ignored, not counted as false positives.
  With a single class both are identical.
"""

from __future__ import annotations

import contextlib
import io
import logging

import numpy as np

from ..annotations import DetectionAnnotations, coco_label_map

log = logging.getLogger("vcmtest")

STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
OBJECT_STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl"]


def to_coco_gt(ann: DetectionAnnotations, frames: list[int], width: int, height: int) -> dict:
    """COCO ground-truth dict of the given frames: pixel boxes, area = box area, all 80 categories."""
    cats = [{"id": cid, "name": name} for name, cid in sorted(coco_label_map().items(), key=lambda x: x[1])]
    images, anns = [], []
    aid = 1
    for fn in frames:
        images.append({"id": fn, "width": width, "height": height, "file_name": str(fn)})
        for o in ann.frames.get(fn, []):
            x, y = o["x_min"] * width, o["y_min"] * height
            w, h = (o["x_max"] - o["x_min"]) * width, (o["y_max"] - o["y_min"]) * height
            anns.append({"id": aid, "image_id": fn, "category_id": int(o["class_id"]),
                         "bbox": [x, y, w, h], "area": w * h, "iscrowd": int(o["iscrowd"])})
            aid += 1
    return {"images": images, "annotations": anns, "categories": cats}


def to_coco_dets(pred_frames: dict, frames: list[int], width: int, height: int) -> tuple[list[dict], int]:
    """Predictions [class_id, score, x1, x2, y1, y2] (normalised) -> COCO detection results in pixels.

    Returns:
        (detections, number of predictions dropped because their class id is not in labels.py).
    """
    valid = set(coco_label_map().values())
    dets, dropped = [], 0
    for fn in frames:
        for p in pred_frames.get(str(fn), []):
            cid, score, x1, x2, y1, y2 = p[:6]
            if cid not in valid:
                dropped += 1
                continue
            dets.append({"image_id": fn, "category_id": int(cid), "score": float(score),
                         "bbox": [x1 * width, y1 * height, (x2 - x1) * width, (y2 - y1) * height]})
    return dets, dropped


def evaluate(ann: DetectionAnnotations, pred_frames: dict, frames: list[int], width: int, height: int) -> dict:
    """Returns {"stats": {AP, AP50, ...}, "per_class_AP": {...}, ...}; AP values in [0, 1]."""
    gt_dict = to_coco_gt(ann, frames, width, height)
    dets, dropped = to_coco_dets(pred_frames, frames, width, height)
    if dropped:
        log.warning(f"{dropped} detections with class ids outside labels.py were ignored")
    n_gt = sum(1 for a in gt_dict["annotations"] if not a["iscrowd"])
    result = {"metric": "COCO bbox AP@[.50:.95] (pycocotools)", "frames": len(frames),
              "image_size": [width, height], "gt_objects": n_gt,
              "gt_crowd_regions": len(gt_dict["annotations"]) - n_gt,
              "detections": len(dets), "detections_dropped": dropped}
    if not dets:  # pycocotools cannot load an empty result list
        result["stats"] = {k: 0.0 for k in STAT_NAMES}
        result["stats_object_weighted"] = {k: 0.0 for k in OBJECT_STAT_NAMES}
        result["per_class_AP"] = {}
        result["note"] = "no detections"
        return result

    result["stats"], result["per_class_AP"], result["stats_object_weighted"] = run_cocoeval(gt_dict, dets, frames,
                                                                                           "bbox")
    return result


def run_cocoeval(gt_dict: dict, dets: list[dict], frames: list[int], iou_type: str) -> tuple[dict, dict, dict]:
    """pycocotools COCOeval ('bbox' or 'segm') -> (stats, per-class AP, object-weighted stats); dets not empty."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with contextlib.redirect_stdout(io.StringIO()):  # silence pycocotools prints
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()
        dt = gt.loadRes(dets)
        ev = COCOeval(gt, dt, iou_type)
        ev.params.imgIds = frames
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    # stats of -1 mean "no ground truth in this area range"; keep them as None
    stats = {k: (float(v) if v > -1 else None) for k, v in zip(STAT_NAMES, ev.stats)}

    names = {cid: name for name, cid in coco_label_map().items()}
    precision = ev.eval["precision"]  # [T, R, K, A, M]
    per_class = {}
    for k, cid in enumerate(ev.params.catIds):
        p = precision[:, :, k, 0, -1]
        p = p[p > -1]
        if p.size:
            per_class[names[cid]] = float(np.mean(p))
    return stats, per_class, object_weighted_stats(ev)


def object_weighted_stats(ev) -> dict:
    """AP pooled over all classes (micro average) from an evaluated COCOeval (after evaluate()).

    Same per-image, per-class matching as COCOeval (evalImgs), and the same accumulation as COCOeval.accumulate
    (score sort, 101-point interpolated precision, maxDets 100 per image and class), except that the detections and
    ground truth of all categories go into one curve. Categories without non-ignored ground truth in the area range
    are left out (COCO leaves them out of the class average too). None when there is no such ground truth."""
    p = ev.params
    n_img, n_area = len(p.imgIds), len(p.areaRng)
    max_det = p.maxDets[-1]
    rec_thrs = p.recThrs
    iou_thrs = p.iouThrs

    def ap_curve(a: int) -> np.ndarray | None:
        """[T] AP per IoU threshold for area range index a (None if no ground truth)."""
        es = []
        for k in range(len(p.catIds)):
            ek = [e for e in ev.evalImgs[k * n_area * n_img + a * n_img:k * n_area * n_img + (a + 1) * n_img]
                  if e is not None]
            if any(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0) for e in ek):
                es += ek                            # only classes with ground truth in this area range
        if not es:
            return None
        gt_ig = np.concatenate([e["gtIgnore"] for e in es])
        npig = int(np.count_nonzero(gt_ig == 0))
        if npig == 0:
            return None
        scores = np.concatenate([e["dtScores"][:max_det] for e in es])
        order = np.argsort(-scores, kind="mergesort")
        dtm = np.concatenate([e["dtMatches"][:, :max_det] for e in es], axis=1)[:, order]
        dt_ig = np.concatenate([e["dtIgnore"][:, :max_det] for e in es], axis=1)[:, order]
        tps = np.logical_and(dtm, np.logical_not(dt_ig))
        fps = np.logical_and(np.logical_not(dtm), np.logical_not(dt_ig))
        tp_sum = np.cumsum(tps, axis=1).astype(float)
        fp_sum = np.cumsum(fps, axis=1).astype(float)
        out = np.zeros(len(iou_thrs))
        for t in range(len(iou_thrs)):
            tp, fp = tp_sum[t], fp_sum[t]
            rc = tp / npig
            pr = (tp / (fp + tp + np.spacing(1))).tolist()
            for i in range(len(pr) - 1, 0, -1):          # precision envelope, as in COCOeval.accumulate
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]
            q = np.zeros(len(rec_thrs))
            idx = np.searchsorted(rc, rec_thrs, side="left")
            for ri, pi in enumerate(idx):
                if pi < len(pr):
                    q[ri] = pr[pi]
            out[t] = q.mean()
        return out

    stats = {}
    curves = {a: ap_curve(a) for a in range(n_area)}       # area ranges: all, small, medium, large
    t50 = int(np.argmin(np.abs(iou_thrs - 0.5)))
    t75 = int(np.argmin(np.abs(iou_thrs - 0.75)))
    c = curves[0]
    stats["AP"] = None if c is None else float(c.mean())
    stats["AP50"] = None if c is None else float(c[t50])
    stats["AP75"] = None if c is None else float(c[t75])
    for name, a in (("APs", 1), ("APm", 2), ("APl", 3)):
        stats[name] = None if curves.get(a) is None else float(curves[a].mean())
    return stats
