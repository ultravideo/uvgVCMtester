"""Multi-object tracking: HOTA (primary), CLEAR MOT (MOTA, MOTP, ...) and Identity (IDF1) with TrackEval.

Follows TrackEval's MOTChallenge 2D-box procedure, applied per class:
- Ground truth = objects of the class with a "track_id". Crowd objects (iscrowd=1) and objects without a
  track_id are distractors (ignore regions).
- Per frame, tracker boxes are matched to all ground-truth boxes of the class, distractors included
  (Hungarian, IoU >= 0.5); tracker boxes matched to a distractor are removed, then the distractors are removed.
- Similarity = box IoU at the decoded resolution; HOTA over alpha 0.05..0.95, CLEAR/Identity at IoU 0.5.
- Classes evaluated = classes with ground-truth tracks; tracker boxes of other classes are ignored.
  The overall value is the class average (TrackEval combine_classes_class_averaged), like COCO AP: "stats".
  Also stored: "stats_object_weighted" = TrackEval combine_classes_det_averaged (the classes' TP / FN / FP and
  association counts summed, then HOTA, MOTA, IDF1 computed once: each class weighs by its detections; matching
  stays per class). Selected for plotting by the config option hota_average: object.
- Within a frame, a (class, track_id) that appears twice is resolved for tracking only: if the two boxes overlap
  (IoU >= 0.5) the first is kept and a warning is returned; otherwise it is an error.

Predictions: per frame [class_id, score, x1, x2, y1, y2, track_id] (normalised coordinates); frames in order.
"""

from __future__ import annotations

import contextlib
import io
import logging

import numpy as np

from ..annotations import AnnotationError, DetectionAnnotations, coco_label_map

log = logging.getLogger("vcmtest")

DUPLICATE_IOU = 0.5
STAT_KEYS = ["HOTA", "DetA", "AssA", "LocA", "DetRe", "DetPr", "AssRe", "AssPr",
             "MOTA", "MOTP", "IDF1", "IDR", "IDP", "IDSW", "Frag", "MT", "PT", "ML",
             "CLR_TP", "CLR_FN", "CLR_FP", "IDTP", "IDFN", "IDFP"]


def _iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def _box(o_or_p, w: int, h: int, pred: bool) -> list[float]:
    if pred:
        _, _, x1, x2, y1, y2 = o_or_p[:6]
    else:
        x1, x2, y1, y2 = o_or_p["x_min"], o_or_p["x_max"], o_or_p["y_min"], o_or_p["y_max"]
    return [x1 * w, y1 * h, (x2 - x1) * w, (y2 - y1) * h]


def ground_truth(ann: DetectionAnnotations, frames: list[int], w: int, h: int) -> tuple[dict, list[str]]:
    """Per frame: {class_id: (track_ids[N], boxes[N,4], boxes_distractor[M,4])}; plus warnings."""
    out, warnings = {}, []
    for fn in frames:
        per_cls: dict[int, tuple[list, list, list]] = {}
        seen: dict[tuple[int, int], list[float]] = {}
        for o in ann.frames.get(fn, []):
            cid = int(o["class_id"])
            ids, boxes, distr = per_cls.setdefault(cid, ([], [], []))
            b = _box(o, w, h, pred=False)
            if o["iscrowd"] or "track_id" not in o:
                distr.append(b)
                continue
            key = (cid, int(o["track_id"]))
            if key in seen:
                iou = float(_iou_xywh(np.array([seen[key]]), np.array([b]))[0, 0])
                if iou >= DUPLICATE_IOU:
                    warnings.append(f"frame {fn}: duplicate track_id {key[1]} (class {cid}, IoU {iou:.2f}); "
                                    f"kept the first box for tracking")
                    continue
                raise AnnotationError(f"frame {fn}: track_id {key[1]} (class {cid}) used for two different objects "
                                      f"(IoU {iou:.2f}); fix the annotations")
            seen[key] = b
            ids.append(key[1])
            boxes.append(b)
        out[fn] = {c: (np.array(i, int), np.array(b, float).reshape(-1, 4), np.array(d, float).reshape(-1, 4))
                   for c, (i, b, d) in per_cls.items()}
    return out, warnings


def _class_data(gt: dict, preds: dict, frames: list[int], cid: int, w: int, h: int) -> dict:
    """TrackEval input for one class, after MOTChallenge-style distractor removal."""
    from scipy.optimize import linear_sum_assignment
    data = {"gt_ids": [], "tracker_ids": [], "similarity_scores": []}
    for fn in frames:
        g_ids, g_boxes, d_boxes = gt[fn].get(cid, (np.zeros(0, int), np.zeros((0, 4)), np.zeros((0, 4))))
        pl = [p for p in preds.get(str(fn), []) if int(p[0]) == cid]
        t_ids = np.array([int(p[6]) for p in pl], int)
        t_boxes = np.array([_box(p, w, h, pred=True) for p in pl], float).reshape(-1, 4)
        if len(d_boxes) and len(t_boxes):
            all_gt = np.vstack([g_boxes, d_boxes])
            sim = _iou_xywh(all_gt, t_boxes)
            sim[sim < 0.5 - np.finfo(float).eps] = 0
            rows, cols = linear_sum_assignment(-sim)
            ok = sim[rows, cols] > np.finfo(float).eps
            rows, cols = rows[ok], cols[ok]
            drop = cols[rows >= len(g_boxes)]          # matched to a distractor
            t_ids, t_boxes = np.delete(t_ids, drop), np.delete(t_boxes, drop, axis=0)
        data["gt_ids"].append(g_ids)
        data["tracker_ids"].append(t_ids)
        data["similarity_scores"].append(_iou_xywh(g_boxes, t_boxes))
    # relabel ids to 0..n-1 (TrackEval requirement)
    for key, n_key, d_key in (("gt_ids", "num_gt_ids", "num_gt_dets"),
                              ("tracker_ids", "num_tracker_ids", "num_tracker_dets")):
        uniq = np.unique(np.concatenate(data[key])) if any(len(x) for x in data[key]) else np.zeros(0, int)
        remap = {int(u): i for i, u in enumerate(uniq)}
        data[key] = [np.array([remap[int(i)] for i in ids], int) for ids in data[key]]
        data[n_key] = len(uniq)
        data[d_key] = int(sum(len(x) for x in data[key]))
    for t in range(len(frames)):
        if len(np.unique(data["tracker_ids"][t])) != len(data["tracker_ids"][t]):
            raise AnnotationError(f"frame {frames[t]}: tracker output has duplicate track ids for class {cid}")
    data["num_timesteps"] = len(frames)
    data["seq"] = "seq"
    return data


def _flatten(res: dict) -> dict:
    out = {}
    for metric in res.values():
        for k, v in metric.items():
            if k in STAT_KEYS:
                out[k] = float(np.mean(v)) if np.ndim(v) else float(v)
    return {k: out.get(k) for k in STAT_KEYS}


def evaluate(ann: DetectionAnnotations, pred_frames: dict, frames: list[int], width: int, height: int) -> dict:
    """Returns {"stats": {...class average...}, "per_class": {name: stats}, "warnings": [...]}."""
    from trackeval.metrics import CLEAR, HOTA, Identity

    gt, warnings = ground_truth(ann, frames, width, height)
    classes = sorted({c for fr in gt.values() for c, (ids, _, _) in fr.items() if len(ids)})
    names = {cid: name for name, cid in coco_label_map().items()}
    with contextlib.redirect_stdout(io.StringIO()):
        metrics = [HOTA(), CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}), Identity({"THRESHOLD": 0.5,
                                                                                      "PRINT_CONFIG": False})]
    per_class_res = {}
    for cid in classes:
        data = _class_data(gt, pred_frames, frames, cid, width, height)
        per_class_res[cid] = {m.get_name(): m.eval_sequence(data) for m in metrics}
        per_class_res[cid]["_counts"] = {"gt_tracks": data["num_gt_ids"], "tracker_tracks": data["num_tracker_ids"],
                                         "gt_dets": data["num_gt_dets"], "tracker_dets": data["num_tracker_dets"]}
    combined, combined_det = {}, {}
    if per_class_res:
        for m in metrics:
            per = {c: r[m.get_name()] for c, r in per_class_res.items()}
            combined[m.get_name()] = m.combine_classes_class_averaged(per)
            combined_det[m.get_name()] = m.combine_classes_det_averaged(per)
    result = {"metric": "HOTA (TrackEval), class average", "frames": len(frames), "image_size": [width, height],
              "classes": [names.get(c, str(c)) for c in classes],
              "stats": _flatten(combined) if combined else {k: None for k in STAT_KEYS},
              "stats_object_weighted": _flatten(combined_det) if combined_det else {k: None for k in STAT_KEYS},
              "per_class": {names.get(c, str(c)): {**_flatten({k: v for k, v in r.items() if k != "_counts"}),
                                                   **r["_counts"]} for c, r in per_class_res.items()},
              "warnings": warnings}
    return result
