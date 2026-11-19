"""Detection F1 at IoU 0.5 at the best score threshold (an operating-point metric, unlike AP).

Per frame and class, detections (highest score first) are matched greedily to unmatched ground-truth boxes of the
same class with IoU >= 0.5. Detections matching a crowd region (iscrowd = 1) only are ignored. All detections
of all classes are then sorted by score; at each score threshold precision = TP / (TP + FP), recall = TP / GT, and
the reported F1 is the maximum over the thresholds (micro average: every ground-truth object weighs the same).
Only classes with ground truth are counted (other classes' detections are ignored, as in COCO AP).
Values in [0, 1]; result field "stats_f1": {F1, precision, recall, score_threshold, tp, fp, fn}.
"""

from __future__ import annotations

import numpy as np

IOU = 0.5


def _iou(a, b) -> float:
    ix = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[2], b[2]))
    inter = ix * iy
    union = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / union if union > 0 else 0.0


def evaluate_f1(ann, pred_frames: dict, frames: list[int], width: int, height: int) -> dict:
    """F1@0.5 at the best score threshold over the given frames (see the module docstring).

    Args:
        ann: detection annotations (frames {frame number: objects with class_id, x_min, x_max, y_min, y_max, iscrowd}).
        pred_frames: predictions {"<frame number>": [[class_id, score, x1, x2, y1, y2], ...]}, normalised.
        width, height: decoded frame size in pixels.

    Returns:
        {F1, precision, recall, score_threshold, tp, fp, fn} at the threshold with the highest F1.
    """
    gt_classes = {int(o["class_id"]) for fn in frames for o in ann.frames.get(fn, []) if not o["iscrowd"]}
    n_gt = 0
    scored = []                                   # (score, is_tp) of every counted detection
    for fn in frames:
        gts = [o for o in ann.frames.get(fn, []) if int(o["class_id"]) in gt_classes]
        boxes = [(o["x_min"] * width, o["x_max"] * width, o["y_min"] * height, o["y_max"] * height) for o in gts]
        crowd = [bool(o["iscrowd"]) for o in gts]
        n_gt += sum(1 for c in crowd if not c)
        used = [False] * len(gts)
        dets = [p for p in pred_frames.get(str(fn), []) if int(p[0]) in gt_classes]
        for p in sorted(dets, key=lambda p: -p[1]):
            d = (p[2] * width, p[3] * width, p[4] * height, p[5] * height)
            best, bi = IOU, None
            for i, (g, o) in enumerate(zip(boxes, gts)):
                if used[i] or crowd[i] or int(o["class_id"]) != int(p[0]):
                    continue
                v = _iou(d, g)
                if v >= best:
                    best, bi = v, i
            if bi is not None:
                used[bi] = True
                scored.append((float(p[1]), True))
            elif any(c and int(o["class_id"]) == int(p[0]) and _iou(d, g) >= IOU
                     for g, o, c in zip(boxes, gts, crowd)):
                continue                          # on a crowd region: ignored
            else:
                scored.append((float(p[1]), False))
    if not scored or n_gt == 0:
        return {"F1": 0.0, "precision": 0.0, "recall": 0.0, "score_threshold": None, "tp": 0,
                "fp": len(scored), "fn": n_gt}
    scored.sort(key=lambda s: -s[0])
    tp = np.cumsum([s[1] for s in scored])
    fp = np.cumsum([not s[1] for s in scored])
    prec, rec = tp / (tp + fp), tp / n_gt
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
    i = int(np.argmax(f1))
    return {"F1": float(f1[i]), "precision": float(prec[i]), "recall": float(rec[i]),
            "score_threshold": scored[i][0], "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(n_gt - tp[i])}
