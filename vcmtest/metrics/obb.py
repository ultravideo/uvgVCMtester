"""Oriented (rotated) object detection: COCO-style AP@[.50:.95] with exact rotated-polygon IoU (primary), and the
DOTA task1 mAP@0.5 (VOC07 11-point) for reference.

Ground truth: <sequence_name>_obb_annotations.json (annotations.load_obb_annotations): 4 corner points per object,
normalised; class ids DOTA v1.0 0-based. Predictions: [class_id, score, x1, y1, x2, y2, x3, y3, x4, y4] normalised.
Both are scaled to the decoded frame size. Polygons are used as given: ground-truth corners outside the frame (objects
cut by the frame edge, annotated with their full extent) are not clipped.

IoU = area(P ∩ Q) / area(P ∪ Q) of the two convex quadrilaterals (exact, Sutherland-Hodgman clipping); crowd ground
truth (iscrowd 1, ignore region): area(P ∩ Q) / area(P) as in pycocotools.

"stats" (primary): pycocotools COCOeval with computeIoU replaced by the polygon IoU. Everything else is COCOeval:
IoU thresholds .50:.05:.95, 101-point interpolated AP, class average over classes with ground truth, crowd = ignore,
area ranges (small < 32^2, medium < 96^2 px; area = polygon area). maxDets [1, 10, 300] instead of COCO's
[1, 10, 100]: frames have up to ~100 objects and Ultralytics OBB models return up to 300 boxes (max_det); AP and
AR300 use 300. "stats_object_weighted": object-weighted (micro) average, as for detection (coco_det).

"stats_dota": DOTA_devkit dota_evaluation_task1: per class, all detections sorted by score; a detection is a true
positive when its best-IoU ground truth (same frame, same class) has IoU >= 0.5 and is not yet matched; VOC07
11-point AP; mean over classes with ground truth. No "difficult" objects (crowd objects are left out of the
ground truth and detections matching them with IoU >= 0.5 are ignored).
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from ..annotations import DetectionAnnotations, dota_label_map
from .coco_det import object_weighted_stats

VEHICLE_CLASSES = (9, 10)        # large vehicle, small vehicle (DOTA v1.0)
VEHICLE_ID = 10                  # merged "vehicle" class keeps the small-vehicle id

STAT_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR300", "ARs", "ARm", "ARl"]
MAX_DETS = [1, 10, 300]
METRIC_VERSION = 1


# ----------------------------------------------------------------------------- polygon geometry

def poly_area(p: np.ndarray) -> float:
    """Area of a polygon given as an [N, 2] vertex array (shoelace formula)."""
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ccw(p: np.ndarray) -> np.ndarray:
    """Vertices in counter-clockwise order (mathematical orientation)."""
    x, y = p[:, 0], p[:, 1]
    s = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
    return p if s >= 0 else p[::-1]


def _left_of(q, ax, ay, bx, by) -> bool:
    """q on the left of (or on) the directed edge a -> b."""
    return (bx - ax) * (q[1] - ay) - (by - ay) * (q[0] - ax) >= 0


def _edge_hit(p, q, ax, ay, bx, by) -> tuple:
    """Intersection of segment p-q with the line through a and b."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    den = (bx - ax) * dy - (by - ay) * dx
    t = ((p[0] - ax) * dy - (p[1] - ay) * dx) / den if den else 0.0   # p + s d = a + t (b - a)
    return (ax + t * (bx - ax), ay + t * (by - ay))


def _clip(subject: list, clipper: np.ndarray) -> list:
    """Sutherland-Hodgman: subject polygon clipped by a convex ccw clipper polygon."""
    out = subject
    n = len(clipper)
    for i in range(n):
        if not out:
            break
        ax, ay = clipper[i]
        bx, by = clipper[(i + 1) % n]
        inp, out = out, []
        prev = inp[-1]
        for cur in inp:
            if _left_of(cur, ax, ay, bx, by):
                if not _left_of(prev, ax, ay, bx, by):
                    out.append(_edge_hit(prev, cur, ax, ay, bx, by))
                out.append(cur)
            elif _left_of(prev, ax, ay, bx, by):
                out.append(_edge_hit(prev, cur, ax, ay, bx, by))
            prev = cur
    return out


def intersection_area(p: np.ndarray, q: np.ndarray) -> float:
    """Area of the intersection of two convex polygons (N x 2 arrays)."""
    r = _clip([tuple(v) for v in _ccw(p)], _ccw(q))
    return poly_area(np.array(r)) if len(r) >= 3 else 0.0


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def poly_areas(p: np.ndarray) -> np.ndarray:
    """[N] areas of N polygons given as an [N, K, 2] array."""
    return 0.5 * np.abs(_cross(p, np.roll(p, -1, axis=1)).sum(1))


def intersection_areas(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """[N] intersection areas of N pairs of convex quadrilaterals p[i], q[i] ([N, 4, 2] arrays), vectorised.

    The intersection of two convex polygons is the convex polygon spanned by the vertices of each that lie inside
    the other plus the edge-edge crossing points; its area = shoelace over these points sorted by angle around their
    centroid. Exact up to floating point (cross-checked against the Sutherland-Hodgman clip, intersection_area)."""
    n = len(p)
    if n == 0:
        return np.zeros(0)
    p = np.where((_cross(p, np.roll(p, -1, 1)).sum(1) < 0)[:, None, None], p[:, ::-1], p)   # counter-clockwise
    q = np.where((_cross(q, np.roll(q, -1, 1)).sum(1) < 0)[:, None, None], q[:, ::-1], q)
    scale = np.maximum(np.abs(p).max((1, 2)), np.abs(q).max((1, 2)))[:, None, None] + 1.0
    eps = 1e-9 * scale ** 2

    def inside(pts, poly):                              # pts [N, M, 2] in convex ccw poly [N, 4, 2] -> [N, M]
        a, b = poly, np.roll(poly, -1, 1)
        c = _cross((b - a)[:, None], pts[:, :, None] - a[:, None])          # [N, M, 4]
        return (c >= -eps).all(2)

    p0, p1 = p, np.roll(p, -1, 1)
    q0, q1 = q, np.roll(q, -1, 1)
    r = (p1 - p0)[:, :, None]                            # [N, 4, 1, 2]
    sv = (q1 - q0)[:, None]                              # [N, 1, 4, 2]
    d = q0[:, None] - p0[:, :, None]                     # [N, 4, 4, 2]
    den = _cross(r, sv)
    ok = np.abs(den) > eps
    den_s = np.where(ok, den, 1.0)
    t = _cross(d, sv) / den_s
    u = _cross(d, r) / den_s
    hit = ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    xpts = (p0[:, :, None] + t[..., None] * r).reshape(n, 16, 2)
    pts = np.concatenate([p, q, xpts], 1)                # [N, 24, 2]
    mask = np.concatenate([inside(p, q), inside(q, p), hit.reshape(n, 16)], 1)
    cnt = mask.sum(1)
    c = (pts * mask[..., None]).sum(1) / np.maximum(cnt, 1)[:, None]
    ang = np.arctan2(pts[..., 1] - c[:, None, 1], pts[..., 0] - c[:, None, 0])
    ang = np.where(mask, ang, np.inf)
    order = np.argsort(ang, 1)
    sp = np.take_along_axis(pts, order[..., None], 1)
    sm = np.take_along_axis(mask, order, 1)
    sp = np.where(sm[..., None], sp, sp[:, :1])          # padding = the first point: zero-area closing segments
    area = 0.5 * np.abs(_cross(sp, np.roll(sp, -1, 1)).sum(1))
    return np.where(cnt >= 3, area, 0.0)


def iou_matrix(dets, gts, crowd=None) -> np.ndarray:
    """[D, G] polygon IoU of D detection and G ground-truth quadrilaterals (lists of 4 x 2 arrays or [D, 4, 2] /
    [G, 4, 2] arrays); crowd ground truth: intersection / detection area. Only pairs whose axis-aligned extents
    overlap are intersected (vectorised over all such pairs)."""
    D, G = len(dets), len(gts)
    out = np.zeros((D, G))
    if not D or not G:
        return out
    dp = np.asarray(dets, float).reshape(D, 4, 2)
    gp = np.asarray(gts, float).reshape(G, 4, 2)
    crowd = np.zeros(G, bool) if crowd is None else np.asarray(crowd, bool)
    dmin, dmax, gmin, gmax = dp.min(1), dp.max(1), gp.min(1), gp.max(1)
    ov = ((dmin[:, None, 0] < gmax[None, :, 0]) & (gmin[None, :, 0] < dmax[:, None, 0])
          & (dmin[:, None, 1] < gmax[None, :, 1]) & (gmin[None, :, 1] < dmax[:, None, 1]))
    i, j = np.nonzero(ov)
    if not len(i):
        return out
    inter = intersection_areas(dp[i], gp[j])
    da, ga = poly_areas(dp), poly_areas(gp)
    union = np.where(crowd[j], da[i], da[i] + ga[j] - inter)
    out[i, j] = np.where(union > 0, inter / np.where(union > 0, union, 1.0), 0.0)
    return out


# ----------------------------------------------------------------------------- COCO-style AP

def _gt_polys(ann: DetectionAnnotations, frames: list[int], w: int, h: int) -> dict:
    return {fn: [(int(o["class_id"]), np.asarray(o["bbox"], float) * [w, h], int(o.get("iscrowd", 0)))
                 for o in ann.frames.get(fn, [])] for fn in frames}


def _det_polys(pred_frames: dict, frames: list[int], w: int, h: int) -> tuple[dict, int]:
    valid = set(dota_label_map().values())
    out, dropped = {}, 0
    for fn in frames:
        lst = []
        for p in pred_frames.get(str(fn), []):
            if int(p[0]) not in valid:
                dropped += 1
                continue
            lst.append((int(p[0]), float(p[1]), np.asarray(p[2:10], float).reshape(4, 2) * [w, h]))
        out[fn] = lst
    return out, dropped


def _cocoeval(gt: dict, dt: dict, frames: list[int], w: int, h: int):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    cats = [{"id": cid, "name": n} for n, cid in sorted(dota_label_map().items(), key=lambda x: x[1])]
    images, anns, dets = [], [], []
    aid = 1
    for fn in frames:
        images.append({"id": fn, "width": w, "height": h, "file_name": str(fn)})
        for cid, poly, crowd in gt[fn]:
            x0, y0 = poly.min(0)
            x1, y1 = poly.max(0)
            anns.append({"id": aid, "image_id": fn, "category_id": cid, "bbox": [x0, y0, x1 - x0, y1 - y0],
                         "area": poly_area(poly), "iscrowd": crowd, "poly": poly})
            aid += 1
        for cid, score, poly in dt[fn]:
            x0, y0 = poly.min(0)
            x1, y1 = poly.max(0)
            dets.append({"image_id": fn, "category_id": cid, "score": score,
                         "bbox": [x0, y0, x1 - x0, y1 - y0], "poly": poly})

    class PolyEval(COCOeval):
        def computeIoU(self, imgId, catId):
            p = self.params
            g = self._gts[imgId, catId] if p.useCats else [x for c in p.catIds for x in self._gts[imgId, c]]
            d = self._dts[imgId, catId] if p.useCats else [x for c in p.catIds for x in self._dts[imgId, c]]
            if not g or not d:
                return []
            inds = np.argsort([-x["score"] for x in d], kind="mergesort")
            d = [d[i] for i in inds][:p.maxDets[-1]]
            return iou_matrix([x["poly"] for x in d], [x["poly"] for x in g], [int(x["iscrowd"]) for x in g])

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = {"images": images, "annotations": anns, "categories": cats}
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        for a in coco_dt.dataset["annotations"]:           # loadRes sets area = box area: use the polygon area
            a["area"] = poly_area(a["poly"])
        ev = PolyEval(coco_gt, coco_dt, "bbox")
        ev.params.imgIds = frames
        ev.params.maxDets = list(MAX_DETS)
        ev.evaluate()
        ev.accumulate()
    return ev


def _summary(ev) -> list[float]:
    """COCOeval.summarize() values for STAT_NAMES, but with maxDets 300 for AP (summarize() hard-codes 100 for the
    first AP value, which is then -1 with maxDets [1, 10, 300])."""
    p = ev.params

    def one(ap: bool, iou=None, area="all", md=MAX_DETS[-1]) -> float:
        a = [i for i, r in enumerate(p.areaRngLbl) if r == area]
        m = [i for i, v in enumerate(p.maxDets) if v == md]
        s = ev.eval["precision"] if ap else ev.eval["recall"]
        if iou is not None:
            s = s[np.where(np.isclose(p.iouThrs, iou))[0]]
        s = s[..., a, m] if ap else s[:, :, a, m]
        s = s[s > -1]
        return float(np.mean(s)) if s.size else -1.0
    return [one(True), one(True, .5), one(True, .75), one(True, area="small"), one(True, area="medium"),
            one(True, area="large"), one(False, md=MAX_DETS[0]), one(False, md=MAX_DETS[1]), one(False),
            one(False, area="small"), one(False, area="medium"), one(False, area="large")]


def dota_voc07_map(gt: dict, dt: dict, frames: list[int], thr: float = 0.5) -> tuple[float | None, dict]:
    """DOTA task1 mAP (VOC07 11-point) at IoU >= thr; (mean over classes with ground truth, per-class AP)."""
    names = {cid: n for n, cid in dota_label_map().items()}
    classes = sorted({c for fn in frames for c, _, crowd in gt[fn] if not crowd})
    per = {}
    for c in classes:
        gts = {fn: [p for cc, p, crowd in gt[fn] if cc == c and not crowd] for fn in frames}
        crowds = {fn: [p for cc, p, crowd in gt[fn] if cc == c and crowd] for fn in frames}
        npos = sum(len(v) for v in gts.values())
        ds, ious, ciou = [], {}, {}
        for fn in frames:                                 # one IoU matrix per frame (detections x GT)
            dd = [(s, p) for cc, s, p in dt[fn] if cc == c]
            if not dd:
                continue
            pp = [p for _, p in dd]
            ious[fn] = iou_matrix(pp, gts[fn])
            ciou[fn] = iou_matrix(pp, crowds[fn], [1] * len(crowds[fn]))
            ds += [(s, fn, k) for k, (s, _) in enumerate(dd)]
        ds.sort(key=lambda x: -x[0])
        used = {fn: np.zeros(len(gts[fn]), bool) for fn in frames}
        tp, fp = [], []
        for _s, fn, k in ds:
            row = ious[fn][k]
            if row.size and row.max() >= thr:
                j = int(row.argmax())
                if not used[fn][j]:
                    used[fn][j] = True
                    tp.append(1)
                    fp.append(0)
                else:
                    tp.append(0)
                    fp.append(1)
            elif ciou[fn].shape[1] and ciou[fn][k].max() >= thr:
                continue                                      # matches an ignore region
            else:
                tp.append(0)
                fp.append(1)
        tpc, fpc = np.cumsum(tp), np.cumsum(fp)
        rec = tpc / npos if len(tpc) else np.zeros(0)
        prec = tpc / np.maximum(tpc + fpc, np.finfo(float).eps) if len(tpc) else np.zeros(0)
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            ap += (prec[rec >= t].max() if np.any(rec >= t) else 0.0) / 11.0
        per[names[c]] = float(ap)
    return (float(np.mean(list(per.values()))) if per else None), per


def merge_vehicles(ann: DetectionAnnotations, pred_frames: dict) -> tuple[DetectionAnnotations, dict]:
    """Ground truth and predictions with large vehicle (9) and small vehicle (10) as one class (id 10)."""
    from dataclasses import replace
    gt = {fn: [dict(o, class_id=VEHICLE_ID) if int(o["class_id"]) in VEHICLE_CLASSES else o for o in objs]
          for fn, objs in ann.frames.items()}
    pf = {k: [[VEHICLE_ID] + list(p[1:]) if int(p[0]) in VEHICLE_CLASSES else p for p in v]
          for k, v in pred_frames.items()}
    return replace(ann, frames=gt), pf


def evaluate(ann: DetectionAnnotations, pred_frames: dict, frames: list[int], width: int, height: int) -> dict:
    """Rotated-box COCO AP (polygon IoU) and DOTA mAP@0.5 (VOC07) over the given frames.

    Overall and per class; AP values in [0, 1].
    """
    gt = _gt_polys(ann, frames, width, height)
    dt, dropped = _det_polys(pred_frames, frames, width, height)
    n_gt = sum(1 for fn in frames for _, _, crowd in gt[fn] if not crowd)
    n_dt = sum(len(v) for v in dt.values())
    result = {"metric": "rotated-box AP@[.50:.95] (pycocotools COCOeval with polygon IoU, maxDets 300)",
              "metric_version": METRIC_VERSION, "frames": len(frames), "image_size": [width, height],
              "gt_objects": n_gt, "gt_crowd_regions": sum(len(gt[fn]) for fn in frames) - n_gt,
              "gt_corners_outside_frame": sum(1 for fn in frames for _, p, _ in gt[fn]
                                              if (p < 0).any() or (p[:, 0] > width).any() or (p[:, 1] > height).any()),
              "detections": n_dt, "detections_dropped": dropped}
    if not n_dt:
        result["stats"] = {k: 0.0 for k in STAT_NAMES}
        result["stats_object_weighted"] = {k: 0.0 for k in ("AP", "AP50", "AP75", "APs", "APm", "APl")}
        result["per_class_AP"] = {}
        result["stats_dota"] = {"mAP50_voc07": 0.0}
        result["per_class_AP50_voc07"] = {}
        result["note"] = "no detections"
        return result
    ev = _cocoeval(gt, dt, frames, width, height)
    result["stats"] = {k: (float(v) if v > -1 else None) for k, v in zip(STAT_NAMES, _summary(ev))}
    names = {cid: n for n, cid in dota_label_map().items()}
    per = {}
    for k, cid in enumerate(ev.params.catIds):
        p = ev.eval["precision"][:, :, k, 0, -1]
        p = p[p > -1]
        if p.size:
            per[names[cid]] = float(np.mean(p))
    result["per_class_AP"] = per
    result["stats_object_weighted"] = object_weighted_stats(ev)
    m, per07 = dota_voc07_map(gt, dt, frames)
    result["stats_dota"] = {"mAP50_voc07": m}
    result["per_class_AP50_voc07"] = per07
    return result
