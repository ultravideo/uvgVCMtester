"""Keypoint detection: COCO keypoint AP (OKS), pycocotools iouType "keypoints" with the standard COCO settings.

- Images = annotated frames; image size from the decoded frames (normalised -> pixel coordinates).
- Ground truth: 17 COCO keypoints per person (x, y, v). OKS uses only labelled keypoints (v > 0) and the person's
  segmentation area ("area" in the JSON = fraction of the image area, scaled by W x H). Persons with no labelled
  keypoints and iscrowd=1 persons are ignore regions (standard pycocotools).
- The GT box given to pycocotools is the extent of the labelled keypoints; pycocotools only uses it for persons
  without labelled keypoints, which are ignored anyway.
- Predictions: per person a score and 17 keypoints; pycocotools takes the box and area of a prediction from its
  keypoints. The per-keypoint scores are not used by the metric.
- Standard COCO keypoint sigmas, OKS thresholds 0.50:0.95, maxDets 20, area ranges all / medium (32^2-96^2) /
  large (> 96^2) in pixels at the decoded resolution.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np

from ..annotations import N_KEYPOINTS, DetectionAnnotations

KP_STAT_NAMES = ["AP", "AP50", "AP75", "APm", "APl", "AR", "AR50", "AR75", "ARm", "ARl"]
KEYPOINT_NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
                  "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
                  "right_knee", "left_ankle", "right_ankle"]
SKELETON = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10],
            [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]


def to_coco_gt(ann: DetectionAnnotations, frames: list[int], width: int, height: int) -> dict:
    """COCO keypoint ground-truth dict of the given frames.

    Pixel keypoints (unlabelled = 0, 0, 0), box = extent of the labelled keypoints, area = JSON area x W x H.
    """
    images, anns = [], []
    aid = 1
    for fn in frames:
        images.append({"id": fn, "width": width, "height": height, "file_name": str(fn)})
        for o in ann.frames.get(fn, []):
            k = np.asarray(o["keypoints"], float).reshape(N_KEYPOINTS, 3)
            v = k[:, 2] > 0
            kp = np.zeros((N_KEYPOINTS, 3))
            kp[v, 0], kp[v, 1], kp[v, 2] = k[v, 0] * width, k[v, 1] * height, k[v, 2]
            if v.any():
                x0, y0 = kp[v, 0].min(), kp[v, 1].min()
                bbox = [x0, y0, kp[v, 0].max() - x0, kp[v, 1].max() - y0]
            else:
                bbox = [0.0, 0.0, 0.0, 0.0]
            area = float(o.get("area") or 0.0) * width * height
            anns.append({"id": aid, "image_id": fn, "category_id": 1, "keypoints": kp.ravel().tolist(),
                         "num_keypoints": int(v.sum()), "bbox": bbox, "area": area,
                         "iscrowd": int(o["iscrowd"])})
            aid += 1
    cats = [{"id": 1, "name": "person", "supercategory": "person", "keypoints": KEYPOINT_NAMES,
             "skeleton": SKELETON}]
    return {"images": images, "annotations": anns, "categories": cats}


def to_coco_dets(pred_frames: dict, frames: list[int], width: int, height: int) -> list[dict]:
    """Predictions [class_id, score, x1, x2, y1, y2, [x, y, s] * 17] (normalised) -> COCO keypoint results."""
    dets = []
    for fn in frames:
        for p in pred_frames.get(str(fn), []):
            if int(p[0]) != 1:
                continue
            k = np.asarray(p[6], float).reshape(N_KEYPOINTS, 3)
            kp = np.stack([k[:, 0] * width, k[:, 1] * height, np.ones(N_KEYPOINTS)], 1)
            dets.append({"image_id": fn, "category_id": 1, "score": float(p[1]), "keypoints": kp.ravel().tolist()})
    return dets


def evaluate(ann: DetectionAnnotations, pred_frames: dict, frames: list[int], width: int, height: int) -> dict:
    """COCO keypoint AP over the given frames. Values in [0, 1] (-1 = no ground truth in that area range)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_dict = to_coco_gt(ann, frames, width, height)
    dets = to_coco_dets(pred_frames, frames, width, height)
    n_gt = sum(1 for a in gt_dict["annotations"] if a["num_keypoints"] > 0 and not a["iscrowd"])
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = gt_dict
        gt.createIndex()
        if dets:
            dt = gt.loadRes(dets)
            ev = COCOeval(gt, dt, "keypoints")
            ev.params.imgIds = frames
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
            stats = [float(s) for s in ev.stats]
        else:
            stats = [0.0] * len(KP_STAT_NAMES)
    return {"metric": "COCO keypoint AP@[.50:.95] (OKS)", "frames": len(frames), "image_size": [width, height],
            "stats": dict(zip(KP_STAT_NAMES, stats)), "gt_persons": n_gt,
            "gt_persons_ignored": len(gt_dict["annotations"]) - n_gt, "predicted_persons": len(dets)}
