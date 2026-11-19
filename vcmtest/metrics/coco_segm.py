"""Instance segmentation: COCO-style mask AP averaged over IoU 0.50:0.95 (pycocotools 'segm').

Ground truth (from <sequence_name>_annotations.json):
- Masks first: <sequence_name>_masks/<name>_NNNN.png (frame number as in the PNG frames). The PNG is
  downscaled to the decoded frame size with nearest neighbour, so colours never change; an
  object's mask is the set of pixels with exactly its "mask_color". Colours without a JSON entry
  are background. Polygons are not used when masks exist.
- Polygons only when there is no masks folder: "polygon" [x1, y1, x2, y2, ...], normalised to
  [0, 1] like the boxes (values > 1 are an error, not guessed to be pixels).
- An object without a mask (no mask_color / polygon, or a mask_color with no pixels in the PNG) is
  an ignore region: its box becomes a pycocotools crowd region, so a prediction on it is neither a
  true nor a false positive. iscrowd=1 objects are crowd regions (their mask if they have one,
  else their box).
- Ground-truth area = mask pixel count at the decoded resolution (COCO small/medium/large ranges).

Predictions: [class_id, score, x1, x2, y1, y2, W, H, rle]; W x H must equal the decoded frame size.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from ..annotations import AnnotationError, DetectionAnnotations, coco_label_map, file_sha256
from .coco_det import OBJECT_STAT_NAMES, STAT_NAMES, run_cocoeval

log = logging.getLogger("vcmtest")

GT_VERSION = 1


# ----------------------------------------------------------------------------- source + fingerprint

def mask_files(masks_dir: Path | None) -> dict[int, Path]:
    """Mask PNGs of the masks folder by frame number; {} if there is no folder."""
    if masks_dir is None or not masks_dir.is_dir():
        return {}
    from ..paths import scan_pngs
    return scan_pngs(masks_dir)


def segmentation_source(ann: DetectionAnnotations, masks_dir: Path | None) -> str | None:
    """'masks' if a masks folder with PNGs exists, else 'polygons' if any object has one, else None."""
    if mask_files(masks_dir):
        return "masks"
    if any("polygon" in o for objs in ann.frames.values() for o in objs):
        return "polygons"
    return None


def segm_annotations_sha256(ann: DetectionAnnotations, masks_dir: Path | None) -> str:
    """Fingerprint of everything the segmentation ground truth depends on (JSON + mask PNGs)."""
    h = hashlib.sha256(f"segm-gt-v{GT_VERSION}|{ann.sha256}|".encode())
    src = segmentation_source(ann, masks_dir)
    h.update(f"{src}|".encode())
    if src == "masks":
        for fn, p in mask_files(masks_dir).items():
            h.update(f"{fn}|{p.name}|{file_sha256(p)}|".encode())
    return h.hexdigest()


# ----------------------------------------------------------------------------- ground truth

def _box_px(o: dict, w: int, h: int) -> list[float]:
    return [o["x_min"] * w, o["y_min"] * h, (o["x_max"] - o["x_min"]) * w, (o["y_max"] - o["y_min"]) * h]


def _box_rle(box: list[float], w: int, h: int) -> dict:
    from pycocotools import mask as mu
    x, y, bw, bh = box
    rle = mu.frPyObjects([[x, y, x + bw, y, x + bw, y + bh, x, y + bh]], h, w)
    return mu.merge(rle)


def _rle_json(rle: dict) -> dict:
    c = rle["counts"]
    return {"size": [int(v) for v in rle["size"]], "counts": c.decode("ascii") if isinstance(c, bytes) else c}


def _load_mask_png(path: Path, w: int, h: int) -> np.ndarray:
    """Colour mask as int32 keys (r<<16 | g<<8 | b), downscaled to w x h with nearest neighbour."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        mw, mh = im.size
        if (mw, mh) != (w, h):
            if mw * h != mh * w:
                raise AnnotationError(f"{path.name}: mask size {mw}x{mh} has a different aspect ratio "
                                      f"than the decoded frames {w}x{h}")
            im = im.resize((w, h), Image.Resampling.NEAREST)
        a = np.asarray(im, dtype=np.int32)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


def build_gt(ann: DetectionAnnotations, masks_dir: Path | None, width: int, height: int) -> dict:
    """COCO ground-truth dict for all annotated frames + counts of how objects were used."""
    from pycocotools import mask as mu

    src = segmentation_source(ann, masks_dir)
    if src is None:
        raise AnnotationError("no instance masks (no masks folder) and no polygons in the annotations")
    files = mask_files(masks_dir) if src == "masks" else {}
    cats = [{"id": cid, "name": name} for name, cid in sorted(coco_label_map().items(), key=lambda x: x[1])]
    images, anns = [], []
    counts = {"masked": 0, "crowd": 0, "ignored_no_mask": 0, "ignored_empty_mask": 0}
    aid = 1
    for fn, objs in ann.frames.items():
        images.append({"id": fn, "width": width, "height": height, "file_name": str(fn)})
        keys = None
        if src == "masks" and any("mask_color" in o for o in objs):
            if fn not in files:
                raise AnnotationError(f"frame {fn}: objects have mask_color but there is no mask PNG "
                                      f"for this frame in {masks_dir}")
            keys = _load_mask_png(files[fn], width, height)
        for o in objs:
            box = _box_px(o, width, height)
            rle = None
            if src == "masks" and "mask_color" in o:
                r, g, b = (int(v) for v in o["mask_color"])
                m = keys == ((r << 16) | (g << 8) | b)
                if m.any():
                    rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))
                else:
                    counts["ignored_empty_mask"] += 1
            elif src == "polygons" and o.get("polygon"):
                poly = [float(v) for v in o["polygon"]]
                if len(poly) < 6 or len(poly) % 2:
                    raise AnnotationError(f"frame {fn}: polygon needs >= 3 x,y pairs")
                if max(poly) > 1.0 + 1e-6 or min(poly) < -1e-6:
                    raise AnnotationError(f"frame {fn}: polygon values outside [0, 1] (expected "
                                          f"normalised coordinates like the boxes)")
                px = [v * (width if i % 2 == 0 else height) for i, v in enumerate(poly)]
                rle = mu.merge(mu.frPyObjects([px], height, width))
            elif src == "masks" or src == "polygons":
                counts["ignored_no_mask"] += 1 if not o["iscrowd"] else 0
            crowd = int(o["iscrowd"]) or rle is None
            if rle is None:
                rle = _box_rle(box, width, height)
            if crowd:
                counts["crowd"] += 1
            else:
                counts["masked"] += 1
            anns.append({"id": aid, "image_id": fn, "category_id": int(o["class_id"]),
                         "segmentation": _rle_json(rle), "area": float(mu.area(rle)),
                         "bbox": box, "iscrowd": int(crowd)})
            aid += 1
    return {"source": src, "counts": counts,
            "coco": {"images": images, "annotations": anns, "categories": cats}}


_GT_MEMO: dict[tuple, dict] = {}


def cached_gt(ann: DetectionAnnotations, masks_dir: Path | None, width: int, height: int,
              cache_dir: Path | None) -> dict:
    """build_gt, cached in memory and as <cache_dir>/segm_gt_<W>x<H>.json (keyed by the GT fingerprint)."""
    sha = segm_annotations_sha256(ann, masks_dir)
    memo_key = (sha, width, height)
    if memo_key in _GT_MEMO:
        return _GT_MEMO[memo_key]
    path = cache_dir / f"segm_gt_{width}x{height}.json" if cache_dir else None
    if path and path.is_file():
        try:
            with open(path) as f:
                data = json.load(f)
            if data.get("key") == {"sha256": sha, "width": width, "height": height}:
                _GT_MEMO[memo_key] = data["gt"]
                return data["gt"]
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    log.info(f"Building instance-mask ground truth at {width}x{height} ...")
    gt = build_gt(ann, masks_dir, width, height)
    c = gt["counts"]
    log.info(f"Mask ground truth ({gt['source']}): {c['masked']} masked objects, {c['crowd']} crowd/ignore "
             f"regions ({c['ignored_no_mask']} without mask, {c['ignored_empty_mask']} with an empty mask)")
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump({"key": {"sha256": sha, "width": width, "height": height}, "gt": gt}, f)
        tmp.replace(path)
    _GT_MEMO[memo_key] = gt
    return gt


# ----------------------------------------------------------------------------- evaluation

def to_coco_dets(pred_frames: dict, frames: list[int], width: int, height: int) -> tuple[list[dict], int]:
    """Predictions [class_id, score, x1, x2, y1, y2, W, H, rle] -> COCO segmentation results.

    Returns:
        (detections, number of predictions dropped because their class id is not in labels.py).

    Raises:
        AnnotationError: predictions without masks, or mask size different from the frame.
    """
    valid = set(coco_label_map().values())
    dets, dropped = [], 0
    for fn in frames:
        for p in pred_frames.get(str(fn), []):
            if len(p) < 9:
                raise AnnotationError("predictions have no masks ([class_id, score, x1, x2, y1, y2, W, H, rle] "
                                      "expected); is the model an instance segmentation model?")
            cid, score, _, _, _, _, w, h, rle = p[:9]
            if (int(w), int(h)) != (width, height):
                raise AnnotationError(f"frame {fn}: predicted mask is {w}x{h}, decoded frame is {width}x{height}")
            if cid not in valid:
                dropped += 1
                continue
            dets.append({"image_id": fn, "category_id": int(cid), "score": float(score),
                         "segmentation": {"size": [height, width], "counts": rle}})
    return dets, dropped


def evaluate(gt: dict, pred_frames: dict, frames: list[int], width: int, height: int) -> dict:
    """Mask AP over the given frames. gt from cached_gt/build_gt. AP values in [0, 1]."""
    fs = set(frames)
    coco = gt["coco"]
    gt_dict = {"images": [i for i in coco["images"] if i["id"] in fs],
               "annotations": [a for a in coco["annotations"] if a["image_id"] in fs],
               "categories": coco["categories"]}
    dets, dropped = to_coco_dets(pred_frames, frames, width, height)
    if dropped:
        log.warning(f"{dropped} detections with class ids outside labels.py were ignored")
    n_gt = sum(1 for a in gt_dict["annotations"] if not a["iscrowd"])
    result = {"metric": "COCO mask AP@[.50:.95] (pycocotools segm)", "frames": len(frames),
              "image_size": [width, height], "gt_source": gt["source"], "gt_objects": n_gt,
              "gt_crowd_or_ignore_regions": len(gt_dict["annotations"]) - n_gt,
              "detections": len(dets), "detections_dropped": dropped}
    if not dets:
        result["stats"] = {k: 0.0 for k in STAT_NAMES}
        result["stats_object_weighted"] = {k: 0.0 for k in OBJECT_STAT_NAMES}
        result["per_class_AP"] = {}
        result["note"] = "no detections"
        return result
    result["stats"], result["per_class_AP"], result["stats_object_weighted"] = run_cocoeval(gt_dict, dets, frames,
                                                                                           "segm")
    return result
