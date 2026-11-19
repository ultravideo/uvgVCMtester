"""Depth estimation metrics against metric ground-truth depth maps.

Ground truth: <sequence_name>_depth/ PNGs (depth_0001.png ...), 16-bit grayscale, depth in millimetres. Resized to the
decoded frame size with nearest neighbour if the sizes differ. A pixel is valid when its depth is > 0 and
>= min_mm (config depth_min_mm); other pixels are ignored.

Predictions (vcmtest/infer.py save_depth_png), by the model's output kind:
- "metric": depth in millimetres (>= 1 mm);
- "relative_disparity": affine-invariant inverse depth (unknown scale and shift; larger = nearer), stored
  normalised per frame with its range.

Per frame, over the valid pixels (d = predicted, g = ground truth, both in metres):
  AbsRel = mean(|d - g| / g)            SqRel = mean((d - g)^2 / g)
  RMSE = sqrt(mean((d - g)^2)) [m]     RMSE_log = sqrt(mean((ln d - ln g)^2))     log10 = mean(|log10 d - log10 g|)
  delta1/2/3 = share of pixels with max(d/g, g/d) < 1.25, 1.25^2, 1.25^3
The sequence value of each metric is the mean of the per-frame values (as in the usual NYU / KITTI evaluation).
Three variants:
- "stats" (metric models only): the predicted depth as is (includes the model's scale error).
- "stats_median_scaled" (metric models only): prediction x median(g) / median(d) per frame (scale removed).
- "stats_aligned" (all models): per-frame least-squares scale and shift fitted to the ground truth on the valid
  pixels, in depth space for metric models (d' = s d + t) and in inverse-depth space for relative_disparity models
  (1/d' = s x + t, the MiDaS protocol). Aligned depth is clamped to [ALIGN_MIN_M, ALIGN_MAX_M] (a non-positive
  fitted inverse depth becomes ALIGN_MAX_M). This compares the structure of all models on the same footing.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

METRIC_KEYS = ["AbsRel", "SqRel", "RMSE", "RMSE_log", "log10", "delta1", "delta2", "delta3"]
GT_VERSION = 1
METRIC_VERSION = 2            # 2: relative models, stats_aligned
KINDS = ("metric", "relative_disparity")
ALIGN_MIN_M, ALIGN_MAX_M = 0.001, 100.0


def load_depth_mm(path: Path, width: int | None = None, height: int | None = None) -> np.ndarray:
    """16-bit depth PNG -> float64 array of its values (nearest-neighbour resize to width x height if given)."""
    from PIL import Image
    with Image.open(path) as im:
        if im.mode not in ("I;16", "I;16B", "I;16L", "I"):
            raise ValueError(f"{path}: depth PNG must be 16-bit grayscale, got mode {im.mode}")
        if width and height and im.size != (width, height):
            im = im.resize((width, height), Image.NEAREST)
        return np.asarray(im).astype(np.float64)


def load_prediction(path: Path, record: list) -> np.ndarray:
    """Stored prediction -> metres (metric record [W, H, png, sha]) or raw model values (relative record + lo, hi)."""
    v = load_depth_mm(path)
    if len(record) >= 6:
        lo, hi = float(record[4]), float(record[5])
        return lo + (v - 1.0) / 65534.0 * (hi - lo)
    return v / 1000.0


def gt_fingerprint(files: dict[int, Path]) -> str:
    """Fingerprint of the depth ground truth (frame numbers, file names and SHA-256 of the PNGs)."""
    from ..annotations import file_sha256
    h = hashlib.sha256(f"depth-gt-v{GT_VERSION}|".encode())
    for fn, p in sorted(files.items()):
        h.update(f"{fn}|{p.name}|{file_sha256(p)}|".encode())
    return h.hexdigest()


def _lstsq(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(s, t) minimising sum (s x + t - y)^2."""
    a = np.stack([x, np.ones_like(x)], 1)
    (s, t), *_ = np.linalg.lstsq(a, y, rcond=None)
    return float(s), float(t)


def frame_metrics(pred: np.ndarray, gt_mm: np.ndarray, min_mm: float, kind: str = "metric") -> dict:
    """{"stats", "stats_median_scaled", "stats_aligned", "info"} for one frame. pred: metres (metric) or raw
    relative values. The stats are None when the frame has no valid pixel (or the variant does not apply)."""
    if kind not in KINDS:
        raise ValueError(f"unknown depth output kind {kind!r}")
    if pred.shape != gt_mm.shape:
        raise ValueError(f"prediction {pred.shape} and ground truth {gt_mm.shape} differ in size")
    valid = (gt_mm > 0) & (gt_mm >= min_mm)
    out = {"stats": None, "stats_median_scaled": None, "stats_aligned": None,
           "info": {"valid_pixels": int(valid.sum()), "pixels": int(valid.size)}}
    if valid.sum() < 2:
        return out
    g = gt_mm[valid] / 1000.0
    x = pred[valid].astype(np.float64)
    if kind == "metric":
        d = np.maximum(x, ALIGN_MIN_M)
        s = float(np.median(g) / np.median(d))
        out["stats"], out["stats_median_scaled"] = _metrics(d, g), _metrics(d * s, g)
        out["info"]["median_scale"] = s
        s, t = _lstsq(d, g)
        da = np.clip(s * d + t, ALIGN_MIN_M, ALIGN_MAX_M)
    else:                                   # relative inverse depth: fit to 1 / g
        s, t = _lstsq(x, 1.0 / g)
        inv = s * x + t
        da = np.where(inv > 1.0 / ALIGN_MAX_M, 1.0 / np.maximum(inv, 1e-12), ALIGN_MAX_M)
        da = np.clip(da, ALIGN_MIN_M, ALIGN_MAX_M)
    out["stats_aligned"] = _metrics(da, g)
    out["info"]["align"] = [s, t]
    return out


def _metrics(d: np.ndarray, g: np.ndarray) -> dict:
    diff = d - g
    ratio = np.maximum(d / g, g / d)
    return {"AbsRel": float(np.mean(np.abs(diff) / g)), "SqRel": float(np.mean(diff ** 2 / g)),
            "RMSE": float(np.sqrt(np.mean(diff ** 2))),
            "RMSE_log": float(np.sqrt(np.mean((np.log(d) - np.log(g)) ** 2))),
            "log10": float(np.mean(np.abs(np.log10(d) - np.log10(g)))),
            "delta1": float(np.mean(ratio < 1.25)), "delta2": float(np.mean(ratio < 1.25 ** 2)),
            "delta3": float(np.mean(ratio < 1.25 ** 3))}


def evaluate(gt_files: dict[int, Path], preds: dict[int, tuple[Path, list]], frames: list[int], width: int,
             height: int, min_mm: float, kind: str = "metric") -> dict:
    """Depth metrics over the given frames. preds: frame -> (png path, stored record).
    Frames without valid ground-truth pixels are left out (counted)."""
    per = {"stats": {}, "stats_median_scaled": {}, "stats_aligned": {}}
    scales = []
    valid_px = total_px = 0
    for fn in frames:
        g = load_depth_mm(gt_files[fn], width, height)
        path, rec = preds[fn]
        p = load_prediction(path, rec)
        if p.shape != (height, width):
            raise ValueError(f"frame {fn}: predicted depth is {p.shape[1]}x{p.shape[0]}, frame is {width}x{height}")
        r = frame_metrics(p, g, min_mm, kind)
        valid_px += r["info"]["valid_pixels"]
        total_px += r["info"]["pixels"]
        for k in per:
            if r[k] is not None:
                per[k][fn] = r[k]
        if "median_scale" in r["info"]:
            scales.append(r["info"]["median_scale"])

    def mean(pf):
        if not pf:
            return None
        return {k: float(np.mean([v[k] for v in pf.values()])) for k in METRIC_KEYS}

    sc = np.array(scales)
    return {"metric": "depth AbsRel (mean of per-frame values; lower is better)", "frames": len(frames),
            "frames_scored": len(per["stats_aligned"]), "image_size": [width, height], "min_depth_mm": min_mm,
            "depth_output": kind, "valid_pixel_share": valid_px / total_px if total_px else None,
            "stats": mean(per["stats"]), "stats_median_scaled": mean(per["stats_median_scaled"]),
            "stats_aligned": mean(per["stats_aligned"]),
            "scale": ({"median": float(np.median(sc)), "mean": float(sc.mean()), "min": float(sc.min()),
                       "max": float(sc.max())} if len(sc) else None),
            "per_frame_AbsRel": {str(k): v["AbsRel"] for k, v in (per["stats"] or per["stats_aligned"]).items()},
            "per_frame_AbsRel_aligned": {str(k): v["AbsRel"] for k, v in per["stats_aligned"].items()}}
