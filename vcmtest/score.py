"""Stage 3: score predictions against annotations (cached).

Scores: data/<seq>/results/scores/<model_id>/QPxx.json (detection) and QPxx_<task>.json (other
tasks; depth: QPxx_depth.json), recomputed when the annotations (JSON, and mask PNGs for segmentation), the
predictions file or the evaluated frame set change.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import replace
from importlib import metadata
from pathlib import Path

from .annotations import AnnotationError, DetectionAnnotations, coco_label_map, file_sha256, frames_to_evaluate

log = logging.getLogger("vcmtest")

SCORE_VERSION = 3          # 2: object-weighted AP added; 3: it ignores classes without ground truth
PANOPTIC_SCORE_VERSION = 2  # panoptic scores only; 2: stats_gt_classes (PQ over the GT classes) added
TRACKING_SCORE_VERSION = 2  # tracking scores only; 2: stats_object_weighted (detection-averaged HOTA) added


PANOPTIC_SCORE_TASKS = ("panoptic_segmentation", "semantic_segmentation")


def scores_path(results_dir: Path, mid: str, qp: int, task: str = "detection") -> Path:
    """Detection: QPxx.json; panoptic and semantic share QPxx_panoptic.json (one evaluation gives PQ and mIoU)."""
    if task in PANOPTIC_SCORE_TASKS:
        return results_dir / "scores" / mid / f"QP{qp}_panoptic.json"
    name = f"QP{qp}.json" if task == "detection" else f"QP{qp}_{task}.json"
    return results_dir / "scores" / mid / name


def class_names(class_ids) -> list[str]:
    """Class names of COCO class ids (the id as text when unknown)."""
    names = {cid: n for n, cid in coco_label_map().items()}
    return [names.get(c, str(c)) for c in class_ids]


def _only_classes(ann: DetectionAnnotations, pred_frames: dict, classes: list[int] | None):
    """Annotations and predictions restricted to the evaluated classes (unchanged when classes is None)."""
    if classes is None:
        return ann, pred_frames
    keep = set(classes)
    ann = replace(ann, frames={fn: [o for o in objs if int(o["class_id"]) in keep] for fn, objs in ann.frames.items()})
    pred_frames = {k: [p for p in v if int(p[0]) in keep] for k, v in pred_frames.items()}
    return ann, pred_frames


def _class_key(key: dict, classes: list[int] | None) -> dict:
    if classes is not None:          # only added when restricted, so unrestricted caches stay valid
        key["classes"] = list(classes)
    return key


def image_size(png: Path) -> tuple[int, int]:
    """(width, height) of a PNG."""
    from PIL import Image
    with Image.open(png) as im:
        return im.size


def score_detection(ann: DetectionAnnotations, pred_path: Path, pngs: dict[int, Path], out_path: Path,
                    label: str, classes: list[int] | None = None) -> dict:
    """Detection scores (COCO bbox AP) of one model and QP, cached in out_path.

    Recomputed if the annotations, predictions, frames or classes change.

    Args:
        classes: class ids to evaluate (sequence_overrides.<seq>.classes), None = all.

    Returns:
        {"key", "annotations", "predictions", "warnings", "result"} as written to out_path.
    """
    frames, warnings = frames_to_evaluate(set(ann.frames), set(pngs), label)
    for w in warnings:
        log.warning(w)
    key = {"version": SCORE_VERSION, "task": "detection", "annotations_sha256": ann.sha256,
           "predictions_sha256": file_sha256(pred_path),
           "frames_sha256": hashlib.sha256(json.dumps(frames).encode()).hexdigest(),
           "pycocotools": metadata.version("pycocotools")}
    _class_key(key, classes)
    if out_path.is_file():
        try:
            with open(out_path) as f:
                cached = json.load(f)
            if cached.get("key") == key:
                log.info(f"{label}: scores cached  AP={_fmt(cached['result']['stats']['AP'])}")
                return cached
        except (OSError, json.JSONDecodeError, KeyError):
            pass

    from .metrics.coco_det import evaluate
    with open(pred_path) as f:
        preds = json.load(f)
    w, h = image_size(pngs[frames[0]])
    ann_c, pf = _only_classes(ann, preds["frames"], classes)
    result = evaluate(ann_c, pf, frames, w, h)
    if classes is not None:
        result["classes_evaluated"] = class_names(classes)
    data = {"key": key, "annotations": str(ann.path), "predictions": str(pred_path),
            "warnings": warnings, "result": result}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    tmp.replace(out_path)
    st = result["stats"]
    log.info(f"{label}: AP={_fmt(st['AP'])} AP50={_fmt(st['AP50'])} AP75={_fmt(st['AP75'])} "
             f"object-weighted AP={_fmt(result['stats_object_weighted']['AP'])} ({result['frames']} frames, "
             f"{result['gt_objects']} GT objects, {result['detections']} detections)")
    return data


def score_obb(ann: DetectionAnnotations, pred_path: Path, pngs: dict[int, Path], out_path: Path, label: str,
              classes: list[int] | None = None, vehicle: bool = False) -> dict:
    """Oriented detection (metrics/obb.py): rotated-box AP@[.50:.95] (primary) + DOTA mAP@0.5 (VOC07).
    classes: DOTA ids to evaluate (sequence_overrides.<seq>.obb_classes), None = all.
    vehicle (obb_score: vehicle): also "stats_vehicle" (+ "vehicle": the full result) with large and small vehicle
    merged into one class in ground truth and predictions; the selected classes then keep "vehicle" if they contain
    either of the two."""
    from .metrics import obb
    frames, warnings = frames_to_evaluate(set(ann.frames), set(pngs), label)
    for w in warnings:
        log.warning(w)
    key = {"version": SCORE_VERSION, "task": "oriented_detection", "annotations_sha256": ann.sha256,
           "predictions_sha256": file_sha256(pred_path),
           "frames_sha256": hashlib.sha256(json.dumps(frames).encode()).hexdigest(),
           "pycocotools": metadata.version("pycocotools"), "obb_metric_version": obb.METRIC_VERSION}
    _class_key(key, classes)
    if vehicle:                      # only added when selected, so dota-only caches stay valid
        key["obb_vehicle"] = True
    cached = _cached_scores(out_path, key, label)
    if cached is not None:
        if vehicle:
            log.info(f"{label}: vehicle AP={_fmt(cached['result']['stats_vehicle']['AP'])}")
        return cached
    with open(pred_path) as f:
        preds = json.load(f)
    w, h = image_size(pngs[frames[0]])
    ann_c, pf = _only_classes(ann, preds["frames"], classes)
    result = obb.evaluate(ann_c, pf, frames, w, h)
    if vehicle:
        ann_v, pf_v = obb.merge_vehicles(ann, preds["frames"])
        keep = None if classes is None else sorted({obb.VEHICLE_ID if c in obb.VEHICLE_CLASSES else c for c in classes})
        ann_v, pf_v = _only_classes(ann_v, pf_v, keep)
        rv = obb.evaluate(ann_v, pf_v, frames, w, h)
        rv["merged"] = "large vehicle + small vehicle -> vehicle (id 10)"
        result["vehicle"] = rv
        result["stats_vehicle"] = rv["stats"]
    if classes is not None:
        from .annotations import dota_label_map
        names = {cid: n for n, cid in dota_label_map().items()}
        result["classes_evaluated"] = [names[c] for c in classes]
    data = {"key": key, "annotations": str(ann.path), "predictions": str(pred_path),
            "warnings": warnings, "result": result}
    _write_scores(out_path, data)
    st = result["stats"]
    log.info(f"{label}: rotated AP={_fmt(st['AP'])} AP50={_fmt(st['AP50'])} AP75={_fmt(st['AP75'])} "
             f"DOTA mAP50 (VOC07)={_fmt(result['stats_dota']['mAP50_voc07'])} ({result['frames']} frames, "
             f"{result['gt_objects']} GT objects, {result['detections']} detections)"
             + (f" | vehicle (small + large merged): AP={_fmt(result['stats_vehicle']['AP'])} "
                f"AP50={_fmt(result['stats_vehicle']['AP50'])}" if vehicle else ""))
    return data


def score_lpr(ann: DetectionAnnotations, pred_path: Path, pngs: dict[int, Path], out_path: Path, label: str,
              iou_thr: float = 0.5, task: str = "plate_recognition") -> dict:
    """End-to-end licence plate (task plate_recognition) or race bib (bib_recognition: digits only) recognition
    (metrics/lpr.py): plate / bib accuracy (primary), character accuracy, box AP. iou_thr = config lpr_iou (box
    match threshold; in the cache key)."""
    from .metrics import lpr
    frames, warnings = frames_to_evaluate(set(ann.frames), set(pngs), label)
    for w in warnings:
        log.warning(w)
    item, digits = ("bib", True) if task == "bib_recognition" else ("plate", False)
    key = {"version": SCORE_VERSION, "task": task, "annotations_sha256": ann.sha256,
           "predictions_sha256": file_sha256(pred_path),
           "frames_sha256": hashlib.sha256(json.dumps(frames).encode()).hexdigest(),
           "pycocotools": metadata.version("pycocotools"), "lpr_metric_version": lpr.METRIC_VERSION,
           "lpr_iou": float(iou_thr)}
    if out_path.is_file():
        try:
            with open(out_path) as f:
                cached = json.load(f)
            if cached.get("key") == key:
                log.info(f"{label}: scores cached  {item} accuracy="
                         f"{_fmt(cached['result']['stats'][f'{item}_accuracy'])}")
                return cached
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    with open(pred_path) as f:
        preds = json.load(f)
    w, h = image_size(pngs[frames[0]])
    result = lpr.evaluate(ann, preds["frames"], frames, w, h, iou_thr, item, digits)
    data = {"key": key, "annotations": str(ann.path), "predictions": str(pred_path),
            "warnings": warnings, "result": result}
    _write_scores(out_path, data)
    st = result["stats"]
    log.info(f"{label}: {item} accuracy={_fmt(st[f'{item}_accuracy'])} char accuracy={_fmt(st['char_accuracy'])} "
             f"detected {_fmt(st['detection_recall'])} (read correctly when detected {_fmt(st['accuracy_detected'])}) "
             f"box AP50={_fmt(st['det_AP50'])} ({result[f'gt_{item}s']} GT {item}s, {result['predictions']} "
             f"predictions, {result[f'false_{item}s']} false)")
    return data


def _fmt(v) -> str:
    return "n/a" if v is None else f"{100 * v:.2f}"


def _cached_scores(out_path: Path, key: dict, label: str) -> dict | None:
    if out_path.is_file():
        try:
            with open(out_path) as f:
                cached = json.load(f)
            if cached.get("key") == key:
                log.info(f"{label}: scores cached  AP={_fmt(cached['result']['stats']['AP'])}")
                return cached
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    return None


def _write_scores(out_path: Path, data: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    tmp.replace(out_path)


def score_segmentation(ann: DetectionAnnotations, masks_dir: Path | None, pred_path: Path,
                       pngs: dict[int, Path], out_path: Path, label: str, gt_cache_dir: Path | None,
                       classes: list[int] | None = None) -> dict:
    """COCO mask AP (primary) plus COCO bbox AP of the same predictions (secondary, detection GT)."""
    from .metrics import coco_det, coco_segm

    frames, warnings = frames_to_evaluate(set(ann.frames), set(pngs), label)
    for w in warnings:
        log.warning(w)
    key = {"version": SCORE_VERSION, "task": "instance_segmentation",
           "annotations_sha256": coco_segm.segm_annotations_sha256(ann, masks_dir),
           "predictions_sha256": file_sha256(pred_path),
           "frames_sha256": hashlib.sha256(json.dumps(frames).encode()).hexdigest(),
           "pycocotools": metadata.version("pycocotools")}
    _class_key(key, classes)
    cached = _cached_scores(out_path, key, label)
    if cached is not None:
        return cached

    with open(pred_path) as f:
        preds = json.load(f)
    w, h = image_size(pngs[frames[0]])
    gt = coco_segm.cached_gt(ann, masks_dir, w, h, gt_cache_dir)
    ann_c, pf = _only_classes(ann, preds["frames"], classes)
    if classes is not None:
        keep = set(classes)
        gt = {**gt, "coco": {**gt["coco"], "annotations": [a for a in gt["coco"]["annotations"]
                                                           if a["category_id"] in keep]}}
    result = coco_segm.evaluate(gt, pf, frames, w, h)
    bbox = coco_det.evaluate(ann_c, pf, frames, w, h)
    if classes is not None:
        result["classes_evaluated"] = class_names(classes)
    result["bbox_stats"] = bbox["stats"]
    result["bbox_stats_object_weighted"] = bbox["stats_object_weighted"]
    result["gt_counts"] = gt["counts"]
    data = {"key": key, "annotations": str(ann.path), "masks_dir": str(masks_dir) if masks_dir else None,
            "predictions": str(pred_path), "warnings": warnings, "result": result}
    _write_scores(out_path, data)
    st = result["stats"]
    log.info(f"{label}: mask AP={_fmt(st['AP'])} AP50={_fmt(st['AP50'])} AP75={_fmt(st['AP75'])} "
             f"object-weighted AP={_fmt(result['stats_object_weighted']['AP'])}  "
             f"(bbox AP={_fmt(bbox['stats']['AP'])}; {result['frames']} frames, {result['gt_objects']} masked "
             f"GT objects, {result['detections']} detections)")
    return data


def score_tracking(ann: DetectionAnnotations, pred_path: Path, pngs: dict[int, Path], out_path: Path,
                   label: str, classes: list[int] | None = None) -> dict:
    """HOTA (primary), CLEAR MOT and IDF1 with TrackEval; class average over classes with GT tracks ("stats") and
    detection-weighted over the classes ("stats_object_weighted")."""
    from .metrics import mot

    frames, warnings = frames_to_evaluate(set(ann.frames), set(pngs), label)
    for w in warnings:
        log.warning(w)
    key = {"version": SCORE_VERSION, "task": "tracking", "annotations_sha256": ann.sha256,
           "predictions_sha256": file_sha256(pred_path),
           "frames_sha256": hashlib.sha256(json.dumps(frames).encode()).hexdigest(),
           "trackeval": metadata.version("trackeval"), "tracking_version": TRACKING_SCORE_VERSION}
    _class_key(key, classes)
    if out_path.is_file():
        try:
            with open(out_path) as f:
                cached = json.load(f)
            if cached.get("key") == key:
                log.info(f"{label}: scores cached  HOTA={_fmt(cached['result']['stats']['HOTA'])}")
                return cached
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    with open(pred_path) as f:
        preds = json.load(f)
    w, h = image_size(pngs[frames[0]])
    ann_c, pf = _only_classes(ann, preds["frames"], classes)
    result = mot.evaluate(ann_c, pf, frames, w, h)
    for msg in result["warnings"]:
        log.warning(f"{label}: {msg}")
    data = {"key": key, "annotations": str(ann.path), "predictions": str(pred_path),
            "warnings": warnings + result["warnings"], "result": result}
    _write_scores(out_path, data)
    st = result["stats"]
    log.info(f"{label}: HOTA={_fmt(st['HOTA'])} DetA={_fmt(st['DetA'])} AssA={_fmt(st['AssA'])} "
             f"MOTA={_fmt(st['MOTA'])} IDF1={_fmt(st['IDF1'])} IDSW={st['IDSW']:.0f} "
             f"(classes {', '.join(result['classes'])}; {result['frames']} frames)")
    return data


def score_panoptic(ann: DetectionAnnotations, masks_dir: Path, pred_path: Path, pngs: dict[int, Path],
                   out_path: Path, label: str, gt_cache_dir: Path | None, classes: list[int] | None = None) -> dict:
    """PQ (panoptic_segmentation) and mIoU (semantic_segmentation) from one evaluation, cached in one file.
    classes (labels.py ids): only these thing classes are kept in the averages; all stuff classes stay."""
    from .metrics import panoptic

    frames, warnings = frames_to_evaluate(set(ann.frames), set(pngs), label)
    for w in warnings:
        log.warning(w)
    key = {"version": SCORE_VERSION, "task": "panoptic", "annotations_sha256": panoptic.gt_fingerprint(ann, masks_dir),
           "predictions_sha256": file_sha256(pred_path),
           "frames_sha256": hashlib.sha256(json.dumps(frames).encode()).hexdigest(),
           "panoptic_version": PANOPTIC_SCORE_VERSION}
    _class_key(key, classes)
    if out_path.is_file():
        try:
            with open(out_path) as f:
                cached = json.load(f)
            if cached.get("key") == key:
                st = cached["result"]["stats"]
                log.info(f"{label}: scores cached  PQ={_fmt(st['PQ'])} mIoU={_fmt(st['mIoU'])}")
                return cached
        except (OSError, json.JSONDecodeError, KeyError):
            pass
    with open(pred_path) as f:
        preds = json.load(f)
    w, h = image_size(pngs[frames[0]])
    gt = panoptic.cached_gt(ann, masks_dir, w, h, gt_cache_dir)
    things = None
    if classes is not None:
        from .annotations import panoptic_categories
        pan_ids = {n: c for c, n, _ in panoptic_categories()}
        things = {pan_ids[n] for n in class_names(classes)}      # thing names are the same in both label sets
    result = panoptic.evaluate(gt, preds["frames"], frames, w, h, things)
    data = {"key": key, "annotations": str(ann.path), "masks_dir": str(masks_dir), "predictions": str(pred_path),
            "warnings": warnings, "result": result}
    _write_scores(out_path, data)
    st = result["stats"]
    log.info(f"{label}: PQ={_fmt(st['PQ'])} SQ={_fmt(st['SQ'])} RQ={_fmt(st['RQ'])} PQth={_fmt(st['PQ_th'])} "
             f"PQst={_fmt(st['PQ_st'])} PQ(GT classes)={_fmt(result['stats_gt_classes']['PQ'])} "
             f"mIoU={_fmt(st['mIoU'])} "
             f"({result['frames']} frames, "
             f"{result['predicted_segments']} predicted segments)")
    return data



def score_keypoints(ann: DetectionAnnotations, pred_path: Path, pngs: dict[int, Path], out_path: Path,
                    label: str) -> dict:
    """COCO keypoint AP (OKS) for one (model, QP); persons only, so per-sequence `classes` does not apply."""
    frames, warnings = frames_to_evaluate(set(ann.frames), set(pngs), label)
    for w in warnings:
        log.warning(w)
    key = {"version": SCORE_VERSION, "task": "keypoints", "annotations_sha256": ann.sha256,
           "predictions_sha256": file_sha256(pred_path),
           "frames_sha256": hashlib.sha256(json.dumps(frames).encode()).hexdigest(),
           "pycocotools": metadata.version("pycocotools")}
    cached = _cached_scores(out_path, key, label)
    if cached is not None:
        return cached
    from .metrics.coco_kp import evaluate
    with open(pred_path) as f:
        preds = json.load(f)
    w, h = image_size(pngs[frames[0]])
    result = evaluate(ann, preds["frames"], frames, w, h)
    data = {"key": key, "annotations": str(ann.path), "predictions": str(pred_path),
            "warnings": warnings, "result": result}
    _write_scores(out_path, data)
    st = result["stats"]
    log.info(f"{label}: keypoint AP={_fmt(st['AP'])} AP50={_fmt(st['AP50'])} AP75={_fmt(st['AP75'])} "
             f"APm={_fmt(st['APm'])} APl={_fmt(st['APl'])} ({result['frames']} frames, {result['gt_persons']} GT "
             f"persons, {result['predicted_persons']} predicted)")
    return data


def _depth_score_key(depth_files: dict[int, Path], pred_path: Path, frames: list[int], min_mm: float,
                     gt_sha: str | None) -> dict:
    from .metrics import depth
    return {"version": SCORE_VERSION, "task": "depth",
            "annotations_sha256": gt_sha or depth.gt_fingerprint(depth_files),
            "predictions_sha256": file_sha256(pred_path),
            "frames_sha256": hashlib.sha256(json.dumps(frames).encode()).hexdigest(),
            "depth_min_mm": min_mm, "depth_version": depth.GT_VERSION, "depth_metric_version": depth.METRIC_VERSION}


def _cached_depth_scores(out_path: Path, key: dict, label: str) -> dict | None:
    if not out_path.is_file():
        return None
    try:
        with open(out_path) as f:
            cached = json.load(f)
        if cached.get("key") == key:
            log.info(f"{label}: scores cached  aligned AbsRel={_fmt(cached['result']['stats_aligned']['AbsRel'])}")
            return cached
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def cached_depth_scores(depth_files: dict[int, Path], pred_path: Path, pngs: dict[int, Path], out_path: Path,
                        label: str, min_mm: float, gt_sha: str | None = None) -> dict | None:
    """The cached depth scores if they are still valid for this predictions file (its depth map PNGs are not
    needed: their sha256 are in the predictions file), else None. Lets a run skip a (model, QP) whose depth
    maps were removed (inference.remove_depth_maps)."""
    if not pred_path.is_file() or not out_path.is_file():
        return None
    frames, _ = frames_to_evaluate(set(depth_files), set(pngs), label)
    return _cached_depth_scores(out_path, _depth_score_key(depth_files, pred_path, frames, min_mm, gt_sha), label)


def score_depth(depth_files: dict[int, Path], pred_path: Path, pngs: dict[int, Path], out_path: Path, label: str,
                min_mm: float, gt_sha: str | None = None) -> dict:
    """Depth metrics (AbsRel primary) for one (model, QP). depth_files: GT depth PNGs by frame number.
    Ground-truth pixels with depth 0 or < min_mm (config depth_min_mm) are ignored."""
    from .metrics import depth
    frames, warnings = frames_to_evaluate(set(depth_files), set(pngs), label)
    for w in warnings:
        log.warning(w)
    key = _depth_score_key(depth_files, pred_path, frames, min_mm, gt_sha)
    cached = _cached_depth_scores(out_path, key, label)
    if cached is not None:
        return cached
    with open(pred_path) as f:
        preds = json.load(f)
    pred_files = {int(fn): (pred_path.parent / rec[2], rec) for fn, rec in preds["frames"].items()}
    kind = preds["meta"].get("model", {}).get("depth_output", "metric")
    missing = [fn for fn in frames if fn not in pred_files]
    if missing:
        raise AnnotationError(f"{label}: no predicted depth for frames {missing[:5]}")
    w, h = image_size(pngs[frames[0]])
    try:
        result = depth.evaluate(depth_files, pred_files, frames, w, h, min_mm, kind)
    except ValueError as e:
        raise AnnotationError(f"{label}: {e}") from e
    data = {"key": key, "annotations": str(next(iter(depth_files.values())).parent), "predictions": str(pred_path),
            "warnings": warnings, "result": result}
    _write_scores(out_path, data)
    st, al = result["stats"], result["stats_aligned"]
    if not result["frames_scored"]:
        log.warning(f"{label}: no frame has valid ground-truth depth (>= {min_mm} mm)")
        return data
    raw = (f"AbsRel={_fmt(st['AbsRel'])} RMSE={st['RMSE']:.3f} m delta1={_fmt(st['delta1'])} "
           f"(median scale {result['scale']['median']:.3f}) | " if st else f"{kind} output, no raw metric | ")
    log.info(f"{label}: {raw}aligned AbsRel={_fmt(al['AbsRel'])} RMSE={al['RMSE']:.3f} m "
             f"delta1={_fmt(al['delta1'])} ({result['frames_scored']} frames, "
             f"{100 * result['valid_pixel_share']:.1f} % valid GT pixels)")
    return data
