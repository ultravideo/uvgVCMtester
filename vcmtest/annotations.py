"""Loading of detection annotations (<sequence_name>_annotations.json, docs/annotation_format.txt).

Format: {"version": ..., "1": [ {class_id, x_min, x_max, y_min, y_max, [iscrowd], [track_id],
[polygon], [mask_color]}, ... ], "2": [...], ...}. A missing "iscrowd" means 0. Keys that are integers are frame numbers
(1-based, same as the PNG frame numbers); other keys (e.g. "version") are metadata.
Coordinates are normalised to [0, 1].
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import resolve_path

log = logging.getLogger("vcmtest")

_REQUIRED = ("class_id", "x_min", "x_max", "y_min", "y_max")


class AnnotationError(ValueError):
    """Invalid or inconsistent annotation file."""
    pass


def coco_label_map() -> dict[str, int]:
    """COCO class name -> id (1..80) from task_models/labels.py."""
    tm = str(resolve_path("task_models"))
    if tm not in sys.path:
        sys.path.insert(0, tm)
    from labels import COCO_label_map
    return COCO_label_map


def dota_label_map() -> dict[str, int]:
    """DOTA v1.0 class name -> 0-based id (task_models/labels_dota.py)."""
    tm = str(resolve_path("task_models"))
    if tm not in sys.path:
        sys.path.insert(0, tm)
    from labels_dota import DOTA_label_map
    return DOTA_label_map


def panoptic_categories() -> list[tuple[int, str, int]]:
    """COCO panoptic categories (id, name, isthing) from task_models/labels_panoptic.py."""
    tm = str(resolve_path("task_models"))
    if tm not in sys.path:
        sys.path.insert(0, tm)
    from labels_panoptic import COCO_PANOPTIC_CATEGORIES
    return list(COCO_PANOPTIC_CATEGORIES)


@dataclass
class DetectionAnnotations:
    """Detection annotations of one sequence: per frame number the list of objects."""
    path: Path
    sha256: str
    version: str | None
    frames: dict[int, list[dict]]   # frame number -> objects

    @property
    def class_ids(self) -> set[int]:
        """Class ids of all annotated objects."""
        return {o["class_id"] for objs in self.frames.values() for o in objs}


def file_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def annotation_version(path: Path | None) -> str:
    """The annotation JSON's "version" value as a string; "" if the key (or the JSON) does not exist."""
    if path is None or not Path(path).is_file():
        return ""
    with open(path) as f:
        raw = json.load(f)
    v = raw.get("version") if isinstance(raw, dict) else None
    return "" if v is None else str(v)


def load_detection_annotations(path: Path) -> DetectionAnnotations:
    """Load and validate <sequence_name>_annotations.json (objects need class_id and box keys; missing iscrowd = 0).

    Raises:
        AnnotationError: invalid structure, frame number, class id, box or iscrowd value, or no frames.
    """
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise AnnotationError(f"{path}: top level must be an object keyed by frame number")
    valid_ids = set(coco_label_map().values())
    frames: dict[int, list[dict]] = {}
    for k, objs in raw.items():
        if not str(k).isdigit():
            continue
        fn = int(k)
        if fn < 1:
            raise AnnotationError(f"{path}: frame numbers start from 1, got '{k}'")
        if not isinstance(objs, list):
            raise AnnotationError(f"{path}: frame {k} must be a list of objects")
        for i, o in enumerate(objs):
            missing = [r for r in _REQUIRED if r not in o]
            if missing:
                raise AnnotationError(f"{path}: frame {k} object {i} missing keys {missing}")
            if o["class_id"] not in valid_ids:
                raise AnnotationError(f"{path}: frame {k} object {i} has class_id {o['class_id']} "
                                      f"not in labels.py")
            if not (o["x_min"] <= o["x_max"] and o["y_min"] <= o["y_max"]):
                raise AnnotationError(f"{path}: frame {k} object {i} has an inverted box")
            o.setdefault("iscrowd", 0)                       # missing iscrowd = 0 (not a crowd region)
            if o["iscrowd"] not in (0, 1):
                raise AnnotationError(f"{path}: frame {k} object {i} has iscrowd={o['iscrowd']}")
        frames[fn] = objs
    if not frames:
        raise AnnotationError(f"{path}: no frame entries")
    return DetectionAnnotations(path=path, sha256=file_sha256(path), version=raw.get("version"),
                                frames=dict(sorted(frames.items())))


def has_track_ids(ann: DetectionAnnotations) -> bool:
    """True if any object has a track_id (tracking ground truth); without any, the tracking task is skipped."""
    return any("track_id" in o for objs in ann.frames.values() for o in objs)


def load_obb_annotations(path: Path) -> DetectionAnnotations:
    """<sequence_name>_obb_annotations.json: per frame [{"class_id": DOTA v1.0 0-based id, "bbox": [[x, y] * 4]}],
    corner points normalised to the frame (may lie slightly outside [0, 1] for objects cut by the frame edge).
    Optional "iscrowd" (missing = 0; 1 = ignore region)."""
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise AnnotationError(f"{path}: top level must be an object keyed by frame number")
    valid_ids = set(dota_label_map().values())
    frames: dict[int, list[dict]] = {}
    for k, objs in raw.items():
        if not str(k).isdigit():
            continue
        if int(k) < 1:
            raise AnnotationError(f"{path}: frame numbers start from 1, got '{k}'")
        if not isinstance(objs, list):
            raise AnnotationError(f"{path}: frame {k} must be a list of objects")
        for i, o in enumerate(objs):
            if "class_id" not in o or "bbox" not in o:
                raise AnnotationError(f"{path}: frame {k} object {i} needs class_id and bbox")
            if o["class_id"] not in valid_ids:
                raise AnnotationError(f"{path}: frame {k} object {i} has class_id {o['class_id']} not in "
                                      f"labels_dota.py (DOTA v1.0, 0..14)")
            b = o["bbox"]
            if (not isinstance(b, list) or len(b) != 4
                    or not all(isinstance(p, list) and len(p) == 2 for p in b)):
                raise AnnotationError(f"{path}: frame {k} object {i}: bbox must be 4 points [[x, y], ...]")
            o.setdefault("iscrowd", 0)
            if o["iscrowd"] not in (0, 1):
                raise AnnotationError(f"{path}: frame {k} object {i} has iscrowd={o['iscrowd']}")
        frames[int(k)] = objs
    if not frames:
        raise AnnotationError(f"{path}: no frame entries")
    return DetectionAnnotations(path=path, sha256=file_sha256(path), version=raw.get("version"),
                                frames=dict(sorted(frames.items())))


def load_bib_annotations(path: Path) -> DetectionAnnotations:
    """<sequence_name>_bib_annotations.json: per frame [{"track_id", "x_min", "x_max", "y_min", "y_max", "bib"}]
    (race bib boxes normalised to the frame and the bib's visible digits)."""
    return load_lpr_annotations(path, text_key="bib")


def load_lpr_annotations(path: Path, text_key: str = "plate") -> DetectionAnnotations:
    """<sequence_name>_lpr_annotations.json: per frame [{"track_id", "x_min", "x_max", "y_min", "y_max", "plate"}]
    (licence plate boxes normalised to the frame and the plate's visible characters). text_key: the text field
    ("bib" for <sequence_name>_bib_annotations.json)."""
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise AnnotationError(f"{path}: top level must be an object keyed by frame number")
    frames: dict[int, list[dict]] = {}
    for k, objs in raw.items():
        if not str(k).isdigit():
            continue
        if int(k) < 1:
            raise AnnotationError(f"{path}: frame numbers start from 1, got '{k}'")
        if not isinstance(objs, list):
            raise AnnotationError(f"{path}: frame {k} must be a list of plates")
        for i, o in enumerate(objs):
            if not isinstance(o, dict):
                raise AnnotationError(f"{path}: frame {k} plate {i} must be an object with named keys")
            missing = [r for r in ("x_min", "x_max", "y_min", "y_max", text_key) if r not in o]
            if missing:
                raise AnnotationError(f"{path}: frame {k} plate {i} missing keys {missing}")
            if not isinstance(o[text_key], str):
                raise AnnotationError(f"{path}: frame {k} plate {i}: '{text_key}' must be a string")
            if not (0 <= o["x_min"] <= o["x_max"] <= 1 and 0 <= o["y_min"] <= o["y_max"] <= 1):
                raise AnnotationError(f"{path}: frame {k} plate {i} has a box outside [0, 1] or inverted")
        frames[int(k)] = objs
    if not frames:
        raise AnnotationError(f"{path}: no frame entries")
    return DetectionAnnotations(path=path, sha256=file_sha256(path), version=raw.get("version"),
                                frames=dict(sorted(frames.items())))


def frames_to_evaluate(ann_frames: set[int], decoded_frames: set[int], label: str) -> tuple[list[int], list[str]]:
    """Frames scored = annotated frames. All must be decoded; decoded frames without annotations
    are not scored (warning)."""
    missing = sorted(ann_frames - decoded_frames)
    if missing:
        raise AnnotationError(f"{label}: {len(missing)} annotated frames have no decoded PNG "
                              f"(first {missing[:5]}); annotations and video do not match")
    warnings = []
    extra = sorted(decoded_frames - ann_frames)
    if extra:
        warnings.append(f"{label}: {len(extra)} decoded frames have no annotation entry and are not "
                        f"scored (first {extra[:5]})")
    return sorted(ann_frames), warnings


def load_panoptic_annotations(path: Path) -> DetectionAnnotations:
    """<sequence_name>_panoptic_annotations.json: per frame the segments {class_id (COCO panoptic id), mask_color,
    [iscrowd], [track_id], [class_name], [box]}. Same container as the detection annotations."""
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise AnnotationError(f"{path}: top level must be an object keyed by frame number")
    valid_ids = {cid for cid, _, _ in panoptic_categories()}
    frames: dict[int, list[dict]] = {}
    for k, segs in raw.items():
        if not str(k).isdigit():
            continue
        fn = int(k)
        if fn < 1:
            raise AnnotationError(f"{path}: frame numbers start from 1, got '{k}'")
        if not isinstance(segs, list):
            raise AnnotationError(f"{path}: frame {k} must be a list of segments")
        colours = set()
        for i, o in enumerate(segs):
            if "class_id" not in o or "mask_color" not in o:
                raise AnnotationError(f"{path}: frame {k} segment {i} needs class_id and mask_color")
            if o["class_id"] not in valid_ids:
                raise AnnotationError(f"{path}: frame {k} segment {i} has class_id {o['class_id']} that is not a "
                                      f"COCO panoptic category (task_models/labels_panoptic.py)")
            c = tuple(int(v) for v in o["mask_color"])
            if len(c) != 3 or c in colours:
                raise AnnotationError(f"{path}: frame {k} segment {i}: mask_color {list(c)} invalid or used twice")
            colours.add(c)
            o.setdefault("iscrowd", 0)
            if o["iscrowd"] not in (0, 1):
                raise AnnotationError(f"{path}: frame {k} segment {i} has iscrowd={o['iscrowd']}")
        frames[fn] = segs
    if not frames:
        raise AnnotationError(f"{path}: no frame entries")
    return DetectionAnnotations(path=path, sha256=file_sha256(path), version=raw.get("version"),
                                frames=dict(sorted(frames.items())))


N_KEYPOINTS = 17     # COCO person keypoints


def load_pose_annotations(path: Path) -> DetectionAnnotations:
    """<sequence_name>_pose_annotations.json: per frame the persons {class_id 1, keypoints (17 x [x, y, v], x/y
    normalised), area (fraction of the image area), [iscrowd]}. A box, if present, is ignored.
    A person without labelled keypoints (all v = 0) is ignored by COCO keypoint AP and needs no area."""
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise AnnotationError(f"{path}: top level must be an object keyed by frame number")
    frames: dict[int, list[dict]] = {}
    for k, objs in raw.items():
        if not str(k).isdigit():
            continue
        fn = int(k)
        if fn < 1:
            raise AnnotationError(f"{path}: frame numbers start from 1, got '{k}'")
        if not isinstance(objs, list):
            raise AnnotationError(f"{path}: frame {k} must be a list of persons")
        for i, o in enumerate(objs):
            where = f"{path}: frame {k} person {i}"
            if o.get("class_id") != 1:
                raise AnnotationError(f"{where}: class_id must be 1 (person), got {o.get('class_id')}")
            kp = o.get("keypoints")
            if not isinstance(kp, list) or len(kp) != 3 * N_KEYPOINTS:
                raise AnnotationError(f"{where}: keypoints must be a list of {3 * N_KEYPOINTS} numbers "
                                      f"(17 COCO keypoints x, y, v)")
            if any(v not in (0, 1, 2) for v in kp[2::3]):
                raise AnnotationError(f"{where}: keypoint visibility must be 0, 1 or 2")
            o.setdefault("iscrowd", 0)
            if o["iscrowd"] not in (0, 1):
                raise AnnotationError(f"{where} has iscrowd={o['iscrowd']}")
            o["num_keypoints"] = sum(1 for v in kp[2::3] if v > 0)
            a = o.get("area")
            if o["num_keypoints"] and not (isinstance(a, (int, float)) and 0 < a <= 1):
                raise AnnotationError(f"{where}: needs 'area' = segmentation area as a fraction of the image "
                                      f"area (0 < area <= 1), got {a!r}")
        frames[fn] = objs
    if not frames:
        raise AnnotationError(f"{path}: no frame entries")
    return DetectionAnnotations(path=path, sha256=file_sha256(path), version=raw.get("version"),
                                frames=dict(sorted(frames.items())))
