"""YAML configuration: defaults, loading, strict validation.

All paths in the config are relative to the project root (the directory that
contains the `vcmtest/` package) unless they are absolute.
Unknown keys are an error, so a typo never silently falls back to a default.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from . import metric_options as _metric_options
from . import wrappers as _wrappers

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"

# Task name -> annotation type it needs (see docs/folder_layout.txt).
TASK_ANNOTATION_TYPE = {
    "detection": "detection",
    "instance_segmentation": "detection",  # needs <sequence_name>_masks/ or polygons too
    "tracking": "detection",
    "keypoints": "pose",
    "panoptic_segmentation": "panoptic",     # <sequence_name>_panoptic_annotations.json + _panoptic_masks/
    "semantic_segmentation": "panoptic",     # same annotations, segments merged by class
    "depth": "depth",
    "oriented_detection": "obb",            # <sequence_name>_obb_annotations.json (DOTA v1.0 classes, 4-point boxes)
    "plate_recognition": "lpr",             # <sequence_name>_lpr_annotations.json (plate boxes + characters)
    "bib_recognition": "bib",               # <sequence_name>_bib_annotations.json (race bib boxes + numbers)
}
IMPLEMENTED_TASKS = {"detection", "instance_segmentation", "tracking", "panoptic_segmentation",
                     "semantic_segmentation", "keypoints", "depth",
                     "oriented_detection", "plate_recognition", "bib_recognition"}
PANOPTIC_TASKS = {"panoptic_segmentation", "semantic_segmentation"}   # one panoptic prediction serves both

# Model families: the registry in vcmtest/wrappers.py (one entry per wrapper); the sets below are derived from it.
WRAPPERS = set(_wrappers.WRAPPERS)
# scene-text readers (any text in the frame): the bib_recognition task
SCENE_TEXT_WRAPPERS = {"rapidocr", "easyocr"}
LPR_WRAPPERS = {"fast_alpr", "plate_two_stage"}   # plate_recognition task
PLATE_OCR_READERS = {"fast_plate_ocr", "rapidocr", "easyocr"}   # plate_two_stage ocr: <reader>:<model>
# Python packages each wrapper needs (optional extras in pyproject.toml) and how to install them.
WRAPPER_REQUIREMENTS = {k: (list(v.requirements), v.install) for k, v in _wrappers.WRAPPERS.items()}
# wrappers that run in their own venv (worker subprocess): config key of that venv's python
WORKER_PYTHON = {k: v.worker for k, v in _wrappers.WRAPPERS.items() if v.worker}


def missing_model_requirements(cfg: dict) -> list[str]:
    """One message per configured wrapper whose Python packages (or venv) are not installed."""
    import importlib.util
    msgs = []
    wrappers = sorted({m["wrapper"] for t in cfg["tasks"] for m in t["models"]})
    if any(m.get("tracker") and not is_ultralytics_tracker(m["tracker"]) for t in cfg["tasks"] for m in t["models"]):
        if importlib.util.find_spec("trackers") is None:
            msgs.append("tracker (Roboflow trackers) needs trackers: uv sync --extra trackers")
    for w in wrappers:
        mods, how = WRAPPER_REQUIREMENTS[w]
        missing = [m for m in mods if importlib.util.find_spec(m) is None]
        if missing:
            msgs.append(f"wrapper '{w}' needs {', '.join(missing)}: {how}")
        if w in WORKER_PYTHON and not resolve_path(cfg[WORKER_PYTHON[w]]).is_file():
            msgs.append(f"wrapper '{w}' needs its venv python {cfg[WORKER_PYTHON[w]]}: {how}")
    return msgs


# RF-DETR detection XLarge/2XLarge need a Roboflow API key / RF-DETR+ licence: not allowed.
# (inference aliases rfdetr-xlarge -> coco/47, rfdetr-2xlarge -> coco/48)
LICENSED_MODELS = {"roboflow": {"rfdetr-xlarge", "rfdetr-2xlarge", "coco/47", "coco/48"},
                   "rfdetr": {"rfdetr-xlarge", "rfdetr-2xlarge"}}

DEFAULTS: dict[str, Any] = {
    "data_root": "data",
    "annotations_root": "All_annotations",
    "results_root": "results",
    "tools_dir": "tools",
    "weights_dir": "weights",   # downloaded model weights (torch hub, Hugging Face, Ultralytics, Roboflow)
    "roboflow_python": ".venv-roboflow/bin/python",   # venv for the 'roboflow' wrapper (envs/roboflow)
    "mmrotate_python": ".venv-mmrotate/bin/python",   # venv for the 'mmrotate' wrapper (envs/mmrotate/setup.sh)
    "ffmpeg": None,             # None -> 'ffmpeg' from PATH; a name -> <tools_dir>/<name>
    "sequences": "all",         # "all" or list of sequence folder names
    "qps": "all",               # "all" or list of ints
    "framerate": None,          # fallback when data/<seq>/info.json has no "framerate"
    "device": "cuda:0",
    "seed": 0,
    # + the plotted-metric options of vcmtest/metric_options.py (ap_average, hota_average, pq_average, depth_eval,
    #   depth_metric, obb_score), added below
    "lpr_iou": 0.5,             # plate_recognition / bib_recognition: box IoU for a predicted box to match a GT one
    "depth_min_mm": 0,          # depth GT: pixels with depth < this (mm) are ignored (0 = only zeros, = no depth)
    "input_size": {},           # {<task>: {<model_id>: N}}: network input size on every sequence (sequence_overrides.
                                # <seq>.input_size wins per task and model; null there = the model's default)
    "sequence_overrides": {},   # {<seq>: {annotations, framerate, qps, tasks, models, classes, ap_average,
                                #          pq_average, depth_min_mm, depth_eval, depth_metric, remove_depth_maps,
                                #          obb_classes, obb_score, lpr_iou, input_size}}
    "decode": {
        "matrix": "bt709",      # bt709 | bt601 | bt2020
        "range": "full",        # full | limited
        "png_bit_depth": 8,     # 8 | 16
        "verify_md5": True,     # compare decoded planes with encoder log MD5s when available
        "remove_frames": False,
        "min_free_gb": 20,
        "timeout_s": 3600,      # per bitstream
        "decoder": "ffmpeg",    # ffmpeg | vtm (vcmtest/decoders.py); not ffmpeg -> part of the decode settings
        "decoder_executable": None,   # null = the decoder's default name (vtm: DecoderAppStatic) in tools_dir
    },
    "plot": {
        "log_x": False,         # bitrate axis linear (true = log scale)
        "y_from_zero": True,    # y axis starts at 0 (false = fitted to the data); also per sequence
    },
    "inference": {
        "timeout_s": 7200,      # per (sequence, QP, model)
        "remove_depth_maps": False,  # delete predicted depth map PNGs (QPxx_depth/) once scored
    },
    "bd": {                     # Bjontegaard comparison of test encodings against an anchor encoding (vcmtest/bd.py)
        "enabled": False,       # run the BD step after the sequences of a run (--bd runs it alone)
        "anchors_file": "configs/anchors.yaml",   # {anchors: {<sequence_name>: <anchor folder>}}; missing file = none
        "anchors": {},          # {<sequence_name>: <anchor folder> | null}: overrides the anchors file per name
        "qps": {},              # {<test folder>: {anchor: [qp, ...], test: [qp, ...]}}: points used (default: all)
        "min_points": 4,        # Pareto points needed on each curve
        "log_x": True,          # BD plots: log bitrate axis (independent of plot.log_x)
        "extended": True,       # extended analysis per pair (vcmtest/bd_extended.py): sub-metric BD, model agreement
        "out_dir": "BD-results",
    },
    "tasks": [],
}

DEFAULTS.update({name: o.default for name, o in _metric_options.METRIC_OPTIONS.items()})   # plotted-metric options

_SEQ_OVERRIDE_KEYS = {"annotations", "framerate", "tasks", "models", "classes", "qps", "depth_min_mm",
                      "remove_depth_maps", "obb_classes", "lpr_iou", "input_size",
                      "y_from_zero"} | set(_metric_options.METRIC_OPTIONS)
# input_size (global) and sequence_overrides.<seq>.input_size: {<task>: {<model_id>: N}} = network input size for
# that model (global: on every sequence; the sequence's entry wins per task and model, null = the model's default)
# (N = positive multiple of 32; Ultralytics imgsz: the longer frame side is resized to N, letterboxed;
# rfdetr: square N x N `resolution` (N also a multiple of the model's patch_size x num_windows); rtdetr: stretched to
# N x N like its 640 x 640 default).
# Implemented for these wrappers; for any other wrapper the model is not run on that sequence (warning, listed as
# skipped). The model id gets "-<N>px" (separate predictions); the override is keyed by the configured model id.
INPUT_SIZE_WRAPPERS = {k for k, v in _wrappers.WRAPPERS.items() if v.input_size}
_TASK_KEYS = {"task", "models"}
_MODEL_KEYS = {"wrapper", "model", "score_threshold", "name", "tracker", "detector", "ocr", "input_size"}
# mmrotate `input_size`: tiles (default; DOTA test protocol: 1024 x 1024 tiles at full resolution, 200 px overlap,
# merged with rotated NMS) | N (whole frame resized to fit N x N, as the released pipelines with N = 1024). DOTA
# models are trained on 1024 crops at full resolution, so resizing a 1920 frame to 1024 halves the objects.
# Always part of the mmrotate model id.
# hf_vitpose (keypoints task): `detector:` = model id of a detection-task model whose cached person boxes on the
# same decoded frame are the ViTPose input boxes.
# Tracking: `tracker:` is either an Ultralytics tracker YAML (name ending in .yaml, or a path; ultralytics
# wrapper only) or an algorithm of the Roboflow `trackers` package applied to any detection wrapper.
TRACKERS_ALGORITHMS = {"bytetrack", "ocsort", "sort", "botsort", "cbiou"}
DEFAULT_TRACKER = "botsort.yaml"     # Ultralytics default tracker (ultralytics wrapper only)


def is_ultralytics_tracker(tracker: str) -> bool:
    """True if the tracker is an Ultralytics tracker YAML (name ending in .yaml or .yml)."""
    return str(tracker).endswith((".yaml", ".yml"))


class ConfigError(ValueError):
    """Invalid configuration (unknown key, invalid value, missing file)."""
    pass


def _merge(defaults: dict, user: dict, where: str) -> dict:
    out = copy.deepcopy(defaults)
    for k, v in user.items():
        if k not in defaults:
            raise ConfigError(f"Unknown config key '{where}{k}'. Allowed: {sorted(defaults)}")
        if isinstance(defaults[k], dict) and defaults[k] and isinstance(v, dict):
            out[k] = _merge(defaults[k], v, f"{where}{k}.")
        else:
            out[k] = v
    return out


def resolve_path(p: str | Path) -> Path:
    """Path relative to the project root unless absolute (~ expanded)."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def project_relative(p: str | Path) -> str:
    """A path as recorded in output files: relative to the project root when inside it, else absolute."""
    p = resolve_path(p).resolve()
    return str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)


# Plotted-metric options (registry: vcmtest/metric_options.py): value -> field in the scores result
def _option_fields(name: str) -> dict:
    return {k: v.stats for k, v in _metric_options.METRIC_OPTIONS[name].variants.items()}


AP_AVERAGES = _option_fields("ap_average")     # class = COCO mean of per-class AP | object = object-weighted
PQ_AVERAGES = _option_fields("pq_average")     # all = panopticapi (TP+FP+FN > 0) | gt = GT classes only
OBB_SCORES = _option_fields("obb_score")       # dota = DOTA classes | vehicle = small + large vehicle merged
DEPTH_EVALS = _option_fields("depth_eval")     # raw = metric prediction as is | aligned = per-frame scale+shift


def sequence_ap_average(cfg: dict, seq: str) -> str:
    """ap_average value for a sequence (sequence override, else global, else default)."""
    return _metric_options.option_value(cfg, seq, "ap_average")


def sequence_obb_score(cfg: dict, seq: str) -> str:
    """obb_score value for a sequence (sequence override, else global, else default)."""
    return _metric_options.option_value(cfg, seq, "obb_score")


def sequence_lpr_iou(cfg: dict, seq: str) -> float:
    """plate_recognition / bib_recognition: minimum box IoU of a prediction with a ground-truth box to match it."""
    ov = (cfg.get("sequence_overrides") or {}).get(seq) or {}
    return float(ov.get("lpr_iou", cfg.get("lpr_iou", 0.5)))


def _valid_iou(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0 < v <= 1


def sequence_pq_average(cfg: dict, seq: str) -> str:
    """pq_average value for a sequence (sequence override, else global, else default)."""
    return _metric_options.option_value(cfg, seq, "pq_average")


def sequence_depth_eval(cfg: dict, seq: str) -> str:
    """depth_eval value for a sequence (sequence override, else global, else default)."""
    return _metric_options.option_value(cfg, seq, "depth_eval")


def sequence_depth_min_mm(cfg: dict, seq: str) -> float:
    """Depth ground truth below this (mm) is ignored in depth scoring; depth 0 (no depth) is always ignored."""
    ov = (cfg.get("sequence_overrides") or {}).get(seq) or {}
    return ov.get("depth_min_mm", cfg.get("depth_min_mm", 0))


def sequence_remove_depth_maps(cfg: dict, seq: str) -> bool:
    """Delete the predicted depth map PNGs once scored: --remove-depth-maps > sequence override > inference.*"""
    if cfg.get("_cli_remove_depth_maps"):
        return True
    ov = (cfg.get("sequence_overrides") or {}).get(seq) or {}
    return bool(ov.get("remove_depth_maps", cfg["inference"].get("remove_depth_maps", False)))


def sequence_y_from_zero(cfg: dict, seq: str) -> bool:
    """RD plots of a sequence: y axis from 0 (true) or fitted to the data: sequence override > plot.y_from_zero."""
    ov = (cfg.get("sequence_overrides") or {}).get(seq) or {}
    return bool(ov.get("y_from_zero", (cfg.get("plot") or {}).get("y_from_zero", True)))


def _valid_depth_min(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v >= 0


def _valid_qps(qps) -> bool:
    return qps == "all" or (isinstance(qps, list) and bool(qps)
                            and all(isinstance(q, int) and not isinstance(q, bool) for q in qps))


def sequence_qps(cfg: dict, seq: str, cli_qps=None):
    """QP selection for one sequence and where it came from: --qp > sequence_overrides.<seq>.qps > qps."""
    if cli_qps:
        return cli_qps, "--qp"
    ov = (cfg.get("sequence_overrides") or {}).get(seq) or {}
    if "qps" in ov:
        return ov["qps"], f"sequence_overrides.{seq}.qps"
    return cfg["qps"], "qps"


def sequence_result_qps(cfg: dict, seq: str) -> list[int] | None:
    """QPs a sequence's results and plots are limited to: sequence_overrides.<seq>.qps when it is a list (QP folders
    on disk outside it are ignored, files kept), else None = every QP folder with valid data."""
    q = ((cfg.get("sequence_overrides") or {}).get(seq) or {}).get("qps")
    return q if isinstance(q, list) else None


def _validate(cfg: dict) -> None:
    sel = cfg["sequences"]
    if not (sel == "all" or (isinstance(sel, list) and all(isinstance(s, str) for s in sel))):
        raise ConfigError("'sequences' must be 'all' or a list of sequence folder names")
    for name, opt in _metric_options.METRIC_OPTIONS.items():          # plotted-metric options
        if cfg[name] not in opt.variants:
            raise ConfigError(f"'{name}' must be one of {list(opt.variants)}")
    if not _valid_iou(cfg["lpr_iou"]):
        raise ConfigError("'lpr_iou' must be a number in (0, 1]")
    if not _valid_qps(cfg["qps"]):
        raise ConfigError("'qps' must be 'all' or a list of integers, e.g. [22, 47]")
    if not _valid_depth_min(cfg["depth_min_mm"]):
        raise ConfigError("'depth_min_mm' must be a number >= 0 (millimetres)")
    fr = cfg["framerate"]
    if fr is not None and (not isinstance(fr, (int, float)) or fr <= 0):
        raise ConfigError("'framerate' must be a positive number or null")
    bd = cfg["bd"]
    if not isinstance(bd["enabled"], bool):
        raise ConfigError("bd.enabled must be true or false")
    if not isinstance(bd["log_x"], bool):
        raise ConfigError("bd.log_x must be true or false")
    if not isinstance(cfg["plot"].get("y_from_zero"), bool):
        raise ConfigError("plot.y_from_zero must be true or false")
    if not isinstance(bd["extended"], bool):
        raise ConfigError("bd.extended must be true or false")
    if not isinstance(bd["min_points"], int) or isinstance(bd["min_points"], bool) or bd["min_points"] < 2:
        raise ConfigError("bd.min_points must be an integer >= 2")
    if not isinstance(bd["anchors"] or {}, dict):
        raise ConfigError("bd.anchors must be a mapping {<sequence_name>: <anchor folder>}")
    if not isinstance(bd["qps"] or {}, dict):
        raise ConfigError("bd.qps must be a mapping {<test folder>: {anchor: [qp, ...], test: [qp, ...]}}")
    for test, sel in (bd["qps"] or {}).items():
        if not isinstance(sel, dict) or set(sel) - {"anchor", "test"} or \
                not all(isinstance(v, list) and v and all(isinstance(q, int) for q in v) for v in sel.values()):
            raise ConfigError(f"bd.qps.{test} must be {{anchor: [qp, ...], test: [qp, ...]}} (either may be omitted)")

    for seq, ov in (cfg["sequence_overrides"] or {}).items():
        if not isinstance(ov, dict) or set(ov) - _SEQ_OVERRIDE_KEYS:
            raise ConfigError(f"sequence_overrides.{seq}: allowed keys {sorted(_SEQ_OVERRIDE_KEYS)}")
        if "y_from_zero" in ov and not isinstance(ov["y_from_zero"], bool):
            raise ConfigError(f"sequence_overrides.{seq}.y_from_zero must be true or false")

    d = cfg["decode"]
    if d["matrix"] not in ("bt709", "bt601", "bt2020"):
        raise ConfigError("decode.matrix must be bt709, bt601 or bt2020")
    if d["range"] not in ("full", "limited"):
        raise ConfigError("decode.range must be full or limited")
    if d["png_bit_depth"] not in (8, 16):
        raise ConfigError("decode.png_bit_depth must be 8 or 16")
    from .decoders import DECODERS
    if d["decoder"] not in DECODERS:
        raise ConfigError(f"decode.decoder must be one of {sorted(DECODERS)}")
    if not isinstance(cfg["inference"].get("remove_depth_maps"), bool):
        raise ConfigError("inference.remove_depth_maps must be true or false")

    if not isinstance(cfg["tasks"], list) or not cfg["tasks"]:
        raise ConfigError("'tasks' must be a non-empty list")
    for i, t in enumerate(cfg["tasks"]):
        if not isinstance(t, dict) or set(t) - _TASK_KEYS or "task" not in t:
            raise ConfigError(f"tasks[{i}]: needs 'task' and 'models'; allowed keys {sorted(_TASK_KEYS)}")
        if t["task"] not in TASK_ANNOTATION_TYPE:
            raise ConfigError(f"tasks[{i}].task '{t['task']}' unknown. Known: {sorted(TASK_ANNOTATION_TYPE)}")
        models = t.get("models")
        if not isinstance(models, list) or not models:
            raise ConfigError(f"tasks[{i}].models must be a non-empty list")
        for j, m in enumerate(models):
            if not isinstance(m, dict) or set(m) - _MODEL_KEYS or not {"wrapper", "model"} <= set(m):
                raise ConfigError(f"tasks[{i}].models[{j}]: needs 'wrapper' and 'model'; allowed {sorted(_MODEL_KEYS)}")
            if m["wrapper"] not in WRAPPERS:
                raise ConfigError(f"tasks[{i}].models[{j}].wrapper must be one of {sorted(WRAPPERS)}")
            if str(m["model"]).strip().lower() in LICENSED_MODELS.get(m["wrapper"], set()):
                raise ConfigError(f"tasks[{i}].models[{j}]: model '{m['model']}' is not allowed: RF-DETR detection "
                                  f"XLarge/2XLarge require a Roboflow API key / RF-DETR+ licence. Use rfdetr-nano, "
                                  f"-small, -medium or -large (or a rfdetr-seg-* model) instead.")
            if t["task"] == "tracking":
                if m["wrapper"] == "ultralytics":
                    m.setdefault("tracker", DEFAULT_TRACKER)
                tr = m.get("tracker")
                if not isinstance(tr, str) or not tr:
                    raise ConfigError(f"tasks[{i}].models[{j}]: tracking needs 'tracker': one of "
                                      f"{sorted(TRACKERS_ALGORITHMS)} (Roboflow trackers, any detection wrapper) or "
                                      f"an Ultralytics tracker YAML such as botsort.yaml (ultralytics wrapper)")
                if is_ultralytics_tracker(tr):
                    if m["wrapper"] != "ultralytics":
                        raise ConfigError(f"tasks[{i}].models[{j}]: Ultralytics tracker '{tr}' needs the ultralytics "
                                          f"wrapper; use one of {sorted(TRACKERS_ALGORITHMS)} with '{m['wrapper']}'")
                elif tr not in TRACKERS_ALGORITHMS:
                    raise ConfigError(f"tasks[{i}].models[{j}]: unknown tracker '{tr}'; use one of "
                                      f"{sorted(TRACKERS_ALGORITHMS)} or an Ultralytics tracker YAML (*.yaml)")
            elif "tracker" in m:
                raise ConfigError(f"tasks[{i}].models[{j}]: 'tracker' is only valid in the tracking task")
            if (t["task"] in PANOPTIC_TASKS) != (m["wrapper"] == "hf_panoptic"):
                raise ConfigError(f"tasks[{i}].models[{j}]: the {t['task']} task needs the hf_panoptic wrapper"
                                  if t["task"] in PANOPTIC_TASKS else
                                  f"tasks[{i}].models[{j}]: the hf_panoptic wrapper is only for the "
                                  f"panoptic_segmentation and semantic_segmentation tasks")
            if (t["task"] == "depth") != (m["wrapper"] == "hf_depth"):
                raise ConfigError(f"tasks[{i}].models[{j}]: the depth task needs the hf_depth wrapper"
                                  if t["task"] == "depth" else
                                  f"tasks[{i}].models[{j}]: the hf_depth wrapper is only for the depth task")
            if m["wrapper"] == "hf_depth" and m.get("score_threshold") is not None:
                raise ConfigError(f"tasks[{i}].models[{j}]: hf_depth has no score threshold (use null)")
            if (t["task"] == "plate_recognition") != (m["wrapper"] in LPR_WRAPPERS):
                raise ConfigError(f"tasks[{i}].models[{j}]: the plate_recognition task needs a plate reader "
                                  f"({sorted(LPR_WRAPPERS)})" if t["task"] == "plate_recognition" else
                                  f"tasks[{i}].models[{j}]: plate readers ({m['wrapper']}) are only for the "
                                  f"plate_recognition task")
            if (t["task"] == "bib_recognition") != (m["wrapper"] in SCENE_TEXT_WRAPPERS):
                raise ConfigError(f"tasks[{i}].models[{j}]: the bib_recognition task needs a scene-text reader "
                                  f"({sorted(SCENE_TEXT_WRAPPERS)})" if t["task"] == "bib_recognition" else
                                  f"tasks[{i}].models[{j}]: scene-text readers ({m['wrapper']}) are only for the "
                                  f"bib_recognition task")
            if m["wrapper"] == "fast_alpr" and not m.get("ocr"):
                raise ConfigError(f"tasks[{i}].models[{j}]: fast_alpr needs 'ocr' (fast-plate-ocr model, e.g. "
                                  f"cct-s-v2-global-model)")
            if m["wrapper"] == "plate_two_stage":
                ocr = str(m.get("ocr") or "")
                reader, sep, rmodel = ocr.partition(":")
                if not sep or not rmodel or reader not in PLATE_OCR_READERS:
                    raise ConfigError(f"tasks[{i}].models[{j}]: plate_two_stage needs 'ocr: <reader>:<model>' with "
                                      f"reader in {sorted(PLATE_OCR_READERS)}, e.g. "
                                      f"fast_plate_ocr:cct-s-v2-global-model, rapidocr:PP-OCRv5-server, easyocr:en")
            if "input_size" in m:
                v = m["input_size"]
                if m["wrapper"] != "mmrotate":
                    raise ConfigError(f"tasks[{i}].models[{j}]: 'input_size' is only valid for the mmrotate wrapper")
                if not (v == "tiles" or (isinstance(v, int) and not isinstance(v, bool) and v > 0 and v % 32 == 0)):
                    raise ConfigError(f"tasks[{i}].models[{j}]: input_size must be 'tiles' or a positive multiple "
                                      f"of 32 (e.g. 1024)")
            if "ocr" in m and m["wrapper"] not in LPR_WRAPPERS:
                raise ConfigError(f"tasks[{i}].models[{j}]: 'ocr' is only valid for the plate readers "
                                  f"({sorted(LPR_WRAPPERS)})")
            is_obb = (m["wrapper"] == "ultralytics" and Path(str(m["model"])).stem.endswith("-obb")) \
                or m["wrapper"] == "mmrotate"
            if (t["task"] == "oriented_detection") != is_obb:
                raise ConfigError(f"tasks[{i}].models[{j}]: the oriented_detection task needs an OBB model "
                                  f"(ultralytics yolo26*-obb.pt or wrapper mmrotate)"
                                  if t["task"] == "oriented_detection"
                                  else f"tasks[{i}].models[{j}]: OBB models (*-obb.pt, mmrotate) are only for the "
                                       f"oriented_detection task")
            if t["task"] == "keypoints":
                name = str(m["model"])
                is_kp = _wrappers.WRAPPERS[m["wrapper"]].keypoints
                ok = bool(is_kp and is_kp(name))
                if not ok:
                    raise ConfigError(f"tasks[{i}].models[{j}]: the keypoints task needs a keypoint model: torchvision "
                                      f"keypointrcnn_resnet50_fpn, ultralytics yolo26*-pose.pt, rfdetr "
                                      f"rfdetr-keypoint-preview or hf_vitpose")
            elif m["wrapper"] == "hf_vitpose":
                raise ConfigError(f"tasks[{i}].models[{j}]: the hf_vitpose wrapper is only for the keypoints task")
            if (m["wrapper"] == "hf_vitpose") != ("detector" in m):
                raise ConfigError(f"tasks[{i}].models[{j}]: " + ("hf_vitpose needs 'detector': the model id of a "
                                  "detection-task model (its person boxes are the ViTPose input)"
                                  if m["wrapper"] == "hf_vitpose" else "'detector' is only valid with hf_vitpose"))
            st = m.get("score_threshold")
            if st is not None and not (isinstance(st, (int, float)) and 0.0 <= st <= 1.0):
                raise ConfigError(f"tasks[{i}].models[{j}].score_threshold must be null or in [0, 1]")

    # per-sequence model selection: a subset of each task's models, by model id (after defaults are applied)
    task_ids = {t["task"]: [model_id(m) for m in t["models"]] for t in cfg["tasks"]}
    order = [t["task"] for t in cfg["tasks"]]
    for i, t in enumerate(cfg["tasks"]):
        for j, m in enumerate(t["models"]):
            if "detector" not in m:
                continue
            if m["detector"] not in task_ids.get("detection", []):
                raise ConfigError(f"tasks[{i}].models[{j}].detector '{m['detector']}' is not a model id of the "
                                  f"detection task (configured: {task_ids.get('detection', [])}; see --plan)")
            if order.index("detection") > i:
                raise ConfigError(f"tasks[{i}]: the detection task must be listed before {t['task']} "
                                  f"(its predictions are the input boxes of {model_id(m)})")
    if cfg.get("input_size") is None:
        cfg["input_size"] = {}
    if cfg["input_size"] != {}:
        _check_input_size(cfg["input_size"], "input_size", task_ids, allow_null=False)
    for seq, ov in (cfg["sequence_overrides"] or {}).items():
        if "classes" in ov:
            sequence_class_ids(cfg, seq)        # validates the names
        if "obb_classes" in ov:
            sequence_obb_class_ids(cfg, seq)
        for name, opt in _metric_options.METRIC_OPTIONS.items():
            if name in ov and ov[name] not in opt.variants:
                raise ConfigError(f"sequence_overrides.{seq}.{name} must be one of {list(opt.variants)}")
        if "remove_depth_maps" in ov and not isinstance(ov["remove_depth_maps"], bool):
            raise ConfigError(f"sequence_overrides.{seq}.remove_depth_maps must be true or false")
        if "depth_min_mm" in ov and not _valid_depth_min(ov["depth_min_mm"]):
            raise ConfigError(f"sequence_overrides.{seq}.depth_min_mm must be a number >= 0 (millimetres)")
        if "lpr_iou" in ov and not _valid_iou(ov["lpr_iou"]):
            raise ConfigError(f"sequence_overrides.{seq}.lpr_iou must be a number in (0, 1]")
        if "qps" in ov and not _valid_qps(ov["qps"]):
            raise ConfigError(f"sequence_overrides.{seq}.qps must be 'all' or a non-empty list of integers, "
                              f"e.g. [37, 42, 47, 50, 52]")
        if ov.get("input_size") is not None:
            _check_input_size(ov["input_size"], f"sequence_overrides.{seq}.input_size", task_ids, allow_null=True)
        if "tasks" in ov:
            ts = ov["tasks"]
            if not isinstance(ts, list) or not ts or not all(isinstance(x, str) for x in ts):
                raise ConfigError(f"sequence_overrides.{seq}.tasks must be a non-empty list of task names")
            for x in ts:
                if x not in task_ids:
                    raise ConfigError(f"sequence_overrides.{seq}.tasks: task '{x}' is not in 'tasks' "
                                      f"(configured: {sorted(task_ids)})")
        sel = ov.get("models")
        if sel is None:
            continue
        if not isinstance(sel, dict) or not sel:
            raise ConfigError(f"sequence_overrides.{seq}.models must be a mapping {{<task>: [<model_id>, ...]}}")
        for task, ids in sel.items():
            if task not in task_ids:
                raise ConfigError(f"sequence_overrides.{seq}.models: task '{task}' is not in 'tasks' "
                                  f"(configured: {sorted(task_ids)})")
            if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
                raise ConfigError(f"sequence_overrides.{seq}.models.{task} must be a non-empty list of model ids")
            unknown = [x for x in ids if x not in task_ids[task]]
            if unknown:
                raise ConfigError(f"sequence_overrides.{seq}.models.{task}: unknown model id(s) {unknown}; "
                                  f"configured for {task}: {task_ids[task]} (see --plan)")


# tasks that sequence_overrides.<seq>.classes can restrict (panoptic / semantic: thing classes in the averages)
CLASS_TASKS = ("detection", "instance_segmentation", "tracking", "panoptic_segmentation", "semantic_segmentation")


def _class_name_ids(names, where: str) -> list[int]:
    from .annotations import coco_label_map
    labels = coco_label_map()
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ConfigError(f"{where} must be a non-empty list of class names, e.g. [car]")
    unknown = [n for n in names if n not in labels]
    if unknown:
        raise ConfigError(f"{where}: unknown class name(s) {unknown} "
                          f"(names from task_models/labels.py, e.g. car, person, truck)")
    return sorted({labels[n] for n in names})


def sequence_class_ids(cfg: dict, seq: str) -> list[int] | dict[str, list[int]] | None:
    """Class ids to evaluate for one sequence (sequence_overrides.<seq>.classes, COCO names), or None = all.

    A list applies to every task in CLASS_TASKS; a mapping {<task>: [names]} only to the tasks it names (e.g.
    {tracking: [person]}), returned as {task: ids}. Use task_class_ids() for the ids of one task.
    Detection, instance segmentation and tracking: ground truth and predictions of other classes are left out of
    the scoring (as if only these categories existed). Panoptic / semantic segmentation: only these thing classes
    (and all stuff classes) are kept in the PQ / mIoU averages."""
    names = ((cfg.get("sequence_overrides") or {}).get(seq) or {}).get("classes")
    if names is None:
        return None
    where = f"sequence_overrides.{seq}.classes"
    if isinstance(names, dict):
        if not names:
            raise ConfigError(f"{where} must be a non-empty list of class names or a mapping {{<task>: [names]}}")
        bad = [t for t in names if t not in CLASS_TASKS]
        if bad:
            raise ConfigError(f"{where}: task(s) {bad} cannot be restricted by classes (allowed: {list(CLASS_TASKS)})")
        return {t: _class_name_ids(v, f"{where}.{t}") for t, v in names.items()}
    return _class_name_ids(names, where)


def task_class_ids(classes, task: str) -> list[int] | None:
    """Class ids of one task from sequence_class_ids()'s value (list = all CLASS_TASKS, mapping = per task)."""
    if classes is None or task not in CLASS_TASKS:
        return None
    if isinstance(classes, dict):
        return classes.get(task)
    return classes


def sequence_obb_class_ids(cfg: dict, seq: str) -> list[int] | None:
    """DOTA class ids scored in oriented_detection for one sequence (sequence_overrides.<seq>.obb_classes, DOTA
    v1.0 names from task_models/labels_dota.py), or None = all. Other classes' GT and predictions are left out."""
    names = ((cfg.get("sequence_overrides") or {}).get(seq) or {}).get("obb_classes")
    if names is None:
        return None
    from .annotations import dota_label_map
    labels = dota_label_map()
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ConfigError(f"sequence_overrides.{seq}.obb_classes must be a non-empty list of DOTA class names, "
                          f"e.g. [small vehicle]")
    unknown = [n for n in names if n not in labels]
    if unknown:
        raise ConfigError(f"sequence_overrides.{seq}.obb_classes: unknown class name(s) {unknown} "
                          f"(DOTA v1.0 names from task_models/labels_dota.py, e.g. small vehicle, large vehicle)")
    return sorted({labels[n] for n in names})



def _check_input_size(isz, where: str, task_ids: dict, allow_null: bool) -> None:
    """Validate {<task>: {<model_id>: N}} (N positive multiple of 32; null only in a sequence override)."""
    if not isinstance(isz, dict) or not isz:
        raise ConfigError(f"{where} must be a mapping {{<task>: {{<model_id>: <size>}}}}")
    for task, sizes in isz.items():
        if task not in task_ids:
            raise ConfigError(f"{where}: task '{task}' is not in 'tasks' (configured: {sorted(task_ids)})")
        if not isinstance(sizes, dict) or not sizes:
            raise ConfigError(f"{where}.{task} must be a mapping {{<model_id>: <size>}}")
        for mid, v in sizes.items():
            if mid not in task_ids[task]:
                raise ConfigError(f"{where}.{task}: unknown model id '{mid}'; configured for {task}: "
                                  f"{task_ids[task]} (see --plan)")
            if v is None and allow_null:
                continue
            if not (isinstance(v, int) and not isinstance(v, bool) and v > 0 and v % 32 == 0):
                raise ConfigError(f"{where}.{task}.{mid} must be a positive multiple of 32 (e.g. 1280)"
                                  + (" or null (model default)" if allow_null else ""))


def sequence_tasks(cfg: dict, seq: str) -> list[str] | None:
    """sequence_overrides.<seq>.tasks (only these tasks run on the sequence) or None (all)."""
    return ((cfg.get("sequence_overrides") or {}).get(seq) or {}).get("tasks")

def sequence_models(cfg: dict, seq: str, task: dict) -> list[dict]:
    """The task's models for one sequence: all of them, or the subset selected in sequence_overrides.<seq>.models."""
    sel = ((cfg.get("sequence_overrides") or {}).get(seq) or {}).get("models") or {}
    if task["task"] not in sel:
        return task["models"]
    keep = set(sel[task["task"]])
    return [m for m in task["models"] if model_id(m) in keep]


def sequence_input_sizes(cfg: dict, seq: str, task: str) -> dict:
    """{configured model id: input size} for one sequence and task: global input_size.<task> updated with
    sequence_overrides.<seq>.input_size.<task> (a null there removes the global size = the model's default)."""
    sizes = dict(((cfg.get("input_size") or {}).get(task)) or {})
    sizes.update((((cfg.get("sequence_overrides") or {}).get(seq) or {}).get("input_size") or {}).get(task) or {})
    return {k: v for k, v in sizes.items() if v is not None}


def resolve_sequence_models(cfg: dict, seq: str, task: dict) -> tuple[list[dict], dict]:
    """The task's models for one sequence with the per-sequence input sizes applied.

    Returns (models, skipped): models = sequence_models() with `input_size` set on the Ultralytics models that have
    an override (model id + "-<N>px"); skipped = {configured model id: reason} for models with an override whose
    wrapper does not implement it (not run), and hf_vitpose models whose detector is skipped. hf_vitpose
    `detector:` ids are mapped to the detector's effective id when the detector has an input size override.
    """
    sizes = sequence_input_sizes(cfg, seq, task["task"])
    det_sizes = sequence_input_sizes(cfg, seq, "detection")
    det_cfg = {model_id(m): m for t in cfg["tasks"] if t["task"] == "detection" for m in t["models"]}
    models, skipped = [], {}
    for m in sequence_models(cfg, seq, task):
        mid = model_id(m)
        if mid in sizes:
            if m["wrapper"] not in INPUT_SIZE_WRAPPERS:
                skipped[mid] = (f"input_size {sizes[mid]} set (input_size / sequence_overrides.{seq}.input_size) "
                                f"but not implemented for the {m['wrapper']} wrapper (implemented: "
                                f"{sorted(INPUT_SIZE_WRAPPERS)}); model not run")
                continue
            m = dict(m, input_size=sizes[mid])
        if m.get("detector") in det_sizes:
            d = det_cfg[m["detector"]]
            if d["wrapper"] not in INPUT_SIZE_WRAPPERS:
                skipped[mid] = f"its detector {m['detector']} is not run on this sequence (input_size not implemented)"
                continue
            m = dict(m, detector=model_id(dict(d, input_size=det_sizes[m["detector"]])))
        models.append(m)
    return models, skipped


def mmrotate_input_size(m: dict) -> str:
    """'tiles' or '<N>px' (resize to fit N x N)."""
    v = m.get("input_size", "tiles")
    return "tiles" if v == "tiles" else f"{int(v)}px"


def model_id(m: dict) -> str:
    """Stable identifier used in output file names."""
    if m.get("name"):
        if m["wrapper"] != "mmrotate" and m.get("input_size") is not None:
            return f"{m['name']}-{int(m['input_size'])}px"
        return m["name"]
    base = f"{m['wrapper']}-{Path(str(m['model'])).name}".replace("/", "_")
    if m.get("tracker"):
        base += f"-{Path(str(m['tracker'])).stem}"
    if m.get("detector"):
        base += f"-on-{m['detector']}"
    if m.get("ocr"):
        base += f"-{m['ocr']}".replace(":", "-").replace("/", "_")
    if m["wrapper"] == "mmrotate":
        base += f"-{mmrotate_input_size(m)}"
    elif m.get("input_size") is not None:
        base += f"-{int(m['input_size'])}px"     # per-sequence input size (sequence_overrides.<seq>.input_size)
    if m.get("score_threshold") is not None:
        base += f"-thr{m['score_threshold']}"
    return base


def load_config(path: str | Path | None) -> dict:
    """Load a YAML config (default configs/config.yaml): merged over the defaults, validated, BD anchors applied.

    Raises:
        ConfigError: file missing, not a mapping, or invalid content.
    """
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_file():
        example = DEFAULT_CONFIG.with_name("example_config.yaml")
        hint = (f" (copy the example: cp {example.relative_to(PROJECT_ROOT)} "
                f"{DEFAULT_CONFIG.relative_to(PROJECT_ROOT)})"
                if path == DEFAULT_CONFIG and example.is_file() else "")
        raise ConfigError(f"Config file not found: {path}{hint}")
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    if not isinstance(user, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    cfg = _merge(DEFAULTS, user, "")
    _validate(cfg)
    cfg["_config_path"] = str(path.resolve())
    _apply_anchors(cfg)
    return cfg


# ----------------------------------------------------------------------------- BD anchors

# Overrides a test folder does not inherit from its anchor: its QP folders are its own.
_NOT_INHERITED = {"qps"}


def load_anchors(cfg: dict) -> tuple[dict[str, str], list[str]]:
    """{sequence_name: anchor folder} from bd.anchors_file, overridden per sequence name by bd.anchors (null removes),
    and warnings (anchor folder missing in data_root, or not named <sequence_name>_...)."""
    bd = cfg["bd"]
    anchors, warnings = {}, []
    f = resolve_path(bd["anchors_file"]) if bd.get("anchors_file") else None
    if f is not None and f.is_file():
        with open(f) as fh:
            doc = yaml.safe_load(fh) or {}
        if not isinstance(doc, dict) or set(doc) - {"anchors"} or not isinstance(doc.get("anchors") or {}, dict):
            raise ConfigError(f"{f}: expected a mapping 'anchors: {{<sequence_name>: <anchor folder>}}'")
        anchors.update(doc.get("anchors") or {})
    elif bd.get("anchors_file"):
        warnings.append(f"bd.anchors_file {bd['anchors_file']} not found (no anchors from file)")
    for rn, folder in (bd.get("anchors") or {}).items():
        if folder is None:
            anchors.pop(rn, None)
        else:
            anchors[rn] = folder
    data_root = resolve_path(cfg["data_root"])
    missing = []
    for rn, folder in sorted(anchors.items()):
        if not isinstance(folder, str) or not folder.startswith(f"{rn}_"):
            raise ConfigError(f"anchor of '{rn}' must be a data folder named {rn}_..., got {folder!r}")
        if not (data_root / folder).is_dir():
            missing.append(folder)
    if len(anchors) >= 2 and 2 * len(missing) > len(anchors):
        # most anchors missing: the anchors were written for another data root (folder names, not paths)
        warnings.append(f"{len(missing)} of {len(anchors)} anchor folders not found in {data_root} (anchors file "
                        f"{bd.get('anchors_file')}, bd.anchors): they seem to belong to another data root; use a "
                        f"separate bd.anchors_file per data root (docs/BD_RATE.md#several-data-roots)")
    else:
        warnings += [f"anchor folder {f} not found in {data_root}" for f in missing]
    return anchors, warnings


def anchor_of(cfg: dict, seq: str) -> str | None:
    """The anchor folder a test folder is compared against (None: no anchor, or seq is the anchor itself)."""
    a = cfg.get("_anchors", {}).get(seq.split("_", 1)[0])
    return a if a and a != seq else None


def _apply_anchors(cfg: dict) -> None:
    """Load the anchors; every other data folder of an anchored sequence name (= a test) inherits the anchor's
    sequence_overrides (except qps), its own overrides winning key by key, so that both encodings run the same
    model ids, classes and metric options."""
    anchors, warnings = load_anchors(cfg)
    cfg["_anchors"], cfg["_anchor_warnings"], cfg["_inherited_overrides"] = anchors, warnings, {}
    data_root = resolve_path(cfg["data_root"])
    if not data_root.is_dir():
        return
    ovs = cfg["sequence_overrides"] = dict(cfg["sequence_overrides"] or {})
    for d in sorted(p for p in data_root.iterdir() if p.is_dir()):
        anchor = anchor_of(cfg, d.name)
        inherited = {k: copy.deepcopy(v) for k, v in (ovs.get(anchor) or {}).items() if k not in _NOT_INHERITED}
        own = ovs.get(d.name) or {}
        keys = sorted(set(inherited) - set(own))
        if anchor and keys:
            ovs[d.name] = inherited | own
            cfg["_inherited_overrides"][d.name] = {"anchor": anchor, "keys": keys}
