"""Registry of task-model wrappers: everything the pipeline needs to know about a model family, in one entry.

A wrapper is a class in task_models/<module>.py (or a worker subprocess in its own venv) that turns a library's
model into the common prediction format (see docs/ADDING_MODELS.md). Adding a model family = one wrapper file +
one WRAPPERS entry here (+ an optional dependency extra in pyproject.toml and a row in docs/MODELS.md).

This module is imported by vcmtest.config, so it must stay light: no model libraries, no torch at import time.
Library imports happen inside the wrapper classes, when a model is loaded.

Entry fields (class Wrapper):
  module, cls      task_models/<module>.py and the wrapper class; constructed as
                   cls(model_cfg_name=<config model>, device=<device>, **kwargs(mcfg, weights_dir)).
                   Worker wrappers (worker=...): cls is a SubprocessWorker subclass in vcmtest/infer.py.
  requirements     import names checked before a run, and the install command printed when one is missing
  install
  input_on_device  True: the frame tensor is moved to the device before forward(); False: forward() gets it on CPU
  threshold        how the score threshold is read / set: an attribute name of the wrapper object, a
                   (getter, setter) pair of functions of the wrapper object, or None (no threshold, e.g. depth)
  torch_module     the torch.nn.Module whose parameters are hashed into weights_sha256: an attribute path
                   ("model", "model.model"), a function of the wrapper object, or None (the wrapper object sets
                   weights_sha256 itself, e.g. a hash of its ONNX files)
  kwargs           function (mcfg, weights_dir) -> extra constructor arguments
  check            function (net) -> None, run after loading (e.g. label-order checks); raise ValueError on a problem
  describe         function (net, mcfg) -> dict merged into the model description (results JSON, cache key): settings
                   that change the predictions but are not in the config line
  input_size       True: sequence_overrides.<seq>.input_size is implemented (kwargs passes mcfg["input_size"])
  keypoints        function (model name) -> bool: the model is a keypoint model (allowed in the keypoints task)
  worker           config key of the venv python for worker subprocess wrappers (e.g. "roboflow_python")
  weights_subdir   worker wrappers: cache folder under weights_dir passed to the worker
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# Pascal-VOC style names used by some model configs (e.g. RT-DETR) -> names in labels.py
LABEL_SYNONYMS = {"motorbike": "motorcycle", "aeroplane": "airplane", "sofa": "couch",
                  "pottedplant": "potted plant", "diningtable": "dining table", "tvmonitor": "tv"}


@dataclass(frozen=True)
class Wrapper:
    """Registry entry of one model family (fields described in the module docstring)."""
    module: str
    cls: str
    requirements: tuple = ()
    install: str = ""
    input_on_device: bool = False
    threshold: str | tuple | None = "score_threshold"
    torch_module: str | Callable | None = "model"
    kwargs: Callable | None = None
    check: Callable | None = None
    describe: Callable | None = None
    input_size: bool = False
    keypoints: Callable | None = None
    worker: str | None = None
    weights_subdir: str | None = None


# ----------------------------------------------------------------------------- shared helpers

def check_contiguous_labels(id2label: dict, what: str) -> None:
    """The wrapper maps label_id -> label_id + 1; valid only for the contiguous 80-class order of labels.py."""
    from labels import COCO_label_map
    bad = [i for i, name in id2label.items()
           if COCO_label_map.get(LABEL_SYNONYMS.get(name, name)) != int(i) + 1]
    if bad:
        raise ValueError(f"{what} label order differs from labels.py for ids {bad[:5]}; "
                         f"wrapper mapping label+1 would be wrong")


def check_dota_labels(id2label: dict) -> None:
    """OBB models output 0-based DOTA v1.0 ids, stored as is: their names must match labels_dota.py."""
    from labels_dota import DOTA_label_map
    bad = [i for i, name in id2label.items() if DOTA_label_map.get(name) != int(i)]
    if bad or len(id2label) != len(DOTA_label_map):
        raise ValueError(f"OBB model classes differ from DOTA v1.0 (labels_dota.py) for ids {bad[:5]}")


def _override_source() -> dict:
    return {"source": "sequence_overrides.<seq>.input_size"}


# ----------------------------------------------------------------------------- per-wrapper functions

def _torchvision_threshold():
    def get(n):
        m = n.model
        return float(m.roi_heads.score_thresh if hasattr(m, "roi_heads") else m.score_thresh)

    def set_(n, t):
        m = n.model
        if hasattr(m, "roi_heads"):
            m.roi_heads.score_thresh = t
        else:
            m.score_thresh = t
    return get, set_


def _detectron2_threshold():
    def get(n):
        return float(n.predictor.model.roi_heads.box_predictor.test_score_thresh)

    def set_(n, t):
        n.predictor.model.roi_heads.box_predictor.test_score_thresh = t
        n.confidence_threshold = t
    return get, set_


def _onnx_threshold():
    """fast_alpr / rapidocr / plate_two_stage: set through set_threshold(), read from confidence_threshold."""
    return (lambda n: float(n.confidence_threshold)), (lambda n, t: n.set_threshold(t))


def _easyocr_threshold():
    def get(n):
        return None if n.confidence_threshold is None else float(n.confidence_threshold)

    def set_(n, t):
        n.confidence_threshold = t
    return get, set_


def _ultralytics_kwargs(mcfg, weights_dir):
    from .config import is_ultralytics_tracker
    from .infer import resolve_tracker
    name = str(mcfg["model"])
    p = Path(name)
    if not p.is_absolute() and len(p.parts) == 1:
        p = Path(weights_dir) / "ultralytics" / name     # download target inside the project
        p.parent.mkdir(parents=True, exist_ok=True)
    tracker = mcfg.get("tracker")
    return {"model_cfg_name": str(p),
            "tracker": resolve_tracker(tracker) if tracker and is_ultralytics_tracker(tracker) else None,
            "imgsz": mcfg.get("input_size")}


def _ultralytics_check(net):
    if getattr(net, "obb", False):
        check_dota_labels(net.model.names)
    else:
        check_contiguous_labels(net.model.names, "Ultralytics")


def _ultralytics_describe(net, mcfg):
    d = {}
    if getattr(net, "obb", False):
        d["input_size"] = {"imgsz": net.imgsz}
    if mcfg.get("input_size") is not None:
        d["input_size"] = {**_override_source(), "imgsz": net.imgsz}
    return d


def _rfdetr_describe(net, mcfg):
    if mcfg.get("input_size") is None:
        return {}
    return {"input_size": {**_override_source(), "resolution": net.input_size, "shape": "square N x N"}}


def _rtdetr_describe(net, mcfg):
    if mcfg.get("input_size") is None:
        return {}
    return {"input_size": {**_override_source(), "size": mcfg["input_size"],
                           "shape": "stretched to N x N (as the 640 x 640 default)"}}


def _vitpose_describe(net, mcfg):
    return {"detector": mcfg["detector"],              # person boxes = this detection model's predictions
            "input_size": net.input_size,
            "score_threshold_meaning": "minimum detector person-box score"}


def _easyocr_describe(net, mcfg):
    return {"score_threshold_source": "library default (none)"} if mcfg.get("score_threshold") is None else {}


def _plate_two_stage_describe(net, mcfg):
    if mcfg.get("input_size") is None:
        return {}
    return {"input_size": {**_override_source(), **net.input_size}}


def _roboflow_describe(net, mcfg):
    d = {"library_versions": net.info["versions"]}
    if net.info.get("unmapped_classes"):
        d["unmapped_classes"] = net.info["unmapped_classes"]
    return d


def _mmrotate_describe(net, mcfg):
    d = {"input_size": net.info["input_size"], "library_versions": net.info["versions"],
         "config": net.info["config"], "ai4rs_commit": net.info["ai4rs_commit"]}
    if mcfg.get("score_threshold") is None:
        d["score_threshold_source"] = "library default (none; max 300 detections)"
    return d


# ----------------------------------------------------------------------------- the registry

WRAPPERS: dict[str, Wrapper] = {
    "torchvision": Wrapper(
        "task_models_torchvision", "TorchvisionTasks", input_on_device=True, threshold=_torchvision_threshold(),
        keypoints=lambda name: name.startswith("keypointrcnn")),
    "ultralytics": Wrapper(
        "task_models_ultralytics", "UltralyticsTasks", ("ultralytics",), "uv sync --extra ultralytics",
        threshold="confidence_threshold", torch_module="model.model", kwargs=_ultralytics_kwargs,
        check=_ultralytics_check, describe=_ultralytics_describe, input_size=True,
        keypoints=lambda name: Path(name).stem.endswith("-pose")),
    "rtdetr": Wrapper(
        "task_models_rtdetr", "RTDetr", ("transformers", "accelerate"), "uv sync --extra rtdetr",
        kwargs=lambda mcfg, wd: {"input_size": mcfg.get("input_size")},
        check=lambda net: check_contiguous_labels(net.model.config.id2label, "RT-DETR"),
        describe=_rtdetr_describe, input_size=True),
    "rfdetr": Wrapper(
        "task_models_rfdetr", "RFDETRTasks", ("rfdetr",), "uv sync --extra rfdetr", input_on_device=True,
        threshold="confidence_threshold", kwargs=lambda mcfg, wd: {"input_size": mcfg.get("input_size")},
        describe=_rfdetr_describe, input_size=True, keypoints=lambda name: name.startswith("rfdetr-keypoint")),
    "detectron2": Wrapper(
        "task_models_detectron2", "Detectron2Tasks", ("detectron2",),
        "build and install it: envs/detectron2/setup.sh", input_on_device=True, threshold=_detectron2_threshold(),
        torch_module="predictor.model"),
    "hf_panoptic": Wrapper(
        "task_models_hf_panoptic", "HFPanoptic", ("transformers",), "uv sync --extra panoptic"),
    "hf_vitpose": Wrapper(
        "task_models_hf_vitpose", "HFVitPose", ("transformers",), "uv sync --extra vitpose",
        describe=_vitpose_describe, keypoints=lambda name: True),
    "hf_depth": Wrapper(
        "task_models_hf_depth", "HFDepth", ("transformers",), "uv sync --extra depth", threshold=None),
    "fast_alpr": Wrapper(
        "task_models_fast_alpr", "FastALPR", ("fast_alpr", "onnxruntime"), "uv sync --extra lpr",
        threshold=_onnx_threshold(), torch_module=None,
        kwargs=lambda mcfg, wd: {"ocr_model": str(mcfg["ocr"]), "weights_dir": wd}),
    "plate_two_stage": Wrapper(
        "task_models_plate_two_stage", "PlateTwoStage", ("ultralytics", "huggingface_hub", "onnxruntime"),
        "uv sync --extra lpr2", threshold=_onnx_threshold(), torch_module=None,
        kwargs=lambda mcfg, wd: {"ocr_model": str(mcfg["ocr"]), "weights_dir": wd,
                                 "imgsz": mcfg.get("input_size")},
        describe=_plate_two_stage_describe, input_size=True),
    "rapidocr": Wrapper(
        "task_models_rapidocr", "RapidOCRText", ("rapidocr", "onnxruntime"), "uv sync --extra ocr",
        threshold=_onnx_threshold(), torch_module=None, kwargs=lambda mcfg, wd: {"weights_dir": wd}),
    "easyocr": Wrapper(
        "task_models_easyocr", "EasyOCRText", ("easyocr",), "uv sync --extra ocr", threshold=_easyocr_threshold(),
        torch_module=lambda n: n.torch_module(), kwargs=lambda mcfg, wd: {"weights_dir": wd},
        describe=_easyocr_describe),
    # worker subprocess wrappers (own venv): cls = SubprocessWorker subclass in vcmtest/infer.py
    "roboflow": Wrapper(
        "vcmtest.infer", "RoboflowWorker", (), "UV_PROJECT_ENVIRONMENT=.venv-roboflow uv sync --project envs/roboflow",
        describe=_roboflow_describe, worker="roboflow_python", weights_subdir="roboflow"),
    "mmrotate": Wrapper(
        "vcmtest.infer", "MMRotateWorker", (), "envs/mmrotate/setup.sh",
        kwargs=lambda mcfg, wd: {"input_size": mcfg.get("input_size", "tiles")}, describe=_mmrotate_describe,
        worker="mmrotate_python", weights_subdir="mmrotate"),
}


def get(name: str) -> Wrapper:
    """Registry entry of a wrapper by name."""
    return WRAPPERS[name]


def resolve_attr(obj, path: str):
    """Attribute of obj at a dotted path (e.g. "model.model")."""
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj
