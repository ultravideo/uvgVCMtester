"""Stage 2: run a task network on every frame (batch size 1), cached and resumable.

Predictions keep the wrapper format, per frame, one list per instance:
    detection:              [class_id, score, x1, x2, y1, y2]
    instance segmentation:  [class_id, score, x1, x2, y1, y2, W, H, rle]
    keypoints:              [1, score, x1, x2, y1, y2, [x, y, s] * 17]   (keypoint x, y normalised, s = score)
coordinates normalised to [0, 1], generic COCO ids 1..80; rle = pycocotools compressed RLE string
of the binary mask at W x H (the decoded frame size).

Cache: data/<seq>/results/predictions/<model_id>/QPxx.json
  - written only when all frames are done; QPxx.partial.jsonl holds finished frames so an
    interrupted run resumes where it stopped.
  - "meta" records the model, effective score threshold, weights hash and the decode settings;
    if any of these differ from the current run, the predictions are recomputed.
Model weights are downloaded under <weights_dir> (torch hub, Hugging Face, Ultralytics).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

from .config import is_ultralytics_tracker, model_id, resolve_path

log = logging.getLogger("vcmtest")

PRED_VERSION = 1
TASK_MODELS_DIR = resolve_path("task_models")


class InferenceError(RuntimeError):
    """Raised when model loading or inference fails."""
    pass


def setup_weight_dirs(weights_dir: Path) -> None:
    """Keep all downloaded weights inside the project (must run before torch hub / HF imports)."""
    weights_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_HOME", str(weights_dir / "torch"))
    os.environ.setdefault("HF_HOME", str(weights_dir / "huggingface"))
    os.environ.setdefault("RF_HOME", str(weights_dir / "rfdetr"))      # rfdetr reference package
    os.environ.setdefault("FVCORE_CACHE", str(weights_dir / "detectron2"))   # detectron2 model zoo weights
    if str(TASK_MODELS_DIR) not in sys.path:
        sys.path.insert(0, str(TASK_MODELS_DIR))


# ----------------------------------------------------------------------------- trackers

def resolve_tracker(name: str) -> str:
    """A tracker YAML given as a path (relative to the project root) or an Ultralytics built-in name."""
    p = Path(name)
    if not p.is_absolute():
        p = resolve_path(name)
    if p.is_file():
        return str(p)
    builtin = name if name.endswith(".yaml") else f"{name}.yaml"
    return builtin


def tracker_config(name: str) -> dict:
    """Tracker settings recorded with the predictions and part of their cache key: the contents of an
    Ultralytics tracker YAML, or library/version/default parameters of a Roboflow `trackers` algorithm."""
    if not is_ultralytics_tracker(name) and not Path(name).is_file():
        if str(TASK_MODELS_DIR) not in sys.path:
            sys.path.insert(0, str(TASK_MODELS_DIR))
        from task_models_trackers import tracker_parameters
        return tracker_parameters(name)
    import yaml
    path = resolve_tracker(name)
    if not Path(path).is_file():
        import importlib.util
        spec = importlib.util.find_spec("ultralytics")
        if spec is None or not spec.origin:
            raise InferenceError(f"tracker '{name}' needs ultralytics (uv sync --extra ultralytics)")
        path = Path(spec.origin).parent / "cfg" / "trackers" / Path(path).name
        if not path.is_file():
            raise InferenceError(f"tracker config '{name}' not found (built-in: botsort.yaml, bytetrack.yaml, ...)")
    with open(path) as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------- worker subprocesses

class SubprocessWorker:
    """Runs a task_models/<script> --worker in a separate venv (its own python) as a subprocess.

    Messages: 4-byte big-endian length + JSON header (+ raw RGB bytes for 'predict'). The worker's
    stderr (library logs, warnings) goes to the debug log; its last lines are added to errors.
    Used by the 'roboflow' (envs/roboflow) and 'mmrotate' (envs/mmrotate) wrappers.
    """

    LOAD_TIMEOUT_S = 1800      # first load downloads the weights
    PREDICT_TIMEOUT_S = 300
    KIND = "worker"
    SCRIPT = ""
    SETUP_HINT = ""

    def __init__(self, python: Path | None, model: str, device: str, score_threshold, cache_dir: Path,
                 seed: int, extra: dict | None = None):
        import collections
        import subprocess
        import threading
        if python is None or not Path(python).is_file():
            raise InferenceError(f"{self.KIND} venv python not found: {python} (create it with: {self.SETUP_HINT})")
        python = Path(python).absolute()     # not resolve(): the venv python is a symlink
        cache_dir = Path(cache_dir).absolute()   # the worker runs in task_models/
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.stderr_tail: collections.deque = collections.deque(maxlen=30)
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "VIRTUAL_ENV")}
        self.proc = subprocess.Popen([str(python), "-u", str(TASK_MODELS_DIR / self.SCRIPT), "--worker"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     cwd=str(TASK_MODELS_DIR), env=env)
        self._t = threading.Thread(target=self._drain_stderr, daemon=True)
        self._t.start()
        try:
            self.info = self._call({"cmd": "load", "model": model, "device": device,
                                    "score_threshold": None if score_threshold is None else float(score_threshold),
                                    "cache_dir": str(cache_dir), "seed": seed, **(extra or {})},
                                   timeout=self.LOAD_TIMEOUT_S)
        except BaseException:
            self.close()          # never leave a worker (and its GPU memory) behind
            raise
        if self.info.get("unmapped_classes"):
            log.info(f"{model}: model classes not in labels.py (predictions dropped): {self.info['unmapped_classes']}")

    def _drain_stderr(self) -> None:
        for line in iter(self.proc.stderr.readline, b""):
            t = line.decode(errors="replace").rstrip()
            if t:
                self.stderr_tail.append(t)
                log.debug(f"[{self.KIND} worker] {t}")

    def _read_exact(self, n: int, deadline: float) -> bytes:
        import select
        buf = b""
        fd = self.proc.stdout.fileno()
        while len(buf) < n:
            left = deadline - time.time()
            if left <= 0:
                raise InferenceError(f"{self.KIND} worker ({self.model}) timed out")
            r, _, _ = select.select([fd], [], [], min(left, 5.0))
            if r:
                chunk = os.read(fd, n - len(buf))
                if not chunk:
                    raise InferenceError(f"{self.KIND} worker ({self.model}) exited (code {self.proc.poll()}): "
                                         + " | ".join(list(self.stderr_tail)[-5:]))
                buf += chunk
            elif self.proc.poll() is not None:
                raise InferenceError(f"{self.KIND} worker ({self.model}) exited (code {self.proc.returncode}): "
                                     + " | ".join(list(self.stderr_tail)[-5:]))
        return buf

    def _call(self, hdr: dict, payload: bytes | None = None, timeout: float = 300) -> dict:
        import struct
        if payload is not None:
            hdr = {**hdr, "nbytes": len(payload)}
        b = json.dumps(hdr).encode()
        try:
            self.proc.stdin.write(struct.pack(">I", len(b)) + b + (payload or b""))
            self.proc.stdin.flush()
        except BrokenPipeError as e:
            raise InferenceError(f"{self.KIND} worker ({self.model}) is not running: "
                                 + " | ".join(list(self.stderr_tail)[-5:])) from e
        deadline = time.time() + timeout
        n = struct.unpack(">I", self._read_exact(4, deadline))[0]
        resp = json.loads(self._read_exact(n, deadline))
        if not resp.get("ok"):
            if resp.get("oom"):
                import torch
                raise torch.cuda.OutOfMemoryError(resp.get("error", "CUDA out of memory (worker)"))
            raise InferenceError(f"{self.KIND} worker ({self.model}): {resp.get('error')}")
        return resp

    def predict(self, rgb) -> list:
        """Send an RGB frame (uint8 array) to the worker and return its predictions."""
        return self._call({"cmd": "predict", "shape": list(rgb.shape)}, rgb.tobytes(),
                          timeout=self.PREDICT_TIMEOUT_S)["preds"]

    def close(self) -> None:
        """Ask a still running worker to quit; kill it if that fails."""
        import struct
        if self.proc.poll() is None:
            try:
                b = json.dumps({"cmd": "quit"}).encode()
                self.proc.stdin.write(struct.pack(">I", len(b)) + b)
                self.proc.stdin.flush()
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=30)


class RoboflowWorker(SubprocessWorker):
    """RF-DETR via Roboflow inference (task_models/task_models_roboflow.py, venv envs/roboflow)."""
    KIND = "Roboflow"
    SCRIPT = "task_models_roboflow.py"
    SETUP_HINT = "UV_PROJECT_ENVIRONMENT=.venv-roboflow uv sync --project envs/roboflow"


AI4RS_DIR = TASK_MODELS_DIR.parent / "envs" / "mmrotate" / "ai4rs"


class MMRotateWorker(SubprocessWorker):
    """MMRotate / ai4rs oriented detectors (task_models/task_models_mmrotate.py, venv envs/mmrotate)."""
    KIND = "MMRotate"
    SCRIPT = "task_models_mmrotate.py"
    SETUP_HINT = "envs/mmrotate/setup.sh"

    def __init__(self, python, model, device, score_threshold, cache_dir, seed, input_size="tiles"):
        if not (AI4RS_DIR / "mmrotate").is_dir():
            raise InferenceError(f"ai4rs checkout missing: {AI4RS_DIR} (create it with: {self.SETUP_HINT})")
        super().__init__(python, model, device, score_threshold, cache_dir, seed,
                         {"ai4rs_dir": str(AI4RS_DIR.absolute()), "input_size": input_size})


# ----------------------------------------------------------------------------- model adapter

class LoadedModel:
    """A task-model wrapper plus what is needed to feed it and to describe it in results.

    Everything wrapper-specific (construction, thresholds, weights hash, checks, extra description) comes from the
    registry in vcmtest/wrappers.py."""

    def __init__(self, mcfg: dict, device: str, weights_dir: Path, seed: int = 0,
                 worker_python: dict | None = None):
        import importlib

        import torch

        from . import wrappers

        self.cfg = mcfg
        self.id = model_id(mcfg)
        self.wrapper_name = w = mcfg["wrapper"]
        self.device = device
        name = str(mcfg["model"])
        if w not in wrappers.WRAPPERS:
            raise InferenceError(f"Unknown wrapper {w}")
        spec = self.spec = wrappers.WRAPPERS[w]
        tracker = mcfg.get("tracker")
        self.tracker_adapter = None
        if tracker and not is_ultralytics_tracker(tracker):
            from task_models_trackers import TrackersAdapter  # Roboflow trackers on any detector's output
            self.tracker_adapter = TrackersAdapter(tracker)
        kwargs = spec.kwargs(mcfg, weights_dir) if spec.kwargs else {}

        if spec.worker:                                          # own venv: worker subprocess
            cls = getattr(importlib.import_module(spec.module), spec.cls)
            python = (worker_python or {}).get(spec.worker)
            self.net = cls(python, name, device, mcfg.get("score_threshold"), weights_dir / spec.weights_subdir,
                           seed, **kwargs)
            self.input_on_device = False
            info = self.net.info
            self.score_threshold = info["score_threshold"]
            self.instance_segmentation = bool(info.get("instance_segmentation", False))
            self.panoptic = self.keypoints = self.depth = self.lpr = self.scene_text = False
            self.obb = bool(info.get("obb", False))
            self.mask_threshold = info.get("mask_threshold")
            self.weights_sha256 = info["weights_sha256"]
            self.transforms = None
            self.torch_module = None
            return

        cls = getattr(importlib.import_module(spec.module), spec.cls)
        kwargs = {"model_cfg_name": name, "device": device, **kwargs}
        try:
            self.net = cls(**kwargs)
        except ValueError as e:                                  # invalid settings reported by the wrapper
            raise InferenceError(str(e)) from None
        self.input_on_device = spec.input_on_device
        if spec.check:
            try:
                spec.check(self.net)
            except ValueError as e:
                raise InferenceError(str(e)) from None
        if mcfg.get("score_threshold") is not None:
            self._set_threshold(float(mcfg["score_threshold"]))
        self.score_threshold = self._get_threshold()
        self.instance_segmentation = bool(getattr(self.net, "instance_segmentation", False))
        self.panoptic = bool(getattr(self.net, "panoptic", False))
        self.keypoints = bool(getattr(self.net, "keypoints", False))
        self.depth = bool(getattr(self.net, "depth", False))
        self.obb = bool(getattr(self.net, "obb", False))
        self.lpr = bool(getattr(self.net, "lpr", False))
        self.scene_text = bool(getattr(self.net, "scene_text", False))
        self.mask_threshold = getattr(self.net, "mask_threshold", None) if self.instance_segmentation else None
        self.transforms = self.net.get_transforms()
        self.torch_module = self._torch_module()
        self.weights_sha256 = (self.net.weights_sha256 if self.torch_module is None   # e.g. ONNX models: file hash
                               else state_dict_hash(self.torch_module))
        if hasattr(self.torch_module, "eval"):
            self.torch_module.eval()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    # -- thresholds: library defaults unless the config sets one (how: wrappers.Wrapper.threshold)
    def _set_threshold(self, t: float) -> None:
        th = self.spec.threshold
        if th is None:
            return
        if isinstance(th, tuple):
            th[1](self.net, t)
        else:
            setattr(self.net, th, t)

    def _get_threshold(self) -> float | None:
        th = self.spec.threshold
        if th is None:
            return None
        if isinstance(th, tuple):
            return th[0](self.net)
        v = getattr(self.net, th)
        return None if v is None else float(v)

    def _torch_module(self):
        from .wrappers import resolve_attr
        tm = self.spec.torch_module
        if tm is None:
            return None
        return tm(self.net) if callable(tm) else resolve_attr(self.net, tm)

    def describe(self) -> dict:
        """Return the model settings that identify its predictions (part of the prediction cache key)."""
        d = {"model_id": self.id, "wrapper": self.wrapper_name, "model": str(self.cfg["model"]),
                "score_threshold": self.score_threshold,
                "score_threshold_source": "config" if self.cfg.get("score_threshold") is not None
                else "library default",
                "weights_sha256": self.weights_sha256, "device": self.device,
                "output": "tracking" if self.tracking else "panoptic" if self.panoptic else
                          "depth" if self.depth else "oriented_detection" if self.obb else
                          "plate_recognition" if self.lpr else "scene_text" if self.scene_text else
                          "keypoints" if self.keypoints else
                          "instance_segmentation" if self.instance_segmentation else "detection"}
        if self.panoptic:
            d["input_size"] = self.net.input_size          # image processor resize (part of the cache key)
            d["family"] = self.net.family
        if self.lpr:
            d["ocr"] = self.net.ocr_name
            d["input_size"] = self.net.input_size
            d["providers"] = self.net.providers
            d["weights_files"] = self.net.weights_files
        if self.scene_text:
            d["input_size"] = self.net.input_size
            if getattr(self.net, "providers", None):
                d["providers"] = self.net.providers
                d["weights_files"] = self.net.weights_files
        if self.obb:
            d["class_ids"] = "DOTA v1.0, 0-based (task_models/labels_dota.py)"
        if self.depth:
            d["input_size"] = self.net.input_size
            d["family"] = self.net.family
            d["depth_output"] = self.net.output_kind       # metric | relative_disparity
            d["max_depth_m"] = self.net.max_depth
            d["score_threshold_source"] = "not applicable"
        if self.tracking:
            d["tracker"] = str(self.cfg["tracker"])
            d["tracker_config"] = tracker_config(self.cfg["tracker"])
        if self.instance_segmentation:
            d["mask_threshold"] = self.mask_threshold
        if self.spec.describe:
            d.update(self.spec.describe(self.net, self.cfg))     # wrapper-specific settings (wrappers.py)
        return d

    @property
    def tracking(self) -> bool:
        """True if the model outputs tracks (separate tracker adapter or the wrapper's own tracker)."""
        return self.tracker_adapter is not None or bool(getattr(self.net, "tracker", None))

    def reset_tracking(self, frame_rate: float | None = None) -> None:
        """Fresh tracker state for a new sequence (Roboflow trackers also get the sequence frame rate)."""
        if self.tracker_adapter is not None:
            if not frame_rate:
                raise InferenceError(f"{self.id}: the sequence frame rate is needed for tracker "
                                     f"'{self.tracker_adapter.name}'")
            self.tracker_adapter.reset(frame_rate)
        elif self.tracking:
            self.net.reset_tracking()

    def close(self) -> None:
        """Release the wrapper's resources (e.g. a worker subprocess), if it has a close method."""
        if hasattr(self.net, "close"):
            self.net.close()

    def predict(self, img, boxes: list | None = None) -> list[list]:
        """Run the model on one frame and return its predictions in the task's prediction format.

        Args:
            img: RGB frame (PIL image).
            boxes: detector person boxes for top-down models (`detector` in the model config).
        """
        preds = self._predict_detections(img, boxes)
        if self.tracker_adapter is not None:
            import numpy as np
            preds = self.tracker_adapter.update(preds, img.size[0], img.size[1], frame=np.asarray(img))
        return preds

    def _predict_detections(self, img, boxes: list | None = None) -> list[list]:
        import torch
        if self.spec.worker:
            import numpy as np
            preds = self.net.predict(np.asarray(img.convert("RGB"), dtype=np.uint8))
            if self.obb:
                return [[int(p[0]), float(p[1])] + [float(v) for v in p[2:10]] for p in preds]
            return self._normalise(preds)
        x = self.transforms(img).unsqueeze(0)
        if self.input_on_device:
            x = x.to(self.device)
        with torch.no_grad():
            preds = self.net(x, boxes) if self.cfg.get("detector") else self.net(x)   # top-down: detector boxes
        if self.depth:
            return preds                                   # float32 H x W array, metres
        if self.keypoints:
            return [[int(p[0]), float(p[1])] + [float(v) for v in p[2:6]] + [[float(v) for v in p[6]]]
                    for p in preds]
        if self.panoptic:
            return [[int(p[0]), float(p[1]), int(p[2]), int(p[3]), str(p[4])] for p in preds]
        if self.obb:
            return [[int(p[0]), float(p[1])] + [float(v) for v in p[2:10]] for p in preds]
        if self.lpr or self.scene_text:
            return [[float(p[0])] + [float(v) for v in p[1:5]] + [str(p[5]), float(p[6])] for p in preds]
        return self._normalise(preds)

    @staticmethod
    def _normalise(preds) -> list[list]:
        out = []
        for p in preds:
            r = [int(p[0]), float(p[1])] + [float(v) for v in p[2:6]]
            if len(p) == 7:                      # tracking: track id
                r += [int(p[6])]
            elif len(p) > 6:                     # instance segmentation: W, H, RLE string
                r += [int(p[6]), int(p[7]), str(p[8])]
            out.append(r)
        return out


def state_dict_hash(module) -> str:
    """SHA-256 over all parameters and buffers (names, dtypes, shapes, values)."""
    import torch
    h = hashlib.sha256()
    sd = module.state_dict()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous().reshape(-1)
        h.update(f"{k}|{t.dtype}|{tuple(sd[k].shape)}|".encode())
        if t.numel():
            h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class ModelPool:
    """Keeps at most one model in GPU memory; loads lazily.

    worker_python: {config key: venv python path} for the worker-subprocess wrappers (e.g. "roboflow_python")."""

    def __init__(self, device: str, weights_dir: Path, seed: int, worker_python: dict | None = None):
        self.device, self.weights_dir, self.seed = device, weights_dir, seed
        self.worker_python = dict(worker_python or {})
        self.current: LoadedModel | None = None

    def get(self, mcfg: dict) -> LoadedModel:
        """Return the loaded model for a model config; release the current one first if it is another model."""
        mid = model_id(mcfg)
        if self.current is not None and self.current.id == mid:
            return self.current
        self.release()
        from .env_info import set_seeds
        set_seeds(self.seed)
        log.info(f"Loading model {mid} on {self.device}")
        t0 = time.time()
        self.current = LoadedModel(mcfg, self.device, self.weights_dir, self.seed, self.worker_python)
        log.info(f"Model {mid} loaded in {time.time() - t0:.1f} s; score threshold "
                 f"{self.current.score_threshold} ({self.current.describe()['score_threshold_source']}); "
                 f"weights sha256 {self.current.weights_sha256[:16]}")
        return self.current

    def release(self) -> None:
        """Close the current model and free its GPU memory."""
        if self.current is not None:
            try:
                self.current.close()
            finally:
                self.current = None
            import gc
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass


# ----------------------------------------------------------------------------- frames

_warned_16bit = False


def load_frame(path: Path):
    """RGB uint8 image (PIL). 16-bit PNGs are rounded to 8 bits for the networks."""
    global _warned_16bit
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("RGB",):
        return im.copy()
    if im.mode in ("I;16", "I;16B", "I", "RGB;16", "RGBA;16") or getattr(im, "bits", 8) > 8:
        import cv2
        import numpy as np
        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        if a.dtype == np.uint16:
            if not _warned_16bit:
                log.warning("16-bit PNG frames are rounded to 8 bits for the task networks")
                _warned_16bit = True
            a = np.clip(np.round(a / 257.0), 0, 255).astype(np.uint8)
        return Image.fromarray(a)
    return im.convert("RGB")


# ----------------------------------------------------------------------------- cache + loop

def predictions_path(results_dir: Path, mid: str, qp: int) -> Path:
    """Return results/predictions/<model_id>/QP<qp>.json."""
    return results_dir / "predictions" / mid / f"QP{qp}.json"


def _cache_key(model_desc: dict, decode_settings: dict | None, n_frames: int) -> dict:
    key = {"version": PRED_VERSION, "model_id": model_desc["model_id"],
           "score_threshold": model_desc["score_threshold"],
           "weights_sha256": model_desc["weights_sha256"],
           "decode_settings": decode_settings, "frames": n_frames}
    if model_desc.get("mask_threshold") is not None:
        key["mask_threshold"] = model_desc["mask_threshold"]
    if model_desc.get("tracker_config") is not None:
        key["tracker_config"] = model_desc["tracker_config"]
    if model_desc.get("input_size") is not None:
        key["input_size"] = model_desc["input_size"]
    if model_desc.get("depth_output") is not None:
        key["depth_output"] = model_desc["depth_output"]
    return key


def cached_predictions(path: Path, mcfg: dict, decode_settings: dict | None, n_frames: int,
                       fps: float | None = None, require_depth_maps: bool = True) -> dict | None:
    """Cached predictions if they match the config without loading the model, else None.

    Without loading the model, the check uses model id, configured threshold (or library default
    recorded in the file), decode settings and frame count. The weights hash is checked again
    whenever the model is loaded anyway.
    """
    if not path.is_file():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    k = data.get("meta", {}).get("key", {})
    if (k.get("version") != PRED_VERSION or k.get("model_id") != model_id(mcfg)
            or k.get("decode_settings") != decode_settings or k.get("frames") != n_frames):
        return None
    if mcfg.get("score_threshold") is not None and k.get("score_threshold") != float(mcfg["score_threshold"]):
        return None
    if mcfg.get("tracker") and k.get("tracker_config") != tracker_config(mcfg["tracker"]):
        return None                               # tracker settings changed
    if (fps is not None and mcfg.get("tracker") and not is_ultralytics_tracker(mcfg["tracker"])
            and k.get("frame_rate") != float(fps)):
        return None                               # sequence frame rate changed (scales the tracker buffers)
    src = data["meta"].get("model", {}).get("score_threshold_source")
    if mcfg.get("score_threshold") is None and not (src or "").startswith(("library default", "not applicable")):
        return None
    if require_depth_maps and data["meta"].get("model", {}).get("output") == "depth" and not all(
            (path.parent / v[2]).is_file() for v in data["frames"].values()):
        return None                               # a depth map PNG is missing
    if mcfg.get("detector"):
        dp = detector_predictions_path(path, mcfg)
        if not dp.is_file() or k.get("detector_predictions_sha256") != _sha256(dp):
            return None                           # the detector's boxes changed (or are missing)
    return data


def detector_predictions_path(pred_path: Path, mcfg: dict) -> Path:
    """predictions/<model_id>/QPxx.json -> predictions/<detector model id>/QPxx.json (same sequence and QP)."""
    return pred_path.parent.parent / mcfg["detector"] / pred_path.name


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_inference(pool: ModelPool, mcfg: dict, frames: dict[int, Path], out_path: Path,
                  decode_settings: dict | None, timeout_s: float, label: str,
                  env: dict | None = None, task: str | None = None, fps: float | None = None) -> dict:
    """Run (or resume) inference for one (sequence, QP, model). Returns the predictions dict."""
    import torch

    n = len(frames)
    cached = cached_predictions(out_path, mcfg, decode_settings, n, fps)
    if cached is not None:
        log.info(f"{label}: predictions cached ({out_path.name}, {n} frames)")
        return cached

    try:
        model = pool.get(mcfg)
    except InferenceError:
        raise
    except Exception as e:  # missing package, download failure, bad model name, ...
        pool.release()
        raise InferenceError(f"{label}: failed to load model: {type(e).__name__}: {e}") from e
    if (task == "tracking") != model.tracking:
        raise InferenceError(f"{label}: model {model.id} " + ("needs a 'tracker' for the tracking task" if
                             task == "tracking" else "is a tracker; use it in the tracking task only"))
    if (task in ("panoptic_segmentation", "semantic_segmentation")) != model.panoptic:
        raise InferenceError(f"{label}: model {model.id} " + ("is not a panoptic model" if model.panoptic is False
                             and task in ("panoptic_segmentation", "semantic_segmentation")
                             else "outputs panoptic segments; use it in the panoptic / semantic tasks only"))
    if (task == "keypoints") != model.keypoints:
        raise InferenceError(f"{label}: model {model.id} " + ("does not output keypoints" if task == "keypoints"
                             else "outputs keypoints; use it in the keypoints task only"))
    is_depth = bool(getattr(model, "depth", False))
    if (task == "depth") != is_depth:
        raise InferenceError(f"{label}: model {model.id} " + ("is not a depth model" if task == "depth"
                             else "outputs depth maps; use it in the depth task only"))
    if (task == "plate_recognition") != bool(getattr(model, "lpr", False)):
        raise InferenceError(f"{label}: model {model.id} " + ("is not a licence plate reader"
                             if task == "plate_recognition" else
                             "reads licence plates; use it in the plate_recognition task only"))
    if (task == "bib_recognition") != bool(getattr(model, "scene_text", False)):
        raise InferenceError(f"{label}: model {model.id} " + ("is not a scene-text reader"
                             if task == "bib_recognition" else
                             "reads scene text; use it in the bib_recognition task only"))
    if (task == "oriented_detection") != bool(getattr(model, "obb", False)):
        raise InferenceError(f"{label}: model {model.id} " + ("is not an oriented box (OBB) model"
                             if task == "oriented_detection" else
                             "outputs oriented boxes; use it in the oriented_detection task only"))
    if task == "instance_segmentation" and not model.instance_segmentation:
        raise InferenceError(f"{label}: model {model.id} does not output instance masks "
                             f"(use a segmentation model, e.g. maskrcnn_*, yolo26*-seg.pt)")
    desc = model.describe()
    key = _cache_key(desc, decode_settings, n)
    if model.tracker_adapter is not None:
        key["frame_rate"] = float(fps) if fps else None
    det_frames = None
    if mcfg.get("detector"):
        dp = detector_predictions_path(out_path, mcfg)
        if not dp.is_file():
            raise InferenceError(f"{label}: predictions of detector {mcfg['detector']} not found ({dp}); the detector "
                                 f"must run in the detection task for this sequence and QP")
        key["detector_predictions_sha256"] = _sha256(dp)
        with open(dp) as f:
            det_frames = json.load(f)["frames"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".partial.jsonl")

    done: dict[str, list] = {}
    if model.tracking:
        # tracker state cannot be restored: a tracking run always starts at the first frame with a fresh tracker
        model.reset_tracking(fps)
        if partial.is_file():
            log.info(f"{label}: tracking cannot resume mid-sequence; restarting from the first frame")
            partial.unlink()
    if partial.is_file():
        with open(partial) as f:
            first = f.readline()
            try:
                hdr = json.loads(first) if first else {}
            except json.JSONDecodeError:
                hdr = {}
            if hdr.get("key") == key:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        break                      # torn last line from an interrupted run
                    done[str(rec["frame"])] = rec["preds"]
        if done:
            log.info(f"{label}: resuming, {len(done)}/{n} frames already done")
    # (re)write the partial file cleanly: header + finished frames (drops a torn last line)
    with open(partial, "w") as f:
        f.write(json.dumps({"key": key}) + "\n")
        for fr, preds in done.items():
            f.write(json.dumps({"frame": int(fr), "preds": preds}) + "\n")

    t0 = last = time.time()
    n_new = 0
    with open(partial, "a") as f:
        for fn, path in frames.items():
            if str(fn) in done:
                continue
            if time.time() - t0 > timeout_s:
                raise InferenceError(f"{label}: inference timed out after {timeout_s} s "
                                     f"({len(done)}/{n} frames done; rerun resumes)")
            img = load_frame(path)
            boxes = det_frames.get(str(fn), []) if det_frames is not None else None
            preds = _predict_with_oom_retry(model, img, label, fn, boxes)
            if is_depth:
                preds = save_depth_png(preds, depth_png_path(out_path, fn), out_path.parent,
                                       desc.get("depth_output", "metric"))
            done[str(fn)] = preds
            f.write(json.dumps({"frame": fn, "preds": preds}) + "\n")
            n_new += 1
            if n_new % 25 == 0:
                f.flush()
            now = time.time()
            if now - last >= 15:
                log.info(f"{label}: {len(done)}/{n} frames ({n_new / (now - t0):.1f} fps)")
                last = now
    elapsed = time.time() - t0

    data = {
        "meta": {"key": key, "model": desc, "frames": n, "inference_seconds": round(elapsed, 1),
                 "fps": round(n_new / elapsed, 2) if elapsed > 0 and n_new else None,
                 "max_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                 if torch.cuda.is_available() else None,
                 "environment": env,
                 "format": DEPTH_FORMAT if is_depth else OBB_FORMAT if desc.get("output") == "oriented_detection" else
                           LPR_FORMAT if desc.get("output") == "plate_recognition" else
                           TEXT_FORMAT if desc.get("output") == "scene_text" else
                           "per frame: [class_id, score, x1, x2, y1, y2"
                           + (", W, H, rle" if desc.get("mask_threshold") is not None else "")
                           + (", [x, y, s] * 17 keypoints" if desc.get("output") == "keypoints" else "")
                           + "], normalised coordinates"
                           + ("; rle = pycocotools compressed RLE of the binary mask at W x H"
                              if desc.get("mask_threshold") is not None else "")},
        "frames": {str(k): done[str(k)] for k in frames},
    }
    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(data, f)
    tmp.replace(out_path)
    partial.unlink(missing_ok=True)
    log.info(f"{label}: {n} frames done in {elapsed:.1f} s -> {out_path}")
    return data


LPR_FORMAT = ("per frame: [score, x1, x2, y1, y2, text, ocr_confidence]; score = plate detector confidence, box "
              "normalised to the frame, text = read characters, ocr_confidence = mean per-character confidence")

TEXT_FORMAT = ("per frame: [score, x1, x2, y1, y2, text, ocr_confidence]; one entry per text box read by a scene-text "
               "OCR; score = ocr_confidence = recognition confidence, axis-aligned box normalised to the frame")

OBB_FORMAT = ("per frame: [class_id, score, x1, y1, x2, y2, x3, y3, x4, y4]; class_id = DOTA v1.0 0-based "
              "(labels_dota.py), corners normalised to the frame")

DEPTH_FORMAT = ("per frame: [W, H, png, sha256] (metric) or [W, H, png, sha256, lo, hi] (relative); png "
                "(relative to this file) = 16-bit grayscale map at the frame size W x H. Metric: depth in "
                "millimetres (rounded, clipped to 1..65535). Relative: the model output linearly normalised per "
                "frame, value = lo + (png - 1) / 65534 * (hi - lo)")


def remove_depth_maps(pred_path: Path) -> int:
    """Delete predictions/<model_id>/QPxx_depth/ (the depth map PNGs; QPxx.json with their sha256 is kept).
    Returns the number of PNGs removed."""
    d = pred_path.parent / f"{pred_path.stem}_depth"
    if not d.is_dir():
        return 0
    files = list(d.glob("depth_*.png"))
    for f in files:
        f.unlink()
    try:
        d.rmdir()
    except OSError:
        pass                                      # other files in it: left alone
    return len(files)


def depth_maps_present(pred_path: Path) -> bool:
    """True if every depth map PNG listed in a depth predictions file exists."""
    try:
        with open(pred_path) as f:
            recs = json.load(f)["frames"].values()
        return all((pred_path.parent / r[2]).is_file() for r in recs)
    except (OSError, json.JSONDecodeError, KeyError, IndexError, TypeError):
        return False


def depth_png_path(pred_path: Path, fn: int) -> Path:
    """predictions/<model_id>/QPxx.json -> predictions/<model_id>/QPxx_depth/depth_<fn>.png"""
    return pred_path.parent / f"{pred_path.stem}_depth" / f"depth_{fn:04d}.png"


def save_depth_png(depth, path: Path, base: Path, kind: str = "metric") -> list:
    """Depth output (H x W) -> 16-bit PNG; returns the per-frame record (see DEPTH_FORMAT).
    metric: metres -> millimetres. relative_*: linear per-frame normalisation to 1..65535, range kept as lo, hi."""
    import numpy as np
    from PIL import Image
    a = np.asarray(depth, dtype=np.float64)
    if a.ndim != 2:
        raise InferenceError(f"depth output must be H x W, got shape {a.shape}")
    extra = []
    if kind == "metric":
        q = np.clip(np.round(np.nan_to_num(a, nan=0.0) * 1000.0), 1, 65535).astype(np.uint16)
    else:
        a = np.nan_to_num(a, nan=float(np.nanmedian(a)) if np.isfinite(a).any() else 0.0)
        lo, hi = float(a.min()), float(a.max())
        q = (np.round((a - lo) / (hi - lo) * 65534) + 1 if hi > lo else np.ones_like(a)).astype(np.uint16)
        extra = [lo, hi]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.png")
    Image.fromarray(q).save(tmp)        # uint16 -> mode I;16
    tmp.replace(path)
    return [int(q.shape[1]), int(q.shape[0]), str(path.relative_to(base)), _sha256(path)] + extra


def _predict_with_oom_retry(model: LoadedModel, img, label: str, fn: int, boxes: list | None = None,
                            retries: int = 2) -> list:
    import torch
    for attempt in range(retries + 1):
        try:
            return model.predict(img, boxes)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            if attempt == retries:
                raise InferenceError(f"{label}: CUDA out of memory on frame {fn} after {retries} retries "
                                     f"(finished frames are kept; rerun resumes)") from e
            log.warning(f"{label}: CUDA OOM on frame {fn}, retrying ({attempt + 1}/{retries})")
            time.sleep(2)
    raise AssertionError("unreachable")
