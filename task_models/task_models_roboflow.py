#!/usr/bin/env python3
"""
Roboflow `inference` (PyPI) wrapper for RF-DETR object detection and instance segmentation.

Runs in its own venv (.venv-roboflow, see envs/roboflow/pyproject.toml), because `inference` pins
older transformers/numpy/pillow/opencv than the main environment. The main pipeline starts this file
as a worker subprocess (python task_models_roboflow.py --worker) and exchanges length-prefixed
messages over stdin/stdout (stdout of the libraries is redirected to stderr).

Models (aliases of `inference`, public COCO weights, no API key needed):
  detection:             rfdetr-nano, rfdetr-small, rfdetr-medium, rfdetr-base, rfdetr-large
                         (rfdetr-xlarge, rfdetr-2xlarge need an API key / licence: not allowed)
  instance segmentation: rfdetr-seg-nano, rfdetr-seg-small, rfdetr-seg-medium, rfdetr-seg-large,
                         rfdetr-seg-xlarge, rfdetr-seg-2xlarge (rfdetr-seg-preview)

Output per instance (same format as the other wrappers):
  detection:             [class_id, score, x1, x2, y1, y2]           (normalised, generic COCO ids)
  instance segmentation: [class_id, score, x1, x2, y1, y2, W, H, rle] (rle = pycocotools compressed
                         RLE string of the mask at the input image size)
Class ids are mapped by class name to labels.py (the models use their own id orders).
Library defaults: torch backend (GPU), confidence 0.4, masks binarised at logit 0 (probability 0.5),
boxes rounded to whole pixels by the library's RF-DETR post-processing.
"""

import hashlib
import json
import os
import struct
import sys

# Only the torch backend: runs on the GPU with the plain `inference` package (the ONNX backend would
# run on CPU onnxruntime; TensorRT is not installed).
_DISABLED_BACKENDS = "onnx,trt,torch-script,hugging-face,ultralytics,custom"

LICENSED_MODELS = {"rfdetr-xlarge", "rfdetr-2xlarge", "coco/47", "coco/48"}

LABEL_SYNONYMS = {"motorbike": "motorcycle", "aeroplane": "airplane", "sofa": "couch",
                  "pottedplant": "potted plant", "diningtable": "dining table", "tvmonitor": "tv"}


class RoboflowTasks:
    """RF-DETR via Roboflow inference. Input: RGB uint8 numpy array (H, W, 3)."""

    def __init__(self, model_cfg_name="rfdetr-base", device="cuda:0", confidence_threshold=None):
        os.environ.setdefault("DISABLED_INFERENCE_MODELS_BACKENDS", _DISABLED_BACKENDS)
        import torch
        from inference import get_model
        from inference_models.configuration import INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE
        from labels import COCO_label_map

        if model_cfg_name.strip().lower() in LICENSED_MODELS:
            raise ValueError("RF-DETR detection XLarge/2XLarge (rfdetr-xlarge, rfdetr-2xlarge) require a Roboflow "
                             "API key / RF-DETR+ licence and are not allowed in this testbench")
        self.model = get_model(model_cfg_name, device=torch.device(device))
        self.instance_segmentation = getattr(self.model, "task_type", "") == "instance-segmentation" \
            or type(self.model).__name__.endswith("InstanceSegmentationAdapter")
        # None = library default (passed as "not set" so the library applies its own default)
        self.confidence_threshold = confidence_threshold
        self.default_confidence = float(INFERENCE_MODELS_RFDETR_DEFAULT_CONFIDENCE)
        self.mask_threshold = 0.5 if self.instance_segmentation else None
        self.class_names = list(self.model.class_names)
        self.id_map = {}
        self.unmapped = []
        for i, name in enumerate(self.class_names):
            gid = COCO_label_map.get(LABEL_SYNONYMS.get(name, name))
            if gid is None:
                self.unmapped.append(name)
            else:
                self.id_map[i] = gid
        missing = sorted(set(COCO_label_map) - {LABEL_SYNONYMS.get(n, n) for n in self.class_names})
        if missing:
            raise RuntimeError(f"{model_cfg_name}: classes of labels.py missing in the model: {missing[:5]}")

    @property
    def effective_threshold(self):
        """Confidence threshold in use: the configured one, or the library default if none is set."""
        return self.default_confidence if self.confidence_threshold is None else self.confidence_threshold

    def torch_module(self):
        """Return the underlying torch model (for the weights hash)."""
        return self.model._model._model

    def forward(self, img_rgb):
        """Run the model on one RGB uint8 frame, a numpy array (H, W, 3).

        Returns:
            Per instance [class_id, score, x1, x2, y1, y2] normalised (labels.py ids; unmapped classes dropped);
            instance segmentation adds [W, H, rle].
        """
        import numpy as np
        bgr = np.ascontiguousarray(img_rgb[:, :, ::-1])
        kw = {}
        if self.confidence_threshold is not None:
            kw["confidence"] = float(self.confidence_threshold)
        if self.instance_segmentation:
            kw["response_mask_format"] = "rle"
        resp = self.model.infer(bgr, **kw)[0]
        W, H = int(resp.image.width), int(resp.image.height)
        out = []
        for p in resp.predictions:
            gid = self.id_map.get(int(p.class_id))
            if gid is None:
                continue
            x1, x2 = (p.x - p.width / 2) / W, (p.x + p.width / 2) / W
            y1, y2 = (p.y - p.height / 2) / H, (p.y + p.height / 2) / H
            r = [gid, float(p.confidence), float(x1), float(x2), float(y1), float(y2)]
            if self.instance_segmentation:
                rle = p.rle
                size = [int(v) for v in rle["size"]]
                assert size == [H, W], f"mask size {size} != image size {[H, W]}"
                c = rle["counts"]
                r += [W, H, c.decode("ascii") if isinstance(c, bytes) else c]
            out.append(r)
        return out


def state_dict_hash(module) -> str:
    """Same algorithm as vcmtest.infer.state_dict_hash (kept in sync; this venv has no vcmtest deps)."""
    import torch
    h = hashlib.sha256()
    sd = module.state_dict()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous().reshape(-1)
        h.update(f"{k}|{t.dtype}|{tuple(sd[k].shape)}|".encode())
        if t.numel():
            h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- worker protocol
# message = 4-byte big-endian length + JSON header [+ raw payload of header["nbytes"] bytes]

def _recv(f):
    n = f.read(4)
    if len(n) < 4:
        return None, None
    hdr = json.loads(f.read(struct.unpack(">I", n)[0]))
    payload = f.read(hdr["nbytes"]) if hdr.get("nbytes") else None
    return hdr, payload


def _send(f, obj):
    b = json.dumps(obj).encode()
    f.write(struct.pack(">I", len(b)) + b)
    f.flush()


def _worker():
    proto_in = sys.stdin.buffer
    proto_out = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)                       # anything printed by libraries goes to stderr
    sys.stdout = sys.stderr
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    net = None
    while True:
        hdr, payload = _recv(proto_in)
        if hdr is None or hdr["cmd"] == "quit":
            return
        try:
            if hdr["cmd"] == "load":
                if hdr.get("cache_dir"):
                    os.environ.setdefault("MODEL_CACHE_DIR", hdr["cache_dir"])
                    os.environ.setdefault("INFERENCE_HOME", hdr["cache_dir"])
                import random

                import numpy as np
                import torch
                random.seed(hdr.get("seed", 0))
                np.random.seed(hdr.get("seed", 0))
                torch.manual_seed(hdr.get("seed", 0))
                net = RoboflowTasks(hdr["model"], hdr["device"], hdr.get("score_threshold"))
                from importlib.metadata import version
                _send(proto_out, {"ok": True, "instance_segmentation": net.instance_segmentation,
                                  "score_threshold": net.effective_threshold,
                                  "mask_threshold": net.mask_threshold,
                                  "weights_sha256": state_dict_hash(net.torch_module()),
                                  "unmapped_classes": net.unmapped,
                                  "versions": {p: version(p) for p in ("inference", "inference-models", "torch",
                                                                       "transformers", "numpy")}})
            elif hdr["cmd"] == "predict":
                import numpy as np
                img = np.frombuffer(payload, dtype=np.uint8).reshape(hdr["shape"])
                _send(proto_out, {"ok": True, "preds": net.forward(img)})
            else:
                _send(proto_out, {"ok": False, "error": f"unknown command {hdr['cmd']}"})
        except Exception as e:  # reported to the main process, which decides what to do
            import torch
            _send(proto_out, {"ok": False, "error": f"{type(e).__name__}: {e}",
                              "oom": isinstance(e, torch.cuda.OutOfMemoryError)})


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker()
    else:
        print(__doc__)
