#!/usr/bin/env python3
"""Licence plate detection + recognition (end to end) with fast-alpr (https://github.com/ankandrew/fast-alpr, MIT):
an open-image-models plate detector (YOLOv9 'yolo-v9-*-license-plate-end2end', ONNX, fixed input size given in the
name, e.g. 640 px; the whole frame is letterboxed to it) followed by a fast-plate-ocr reader on each detected plate
crop (e.g. 'cct-s-v2-global-model', global model with a region head).

Runtime: ONNX Runtime GPU. onnxruntime-gpu < 1.27 is the CUDA 12 build; onnxruntime.preload_dlls() loads the CUDA 12
/ cuDNN 9 libraries that torch installs (nvidia-* wheels), so no system CUDA is needed. Providers: CUDA, then CPU
(TensorRT is not used).

Weights: downloaded to <weights_dir>/fast_alpr/{open-image-models,fast-plate-ocr}/<model>/ (the libraries' cache
directories, normally ~/.cache/..., are redirected); weights_sha256 = SHA-256 over the downloaded files of both models.

Output per plate: [score, x1, x2, y1, y2, text, ocr_confidence] = detector confidence, the box normalised to the frame,
the read characters ("" if the OCR returned nothing) and the mean per-character OCR confidence.
"""

import numpy as np
import torch
from torchvision.transforms import v2

PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class FastALPR(torch.nn.Module):
    """Assumes a single RGB uint8 image (1, C, H, W)."""

    lpr = True

    def __init__(self, model_cfg_name="yolo-v9-t-640-license-plate-end2end", ocr_model="cct-s-v2-global-model",
                 device="cuda:0", weights_dir=None):
        super().__init__()
        import hashlib
        import pathlib

        import onnxruntime
        if str(device).startswith("cuda"):
            onnxruntime.preload_dlls()
        import fast_plate_ocr.inference.hub as ocr_hub
        import open_image_models.detection.core.hub as det_hub
        from fast_alpr import ALPR
        if weights_dir is not None:
            base = pathlib.Path(weights_dir) / "fast_alpr"
            det_hub.MODEL_CACHE_DIR = base / "open-image-models"
            ocr_hub.MODEL_CACHE_DIR = base / "fast-plate-ocr"
        providers = PROVIDERS if str(device).startswith("cuda") else ["CPUExecutionProvider"]
        self.confidence_threshold = 0.4                  # fast-alpr default detector_conf_thresh
        self.detector_name, self.ocr_name = model_cfg_name, ocr_model
        self.alpr = ALPR(detector_model=model_cfg_name, ocr_model=ocr_model, detector_providers=providers,
                         ocr_providers=providers, ocr_device="cuda" if str(device).startswith("cuda") else "cpu")
        self.providers = {"detector": self.alpr.detector.detector.model.get_providers()[0]}
        files = sorted((pathlib.Path(det_hub.MODEL_CACHE_DIR) / model_cfg_name).glob("*")) + \
            sorted((pathlib.Path(ocr_hub.MODEL_CACHE_DIR) / ocr_model).glob("*"))
        h = hashlib.sha256()
        for p in files:
            h.update(p.name.encode() + b"|" + p.read_bytes())
        self.weights_sha256 = h.hexdigest()
        self.weights_files = [p.name for p in files]
        self.input_size = {"detector": model_cfg_name.split("-")[3]}     # e.g. 640 from yolo-v9-t-640-...

    def set_threshold(self, t: float) -> None:
        """Set the detector confidence threshold, here and in the fast-alpr plate detector."""
        self.confidence_threshold = t
        self.alpr.detector.detector.conf_thresh = t

    def get_transforms(self):
        """Return the frame transform: tensor image as uint8, values in [0, 255]."""
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=False)])

    def forward(self, X):
        """Detect and read licence plates in one RGB uint8 frame (1, C, H, W).

        Returns:
            Per plate [score, x1, x2, y1, y2, text, ocr_confidence]: detector confidence, box normalised, read text
            ("" if the OCR returned nothing), mean per-character OCR confidence. Plates below confidence_threshold
            are dropped.
        """
        rgb = X[0].permute(1, 2, 0).cpu().numpy()
        h, w = rgb.shape[:2]
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        out = []
        for r in self.alpr.predict(bgr):
            d, b = r.detection, r.detection.bounding_box
            if d.confidence < self.confidence_threshold:
                continue
            text, conf = "", 0.0
            if r.ocr is not None:
                text = r.ocr.text or ""
                c = r.ocr.confidence
                conf = float(np.mean(c)) if isinstance(c, (list, tuple, np.ndarray)) and len(c) else float(c or 0.0)
            out.append([float(d.confidence), b.x1 / w, b.x2 / w, b.y1 / h, b.y2 / h, text, conf])
        return out
