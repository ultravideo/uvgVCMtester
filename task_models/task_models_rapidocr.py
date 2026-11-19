#!/usr/bin/env python3
"""Scene-text detection + recognition (end to end) with RapidOCR (https://github.com/RapidAI/RapidOCR, Apache-2.0):
PaddleOCR PP-OCR models (Apache-2.0) converted to ONNX, run on ONNX Runtime GPU.

Model name = '<PP-OCR version>-<model type>', e.g. 'PP-OCRv5-server', 'PP-OCRv5-mobile', 'PP-OCRv6-small' (detector
and recognizer of that version and type, multilingual 'ch' models that also read Latin letters and digits). The text
line orientation classifier is the library default (ch_ppocr_mobile_v2.0_cls). Library defaults otherwise: detector
input = frame with its shorter side >= 736 (a 1920x1080 frame is used as is), DB box threshold 0.5, unclip 1.6; text
score threshold (Global.text_score) 0.5, set by score_threshold.

Runtime: onnxruntime-gpu < 1.27 (CUDA 12 build) with onnxruntime.preload_dlls() (torch's CUDA libraries), as
fast_alpr. Weights: downloaded to <weights_dir>/rapidocr/ (Global.model_root_dir); weights_sha256 = SHA-256 over the
three ONNX files actually loaded.

Output per text line: [score, x1, x2, y1, y2, text, ocr_confidence] (same format as the plate readers): score =
ocr_confidence = recognition confidence (RapidOCR exposes no detection score), box = axis-aligned bounds of the
detected quadrilateral, normalised to the frame.
"""

import numpy as np
import torch
from torchvision.transforms import v2


class RapidOCRText(torch.nn.Module):
    """Assumes a single RGB uint8 image (1, C, H, W)."""

    scene_text = True

    def __init__(self, model_cfg_name="PP-OCRv5-server", device="cuda:0", weights_dir=None):
        super().__init__()
        import hashlib
        import pathlib

        import onnxruntime
        from rapidocr import ModelType, OCRVersion, RapidOCR
        cuda = str(device).startswith("cuda")
        if cuda:
            onnxruntime.preload_dlls()
        try:
            ver, typ = model_cfg_name.rsplit("-", 1)
            version, mtype = OCRVersion(ver), ModelType(typ)
        except ValueError as e:
            raise ValueError(f"RapidOCR model '{model_cfg_name}': expected '<version>-<type>' with version in "
                             f"{[v.value for v in OCRVersion]} and type in {[t.value for t in ModelType]}") from e
        self.default_threshold = 0.5                      # Global.text_score in rapidocr/config.yaml
        self.confidence_threshold = self.default_threshold
        params = {"Global.log_level": "error", "Det.ocr_version": version, "Det.model_type": mtype,
                  "Rec.ocr_version": version, "Rec.model_type": mtype,
                  "EngineConfig.onnxruntime.use_cuda": cuda}
        if weights_dir is not None:
            d = pathlib.Path(weights_dir) / "rapidocr"
            d.mkdir(parents=True, exist_ok=True)
            params["Global.model_root_dir"] = str(d)
        self.engine = RapidOCR(params=params)
        import cv2
        warm = np.full((96, 320, 3), 255, np.uint8)       # models load lazily on the first text found: one
        cv2.putText(warm, "123", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 5)   # call with text loads all
        self.engine(warm)
        if any(getattr(self.engine, n) is None for n in ("text_det", "text_cls", "text_rec")):
            raise RuntimeError("RapidOCR models were not loaded by the warm-up call")
        sessions = {n: getattr(self.engine, n).session.session for n in ("text_det", "text_cls", "text_rec")}
        self.providers = {n: s.get_providers()[0] for n, s in sessions.items()}
        files = [pathlib.Path(s._model_path) for s in sessions.values()]
        h = hashlib.sha256()
        for p in files:
            h.update(p.name.encode() + b"|" + p.read_bytes())
        self.weights_sha256 = h.hexdigest()
        self.weights_files = [p.name for p in files]
        self.input_size = {"det_limit_side_len": int(self.engine.cfg.Det.limit_side_len),
                           "det_limit_type": str(self.engine.cfg.Det.limit_type),
                           "max_side_len": int(self.engine.cfg.Global.max_side_len)}

    def set_threshold(self, t: float) -> None:
        """Set the text score threshold (passed to RapidOCR as text_score)."""
        self.confidence_threshold = t

    def get_transforms(self):
        """Return the frame transform: tensor image as uint8, values in [0, 255]."""
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=False)])

    def forward(self, X):
        """Detect and read text lines in one RGB uint8 frame (1, C, H, W).

        Returns:
            Per text line [score, x1, x2, y1, y2, text, ocr_confidence]: score = ocr_confidence = recognition
            confidence, box = axis-aligned bounds of the detected quadrilateral, normalised.
        """
        rgb = X[0].permute(1, 2, 0).cpu().numpy()
        h, w = rgb.shape[:2]
        r = self.engine(np.ascontiguousarray(rgb[:, :, ::-1]), text_score=self.confidence_threshold)
        out = []
        if r.boxes is None:
            return out
        for box, text, score in zip(r.boxes, r.txts, r.scores):
            b = np.asarray(box, float)
            s = float(score)
            out.append([s, float(b[:, 0].min() / w), float(b[:, 0].max() / w), float(b[:, 1].min() / h),
                        float(b[:, 1].max() / h), str(text), s])
        return out
