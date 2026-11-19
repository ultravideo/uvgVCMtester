#!/usr/bin/env python3
"""Scene-text detection + recognition (end to end) with EasyOCR (https://github.com/JaidedAI/EasyOCR, Apache-2.0):
CRAFT text detector + CRNN recognizer (PyTorch).

Model name = the recognition language(s), '+'-separated EasyOCR codes, e.g. 'en' (english_g2 recognizer; reads Latin
letters and digits). readtext() defaults (detector canvas 2560, text_threshold 0.7, low_text 0.4, link 0.4, greedy
decoder, no allowlist). The frame goes to Reader.detect (RGB) and Reader.recognize (greyscale, cv2 RGB2GRAY) directly:
readtext() on a numpy array treats a 3-channel array as BGR for the greyscale conversion, while its detector (and its
file-path input) uses RGB; calling the two stages ourselves gives the file-path behaviour.

Weights: <weights_dir>/easyocr/ (model_storage_directory); weights_sha256 = SHA-256 of the detector and recognizer
state dicts (vcmtest state_dict_hash over both).

Output per text box: [score, x1, x2, y1, y2, text, ocr_confidence] (same format as the plate readers): score =
ocr_confidence = recognition confidence (EasyOCR exposes no detection score); box = axis-aligned bounds, normalised.
No score threshold by default (EasyOCR returns every recognised box); score_threshold filters on the confidence.
"""

import numpy as np
import torch
from torchvision.transforms import v2


class EasyOCRText(torch.nn.Module):
    """Assumes a single RGB uint8 image (1, C, H, W)."""

    scene_text = True

    def __init__(self, model_cfg_name="en", device="cuda:0", weights_dir=None):
        super().__init__()
        import pathlib

        import easyocr
        langs = [s for s in str(model_cfg_name).split("+") if s]
        kw = {}
        if weights_dir is not None:
            d = pathlib.Path(weights_dir) / "easyocr"
            (d / "user_network").mkdir(parents=True, exist_ok=True)
            kw = {"model_storage_directory": str(d), "user_network_directory": str(d / "user_network")}
        cuda = str(device).startswith("cuda")
        self.reader = easyocr.Reader(langs, gpu=device if cuda else False, verbose=False, **kw)
        self.confidence_threshold = None
        self.input_size = {"canvas_size": 2560, "mag_ratio": 1.0}

    def torch_module(self):
        """Return the detector and recognizer networks as one ModuleDict (for the weights hash)."""
        return torch.nn.ModuleDict({"detector": _unwrap(self.reader.detector),
                                    "recognizer": _unwrap(self.reader.recognizer)})

    def get_transforms(self):
        """Return the frame transform: tensor image as uint8, values in [0, 255]."""
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=False)])

    def forward(self, X):
        """Detect and read text in one RGB uint8 frame (1, C, H, W).

        Returns:
            Per text box [score, x1, x2, y1, y2, text, ocr_confidence]: score = ocr_confidence = recognition
            confidence, box = axis-aligned bounds, normalised; boxes below confidence_threshold (if set) are dropped.
        """
        import cv2
        rgb = np.ascontiguousarray(X[0].permute(1, 2, 0).cpu().numpy())
        h, w = rgb.shape[:2]
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        hor, free = self.reader.detect(rgb, reformat=False)
        res = self.reader.recognize(grey, hor[0], free[0], reformat=False)
        out = []
        for box, text, conf in res:
            b = np.asarray(box, float)
            s = float(conf)
            if self.confidence_threshold is not None and s < self.confidence_threshold:
                continue
            out.append([s, float(b[:, 0].min() / w), float(b[:, 0].max() / w), float(b[:, 1].min() / h),
                        float(b[:, 1].max() / h), str(text), s])
        return out


def _unwrap(m):
    return m.module if isinstance(m, torch.nn.DataParallel) else m
