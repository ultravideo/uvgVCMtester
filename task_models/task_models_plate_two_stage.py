#!/usr/bin/env python3
"""Licence plate recognition in two stages: an Ultralytics YOLO plate detector on the whole frame, then a text
recogniser on each detected plate crop. Detector and reader are chosen independently.

Detector (`model:`): an Ultralytics plate-detection checkpoint (.pt, single class), either
'<hf user>/<hf repo>/<file>.pt' (downloaded from the Hugging Face Hub at the pinned revision of REVISIONS, or the
latest revision with a warning if the repo is not listed there) or a local .pt path. E.g.
'morsetechlab/yolov11-license-plate-detection/license-plate-finetune-v1s.pt' (YOLOv11 n/s/m/l/x fine-tuned on a
Roboflow plate dataset, AGPL-3.0). Ultralytics predict() defaults: conf 0.25 (score_threshold), IoU 0.7, imgsz =
the checkpoint's training size (640) unless `input_size` is given; the frame is letterboxed to it.

Reader (`ocr:` = '<reader>:<model>'), on each crop:
- fast_plate_ocr:<model>, e.g. fast_plate_ocr:cct-s-v2-global-model: fast-plate-ocr (MIT) as in fast-alpr
  (crop converted to the model's colour mode; text = plate characters, pad characters removed);
- rapidocr:<PP-OCR version>-<type>, e.g. rapidocr:PP-OCRv5-server: RapidOCR (Apache-2.0) recogniser only
  (use_det/use_cls off: the crop is the text line);
- easyocr:<langs>, e.g. easyocr:en: EasyOCR (Apache-2.0) as readtext() on the crop: CRAFT text detection in the crop
  (RGB), recognition on the greyscale crop (cv2 RGB2GRAY), pieces joined left to right, confidence = their mean.
  (Recognising the whole crop as one box without detection read worse on CarPark VTM QP27: 0 of 4 large plates.)
No character allowlist or plate-format rules for any reader; the metric normalises the text (upper case, spaces,
hyphens, dots removed).

Crop = the detector box rounded outwards to whole pixels and clipped to the frame (fast-alpr crops the same way).

Weights: <weights_dir>/plate_two_stage/<repo>/<revision>/<file> (detector), the readers' own folders under
<weights_dir> (fast_alpr/fast-plate-ocr, rapidocr, easyocr, shared with those wrappers); weights_sha256 = SHA-256 over
the detector file and the reader's weight files (EasyOCR: its recogniser state dict).

Output per plate: [score, x1, x2, y1, y2, text, ocr_confidence] = detector confidence, the box normalised to the frame,
the read text ("" if nothing was read) and the reader's confidence (fast-plate-ocr: mean per-character probability).
"""

import hashlib
import pathlib
import warnings

import numpy as np
import torch
from torchvision.transforms import v2

# Hugging Face revisions the detector checkpoints are pinned to (repo id -> commit).
REVISIONS = {"morsetechlab/yolov11-license-plate-detection": "251a30d7daedca065f56e04b0af04052c907c68f"}
READERS = ("fast_plate_ocr", "rapidocr", "easyocr")
ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def parse_ocr(spec: str) -> tuple[str, str]:
    """Split '<reader>:<model>' and check the reader name.

    Raises:
        ValueError: no ':' or an unknown reader.
    """
    reader, sep, model = str(spec).partition(":")
    if not sep or not model or reader not in READERS:
        raise ValueError(f"ocr '{spec}': expected '<reader>:<model>' with reader in {list(READERS)}, e.g. "
                         f"fast_plate_ocr:cct-s-v2-global-model, rapidocr:PP-OCRv5-server, easyocr:en")
    return reader, model


def _file_hash(h, p: pathlib.Path) -> None:
    h.update(p.name.encode() + b"|" + p.read_bytes())


class PlateTwoStage(torch.nn.Module):
    """Assumes a single RGB uint8 image (1, C, H, W)."""

    lpr = True

    def __init__(self, model_cfg_name, ocr_model, device="cuda:0", weights_dir=None, imgsz=None):
        super().__init__()
        from ultralytics import YOLO
        self.device = device
        self.cuda = str(device).startswith("cuda")
        self.weights_dir = pathlib.Path(weights_dir) if weights_dir is not None else pathlib.Path("weights")
        self.detector_path = self._detector_file(str(model_cfg_name))
        self.model = YOLO(str(self.detector_path), task="detect")
        names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        if len(names) != 1:
            raise ValueError(f"plate detector {model_cfg_name}: expected one class, got {names}")
        self.confidence_threshold = 0.25                  # Ultralytics predict() default conf
        train_imgsz = (getattr(self.model.model, "args", {}) or {}).get("imgsz", 640)
        self.imgsz = int(imgsz) if imgsz is not None else int(train_imgsz)
        self.ocr_name = str(ocr_model)
        self.reader, self.reader_model = parse_ocr(ocr_model)
        h = hashlib.sha256()
        _file_hash(h, self.detector_path)
        self.weights_files = [self.detector_path.name]
        self.providers = {"detector": "torch " + ("cuda" if self.cuda else "cpu")}
        getattr(self, f"_init_{self.reader}")(h)
        self.weights_sha256 = h.hexdigest()
        self.input_size = {"detector_imgsz": self.imgsz, "detector_letterbox": True,
                           "ocr_input": "detector box crop (rounded outwards, clipped)"}

    # --- detector -----------------------------------------------------------------------------------------------
    def _detector_file(self, name: str) -> pathlib.Path:
        p = pathlib.Path(name)
        if p.suffix == ".pt" and p.exists():
            return p
        parts = name.split("/")
        if len(parts) != 3 or not parts[2].endswith(".pt"):
            raise ValueError(f"plate detector '{name}': give '<hf user>/<hf repo>/<file>.pt' or a local .pt path")
        from huggingface_hub import hf_hub_download
        repo, fname = "/".join(parts[:2]), parts[2]
        rev = REVISIONS.get(repo)
        if rev is None:
            warnings.warn(f"plate detector repo {repo} has no pinned revision (REVISIONS): using the latest",
                          stacklevel=2)
        local = self.weights_dir / "plate_two_stage" / repo / (rev[:12] if rev else "latest")
        return pathlib.Path(hf_hub_download(repo, fname, revision=rev, local_dir=str(local)))

    # --- readers ------------------------------------------------------------------------------------------------
    def _init_fast_plate_ocr(self, h) -> None:
        import onnxruntime
        if self.cuda:
            onnxruntime.preload_dlls()
        import fast_plate_ocr.inference.hub as ocr_hub
        from fast_plate_ocr import LicensePlateRecognizer
        ocr_hub.MODEL_CACHE_DIR = self.weights_dir / "fast_alpr" / "fast-plate-ocr"   # shared with fast_alpr
        providers = ORT_PROVIDERS if self.cuda else ["CPUExecutionProvider"]
        self.ocr = LicensePlateRecognizer(hub_ocr_model=self.reader_model, providers=providers)
        self.color_mode = self.ocr.config.image_color_mode
        files = sorted((pathlib.Path(ocr_hub.MODEL_CACHE_DIR) / self.reader_model).glob("*"))
        for p in files:
            _file_hash(h, p)
        self.weights_files += [p.name for p in files]
        self.providers["ocr"] = self.ocr.model.get_providers()[0]

    def _init_rapidocr(self, h) -> None:
        import onnxruntime
        if self.cuda:
            onnxruntime.preload_dlls()
        from rapidocr import ModelType, OCRVersion, RapidOCR
        try:
            ver, typ = self.reader_model.rsplit("-", 1)
            version, mtype = OCRVersion(ver), ModelType(typ)
        except ValueError as e:
            raise ValueError(f"rapidocr model '{self.reader_model}': expected '<version>-<type>', e.g. "
                             f"PP-OCRv5-server") from e
        d = self.weights_dir / "rapidocr"
        d.mkdir(parents=True, exist_ok=True)
        params = {"Global.log_level": "error", "Global.model_root_dir": str(d), "Global.use_det": False,
                  "Global.use_cls": False, "Rec.ocr_version": version, "Rec.model_type": mtype,
                  "EngineConfig.onnxruntime.use_cuda": self.cuda}
        self.ocr = RapidOCR(params=params)
        import cv2
        warm = np.full((48, 200, 3), 255, np.uint8)       # the recogniser loads lazily: one call loads it
        cv2.putText(warm, "AB123", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        self.ocr(warm, use_det=False, use_cls=False, use_rec=True)
        sess = self.ocr.text_rec.session.session
        p = pathlib.Path(sess._model_path)
        _file_hash(h, p)
        self.weights_files.append(p.name)
        self.providers["ocr"] = sess.get_providers()[0]

    def _init_easyocr(self, h) -> None:
        import easyocr

        from vcmtest.infer import state_dict_hash
        d = self.weights_dir / "easyocr"
        (d / "user_network").mkdir(parents=True, exist_ok=True)
        langs = [s for s in self.reader_model.split("+") if s]
        self.ocr = easyocr.Reader(langs, gpu=self.device if self.cuda else False, verbose=False,
                                  model_storage_directory=str(d), user_network_directory=str(d / "user_network"))
        rec = self.ocr.recognizer
        rec = rec.module if isinstance(rec, torch.nn.DataParallel) else rec
        h.update(b"easyocr-recognizer|" + state_dict_hash(rec).encode())
        self.weights_files.append(f"easyocr recognizer ({'+'.join(langs)})")
        self.providers["ocr"] = "torch " + ("cuda" if self.cuda else "cpu")

    def _read(self, crop_rgb: np.ndarray) -> tuple[str, float]:
        import cv2
        if self.reader == "fast_plate_ocr":
            img = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY) if self.color_mode == "grayscale" else crop_rgb
            p = self.ocr.run_one(img, return_confidence=True)
            probs = p.char_probs
            return str(p.plate or ""), float(np.mean(probs)) if probs is not None and len(probs) else 0.0
        if self.reader == "rapidocr":
            r = self.ocr(np.ascontiguousarray(crop_rgb[:, :, ::-1]), use_det=False, use_cls=False, use_rec=True)
            if not r.txts:
                return "", 0.0
            return str(r.txts[0]), float(r.scores[0])
        grey = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        hor, free = self.ocr.detect(crop_rgb, reformat=False)
        res = self.ocr.recognize(grey, hor[0], free[0], reformat=False) if (hor[0] or free[0]) else []
        if not res:
            return "", 0.0
        res = sorted(res, key=lambda r: min(p[0] for p in r[0]))           # pieces left to right
        return "".join(str(r[1]) for r in res), float(np.mean([float(r[2]) for r in res]))

    # --- interface ----------------------------------------------------------------------------------------------
    def set_threshold(self, t: float) -> None:
        """Set the detector confidence threshold."""
        self.confidence_threshold = t

    def get_transforms(self):
        """Return the frame transform: tensor image as uint8, values in [0, 255]."""
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=False)])

    @torch.no_grad()
    def forward(self, X):
        """Detect plates in one RGB uint8 frame (1, C, H, W) and read each crop.

        Returns:
            Per plate [score, x1, x2, y1, y2, text, ocr_confidence]: detector confidence, box normalised, read text
            ("" if nothing was read), reader confidence. Plates below confidence_threshold are not returned.
        """
        rgb = np.ascontiguousarray(X[0].permute(1, 2, 0).cpu().numpy())
        H, W = rgb.shape[:2]
        r = self.model.predict(np.ascontiguousarray(rgb[:, :, ::-1]), conf=self.confidence_threshold,
                               imgsz=self.imgsz, device=self.device, verbose=False)[0]
        out = []
        for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            x1, y1 = max(int(np.floor(b[0])), 0), max(int(np.floor(b[1])), 0)
            x2, y2 = min(int(np.ceil(b[2])), W), min(int(np.ceil(b[3])), H)
            text, conf = ("", 0.0) if x2 <= x1 or y2 <= y1 else self._read(np.ascontiguousarray(rgb[y1:y2, x1:x2]))
            out.append([float(c), float(b[0]) / W, float(b[2]) / W, float(b[1]) / H, float(b[3]) / H, text, conf])
        return out
