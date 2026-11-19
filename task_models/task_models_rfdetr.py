#!/usr/bin/env python3
"""
RF-DETR wrapper using Roboflow's reference implementation (PyPI package `rfdetr`, Apache-2.0),
running in the main venv.

Models (COCO weights downloaded by rfdetr into $RF_HOME, set to <weights_dir>/rfdetr):
  detection:             rfdetr-nano, rfdetr-small, rfdetr-medium, rfdetr-large,
                         rfdetr-base (deprecated in rfdetr, kept for comparison)
  instance segmentation: rfdetr-seg-nano, rfdetr-seg-small, rfdetr-seg-medium, rfdetr-seg-large,
                         rfdetr-seg-xlarge, rfdetr-seg-2xlarge
  keypoints:             rfdetr-keypoint-preview (RFDETRKeypointPreview, weights rf-detr-keypoint-preview-xlarge.pth,
                         17 COCO person keypoints; an open model of the package, not RF-DETR+)
Not allowed: rfdetr-xlarge, rfdetr-2xlarge (detection XL/2XL need an RF-DETR+ licence / API key;
they are not part of the open `rfdetr` package).

Output per instance (same format as the other wrappers):
  detection:             [class_id, score, x1, x2, y1, y2]           (normalised, generic COCO ids)
  instance segmentation: [class_id, score, x1, x2, y1, y2, W, H, rle] (rle = pycocotools compressed
                         RLE string of the mask at the input image size)
  keypoints:             [1, score, x1, x2, y1, y2, [x, y, s] * 17]  (score = detection confidence,
                         keypoint x, y normalised, s = keypoint confidence)
Class ids: rfdetr returns COCO 91-category ids; they are mapped by the returned class name to labels.py.
Library defaults: predict threshold 0.5, masks binarised at logit 0 (probability 0.5).
"""

import numpy as np
import torch
from labels import COCO_label_map
from torchvision.transforms import v2

LICENSED_MODELS = {"rfdetr-xlarge", "rfdetr-2xlarge"}
LICENSE_MESSAGE = ("RF-DETR detection XLarge/2XLarge (rfdetr-xlarge, rfdetr-2xlarge) require a Roboflow "
                   "API key / RF-DETR+ licence and are not allowed in this testbench")


def _model_classes():
    import rfdetr
    return {
        "rfdetr-nano": rfdetr.RFDETRNano,
        "rfdetr-small": rfdetr.RFDETRSmall,
        "rfdetr-medium": rfdetr.RFDETRMedium,
        "rfdetr-large": rfdetr.RFDETRLarge,
        "rfdetr-base": rfdetr.variants.RFDETRBase,
        "rfdetr-seg-nano": rfdetr.RFDETRSegNano,
        "rfdetr-seg-small": rfdetr.RFDETRSegSmall,
        "rfdetr-seg-medium": rfdetr.RFDETRSegMedium,
        "rfdetr-seg-large": rfdetr.RFDETRSegLarge,
        "rfdetr-seg-xlarge": rfdetr.RFDETRSegXLarge,
        "rfdetr-seg-2xlarge": rfdetr.RFDETRSeg2XLarge,
        "rfdetr-keypoint-preview": rfdetr.RFDETRKeypointPreview,
    }


class RFDETRTasks(torch.nn.Module):
    """Input: (1, 3, H, W) float tensor, RGB, values in [0, 1]."""

    def __init__(self, model_cfg_name="rfdetr-medium", device="cuda:0", input_size=None):
        super().__init__()
        if model_cfg_name in LICENSED_MODELS:
            raise ValueError(LICENSE_MESSAGE)
        classes = _model_classes()
        if model_cfg_name not in classes:
            raise ValueError(f"Unknown rfdetr model '{model_cfg_name}'; known: {sorted(classes)}")
        from rfdetr.variants import RFDETRSeg
        cls = classes[model_cfg_name]
        # input_size: None = the model's training resolution; N = square N x N input (rfdetr `resolution`; the
        # library rescales the positional encoding). N must be a multiple of patch_size * num_windows (32 for
        # rfdetr-medium, 24 for rfdetr-seg-medium); rfdetr itself does not check this.
        kw = {}
        if input_size is not None:
            f = cls._model_config_class.model_fields
            block = f["patch_size"].default * f["num_windows"].default
            if int(input_size) <= 0 or int(input_size) % block:
                raise ValueError(f"{model_cfg_name}: input_size must be a positive multiple of {block} "
                                 f"(patch_size x num_windows), got {input_size}")
            kw["resolution"] = int(input_size)
        self.rf = cls(device=device, **kw)
        self.input_size = int(self.rf.model_config.resolution)
        self.instance_segmentation = issubclass(cls, RFDETRSeg)
        self.keypoints = model_cfg_name == "rfdetr-keypoint-preview"
        # the checkpoint lists [0, 17]: a background-first slot without keypoints, then person with 17
        if self.keypoints and sorted(k for k in self.rf.model_config.num_keypoints_per_class if k) != [17]:
            raise ValueError(f"{model_cfg_name}: expected 17 COCO keypoints, got "
                             f"{self.rf.model_config.num_keypoints_per_class}")
        self.confidence_threshold = 0.5      # rfdetr predict() default
        self.mask_threshold = 0.5 if self.instance_segmentation else None
        self.model = self.rf.model.model     # torch module (LWDETR), for the weights hash

    def get_transforms(self):
        """Return the frame transform: tensor image as float32, scaled to [0, 1]."""
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])

    def forward(self, x):
        """Run RF-DETR on one RGB frame (1, 3, H, W), values in [0, 1].

        Returns:
            Per instance [class_id, score, x1, x2, y1, y2] normalised (labels.py ids; unmapped classes dropped);
            instance segmentation adds [W, H, rle]; the keypoint model gives [1, score, x1, x2, y1, y2, [x, y, s] * 17].
        """
        from pycocotools import mask as mask_util
        with torch.no_grad():
            det = self.rf.predict(x.squeeze(0), threshold=self.confidence_threshold)
        H, W = int(x.shape[-2]), int(x.shape[-1])
        if self.keypoints:
            return self._parse_keypoints(det, W, H)
        names = det.data.get("class_name") if det.data else None
        out = []
        for i in range(len(det)):
            gid = COCO_label_map.get(str(names[i])) if names is not None else None
            if gid is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in det.xyxy[i])
            r = [gid, float(det.confidence[i]), x1 / W, x2 / W, y1 / H, y2 / H]
            if self.instance_segmentation:
                m = det.mask[i]
                assert m.shape == (H, W), f"mask size {m.shape} != image size {(H, W)}"
                rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
                r += [W, H, rle["counts"].decode("ascii")]
            out.append(r)
        return out

    def _parse_keypoints(self, kp, W, H):
        """supervision KeyPoints: xy (N, 17, 2) pixels, keypoint_confidence (N, 17), detection_confidence (N,),
        box in data["xyxy"], class names in data["class_name"]; persons only."""
        out = []
        if kp is None or len(kp) == 0:
            return out
        names = kp.data.get("class_name")
        boxes = kp.data["xyxy"]
        for i in range(len(kp)):
            if names is not None and str(names[i]) != "person":
                continue
            xy = kp.xy[i] / np.array([W, H], np.float32)
            k = np.concatenate([xy, kp.keypoint_confidence[i][:, None]], 1).ravel()
            x1, y1, x2, y2 = (float(v) for v in boxes[i])
            out.append([1, float(kp.detection_confidence[i]), x1 / W, x2 / W, y1 / H, y2 / H,
                        [round(float(v), 6) for v in k]])
        return out
