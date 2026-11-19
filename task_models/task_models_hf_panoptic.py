#!/usr/bin/env python3
"""Panoptic segmentation with Hugging Face transformers universal-segmentation models trained on COCO panoptic:
Mask2Former (facebook/mask2former-swin-*-coco-panoptic), OneFormer (shi-labs/oneformer_coco_*) and EoMT
(tue-mps/coco_panoptic_eomt_*).

Inference: the model's own image processor (resize / normalise as for its COCO evaluation; Mask2Former: shorter side
800, longer side <= 1333 instead of the Hugging Face default of a fixed 384x384, see MASK2FORMER_SIZE), then
post_process_panoptic_segmentation at the frame size with the processor defaults (threshold 0.8, mask_threshold 0.5,
overlap_mask_area_threshold 0.8), except label_ids_to_fuse (EoMT: stuff_classes) = all stuff classes: stuff segments
of the same class are merged into one segment, as in the original Mask2Former / OneFormer / EoMT COCO panoptic
evaluation (the transformers default fuses nothing). OneFormer is run with task input "panoptic".
transformers reports swin.layernorm as MISSING for the Swin-based checkpoints: that layer only produces the backbone's
unused last_hidden_state; changing its weights does not change the outputs (checked).

Output per frame: [class_id, score, W, H, rle] with class_id = COCO panoptic category id
(task_models/labels_panoptic.py, mapped by class name from the model's id2label), rle = pycocotools compressed RLE
of the segment at the frame size. Segments do not overlap.
"""

import numpy as np
import torch
from labels_panoptic import COCO_PANOPTIC_isthing, COCO_PANOPTIC_label_map
from torchvision.transforms import v2

DEFAULT_THRESHOLD = 0.8          # transformers post_process_panoptic_segmentation default
# The Hugging Face Mask2Former COCO checkpoints resize to a fixed 384x384 (aspect ratio lost); the original
# Mask2Former COCO evaluation keeps the aspect ratio with the shorter side 800 and the longer side at most 1333
# (as OneFormer's processor does by default). EoMT and OneFormer keep their processor settings.
MASK2FORMER_SIZE = {"shortest_edge": 800, "longest_edge": 1333}


class HFPanoptic(torch.nn.Module):
    """Assumes a single RGB uint8 image (1, C, H, W)."""

    panoptic = True

    def __init__(self, model_cfg_name="facebook/mask2former-swin-tiny-coco-panoptic", device="cuda:0"):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation, AutoProcessor
        self.model = AutoModelForUniversalSegmentation.from_pretrained(model_cfg_name).to(device).eval()
        self.family = self.model.config.model_type            # mask2former | oneformer | eomt
        self.processor = (AutoProcessor if self.family == "oneformer" else AutoImageProcessor).from_pretrained(
            model_cfg_name)
        ip = getattr(self.processor, "image_processor", self.processor)    # OneFormerProcessor wraps one
        if self.family == "mask2former":
            ip.size = dict(MASK2FORMER_SIZE)
        self.input_size = {k: v for k, v in dict(ip.size).items() if v}
        self.model_cfg_name = model_cfg_name
        self.device = device
        self.score_threshold = DEFAULT_THRESHOLD
        id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        unknown = [n for n in id2label.values() if n not in COCO_PANOPTIC_label_map]
        if unknown:
            raise ValueError(f"{model_cfg_name}: classes not in the COCO panoptic categories: {unknown[:5]}")
        self.label_to_coco = {i: COCO_PANOPTIC_label_map[n] for i, n in id2label.items()}
        self.stuff_label_ids = {i for i, c in self.label_to_coco.items() if not COCO_PANOPTIC_isthing[c]}

    def get_transforms(self):
        """Return the frame transform: tensor image as uint8, values in [0, 255]."""
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=False)])

    def forward(self, X):
        """Panoptic segmentation of one RGB uint8 frame (1, C, H, W).

        Returns:
            Per segment [class_id, score, W, H, rle]: COCO panoptic class id, rle = pycocotools compressed RLE at the
            frame size. Segments without pixels are skipped.
        """
        from pycocotools import mask as mu
        img = v2.functional.to_pil_image(X[0])
        w, h = img.size
        if self.family == "oneformer":
            inp = self.processor(images=img, task_inputs=["panoptic"], return_tensors="pt")
        else:
            inp = self.processor(images=img, return_tensors="pt")
        inp = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        with torch.no_grad():
            out = self.model(**inp)
        fuse = ({"stuff_classes": sorted(self.stuff_label_ids)} if self.family == "eomt"     # EoMT's name for it
                else {"label_ids_to_fuse": self.stuff_label_ids})
        res = self.processor.post_process_panoptic_segmentation(
            out, threshold=self.score_threshold, target_sizes=[(h, w)], **fuse)[0]
        seg = res["segmentation"].cpu().numpy()
        preds = []
        for s in res["segments_info"]:
            m = seg == s["id"]
            if not m.any():
                continue
            rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")
            preds.append([self.label_to_coco[int(s["label_id"])], float(s["score"]), w, h, rle])
        return preds
