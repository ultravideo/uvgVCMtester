#!/usr/bin/env python3
"""Monocular depth estimation with Hugging Face transformers depth models (AutoModelForDepthEstimation).

Supported families and what they output (OUTPUT_KIND):
- Depth Anything V2 (depth_anything): the config says depth_estimation_type "metric" (metres; e.g.
  depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf) or "relative" (affine-invariant inverse depth;
  depth-anything/Depth-Anything-V2-{Small,Base,Large}-hf);
- ZoeDepth (zoedepth, e.g. Intel/zoedepth-nyu-kitti) and GLPN (glpn): metric, metres;
- DPT / MiDaS (dpt, e.g. Intel/dpt-large, Intel/dpt-hybrid-midas): relative inverse depth.
Other families are rejected (their output kind would have to be added here after checking it).

Inference: the model's own image processor, then post_process_depth_estimation at the frame size; ZoeDepth's
post-processing gets the source size (padding removal). DPT runs at the Hugging Face default fixed 384x384 (the
frame is squashed, aspect ratio lost): the transformers DPT models (dpt-large, dpt-hybrid-midas) do not
interpolate their position embeddings and fail on non-square input, unlike the original MiDaS / DPT code, which
keeps the aspect ratio (shorter side 384).
Intel/dpt-large reports neck.fusion_stage.layers.0.residual_layer1 as MISSING: the first fusion layer has no
residual input, so that layer is never used (as in the original DPT); the outputs do not depend on it.

Output per frame: float32 array H x W at the frame size: metres ("metric") or relative inverse depth
("relative_disparity"; larger = nearer, unknown scale and shift).
"""

import numpy as np
import torch
from torchvision.transforms import v2

OUTPUT_KIND = {"zoedepth": "metric", "glpn": "metric", "dpt": "relative_disparity"}


class HFDepth(torch.nn.Module):
    """Assumes a single RGB uint8 image (1, C, H, W)."""

    depth = True

    def __init__(self, model_cfg_name="depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", device="cuda:0"):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.model = AutoModelForDepthEstimation.from_pretrained(model_cfg_name).to(device).eval()
        self.processor = AutoImageProcessor.from_pretrained(model_cfg_name)
        self.family = self.model.config.model_type
        if self.family == "depth_anything":
            kind = getattr(self.model.config, "depth_estimation_type", None)
            self.output_kind = {"metric": "metric", "relative": "relative_disparity"}.get(kind)
        else:
            self.output_kind = OUTPUT_KIND.get(self.family)
        if self.output_kind is None:
            raise ValueError(f"{model_cfg_name}: depth model family '{self.family}' is not supported "
                             f"(supported: depth_anything, {', '.join(sorted(OUTPUT_KIND))})")
        self.max_depth = getattr(self.model.config, "max_depth", None) if self.output_kind == "metric" else None
        self.input_size = {k: v for k, v in dict(self.processor.size).items() if v}
        if hasattr(self.processor, "keep_aspect_ratio"):
            self.input_size["keep_aspect_ratio"] = bool(self.processor.keep_aspect_ratio)
        self.model_cfg_name = model_cfg_name
        self.device = device

    def get_transforms(self):
        """Return the frame transform: tensor image as uint8, values in [0, 255]."""
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=False)])

    def forward(self, X):
        """Estimate depth for one RGB uint8 frame (1, C, H, W).

        Returns:
            float32 array (H, W) at the frame size: metres (output_kind "metric") or relative inverse depth
            ("relative_disparity").
        """
        img = v2.functional.to_pil_image(X[0])
        w, h = img.size
        inp = self.processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inp)
        kw = {"source_sizes": [(h, w)]} if self.family == "zoedepth" else {}
        d = self.processor.post_process_depth_estimation(out, target_sizes=[(h, w)], **kw)[0]["predicted_depth"]
        return d.float().cpu().numpy().astype(np.float32)
