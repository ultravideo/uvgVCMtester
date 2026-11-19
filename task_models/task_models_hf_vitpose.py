#!/usr/bin/env python3
"""Top-down keypoint detection with Hugging Face ViTPose / ViTPose++ (usyd-community/vitpose-*), COCO 17 keypoints.

The person boxes come from a detection model of the detection task (config key `detector`): the pipeline passes
that detector's cached predictions for the same decoded frame, so the compression affects both stages.

Per frame:
1. Person boxes (class_id 1) of the detector with score >= score_threshold (default 0.0 = all boxes, as in the
   ViTPose / mmpose COCO evaluation with detector results).
2. ViTPose on each box (the model's image processor: box crop with padding, 256x192 input, UDP heatmap decoding
   in post_process_pose_estimation). ViTPose++ checkpoints have several dataset heads: the COCO head (index 0).
3. Instance score = box score x mean of the keypoint scores above 0.2 (0 if none), then OKS non-maximum
   suppression at 0.9 (both as in mmpose's CocoMetric, used for the ViTPose COCO results).

Output per person: [1, score, x1, x2, y1, y2, [x, y, s] * 17], box = the detector box, keypoint x, y normalised,
s = keypoint score.
"""

import numpy as np
import torch
from torchvision.transforms import v2

COCO_SIGMAS = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0


def oks_nms(kps: np.ndarray, scores: np.ndarray, areas: np.ndarray, thr: float = 0.9) -> list[int]:
    """Greedy OKS NMS (mmpose oks_nms). kps (N, 17, 2) pixels; returns the kept indices, best score first."""
    order = list(np.argsort(-scores))
    var = (2 * COCO_SIGMAS) ** 2
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        if not order:
            break
        rest = np.array(order)
        d2 = ((kps[rest] - kps[i]) ** 2).sum(-1)                        # (M, 17)
        a = (areas[i] + areas[rest])[:, None] / 2 + np.spacing(1)
        oks = np.exp(-d2 / var / a / 2).mean(1)
        order = [j for j, o in zip(order, oks) if o <= thr]
    return keep


class HFVitPose(torch.nn.Module):
    """Top-down COCO keypoint detection with ViTPose on detector person boxes.

    Input: a single RGB uint8 image (1, C, H, W).
    """
    keypoints = True
    DEFAULT_THRESHOLD = 0.0          # minimum detector person-box score
    KEYPOINT_SCORE_THR = 0.2
    OKS_NMS_THR = 0.9
    MAX_BOXES_PER_BATCH = 32

    def __init__(self, model_cfg_name="usyd-community/vitpose-plus-large", device="cuda:0"):
        super().__init__()
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        self.processor = AutoProcessor.from_pretrained(model_cfg_name)
        self.model = VitPoseForPoseEstimation.from_pretrained(model_cfg_name).to(device).eval()
        names = [n.lower() for _, n in sorted(self.model.config.id2label.items())]
        if len(names) != 17 or not names[0].startswith("nose") or not names[16].endswith("ankle"):
            raise ValueError(f"{model_cfg_name}: expected the 17 COCO keypoints, got {names}")
        self.num_experts = int(getattr(self.model.config.backbone_config, "num_experts", 1) or 1)
        self.device = device
        self.score_threshold = self.DEFAULT_THRESHOLD
        self.input_size = dict(self.processor.size) if hasattr(self.processor, "size") else None
        self.input_size = {k: v for k, v in (self.input_size or {}).items() if v}

    def get_transforms(self):
        """Return the frame transform: tensor image as uint8, values in [0, 255]."""
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=False)])

    def forward(self, X, boxes):
        """X (1, 3, H, W) uint8; boxes = detector predictions [class_id, score, x1, x2, y1, y2, ...] (normalised)."""
        img = v2.functional.to_pil_image(X[0])
        w, h = img.size
        person = [b for b in boxes if int(b[0]) == 1 and float(b[1]) >= self.score_threshold]
        if not person:
            return []
        bs = np.array([float(b[1]) for b in person])
        xyxy = np.array([[b[2] * w, b[4] * h, b[3] * w, b[5] * h] for b in person], np.float32)
        xywh = np.concatenate([xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]], 1)
        valid = (xywh[:, 2] > 1) & (xywh[:, 3] > 1)
        bs, xyxy, xywh = bs[valid], xyxy[valid], xywh[valid]
        kps, ksc = [], []
        for s in range(0, len(xywh), self.MAX_BOXES_PER_BATCH):
            chunk = xywh[s:s + self.MAX_BOXES_PER_BATCH]
            inp = self.processor(img, boxes=[chunk], return_tensors="pt").to(self.device)
            extra = {}
            if self.num_experts > 1:
                extra["dataset_index"] = torch.zeros(len(chunk), dtype=torch.long, device=self.device)   # COCO
            with torch.no_grad():
                out = self.model(**inp, **extra)
            res = self.processor.post_process_pose_estimation(out, boxes=[chunk])[0]
            for r in res:
                kps.append(r["keypoints"].cpu().numpy())
                ksc.append(r["scores"].cpu().numpy())
        if not kps:
            return []
        kps, ksc = np.stack(kps), np.stack(ksc)                                     # (N, 17, 2), (N, 17)
        above = ksc > self.KEYPOINT_SCORE_THR
        mean_k = np.where(above.any(1), (ksc * above).sum(1) / np.maximum(above.sum(1), 1), 0.0)
        scores = bs * mean_k
        areas = xywh[:, 2] * xywh[:, 3]
        keep = oks_nms(kps, scores, areas, self.OKS_NMS_THR)
        preds = []
        for i in keep:
            kp = np.concatenate([kps[i] / [w, h], ksc[i][:, None]], 1).ravel()
            preds.append([1, float(scores[i]), float(xyxy[i, 0] / w), float(xyxy[i, 2] / w), float(xyxy[i, 1] / h),
                          float(xyxy[i, 3] / h), [round(float(v), 6) for v in kp]])
        return preds
