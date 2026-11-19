"""Ultralytics pre-trained model wrappers (YOLO26 detection, segmentation, tracking, pose, oriented boxes).

COCO class ids come from labels.py (COCO_label_map). Output per box [class_id, score, x1, x2, y1, y2], normalised.
- conf=self.confidence_threshold is passed to the model.
- Instance segmentation models (yolo26{n,s,m,l,x}-seg.pt, model.task == "segment"): output per
  instance [class_id, score, x1, x2, y1, y2, W, H, rle], rle = pycocotools compressed RLE string of the
  binary mask at the input image size. retina_masks=True gives masks at the original image
  resolution (same as Ultralytics' own COCO-JSON mask evaluation, process_mask_native); without it
  the masks are at the letterboxed inference size.
- Tracking (tracker="botsort.yaml", "bytetrack.yaml", ... or a path to a tracker YAML): frames must be fed in
  order; output per tracked box [class_id, score, x1, x2, y1, y2, track_id] (only boxes with a track id).
  Detection confidence in track mode defaults to 0.1 (Ultralytics' track() default). reset_tracking() starts a
  new sequence.
- Pose models (yolo26{n,s,m,l,x}-pose.pt, model.task == "pose"): output per person
  [1, score, x1, x2, y1, y2, [x, y, s] * 17] with keypoint x, y normalised and s = keypoint confidence.
- Oriented box models (yolo26{n,s,m,l,x}-obb.pt, model.task == "obb", trained on DOTA v1.0): output per box
  [class_id, score, x1, y1, x2, y2, x3, y3, x4, y4] = DOTA 0-based class id (labels_dota.py, no +1) and the 4
  corners normalised (Results.obb.xyxyxyxyn). They run at their training size (imgsz 1024 from the checkpoint's
  train_args); all other models keep imgsz 640.
- imgsz: optional input size (positive multiple of 32) that replaces the default above (vcmtest config
  sequence_overrides.<seq>.input_size): the longer frame side is resized to imgsz and the frame letterboxed
  (Ultralytics rect inference). Used for prediction and tracking.
"""

import numpy as np
import torch
from torchvision.transforms import v2


class UltralyticsTasks(torch.nn.Module):
    """Ultralytics YOLO models (detection, segmentation, tracking, pose, oriented boxes).

    Input: a single RGB uint8 image (1, C, H, W).
    """

    def __init__(self, model_cfg_name = "yolo26m.pt", device = "cuda:0", tracker = None, imgsz = None):
        super().__init__()

        from ultralytics import YOLO

        self.confidence_threshold = 0.25 # [0.0:1.0], default 0.25
        self.tracker = tracker
        if tracker:
            self.confidence_threshold = 0.1  # Ultralytics track() default: trackers need low-score detections

        self.device = device
        self.model = YOLO(model_cfg_name)
        self.model.to(device)
        self.instance_segmentation = self.model.task == "segment"
        self.keypoints = self.model.task == "pose"
        self.obb = self.model.task == "obb"
        self.imgsz = 640
        if self.obb:
            ckpt = getattr(self.model, "ckpt", None) or {}
            self.imgsz = int((ckpt.get("train_args") or {}).get("imgsz") or 1024)
        if imgsz is not None:
            if int(imgsz) <= 0 or int(imgsz) % 32:
                raise ValueError(f"imgsz must be a positive multiple of 32, got {imgsz}")
            self.imgsz = int(imgsz)
        self.mask_threshold = 0.5   # Ultralytics binarises mask logits at 0 (= probability 0.5)

    def get_transforms(self,):
        """Return the frame transform: tensor image (uint8, values in [0, 255])."""

        transforms = v2.Compose([v2.ToImage(),
        ])
        return transforms

    def _parse_predictions(self, results):

        predictions = []
        if self.instance_segmentation:
            return self._parse_segmentation(results[0])
        if self.keypoints:
            return self._parse_pose(results[0])
        if self.obb:
            return self._parse_obb(results[0])
        for r in results[0]:
            if(r.boxes.conf.cpu().numpy()[0] >= self.confidence_threshold):
                predictions.append([int((r.boxes.cls+1).cpu()), r.boxes.conf.cpu().numpy()[0],
                                r.boxes.xyxyn[0][0].cpu().numpy(),
                                r.boxes.xyxyn[0][2].cpu().numpy(),
                                r.boxes.xyxyn[0][1].cpu().numpy(),
                                r.boxes.xyxyn[0][3].cpu().numpy()])

        return predictions

    def reset_tracking(self):
        """Start a new sequence: the next track() call creates a fresh predictor and tracker state."""
        self.model.predictor = None

    def _parse_tracks(self, res):
        predictions = []
        b = res.boxes
        if b is None or b.id is None or len(b) == 0:
            return predictions
        cls, conf = b.cls.cpu().numpy(), b.conf.cpu().numpy()
        xyxyn, ids = b.xyxyn.cpu().numpy(), b.id.cpu().numpy()
        for i in range(len(cls)):
            predictions.append([int(cls[i]) + 1, float(conf[i]), float(xyxyn[i][0]), float(xyxyn[i][2]),
                                float(xyxyn[i][1]), float(xyxyn[i][3]), int(ids[i])])
        return predictions

    def _parse_pose(self, res):
        predictions = []
        if res.keypoints is None or len(res.boxes) == 0:
            return predictions
        conf, xyxyn = res.boxes.conf.cpu().numpy(), res.boxes.xyxyn.cpu().numpy()
        kxy = res.keypoints.xyn.cpu().numpy()                                  # (N, 17, 2) normalised
        kc = (res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None
              else np.ones(kxy.shape[:2], np.float32))
        for i in range(len(conf)):
            if conf[i] < self.confidence_threshold:
                continue
            kp = np.concatenate([kxy[i], kc[i][:, None]], 1).ravel()
            predictions.append([1, float(conf[i]), float(xyxyn[i][0]), float(xyxyn[i][2]), float(xyxyn[i][1]),
                                float(xyxyn[i][3]), [round(float(v), 6) for v in kp]])
        return predictions

    def _parse_obb(self, res):
        predictions = []
        if res.obb is None or len(res.obb) == 0:
            return predictions
        cls, conf = res.obb.cls.cpu().numpy(), res.obb.conf.cpu().numpy()
        pts = res.obb.xyxyxyxyn.cpu().numpy()                  # (N, 4, 2) normalised corners
        for i in range(len(cls)):
            if conf[i] < self.confidence_threshold:
                continue
            predictions.append([int(cls[i]), float(conf[i])] + [float(v) for v in pts[i].ravel()])
        return predictions

    def _parse_segmentation(self, res):
        from pycocotools import mask as mask_util
        predictions = []
        if res.masks is None or len(res.boxes) == 0:
            return predictions
        H, W = res.orig_shape
        masks = (res.masks.data > 0.5).cpu().numpy()          # (N, H, W) with retina_masks=True
        assert masks.shape[1:] == (H, W), f"mask size {masks.shape[1:]} != image size {(H, W)}"
        cls = res.boxes.cls.cpu().numpy()
        conf = res.boxes.conf.cpu().numpy()
        xyxyn = res.boxes.xyxyn.cpu().numpy()
        for i in range(len(cls)):
            if conf[i] < self.confidence_threshold:
                continue
            rle = mask_util.encode(np.asfortranarray(masks[i].astype(np.uint8)))
            predictions.append([int(cls[i]) + 1, float(conf[i]),
                                float(xyxyn[i][0]), float(xyxyn[i][2]), float(xyxyn[i][1]), float(xyxyn[i][3]),
                                int(W), int(H), rle["counts"].decode("ascii")])
        return predictions

    def forward(self, x):
        """Run the model, or the tracker if set, on one RGB uint8 frame (1, C, H, W).

        Returns:
            The per-frame list in the output format of the model's task (see the module docstring).
        """
        if self.tracker:
            with torch.no_grad():
                res = self.model.track(v2.functional.to_pil_image(x.squeeze()), persist=True, tracker=self.tracker,
                                       device=self.device, verbose=False, imgsz=self.imgsz,
                                       conf=self.confidence_threshold)
            return self._parse_tracks(res[0])
        with torch.no_grad():
            preds = self.model(v2.functional.to_pil_image(x.squeeze()), device=self.device, verbose=False,
                               imgsz=self.imgsz, conf=self.confidence_threshold,
                               retina_masks=self.instance_segmentation)

        return self._parse_predictions(preds)
