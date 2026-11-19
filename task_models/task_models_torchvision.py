"""Torchvision pre-trained object detection, instance segmentation and keypoint model wrappers.

COCO class ids come from labels.py (COCO_label_map). Output per box [class_id, score, x1, x2, y1, y2], normalised.
- Model names are matched exactly: "fasterrcnn_resnet50_fpn" is v1 and "fasterrcnn_resnet50_fpn_v2" is v2.
- Mask R-CNN (instance segmentation): maskrcnn_resnet50_fpn and maskrcnn_resnet50_fpn_v2.
  Output per instance: [class_id, score, x1, x2, y1, y2, W, H, rle], where rle is the
  pycocotools compressed RLE string ("counts") of the mask binarised at mask_threshold (0.5).
- Keypoint R-CNN (keypoint detection): keypointrcnn_resnet50_fpn (COCO_V1 weights). Output per person:
  [1, score, x1, x2, y1, y2, [x, y, s] * 17], keypoint x, y normalised, s = torchvision keypoint score (logit).
"""

import numpy as np
import torch
from labels import COCO_label_map
from torchvision.transforms import v2


class TorchvisionTasks(torch.nn.Module):
    """
    Torchvision pre-trained object detection models.
    Assumes single input image (1, C, H, W) in RGB format,
    pixel value range [0.0, 1.0]

    """

    CLASSES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant', None, 'stop sign',
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', None, 'backpack', 'umbrella', None,
               None, 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', None, 'wine glass', 'cup', 'fork',
               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
               'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
               'bed', None, 'dining table', None, None, 'toilet', None, 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', None, 'book', 'clock', 'vase', 'scissors', 'teddy bear',
               'hair drier', 'toothbrush']

    def __init__(self, model_cfg_name="retinanet_resnet50_fpn_v2", device="cuda:0"):
        super().__init__()

        self.instance_segmentation = False
        self.keypoints = False
        if model_cfg_name == "retinanet_resnet50_fpn_v2":
            from torchvision.models.detection.retinanet import (
                RetinaNet_ResNet50_FPN_V2_Weights,
                retinanet_resnet50_fpn_v2,
            )
            self.model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1).to(device)
            self.model.eval()
        elif model_cfg_name == "fasterrcnn_resnet50_fpn":
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
            self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1).to(device)
            self.model.eval()
        elif model_cfg_name == "fasterrcnn_resnet50_fpn_v2":
            from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
            self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1).to(device)
            self.model.eval()
        elif model_cfg_name == "fcos_resnet50_fpn":
            from torchvision.models.detection import FCOS_ResNet50_FPN_Weights, fcos_resnet50_fpn
            self.model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT).to(device)
            self.model.eval()
        elif model_cfg_name == "maskrcnn_resnet50_fpn":
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
            self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1).to(device)
            self.model.eval()
            self.instance_segmentation = True
        elif model_cfg_name == "maskrcnn_resnet50_fpn_v2":
            from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
            self.model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1).to(device)
            self.model.eval()
            self.instance_segmentation = True
        elif model_cfg_name == "keypointrcnn_resnet50_fpn":
            from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights, keypointrcnn_resnet50_fpn
            self.model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1).to(device)
            self.model.eval()
            self.keypoints = True
        else:
            raise AssertionError(f'Unknown model cfg name: {model_cfg_name}')

        self.prediction_limit = 100  # limit predictions to max 100 for each input
        self.mask_threshold = 0.5    # soft mask -> binary mask (torchvision / COCO convention)

    def _parse_detections(self, detections, original_image_sizes):
        # Should return a list of results, with each result in the following form:
        # [class_name, scores[ii]] + bboxes[ii].tolist()
        results = []
        H, W = original_image_sizes[-2:]
        for det in detections:
            masks = None
            if self.instance_segmentation:
                from pycocotools import mask as mask_util
                n = min(len(det["labels"]), self.prediction_limit)
                # (N, 1, H, W) soft masks at input resolution -> binary, Fortran order for pycocotools
                binm = (det["masks"][:n, 0] > self.mask_threshold).to("cpu").numpy()
                masks = []
                for m in binm:
                    rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))
                    masks.append(rle["counts"].decode("ascii"))
            for ii, (label, score, box) in enumerate(zip(det["labels"], det["scores"], det["boxes"])):
                if ii >= self.prediction_limit:
                    break
                class_name = self.CLASSES[label]
                class_id = COCO_label_map[class_name]  # to generic IDs
                scores = score.item()
                bboxes = box  # Should be a tensor of shape [N, 4]
                # x1, y1, x2, y2 in pixels above, now scaling coordinates to percentages
                # and changing to x1, x2, y1, y2
                bboxes = bboxes / torch.tensor([W, H, W, H], device=bboxes.device)
                bboxes = bboxes[[0, 2, 1, 3]]

                # Final format below is class_id, confidence score, x1, x2, y1, y2
                rslt = [class_id, scores] + bboxes.tolist()
                if self.keypoints:
                    kp = det["keypoints"][ii, :, :2] / torch.tensor([W, H], device=bboxes.device)
                    ks = det["keypoints_scores"][ii][:, None]
                    rslt += [[round(float(v), 6) for v in torch.cat([kp, ks], 1).reshape(-1).tolist()]]
                if masks is not None:
                    rslt += [int(W), int(H), masks[ii]]
                results.append(rslt)
        return results[:self.prediction_limit]  # assumes detections were ordered from highest confidence to lowest

    # for dataloader to transform the data to correct format for this model
    def get_transforms(self,):
        """Return the frame transform: tensor image as float32, scaled to [0, 1]."""
        transforms = v2.Compose([v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
        ])
        return transforms

    # assume pixel values are in range [0.0 1.0], RGB
    def forward(self, x):
        """Run the model on RGB frames (N, C, H, W), values in [0, 1].

        Returns:
            At most prediction_limit entries [class_id, score, x1, x2, y1, y2] normalised (labels.py ids); Mask R-CNN
            adds [W, H, rle], Keypoint R-CNN adds [[x, y, s] * 17].
        """
        with torch.no_grad():

            preds = self.model(x)
            parsed_preds = self._parse_detections(preds, x.shape)
        return parsed_preds
