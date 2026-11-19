"""Detectron2 pre-trained object detection and instance segmentation model wrappers.

COCO class ids come from labels.py (COCO_label_map). Output per box [class_id, score, x1, x2, y1, y2], normalised;
instance segmentation models (mask_rcnn_*) add [W, H, rle] (pycocotools compressed RLE of detectron2's binary mask
at the frame size), as the other wrappers. Detectron2 is installed manually (docs/MODELS.md).
"""

import os

import torch
from labels import COCO_label_map
from torchvision.transforms import v2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silence tensorflow messages

class Detectron2Tasks(torch.nn.Module):
    """
    Detectron2 models.
    Assumes single input image (1, C, H, W) in RGB format,
    pixel value range [0.0, 255.0]

    """

    def __init__(self, model_cfg_name = "faster_rcnn_X_101_32x8d_FPN_3x.yaml", device = "cuda:0"):
        super().__init__()

        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        self.confidence_threshold = 0.05 # detectron2 default SCORE_THRESH_TEST


        COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane',
                        'bus', 'train', 'truck', 'boat', 'traffic light',
                        'fire hydrant', 'stop sign', 'parking meter', 'bench',
                        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                        'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                        'book', 'clock', 'vase', 'scissors', 'teddy bear',
                        'hair drier', 'toothbrush']

        #add the the required "COCO-Detection/" or "COCO-InstanceSegmentation"
        # into the model_cfg_name
        if("/" not in model_cfg_name):
            if("mask" in model_cfg_name):
                model_cfg_name = "COCO-InstanceSegmentation/" + model_cfg_name
            else:
                model_cfg_name = "COCO-Detection/" + model_cfg_name

        # THIS IS DONE IN A JVET-CTC WAY, check detectron2/detectron2_predict.py and JVET_template/
        self.cfg = get_cfg()
        #CTC Detection: "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
        #CTC Segmentation: "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
        self.cfg.merge_from_file(model_zoo.get_config_file(model_cfg_name))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold  # set threshold for this model
        # Find a model from detectron2's model zoo. https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg_name)
        self.cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(self.cfg)
        self.coco_classes = COCO_CLASSES

        #used in predictions reformulation
        if("Detection" in model_cfg_name):
            # this should be correct to later select a valid evaluation metric and labels
            self.task = "detection"
        elif("Segmentation" in model_cfg_name):
            self.task = "instance_segmentation"
        else: # this should not be reached
            raise AssertionError(f'Unknown model cfg name: {model_cfg_name}')
        # instance segmentation output (vcmtest): masks pasted at the input frame size and binarised by
        # detectron2's postprocess at 0.5 (paste_masks_in_image default)
        self.instance_segmentation = self.task == "instance_segmentation"
        self.mask_threshold = 0.5


    # Reformulate the output from predictor, returns a list, which each item is
    # a detection: [class, confidence, box_x1, box_x2, box_y1, box_y2], coordinates in percentage
    def _parse_predictions(self, all_outputs):
        results = []                       # detections of all frames (as the other wrappers; batch size 1 in use)
        for outputs in all_outputs:
            #copy from detectron2/detectron2_predict.py
            classes = outputs['instances'].pred_classes.to('cpu').numpy()
            if(len(classes) > 0):
                scores = outputs['instances'].scores.to('cpu').numpy()
                bboxes = outputs['instances'].pred_boxes.tensor.to('cpu').numpy()
                H, W = outputs['instances'].image_size
                # convert bboxes to 0-1
                # detectron: x1, y1, x2, y2 in pixels
                bboxes = bboxes / [W, H, W, H] #in percentage

                # Rearrange output x1, x2, y1, y2
                bboxes = bboxes[:, [0, 2, 1, 3]]
                masks = None
                if self.instance_segmentation:
                    import numpy as np
                    from pycocotools import mask as mask_util
                    binm = outputs['instances'].pred_masks.to('cpu').numpy()   # (N, H, W) bool
                    masks = [mask_util.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode("ascii")
                             for m in binm]

                for ii in range(len(classes)):
                    coco_cnt_id = classes[ii]
                    class_name = self.coco_classes[coco_cnt_id]
                    class_id = COCO_label_map[class_name] #to generic IDs

                    # if(self.instance_segmentation): #segmentation':
                    #     assert (masks[ii].shape[1]==W) and (masks[ii].shape[0]==H), \
                    #         print('Detected result does not match the input image size')

                    rslt = [class_id, scores[ii]] + bboxes[ii].tolist()
                    if masks is not None:
                        rslt += [int(W), int(H), masks[ii]]
                    # rslt = [imageId, class_name, scores[ii]] + bboxes[ii].tolist()

                    # if(self.instance_segmentation): # 'segmentation':
                    #     rslt += [masks[ii].shape[1], masks[ii].shape[0], \
                    #          oid_mask_encoding.encode_binary_mask(masks[ii]).decode('ascii')]

                    results.append(rslt)

        return results #return, in case if needed


    def get_transforms(self,):
        """Return the frame transform: tensor image as float32, values kept in [0, 255]."""
        transforms = v2.Compose([v2.ToImage(),
            v2.ToDtype(torch.float32, scale=False), #keep between [0 255]
        ])
        return transforms

    #assume pixel values are in range [0 255]
    def forward(self, X):
        """Run the model on RGB frames (N, C, H, W), values in [0, 255].

        Each frame is converted to BGR and resized as detectron2's test resize (shorter side 800, longer side <= 1333).

        Returns:
            Per detection [class_id, score, x1, x2, y1, y2] normalised (labels.py ids); instance segmentation adds
            [W, H, rle]. With several frames, the detections of all frames are returned in one list (as the
            other wrappers; the pipeline uses batch size 1).
        """

        with torch.no_grad():
            inputs = []
            predictions = []
            for x in X:
                ## OPTION 1 pre-processing ##
                # Moved pre-processing functionalities from Detectron2 codes
                # here to make this faster
                height, width = x.shape[1:]
                new_size = self.predictor.aug.get_output_shape(height, width, 800, 1333)
                #RGB to BGR and resize
                image = torch.nn.functional.interpolate(x[[2, 1, 0], ...].unsqueeze(0), (new_size[0], new_size[1]),
                                                        mode="bilinear", align_corners=None).squeeze(0)
                inputs.append({"image": image, "height": height, "width": width})

            predictions = self.predictor.model(inputs)

                ## ##


                # ## OPTION 2 pre-processing ##
                # # keep pre-processing in the model input
                # # pre-processing does not accept very high resolutions, scale to 2.5K
                # height, width = x.shape[1:]
                # new_size = self.predictor.aug.get_output_shape(height, width, 1440, 2560)
                # image = F.interpolate(x[[2, 1, 0],...].unsqueeze(0), (new_size[0], new_size[1]), mode="bilinear",
                #                       align_corners=None).squeeze(0)
                # predictions.append(self.predictor(image.cpu().numpy().transpose(1,2,0)))
                ## ##

            parsed_predictions = self._parse_predictions(predictions)

        return parsed_predictions
