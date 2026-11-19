"""RT-DETR / RT-DETRv2 (Hugging Face transformers) pre-trained object detection model wrapper.

COCO class ids come from labels.py (COCO_label_map). Output per box [class_id, score, x1, x2, y1, y2], normalised.
"""
import torch
from torchvision.transforms import v2


class RTDetr(torch.nn.Module):
    """
    RT-DETR via huggingface.
    Assumes single input image (1, C, H, W) in RGB format and uint8,
    pixel value range [0.0, 255.0]

    Using transformers library.

    """
    def __init__(self, model_cfg_name = "PekingU/rtdetr_v2_r18vd", device = "cuda:0", input_size = None):
        super().__init__()

        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        self.image_processor = RTDetrImageProcessor.from_pretrained(model_cfg_name)
        self.model = RTDetrV2ForObjectDetection.from_pretrained(model_cfg_name, device_map=device)

        self.model_cfg_name = model_cfg_name

        self.device = device
        self.score_threshold = 0.5 # transformers post_process_object_detection default
        # Input size: None = image processor default (stretch to 640 x 640). N (multiple of 32) = stretch to N x N
        # like the default, i.e. the geometry RT-DETR was trained with (COCO resized to a square without keeping the
        # aspect ratio). Keeping the aspect ratio (long side N) was tried first and was worse at the same pixel count
        # (WinterDrive QP29: 1280 x 736 37.7 AP vs 960 x 960 52.0 / 640 x 640 51.6). RT-DETR builds its sinusoidal
        # positional embeddings and anchors for the actual feature-map size.
        if input_size is not None and (int(input_size) <= 0 or int(input_size) % 32):
            raise ValueError(f"input_size must be a positive multiple of 32, got {input_size}")
        self.input_size = None if input_size is None else int(input_size)

    def resize_to(self, width, height):
        """(height, width) given to the image processor, or None = processor default (640 x 640)."""
        if self.input_size is None:
            return None
        return self.input_size, self.input_size

    def get_transforms(self,):
        """Return the frame transform: tensor image as uint8, values in [0, 255]."""
        transforms = v2.Compose([v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=False), #keep between [0 255]
        ])
        return transforms

    def _parse_predictions(self, results, size):

        predictions = []
        for r in results:
            for score, label_id, box in zip(r["scores"], r["labels"], r["boxes"]):
                box = box.tolist()
                predictions.append([label_id.item()+1, score.item(),
                            box[0] / size[0],
                            box[2] / size[0],
                            box[1] / size[1],
                            box[3] / size[1]])

        return predictions

    def forward(self, X):
        """Run RT-DETR on one RGB uint8 frame (1, C, H, W).

        Returns:
            Per box [class_id, score, x1, x2, y1, y2] normalised, class_id = model label + 1 (labels.py ids); boxes
            below score_threshold are dropped by the post-processing.
        """

        X = v2.functional.to_pil_image(X[0])

        size = self.resize_to(X.width, X.height)
        kw = {} if size is None else {"size": {"height": size[0], "width": size[1]}}
        inp = self.image_processor(images=X, return_tensors="pt", **kw).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inp)

        preds = self.image_processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([(X.height, X.width)]), threshold=self.score_threshold)

        return self._parse_predictions(preds, (X.width, X.height))
