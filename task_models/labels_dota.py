"""DOTA v1.0 categories (15), 0-based ids in the order of Ultralytics DOTAv1.yaml (= the yolo*-obb.pt model outputs).

Used by the oriented_detection task: <sequence_name>_obb_annotations.json class_id and the predictions use these ids.
"""

DOTA_label_map = {
    "plane": 0,
    "ship": 1,
    "storage tank": 2,
    "baseball diamond": 3,
    "tennis court": 4,
    "basketball court": 5,
    "ground track field": 6,
    "harbor": 7,
    "bridge": 8,
    "large vehicle": 9,
    "small vehicle": 10,
    "helicopter": 11,
    "roundabout": 12,
    "soccer ball field": 13,
    "swimming pool": 14,
}
