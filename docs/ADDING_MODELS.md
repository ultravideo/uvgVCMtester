# Adding task models

How to add your own models. There are three cases, from least to most work:

1. **A new model of a family that is already supported**, e.g. another YOLO26 size, a different Hugging Face
   RT-DETR checkpoint or another torchvision detector: **config only** (section 1).
2. **A new model family** (a library or architecture with no wrapper yet) that runs in the main Python
   environment: **a wrapper file plus one registry entry** (sections 2–4).
3. **A new family that needs its own Python environment** because its dependencies clash with the main venv
   (as Roboflow `inference` and MMRotate do): a wrapper that runs as a **worker subprocess** (section 5).

New *tasks* (a new metric or annotation type) are a larger change and are not covered here.

---

## 1. A new model of an existing family: config only

Add a line under the task in `configs/config.yaml`:

```yaml
tasks:
  - task: detection
    models:
      - {wrapper: ultralytics, model: yolo26l.pt, score_threshold: 0.05}
      - {wrapper: rtdetr, model: PekingU/rtdetr_v2_r50vd, score_threshold: 0.05}
```

| `wrapper` | `model` accepts | Tasks |
|---|---|---|
| `torchvision` | torchvision model builder names, e.g. `fasterrcnn_resnet50_fpn_v2`, `maskrcnn_resnet50_fpn_v2`, `keypointrcnn_resnet50_fpn` | detection, instance seg, keypoints |
| `ultralytics` | Ultralytics weight files, e.g. `yolo26m.pt`, `yolo26x-seg.pt`, `yolo26m-pose.pt`, `yolo26m-obb.pt`, or a path to your own `.pt` | detection, instance seg, keypoints, oriented detection, tracking |
| `rtdetr` | Hugging Face RT-DETRv2 ids, e.g. `PekingU/rtdetr_v2_r18vd` | detection |
| `rfdetr` | `rfdetr-{nano,small,medium,large}`, `rfdetr-seg-*`, `rfdetr-keypoint-preview` | detection, instance seg, keypoints |
| `hf_panoptic`, `hf_vitpose`, `hf_depth` | Hugging Face ids of the supported families (see `docs/MODELS.md`) | panoptic / semantic, keypoints, depth |

The full list with licences is in [MODELS.md](MODELS.md). Then:

```bash
python vcmtest --plan                    # the model id appears in the plan; config errors are reported here
python vcmtest --seq <sequence> --qp 27 47   # validate on two QPs before a full run
```

Each model gets an id (e.g. `ultralytics-yolo26l.pt-thr0.05`). Its predictions, scores and per-model plot are
stored under that id, so models never overwrite each other. Add `name: <id>` to a model line to choose the id
yourself.

**Your own fine-tuned weights:** for `ultralytics`, give the path to the `.pt` file. The class ids must follow the
80-class COCO order of `task_models/labels.py`; this is checked when the model loads. A model trained on other
classes needs its own label mapping, which means a new wrapper (section 3).

---

## 2. How a wrapper works

A *wrapper* is one Python file in `task_models/` that turns a library's model into the common interface. The
pipeline does everything else: decoding, the loop over frames, caching and resume, OOM handling, scoring,
bitrates and plots.

### What the pipeline gives your wrapper
- One frame at a time (batch size 1), as an **RGB uint8 PIL image** at the decoded resolution (e.g. 1920×1080).
- Before the call, the image goes through your `get_transforms()` (usually: to a uint8 or float tensor) and gets
  a batch dimension: `forward(X)` receives `X` of shape `(1, 3, H, W)`.
  - The tensor is moved to the GPU first only if the wrapper is registered with `input_on_device = True`
    (section 3, step 4).
  - Do the model-specific resizing and normalisation inside `forward` (or with the library's image processor).

### What your wrapper must return: one list per detected object, per frame

| Task | Per object | Notes |
|---|---|---|
| detection | `[class_id, score, x1, x2, y1, y2]` | **coordinates normalised to [0, 1] and in the order x1, x2, y1, y2** (not x1, y1, x2, y2) |
| instance segmentation | `[class_id, score, x1, x2, y1, y2, W, H, rle]` | `rle` = pycocotools compressed RLE string of the binary mask at the frame size W × H: `mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")` |
| keypoints | `[1, score, x1, x2, y1, y2, [x, y, s] * 17]` | 17 COCO keypoints, x, y normalised, s = keypoint score |

- `class_id` is the **generic COCO id 1..80** of `task_models/labels.py` (`COCO_label_map`, name → id). Libraries
  differ: some use 0..79, some the 91-id COCO order with gaps. **Map by class name** (the model's `id2label` or
  `names`) through `COCO_label_map` to be safe. Drop classes that are not in `labels.py`.
- Return every detection above the score threshold. Do not cap the number of boxes yourself; COCO AP uses the top
  100 per frame.

### What the wrapper object should provide

| Attribute / method | Meaning |
|---|---|
| `get_transforms()` | a torchvision `v2` transform from the PIL image to a tensor |
| `forward(X)` | the predictions of one frame, in the format above |
| a score threshold attribute | e.g. `self.score_threshold`, set to the **library default**; the pipeline overrides it when the config gives `score_threshold` |
| `self.model` (or another `torch.nn.Module`) | its parameters are hashed into `weights_sha256`, which is recorded in the results and part of the cache key |
| `instance_segmentation = True` and `mask_threshold` | only for mask models |
| `keypoints = True` | only for keypoint models |

**Tracking comes for free:** any detection wrapper can be combined with the Roboflow `trackers` algorithms
(`tracker: bytetrack`, `ocsort`, `sort`, `botsort`, `cbiou`) in the tracking task, with no extra code.

---

## 3. A new model family in the main environment: step by step

The example adds a generic Hugging Face object-detection wrapper (`AutoModelForObjectDetection`). It covers
D-FINE, YOLOS, Conditional DETR and others; the same steps apply to any library. The wrapper name in the
config is `hf_detection`.

### Step 1: the wrapper file `task_models/task_models_hf_detection.py`

```python
"""Object detection with Hugging Face transformers models (AutoModelForObjectDetection), e.g.
ustc-community/dfine-small-coco (D-FINE), hustvl/yolos-small (YOLOS).
Output per instance: [class_id, score, x1, x2, y1, y2], generic COCO ids, normalised coordinates."""

import torch
from torchvision.transforms import v2

from labels import COCO_label_map

SYNONYMS = {"motorbike": "motorcycle", "aeroplane": "airplane", "sofa": "couch", "pottedplant": "potted plant",
            "diningtable": "dining table", "tvmonitor": "tv"}

class HFDetection(torch.nn.Module):
    """Input: (1, 3, H, W) uint8 tensor, RGB."""

    def __init__(self, model_cfg_name="ustc-community/dfine-small-coco", device="cuda:0"):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForObjectDetection   # import inside the wrapper
        self.processor = AutoImageProcessor.from_pretrained(model_cfg_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_cfg_name).to(device).eval()
        self.device = device
        self.score_threshold = 0.5            # library default of post_process_object_detection
        self.id_map, self.unmapped = {}, []   # model label id -> generic COCO id, by class name
        for i, name in self.model.config.id2label.items():
            cid = COCO_label_map.get(SYNONYMS.get(name, name))
            if cid is None:
                self.unmapped.append(name)     # e.g. "N/A" slots of the 91-id order
            else:
                self.id_map[int(i)] = cid

    def get_transforms(self):
        return v2.Compose([v2.ToImage(), v2.ToDtype(torch.uint8, scale=False)])

    @torch.no_grad()
    def forward(self, X):
        img = v2.functional.to_pil_image(X[0])
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        out = self.model(**inputs)
        r = self.processor.post_process_object_detection(out, threshold=self.score_threshold,
                                                         target_sizes=[(img.height, img.width)])[0]
        W, H = img.width, img.height
        preds = []
        for s, l, b in zip(r["scores"].tolist(), r["labels"].tolist(), r["boxes"].tolist()):
            if int(l) in self.id_map:
                preds.append([self.id_map[int(l)], s, b[0] / W, b[2] / W, b[1] / H, b[3] / H])
        return preds
```

This file was checked: with `ustc-community/dfine-small-coco` at threshold 0.05 it gives 49.5 AP (car, truck)
on 60 frames of a test sequence, close to RT-DETRv2-R18 on the same frames.

**Rules for wrapper files**
- Import the model library **inside** the wrapper (in `__init__`), never at the top of `vcmtest/`, so the core
  runs without it.
- Do not download weights to your home folder. The pipeline sets `TORCH_HOME`, `HF_HOME` and `RF_HOME` to
  `weights/`; for other libraries, pass a path under `weights/` (as `fast_alpr` and `rapidocr` do).
- Keep library defaults (threshold, input size) unless there is a measured reason. Record any change and its reason
  in the docstring and in `docs/MODELS.md`.

### Step 2: register the wrapper in `vcmtest/wrappers.py`
Add one entry to the `WRAPPERS` registry. The key is the name used as `wrapper:` in the config:

```python
    "hf_detection": Wrapper(
        "task_models_hf_detection", "HFDetection",      # task_models/<module>.py and the class
        ("transformers",), "uv sync --extra hfdet"),     # import names checked before a run + install command
```

This is the only code outside the wrapper file. The rest has defaults that fit the example; set a field only when
your wrapper differs:

| Field | Default | Set it when |
|---|---|---|
| `input_on_device` | `False`: `forward()` gets the tensor on the CPU | `forward()` expects `X` already on the GPU (`True`) |
| `threshold` | `"score_threshold"`: the attribute the config's `score_threshold` is written to and the effective value read from | the attribute has another name (`"confidence_threshold"`), it needs code (a `(getter, setter)` pair of functions of the wrapper object), or there is no threshold (`None`) |
| `torch_module` | `"model"`: the attribute whose parameters are hashed into `weights_sha256` | another path (`"model.model"`), a function of the wrapper object, or `None` for non-torch models; then the wrapper sets `self.weights_sha256` itself (e.g. a hash of its ONNX files, as `fast_alpr` and `rapidocr` do) |
| `kwargs` | none | the constructor needs more than `model_cfg_name` and `device`: a function `(mcfg, weights_dir) -> dict` (e.g. `weights_dir` for a download folder, `mcfg["input_size"]`) |
| `check` | none | a check after loading, e.g. the label order: a function `(net) -> None` that raises `ValueError` |
| `describe` | none | a setting changes the predictions but is not in the config line (e.g. a fixed input size): a function `(net, mcfg) -> dict`; it is recorded in the results and is part of the cache key, so a change of it triggers recomputation |
| `keypoints` | none | keypoint models: a function `(model name) -> bool` that tells whether the model may be used in the keypoints task |
| `input_size` | `False` | `input_size` / `sequence_overrides.<seq>.input_size` is implemented (see *Optional features*) |

- `--plan` reports a missing package with its install command, so a run does not fail at model load.
- The config's set of valid wrappers and its install hints come from this registry.

### Step 3: the dependency in `pyproject.toml`
```toml
[project.optional-dependencies]
hfdet = ["transformers"]        # generic HF object detectors (licence of the package)
```
- Then run `uv sync --all-extras` for a development venv, or `uv sync --extra hfdet`.
- If the library cannot share the main venv (it pins other versions of torch, numpy, …), use section 5 instead.

### Step 4: use it
```yaml
  - task: detection
    models:
      - {wrapper: hf_detection, model: ustc-community/dfine-small-coco, score_threshold: 0.05}
  - task: tracking
    models:
      - {wrapper: hf_detection, model: ustc-community/dfine-small-coco, tracker: bytetrack, score_threshold: 0.1}
```
- Score threshold: for AP, use a low value such as 0.05. Library defaults of 0.25–0.5 cut recall and cap AP.
- Score threshold for tracking: 0.1, so the trackers get the low-score detections.

### Step 5: documentation and a test
- `docs/MODELS.md`: a row in the wrapper table (model names, tasks, install, default threshold, **licence of
  the code and of the weights**) and a short note on anything special (class mapping, input size).
- A test: at least a config test (the wrapper is accepted and gets the expected model id). A test that loads
  the model should be skipped when the package is missing:
  `pytest.importorskip("transformers")`.

---

## 4. Check the new model before a full run

1. `python vcmtest --plan`: the model appears under its task, with no missing-package warning.
2. `python vcmtest --seq <sequence> --qp <high-quality QP> <low-quality QP>`, then look at
   `data/<sequence>/results/logs/run_*.log` and the results JSON:
   - **Accuracy should fall as QP rises.** If it does not, check the class mapping and the box order first.
   - Compare with an existing model on the same sequence. AP near 0 almost always means a wrong class mapping or
     box format (x1, x2, y1, y2; normalised). AP clearly lower than expected can mean a wrong input size or
     normalisation (compare with the library's own example on one image).
3. `python tools/check_alignment.py <sequence> <QP> <model_id>` (on a high-quality QP) scores the predictions against
   the annotations shifted by −2..+2 frames: offset 0 should be clearly best. A frame offset in the wrapper's
   output order shows up here.
4. Then run all QPs. The predictions are cached; when the threshold, decode settings or frame count change, they
   are recomputed automatically. Changed weights under the same model name are detected only when the model is
   loaded, not on a QP whose predictions are cached: delete `predictions/<model_id>/` to rerun (README, Caching and
   removed models).

---

## 5. A model family that needs its own Python environment (worker)

Use this when the library pins versions that conflict with the main venv. The model runs in a subprocess with the
other venv's Python; the main process sends frames and receives predictions.

- **Examples to copy:** `task_models/task_models_roboflow.py` (with `envs/roboflow/pyproject.toml`) and
  `task_models/task_models_mmrotate.py` (with `envs/mmrotate/setup.sh`).
- **Protocol** (already implemented on the main side by `SubprocessWorker` in `vcmtest/infer.py`). Each message is
  a 4-byte big-endian length, then a JSON header, then an optional raw payload of `nbytes` bytes:
  - `{"cmd": "load", "model", "device", "score_threshold", "cache_dir", "seed", ...}`
    → `{"ok": true, "score_threshold", "weights_sha256", "instance_segmentation", "mask_threshold", "versions"}`
  - `{"cmd": "predict", "shape": [H, W, 3], "nbytes"}` + RGB uint8 bytes → `{"ok": true, "preds": [...]}`
    (same per-object format as above)
  - `{"cmd": "quit"}`
  - On an error: `{"ok": false, "error": "...", "oom": true|false}`.
  - The worker must write only protocol messages to stdout. The examples redirect library prints to stderr.
- **Main side:**
  - A `SubprocessWorker` subclass in `vcmtest/infer.py` with `KIND`, `SCRIPT` (your worker script in
    `task_models/`) and `SETUP_HINT`.
  - A registry entry in `vcmtest/wrappers.py` with `module="vcmtest.infer"`, `cls=<your SubprocessWorker
    subclass>`, `worker="<wrapper>_python"`, `weights_subdir="<folder under weights/>"` and `install="<setup
    command>"`. Worker-specific settings go to the worker through `kwargs`; extra fields for the results go
    through `describe` (see the `roboflow` and `mmrotate` entries).
  - A `<wrapper>_python` config key with the default venv path (`DEFAULTS` in `vcmtest/config.py`).

---

## Optional features

- **Input size** (global `input_size`, per sequence `sequence_overrides.<seq>.input_size`): accept an `input_size` argument in the
  wrapper. In the registry entry, set `input_size=True`, pass it with
  `kwargs=lambda mcfg, wd: {"input_size": mcfg.get("input_size")}`, and record the effective size with `describe`
  (see the `rtdetr` entry). Until then, a model given an input size is not run on that sequence (a warning).
- **Keypoints** (top-down on another model's boxes, like `hf_vitpose`): see that wrapper and the `detector:` key.
- **Other outputs** (panoptic, depth, oriented boxes, text) have their own formats and are tied to specific
  wrappers in `_validate` (`config.py`) and in `run_inference` (`infer.py`). See the existing wrapper for that
  task (`task_models_hf_panoptic.py`, `task_models_hf_depth.py`, `task_models_mmrotate.py`,
  `task_models_fast_alpr.py`) and the format notes at the top of `vcmtest/infer.py`.

---

## A different bitstream decoder

Not a task model, but the same idea: `decode.decoder` selects the decoder (`ffmpeg` or `vtm`). An own decoder
executable that writes planar YUV is added by subclassing `YuvPipeDecoder` in `vcmtest/decoders.py` (example:
`VTMDecoder`, about 10 lines) and adding it to `DECODERS`; see the module docstring and README (Decoding).
