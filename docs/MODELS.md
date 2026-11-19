# Supported task models

Every model is selected in `configs/config.yaml` as `{wrapper: <wrapper>, model: <name>, score_threshold: <t|null>}`
under a task (`detection`, `instance_segmentation`, `tracking`, `keypoints`, `panoptic_segmentation`, `semantic_segmentation`, `depth`). `score_threshold: null` uses the library default listed
below; a number overrides it (and is recorded in the results). Only the libraries of the wrappers that appear in
the config need to be installed: the pipeline checks this at start and prints the install command.
To add a model or a new model family (wrapper), see [ADDING_MODELS.md](ADDING_MODELS.md).

All wrappers output the same format per frame, one list per detected instance, with generic COCO class ids
1–80 (`task_models/labels.py`) and coordinates normalised to [0, 1]:

| Task | Per instance |
|---|---|
| detection | `[class_id, score, x1, x2, y1, y2]` |
| tracking | `[class_id, score, x1, x2, y1, y2, track_id]` |
| instance segmentation | `[class_id, score, x1, x2, y1, y2, W, H, rle]` — `rle` = pycocotools compressed RLE of the binary mask at W×H (the decoded frame size) |
| depth | `[W, H, png, sha256]` per frame: `png` = 16-bit depth map in millimetres at the frame size, stored next to the predictions file (`QPxx_depth/depth_NNNN.png`) |
| plate recognition | `[score, x1, x2, y1, y2, text, ocr_confidence]` — plate box normalised, read characters, mean per-character OCR confidence |
| oriented detection | `[class_id, score, x1, y1, x2, y2, x3, y3, x4, y4]` — DOTA v1.0 0-based class id (`task_models/labels_dota.py`), 4 corners normalised |
| panoptic / semantic segmentation | `[class_id, score, W, H, rle]` per segment (non-overlapping); `class_id` = COCO **panoptic** category id (`task_models/labels_panoptic.py`: things 1–90 with gaps, stuff 92–200) |

## Wrappers

| Wrapper | Models (`model:`) | Tasks | Install | Library default score threshold | Code licence |
|---|---|---|---|---|---|
| `torchvision` | `fasterrcnn_resnet50_fpn`, `fasterrcnn_resnet50_fpn_v2`, `retinanet_resnet50_fpn_v2`, `fcos_resnet50_fpn` | detection | core | 0.05 (FCOS 0.2) | BSD-3-Clause |
| `torchvision` | `maskrcnn_resnet50_fpn`, `maskrcnn_resnet50_fpn_v2` | instance segmentation | core | 0.05 | BSD-3-Clause |
| `ultralytics` | `yolo26{n,s,m,l,x}.pt` | detection | `uv sync --extra ultralytics` | 0.25 | **AGPL-3.0** |
| `ultralytics` | `yolo26{n,s,m,l,x}-seg.pt` | instance segmentation | `uv sync --extra ultralytics` | 0.25 | **AGPL-3.0** |
| `ultralytics` + `tracker:` | `yolo26{n,s,m,l,x}.pt` with `botsort.yaml` (default), `bytetrack.yaml`, `ocsort.yaml`, `deepocsort.yaml`, `fasttrack.yaml`, `tracktrack.yaml` or a path to an own tracker YAML | tracking | `uv sync --extra ultralytics` | 0.1 (track mode) | **AGPL-3.0** |
| any detection wrapper + `tracker:` | `bytetrack`, `ocsort`, `sort`, `botsort`, `cbiou` (Roboflow `trackers`) on the boxes of any detection model above, e.g. `{wrapper: rfdetr, model: rfdetr-medium, tracker: bytetrack}` | tracking | `uv sync --extra trackers` | detector's own default (0.1 recommended) | Apache-2.0 |
| `rtdetr` | Hugging Face ids, e.g. `PekingU/rtdetr_v2_r18vd` | detection | `uv sync --extra rtdetr` | 0.5 | Apache-2.0 |
| `rfdetr` | `rfdetr-nano`, `-small`, `-medium`, `-large` (`rfdetr-base`, deprecated) | detection | `uv sync --extra rfdetr` | 0.5 | Apache-2.0 |
| `rfdetr` | `rfdetr-seg-nano`, `-small`, `-medium`, `-large`, `-xlarge`, `-2xlarge` | instance segmentation | `uv sync --extra rfdetr` | 0.5 | Apache-2.0 |
| `roboflow` | `rfdetr-nano`, `-small`, `-medium`, `-base`, `-large` | detection | separate venv, see below | 0.4 | Apache-2.0 |
| `roboflow` | `rfdetr-seg-nano` … `rfdetr-seg-2xlarge` | instance segmentation | separate venv, see below | 0.4 | Apache-2.0 |
| `hf_panoptic` | Hugging Face COCO panoptic checkpoints: `facebook/mask2former-swin-{tiny,small,base,large}-coco-panoptic`, `shi-labs/oneformer_coco_swin_large`, `tue-mps/coco_panoptic_eomt_{small,base}_640_2x`, `tue-mps/coco_panoptic_eomt_large_{640,1280}` | panoptic + semantic segmentation | `uv sync --extra panoptic` | 0.8 (post-processing) | MIT |
| `torchvision` | `keypointrcnn_resnet50_fpn` (COCO_V1) | keypoints | core | 0.05 | BSD-3-Clause |
| `rfdetr` | `rfdetr-keypoint-preview` (`RFDETRKeypointPreview`, weights `rf-detr-keypoint-preview-xlarge.pth`; open, not RF-DETR+) | keypoints | `uv sync --extra rfdetr` | 0.5 | Apache-2.0 |
| `ultralytics` | `yolo26{n,s,m,l,x}-obb.pt` (DOTA v1.0, 15 classes) | oriented detection | `uv sync --extra ultralytics` | 0.25 | **AGPL-3.0** |
| `mmrotate` (+ `input_size:`) | ai4rs config names: `o2_rtdetr_r{18,34,50}vd_2xb4_72e_dota` (O2-RTDETR), `rotated_rtmdet_{tiny,s,m,l}-3x-dota` (Rotated RTMDet), `oriented-rcnn-le90_r50_fpn_1x_dota` (Oriented R-CNN); DOTA v1.0, 15 classes | oriented detection | separate venv: `envs/mmrotate/setup.sh` | O2-RTDETR none (top 300); RTMDet-R / Oriented R-CNN 0.05 | Apache-2.0 (ai4rs / MMRotate code; RTMDet-R and Oriented R-CNN weights from OpenMMLab); O2-RTDETR weight licence not stated |
| `ultralytics` | `yolo26{n,s,m,l,x}-pose.pt` | keypoints | `uv sync --extra ultralytics` | 0.25 | **AGPL-3.0** |
| `hf_vitpose` + `detector:` | `usyd-community/vitpose-base-simple`, `usyd-community/vitpose-plus-{small,base,large,huge}`, top-down on the person boxes of `detector` = a detection-task model id, e.g. `{wrapper: hf_vitpose, model: usyd-community/vitpose-plus-large, detector: rfdetr-rfdetr-medium-thr0.05}` | keypoints | `uv sync --extra vitpose` | 0.0 (min. person-box score) | Apache-2.0 |
| `hf_depth` | metric: `depth-anything/Depth-Anything-V2-Metric-{Indoor,Outdoor}-{Small,Base,Large}-hf`, `Intel/zoedepth-nyu-kitti` (also `-nyu`, `-kitti`), GLPN; relative (inverse depth): `depth-anything/Depth-Anything-V2-{Small,Base,Large}-hf`, `Intel/dpt-large`, `Intel/dpt-hybrid-midas`. Other families are rejected | depth | `uv sync --extra depth` | none (null) | DA-V2 Small: Apache-2.0, Base / Large: CC-BY-NC-4.0; ZoeDepth: MIT; DPT: Apache-2.0 (weights). Depth Pro not used: Apple research-only licence |
| `fast_alpr` + `ocr:` | detector (`model:`) `yolo-v9-{t-640,t-512,t-416,t-384,t-256,s-608}-license-plate-end2end` (open-image-models) with OCR (`ocr:`) e.g. `cct-s-v2-global-model`, `cct-xs-v2-global-model` (fast-plate-ocr) | plate recognition | `uv sync --extra lpr` | 0.4 | code MIT (fast-alpr, open-image-models, fast-plate-ocr); weight licence not stated |
| `plate_two_stage` + `ocr:` | detector (`model:`) an Ultralytics plate checkpoint `<hf user>/<repo>/<file>.pt` (pinned revision) or a local `.pt`, e.g. `morsetechlab/yolov11-license-plate-detection/license-plate-finetune-v1{n,s,m,l,x}.pt` (YOLOv11); reader (`ocr:`) `fast_plate_ocr:<model>` (e.g. `cct-s-v2-global-model`), `rapidocr:<PP-OCR version>-<type>` (e.g. `PP-OCRv5-server`) or `easyocr:<langs>` (e.g. `en`) | plate recognition | `uv sync --extra lpr2` | 0.25 (Ultralytics conf) | detector weights **AGPL-3.0** (morsetechlab model card; Ultralytics); readers: MIT (fast-plate-ocr), Apache-2.0 (RapidOCR / PaddleOCR, EasyOCR) |
| `rapidocr` | `<PP-OCR version>-<type>`: `PP-OCRv5-server`, `PP-OCRv5-mobile`, `PP-OCRv6-{tiny,small,medium}`, `PP-OCRv4-{mobile,server}` (RapidOCR ONNX of PaddleOCR PP-OCR) | bib recognition (scene text) | `uv sync --extra ocr` | 0.5 (text_score) | Apache-2.0 (RapidOCR, PaddleOCR models) |
| `easyocr` | language code(s), `+`-separated, e.g. `en` (CRAFT detector + english_g2 CRNN) | bib recognition (scene text) | `uv sync --extra ocr` | none | Apache-2.0 |
| `detectron2` | e.g. `faster_rcnn_X_101_32x8d_FPN_3x.yaml`, `mask_rcnn_X_101_32x8d_FPN_3x.yaml` | detection; instance segmentation (`mask_rcnn_*`) | `envs/detectron2/setup.sh` (built from source, below) | 0.05 | Apache-2.0 |

Licences of the pretrained **weights** can differ from the code licence; check the model cards before
publishing results or redistributing weights.

## Notes per wrapper

- **torchvision** — model names are matched exactly (`fasterrcnn_resnet50_fpn` is v1, `..._v2` is v2). Masks are
  binarised at 0.5; at most 100 detections per frame.
- **fast_alpr** — ONNX Runtime GPU. `onnxruntime-gpu` is pinned below 1.27 (the last CUDA 12 build; 1.27+ needs
  CUDA 13 libraries that are not installed); `onnxruntime.preload_dlls()` loads torch's CUDA 12 / cuDNN 9 libraries.
  The detector letterboxes the whole frame to its fixed size (e.g. 640 px), so plates below about 40 px wide in a
  1920-px frame are mostly missed. Weights are redirected from `~/.cache` to `weights/fast_alpr/`.
- **plate_two_stage** — two stages: the Ultralytics plate detector on the whole frame (letterboxed to the checkpoint's
  training size, 640, or `input_size`), then the reader on each box crop (rounded outwards to whole pixels, clipped to
  the frame, as fast-alpr crops). Readers: fast-plate-ocr as in fast-alpr (colour mode from the model config, mean
  per-character probability as confidence); RapidOCR recogniser only (`use_det` / `use_cls` off: the crop is the text
  line); EasyOCR CRAFT detection + recognition inside the crop, pieces joined left to right (recognising the whole
  crop as one box read 0 of 4 large CarPark plates). No allowlist or plate-format rules. The detector repo revision is
  pinned in the wrapper (`REVISIONS`); weights in `weights/plate_two_stage/<repo>/<revision>/`, the readers' weights
  in their usual folders (`weights/fast_alpr/fast-plate-ocr`, `weights/rapidocr`, `weights/easyocr`). The
  morsetechlab model card notes train/test overlap in its Roboflow dataset (its mAP figures are optimistic). EULPR
  (0xnu/european-license-plate-recognition) was checked on 2026-11-17 and not added: its YOLOv12n detector returns
  one box over the whole image on every frame (its training set has no plate boxes).
  Configured since 2026-11-18: v1s + `fast_plate_ocr:cct-s-v2-global-model` only. v1x + cct-s (found fewer small
  plates than v1s), v1s + `rapidocr:PP-OCRv5-server` (I / 1 confusion) and v1s + `easyocr:en` (3–18 %) were run on
  2026-11-17 and dropped; the readers stay supported by the wrapper.
- **ultralytics** — inference at `imgsz=640`; OBB models (`*-obb.pt`) at their training size from the checkpoint
  (`imgsz=1024`, DOTA v1.0); both replaced by the global `input_size` and per sequence by `sequence_overrides.<seq>.input_size` (model id +
  `-<N>px`; also implemented for rfdetr and rtdetr; others with an override are not run); class ids are kept 0-based (DOTA order, checked against `labels_dota.py`). On
  SyntheticAerial, inference at the native 1920 was much worse than at 1024 (yolo26n AP 22 vs 40, 20 frames). Segmentation models run with `retina_masks=True`, so masks are at
  frame resolution (as in Ultralytics' own COCO-JSON evaluation); without it they would be at the letterboxed
  inference size. Weights are downloaded to `weights/ultralytics/`.
- **ultralytics tracking** — `model.track(..., persist=True)` frame by frame in order; a fresh tracker for each
  QP (an interrupted tracking run restarts from the first frame). The tracker YAML contents are recorded with
  the predictions and are part of their cache key. Only boxes with a track id are output.
- **Roboflow trackers** — clean-room implementations (PyPI `trackers`, Apache-2.0) run on each frame's detections of
  the configured detector. Library defaults, except `frame_rate` = the sequence frame rate (the lost-track buffer is
  defined in 30 fps frames and rescaled). Only detections assigned to a confirmed track are output. Feed the
  trackers low-confidence detections (`score_threshold: 0.1`): ByteTrack uses them in its second association.
  OC-SORT's library default ignores detections below 0.6.
- **rtdetr** — Hugging Face Transformers; weights in `weights/huggingface/`. The wrapper maps label id + 1 to the
  generic id; the label order is checked against `labels.py` at load time.
  Input: the image processor's 640 × 640 stretch (no aspect ratio kept, as in RT-DETR training); a per-sequence
  `input_size` N stretches to N × N. Measured 2026-11-13 (R18, every 10th frame): 1280 × 736 with the aspect kept
  was never the best size (WinterDrive QP29 37.7 AP vs 51.6 at 640 × 640 and 55.2 at 800 × 800; CampusView 25.8 vs
  30.0); larger inputs cut its large-object AP (WinterDrive APl 74 → 52). Only very small objects gain (SyntheticCity
  960 × 960 29.0 vs 21.4 at 640; SyntheticDrone 19.9 vs 13.0). Larger backbones at 640: WinterDrive R50 58.7, R101
  61.0 (R18 51.6).
- **rfdetr** — Roboflow's reference implementation (PyPI `rfdetr`); weights in `weights/rfdetr/`. It returns COCO
  91-category ids; the wrapper maps the returned class *name* to `labels.py`.
- **roboflow** — Roboflow `inference` (PyPI). It pins older transformers / numpy / pillow / opencv than the main
  environment, so it has its own locked environment and runs as a worker subprocess:
  `UV_PROJECT_ENVIRONMENT=.venv-roboflow uv sync --project envs/roboflow` (path set by `roboflow_python` in the
  config). Uses the torch backend on the GPU; public COCO models need no API key. Boxes are rounded to whole
  pixels by the library.
- **hf_panoptic** — `AutoModelForUniversalSegmentation` + the model's image processor, then
  `post_process_panoptic_segmentation` at the frame size with the transformers defaults (threshold 0.8, mask
  threshold 0.5, overlap 0.8), except that stuff segments of one class are merged (`label_ids_to_fuse` / EoMT
  `stuff_classes` = all stuff classes), as in the models' own COCO evaluation. Class names are mapped to the COCO
  panoptic ids (an unknown name is an error). One prediction serves both tasks: list the same model under
  `panoptic_segmentation` and `semantic_segmentation`, it is run once. Input size: OneFormer shorter side 800 /
  longer ≤ 1333 and EoMT 640 or 1280 (processor defaults, as in their papers); **Mask2Former** is run at shorter
  side 800 / longer ≤ 1333 as in the Mask2Former paper, because the Hugging Face processors of these checkpoints
  resize to a fixed 384×384 (aspect ratio lost; PQ 0.19 vs 0.35 for Swin-L on HighwayView). The input size is
  recorded with the predictions. transformers reports `swin.layernorm` as MISSING for the Swin checkpoints: it only
  feeds the unused `last_hidden_state` (changing it does not change the outputs). OneFormer DiNAT needs `natten`
  (not installed). EoMT DINOv3 checkpoints use the DINOv3 licence. Weights in `weights/huggingface/`.
- **Keypoints** — output per person `[1, score, x1, x2, y1, y2, [x, y, s] * 17]` (17 COCO keypoints, x, y
  normalised, s = keypoint score, unused by the metric). Keypoint R-CNN: score = box score, s = torchvision keypoint
  logit. YOLO26-pose: score = box confidence, s = keypoint confidence; imgsz 640. RF-DETR keypoint preview:
  `predict()` returns supervision `KeyPoints`; score = detection confidence, s = keypoint confidence, box =
  `data["xyxy"]`; configured at 0.05 like the other RF-DETR models (library default 0.5). The checkpoint's
  keypoint schema is `[0, 17]` (background-first slot, then person).
- **hf_vitpose** — top-down ViTPose / ViTPose++ (`VitPoseForPoseEstimation` + its image processor: box crop,
  256×192, UDP heatmap decoding). The person boxes are the cached predictions of `detector` (a model of the
  detection task, listed before the keypoints task) on the same decoded frame, so compression affects both
  stages; the detector's predictions hash is part of the cache key. `score_threshold` = minimum person-box score
  (null = 0.0, all boxes, as in the ViTPose COCO evaluation with detector results). ViTPose++ uses its COCO head
  (dataset index 0). Instance score = box score × mean keypoint score above 0.2, then OKS-NMS at 0.9 (mmpose's
  CocoMetric). Weights in `weights/huggingface/`.
- **hf_depth** — `AutoModelForDepthEstimation` + its image processor, then `post_process_depth_estimation` at the
  frame size (ZoeDepth: with the source size, padding removed). The output kind comes from the model: Depth
  Anything V2 by its config `depth_estimation_type` (metric = metres: Indoor trained on Hypersim, `max_depth`
  20 m; Outdoor on Virtual KITTI 2, 80 m; relative = affine-invariant inverse depth); ZoeDepth and GLPN metric;
  DPT relative inverse depth. Depth Anything V2: shorter side 518, aspect ratio kept, multiple of 14. DPT runs at
  the Hugging Face default 384×384 (the frame is squashed): the transformers DPT models fail on non-square input
  because they do not interpolate their position embeddings (the original MiDaS / DPT code keeps the aspect
  ratio). `Intel/dpt-large` reports `neck.fusion_stage.layers.0.residual_layer1` as MISSING: the first fusion
  layer has no residual input, so that layer is never used. Metric maps are stored in millimetres, relative maps
  normalised per frame (16-bit PNGs). Weights in `weights/huggingface/`. Mind the non-commercial licence of the
  Depth Anything V2 Base / Large weights.
- **rapidocr** — RapidOCR 3.x on ONNX Runtime GPU (the same CUDA 12 build and `preload_dlls()` as fast_alpr). The
  model name picks the detector and recognizer version and size (`PP-OCRv5-server`); orientation classifier = the
  library default. Library defaults otherwise (detector input: shorter side ≥ 736, so 1080p frames run as is).
  Models load lazily on the first text found, so the wrapper runs one warm-up image with text. score = recognition
  confidence (no detection score is exposed). Weights in `weights/rapidocr/`; hash over the 3 ONNX files used.
  On KartingTime, the PP-OCRv5-mobile and PP-OCRv6-small detectors merge a bib with neighbouring sponsor text
  into one wide line; PP-OCRv6-small also detects no text on many bibs of the sharp frames.
- **easyocr** — EasyOCR 1.7 (PyTorch). `readtext()` on a numpy array uses the array as RGB for detection but
  converts it to grey as if it were BGR; the wrapper calls `detect` (RGB) and `recognize` (cv2 RGB2GRAY) itself, as
  for a file-path input, with the `readtext` defaults. score = recognition confidence. Weights in `weights/easyocr/`.
- **mmrotate** — OpenMMLab MMRotate 1.x models from ai4rs (https://github.com/wokaikaixinxin/ai4rs, pinned commit in
  `envs/mmrotate/setup.sh`), run as a worker subprocess (`task_models/task_models_mmrotate.py --worker`, same
  protocol as `roboflow`) in `.venv-mmrotate`: Python 3.11, torch 2.1.2 + CUDA 12.1, mmcv 2.1.0 (the prebuilt
  OpenMMLab wheel with CUDA ops; no CUDA compiler in the container), mmdet 3.3.0, mmengine 0.10.7, mmsegmentation
  1.2.2 (imported by ai4rs). The ai4rs checkout (`envs/mmrotate/ai4rs/`, git-ignored) is put on `sys.path`, not
  installed, because the models live in its `projects/` folder. `input_size: tiles` (default) = the DOTA single-scale
  test protocol: 1024×1024 tiles of the full-resolution frame with 200 px overlap (1920×1080 → 6 tiles, no
  padding; tile pipeline with resize factor 1), tile boxes merged per class with rotated NMS 0.1 (as MMRotate's
  DOTA merge_patches); `input_size: N` = the whole frame resized to fit N×N (the released test pipelines, N = 1024).
  The setting is part of the model id (`-tiles`, `-1024px`). Boxes (cx, cy, w, h, angle) are converted to 4
  corners; classes are mapped by name from the MMRotate DOTA order to `labels_dota.py`. Weights in
  `weights/mmrotate/` (ModelScope for O2-RTDETR, download.openmmlab.com for RTMDet-R and Oriented R-CNN; the Oriented
  R-CNN checkpoint is the MMRotate 0.x one linked by the 1.x config). Why tiles: DOTA models are trained on 1024
  crops at full resolution; on SyntheticAerial a frame resized to 1024 halves the cars and RTMDet-R / Oriented R-CNN
  found almost none (3–43 of 267 cars on 3 frames), O2-RTDETR drew boxes 37 % too wide; a whole 1920 frame padded
  with grey makes RTMDet-R's scores depend on the amount of padding (CSPNeXt global channel attention: 8 vs 45 cars
  on one frame). With tiles all models match the GT box size.
- **detectron2** — no wheels for current torch, so it is built from source: `envs/detectron2/setup.sh` clones
  detectron2 at a pinned commit (main 247691a, tested 2026-11-16 with torch 2.10 + cu128) into `envs/detectron2/src/`,
  replaces `pkg_resources` (removed in setuptools >= 81) in `model_zoo/model_zoo.py` by `importlib.resources`, builds
  a wheel with the main venv's torch (`--no-build-isolation`; kept in `envs/detectron2/dist/<commit>-torch<ver>-py<ver>/`,
  rebuilt only when torch or Python change) and installs it with `--no-deps` (its hydra-core pin would downgrade
  omegaconf) plus its runtime dependencies (fvcore, iopath, yacs, termcolor, cloudpickle, tabulate). Only the C++
  ops are compiled (no `nvcc` needed); the standard Faster / Mask R-CNN models use torchvision's CUDA ops
  (deformable conv / rotated-box ops unavailable). The build needs the Python headers: if the venv's Python has
  none (a system Python without `python3.x-dev`), the script uses the headers of a uv-managed Python of the same
  version (same ABI). A plain `uv sync` removes detectron2 (not in the lock file): rerun the script (seconds) or
  use `uv sync --inexact`. Weights: weights/detectron2 (FVCORE_CACHE).
  Masks: detectron2's own binary masks at the frame size (threshold 0.5).

## Not allowed

`rfdetr-xlarge` and `rfdetr-2xlarge` (RF-DETR **detection** XLarge/2XLarge) require a Roboflow API key /
RF-DETR+ licence. The config is rejected at load time with a clear message (both RF-DETR wrappers).
The segmentation XLarge/2XLarge models are part of the open `rfdetr` package and are allowed.

## Known issues

- RF-DETR segmentation masks are offset from the ground truth: about (+2, +1) px with the `rfdetr` package and
  about (+4, +2) px with `inference` 1.8.4 at 1920×1080 (boxes are aligned). This lowers mask AP75.
- Library default thresholds differ a lot (0.05 … 0.5). COCO AP rewards low-confidence detections, so a high
  threshold caps AP; the example config uses 0.05 for RF-DETR for that reason.
