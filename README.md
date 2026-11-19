# uvgVCMtester — VCM evaluation testbench

Intended for Video Coding for Machines (VCM) experiments. Measures how well machine-vision networks perform on **compressed video**: decodes AVC/HEVC/VVC/other bitstreams at several QPs (or takes already reconstructed PNG frames), runs task networks (object detection, instance segmentation, multi-object tracking and more) on
every frame, scores them against annotations (COCO AP, HOTA/MOTA, ...), plots **rate–accuracy curves** (accuracy vs bitrate), **calculates BD-rates**, **analyzes results** (flags non-monotonic behavior, results consistency etc.).

Developed and tested with the comprehensive [UVG-VCM benchmarking dataset](https://github.com/ultravideo/UVG-VCM).

```
                                                         ┌─▶ RD curves
  VIDEO ──(decode)──▶ Task predictions ──▶ Evaluation ──┼─▶ BD-rate
                                                         └─▶ Result analysis
```

## Contents

- [Integrated models](#integrated-models)
- [Integrated metrics](#integrated-metrics)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data layout](#data-layout)
- [Configuration](#configuration)
- [Running](#running)
- [Outputs](#outputs)
- [BD-rate against an anchor](#bd-rate-against-an-anchor)
- [Repository layout](#repository-layout)

## Integrated models

The testbench does not re-implement the networks: each model family has a wrapper that runs the original library
with its published pretrained weights (mostly COCO), so the results are those of the published models. Selected in
the config by `wrapper` and `model` per task.

| Task | Models (library) |
|---|---|
| Object detection | Faster R-CNN, RetinaNet, FCOS (torchvision); YOLO26 (Ultralytics); RT-DETR / RT-DETRv2 (Hugging Face); RF-DETR (`rfdetr` package or Roboflow `inference`); Faster R-CNN X101 etc. (detectron2) |
| Instance segmentation | Mask R-CNN (torchvision, detectron2); YOLO26-seg (Ultralytics); RF-DETR-Seg (`rfdetr`, Roboflow `inference`) |
| Multi-object tracking | any detector above + ByteTrack, OC-SORT, SORT, BoT-SORT or C-BIoU (Roboflow `trackers`); YOLO26 + Ultralytics trackers |
| Panoptic and semantic segmentation | Mask2Former, OneFormer, EoMT (Hugging Face) |
| Keypoints (human pose) | Keypoint R-CNN (torchvision); YOLO26-pose (Ultralytics); RF-DETR keypoint preview (`rfdetr`); ViTPose / ViTPose++ (Hugging Face, top-down on a configured detector's boxes) |
| Depth | Depth Anything V2 metric and relative, ZoeDepth, GLPN, DPT (Hugging Face) |
| Oriented object detection (DOTA v1.0) | YOLO26-obb (Ultralytics); O2-RTDETR, Rotated RTMDet, Oriented R-CNN (ai4rs / MMRotate) |
| Licence plate recognition | fast-alpr (YOLOv9 plate detector + fast-plate-ocr); two-stage: YOLOv11 plate detector + fast-plate-ocr, RapidOCR or EasyOCR reader |
| Race bib recognition | RapidOCR (PP-OCRv4 / v5 / v6), EasyOCR |

More:
- **[docs/MODELS.md](docs/MODELS.md)**: every model name, install command (optional extras), default threshold, licence,
  notes per wrapper and known issues.
- **[docs/ADDING_MODELS.md](docs/ADDING_MODELS.md)**: adding a model or a new model family (wrapper).
- **[docs/ADDING_METRICS.md](docs/ADDING_METRICS.md)**: adding a metric.

## Integrated metrics

One primary metric per task is plotted and used for BD-rate; the variants in brackets are config options (global
or per sequence; all variants are always stored). Definitions, ground-truth handling and the BD method:
**[docs/METRICS.md](docs/METRICS.md)**.

- **Object detection:** COCO bbox AP@[.50:.95] (`ap_average`: class mean | object-weighted).
- **Instance segmentation:** COCO mask AP@[.50:.95] (`ap_average` as detection).
- **Tracking:** HOTA; MOTA, IDF1 stored (`hota_average`: class mean | detection-weighted).
- **Keypoints:** COCO keypoint AP@[.50:.95] (OKS).
- **Panoptic segmentation:** PQ; SQ, RQ stored (`pq_average`: all classes as panopticapi | classes with ground truth only).
- **Semantic segmentation:** mIoU, from the panoptic predictions.
- **Depth:** AbsRel, lower is better; RMSE, δ1 etc. stored (`depth_eval`: raw | per-frame aligned; `depth_min_mm`).
- **Oriented object detection:** rotated-box AP@[.50:.95]; DOTA mAP@0.5 stored (`obb_score`: DOTA classes | vehicles
  merged; `obb_classes`).
- **Licence plate recognition:** plate accuracy = plates found and read exactly / all plates (`lpr_iou`).
- **Race bib recognition:** bib accuracy, as plates with digits only (`lpr_iou`).
- **All box and mask tasks:** `classes` limits scoring to the listed classes.
- **BD-rate / BD-accuracy** of a test encoding against an anchor: see [BD-rate against an anchor](#bd-rate-against-an-anchor).

## Requirements

- Linux, Python 3.12, [uv](https://docs.astral.sh/uv/).
- NVIDIA GPU with a CUDA 12.8 capable driver (torch 2.10 wheels). CPU works but is slow (`device: cpu`).
- **ffmpeg ≥ 7.1** with the native `vvc` decoder (and `hevc`) — on `PATH`, or a static build placed in `tools/`
  and selected with `ffmpeg: <name>` in the config. ffmpeg / ffprobe are needed with every decoder: the stream
  properties and the YUV → RGB conversion always come from them.
- Optional: the **VTM reference decoder** (`DecoderApp`, e.g. `DecoderAppStatic` placed in `tools/`) to decode with
  `decode.decoder: vtm` instead of ffmpeg's own VVC decoder.
- Disk space for decoded PNG frames (about 1–1.5 MB per 1080p frame, 8-bit; the run checks free space first).

## Installation

```bash
uv sync                          # core: decoding, scoring, plots, torchvision models
uv sync --extra ultralytics      # + YOLO26 (AGPL-3.0)
uv sync --extra rtdetr           # + RT-DETR (Hugging Face transformers)
uv sync --extra rfdetr           # + RF-DETR reference package
uv sync --extra trackers         # + ByteTrack / OC-SORT / ... for any detector (tracking)
uv sync --extra panoptic         # + Mask2Former / OneFormer / EoMT (panoptic + semantic segmentation)
uv sync --extra vitpose          # + ViTPose / ViTPose++ (top-down keypoints)
uv sync --extra depth            # + Depth Anything V2 metric depth
uv sync --extra lpr              # + fast-alpr licence plate detection + OCR (ONNX Runtime GPU)
uv sync --extra lpr2             # + two-stage plate reader: YOLO plate detector + fast-plate-ocr / PP-OCR / EasyOCR
uv sync --extra ocr              # + RapidOCR (PP-OCR) and EasyOCR scene-text readers (race bibs)
uv sync --all-extras             # everything above
```

Only the model families used in your config need to be installed; at start the pipeline lists any missing
package together with the command that installs it. Note that `uv sync` removes extras that are not requested,
so pass the same `--extra` options each time (or use `--all-extras`).

Optional, Roboflow `inference` (wrapper `roboflow`) in its own environment, because it pins older
transformers/numpy/pillow/opencv:

```bash
UV_PROJECT_ENVIRONMENT=.venv-roboflow uv sync --project envs/roboflow
```

Optional, MMRotate / ai4rs oriented detectors (wrapper `mmrotate`: O2-RTDETR, Rotated RTMDet, Oriented R-CNN) in their own environment, because
mmcv needs compiled CUDA ops (prebuilt only for torch 2.1 + CUDA 12.1); the script also clones ai4rs at the pinned
commit into `envs/mmrotate/ai4rs/`:

```bash
envs/mmrotate/setup.sh
```

Optional, detectron2 (wrapper `detectron2`), built from source into the main venv because it has no wheels for current
torch (pinned commit; the C++ ops build without a CUDA compiler; a few minutes once, the wheel is kept in
`envs/detectron2/dist/`):

```bash
envs/detectron2/setup.sh
```

A plain `uv sync` removes detectron2 again (it is not in the lock file): rerun the script (seconds, cached wheel) or
sync with `uv sync --inexact ...`, which keeps it.

Model weights are downloaded on first use into `weights/` (git-ignored).

## Data layout

```
data/<sequence_folder>/
    QP22/  <bitstream>.bin [+ encoder log <bitstream>.bin.log]  and/or  <name>_0001.png ...
    QP27/  ...
    info.json            optional: {"framerate": 60}
<annotations_root>/<sequence_name>/          (annotations_root set in the config)
    <sequence_name>_annotations.json          detection / instance segmentation / tracking (boxes, iscrowd, track_id, mask_color, polygon); no track_id at all -> tracking skipped
    <sequence_name>_masks/                     colour-coded instance masks per frame, linked by "mask_color"
    <sequence_name>_pose_annotations.json      keypoints (17 COCO keypoints + normalised area per person)
    <sequence_name>_panoptic_annotations.json  panoptic and semantic segmentation
    <sequence_name>_panoptic_masks/            colour-coded panoptic segments per frame, linked by "mask_color"
    <sequence_name>_depth/                     depth maps: depth_0001.png ..., 16-bit grayscale, millimetres
    <sequence_name>_obb_annotations.json       oriented object detection: 4 corner points per object, DOTA v1.0 classes
    <sequence_name>_lpr_annotations.json       licence plates: box + visible characters per plate
    <sequence_name>_bib_annotations.json       race bibs: box + visible digits per bib
```

- `sequence_name` is the sequence folder name up to the first `_`
  (`VolleyballGame_1920x1080_444_LD` → `VolleyballGame`). Bitstream names should start with it (warning only).
- Mask PNG colours are RGB (RGBA allowed, alpha ignored) and must equal the JSON `mask_color` [R, G, B].
- A missing `iscrowd` means 0.
- Frame numbers start at 1 in PNG names and annotation keys (encoder POC 0 = frame 1).
- If a QP folder already contains PNG frames and no decode marker, they are used as is and never deleted.
- Formats: [docs/annotation_format.txt](docs/annotation_format.txt), [docs/folder_layout.txt](docs/folder_layout.txt).

From a QP folder to an RD point:

```
QPxx/ bitstream ──decode (ffmpeg | VTM)──┐
QPxx/ PNG frames (given, used as is) ────┴─▶ RGB PNGs ──task network──▶ predictions ──metric──▶ accuracy ─┐
QPxx/ bitstream: size × fps / frames ──▶ bitrate (kbps) ──────────────────────────────────────────────────┼─▶ RD curve + JSON
QPxx/ bitrate.json (PNG-only) ──▶ bitrate (as given) ─────────────────────────────────────────────────────┘
```

The bitrate of a bitstream is computed from its file size; a folder with PNG frames only needs a `bitrate.json`
(used as given; without it the point is not plotted).

Decoding is verified bit-exact against the encoder log MD5s when available; YUV→RGB conversion
(matrix, range, bit depth) is set in the config.

**Decoder** (`decode.decoder`): `ffmpeg` (default, its native vvc / hevc decoders) or `vtm` (the VVC reference
decoder; it writes raw YUV into a named pipe that ffmpeg converts, so no YUV file is stored). On the test
sequences both give bit-identical frames. A decoder other than ffmpeg is part of the decode settings, so
switching decoders re-decodes the frames and recomputes the predictions. The bitstream must be a standard stream
that ffprobe recognises. To add your own decoder (any executable that writes planar YUV), subclass
`YuvPipeDecoder` in `vcmtest/decoders.py` (`VTMDecoder` is the example: `name`, `codecs`, `default_executable`,
`command()`) and add it to `DECODERS`.

## Configuration

One YAML file, by default `configs/config.yaml`, selects the sequences, QPs, tasks and models. All options,
defaults and per-sequence overrides: **[docs/CONFIGURATION.md](docs/CONFIGURATION.md)**.

Quick start:

```bash
cp configs/example_config.yaml configs/config.yaml    # documented example; edit it
uv run python -m vcmtest --plan                       # check: sequences, QPs, tasks, model ids, missing extras
```

A minimal config (everything left out takes its default):

```yaml
ffmpeg: ffmpeg            # executable in tools/; null = ffmpeg from PATH (>= 7.1 for VVC)
sequences: all            # or a list of folders in data/
qps: all                  # or e.g. [22, 27, 32, 37]
tasks:
  - task: detection
    models:
      - {wrapper: torchvision, model: fasterrcnn_resnet50_fpn_v2, score_threshold: null}   # null = library default
      - {wrapper: ultralytics, model: yolo26m.pt, score_threshold: 0.05}
  - task: tracking
    models:
      - {wrapper: rfdetr, model: rfdetr-medium, tracker: bytetrack, score_threshold: 0.1}
sequence_overrides:       # optional, per sequence folder
  MySeq_1920x1080_RA:
    qps: [37, 42, 47, 50]
    classes: [car]        # score only these classes
```

Main option groups:
- **Paths:** `data_root`, `annotations_root`, `results_root`, `weights_dir`, `ffmpeg`.
- **Tasks and models:** `tasks` (a task runs on a sequence only if its annotations exist); model ids as printed by
  `--plan`. For AP use a low `score_threshold` (0.05); trackers need 0.1. Models: [docs/MODELS.md](docs/MODELS.md).
- **Metric variants** (all always stored, one plotted): `ap_average`, `hota_average`, `pq_average`, `depth_eval`,
  `depth_min_mm`, `obb_score`, `lpr_iou`. See [Integrated metrics](#integrated-metrics).
- **`input_size`:** network input size per task and model (YOLO26, RF-DETR, RT-DETR).
- **`sequence_overrides`:** per sequence `qps`, `tasks`, `models`, `classes`, `input_size`, metric variants, ...
- **`decode`**, **`inference`**, **`plot`**, **`bd`**: decoder and colour conversion, timeouts, plot axes, BD comparison.

**BD anchors change the configuration that is run**, also with `bd.enabled: false`: if a sequence name has an anchor
(`bd.anchors_file`, `bd.anchors`), every other folder of that sequence name inherits the anchor's
`sequence_overrides` (all keys except `qps`; its own keys win). `--plan` and the run log list the inherited keys;
details in [BD-rate against an anchor](#bd-rate-against-an-anchor).

## Running

```bash
uv run python vcmtest                         # everything in configs/config.yaml
uv run python vcmtest my_config.yaml          # another config (also: python -m vcmtest ...)
uv run python vcmtest --seq <sequence_folder> --qp 22 47
uv run python vcmtest --plan                  # only discover inputs and print the work plan
uv run python vcmtest --remove-frames         # delete decoded PNGs after evaluation
uv run python vcmtest --remove-depth-maps     # delete predicted depth maps after scoring (inference.remove_depth_maps)
uv run python vcmtest --bd                    # only the BD comparisons of the configured anchor/test pairs
uv run python vcmtest --bd --anchor <folder> --test <folder>   # BD comparison of one pair
```

Run logs go to `results/logs/run_<timestamp>.log`. A frame-alignment sanity check (use a high-quality QP):
`uv run python tools/check_alignment.py <sequence_folder> <qp> <model_id>`.

## Outputs

| File | Content |
|---|---|
| `data/<seq>/results/<seq>_results.json` | per QP bitrate and decode record; per task, model and QP the accuracy (all stats of the metric, per class), thresholds, weights hash, versions, sanity check |
| `data/<seq>/results/<seq>_<task>_rd.png` | RD curve per task, one line per model; `rd_per_model/` one plot per model |
| `data/<seq>/results/predictions/`, `scores/` | cached predictions and scores per model and QP |
| `results/summary.json` | all sequences collected into one file |
| `BD-results/<test>__vs__<anchor>/` | BD comparison against the anchor: `bd.json`, RD plots, Markdown tables, extended analysis |
| `BD-results/summary.json`, `overview.md` | all BD comparisons and the cross-sequence overview |
| `results/logs/` | run logs |

The results file holds every valid data point on disk (all QPs run so far); points that no longer match the config
or annotations are listed under `not_plotted`. Bitrate = bitstream size × 8 × fps / frames (kbps); a QP folder with
PNGs only needs `QPxx/bitrate.json` (`{"bitrate": 0.35, "unit": "bpp"}`, used as given). Removing a model from the
config keeps its files on disk, and putting it back reuses its cached predictions and scores.

More: [docs/OUTPUTS.md](docs/OUTPUTS.md) (file contents, caching rules, bitrate details, BD and extended analysis
files).

## BD-rate against an anchor

Encodings of the same source (data folders with the same sequence name, e.g. `CampusView_..._VTM_RA` and
`CampusView_..._kvazaar_medium`) can be compared per task and model. An anchors file names one anchor folder per
sequence name; every other folder of that name is a test:

```yaml
# configs/anchors.yaml (bd.anchors_file; bd.anchors in config.yaml overrides it per name)
anchors:
  CampusView: CampusView_1920x1080_444_10b_VTM_RA
```

- **BD-rate (%)**: bitrate change of the test at equal accuracy (negative = better);
  **BD-accuracy (pp)**: accuracy change at equal bitrate.
- Curves reduced to their Pareto front, PCHIP over log10 bitrate, integrated over the overlap; n/a below
  `bd.min_points` (4) points. A task or model is skipped when the two sides are not measured the same way (metric
  options, classes, annotations, bitrate unit, weights, frame count); a different frame size is allowed and noted.
- **Tests inherit the anchor's `sequence_overrides`** (all except `qps`; their own keys win), also with
  `bd.enabled: false`: shown by `--plan`, recorded in the results.
- Runs at the end of a run with `bd.enabled: true`, or alone from the results on disk: `--bd` (configured pairs) or
  `--bd --anchor A --test T` (one pair).
- Output in `BD-results/`: per pair `bd.json`, RD plots, tables and an extended analysis (sub-metric and per-class
  BD, model agreement); `summary.json` and `overview.md` across sequences.

More: [docs/BD_RATE.md](docs/BD_RATE.md) (anchors, inheritance, checks, method, extended analysis, outputs).

## Repository layout

```
vcmtest/        pipeline: config, discovery, decode, inference, scoring, bitrate, results, plots, BD (bd.py)
task_models/    model wrappers (one file per library) and labels.py (generic COCO ids)
configs/        example_config.yaml (copy to config.yaml = the default config); BD anchors file (optional)
envs/roboflow/  separate locked environment for Roboflow inference
envs/mmrotate/  separate locked environment (+ setup.sh, ai4rs checkout) for MMRotate oriented detectors
tools/          check_alignment.py (+ local ffmpeg / VTM decoder binaries, git-ignored)
docs/           documentation (index: docs/README.md): configuration, data formats, outputs, metrics, models, adding models / metrics
results/        summary.json
BD-results/     BD comparisons (generated)
```

## Licence

The Clear BSD License (BSD-3-Clause-Clear), see [LICENCE](LICENCE). It grants no patent rights.

The optional model libraries keep their own licences (listed in [docs/MODELS.md](docs/MODELS.md)); in particular the
Ultralytics models (extra `ultralytics`) are AGPL-3.0, and some model weights are non-commercial. Installing an
extra is your choice and subject to its licence.
