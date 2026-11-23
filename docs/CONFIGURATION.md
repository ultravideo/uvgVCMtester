# Configuration

The run is controlled by one YAML file, by default `configs/config.yaml`. Start from the documented example:

```bash
cp configs/example_config.yaml configs/config.yaml
```

- Paths are relative to the repository root unless absolute.
- Unknown keys are an error; the config is validated before anything runs.
- Keys that are left out take their defaults (listed below).
- `python -m vcmtest <config.yaml> --plan` prints the resulting work plan (sequences, QPs, tasks, model ids, missing
  extras, inherited overrides) without running anything.

Contents: [Paths and environment](#paths-and-environment) · [Sequences and QPs](#sequences-and-qps) ·
[Tasks and models](#tasks-and-models) · [Metric options](#metric-options) · [Input size](#input-size) ·
[Per-sequence overrides](#per-sequence-overrides) · [BD anchors and inheritance](#bd-anchors-and-inheritance) ·
[Decoding](#decoding) · [Inference](#inference) · [Plots](#plots) · [BD comparison](#bd-comparison) ·
[Command-line options](#command-line-options)

## Paths and environment

| Key | Default | Meaning |
|---|---|---|
| `data_root` | `data` | one folder per sequence: `data/<sequence_folder>/QPxx/` |
| `annotations_root` | `All_annotations` | one sub-folder per sequence name (folder name up to the first `_`) |
| `results_root` | `results` | `results/summary.json`, `results/logs/` |
| `tools_dir` | `tools` | ffmpeg and decoder executables |
| `weights_dir` | `weights` | downloaded model weights (torch hub, Hugging Face, Ultralytics, Roboflow) |
| `ffmpeg` | `null` | executable name in `tools_dir`; `null` = ffmpeg from PATH (≥ 7.1 for VVC) |
| `roboflow_python` | `.venv-roboflow/bin/python` | Python of the separate venv of the `roboflow` wrapper |
| `mmrotate_python` | `.venv-mmrotate/bin/python` | Python of the separate venv of the `mmrotate` wrapper |
| `device` | `cuda:0` | inference device |
| `seed` | `0` | random seed |

### Several data roots

Each sequence's results stay in its own folder (`<data_root>/<seq>/results/`), but three things are shared by name
or by folder and do **not** follow `data_root`:

- **Anchors** (`bd.anchors_file`, `bd.anchors`) name folders, not paths: they are looked up under the current
  `data_root`. An anchor folder that is not there gives a warning and its pairs are skipped (a test in another data
  root cannot be compared with it); when most anchors are missing, one warning says the anchors seem to belong to
  another data root.
- **`results/summary.json`** is rebuilt from the current `data_root` only: switching the root replaces its
  sequences (their results files are kept). It records `data_root`, and a switch is warned.
- **`BD-results/`** keeps the pairs of every run: pairs of two data roots would be mixed in `summary.json` and
  `overview.md` (warned; `data_roots` lists them), and a pair folder of the same name is replaced (warned).

`sequence_overrides` are keyed by folder name, so a folder of the same name in another data root gets the same
overrides. To keep a second data root separate, give it its own config with its own outputs:

```yaml
# configs/config_data2.yaml
data_root: data2
results_root: results_data2
bd:
  anchors_file: configs/anchors_data2.yaml
  out_dir: BD-results_data2
```

and run `python -m vcmtest configs/config_data2.yaml`.

## Sequences and QPs

| Key | Default | Meaning |
|---|---|---|
| `sequences` | `all` | `all` or a list of folder names under `data_root` |
| `qps` | `all` | `all` or a list, e.g. `[22, 27, 32, 37]` |
| `framerate` | `null` | fallback when `data/<seq>/info.json` has no `"framerate"` |

QP selection order: `--qp` on the command line > `sequence_overrides.<seq>.qps` > `qps`.

## Tasks and models

`tasks` is a list; each entry names a task and its models. A task runs on a sequence only if that sequence has the
task's annotations; otherwise it is skipped.

```yaml
tasks:
  - task: detection
    models:
      - {wrapper: torchvision, model: fasterrcnn_resnet50_fpn_v2, score_threshold: null}
      - {wrapper: ultralytics, model: yolo26m.pt, score_threshold: 0.05}
  - task: tracking
    models:
      - {wrapper: ultralytics, model: yolo26m.pt, tracker: bytetrack.yaml, score_threshold: null}
      - {wrapper: rfdetr, model: rfdetr-medium, tracker: bytetrack, score_threshold: 0.1}
```

Tasks: `detection`, `instance_segmentation`, `tracking`, `keypoints`, `panoptic_segmentation`,
`semantic_segmentation`, `depth`, `oriented_detection`, `plate_recognition`, `bib_recognition`.

Model keys:

| Key | Meaning |
|---|---|
| `wrapper` | model family, e.g. `torchvision`, `ultralytics`, `rtdetr`, `rfdetr`, `roboflow`, `detectron2`, `hf_panoptic`, `hf_depth`, `hf_vitpose`, `mmrotate`, `fast_alpr`, `plate_two_stage`, `rapidocr`, `easyocr` ([MODELS.md](MODELS.md)) |
| `model` | model name within the wrapper |
| `score_threshold` | detections below it are dropped; `null` = the library default |
| `tracker` | tracking only: an Ultralytics tracker YAML (`botsort.yaml`, `bytetrack.yaml`, ...; ultralytics wrapper) or a Roboflow `trackers` algorithm on any detection wrapper (`bytetrack`, `ocsort`, `sort`, `botsort`, `cbiou`) |
| `detector` | `hf_vitpose` only: model id of a detection-task model whose cached person boxes are the pose input |
| `ocr` | `fast_alpr` / `plate_two_stage`: the plate reader |
| `input_size` | `mmrotate` only: `tiles` (default, DOTA protocol) or N; other wrappers use the [input_size](#input-size) option |
| `name` | optional own model id instead of the generated one |

**Score threshold.** For AP use a low threshold (0.05): library defaults of 0.25–0.5 cut recall and cap AP. Trackers
need the low-score detections (0.1). Defaults per model: [MODELS.md](MODELS.md).

**Model id.** Each model gets an id used in file names, plots and other options:
`<wrapper>-<model>[-<tracker>][-on-<detector>][-<ocr>][-<N>px][-thr<score_threshold>]`, e.g.
`ultralytics-yolo26m.pt-thr0.05`. `--plan` prints the ids. Changing anything that is part of the id creates new
predictions; the old ones stay on disk.

**Optional extras.** Most model families are optional extras (`uv sync --extra <name>`); `--plan` reports models whose
extra is not installed.

## Metric options

These choose which variant of a metric is plotted; all variants are always stored. Each can also be set per
sequence. Definitions: [METRICS.md](METRICS.md).

| Key | Default | Values |
|---|---|---|
| `ap_average` | `class` | detection / instance segmentation AP: `class` (COCO mean of per-class AP) or `object` (object-weighted, one PR curve over the classes) |
| `hota_average` | `class` | tracking: `class` (class average) or `object` (TrackEval detection-averaged) |
| `pq_average` | `all` | panoptic PQ: `all` (panopticapi, classes with TP+FP+FN > 0) or `gt` (classes with ground truth only) |
| `depth_eval` | `raw` | depth: `raw` (metric models as is; relative models not plotted) or `aligned` (per-frame scale + shift, all models) |
| `depth_min_mm` | `0` | depth ground truth below this (mm) is ignored (0 is always ignored), e.g. a stereo camera's minimum range |
| `obb_score` | `dota` | oriented detection: `dota` (classes as annotated) or `vehicle` (large + small vehicle merged) |
| `lpr_iou` | `0.5` | plate / bib recognition: box IoU for a prediction to match a ground-truth box |

## Input size

Network input size per task and model, e.g. a larger input for small objects:

```yaml
input_size:
  detection: {ultralytics-yolo26m.pt-thr0.05: 1280, rfdetr-rfdetr-medium-thr0.05: 1152}
  tracking: {ultralytics-yolo26m.pt-bytetrack: 1280}
```

- Keys are the configured model ids (as printed by `--plan`); N is a positive multiple of 32.
- The global `input_size` applies to every sequence; `sequence_overrides.<seq>.input_size` (same form) wins per task
  and model, and `null` there restores the model's default on that sequence.
- The model id gets `-<N>px`, so the resized model has its own predictions.
- Supported wrappers:
  - `ultralytics`: `imgsz`, the longer frame side is resized to N and letterboxed (also tracking, seg, pose, OBB);
  - `rfdetr`: square N × N `resolution`; N must also be a multiple of the model's patch size × windows (32 for
    rfdetr-medium, 24 for rfdetr-seg-medium);
  - `rtdetr`: stretched to N × N like its 640 × 640 default, the geometry it was trained with (keeping the aspect
    ratio, used before 2026-11-13, was worse). A larger input helps RT-DETR only on very small objects (synthetic
    sequences, 960) and costs its large objects on real footage.
- Any other wrapper: the model is **not run** on that sequence; `--plan` and the run log warn, and the results JSON
  lists it under `models_not_run`.
- A ViTPose model whose `detector` has an input size reads that detector's resized predictions (its id then contains
  the resized detector id).

The sizes used in this project's evaluation (since 2026-11-16): YOLO26 1280 and RF-DETR-M / RF-DETR-Seg-M 1152 in
detection, instance segmentation and tracking on every sequence; RT-DETRv2 960 on the synthetic sequences only.

## Per-sequence overrides

`sequence_overrides` is keyed by sequence folder name. Allowed keys:

```yaml
sequence_overrides:
  MySeq_1920x1080_RA:
    framerate: 50                       # if there is no data/<seq>/info.json
    qps: [37, 42, 47, 50]               # 'all' or a list; also limits the results and plots
    annotations: MySeq                  # annotation sub-folder, if it differs from the sequence name
    tasks: [detection, tracking]        # run only these tasks
    models:                             # run only these of a task's models (ids as printed by --plan)
      detection: [ultralytics-yolo26m.pt-thr0.05]
    classes: [car]                      # score only these classes, or per task {tracking: [person]}
    input_size:                         # as the global input_size; wins per task and model; null = model default
      detection: {ultralytics-yolo26m.pt-thr0.05: 960}
    ap_average: object                  # any metric option of the table above
    obb_classes: [small vehicle]        # oriented detection: DOTA class names scored
    remove_depth_maps: true             # as inference.remove_depth_maps
```

- **`qps`**: a list also limits the sequence's results and plots to those QPs; other QP folders are ignored (listed
  under `qps_excluded`), their files kept. `--qp` on the command line wins.
- **`tasks`**: tasks not listed are skipped on this sequence.
- **`models`**: tasks not listed run all their models.
- **`classes`**: detection, instance segmentation and tracking score only the listed classes (names from
  `task_models/labels.py`); ground truth and predictions of other classes are left out, as if only these categories
  existed, and the reported value is the average over the listed classes. Useful when a class has only one or two
  objects in a sequence, which would otherwise weigh as much as a class with thousands. In panoptic and semantic
  segmentation the list selects the thing classes only: the listed things and all stuff classes enter the PQ / mIoU
  averages. A mapping `{<task>: [names]}` restricts only the tasks it names (detection, instance_segmentation,
  tracking, panoptic_segmentation, semantic_segmentation).
- **Metric options** (`ap_average`, `hota_average`, `pq_average`, `depth_eval`, `depth_min_mm`, `obb_score`,
  `lpr_iou`) override the global value on this sequence.

## BD anchors and inheritance

BD anchors change the configuration that is run, also when the BD step is disabled.

- The anchors come from `bd.anchors_file` (default `configs/anchors.yaml`, a missing file = no anchors) and
  `bd.anchors` (per sequence name; wins over the file; `null` removes an anchor):

  ```yaml
  anchors:
    CampusView: CampusView_1920x1080_444_10b_VTM_RA
  ```
- Every other folder with the same sequence name is a **test** encoding. When the config is loaded, it inherits the
  anchor's `sequence_overrides`: all keys except `qps` (a test has its own QP folders). The test's own keys win key by
  key. The global settings are never changed.
- The purpose is that both encodings run the same model ids (e.g. the anchor's `input_size`), classes and metric
  options, so that they can be compared.
- `--plan` and the run log list the inherited keys, and the test's results file records them
  (`overrides_inherited`).
- To run a folder without them: remove its sequence name's anchor, or set the key in the test's own
  `sequence_overrides`.

## Decoding

| Key | Default | Meaning |
|---|---|---|
| `decode.matrix` | `bt709` | YUV → RGB matrix: `bt709`, `bt601`, `bt2020` |
| `decode.range` | `full` | `full` or `limited` |
| `decode.png_bit_depth` | `8` | `8` or `16` |
| `decode.verify_md5` | `true` | compare the decoded planes with the encoder log MD5s, when available |
| `decode.remove_frames` | `false` | delete decoded PNGs after evaluation (same as `--remove-frames`) |
| `decode.min_free_gb` | `20` | free disk space (GiB) that must remain after writing the decoded frames; otherwise the decode stops with an error |
| `decode.timeout_s` | `3600` | per bitstream |
| `decode.decoder` | `ffmpeg` | `ffmpeg` (native vvc / hevc decoders) or `vtm` (VVC reference decoder, raw YUV piped to ffmpeg) |
| `decode.decoder_executable` | `null` | `null` = the decoder's default in `tools_dir` (vtm: `DecoderAppStatic`), or a path |

The decode settings are recorded with the frames; changing them re-decodes the frames and recomputes the
predictions. PNG frames without a decode marker are user-provided and used as they are.

## Inference

| Key | Default | Meaning |
|---|---|---|
| `inference.timeout_s` | `7200` | per (sequence, QP, model) |
| `inference.remove_depth_maps` | `false` | delete predicted depth map PNGs after scoring (same as `--remove-depth-maps`; also per sequence); a rerun skips inference while the scores stay valid |

## Plots

| Key | Default | Meaning |
|---|---|---|
| `plot.log_x` | `false` | bitrate axis of the sequence RD plots: linear, or log scale with `true` |

## BD comparison

| Key | Default | Meaning |
|---|---|---|
| `bd.enabled` | `false` | run the BD step at the end of a run, for the pairs in which a sequence of the run is the anchor or the test |
| `bd.anchors_file` | `configs/anchors.yaml` | `{anchors: {<sequence_name>: <anchor folder>}}`; see [BD anchors](#bd-anchors-and-inheritance) |
| `bd.anchors` | `{}` | per sequence name, wins over the file; `null` removes |
| `bd.qps` | `{}` | points used per pair, e.g. `{<test folder>: {anchor: [27, 32, 37, 42], test: [27, 32, 37, 42]}}`; default all |
| `bd.min_points` | `4` | Pareto points needed on each curve (fewer → n/a) |
| `bd.log_x` | `true` | BD plots: log bitrate axis (independent of `plot.log_x`) |
| `bd.extended` | `true` | extended analysis per pair: sub-metric and per-class BD, model agreement |
| `bd.out_dir` | `BD-results` | output folder |

`bd.enabled` affects only the BD step, not the configuration. Anchors, checks and outputs: [BD_RATE.md](BD_RATE.md); the method:
[METRICS.md](METRICS.md#bd-rate-and-bd-accuracy).

## Command-line options

| Option | Effect |
|---|---|
| `<config.yaml>` | config file (default `configs/config.yaml`) |
| `--seq <folder> ...` | only these sequences (instead of `sequences`) |
| `--qp <n> ...` | only these QPs (wins over `qps` and the per-sequence `qps`) |
| `--plan` | print the work plan and stop |
| `--remove-frames` | as `decode.remove_frames: true` |
| `--remove-depth-maps` | as `inference.remove_depth_maps: true` |
| `--bd` | only the BD step, from the results on disk (no decoding, inference or scoring) |
| `--bd --anchor <folder> --test <folder>` | one given pair (anchor = test is a self-check: all zeros) |
| `-v`, `--verbose` | more log output |
