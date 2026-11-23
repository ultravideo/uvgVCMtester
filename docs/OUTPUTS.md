# Outputs

Everything a run writes: per-sequence results, plots and caches, the collected summary, BD comparisons and logs.
`<seq>` = the sequence folder name under `data_root`. Configuration options: [CONFIGURATION.md](CONFIGURATION.md);
metric definitions: [METRICS.md](METRICS.md).

## Overview

| File | Content |
|---|---|
| `data/<seq>/results/<seq>_results.json` | the sequence's results: bitrates, accuracies per task, model and QP, settings, versions, warnings |
| `data/<seq>/results/<seq>_<task>_rd.png` | RD curve per task, one line per model |
| `data/<seq>/results/rd_per_model/<seq>_<task>_<model_id>_rd.png` | RD curve of one model |
| `data/<seq>/results/predictions/<model_id>/` | cached predictions per QP |
| `data/<seq>/results/scores/<model_id>/` | cached scores per QP |
| `data/<seq>/results/cache/` | ground-truth masks converted to the decoded frame size |
| `data/<seq>/results/logs/` | per-sequence logs |
| `data/<seq>/QPxx/<bitstream_stem>_NNNN.png`, `<bitstream_stem>.decode.json` | decoded frames and their decode marker |
| `results/summary.json` | all sequences' results collected into one file |
| `results/logs/run_<timestamp>.log` | run log (`bd_<timestamp>.log` for `--bd`) |
| `BD-results/<test>__vs__<anchor>/` | BD comparison of a test encoding against its anchor |
| `BD-results/summary.json`, `BD-results/overview.md` | all BD comparisons and the cross-sequence overview |

## Per-sequence results file

`data/<seq>/results/<seq>_results.json`:

- **Sequence:** name, frame rate and its source (`info.json` or config), annotation folder, decode settings, the
  bitrate unit of the sequence, QP folders excluded by a `qps` list (`qps_excluded`), the `sequence_overrides`
  inherited from the BD anchor (`overrides_inherited`), warnings and errors, the config file and seed.
- **Per QP (`qps`):** bitstream, size in bytes, frame count, fps, bitrate (`bitrate`, `bitrate_unit`,
  `bitrate_source`; `bitrate_kbps` for bitstreams), the encoder-log bitrate it was cross-checked with,
  `bitrate.json` if present, the size of the evaluated frames (`frame_size`, [width, height]), and the decode record (decoded or user-provided PNGs, settings, MD5 check, ffmpeg /
  decoder version, stream codec, profile, size and pixel format).
- **Per task (`tasks.<task>`):** annotation type, version (`""` if missing) and sha256, the metric and the plotted
  key (`metric_key`) with its option values (e.g. `ap_average`), class selection, models not run
  (`models_not_run`), and per model:
  - `model`: model id, wrapper, model name, score threshold and its source, weights sha256, device;
  - `points`: one per valid QP: bitrate, the plotted value, all stats of the metric (e.g. AP, AP50, AP75, APs/m/l,
    AR; both averages such as `stats` and `stats_object_weighted`), per-class values, frame and object counts,
    the scores file;
  - `not_plotted`: QPs whose predictions or scores no longer match the current config or annotations, or that
    have no bitrate or another bitrate unit, with the reason;
  - `sanity`: accuracy should rise with the bitrate; violations are listed.
- **Environment:** vcmtest version, git commit, Python, platform, package versions, ffmpeg, CUDA, cuDNN, GPU.

Accuracies are fractions in [0, 1] (AP 0.25 = 25 %); the plots show percent. The file holds every valid data point
on disk (all QPs run so far), so a run with new QPs extends the curve.

## Plots

- `<seq>_<task>_rd.png`: one line per model, the plotted metric vs bitrate in the sequence's unit; linear bitrate
  axis (`plot.log_x: true` for log).
- `rd_per_model/<seq>_<task>_<model_id>_rd.png`: one model, with the same colour and marker as in the combined plot.
  Never deleted: plots of models no longer configured, or without valid points, stay as last drawn.

## Predictions, scores and caches

- `predictions/<model_id>/QPxx.json`: per-frame predictions (format per task: [annotation_format.txt](annotation_format.txt));
  `QPxx.partial.jsonl` while a model is running, so an interrupted run resumes. Depth models also write
  `QPxx_depth/` (predicted depth maps; deleted after scoring with `inference.remove_depth_maps` /
  `--remove-depth-maps`, the cached scores stay valid).
- `scores/<model_id>/QPxx.json` (detection) and `QPxx_<task>.json` (other tasks): the metric's stats per QP, with
  the settings they were computed with.
- `cache/segm_gt_<W>x<H>.json`, `cache/panoptic_gt_<W>x<H>.json`: ground-truth masks resized to the decoded size.
- Decoded frames stay in the QP folder unless `--remove-frames` / `decode.remove_frames`. The marker
  `<bitstream_stem>.decode.json` records the decode settings, the decoder and its version and the MD5 check; a
  change of settings re-decodes. PNGs without a marker are user-provided: used as is, never deleted.

### Caching and removed models

Removing a model from the config drops it from the results file and the combined plot only: its predictions,
scores and per-model plot stay on disk (nothing under `results/` is deleted). Putting it back reuses its files: if
its predictions still match, it is not run again, its scores are read from disk and its points reappear in the
results file and the plots.

- Predictions are reused when the model id (including `-<N>px`), the configured score threshold, the tracker
  settings and sequence frame rate (tracking), the decode settings, the frame count and, for top-down pose, the
  detector's predictions are unchanged; otherwise the model runs again.
- Scores are reused when the annotations (and masks), the predictions file, the frame set, the metric version
  (pycocotools / TrackEval version, vcmtest scoring version) and the scoring settings that change the value (class
  selection, `lpr_iou`, `depth_min_mm`) are unchanged; otherwise they are recomputed from the cached predictions.
- Not detected on a fully cached QP: new weights under the same model name (the weights hash is checked only when
  the model is loaded anyway), and a changed library default threshold when the config sets no `score_threshold`.
  In both cases delete `data/<seq>/results/predictions/<model_id>/` by hand to force a rerun.

## Bitrate

`kbps = bytes × 8 × fps / frames / 1000` from the bitstream in `QPxx/`, cross-checked with the encoder log when
there is one. A QP folder with decoded PNGs only (e.g. from a codec ffmpeg cannot decode) needs `QPxx/bitrate.json`:

```json
{"bitrate": 0.35, "unit": "bpp"}
```

The value is used as given (never converted); `unit` is optional free text and only labels the plot's x axis
("Bitrate" without a unit if absent). With a bitstream, the bitstream wins and `bitrate.json` is only
cross-checked (when its unit is exactly `kbps`). A QP without any bitrate, or whose unit differs from the rest of
the sequence, is listed under `not_plotted`. A sequence-level `bitrates.json` is not read. QP folder numbers are
only labels for the pipeline: `QP0`, `QP1`, ... work as rate points (the QP-order sanity note then only warns).

## Summary

`results/summary.json`: the results files of all sequences under `data_root`, condensed into one file and rewritten at
the end of every run. It records the `data_root` it was collected from; rewriting it from another data root replaces
its sequences and is warned ([several data roots](CONFIGURATION.md#several-data-roots)). Per sequence: results file, frame rate, warnings and errors; per task: metric and annotation
version; per model: score threshold, weights sha256 and per QP the bitrate and the plotted value.

## BD comparisons

Written to `bd.out_dir` (default `BD-results/`); anchors, method and options: [BD_RATE.md](BD_RATE.md).

| File | Content |
|---|---|
| `<test>__vs__<anchor>/bd.json` | per task and model: BD-rate %, BD-accuracy pp, the points of both sides, Pareto / removed QPs, the overlap intervals, n/a or skip reasons; per-task mean over the models; settings checked; encodings (codec, profile, pixel format, frame size) of both sides, with `frame_size_note` when the sizes differ or are unknown; `data_root` (where both results files were read); vcmtest version |
| `<test>__vs__<anchor>/<task>_rd.png`, `rd_per_model/` | anchor solid, test dashed, same colour per model; Pareto points filled and joined, removed points hollow; log bitrate axis (`bd.log_x`) |
| `<test>__vs__<anchor>/<test>__vs__<anchor>.md` | tables of the BD numbers per task |
| `<test>__vs__<anchor>/extended.json`, `<test>__vs__<anchor>_extended.md` | extended analysis (`bd.extended`, default on; same results, no rescoring), see below |
| `summary.json` | all BD comparisons collected into one file, with an `overview`; `data_root` per pair and `data_roots` (more than one = pairs of several data roots mixed, warned) |
| `overview.md` | cross-sequence overview; names the data root(s) |

**Extended analysis:**
1. BD of the sub-metrics per model: components of the plotted value (detection AP50 / AP75 / APs / APm / APl,
   tracking DetA / AssA / LocA / MOTA / IDF1, panoptic SQ / RQ / PQ things / stuff, plate / bib detection recall vs
   recognition, ...) and per class, with the anchor's best value.
2. Model agreement per task: the BD-rate sign over the models (consistent / mostly / mixed, spread flag), and per
   QP step the mean change (also per halving of the bitrate), the models that get better at the lower rate (one =
   model noise, all = content or codec effect) and where each model's largest drop is.

**Overview:** pairs grouped by comparison = the encoding spec of the test vs the anchor (folder name without the
sequence name, e.g. `1920x1080_444_10b_VCMRS_RA vs 1920x1080_444_10b_VTM_RA`); per task each sequence's mean over
its models and the mean over the sequences (each weighs equally), per model the mean over the sequences where it was
compared.

## Logs

- `results/logs/run_<timestamp>.log`: the whole run (plan, inherited overrides, progress, warnings, timings);
  `results/logs/bd_<timestamp>.log` for `--bd`.
- `data/<seq>/results/logs/`: per-sequence logs.
