# BD-rate against an anchor

Bjøntegaard comparison of a **test** encoding against an **anchor** encoding of the same source video, per task and
model (`vcmtest/bd.py`, `vcmtest/bd_extended.py`). The comparison works on the results files already on disk; it
never decodes, runs models or scores.

Related: options in [CONFIGURATION.md](CONFIGURATION.md#bd-comparison), formulas in
[METRICS.md](METRICS.md#bd-rate-and-bd-accuracy), output files in [OUTPUTS.md](OUTPUTS.md#bd-comparisons).

## Anchors and tests

Several data folders can hold encodings of the same source. The sequence name (folder name up to the first `_`)
identifies the source, e.g. `CampusView_1920x1080_444_10b_VTM_RA` and `CampusView_1920x1080_420_10b_kvazaar_medium`
are both `CampusView`.

An anchors file names one anchor folder per sequence name:

```yaml
# configs/anchors.yaml
anchors:
  CampusView: CampusView_1920x1080_444_10b_VTM_RA
```

- The file is set with `bd.anchors_file` (default `configs/anchors.yaml`; a missing file = no anchors, with a
  warning when the option names a file).
- `bd.anchors` in config.yaml overrides the file per sequence name; `null` removes an anchor:
  `bd.anchors: {CampusView: null}`.
- The anchor folder must be named `<sequence_name>_...` (otherwise a config error). An anchor folder that is not in
  `data_root` gives a warning.
- Every other folder of that sequence name is a **test** and is compared against the anchor. The anchor itself is
  never a test, and a test is compared only against its own anchor.

### Several data roots

The anchors name folders, not paths: they always refer to the current `data_root`. Anchor and test must be in the
same data root. With a second data root, give it its own anchors file and outputs (`bd.anchors_file`, `bd.out_dir`,
`results_root`), otherwise `BD-results/summary.json` and `overview.md` average over the pairs of both roots. The data
root is recorded in `bd.json`, `summary.json` and `overview.md`; most anchors missing from `data_root`, a pair
replaced by one from another data root, and an output folder with several data roots are warned. Details and an
example config: [CONFIGURATION.md](CONFIGURATION.md#several-data-roots).

## Anchors change the run configuration of the tests

So that both sides are comparable by default, a test folder **inherits the anchor's `sequence_overrides`** when the
config is loaded:

- all keys except `qps` (a test has its own QP folders): e.g. `classes`, `models`, `tasks`, `input_size` (which
  changes the model ids), `ap_average` / `hota_average` / `pq_average` and the other metric options, `lpr_iou`,
  `depth_min_mm`, `framerate`, `annotations`;
- the test's own `sequence_overrides` win key by key (a key the test sets is not inherited);
- the global settings (tasks, models, decode, inference, ...) are never changed;
- the inherited keys are printed by `--plan`, written to the run log ("sequence_overrides [...] inherited from its
  BD anchor ...") and recorded in the test's results file (`overrides_inherited`).

This depends only on the anchors (`bd.anchors_file`, `bd.anchors`), **not on `bd.enabled`**: with BD disabled the
tests still run with the anchor's overrides. To run a folder without them, remove its sequence name's anchor
(delete the line in the anchors file, or `bd.anchors: {<sequence_name>: null}`), or set the key in the test's own
`sequence_overrides`.

## When the comparison runs

| How | Pairs compared |
|---|---|
| `bd.enabled: true` + a normal run | at the end of the run (after all sequences and `results/summary.json`): the pairs in which a sequence of the run is the anchor or the test |
| `--bd` | only the BD step, all configured pairs whose results files exist |
| `--bd --seq <folder> ...` | only the BD step, the configured pairs involving those folders |
| `--bd --anchor A --test T` | one given pair, also without an anchors entry; anchor = test is a self-check (all 0) |

- `bd.enabled` controls only this step, not the configuration. With `bd.enabled: false` no BD step follows a run.
- The other side of a pair is read from its results file on disk and is not re-evaluated. A missing results file
  gives a warning and the pair is skipped (with `--anchor` / `--test`: an error naming the folder).
- `bd.qps`, `bd.min_points`, `bd.log_x`, `bd.extended` and `bd.out_dir` act on this step only.
- `--bd` writes its log to `results/logs/bd_<timestamp>.log`.

## Comparability checks

Both sides must be measured the same way. The two results files are compared per task and model:

- **Task skipped** (with the reason) when the two sides differ in: the plotted metric key or its options
  (`ap_average`, `hota_average`, `pq_average`, `depth_eval`, `depth_metric`, `obb_score`), `classes`, `lpr_iou`, `depth_min_mm`,
  annotation version or fingerprint (sha256), or bitrate unit (bitrates are compared as given, never converted).
  A task run on one side only is skipped as well.
- **Model n/a** when its weights (sha256) or frame counts differ.
- The codec, profile and pixel format of both sides are recorded in `bd.json` (e.g. 4:4:4 anchor vs 4:2:0 test is
  allowed: it is part of the encoding under test).
- **Frame size** (width × height of the evaluated frames) is recorded per side in `bd.json` (`encodings.*.frame_size`)
  but does not skip the comparison: a reduced resolution is part of the encoding under test, and each side is scored
  against the same annotations at its own frame size (boxes, masks and depth scaled to it). When the sizes differ,
  `bd.json` (`frame_size_note`), the Markdown table and the log say so. The size-range sub-metrics of the extended
  analysis (APs / APm / APl) use absolute pixel areas and are then not directly comparable. A results file written
  before the frame size was recorded gives the note "frame size not checked" (PNG-only folders; rerun the sequence
  to record it).

## Method

Per task and model, each side is a curve of the plotted metric (%) against bitrate.

1. **Points:** the plotted points of the results file (its `qps` limits already applied); `bd.qps` can select the
   points per pair, e.g. 4 QPs as in a CTC:
   ```yaml
   bd:
     qps:
       CampusView_1920x1080_420_10b_kvazaar_medium: {anchor: [27, 32, 37, 42], test: [27, 32, 37, 42]}
   ```
2. **Pareto front:** a point is dropped when another point of the same curve has a lower or equal bitrate and a
   better or equal accuracy, one of them strictly (non-monotonic points). The removed QPs are listed in `bd.json`.
3. **Interpolation:** PCHIP (piecewise cubic Hermite, as the `bjontegaard` package / JVET CTC sheets) over log10
   bitrate, integrated exactly over the overlapping range only.
4. **n/a** with the reason: fewer than `bd.min_points` (default 4) Pareto points on either curve, or no overlap.
5. **Mean:** per task, the mean over the models with a value (n/a models excluded; the number of models is recorded).

Results:

- **BD-rate (%)**: bitrate change of the test at equal task accuracy. Negative = the test needs less bitrate
  (better), for every metric direction.
- **BD-accuracy (pp)**: change of the plotted metric at equal bitrate, in percentage points. Positive = the test is
  better, for every metric direction: for lower-is-better metrics such as depth AbsRel the sign is flipped (anchor −
  test). With `depth_metric: one_minus_abs_rel` depth is plotted as 1 − AbsRel (higher is better): same BD-rate and
  same BD-accuracy as AbsRel.

Formulas: [METRICS.md](METRICS.md#bd-rate-and-bd-accuracy).

## Extended analysis

With `bd.extended: true` (default), each pair also gets an extended analysis from the same results (no rescoring):

1. **Breakdown:** BD-rate / BD-accuracy of the sub-metrics of each model (e.g. AP50 / AP75 / APs / APm / APl,
   DetA / AssA / LocA / MOTA / IDF1, SQ / RQ / PQ things / stuff, plate detection recall vs recognition) and per
   class, with the anchor's best value (a BD of a sub-metric near 0 on the anchor is not meaningful).
2. **Model agreement** per task: whether the models agree on the sign of the BD-rate (consistent / mostly / mixed,
   spread flag), and per QP step the mean change, inversions (models better at the lower bitrate: one model = model
   noise, all = content or codec effect) and where each model's largest drop is.

Details: [METRICS.md](METRICS.md#extended-bd-analysis).

## Outputs

In `bd.out_dir` (default `BD-results/`):

- `<test>__vs__<anchor>/`: `bd.json` (all numbers, points, removed QPs, overlaps, reasons, encodings),
  `<task>_rd.png` and `rd_per_model/` (anchor solid, test dashed, same colour per model; Pareto points filled and
  joined, removed points hollow; log bitrate axis with `bd.log_x: true`, default, independent of `plot.log_x`),
  `<test>__vs__<anchor>.md` (tables), and with the extended analysis `extended.json` and
  `<test>__vs__<anchor>_extended.md`.
- `summary.json`: all comparisons in one file, with an `overview`.
- `overview.md`: cross-sequence overview: pairs grouped by comparison = encoding spec of the test vs the anchor
  (folder name without the sequence name), per task each sequence's mean over its models and the mean over the
  sequences (equal weight), per model the mean over the sequences where it was compared.

Details: [OUTPUTS.md](OUTPUTS.md#bd-comparisons).

## Example

```bash
# configs/anchors.yaml names the anchor; both folders exist under data/
uv run python -m vcmtest --plan --seq CampusView_1920x1080_420_10b_kvazaar_medium   # shows the inherited overrides
uv run python -m vcmtest --seq CampusView_1920x1080_420_10b_kvazaar_medium           # evaluate the test
uv run python -m vcmtest --bd --seq CampusView_1920x1080_420_10b_kvazaar_medium      # compare it with its anchor
# or, without an anchors entry:
uv run python -m vcmtest --bd --anchor CampusView_1920x1080_444_10b_VTM_RA --test CampusView_1920x1080_420_10b_kvazaar_medium
```
