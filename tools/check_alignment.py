#!/usr/bin/env python3
"""Frame-alignment sanity check: score predictions of decoded frame k against annotation frame k+offset.

If decoded frames and annotations are aligned, offset 0 gives clearly the best AP. Use a high-quality
QP: at low quality neighbouring frames score almost the same and the check becomes inconclusive.
Exit code: 0 aligned, 1 misaligned (another offset better by >= --margin AP points), 2 inconclusive.

  .venv/bin/python tools/check_alignment.py <sequence_folder> <qp> <model_id> [--offsets -2 -1 0 1 2]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vcmtest.annotations import load_detection_annotations  # noqa: E402
from vcmtest.config import load_config, resolve_path  # noqa: E402
from vcmtest.metrics.coco_det import evaluate  # noqa: E402
from vcmtest.paths import find_annotations, scan_pngs  # noqa: E402
from vcmtest.score import image_size  # noqa: E402


def main() -> int:
    """Score the predictions at each frame offset, print the verdict and return the exit code (0, 1 or 2)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sequence")
    ap.add_argument("qp", type=int)
    ap.add_argument("model_id")
    ap.add_argument("--config", default=None)
    ap.add_argument("--offsets", type=int, nargs="+", default=[-2, -1, 0, 1, 2])
    ap.add_argument("--margin", type=float, default=1.0, help="AP points needed for a verdict")
    a = ap.parse_args()

    cfg = load_config(a.config)
    seq_dir = resolve_path(cfg["data_root"]) / a.sequence
    override = (cfg["sequence_overrides"] or {}).get(a.sequence, {}).get("annotations")
    inv = find_annotations(resolve_path(cfg["annotations_root"]), a.sequence, override)
    ann = load_detection_annotations(inv.detection_json)
    pred_path = seq_dir / "results" / "predictions" / a.model_id / f"QP{a.qp}.json"
    preds = json.loads(pred_path.read_text())["frames"]
    w, h = image_size(next(iter(scan_pngs(seq_dir / f"QP{a.qp}").values())))

    results = {}
    for off in a.offsets:
        # prediction of decoded frame k is scored as if it were annotation frame k + off
        shifted = {str(int(k) + off): v for k, v in preds.items()}
        frames = sorted(f for f in ann.frames if str(f) in shifted)
        r = evaluate(ann, shifted, frames, w, h)
        results[off] = r["stats"]["AP"]
        print(f"offset {off:+d}: AP = {100 * r['stats']['AP']:6.2f}  ({len(frames)} frames)")
    if 0 not in results:
        print("offset 0 not evaluated")
        return 2
    others = {k: v for k, v in results.items() if k != 0}
    best_other = max(others, key=others.get)
    gap = 100 * (results[0] - others[best_other])
    if gap >= a.margin:
        print(f"aligned: offset 0 beats the best other offset ({best_other:+d}) by {gap:.2f} AP points")
        return 0
    if gap <= -a.margin:
        print(f"MISALIGNED: offset {best_other:+d} beats offset 0 by {-gap:.2f} AP points")
        return 1
    print(f"inconclusive: offset 0 vs best other ({best_other:+d}) differ by {gap:+.2f} AP points "
          f"(< {a.margin}); try a higher-quality QP")
    return 2


if __name__ == "__main__":
    sys.exit(main())
