#!/usr/bin/env python3
"""
Multi-object trackers from Roboflow `trackers` (PyPI, Apache-2.0), applied to the per-frame output of any
detection wrapper (torchvision, rfdetr, rtdetr, ultralytics, roboflow, ...).

Algorithms (config `tracker:` value): bytetrack, ocsort, sort, botsort, cbiou.
Library defaults are used for all tracker parameters except `frame_rate`, which is set to the sequence's
frame rate (it describes the video: the lost-track buffer is defined in 30 fps frames and rescaled with it).

Input per frame:  detections [class_id, score, x1, x2, y1, y2(, ...)] normalised to [0, 1].
Output per frame: [class_id, score, x1, x2, y1, y2, track_id] for detections assigned to a confirmed track
                  (the library returns tracker_id -1 for the others; they are dropped).
"""

import inspect

import numpy as np

ALGORITHMS = {
    "bytetrack": "ByteTrackTracker",
    "ocsort": "OCSORTTracker",
    "sort": "SORTTracker",
    "botsort": "BoTSORTTracker",
    "cbiou": "CBIoUTracker",
}


def _plain(v):
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return getattr(v, "__name__", repr(v))


def tracker_parameters(name: str) -> dict:
    """Library, version, algorithm and default constructor parameters (for the results and the cache key)."""
    from importlib.metadata import version

    import trackers
    cls = getattr(trackers, ALGORITHMS[name])
    params = {k: _plain(p.default) for k, p in inspect.signature(cls.__init__).parameters.items()
              if k != "self" and p.default is not inspect.Parameter.empty}
    return {"library": "trackers", "version": version("trackers"), "algorithm": name, "class": ALGORITHMS[name],
            "params": params}


class TrackersAdapter:
    """Roboflow `trackers` multi-object tracker run on the per-frame output of a detection wrapper."""
    def __init__(self, name: str):
        if name not in ALGORITHMS:
            raise ValueError(f"Unknown tracker '{name}'; known: {sorted(ALGORITHMS)}")
        import trackers
        self.name = name
        self.cls = getattr(trackers, ALGORITHMS[name])
        self.tracker = None
        self.frame_rate = None

    def reset(self, frame_rate: float) -> None:
        """Start a new sequence with a fresh tracker."""
        self.frame_rate = float(frame_rate)
        self.tracker = self.cls(frame_rate=self.frame_rate)

    def update(self, preds: list, width: int, height: int, frame=None) -> list:
        """Track one frame's detections.

        Args:
            preds: Detections [class_id, score, x1, x2, y1, y2, ...] normalised to [0, 1].
            width: Frame width in pixels.
            height: Frame height in pixels.
            frame: Optional frame image, passed to the tracker.

        Returns:
            [class_id, score, x1, x2, y1, y2, track_id] normalised, only for detections on a confirmed track.
        """
        import supervision as sv
        if self.tracker is None:
            raise RuntimeError("reset(frame_rate) must be called before the first frame")
        if preds:
            a = np.array([p[:6] for p in preds], dtype=float)
            xyxy = np.stack([a[:, 2] * width, a[:, 4] * height, a[:, 3] * width, a[:, 5] * height], axis=1)
            det = sv.Detections(xyxy=xyxy, confidence=a[:, 1], class_id=a[:, 0].astype(int))
        else:
            det = sv.Detections.empty()
        out = self.tracker.update(det, frame=frame)
        res = []
        for i in range(len(out)):
            tid = int(out.tracker_id[i])
            if tid < 0:
                continue
            x1, y1, x2, y2 = (float(v) for v in out.xyxy[i])
            res.append([int(out.class_id[i]), float(out.confidence[i]), x1 / width, x2 / width,
                        y1 / height, y2 / height, tid])
        return res
