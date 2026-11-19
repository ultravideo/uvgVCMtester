"""Bitrate of one QP point.

Source 1 (when the QP folder has a bitstream): bitrate_kbps = bytes * 8 * fps / frames / 1000, unit "kbps".
Source 2 (PNG frames only): QPxx/bitrate.json {"bitrate": <number>, "unit": <optional free text>}; the value is used
as given, never converted, and the unit is only a label (None if not given). With a bitstream, bitrate.json is only
cross-checked (when its unit is exactly "kbps") and recorded. A sequence-level bitrates.json is not read.
"""

from __future__ import annotations

import re
from pathlib import Path

_LOG_KBPS = re.compile(r"Bytes written to file:\s*(\d+)\s*\(([\d.]+)\s*kbps\)")


def bitrate_kbps(n_bytes: int, fps: float, frames: int) -> float:
    """Bitrate in kbps: bytes * 8 * fps / frames / 1000.

    Raises:
        ValueError: frames or fps not positive.
    """
    if frames <= 0 or fps <= 0:
        raise ValueError(f"invalid frames={frames} or fps={fps}")
    return n_bytes * 8.0 * fps / frames / 1000.0


def encoder_log_kbps(log_file: Path | None) -> tuple[int, float] | None:
    """(bytes, kbps) reported by a VTM/HM encoder log, if present."""
    if not log_file or not log_file.is_file():
        return None
    with open(log_file, errors="replace") as f:
        for line in f:
            m = _LOG_KBPS.search(line)
            if m:
                return int(m.group(1)), float(m.group(2))
    return None


BITRATE_JSON = "bitrate.json"


def read_bitrate_json(qp_dir: Path | None) -> tuple[dict | None, str | None]:
    """(provided, error): provided = {"bitrate": float, "unit": str | None} from QPxx/bitrate.json, or None if the
    file does not exist; error = why the file cannot be used."""
    import json
    import math
    p = qp_dir / BITRATE_JSON if qp_dir else None
    if p is None or not p.is_file():
        return None, None
    try:
        with open(p) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return None, f"{p.name}: not readable JSON ({e})"
    v = d.get("bitrate") if isinstance(d, dict) else None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0:
        return None, f'{p.name}: needs "bitrate": a positive number'
    unit = d.get("unit")
    if unit is not None and not (isinstance(unit, str) and unit.strip()):
        return None, f'{p.name}: "unit" must be a non-empty string if given'
    return {"bitrate": float(v), "unit": unit.strip() if unit else None}, None


def bitrate_info(bitstream: Path | None, fps: float, frames: int | None, log_file: Path | None,
                 qp_dir: Path | None = None) -> dict:
    """Bitrate entry for one QP: "bitrate", "bitrate_unit", "bitrate_source" (+ "bitrate_kbps" for bitstreams), or
    "bitrate": None and "bitrate_error". Bitstream: cross-checked with the encoder log and a "kbps" bitrate.json."""
    provided, perr = read_bitrate_json(qp_dir)
    if bitstream is None:
        if perr:
            return {"bitstream": None, "bitrate": None, "bitrate_error": perr}
        if provided is None:
            return {"bitstream": None, "bitrate": None,
                    "bitrate_error": f"bitrate unknown: no bitstream and no QPxx/{BITRATE_JSON}"}
        return {"bitstream": None, "bitrate": provided["bitrate"], "bitrate_unit": provided["unit"],
                "bitrate_source": BITRATE_JSON}
    if not frames:
        return {"bitstream": bitstream.name, "bitrate": None, "bitrate_error": "bitrate unknown (frame count unknown)"}
    n = bitstream.stat().st_size
    kbps = bitrate_kbps(n, fps, frames)
    info = {"bitstream": bitstream.name, "bytes": n, "frames": frames, "fps": fps, "bitrate_kbps": kbps,
            "bitrate": kbps, "bitrate_unit": "kbps", "bitrate_source": "bitstream"}
    warn = []
    ref = encoder_log_kbps(log_file)
    if ref:
        info["encoder_log_bytes"], info["encoder_log_kbps"] = ref
        if ref[0] != n:
            warn.append(f"file size {n} B differs from encoder log {ref[0]} B")
        if abs(ref[1] - kbps) > 0.005 * ref[1]:
            warn.append(f"bitrate {kbps:.3f} kbps differs from encoder log {ref[1]:.3f} kbps by more than 0.5%")
    if perr:
        info["bitrate_json"] = {"error": perr}
    elif provided:
        info["bitrate_json"] = dict(provided)
        if provided["unit"] == "kbps":
            if abs(provided["bitrate"] - kbps) > 0.005 * kbps:
                warn.append(f"bitrate {kbps:.3f} kbps differs from {BITRATE_JSON} {provided['bitrate']:.3f} kbps "
                            f"by more than 0.5%")
        else:
            info["bitrate_json"]["note"] = ("not cross-checked (unit not given)" if provided["unit"] is None else
                                            f"not cross-checked (unit '{provided['unit']}' is not kbps)")
    if warn:
        info["warnings"] = warn
    return info
