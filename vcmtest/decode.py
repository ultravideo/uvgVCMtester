"""Stage 1: decode a bitstream directly into RGB PNGs with ffmpeg (cached, resumable).

The bitstream decoder is selectable (decode.decoder, vcmtest/decoders.py): ffmpeg's own (default) or an external
decoder such as VTM whose raw YUV output ffmpeg reads through a pipe; the YUV -> RGB conversion and the MD5 check
are always done by ffmpeg.

- Output: QPxx/<bitstream stem>_%04d.png, frame numbers start at 1 (= encoder POC + 1).
- Decoding goes into QPxx/.decode_tmp/ and is moved into place only when complete and verified,
  so an interrupted run never leaves a partial set that looks finished.
- A marker QPxx/<stem>.decode.json records the settings, ffmpeg version and checks. If the
  marker's settings differ from the current config, the frames are decoded again.
- PNGs in a QP folder without a bitstream (user-provided) are used as they are and never deleted.
- Optional MD5 check: per-frame Y/U/V plane MD5s from ffmpeg are compared with the
  [MD5:y,u,v] entries of the encoder log (VTM/HM), keyed by POC.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path

from .decoders import DecoderError, get_decoder
from .paths import QPItem, scan_pngs

log = logging.getLogger("vcmtest")

MARKER_VERSION = 1
PNG_SIZE_FACTOR = 0.5   # conservative PNG size / raw RGB size, for the disk-space estimate
_MD5_RE = re.compile(r"^POC\s+(\d+)\b.*\[MD5:([0-9a-fA-F]{32}),([0-9a-fA-F]{32}),([0-9a-fA-F]{32})\]")


class DecodeError(RuntimeError):
    """Raised when decoding a bitstream into PNG frames fails."""
    pass


# ----------------------------------------------------------------------------- helpers

def decode_settings(cfg_decode: dict) -> dict:
    """Return the decode settings recorded in the marker (matrix, range, PNG bit depth; decoder if not ffmpeg)."""
    s = {"matrix": cfg_decode["matrix"], "range": cfg_decode["range"],
         "png_bit_depth": cfg_decode["png_bit_depth"]}
    if cfg_decode.get("decoder", "ffmpeg") != "ffmpeg":
        s["decoder"] = cfg_decode["decoder"]     # (ffmpeg: not recorded, so existing caches stay valid)
    return s


def color_filter(settings: dict) -> str:
    """Return the ffmpeg filter that converts the decoded YUV to RGB PNG pixels with the given settings."""
    pix = "rgb24" if settings["png_bit_depth"] == 8 else "rgb48be"
    return (f"scale=in_color_matrix={settings['matrix']}:in_range={settings['range']}:out_range=full"
            f":flags=accurate_rnd+full_chroma_int,format={pix}")


def read_log_md5(log_file: Path | None) -> dict[int, tuple[str, str, str]]:
    """POC -> (y, u, v) MD5s from an encoder log; empty if the log has none."""
    out: dict[int, tuple[str, str, str]] = {}
    if not log_file or not log_file.is_file():
        return out
    with open(log_file, errors="replace") as f:
        for line in f:
            m = _MD5_RE.match(line)
            if m:
                out[int(m.group(1))] = (m.group(2).lower(), m.group(3).lower(), m.group(4).lower())
    return out


def read_framemd5(path: Path) -> list[str]:
    """Return the per-frame MD5s (lower case, last field of each line) from an ffmpeg framemd5 file."""
    out = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            out.append(line.rsplit(",", 1)[1].strip().lower())
    return out


def ffprobe_for(ffmpeg_path: str) -> str | None:
    """Return the ffprobe next to the ffmpeg executable, else ffprobe on PATH, or None."""
    sib = Path(ffmpeg_path).with_name("ffprobe")
    if sib.is_file():
        return str(sib)
    return shutil.which("ffprobe")


def probe_stream(ffmpeg_path: str, bitstream: Path, timeout: float = 60) -> dict:
    """Return ffprobe's properties of the first video stream (codec_name, width, height, pix_fmt, profile).

    Returns an empty dict if ffprobe is not available or finds no video stream.

    Raises:
        DecodeError: ffprobe fails.
    """
    probe = ffprobe_for(ffmpeg_path)
    if not probe:
        return {}
    r = subprocess.run([probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=codec_name,width,height,pix_fmt,profile", "-of", "json", str(bitstream)],
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise DecodeError(f"ffprobe failed on {bitstream}: {r.stderr.strip()[-500:]}")
    streams = json.loads(r.stdout).get("streams", [])
    return streams[0] if streams else {}


def check_disk_space(folder: Path, need_bytes: float, min_free_gb: float) -> None:
    """Raise DecodeError if writing need_bytes would leave less than min_free_gb GiB free in the folder."""
    free = shutil.disk_usage(folder).free
    if free - need_bytes < min_free_gb * 2**30:
        raise DecodeError(f"Not enough disk space in {folder}: free {free / 2**30:.1f} GiB, "
                          f"estimated need {need_bytes / 2**30:.1f} GiB + reserve {min_free_gb} GiB")


def marker_path(item: QPItem) -> Path:
    """Return the path of the decode marker QPxx/<bitstream stem>.decode.json."""
    return item.dir / f"{item.bitstream.stem}.decode.json"


def _load_marker(item: QPItem) -> dict | None:
    p = marker_path(item)
    if not p.is_file():
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _contiguous(frames: dict[int, Path]) -> bool:
    return list(frames) == list(range(1, len(frames) + 1))


# ----------------------------------------------------------------------------- main entry

def ensure_frames(item: QPItem, cfg: dict, ffmpeg_path: str | None, ffmpeg_version: str | None,
                  expected_frames: int | None) -> dict:
    """Make sure item.dir holds the PNG frames. Returns a status dict (also stored in results)."""
    settings = decode_settings(cfg["decode"])

    if item.bitstream is None:  # user-provided PNGs only
        if not _contiguous(item.pngs):
            raise DecodeError(f"{item.dir}: PNG frame numbers are not contiguous from 1")
        log.info(f"QP{item.qp}: using {len(item.pngs)} provided PNGs (no bitstream)")
        return {"source": "provided_pngs", "frames": len(item.pngs)}

    marker = _load_marker(item)
    if marker and item.pngs:
        same = marker.get("settings") == settings and marker.get("marker_version") == MARKER_VERSION
        complete = len(item.pngs) == marker.get("frames") and _contiguous(item.pngs)
        if same and complete:
            log.info(f"QP{item.qp}: {len(item.pngs)} PNGs already decoded (cached)")
            return {**marker, "cached": True}
        why = "decode settings changed" if not same else "frame set incomplete"
        log.warning(f"QP{item.qp}: existing PNGs are stale ({why}); decoding again")
        _delete_pngs(item)
    elif item.pngs and not marker:
        # PNGs next to a bitstream but without our marker: provided by the user. Use, don't touch.
        if not _contiguous(item.pngs):
            raise DecodeError(f"{item.dir}: PNG frame numbers are not contiguous from 1")
        if expected_frames and len(item.pngs) != expected_frames:
            raise DecodeError(f"{item.dir}: {len(item.pngs)} provided PNGs, encoder log says {expected_frames}")
        log.info(f"QP{item.qp}: using {len(item.pngs)} existing PNGs (no decode marker; not re-decoded)")
        return {"source": "provided_pngs", "frames": len(item.pngs)}

    if ffmpeg_path is None:
        raise DecodeError("ffmpeg is needed for decoding but was not resolved")
    return _decode(item, cfg, settings, ffmpeg_path, ffmpeg_version, expected_frames)


def _delete_pngs(item: QPItem) -> None:
    for p in item.pngs.values():
        p.unlink(missing_ok=True)
    item.pngs = {}
    if item.bitstream is not None:
        marker_path(item).unlink(missing_ok=True)


def _decode(item: QPItem, cfg: dict, settings: dict, ffmpeg_path: str, ffmpeg_version: str | None,
            expected_frames: int | None) -> dict:
    dcfg = cfg["decode"]
    stream = probe_stream(ffmpeg_path, item.bitstream)
    w, h = stream.get("width"), stream.get("height")
    n_est = expected_frames or 0
    if w and h and n_est:
        need = n_est * w * h * 3 * (settings["png_bit_depth"] // 8) * PNG_SIZE_FACTOR
    else:
        need = 0
        log.warning(f"QP{item.qp}: cannot estimate PNG size (no resolution or frame count); "
                    f"checking only the {dcfg['min_free_gb']} GiB reserve")
    check_disk_space(item.dir, need, dcfg["min_free_gb"])

    log_md5 = read_log_md5(item.log_file) if dcfg["verify_md5"] else {}
    do_md5 = bool(log_md5)
    if dcfg["verify_md5"] and not log_md5:
        log.info(f"QP{item.qp}: no MD5s in encoder log; skipping MD5 verification")

    decoder = get_decoder(cfg)
    try:
        decoder.check_stream(stream, item.bitstream)
    except DecoderError as e:
        raise DecodeError(f"QP{item.qp}: {e}") from None
    tmp = item.dir / ".decode_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    stem = item.bitstream.stem
    try:
        input_args = decoder.start(item.bitstream, stream, tmp)
    except DecoderError as e:
        shutil.rmtree(tmp, ignore_errors=True)
        raise DecodeError(f"QP{item.qp}: {e}") from None
    cmd = [ffmpeg_path, "-hide_banner", "-nostdin", "-v", "error"] + input_args
    vf = color_filter(settings)
    if do_md5:
        cmd += ["-filter_complex",
                f"[0:v]split=4[a][b][c][d];[a]{vf}[rgb];"
                "[b]extractplanes=y[py];[c]extractplanes=u[pu];[d]extractplanes=v[pv]",
                "-map", "[rgb]", "-start_number", "1", str(tmp / f"{stem}_%04d.png")]
        for p in ("y", "u", "v"):
            cmd += ["-map", f"[p{p}]", "-f", "framemd5", str(tmp / f"md5_{p}.txt")]
    else:
        cmd += ["-vf", vf, "-start_number", "1", str(tmp / f"{stem}_%04d.png")]

    log.info(f"QP{item.qp}: decoding {item.bitstream.name} "
             f"({stream.get('codec_name', '?')} {w}x{h} {stream.get('pix_fmt', '?')}) with {decoder.name}"
             f"{' (' + decoder.version + ')' if decoder.version else ''} -> PNG {settings}")
    log.debug("cmd: " + " ".join(cmd))
    t0 = time.time()
    ok = False
    try:
        _run_with_progress(cmd, tmp, f"{stem}_", expected_frames, dcfg["timeout_s"], f"QP{item.qp}",
                           decoder.poll_failure)
        ok = True
    finally:
        try:
            decoder.finish(ok)
        except DecoderError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            raise DecodeError(f"QP{item.qp}: {e}") from None
        if not ok:
            shutil.rmtree(tmp, ignore_errors=True)
    elapsed = time.time() - t0

    frames = scan_pngs(tmp)
    n = len(frames)
    try:
        if n == 0 or not _contiguous(frames):
            raise DecodeError(f"QP{item.qp}: decoded frames missing or not contiguous ({n} PNGs)")
        if expected_frames is not None and n != expected_frames:
            raise DecodeError(f"QP{item.qp}: decoded {n} frames, encoder log says {expected_frames}")
        md5_result = None
        if do_md5:
            md5_result = _verify_md5(tmp, log_md5, n, item.qp)
    except DecodeError:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    for p in frames.values():
        p.rename(item.dir / p.name)
    shutil.rmtree(tmp, ignore_errors=True)
    item.pngs = scan_pngs(item.dir)

    status = {
        "marker_version": MARKER_VERSION,
        "source": "decoded",
        "bitstream": item.bitstream.name,
        "bitstream_bytes": item.bitstream.stat().st_size,
        "stream": stream,
        "settings": settings,
        "frames": n,
        "md5_check": md5_result if md5_result else ("skipped" if not do_md5 else None),
        "ffmpeg": ffmpeg_version,
        **({"decoder": {"name": decoder.name, "version": decoder.version,
                        "executable": getattr(decoder, "executable", None)}} if decoder.name != "ffmpeg" else {}),
        "decode_seconds": round(elapsed, 1),
    }
    with open(marker_path(item), "w") as f:
        json.dump(status, f, indent=2)
    log.info(f"QP{item.qp}: {n} frames decoded in {elapsed:.1f} s; MD5 check: {status['md5_check']}")
    return {**status, "cached": False}


def _verify_md5(tmp: Path, log_md5: dict, n: int, qp: int) -> str:
    planes = [read_framemd5(tmp / f"md5_{p}.txt") for p in ("y", "u", "v")]
    if any(len(pl) != n for pl in planes):
        raise DecodeError(f"QP{qp}: framemd5 output has {[len(pl) for pl in planes]} entries, expected {n}")
    missing = [poc for poc in range(n) if poc not in log_md5]
    if missing:
        raise DecodeError(f"QP{qp}: encoder log has no MD5 for {len(missing)} POCs (first {missing[:5]})")
    bad = [poc for poc in range(n)
           if (planes[0][poc], planes[1][poc], planes[2][poc]) != log_md5[poc]]
    if bad:
        raise DecodeError(f"QP{qp}: decoded frames do not match encoder log MD5 for {len(bad)} of {n} "
                          f"frames (first POCs {bad[:5]})")
    return f"ok ({n}/{n} frames match encoder log)"


def _run_with_progress(cmd: list[str], out_dir: Path, prefix: str, expected: int | None,
                       timeout_s: float, label: str, poll_failure=None) -> None:
    """Run ffmpeg with progress logs and a timeout; poll_failure() -> message stops it when a helper decoder
    process has failed (else ffmpeg could wait forever on the pipe)."""
    t0 = time.time()
    last = t0
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        while True:
            try:
                proc.wait(timeout=2)
                break
            except subprocess.TimeoutExpired:
                pass
            failure = poll_failure() if poll_failure else None
            if failure:
                proc.kill()
                proc.wait()
                raise DecodeError(f"{label}: {failure}")
            now = time.time()
            if now - t0 > timeout_s:
                proc.kill()
                proc.wait()
                raise DecodeError(f"{label}: decoding timed out after {timeout_s} s")
            if now - last >= 10:
                n = sum(1 for p in out_dir.iterdir() if p.name.startswith(prefix))
                log.info(f"{label}: decoded {n}{'/' + str(expected) if expected else ''} frames "
                         f"({now - t0:.0f} s)")
                last = now
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    err = proc.stderr.read() if proc.stderr else ""
    if proc.returncode != 0:
        raise DecodeError(f"{label}: ffmpeg failed (exit {proc.returncode}): {err.strip()[-1000:]}")
    if err.strip():
        log.warning(f"{label}: ffmpeg stderr: {err.strip()[-500:]}")


def remove_frames(item: QPItem) -> int:
    """Delete PNGs that this tool decoded (never user-provided ones). Returns number removed."""
    if item.bitstream is None or _load_marker(item) is None:
        return 0
    n = len(item.pngs)
    _delete_pngs(item)
    return n
