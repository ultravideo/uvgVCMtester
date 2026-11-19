"""Bitstream decoders (config decode.decoder): how a bitstream becomes the YUV frames that ffmpeg converts to PNG.

- ffmpeg (default): ffmpeg decodes the bitstream itself (its native vvc / hevc decoders).
- vtm: the VVC reference decoder (VTM DecoderApp, executable in tools/). It writes raw YUV into a named pipe (FIFO);
  ffmpeg reads the pipe as rawvideo and does the same YUV -> RGB PNG conversion and MD5 check as for its own
  decoding. No YUV file is written to disk.

The stream properties (codec, size, pixel format) always come from ffprobe, so the bitstream must be a standard
stream that ffprobe recognises. ffmpeg / ffprobe are therefore needed with every decoder.

Adding a decoder that writes YUV (the usual case for reference / research decoders): subclass YuvPipeDecoder, set
`name`, `codecs` and `default_executable`, implement `command()` (and `version()` if `--help` does not print it),
and add the class to DECODERS. Its output must be planar YUV in the stream's pixel format (ffprobe pix_fmt; 8-bit
samples as bytes, > 8-bit as 16-bit little endian), frames in output (display) order. See VTMDecoder.

A decoder other than ffmpeg is part of the decode settings (`"decoder": <name>`), so switching decoders decodes the
frames again and recomputes predictions (a non-conforming or modified decoder may give other frames).
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

from .config import ConfigError, resolve_path


class Decoder:
    """Base: provides ffmpeg input arguments for one bitstream; may run a helper process meanwhile."""

    name = ""
    codecs: set[str] = set()          # ffprobe codec names this decoder accepts (empty = any)

    def __init__(self, cfg: dict):
        self.cfg = cfg

    @property
    def version(self) -> str | None:
        """Decoder version string recorded in the decode marker, or None."""
        return None

    def check_stream(self, stream: dict, bitstream: Path) -> None:
        """Raise DecoderError if the stream's ffprobe codec is not one this decoder accepts."""
        codec = stream.get("codec_name")
        if self.codecs and codec not in self.codecs:
            raise DecoderError(f"{bitstream.name}: decoder '{self.name}' decodes {sorted(self.codecs)}, "
                               f"ffprobe reports '{codec}'")

    def start(self, bitstream: Path, stream: dict, tmp: Path) -> list[str]:
        """Start decoding; return the ffmpeg input arguments (before the filters / outputs)."""
        raise NotImplementedError

    def poll_failure(self) -> str | None:
        """While ffmpeg runs: an error message if the helper process has failed, else None."""
        return None

    def finish(self, ok: bool, timeout: float = 120) -> None:
        """After ffmpeg: stop / check the helper process; raise DecoderError on failure (if ok)."""


class DecoderError(RuntimeError):
    """Raised when a bitstream decoder fails or cannot decode a stream."""
    pass


class FFmpegDecoder(Decoder):
    """ffmpeg's own decoder (vvc, hevc, ...)."""

    name = "ffmpeg"

    def start(self, bitstream: Path, stream: dict, tmp: Path) -> list[str]:
        """Return ffmpeg input arguments that decode the bitstream with ffmpeg itself."""
        return ["-threads", "0", "-i", str(bitstream)]


class YuvPipeDecoder(Decoder):
    """An external decoder executable that writes raw planar YUV to a file: given a FIFO instead of a file."""

    default_executable = ""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.executable = resolve_decoder_executable(cfg, self.default_executable)
        self.proc: subprocess.Popen | None = None
        self.log_path: Path | None = None
        self._version: str | None = None

    def command(self, bitstream: Path, yuv_out: Path) -> list[str]:
        """Return the decoder command line that decodes the bitstream into the YUV file (the FIFO)."""
        raise NotImplementedError

    def version_command(self) -> list[str]:
        """Return the command whose output contains the decoder version."""
        return [self.executable, "--help"]

    def parse_version(self, text: str) -> str | None:
        """Return the first output line that contains 'version', or None."""
        for line in text.splitlines():
            if "version" in line.lower():
                return line.strip()
        return None

    @property
    def version(self) -> str | None:
        """Decoder version, from the version command output (cached; 'unknown' if not found).

        Raises:
            ConfigError: the executable cannot be run.
        """
        if self._version is None:
            try:
                r = subprocess.run(self.version_command(), capture_output=True, text=True, timeout=30)
                self._version = self.parse_version(r.stdout + r.stderr) or "unknown"
            except (OSError, subprocess.TimeoutExpired) as e:
                raise ConfigError(f"decoder '{self.name}': cannot run {self.executable}: {e}") from None
        return self._version

    def start(self, bitstream: Path, stream: dict, tmp: Path) -> list[str]:
        """Create the FIFO, start the decoder writing into it and return ffmpeg rawvideo input arguments.

        The decoder output goes to <tmp>/<name>_decoder.log.

        Raises:
            DecoderError: ffprobe gave no frame size or pixel format.
        """
        w, h, pix = stream.get("width"), stream.get("height"), stream.get("pix_fmt")
        if not (w and h and pix):
            raise DecoderError(f"{bitstream.name}: ffprobe gives no size / pixel format; needed to read the "
                               f"'{self.name}' decoder's raw YUV output")
        fifo = tmp / "decoded.yuv"
        os.mkfifo(fifo)
        self.log_path = tmp / f"{self.name}_decoder.log"
        logf = open(self.log_path, "w")
        try:
            self.proc = subprocess.Popen(self.command(bitstream, fifo), stdout=logf, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL)
        finally:
            logf.close()
        return ["-f", "rawvideo", "-pix_fmt", pix, "-s:v", f"{w}x{h}", "-i", str(fifo)]

    def _log_tail(self) -> str:
        try:
            return self.log_path.read_text(errors="replace").strip()[-800:] if self.log_path else ""
        except OSError:
            return ""

    def poll_failure(self) -> str | None:
        """Return an error message with the decoder log tail if the decoder exited with an error, else None."""
        if self.proc is not None and self.proc.poll() not in (None, 0):
            return f"decoder '{self.name}' failed (exit {self.proc.returncode}): {self._log_tail()}"
        return None

    def finish(self, ok: bool, timeout: float = 120) -> None:
        """Wait for the decoder process to end (killed after 1 s if ffmpeg failed).

        Raises:
            DecoderError: with ok, the decoder did not finish within the timeout or exited with an error.
        """
        p, self.proc = self.proc, None
        if p is None:
            return
        try:
            p.wait(timeout=timeout if ok else 1)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
            if ok:
                raise DecoderError(f"decoder '{self.name}' did not finish after ffmpeg ended") from None
        if ok and p.returncode != 0:
            raise DecoderError(f"decoder '{self.name}' failed (exit {p.returncode}): {self._log_tail()}")


class VTMDecoder(YuvPipeDecoder):
    """VVC reference decoder (VTM DecoderApp): -b bitstream -o YUV, output at the native bit depth (-d 0)."""

    name = "vtm"
    codecs = {"vvc"}
    default_executable = "DecoderAppStatic"

    def command(self, bitstream: Path, yuv_out: Path) -> list[str]:
        """Return the VTM DecoderApp command line."""
        return [self.executable, "-b", str(bitstream), "-o", str(yuv_out), "-d", "0"]

    def parse_version(self, text: str) -> str | None:
        """Return 'VTM <version>' from the DecoderApp output, else the generic version line."""
        m = re.search(r"VTM Decoder Version\s+(\S+)", text)
        return f"VTM {m.group(1)}" if m else super().parse_version(text)


DECODERS: dict[str, type[Decoder]] = {"ffmpeg": FFmpegDecoder, "vtm": VTMDecoder}


def resolve_decoder_executable(cfg: dict, default_name: str) -> str:
    """decode.decoder_executable (or the decoder's default name): a file in tools_dir, a path, or on PATH."""
    name = cfg["decode"].get("decoder_executable") or default_name
    for p in (resolve_path(cfg["tools_dir"]) / name, resolve_path(name)):
        if p.is_file():
            if not os.access(p, os.X_OK):
                raise ConfigError(f"decoder executable {p} is not executable (chmod +x)")
            return str(p)
    found = shutil.which(name)
    if found:
        return found
    raise ConfigError(f"decoder '{cfg['decode'].get('decoder')}': executable '{name}' not found in "
                      f"{resolve_path(cfg['tools_dir'])}, as a path, or on PATH (decode.decoder_executable)")


_cache: dict = {}


def get_decoder(cfg: dict) -> Decoder:
    """The configured decoder (one instance per name and executable setting)."""
    d = cfg["decode"]
    name = d.get("decoder", "ffmpeg")
    key = (name, d.get("decoder_executable"), cfg.get("tools_dir"))
    if key not in _cache:
        if name not in DECODERS:
            raise ConfigError(f"decode.decoder must be one of {sorted(DECODERS)}")
        _cache[key] = DECODERS[name](cfg)
    return _cache[key]
