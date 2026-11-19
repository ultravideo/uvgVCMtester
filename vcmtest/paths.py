"""Discovery of sequences, QP folders, bitstreams/PNGs, annotations and tools.

Layout (docs/folder_layout.txt):
  <data_root>/<sequence_folder>/QPxx/       bitstream and/or <name>_NNN(N).png, optional encoder log
  <data_root>/<sequence_folder>/info.json   optional, key "framerate"
  <annotations_root>/<sequence_name>/       annotations; sequence_name = sequence folder name up to the first "_"
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .config import ConfigError, resolve_path

log = logging.getLogger("vcmtest")

BITSTREAM_EXTS = {".bin", ".bit", ".266", ".vvc", ".h266", ".265", ".hevc", ".h265"}
PNG_RE = re.compile(r"^(?P<stem>.+?)_(?P<num>\d{3,4})\.png$")
QP_DIR_RE = re.compile(r"^QP(\d+)$")
MIN_FFMPEG = (7, 1)


class DiscoveryError(RuntimeError):
    """Raised when the input data layout (sequences, QP folders, annotations) is invalid."""
    pass


def sequence_name(sequence_folder: str) -> str:
    """'VolleyballGame_1920x1080_444_LD' -> 'VolleyballGame'."""
    return sequence_folder.split("_", 1)[0]


# ----------------------------------------------------------------------------- QP folders

@dataclass
class QPItem:
    """One QPxx folder: its bitstream, encoder log and PNG frames (each may be missing)."""
    qp: int
    dir: Path
    bitstream: Path | None
    log_file: Path | None
    pngs: dict[int, Path]  # frame number (1-based) -> path

    @property
    def has_pngs(self) -> bool:
        """True if the folder contains PNG frames."""
        return bool(self.pngs)


def _find_log(bitstream: Path) -> Path | None:
    for cand in (bitstream.with_name(bitstream.name + ".log"), bitstream.with_suffix(".log")):
        if cand.is_file():
            return cand
    logs = sorted(bitstream.parent.glob("*.log"))
    return logs[0] if len(logs) == 1 else None


def scan_pngs(folder: Path) -> dict[int, Path]:
    """Return the frame number (1-based) -> path of the <stem>_NNN(N).png files in a folder.

    Raises:
        DiscoveryError: two PNGs for the same frame number, or more than one name stem.
    """
    frames: dict[int, Path] = {}
    stems = set()
    for p in folder.iterdir():
        m = PNG_RE.match(p.name)
        if m and p.is_file():
            n = int(m["num"])
            if n in frames:
                raise DiscoveryError(f"{folder}: duplicate PNG for frame {n}: {frames[n].name}, {p.name}")
            frames[n] = p
            stems.add(m["stem"])
    if len(stems) > 1:
        raise DiscoveryError(f"{folder}: PNGs with several name stems {sorted(stems)}")
    return dict(sorted(frames.items()))


def scan_qp_dir(qp: int, folder: Path) -> QPItem:
    """Scan one QP folder for its bitstream, encoder log and PNG frames.

    Raises:
        DiscoveryError: more than one bitstream, or neither a bitstream nor PNG frames.
    """
    bins = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in BITSTREAM_EXTS)
    if len(bins) > 1:
        raise DiscoveryError(f"{folder}: more than one bitstream: {[b.name for b in bins]}")
    bitstream = bins[0] if bins else None
    pngs = scan_pngs(folder)
    if bitstream is None and not pngs:
        raise DiscoveryError(f"{folder}: no bitstream and no PNG frames")
    log_file = _find_log(bitstream) if bitstream else None
    return QPItem(qp=qp, dir=folder, bitstream=bitstream, log_file=log_file, pngs=pngs)


def qp_dirs(seq_dir: Path) -> dict[int, Path]:
    """Return QP number -> folder for the QPxx sub-folders of a sequence folder."""
    found = {}
    for p in seq_dir.iterdir():
        m = QP_DIR_RE.match(p.name)
        if m and p.is_dir():
            found[int(m.group(1))] = p
    return found


def discover_qps(seq_dir: Path, selection) -> list[QPItem]:
    """Scan the selected QP folders ('all' or a list of QP numbers) of a sequence, sorted by QP.

    Raises:
        DiscoveryError: a requested QP folder is missing, or the sequence has no QP folders.
    """
    found = qp_dirs(seq_dir)
    if selection == "all":
        wanted = sorted(found)
    else:
        missing = sorted(set(selection) - set(found))
        if missing:
            raise DiscoveryError(f"{seq_dir.name}: requested QPs not found: {missing}; available {sorted(found)}")
        wanted = sorted(selection)
    if not wanted:
        raise DiscoveryError(f"{seq_dir.name}: no QPxx folders")
    return [scan_qp_dir(q, found[q]) for q in wanted]


def discover_sequences(data_root: Path, selection) -> list[Path]:
    """Return the selected sequence folders ('all' or a list of names) that contain QPxx sub-folders.

    Raises:
        ConfigError: data_root is missing, or a selected sequence is missing or has no QPxx sub-folders.
    """
    if not data_root.is_dir():
        raise ConfigError(f"data_root does not exist: {data_root}")
    all_seqs = sorted(p for p in data_root.iterdir() if p.is_dir() and qp_dirs(p))
    if selection == "all":
        return all_seqs
    by_name = {p.name: p for p in all_seqs}
    missing = [s for s in selection if s not in by_name]
    no_qp = [s for s in missing if (data_root / s).is_dir()]
    if no_qp:
        raise ConfigError(f"Sequence folder(s) without QPxx/ sub-folders: {no_qp} "
                          f"(expected {data_root}/<sequence_folder>/QPxx/)")
    if missing:
        raise ConfigError(f"Sequences not found in {data_root}: {missing}. Available: {sorted(by_name)}")
    return [by_name[s] for s in selection]


def bitstream_name_warnings(seq_folder: str, qps: list[QPItem]) -> list[str]:
    """Return warnings for bitstreams whose name does not start with the sequence name."""
    rn = sequence_name(seq_folder)
    bad: dict[str, list[int]] = {}
    for it in qps:
        if it.bitstream and not it.bitstream.stem.startswith(rn):
            bad.setdefault(it.bitstream.name, []).append(it.qp)
    return [f"Bitstream name '{name}' (QP {', '.join(map(str, q))}) does not start with "
            f"sequence name '{rn}'" for name, q in bad.items()]


# ----------------------------------------------------------------------------- annotations

@dataclass
class AnnotationInventory:
    """Annotation files and folders found for one sequence (None = not present)."""
    folder: Path
    name: str
    detection_json: Path | None = None
    masks_dir: Path | None = None
    pose_json: Path | None = None
    panoptic_json: Path | None = None
    panoptic_masks_dir: Path | None = None
    depth_dir: Path | None = None
    obb_json: Path | None = None
    lpr_json: Path | None = None
    bib_json: Path | None = None
    unknown: list[Path] = field(default_factory=list)

    def has(self, ann_type: str) -> bool:
        """True if the annotation type (detection, pose, panoptic, depth, obb, lpr, bib) is present."""
        return {
            "detection": self.detection_json,
            "pose": self.pose_json,
            "panoptic": self.panoptic_json,
            "depth": self.depth_dir,
            "obb": self.obb_json,
            "lpr": self.lpr_json,
            "bib": self.bib_json,
        }[ann_type] is not None

    def json_for(self, ann_type: str) -> Path | None:
        """Return the annotation JSON of a type, or None if missing or the type has no JSON (depth)."""
        return {"detection": self.detection_json, "pose": self.pose_json,
                "panoptic": self.panoptic_json, "obb": self.obb_json, "lpr": self.lpr_json,
                "bib": self.bib_json}.get(ann_type)

    def types(self) -> list[str]:
        """Return the annotation types present."""
        return [t for t in ("detection", "pose", "panoptic", "depth", "obb", "lpr", "bib") if self.has(t)]

    def as_dict(self) -> dict:
        """Return the inventory as a JSON-serialisable dict (paths as strings)."""
        f = lambda p: str(p) if p else None  # noqa: E731
        return {"folder": str(self.folder), "detection_json": f(self.detection_json),
                "masks_dir": f(self.masks_dir), "pose_json": f(self.pose_json),
                "panoptic_json": f(self.panoptic_json),
                "panoptic_masks_dir": f(self.panoptic_masks_dir), "depth_dir": f(self.depth_dir),
                "obb_json": f(self.obb_json), "lpr_json": f(self.lpr_json),
                "bib_json": f(self.bib_json),
                "unknown": [str(p) for p in self.unknown]}


def find_annotations(annotations_root: Path, seq_folder: str, override: str | None = None) -> AnnotationInventory:
    """Find the annotation files of a sequence in <annotations_root>/<sequence_name or override>/.

    Files and folders with unexpected names are listed in `unknown`.

    Raises:
        DiscoveryError: the annotation folder does not exist.
    """
    name = override or sequence_name(seq_folder)
    folder = annotations_root / name
    if not folder.is_dir():
        raise DiscoveryError(f"Sequence '{seq_folder}': annotation folder not found: {folder} "
                             f"(set sequence_overrides.{seq_folder}.annotations to use another sub-folder)")
    inv = AnnotationInventory(folder=folder, name=name)
    known = {
        f"{name}_annotations.json": "detection_json",
        f"{name}_pose_annotations.json": "pose_json",
        f"{name}_panoptic_annotations.json": "panoptic_json",
        f"{name}_obb_annotations.json": "obb_json",
        f"{name}_lpr_annotations.json": "lpr_json",
        f"{name}_bib_annotations.json": "bib_json",
    }
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.name in known:
            setattr(inv, known[p.name], p)
        elif p.is_file() and p.suffix.lower() == ".json":
            inv.unknown.append(p)
        elif p.is_dir() and p.name == f"{name}_masks":
            inv.masks_dir = p
        elif p.is_dir() and p.name == f"{name}_panoptic_masks":
            inv.panoptic_masks_dir = p
        elif p.is_dir() and p.name == f"{name}_depth":
            inv.depth_dir = p
        elif p.is_dir():
            inv.unknown.append(p)
    return inv


# ----------------------------------------------------------------------------- frame rate

def read_encoder_log_info(log_file: Path | None) -> dict:
    """Frame rate and frame count as reported by a VTM/HM encoder log (best effort)."""
    info: dict = {}
    if not log_file or not log_file.is_file():
        return info
    with open(log_file, errors="replace") as f:
        for line in f:
            if "fps" not in info:
                m = re.match(r"\s*(?:Real|Input)\s+Format\s*:\s*\d+x\d+\s+([\d.]+)Hz", line)
                if m:
                    info["fps"] = float(m.group(1))
            if "frames" not in info:
                m = re.match(r"\s*Frame index\s*:\s*\d+\s*-\s*\d+\s*\((\d+) frames\)", line)
                if m:
                    info["frames"] = int(m.group(1))
            if "fps" in info and "frames" in info:
                break
    return info


def resolve_framerate(seq_dir: Path, cfg: dict, qps: list[QPItem]) -> tuple[float, str, list[str]]:
    """Returns (fps, source, warnings). info.json > sequence_overrides > config 'framerate'."""
    warnings = []
    fps, source = None, None
    info_path = seq_dir / "info.json"
    if info_path.is_file():
        with open(info_path) as f:
            info = json.load(f)
        if "framerate" in info:
            fps, source = float(info["framerate"]), str(info_path)
    if fps is None:
        ov = (cfg.get("sequence_overrides") or {}).get(seq_dir.name, {})
        if ov.get("framerate") is not None:
            fps, source = float(ov["framerate"]), f"config sequence_overrides.{seq_dir.name}.framerate"
        elif cfg.get("framerate") is not None:
            fps, source = float(cfg["framerate"]), "config framerate"
    if fps is None:
        raise DiscoveryError(f"Sequence '{seq_dir.name}': no frame rate (no info.json 'framerate' and "
                             f"no 'framerate' in config)")
    for it in qps:
        li = read_encoder_log_info(it.log_file)
        if "fps" in li and abs(li["fps"] - fps) > 1e-6:
            warnings.append(f"QP{it.qp}: encoder log frame rate {li['fps']} Hz differs from {fps} ({source})")
    return fps, source, warnings


# ----------------------------------------------------------------------------- tools

@dataclass
class FFmpegInfo:
    """ffmpeg executable, its version and the names of its available decoders."""
    path: str
    version_line: str
    version: tuple[int, ...] | None
    decoders: set[str]


def resolve_ffmpeg(cfg: dict) -> str:
    """Return the ffmpeg executable: config 'ffmpeg' (a file in tools_dir) or ffmpeg on PATH.

    Raises:
        ConfigError: the executable is not found.
    """
    name = cfg.get("ffmpeg")
    if name:
        p = resolve_path(cfg["tools_dir"]) / name
        if not p.is_file():
            raise ConfigError(f"ffmpeg executable '{name}' not found in tools dir: {p} (place a static ffmpeg >= 7.1 "
                              f"and ffprobe there, or set 'ffmpeg: null' to use the ffmpeg on PATH)")
        return str(p)
    found = shutil.which("ffmpeg")
    if not found:
        raise ConfigError("ffmpeg not found on PATH (set 'ffmpeg: <name>' to use one from tools/)")
    return found


def probe_ffmpeg(path: str, timeout: float = 30) -> FFmpegInfo:
    """Run ffmpeg -version and -decoders and return the parsed version and decoder names."""
    out = subprocess.run([path, "-hide_banner", "-version"], capture_output=True, text=True,
                         timeout=timeout, check=True).stdout
    line = out.splitlines()[0] if out else ""
    m = re.search(r"ffmpeg version n?(\d+)\.(\d+)(?:\.(\d+))?", line)
    version = tuple(int(g) for g in m.groups() if g is not None) if m else None
    dec_out = subprocess.run([path, "-hide_banner", "-decoders"], capture_output=True, text=True,
                             timeout=timeout, check=True).stdout
    decoders = set()
    for ln in dec_out.splitlines():
        parts = ln.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VASDT.":
            decoders.add(parts[1])
    return FFmpegInfo(path=path, version_line=line, version=version, decoders=decoders)


def check_ffmpeg(info: FFmpegInfo, needed: set[str]) -> list[str]:
    """Raises if a needed decoder is missing; returns warnings."""
    warnings = []
    if info.version is None:
        warnings.append(f"Could not parse ffmpeg version from '{info.version_line}' (need >= 7.1)")
    elif info.version[:2] < MIN_FFMPEG:
        raise ConfigError(f"{info.path}: ffmpeg {'.'.join(map(str, info.version))} is too old, need >= 7.1")
    missing = needed - info.decoders
    if missing:
        raise ConfigError(f"{info.path}: missing decoder(s) {sorted(missing)}")
    return warnings


def codec_for(bitstream: Path) -> str:
    """Return the ffmpeg codec name for a bitstream extension: hevc, vvc, or auto (.bin / .bit)."""
    ext = bitstream.suffix.lower()
    if ext in {".265", ".hevc", ".h265"}:
        return "hevc"
    if ext in {".266", ".vvc", ".h266"}:
        return "vvc"
    return "auto"  # .bin/.bit: let ffprobe decide at decode time
