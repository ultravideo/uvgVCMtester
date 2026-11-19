"""Run environment: versions, seeds, logging setup."""

from __future__ import annotations

import logging
import platform
import random
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

from . import __version__
from .config import PROJECT_ROOT

log = logging.getLogger("vcmtest")
_FMT = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%Y-%m-%d %H:%M:%S")


def timestamp() -> str:
    """Current local time as YYYYmmdd-HHMMSS (log file names)."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the vcmtest logger with a stdout handler at the given level (added only once)."""
    log.setLevel(logging.DEBUG)
    log.propagate = False
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in log.handlers):
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(_FMT)
        h.setLevel(level)
        log.addHandler(h)


def add_file_log(path: Path) -> logging.FileHandler:
    """Add a DEBUG-level file handler writing to path (parent folders created) and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h = logging.FileHandler(path)
    h.setFormatter(_FMT)
    h.setLevel(logging.DEBUG)
    log.addHandler(h)
    return h


def remove_handler(h: logging.Handler) -> None:
    """Detach a handler from the vcmtest logger and close it."""
    log.removeHandler(h)
    h.close()


def set_seeds(seed: int) -> None:
    """Seed Python's random, NumPy and torch (CPU and CUDA) where installed."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _pkg(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    """Short HEAD commit of the project repository, with '-dirty' if tracked files are modified; None on failure."""
    try:
        out = subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(PROJECT_ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "") if out.returncode == 0 else None
    except Exception:
        return None


def environment_info(ffmpeg_version_line: str | None = None) -> dict:
    """Versions of vcmtest, git, Python, the platform, key packages and ffmpeg.

    Plus CUDA, cuDNN and GPU name when torch is installed.
    """
    info = {
        "vcmtest": __version__,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: _pkg(p) for p in ("torch", "torchvision", "ultralytics", "transformers",
                                          "pycocotools", "numpy", "pillow")},
        "ffmpeg": ffmpeg_version_line,
    }
    try:
        import torch
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    return info
