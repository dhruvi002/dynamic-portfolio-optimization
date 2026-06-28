"""
Run stamping for reproducibility.
=================================

Phase 0 acceptance criterion: every artifact directory contains a
`run_meta.json` capturing the seed, git commit, and full config so any
reported number is attributable to an exact configuration and code state.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

# Libraries whose versions materially affect numerical results.
_TRACKED_PACKAGES = [
    "torch", "numpy", "pandas", "gymnasium", "scikit-learn",
    "yfinance", "finrl", "ray", "transformers", "ta",
]


def _git_sha() -> Optional[str]:
    """Return the current commit SHA (with -dirty suffix if uncommitted)."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"],
            stderr=subprocess.DEVNULL,
        ) != 0
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return None


def _library_versions() -> dict:
    versions = {}
    for pkg in _TRACKED_PACKAGES:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    return versions


def build_run_meta(seed: int, config: dict, device: str = "", **extra) -> dict:
    """Assemble the run-metadata dict (does not write to disk)."""
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "seed": seed,
        "config": config,
        "device": str(device),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "library_versions": _library_versions(),
    }
    meta.update(extra)
    return meta


def write_run_meta(out_dir: str, seed: int, config: dict,
                   device: str = "", **extra) -> str:
    """
    Write `run_meta.json` into `out_dir` and return its path.

    Pass any extra run-scoped facts as keyword args (e.g. mode="train",
    episodes=500, encoder="mlp", sentiment=False).
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = build_run_meta(seed, config, device=device, **extra)
    path = os.path.join(out_dir, "run_meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path
