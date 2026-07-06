"""
Global seeding for reproducible runs.
=====================================

Phase 0 of the remediation plan: nothing is re-measured until runs are
deterministic, so every later number is trustworthy and re-runnable.

`set_global_seed(seed)` seeds every RNG the project touches:
  - Python's `random`            (used by the SAC ReplayBuffer.sample)
  - NumPy's legacy global RNG    (used across data/metrics code)
  - PyTorch CPU + CUDA RNGs      (network init, Dirichlet sampling)
  - PYTHONHASHSEED               (hash-ordering determinism)
  - cuDNN / deterministic algos  (kernel-level determinism)

The Gymnasium environment is seeded separately via `env.reset(seed=...)`,
which is threaded from the CLI through the trainer.
"""

from __future__ import annotations

import os
import random

import numpy as np

# ── Single-thread pinning for CPU determinism (Phase 5, Task C, §12) ──────────
# Two identical-seed runs previously produced different per-seed results. On a
# CPU-only machine the dominant cause is multi-threaded intra-op parallelism:
# floating-point reductions are summed in a nondeterministic order across
# OpenMP/MKL threads. These env vars must be set *before* the numerical
# backends (torch / numpy-MKL / OpenMP) spin up their thread pools, so they are
# assigned at import time of this module. Entry points import `utils.seeding`
# up front, and `torch.set_num_threads(1)` in `set_global_seed` enforces the
# same at runtime as a belt-and-braces measure regardless of import order.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")


def set_global_seed(seed: int, deterministic: bool = True) -> int:
    """
    Seed all RNGs used in the project for reproducible runs.

    Parameters
    ----------
    seed : int
        The seed applied to random / numpy / torch.
    deterministic : bool, default True
        If True, force deterministic algorithms and cuDNN determinism.
        Set False only if a CUDA op without a deterministic implementation
        raises and you accept the loss of bit-for-bit reproducibility.

    Returns
    -------
    int
        The seed that was set (echoed for logging/stamping).
    """
    # Must be set before hash-randomised structures are built.
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Pin intra-op (and, best-effort, inter-op) threads to 1 so CPU float
    # reductions have a fixed summation order (Phase 5, Task C). This is the
    # runtime counterpart to the env vars set at module import above and takes
    # effect even if torch was imported before this module.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Can only be set once, before any parallel work has started; ignore if
        # torch has already launched its inter-op pool.
        pass

    if deterministic:
        # cuDNN determinism (no-op on CPU-only machines).
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Required for some CUDA GEMM kernels to behave deterministically.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Older torch builds may not support warn_only / the call at all.
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass

    return seed
