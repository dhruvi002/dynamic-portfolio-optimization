"""
RunningNormalizer — online z-score normalizer with train/eval freeze.

Only updates running stats while in training mode. Freezing at eval time
prevents look-ahead bias: test-set observations must not shift the normalizer
statistics that were computed on training data.

The first n_skip dimensions (portfolio weights) are passed through unchanged
since they are already bounded on the simplex and normalizing them is harmful.
"""

import numpy as np


class RunningNormalizer:

    def __init__(self, size: int, n_skip: int = 0, epsilon: float = 1e-8):
        self.size    = size
        self.n_skip  = n_skip
        self.epsilon = epsilon
        self.training = True

        # Welford online algorithm state
        self.count = 0
        self.mean  = np.zeros(size, dtype=np.float64)
        self.M2    = np.ones(size,  dtype=np.float64)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def update(self, x: np.ndarray):
        """Update running stats. No-op when in eval mode."""
        if not self.training:
            return
        x = np.asarray(x, dtype=np.float64).ravel()
        self.count += 1
        delta       = x - self.mean
        self.mean  += delta / self.count
        self.M2    += delta * (x - self.mean)   # Welford update

    def normalize(self, x: np.ndarray) -> np.ndarray:
        out = np.asarray(x, dtype=np.float32).copy()
        if self.count < 2:
            return out
        var = self.M2 / (self.count - 1)
        std = np.sqrt(var + self.epsilon).astype(np.float32)
        mu  = self.mean.astype(np.float32)
        out[self.n_skip:] = (out[self.n_skip:] - mu[self.n_skip:]) / std[self.n_skip:]
        return out

    def state_dict(self) -> dict:
        return {
            "count":  self.count,
            "mean":   self.mean.copy(),
            "M2":     self.M2.copy(),
            "n_skip": self.n_skip,
            "size":   self.size,
        }

    def load_state_dict(self, d: dict):
        self.count  = d["count"]
        self.mean   = d["mean"].copy()
        self.M2     = d["M2"].copy()
        self.n_skip = d["n_skip"]
        self.size   = d["size"]
