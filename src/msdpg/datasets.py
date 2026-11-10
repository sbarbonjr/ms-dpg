"""Small deterministic diagnostic datasets."""

from __future__ import annotations

import numpy as np


def make_scenario(
    name: str, random_state: int = 0, n_samples: int = 1500, n_features: int = 50
):
    """Generate a compact D1--D5-style multi-label diagnostic dataset."""
    rng = np.random.default_rng(random_state)
    X = rng.normal(size=(n_samples, n_features))
    Y = np.zeros((n_samples, 4), dtype=int)
    if name in {"D1", "G1"}:
        Y[:, 0] = (X[:, 0] > 0) & (X[:, 1] > 0)
        Y[:, 1] = (X[:, 0] > 0) & (X[:, 2] > 0)
    elif name in {"D2", "G2"}:
        X[:, 3] = 0.9 * X[:, 1] + np.sqrt(0.19) * rng.normal(size=n_samples)
        Y[:, 0] = X[:, 1] > 0
        Y[:, 1] = X[:, 3] > 0
    elif name in {"D3", "G3"}:
        Y[:, 0] = X[:, 4] > 0
        Y[:, 1] = X[:, 5] > 0
    elif name in {"D4", "G4"}:
        Y[:, 0] = X[:, 6] > 0
        Y[:, 1] = Y[:, 0] & (X[:, 7] > 0)
    elif name in {"D5", "G5"}:
        Y[:, 0] = X[:, 8] > 0.5
        Y[:, 1] = X[:, 8] < -0.5
    else:
        raise ValueError("name must be D1, D2, D3, D4, or D5")
    Y[:, 2] = X[:, 10] > 0
    Y[:, 3] = X[:, 11] > 0.3
    return X, Y.astype(int)
