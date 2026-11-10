"""Directional predicate keys and matching rules."""

from __future__ import annotations
from typing import TypeAlias
import numpy as np

Predicate: TypeAlias = tuple[int, float | int, str]


def quantile_edges(X: np.ndarray, bins: int = 32) -> list[np.ndarray]:
    """Return feature-wise quantile edges computed from the training matrix."""
    if not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional training matrix")
    return [
        np.quantile(X[:, j], np.linspace(0.0, 1.0, bins + 1)) for j in range(X.shape[1])
    ]


def predicate_key(
    feature: int,
    threshold: float,
    direction: str,
    matching: str,
    edges: list[np.ndarray] | None = None,
) -> Predicate:
    """Return ``(feature, threshold-or-bin, direction)`` under a matching rule."""
    if direction not in ("<=", ">"):
        raise ValueError("direction must be '<=' or '>'")
    if matching in ("exact", "round6"):
        return (int(feature), round(float(threshold), 6), direction)
    if matching in ("quantile", "q16", "q32") and edges is not None:
        return (
            int(feature),
            int(np.searchsorted(edges[feature], threshold)),
            direction,
        )
    raise ValueError("quantile matching requires training quantile edges")
