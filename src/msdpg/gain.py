"""Label-wise directional Gini gains."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.tree import _tree

from .predicates import Predicate, predicate_key


def positive_index(classes: np.ndarray) -> int | None:
    """Return the class-array index of positive class 1, if present."""
    found = np.flatnonzero(np.asarray(classes) == 1)
    return int(found[0]) if found.size else None


def _gini(probabilities: np.ndarray) -> float:
    return float(1.0 - np.dot(probabilities, probabilities))


def directional_gains(
    estimator,
    label: int,
    classes: np.ndarray,
    matching: str,
    edges: list[np.ndarray] | None = None,
) -> dict[Predicate, float]:
    """Aggregate ``w_u Delta G_l(s_u)`` to the child with higher positive fraction."""
    tree = estimator.tree_
    pos = positive_index(classes)
    output: defaultdict[Predicate, float] = defaultdict(float)
    if pos is None:
        return {}
    weights = tree.weighted_n_node_samples
    for node, left_node in enumerate(tree.children_left):
        if left_node == _tree.TREE_LEAF:
            continue
        left, right = int(left_node), int(tree.children_right[node])
        gain = (
            weights[node] * _gini(tree.value[node, label, :])
            - weights[left] * _gini(tree.value[left, label, :])
            - weights[right] * _gini(tree.value[right, label, :])
        )
        if gain <= 0.0:
            continue
        left_p, right_p = (
            float(tree.value[left, label, pos]),
            float(tree.value[right, label, pos]),
        )
        feature, threshold = int(tree.feature[node]), float(tree.threshold[node])
        if left_p == right_p:
            output[predicate_key(feature, threshold, "<=", matching, edges)] += (
                gain / 2.0
            )
            output[predicate_key(feature, threshold, ">", matching, edges)] += (
                gain / 2.0
            )
        else:
            direction = "<=" if left_p > right_p else ">"
            output[predicate_key(feature, threshold, direction, matching, edges)] += (
                gain
            )
    return dict(output)
