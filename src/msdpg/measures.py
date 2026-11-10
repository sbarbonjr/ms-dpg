"""Shared Predicate Flow and related structural measures."""

from __future__ import annotations
import numpy as np


def active_predicates(
    gains: dict, top_percent: float = 25, minimum_fraction: float = 0.05
) -> set:
    """Return positive-gain predicates meeting the percentile and relative-gain cutoffs."""
    positive = {key: value for key, value in gains.items() if value > 0}
    if not positive:
        return set()
    threshold = np.percentile(list(positive.values()), 100 - top_percent)
    maximum = max(positive.values())
    return {
        key
        for key, value in positive.items()
        if value >= threshold and value >= minimum_fraction * maximum
    }


def spf(
    records: list[dict],
    gains: list[dict],
    source: int,
    target: int,
    top_percent: float = 25,
    minimum_fraction: float = 0.05,
) -> float:
    """Compute SPF(A->B), the target mass on traces with a shared active predicate."""
    shared = active_predicates(gains[source], top_percent, minimum_fraction)
    shared &= active_predicates(gains[target], top_percent, minimum_fraction)
    target_records = [r for r in records if r["label"] == target]
    denominator = sum(r["mass"] for r in target_records)
    if not denominator or not shared:
        return 0.0
    return float(
        sum(r["mass"] for r in target_records if set(r["predicates"]) & shared)
        / denominator
    )


def classify_lifts(source: float, target: float) -> str:
    """Classify predicate enrichment across a pair of labels."""
    if source > 1 and target > 1:
        return "co-enriching"
    if source < 1 and target < 1:
        return "co-depleting"
    return "separating"


def containment(
    records: list[dict],
    predicate_mass: list[dict],
    sink_mass: np.ndarray,
    child: int,
    parent: int,
    threshold: float = 0.01,
) -> float:
    """Fraction of child decision mass traversing a parent-supported predicate."""
    if sink_mass[parent] <= 0:
        return 0.0
    supported = {
        key
        for key, mass in predicate_mass[parent].items()
        if mass / sink_mass[parent] >= threshold
    }
    child_records = [record for record in records if record["label"] == child]
    total = sum(record["mass"] for record in child_records)
    if not total:
        return 0.0
    return float(
        sum(
            record["mass"]
            for record in child_records
            if set(record["predicates"]) & supported
        )
        / total
    )


def jaccard(left: set, right: set) -> float:
    """Return unweighted Jaccard overlap, using 0 for two empty sets."""
    return len(left & right) / len(left | right) if left or right else 0.0
