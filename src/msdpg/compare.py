"""Native-versus-Binary-Relevance comparison helpers."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .graph import MSDPG


@dataclass
class ArchitectureComparison:
    native: object
    br: object
    excess: pd.DataFrame
    jaccard_excess: pd.DataFrame

    @property
    def median_excess(self) -> float:
        values = self.excess.to_numpy()[~np.eye(len(self.excess), dtype=bool)]
        return float(np.nanmedian(values))


def compare_architectures(
    native_forest, br_model, X_train, **kwargs
) -> ArchitectureComparison:
    """Compare SPF matrices of native and Binary-Relevance forests."""
    native = MSDPG(**kwargs).fit(native_forest, X_train).explain()
    br = MSDPG(**kwargs).fit(br_model, X_train).explain()
    excess = native.spf - br.spf
    n = len(native.spf)
    jaccard = np.full((n, n), np.nan)
    for i in range(n):
        ai = set(native.active_predicates(i).predicate)
        bi = set(br.active_predicates(i).predicate)
        for j in range(n):
            if i == j:
                continue
            aj, bj = (
                set(native.active_predicates(j).predicate),
                set(br.active_predicates(j).predicate),
            )
            ni, nb = ai & aj, bi & bj
            union = ni | nb
            jaccard[i, j] = len(ni) / len(union) if union else 0.0
    return ArchitectureComparison(
        native,
        br,
        excess,
        pd.DataFrame(jaccard, index=native.spf.index, columns=native.spf.columns),
    )
