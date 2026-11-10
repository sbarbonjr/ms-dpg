"""Public global and pair-level explanation objects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .measures import active_predicates, classify_lifts


def _predicate_text(key, feature_names: list[str], thresholds: dict) -> str:
    feature, value, direction = key
    threshold = thresholds.get(key, value)
    return f"{feature_names[feature]} {direction} {threshold:.3g}"


@dataclass
class Explanation:
    explainer: object

    def __post_init__(self) -> None:
        matrix = self.explainer._graph.spf_matrix(
            self.explainer.top_percent, self.explainer.minimum_fraction
        )
        labels = self.explainer.label_names_
        self.spf = pd.DataFrame(matrix, index=labels, columns=labels)

    def _label(self, label: int | str) -> int:
        if isinstance(label, str):
            if label not in self.explainer.label_names_:
                raise KeyError(f"unknown label: {label}")
            return self.explainer.label_names_.index(label)
        if not 0 <= label < len(self.explainer.label_names_):
            raise IndexError("label index out of range")
        return label

    def top_pairs(self, k: int = 5) -> pd.DataFrame:
        rows = [
            {
                "source": self.explainer.label_names_[i],
                "target": self.explainer.label_names_[j],
                "spf": value,
            }
            for i, row in enumerate(self.spf.to_numpy())
            for j, value in enumerate(row)
            if i != j
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("spf", ascending=False)
            .head(k)
            .reset_index(drop=True)
        )

    def active_predicates(self, label: int | str) -> pd.DataFrame:
        index = self._label(label)
        gains = self.explainer._graph.gains[index]
        active = active_predicates(
            gains, self.explainer.top_percent, self.explainer.minimum_fraction
        )
        rows = [
            {
                "predicate": key,
                "text": _predicate_text(
                    key,
                    self.explainer.feature_names_,
                    self.explainer.predicate_thresholds_,
                ),
                "gain": gains[key],
            }
            for key in active
        ]
        columns = ["predicate", "text", "gain"]
        return (
            pd.DataFrame(rows, columns=columns)
            .sort_values("gain", ascending=False)
            .reset_index(drop=True)
        )

    def pair(self, source: int | str, target: int | str) -> PairExplanation:
        return PairExplanation(self, self._label(source), self._label(target))

    def plot_spf(self, ax=None):
        from .plotting import plot_spf

        return plot_spf(self, ax=ax)


@dataclass
class PairExplanation:
    explanation: Explanation
    source_index: int
    target_index: int

    def __post_init__(self) -> None:
        graph = self.explanation.explainer._graph
        self.source, self.target = (
            self.explanation.explainer.label_names_[self.source_index],
            self.explanation.explainer.label_names_[self.target_index],
        )
        self.spf = float(
            self.explanation.spf.iloc[self.source_index, self.target_index]
        )
        if graph.is_br:
            self.shared = pd.DataFrame(
                columns=["predicate", "text", "lift_A", "lift_B", "mass", "kind"]
            )
            self.separating_share = np.nan
            return
        lifts = graph.lifts()
        _records, _, predicate_mass = graph.records()
        shared = active_predicates(
            graph.gains[self.source_index],
            self.explanation.explainer.top_percent,
            self.explanation.explainer.minimum_fraction,
        )
        shared &= active_predicates(
            graph.gains[self.target_index],
            self.explanation.explainer.top_percent,
            self.explanation.explainer.minimum_fraction,
        )
        rows = []
        for key in shared:
            a, b = (
                float(lifts[key][self.source_index]),
                float(lifts[key][self.target_index]),
            )
            rows.append(
                {
                    "predicate": key,
                    "text": _predicate_text(
                        key,
                        self.explanation.explainer.feature_names_,
                        self.explanation.explainer.predicate_thresholds_,
                    ),
                    "lift_A": a,
                    "lift_B": b,
                    "mass": predicate_mass[self.source_index].get(key, 0.0)
                    + predicate_mass[self.target_index].get(key, 0.0),
                    "kind": classify_lifts(a, b),
                }
            )
        self.shared = (
            pd.DataFrame(rows)
            .sort_values("mass", ascending=False)
            .reset_index(drop=True)
            if rows
            else pd.DataFrame(
                columns=["predicate", "text", "lift_A", "lift_B", "mass", "kind"]
            )
        )
        total = self.shared["mass"].sum()
        self.separating_share = (
            float(
                self.shared.loc[self.shared.kind == "separating", "mass"].sum() / total
            )
            if total
            else 0.0
        )

    def contracted(self, n_nodes: int = 14):
        if self.explanation.explainer.is_br_:
            raise NotImplementedError(
                "contracted views are defined for native multi-output forests only"
            )
        from .contracted import ContractedView

        return ContractedView(self, n_nodes)

    def plot_lifts(self, ax=None):
        if self.explanation.explainer.is_br_:
            raise NotImplementedError(
                "lifts are defined for native multi-output forests only"
            )
        from .plotting import plot_lifts

        return plot_lifts(self, ax=ax)
