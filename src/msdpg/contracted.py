"""Contracted predicate--sink views."""
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd

@dataclass
class ContractedView:
    pair: object
    n_nodes: int = 14
    def __post_init__(self) -> None:
        shared = self.pair.shared.copy()
        co = shared[shared.kind == "co-enriching"].copy()
        co["score"] = co["mass"] * co[["lift_A", "lift_B"]].min(axis=1)
        self.predicates = co.sort_values("score", ascending=False).head(self.n_nodes).reset_index(drop=True)
        graph = self.pair.explanation.explainer._graph
        _, sink_mass, predicate_mass = graph.records()
        self.coverage = {"source": float(sum(predicate_mass[self.pair.source_index].get(key, 0.) for key in self.predicates.predicate) / sink_mass[self.pair.source_index]) if sink_mass[self.pair.source_index] else 0.,
                         "target": float(sum(predicate_mass[self.pair.target_index].get(key, 0.) for key in self.predicates.predicate) / sink_mass[self.pair.target_index]) if sink_mass[self.pair.target_index] else 0.}
        self.omitted_by_class = {kind: int((shared.kind == kind).sum() - (self.predicates.kind == kind).sum()) for kind in ("co-enriching", "separating", "co-depleting")}
    def to_graphviz(self, path):
        try:
            import graphviz
        except ImportError as exc:
            raise ImportError("install msdpg[graphviz] and system graphviz") from exc
        dot = graphviz.Digraph()
        for _, row in self.predicates.iterrows():
            dot.node(str(row.predicate), row.text)
        return dot.render(str(path), cleanup=True)

def match_predicates(left: set, right: set) -> set:
    """Return exact predicate-key intersection for compatible fitted graphs."""
    return set(left) & set(right)
