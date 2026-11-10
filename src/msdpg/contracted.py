"""Selection, coverage, and rendering of contracted pair views."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

_CLASSES = {
    (1, 1): "both",
    (1, 0): "source only",
    (0, 1): "target only",
    (0, 0): "neither",
}


def _interval(key, edges):
    feature, bin_id, direction = key
    edge = edges[feature]
    return (
        feature,
        direction,
        edge[bin_id - 1] if bin_id > 0 else -math.inf,
        edge[bin_id] if bin_id < len(edge) else math.inf,
    )


def _overlap(left, right) -> bool:
    return (
        left[0] == right[0]
        and left[1] == right[1]
        and left[2] <= right[3]
        and right[2] <= left[3]
    )


def match_predicates(
    left: Iterable, right: Iterable, edges_left=None, edges_right=None
) -> set:
    """Match predicates by feature, direction, and overlapping quantile intervals.

    If quantile edges are omitted, the function falls back to exact key matching.
    """
    left, right = list(left), list(right)
    if edges_left is None or edges_right is None:
        return set(left) & set(right)
    used, matches = set(), set()
    for key in left:
        interval = _interval(key, edges_left)
        for index, candidate in enumerate(right):
            if index not in used and _overlap(
                interval, _interval(candidate, edges_right)
            ):
                used.add(index)
                matches.add(key)
                break
    return matches


@dataclass
class ContractedView:
    """A compact native-forest view of co-enriching predicates for one label pair."""

    pair: object
    n_nodes: int = 14

    def __post_init__(self) -> None:
        if self.n_nodes < 1:
            raise ValueError("n_nodes must be positive")
        graph = self.pair.explanation.explainer._graph
        shared = self.pair.shared.copy()
        candidate = shared[shared.kind == "co-enriching"].copy()
        candidate["score"] = candidate["mass"] * candidate[["lift_A", "lift_B"]].min(
            axis=1
        )
        selected = candidate.nlargest(self.n_nodes, "score").copy()
        depth_sum, depth_weight = defaultdict(float), defaultdict(float)
        chosen = set(selected.predicate)
        for trace in graph.traces:
            for depth, predicate in enumerate(trace.predicates):
                if predicate in chosen:
                    depth_sum[predicate] += trace.weight * depth
                    depth_weight[predicate] += trace.weight
        selected["mean_depth"] = [
            depth_sum[key] / depth_weight[key] for key in selected.predicate
        ]
        self.predicates = selected.sort_values(
            ["mean_depth", "score"], ascending=[True, False]
        ).reset_index(drop=True)
        chosen = set(self.predicates.predicate)
        total_mass = {self.pair.source_index: 0.0, self.pair.target_index: 0.0}
        covered = {self.pair.source_index: 0.0, self.pair.target_index: 0.0}
        class_weight = defaultdict(float)
        class_omitted = defaultdict(float)
        for trace in graph.traces:
            hit = bool(set(trace.predicates) & chosen)
            label_class = (
                int(trace.probabilities[self.pair.source_index] >= 0.5),
                int(trace.probabilities[self.pair.target_index] >= 0.5),
            )
            class_weight[label_class] += trace.weight
            if not hit:
                class_omitted[label_class] += trace.weight
            for label in total_mass:
                mass = trace.weight * trace.probabilities[label]
                total_mass[label] += mass
                if hit:
                    covered[label] += mass
        self.coverage = {
            "source": covered[self.pair.source_index]
            / total_mass[self.pair.source_index]
            if total_mass[self.pair.source_index]
            else 0.0,
            "target": covered[self.pair.target_index]
            / total_mass[self.pair.target_index]
            if total_mass[self.pair.target_index]
            else 0.0,
        }
        self.omitted_by_class = {
            name: class_omitted[key] / class_weight[key]
            if class_weight[key]
            else float("nan")
            for key, name in _CLASSES.items()
        }

    def to_graphviz(self, path):
        """Write a DOT, PDF, and PNG rendering of the selected predicate--sink view."""
        try:
            import graphviz
        except ImportError as exc:
            raise ImportError(
                "install msdpg[graphviz] and system graphviz to render contracted views"
            ) from exc
        path = Path(path)
        dot = graphviz.Digraph("contracted", format="pdf")
        dot.attr(rankdir="LR", splines="polyline", nodesep="0.22", ranksep="0.45")
        dot.attr(
            "node", shape="box", style="rounded", fontname="Helvetica", fontsize="10"
        )
        source, target = self.pair.source, self.pair.target
        # A single bridge represents the retained executed predicate mass.  It
        # avoids drawing two crossing sink edges from every predicate while the
        # predicate labels themselves retain pair-specific lift information.
        dot.node(
            "paths",
            "retained executed\\nshared-predicate paths",
            shape="diamond",
            fontsize="9",
        )
        dot.node("source", f"source sink\\n{source}", shape="ellipse")
        dot.node("target", f"target sink\\n{target}", shape="ellipse")
        maximum = max(self.predicates.mass, default=1.0)
        for index, row in self.predicates.iterrows():
            name = f"p{index}"
            label = f"{row.text}\\nlift=({row.lift_A:.2f}, {row.lift_B:.2f})"
            dot.node(name, label)
            dot.edge(name, "paths", penwidth=str(0.6 + 2.4 * row.mass / maximum))
        dot.edge("paths", "source", label=f"coverage {self.coverage['source']:.1%}")
        dot.edge("paths", "target", label=f"coverage {self.coverage['target']:.1%}")
        dot.save(str(path.with_suffix(".dot")))
        rendered = dot.render(filename=str(path), cleanup=True)
        # Graphviz produces PDF; a matching PNG is useful in documentation.
        dot.format = "png"
        dot.render(filename=str(path), cleanup=True)
        return Path(rendered)
