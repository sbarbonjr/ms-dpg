"""Fitting and extraction of multi-sink Decision Predicate Graphs."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
import networkx as nx
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import _tree
from .gain import directional_gains, positive_index
from .predicates import Predicate, predicate_key, quantile_edges
from .measures import active_predicates

@dataclass(frozen=True)
class Trace:
    tree: Any
    predicates: tuple[Predicate, ...]
    weight: float
    probabilities: np.ndarray

def sink(label: int) -> tuple[str, int]:
    return ("sink", label)

@dataclass
class _Graph:
    n_labels: int
    matching: str
    weighting: str
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    traces: list[Trace] = field(default_factory=list)
    gains: list[dict[Predicate, float]] = field(default_factory=list)
    is_br: bool = False

    def _bump_edge(self, source: Any, target: Any, weight: float) -> None:
        if self.graph.has_edge(source, target):
            self.graph[source][target]["weight"] += weight
        else:
            self.graph.add_edge(source, target, weight=weight)

    def _add_tree(self, tree_id: Any, estimator: Any, label_columns: list[int],
                  classes_per_label: list[np.ndarray], X: np.ndarray, edges: list[np.ndarray] | None) -> None:
        tree = estimator.tree_
        weights = (tree.weighted_n_node_samples if self.weighting == "inbag" else
                   np.bincount(estimator.apply(X), minlength=tree.node_count).astype(float))
        positives = [positive_index(classes) for classes in classes_per_label]
        def walk(node: int, path: tuple[Predicate, ...]) -> None:
            left = int(tree.children_left[node])
            if left == _tree.TREE_LEAF:
                weight = float(weights[node])
                if weight <= 0:
                    return
                probabilities = np.full(self.n_labels, np.nan)
                for output, label in enumerate(label_columns):
                    pos = positives[output]
                    probabilities[label] = 0.0 if pos is None else float(tree.value[node, output, pos])
                trace = Trace(tree_id, path, weight, probabilities)
                self.traces.append(trace)
                for a, b in zip(path, path[1:]):
                    self._bump_edge(a, b, weight)
                if path:
                    for label in label_columns:
                        self._bump_edge(path[-1], sink(label), weight * probabilities[label])
                return
            right = int(tree.children_right[node])
            feature, threshold = int(tree.feature[node]), float(tree.threshold[node])
            walk(left, path + (predicate_key(feature, threshold, "<=", self.matching, edges),))
            walk(right, path + (predicate_key(feature, threshold, ">", self.matching, edges),))
        walk(0, ())

    def finish(self) -> None:
        self.gains = [dict(gain) for gain in self.gains]
        for label in range(self.n_labels):
            self.graph.add_node(sink(label), kind="sink")
        for node in self.graph.nodes:
            if node[0] != "sink":
                self.graph.nodes[node]["kind"] = "predicate"

    def records(self) -> tuple[list[dict], np.ndarray, list[dict]]:
        masses = np.zeros(self.n_labels)
        predicate_mass = [defaultdict(float) for _ in range(self.n_labels)]
        records: list[dict] = []
        for trace in self.traces:
            for label, probability in enumerate(trace.probabilities):
                if np.isnan(probability):
                    continue
                mass = trace.weight * probability
                masses[label] += mass
                for predicate in trace.predicates:
                    predicate_mass[label][predicate] += mass
                records.append({"tree": trace.tree, "label": label, "mass": mass, "predicates": trace.predicates})
        return records, masses, [dict(item) for item in predicate_mass]

    def node_flow(self) -> tuple[dict, dict]:
        weight: defaultdict = defaultdict(float)
        flow: defaultdict = defaultdict(lambda: np.zeros(self.n_labels))
        for trace in self.traces:
            probabilities = np.nan_to_num(trace.probabilities)
            for predicate in set(trace.predicates):
                weight[predicate] += trace.weight
                flow[predicate] += trace.weight * probabilities
        return dict(weight), dict(flow)

    def spf_matrix(self, top_percent: float, minimum_fraction: float) -> np.ndarray:
        n = self.n_labels
        bits: defaultdict = defaultdict(int)
        for label, gains in enumerate(self.gains):
            for predicate in active_predicates(gains, top_percent, minimum_fraction):
                bits[predicate] |= 1 << label
        numerator, denominator = np.zeros((n, n)), np.zeros(n)
        for trace in self.traces:
            valid = ~np.isnan(trace.probabilities)
            masses = np.where(valid, trace.weight * np.nan_to_num(trace.probabilities), 0.0)
            denominator += masses
            reach = [0] * n
            for predicate in set(trace.predicates):
                mask = bits.get(predicate, 0)
                for source in range(n):
                    if mask >> source & 1:
                        reach[source] |= mask
            for source, mask in enumerate(reach):
                for target in range(n):
                    if mask >> target & 1:
                        numerator[source, target] += masses[target]
        with np.errstate(invalid="ignore", divide="ignore"):
            result = np.where(denominator > 0, numerator / denominator, 0.0)
        np.fill_diagonal(result, np.nan)
        return result

    def lifts(self) -> dict[Predicate, np.ndarray]:
        if self.is_br:
            raise NotImplementedError("lifts are defined for native multi-output forests only")
        weight, flow = self.node_flow()
        total_weight = sum(trace.weight for trace in self.traces)
        baseline = sum(trace.weight * trace.probabilities for trace in self.traces) / total_weight
        return {key: (flow[key] / weight[key]) / baseline for key in weight}

class MSDPG:
    """Explain a fitted multi-output random forest with a multi-sink DPG.

    Parameters
    ----------
    matching : {"quantile", "exact"}, default="quantile"
        Predicate matching rule. Quantile matching maps thresholds into bins.
    bins : int, default=32
        Number of feature-wise training quantile bins.
    top_percent, minimum_fraction : float
        Active-predicate gain cutoffs used by Shared Predicate Flow.
    """
    def __init__(self, matching: str = "quantile", bins: int = 32,
                 top_percent: float = 25, minimum_fraction: float = .05,
                 weighting: str = "inbag") -> None:
        if matching not in {"quantile", "exact"}:
            raise ValueError("matching must be 'quantile' or 'exact'")
        if weighting not in {"inbag", "train"}:
            raise ValueError("weighting must be 'inbag' or 'train'")
        self.matching, self.bins = matching, bins
        self.top_percent, self.minimum_fraction, self.weighting = top_percent, minimum_fraction, weighting

    def fit(self, forest: Any, X: np.ndarray, label_names: list[str] | None = None,
            feature_names: list[str] | None = None) -> "MSDPG":
        """Extract traces, sinks, and gains from a fitted supported forest."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("X must be a two-dimensional training matrix")
        edges = quantile_edges(X, self.bins) if self.matching == "quantile" else None
        if isinstance(forest, RandomForestClassifier):
            if getattr(forest, "n_outputs_", 1) < 2:
                raise ValueError("single-output forests should be explained with DPG")
            classes = list(forest.classes_)
            if any(not set(np.asarray(item)).issubset({0, 1}) for item in classes):
                raise ValueError("ms-DPG supports binary labels only")
            graph = _Graph(len(classes), self.matching, self.weighting, gains=[defaultdict(float) for _ in classes])
            for tree_id, estimator in enumerate(forest.estimators_):
                for label, class_values in enumerate(classes):
                    for key, value in directional_gains(estimator, label, class_values, self.matching, edges).items():
                        graph.gains[label][key] += value
                graph._add_tree(tree_id, estimator, list(range(len(classes))), classes, X, edges)
            is_br = False
        elif isinstance(forest, MultiOutputClassifier):
            estimators = forest.estimators_
            if len(estimators) < 2 or not all(isinstance(item, RandomForestClassifier) for item in estimators):
                raise ValueError("BR requires a MultiOutputClassifier of RandomForestClassifier estimators")
            classes = [item.classes_ for item in estimators]
            if any(not set(np.asarray(item)).issubset({0, 1}) for item in classes):
                raise ValueError("ms-DPG supports binary labels only")
            graph = _Graph(len(estimators), self.matching, self.weighting, gains=[defaultdict(float) for _ in estimators], is_br=True)
            for label, rf in enumerate(estimators):
                for tree_id, estimator in enumerate(rf.estimators_):
                    for key, value in directional_gains(estimator, 0, rf.classes_, self.matching, edges).items():
                        graph.gains[label][key] += value
                    graph._add_tree((label, tree_id), estimator, [label], [rf.classes_], X, edges)
            is_br = True
        else:
            raise ValueError("forest must be RandomForestClassifier or MultiOutputClassifier of random forests")
        graph.finish()
        self._graph, self.forest_, self.is_br_ = graph, forest, is_br
        # Expose the extracted representation for structural analyses while
        # keeping construction internals private.
        self.graph, self.traces, self.gains = graph.graph, graph.traces, graph.gains
        self.label_names_ = list(label_names) if label_names is not None else [str(i) for i in range(graph.n_labels)]
        if len(self.label_names_) != graph.n_labels:
            raise ValueError("label_names length must equal the number of labels")
        self.feature_names_ = list(feature_names) if feature_names is not None else [f"x{j}" for j in range(X.shape[1])]
        if len(self.feature_names_) != X.shape[1]:
            raise ValueError("feature_names length must equal X.shape[1]")
        return self

    def _require_fit(self) -> None:
        if not hasattr(self, "_graph"):
            raise ValueError("call fit before querying an ms-DPG")

    def spf_matrix(self) -> np.ndarray:
        """Return SPF(A→B) for all ordered label pairs; the diagonal is NaN."""
        self._require_fit()
        return self._graph.spf_matrix(self.top_percent, self.minimum_fraction)

    def node_flow(self) -> tuple[dict, dict]:
        """Return predicate trace mass ``W(v)`` and per-label decision flow ``F_l(v)``."""
        self._require_fit()
        return self._graph.node_flow()

    def lift(self) -> dict[Predicate, np.ndarray]:
        """Return native-forest predicate lifts ``(F_l(v)/W(v))/pi_l``."""
        self._require_fit()
        return self._graph.lifts()

    def extraction(self) -> tuple[list[dict], np.ndarray, list[dict]]:
        """Return legacy path records, label sink masses, and predicate masses."""
        self._require_fit()
        return self._graph.records()

    def explain(self):
        """Return the global SPF and pair-level explanation interface."""
        self._require_fit()
        from .explanation import Explanation
        return Explanation(self)
