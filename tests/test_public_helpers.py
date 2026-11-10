import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputClassifier

from msdpg import MSDPG
from msdpg.compare import compare_architectures
from msdpg.datasets import make_scenario
from msdpg.measures import containment, jaccard, spf
from msdpg.predicates import predicate_key, quantile_edges
from msdpg.sinks import SinkExtraction


def _native_and_data():
    X, Y = make_scenario("D1", n_samples=160)
    return X, Y, RandomForestClassifier(n_estimators=10, random_state=4).fit(X, Y)


def test_predicate_modes_and_validation():
    X = np.arange(30, dtype=float).reshape(15, 2)
    edges = quantile_edges(X, 8)
    assert predicate_key(0, 1.23456789, ">", "exact") == (0, 1.234568, ">")
    assert predicate_key(0, 1.0, ">", "quantile", edges)[0] == 0
    with pytest.raises(ValueError):
        quantile_edges(X, 1)
    with pytest.raises(ValueError):
        predicate_key(0, 1, "=", "exact")
    with pytest.raises(ValueError):
        predicate_key(0, 1, ">", "quantile")


def test_measure_helpers_and_zero_conventions():
    records = [{"label": 1, "mass": 2.0, "predicates": ((0, 1, ">"),)}]
    gains = [{(0, 1, ">"): 1.0}, {(0, 1, ">"): 1.0}]
    assert spf(records, gains, 0, 1) == 1.0
    assert spf([], gains, 0, 1) == 0.0
    assert jaccard(set(), set()) == 0.0
    assert jaccard({1}, {1, 2}) == 0.5
    assert (
        containment(records, [{}, {(0, 1, ">"): 2.0}], np.array([1.0, 2.0]), 1, 1)
        == 1.0
    )
    assert containment(records, [{}, {}], np.array([1.0, 0.0]), 1, 1) == 0.0


def test_input_validation_and_explanation_errors():
    X, Y, native = _native_and_data()
    with pytest.raises(ValueError):
        MSDPG(matching="bad")
    with pytest.raises(ValueError):
        MSDPG().fit(native, X[:, 0])
    single = RandomForestClassifier(n_estimators=2, random_state=0).fit(X, Y[:, 0])
    with pytest.raises(ValueError):
        MSDPG().fit(single, X)
    reg = RandomForestRegressor(n_estimators=2, random_state=0).fit(X, Y)
    with pytest.raises(TypeError):
        MSDPG().fit(reg, X)
    exp = MSDPG().fit(native, X).explain()
    with pytest.raises(KeyError):
        exp.pair("missing", 0)
    with pytest.raises(IndexError):
        exp.active_predicates(99)


def test_plots_compare_and_legacy_container():
    X, Y, native = _native_and_data()
    br = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=8, random_state=4)
    ).fit(X, Y)
    exp = MSDPG().fit(native, X).explain()
    assert exp.plot_spf().images
    assert exp.pair(0, 1).plot_lifts().get_xscale() == "log"
    result = compare_architectures(native, br, X)
    assert result.excess.shape == (Y.shape[1], Y.shape[1])
    assert np.isfinite(result.median_excess)
    legacy = SinkExtraction([], np.array([0.0]), [{}], [{}])
    assert legacy.sink_mass.shape == (1,)
