import numpy as np
from sklearn.ensemble import RandomForestClassifier
from msdpg import MSDPG
from msdpg.datasets import make_scenario
from msdpg.predicates import predicate_key, quantile_edges

def test_predicate_quantile_key_any_bin_count():
    X = np.arange(40, dtype=float).reshape(20, 2)
    for bins in (8, 16, 32, 64):
        edges = quantile_edges(X, bins)
        assert predicate_key(0, 5.0, "<=", "quantile", edges)[1] == np.searchsorted(edges[0], 5.0)

def test_spf_bounds_and_native_api():
    X, Y = make_scenario("D1", n_samples=200)
    forest = RandomForestClassifier(n_estimators=20, random_state=0).fit(X, Y)
    exp = MSDPG().fit(forest, X).explain()
    values = exp.spf.to_numpy()
    assert np.isnan(np.diag(values)).all()
    assert np.nanmin(values) >= 0 and np.nanmax(values) <= 1
    assert {"predicate", "lift_A", "lift_B", "mass", "kind"}.issubset(exp.pair(0, 1).shared.columns)
