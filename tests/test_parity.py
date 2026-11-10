"""Regression fixture exported with the reference dpg_mtl implementation."""

import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from msdpg import MSDPG
from msdpg.datasets import make_scenario
from msdpg.measures import spf

FIXTURE = Path(__file__).parent / "data" / "reference_d1.json"


def test_reference_d1_parity_fixture():
    expected = json.loads(FIXTURE.read_text())
    X, Y = make_scenario("D1", random_state=7, n_samples=240)
    model = RandomForestClassifier(n_estimators=16, random_state=11).fit(X, Y)
    fitted = MSDPG().fit(model, X)
    assert len(fitted.traces) == expected["n_traces"]
    _records, sink_mass, _ = fitted.extraction()
    assert np.allclose(sink_mass, expected["sink_mass"], rtol=0, atol=1e-12)
    assert np.allclose(
        fitted.spf_matrix(), expected["spf"], equal_nan=True, rtol=0, atol=1e-12
    )
    assert np.allclose(
        [sum(g.values()) for g in fitted.gains],
        expected["gain_totals"],
        rtol=0,
        atol=1e-12,
    )


def test_matrix_equals_pairwise_spf():
    X, Y = make_scenario("D1", n_samples=200)
    model = RandomForestClassifier(n_estimators=12, random_state=1).fit(X, Y)
    fitted = MSDPG().fit(model, X)
    records, _, _ = fitted.extraction()
    matrix = fitted.spf_matrix()
    for source in range(Y.shape[1]):
        for target in range(Y.shape[1]):
            if source != target:
                assert matrix[source, target] == spf(
                    records, fitted.gains, source, target
                )
