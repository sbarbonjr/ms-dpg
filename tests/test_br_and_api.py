import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from msdpg import MSDPG
from msdpg.datasets import make_scenario

def test_public_structural_queries_and_br_limitations():
    X, Y = make_scenario("D1", n_samples=180)
    native = RandomForestClassifier(n_estimators=12, random_state=0).fit(X, Y)
    fitted = MSDPG().fit(native, X)
    assert fitted.spf_matrix().shape == (4, 4)
    assert fitted.traces and fitted.gains and fitted.node_flow()[0]
    assert fitted.lift()
    br = MultiOutputClassifier(RandomForestClassifier(n_estimators=8, random_state=0)).fit(X, Y)
    br_fitted = MSDPG().fit(br, X)
    with pytest.raises(NotImplementedError):
        br_fitted.lift()
    with pytest.raises(NotImplementedError):
        br_fitted.explain().pair(0, 1).contracted()
