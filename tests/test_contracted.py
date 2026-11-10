import importlib.util

import pytest
from sklearn.ensemble import RandomForestClassifier

from msdpg import MSDPG
from msdpg.datasets import make_scenario


def test_contracted_view_reports_selection_coverage_and_classes():
    X, Y = make_scenario("D1", n_samples=220)
    forest = RandomForestClassifier(n_estimators=15, random_state=0).fit(X, Y)
    view = MSDPG().fit(forest, X).explain().pair(0, 1).contracted(5)
    assert len(view.predicates) <= 5
    assert set(view.coverage) == {"source", "target"}
    assert all(0 <= value <= 1 for value in view.coverage.values())
    assert set(view.omitted_by_class) == {
        "both",
        "source only",
        "target only",
        "neither",
    }


@pytest.mark.skipif(
    importlib.util.find_spec("graphviz") is None, reason="python-graphviz unavailable"
)
def test_contracted_graphviz_writes_requested_artifacts(tmp_path):
    X, Y = make_scenario("D1", n_samples=160)
    forest = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, Y)
    view = MSDPG().fit(forest, X).explain().pair(0, 1).contracted(4)
    view.to_graphviz(tmp_path / "view")
    assert (tmp_path / "view.dot").exists()
    assert (tmp_path / "view.pdf").exists()
    assert (tmp_path / "view.png").exists()
