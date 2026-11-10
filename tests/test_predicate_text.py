from sklearn.ensemble import RandomForestClassifier

from msdpg import MSDPG
from msdpg.datasets import make_scenario


def test_predicate_text_uses_merged_mean_threshold_not_bin_identifier():
    X, Y = make_scenario("D1", n_samples=220)
    forest = RandomForestClassifier(n_estimators=12, random_state=3).fit(X, Y)
    exp = MSDPG().fit(forest, X).explain()
    table = exp.active_predicates(0)
    assert not table.empty
    assert all(" q" not in text for text in table.text)
    assert any("." in text or "-" in text for text in table.text)
