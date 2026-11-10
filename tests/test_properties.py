import numpy as np
from sklearn.ensemble import RandomForestClassifier
from msdpg import MSDPG
from msdpg.datasets import make_scenario


def test_train_weighting_recovers_ensemble_probability_mass():
    X, Y = make_scenario("D1", n_samples=220)
    forest = RandomForestClassifier(n_estimators=15, random_state=5).fit(X, Y)
    fitted = MSDPG(weighting="train").fit(forest, X)
    _, mass, _ = fitted.extraction()
    expected = [
        len(forest.estimators_) * probabilities[:, 1].sum()
        for probabilities in forest.predict_proba(X)
    ]
    assert np.allclose(mass, expected, rtol=0, atol=1e-10)


def test_duplicate_labels_have_identical_sink_and_gain_structure():
    X, Y = make_scenario("D1", n_samples=220)
    duplicated = np.column_stack([Y[:, 0], Y[:, 0]])
    forest = RandomForestClassifier(n_estimators=15, random_state=5).fit(X, duplicated)
    fitted = MSDPG().fit(forest, X)
    _, mass, predicate_mass = fitted.extraction()
    assert mass[0] == mass[1]
    assert predicate_mass[0] == predicate_mass[1]
    assert fitted.gains[0] == fitted.gains[1]


def test_all_diagnostic_generators_have_binary_four_label_output():
    for name in ("D1", "D2", "D3", "D4", "D5"):
        X, Y = make_scenario(name, n_samples=80)
        assert X.shape == (80, 50)
        assert Y.shape == (80, 4)
        assert set(np.unique(Y)).issubset({0, 1})
