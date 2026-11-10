from sklearn.ensemble import RandomForestClassifier
from msdpg import MSDPG
from msdpg.datasets import make_scenario

X, Y = make_scenario("D1", random_state=0)
forest = RandomForestClassifier(n_estimators=30, random_state=0).fit(X, Y)
explanation = MSDPG().fit(forest, X).explain()
print(explanation.top_pairs())
print(explanation.pair(0, 1).shared.head())
