"""Fetch emotions from OpenML and inspect its strongest ms-DPG label pairs."""
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from msdpg import MSDPG

# Requires network access; OpenML target conversion is intentionally explicit.
data = fetch_openml(name="emotions", version=1, as_frame=True)
X = data.data.to_numpy()
Y = data.target.to_numpy().astype(int)
forest = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(X, Y)
exp = MSDPG().fit(forest, X, label_names=list(data.target_names)).explain()
print(exp.top_pairs())
