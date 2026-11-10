# ms-DPG

**Explain how a multi-output tree ensemble shares its decisions across labels.**

Multi-sink Decision Predicate Graphs (ms-DPG) extend Decision Predicate Graphs to binary multi-output random forests. A fitted forest produces one graph with one sink per label. Each executed root-to-leaf trace sends label-specific decision mass to its sinks, making shared directional conditions inspectable.

| Question | Answer |
|---|---|
| Which label pairs reuse decision conditions? | SPF matrix |
| Which conditions are reused? | Pair explanation and contracted view |
| Does fitting jointly change structure? | Native vs BR comparison |

## Installation

```bash
pip install git+https://github.com/sbarbonjr/ms-dpg.git
```

## Quick start

```python
from sklearn.ensemble import RandomForestClassifier
from msdpg import MSDPG
from msdpg.datasets import make_scenario

X, Y = make_scenario("D1", random_state=0)
forest = RandomForestClassifier(n_estimators=200, random_state=0).fit(X, Y)
exp = MSDPG().fit(forest, X).explain()
print(exp.top_pairs())
pair = exp.pair(0, 1)
print(pair.shared.head())
```

SPF describes fitted decision structure; it is not a measure of statistical label dependence or perturbation-based leakage.

## Development

```bash
pip install -e .[dev]
pytest
```

## License

MIT.
