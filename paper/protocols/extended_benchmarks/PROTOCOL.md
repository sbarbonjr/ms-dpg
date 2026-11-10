# Extended real-data benchmarks: protocol

Written before any model was fitted on the new datasets. At that point only the
dataset descriptors had been inspected (instances, features, labels, cardinality,
label prevalence, column types). The checksum in `PROTOCOL.sha256` is that of the
version written then, kept unchanged in `PROTOCOL.original.md`; this file differs
from it only by this sentence and the "Deviations" section.

## Purpose

Extend the real-data evaluation from three benchmarks to nine, using the same
protocol as the existing experiments, to answer the comment that three datasets
are too few. Existing scripts and result files are not modified. Outputs are in
`results/extended_benchmarks/`.

## Datasets (OpenML, pinned name and version)

Same OpenML series as the existing benchmarks (emotions v3, scene v4, yeast v4).

| Dataset | Version | OpenML id | Domain |
|---|---|---|---|
| image | 2 | 40592 | Images |
| birds | 3 | 40588 | Audio |
| slashdot | 3 | 40596 | Text |
| langlog | 1 | 40593 | Text |
| enron | 2 | 40590 | Email text |
| genbase | 2 | 40591 | Proteins |

All six are included whatever their results. No dataset is added or removed after
results are seen. Datasets that could not be obtained (medical, CAL500, reuters)
are out of scope for this protocol.

## Features

- Numeric columns are kept; missing values are imputed with the column median, as in
  the existing loader.
- Categorical columns with exactly two values are encoded 0/1. This is needed for
  enron and genbase, whose features are all categorical.
- Other categorical columns are dropped, as in the existing loader.
- Columns constant over the whole dataset are dropped.

## Labels

- **Primary analysis (`top8`):** the 8 most prevalent labels, the rule already used
  for yeast. Datasets with at most 8 labels keep them all.
- **Sensitivity analysis (`min20`):** all labels with at least 20 positive instances
  in the whole dataset. Not run for image, where it is identical to `top8`.
- Instances without any retained label are kept, as for yeast.

## Models and splits

Identical to the existing protocol: `split_60_20_20` with seeds 0 to 9; 200-tree
forests; native forest with default hyperparameters; BR hyperparameters selected
from the fixed grid in `dpg_mtl.models.GRID` to match the validation macro-F1 of the
native forest; q32 predicate matching; default activity cut-offs (top 25%, 5%).

## Quantities

For each dataset, label mode and seed:

1. Test macro-F1 of native and BR, and their absolute gap. Parity is a gap of at most
   .015.
2. SPF for all ordered label pairs, native and BR. The excess is native minus BR per
   pair, summarized by its median over pairs.
3. Equal-hyperparameter control: a native forest refitted with the BR
   hyperparameters. Its excess over BR, and the seed-only reference (the same forest
   with seed + 1000).
4. Label nulls for the native forest: observed, independent, joint and conditional
   permutation, reporting median SPF, mean |phi| and test AUC.
5. Per-label TreeSHAP ranking agreement (Spearman) for the comparisons of
   `experiments/postreg_hp_control.py`, on 200 test instances.

## Inference

One-sided Wilcoxon signed-rank test of the seed-level median excess, as in
`experiments/t4_real_case.py`. Holm correction over the nine primary analyses (six
new `top8` analyses plus the three existing ones, whose p-values are read from
`results/t4_wilcoxon.csv`). Sensitivity analyses are reported without entering the
correction.

## Reporting

- Every dataset is reported, including those where native does not exceed BR, or
  where the macro-F1 gap exceeds the tolerance.
- Datasets outside the parity tolerance are flagged, as scene already is, and are not
  removed.
- Any deviation from this file is recorded under "Deviations" below, with its reason.

## Replication check

Before the new datasets are run, the new runner is applied to emotions and yeast at
seed 0. It must reproduce the corresponding values in `results/t4_real.csv`.

## Deviations

### D-1. TreeSHAP validity rule (quantity 5)

Recorded after the core quantities 1 to 4 had been computed for all jobs and after
the TreeSHAP failures below had been seen; approved by the authors.

**What happened.** The first seed-0 run, with SHAP, aborted on the native forest of
one of the text datasets (enron, langlog, slashdot were running) with shap's
additivity error (`ExplainerError: Additivity check failed in
TreeExplainer!`). A diagnostic at seed 0 (top8) measured, for the native forest and
each label, the largest difference between the sum of the SHAP values plus the base value
and the model output:

| Dataset | Native tree depth (mean / max) | Labels failing the check | Largest error |
|---|---|---|---|
| slashdot | 241 / 286 | 8 of 8 | 6.3e62 |
| enron | 61 / 79 | 4 of 8 | 18.9 |
| langlog | 32 / 47 | 0 of 8 | 6.5e-9 |

The tree view passed to shap reproduces the forest output to 1e-15, so the failures
are numerical instability of path-dependent TreeSHAP on very deep trees, not a
mismatch of inputs. On slashdot each label also took about 40 minutes.

**Rule adopted** (`run_shap.py`, outputs in `results/extended_benchmarks/shap/`):

1. TreeSHAP is computed with the additivity check disabled and the error measured
   for every forest and label. A label is valid for a forest when every explained
   row passes shap's own criterion (`np.isclose` with `atol=1e-2`, `rtol=1e-2`).
2. Forests whose deepest tree exceeds 100 levels are not computed. This excludes
   the default native forests of slashdot (both label modes), and the depth-unbounded
   BR configurations selected on some slashdot seeds.
3. Each comparison uses only the labels valid in both forests. The number of seeds
   and of valid labels is reported with every value.

The same check is applied to the BR forests, for which the existing pipeline already
disabled the additivity check; no BR forest failed it.

**Consequences.** All labels are valid for birds, genbase, image and langlog. On
enron the default native forests are valid for about half of the labels (37 of 80
with top8), so comparisons involving them rest on a subset. On slashdot only the
comparisons between forests with the BR hyperparameters are available, on 9 of 10
seeds (top8) and 5 of 10 (min20). The core quantities 1 to 4 are unaffected.
Before the full stage, the rule reproduced the stored seed-0 agreement values of
genbase and birds exactly, and every refitted forest reproduced its stored test
macro-F1 exactly.
