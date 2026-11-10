# Robustness of the native-versus-BR comparison: protocol

Written before any forest was refitted for this analysis. The results it uses as
reference (the q32 / top 25% / 5% comparison on nine benchmarks) were already known,
since they are reported in the manuscript. The checksum of this file is in
`PROTOCOL.sha256`. Outputs are in `results/robustness/`. Existing scripts and result
files are not modified.

## Purpose

Four analyses requested in the improvement plan (`msDPG_Complete_Improvement_Plan.md`):

- **E2.** Whether the SPF excess tracks the macro-F1 gap between native and BR.
- **E3.** Whether the native-versus-BR comparison depends on the predicate-matching
  rule and on the activity cut-offs, which so far were varied only on synthetic data.
- **E4.** What the mass weighting of SPF adds over an unweighted overlap of the active
  predicate sets.
- **E5.** How the size of the native and BR models differs.

## Data, splits and models

The nine benchmarks of the manuscript with their primary label sets: emotions, scene
and yeast (eight most prevalent labels) loaded with `dpg_mtl.data.load_openml`; image,
birds, genbase, enron, slashdot and langlog (`top8`) loaded with
`experiments.extended_benchmarks.datasets.load_benchmark`. Seeds 0 to 9 and
`split_60_20_20`, as before.

No hyperparameters are selected again. For each dataset and seed, the native forest
(200 trees, defaults, `random_state=seed`) and the BR forests with the hyperparameters
stored for that seed (`results/t4_real.csv` for the first three benchmarks,
`results/extended_benchmarks/raw/` for the others) are refitted. Fits are deterministic
given the seed, so each job first checks that the test macro-F1 of both models equals
the stored value; a job that does not reproduce it is reported and excluded.

## E3: settings

- **Matching:** `round6` (thresholds equal to six decimals), `q16`, `q32`, `q64`
  (feature-wise quantile bins of the training data). q64 uses edges at 65 equally
  spaced training quantiles with the existing `predicate_key`, which only needs the
  edges.
- **Activity:** top {10, 25, 50}% of predicates with positive gain, crossed with a
  minimum of {1, 5, 10}% of the label's maximum gain.

All 4 x 9 = 36 combinations are computed for both models. The manuscript's setting
(q32, 25%, 5%) is the reference; its excess must reproduce the stored values.

Reported per dataset and setting: median over seeds of the median SPF excess over
ordered pairs (native minus BR), number of seeds with positive excess, median native
and BR SPF, mean active-set size per label, fraction of ordered pairs whose shared
active set is empty, and the Spearman correlation between the pair rankings by excess
under the setting and under the reference (median over seeds).

Summary statement: for each dataset, the number of the 36 settings in which the sign
of the median excess equals its sign under the reference. No setting is chosen or
preferred on the basis of these results.

## E4: unweighted comparator

For every ordered pair and setting, the Jaccard index |A_A ∩ A_B| / |A_A ∪ A_B| of the
two labels' active predicate sets (0 when both are empty), for both models. Reported:
the native-minus-BR Jaccard excess (median over pairs and seeds) at every setting, and
the Spearman correlation across pairs between SPF and Jaccard in the native forest at
the reference setting. Jaccard is symmetric and ignores where the mass flows; the
comparison describes how the two quantities differ, not which is better.

## E5: model size

For each model: number of trees (200 for native, 200 x L for BR), total nodes, total
leaves, mean and maximum tree depth. Reported as the median over seeds per dataset.

## E2: macro-F1 gap and excess (existing results only)

From the stored per-seed results of the nine benchmarks: a scatter of the absolute
test macro-F1 gap against the seed-level median excess, by dataset; the Spearman
correlation between the two within each dataset (ten seeds) and its median over
datasets; and, as a sensitivity analysis, the median excess per dataset over the
seeds whose absolute gap is at most .015, with the number of such seeds. This is
descriptive: seeds are not independent datasets, and selecting seeds on test
macro-F1 is not a model-selection procedure.

## Reporting

All datasets and settings are reported. Any deviation from this file is recorded
under "Deviations" below, with its reason.

## Deviations

None so far.
