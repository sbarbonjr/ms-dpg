# Explanation atlas of the nine benchmarks: protocol

Written before the analysis was run. The checksum of this file is in `PROTOCOL.sha256`.
Outputs are in `results/atlas/` and `XAICONF2027_MS-DPG/figures/atlas/`. Existing
scripts and result files are not modified.

## Purpose

Show the two levels of the ms-DPG explanation (the SPF matrix, and the lifts of the
shared predicates of a selected pair) for every real benchmark, with an indication of
which displayed predicates recur across repetitions.

## Models

For each of the nine benchmarks of the manuscript (emotions, scene, yeast with eight
labels, loaded with `dpg_mtl.data.load_openml`; image, birds, genbase, enron, slashdot,
langlog with `top8`, loaded with `experiments.extended_benchmarks.datasets.load_benchmark`):
`split_60_20_20`, a 200-tree native forest with default hyperparameters and
`random_state` equal to the repetition, matching by 32 training quantile bins, and the
reference activity cut-offs (top 25%, minimum 5%). Native forests only: the lifts are
defined for native forests.

## Explanation shown (first repetition)

1. The SPF matrix over all ordered label pairs.
2. The selected pair: the ordered pair with the highest SPF; ties broken by the lower
   source index, then the lower target index. No other selection is made.
3. For the selected pair, every shared active predicate with its two lifts and its
   decision mass for the two labels, F_A(v) + F_B(v) (each trace counted once). A
   predicate is co-enriching when both lifts exceed 1, separating when one exceeds 1 and
   the other is below 1, and co-depleting when both are below 1.
4. The separating share: the decision mass of the separating predicates over that of all
   shared active predicates of the pair.
5. A table of the six shared active predicates with the largest decision mass, with
   their lifts, mass share and recurrence.

Predicates are written with the mean threshold of the splits they merge; a feature with
two values in the training set is written as present/absent.

## Recurrence

For repetitions 1 to 9, the shared active predicates of the same pair are computed. A
predicate of repetition 0 recurs in a repetition when a predicate of that repetition
tests the same feature in the same direction with an overlapping threshold bin (as in
`experiments/contracted_view/PROTOCOL.md`). Reported: the number of the nine
repetitions in which each tabulated predicate recurs, and the SPF of the pair in each
repetition.

## Main-text figure

The SPF matrix of emotions (first repetition), with full label names.

## Reporting

All nine benchmarks are reported. Any deviation from this file is recorded under
"Deviations" below, with its reason.

## Deviations

None so far.
