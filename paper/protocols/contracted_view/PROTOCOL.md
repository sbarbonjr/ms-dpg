# Coverage and stability of the contracted views (E6): protocol

Written before the analysis was run. The checksum of this file is in `PROTOCOL.sha256`.
Outputs are in `results/contracted_view/`. Existing scripts and result files are not
modified.

## Purpose

The manuscript shows two contracted ms-DPGs: `sad.lonely -> quiet.still` on emotions
and `white -> blue` on flags, each from the first repetition. E6 measures (i) how much
of the decision mass these views display and omit, and (ii) whether the displayed
predicates and their lift classification recur across repetitions.

## Views

For each pair and each repetition 0 to 9: `split_60_20_20` with that repetition's seed, a
200-tree native forest with default hyperparameters and `random_state` equal to the
seed, matching by 32 training quantile bins, and the selection rule of
`experiments/plot_msdpg_graph.py`: the shared active predicates (top 25%, minimum 5%) that
are co-enriching (both lifts above 1), ranked by their mass for the two labels times the
smaller lift, keeping the 14 highest. Data loading as in `experiments/plot_msdpg_graph.py`
(emotions) and `experiments/flags_illustration.py` (flags). The pair is fixed across
repetitions.

## Quantities per repetition

1. Counts: shared active predicates, co-enriching shared active predicates, displayed
   predicates.
2. Coverage: the fraction of the decision mass of each of the two labels that lies on
   traces containing at least one displayed predicate (each trace counted once).
3. Omission by class: for each of the four classes of the view (leaves storing both
   labels, only the source, only the target, neither; a leaf's class is given by
   p >= .5 for each label, as in the figures), the fraction of the class's leaf mass on
   traces that contain no displayed predicate.
4. SPF of the pair.

The repetition-0 values must reproduce the counts and class shares in the captions of the
two figures.

## Stability across repetitions

Quantile bins differ between repetitions, because they are computed on each training
set. Two predicates of different repetitions are therefore matched when they test the
same feature in the same direction and their threshold bins, as intervals of feature
values, overlap.

5. For each of the 45 pairs of repetitions, the Jaccard index of the two sets of
   displayed predicates under this matching (matched pairs counted once), and the same
   index at the level of (feature, direction). Reported: median and range.
6. Recurrence: for each predicate displayed in repetition 0 (the figure), the number of
   other repetitions in which a matching predicate is displayed, and in which a matching
   predicate is shared and active.
7. Lift agreement: for every predicate shared and active in repetition 0, and every other
   repetition in which a matching predicate is shared and active, whether both are
   classified the same way (co-enriching, separating, or neither). Reported: the fraction
   of agreeing comparisons, weighted by the repetition-0 mass of the predicate.

## Reporting

Both pairs and all repetitions are reported. Any deviation from this file is recorded
under "Deviations" below, with its reason.

## Deviations

None so far.
