# Concepts

## Predicate and trace

A predicate is a directional split `(feature, threshold-or-bin, direction)`. ms-DPG records only root-to-leaf traces executed by the fitted forest. A trace ending at leaf `r` has in-bag mass `w_r` and a positive-class fraction `p_{r,l}` for each label.

## Fractional sinks

Each label has a sink. The decision mass sent from a leaf to label `l` is `m(r,S_l)=w_r p_{r,l}`. Thus, one executed path can contribute different mass to different label sinks.

## Shared Predicate Flow

For active predicate sets `A_A` and `A_B`, shared predicates are `C_AB=A_A ∩ A_B`. `SPF(A→B)` is the fraction of B's sink mass on executed traces containing at least one element of `C_AB`. The measure is directional, lies in `[0,1]`, and has an undefined diagonal.

## Lifts and contracted views

For predicate `v`, the native-forest lift is `(F_l(v)/W(v))/pi_l`. A predicate is co-enriching when both pair lifts exceed one, separating when they fall on opposite sides of one, and co-depleting when both are below one. The contracted view retains the co-enriching shared predicates with the highest `mass × min(lift_A, lift_B)`, then orders them by mean depth.
