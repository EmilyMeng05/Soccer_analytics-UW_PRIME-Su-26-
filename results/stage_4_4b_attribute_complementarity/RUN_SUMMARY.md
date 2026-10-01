# Stage 4.4B/4.4C verified run summary

## Reproduction check

All seven gamma=0 lineups reproduced the Stage 4.2 reference player sets and
role-adjusted-quality totals. See `gamma_zero_reproduction_check.csv`.

## Internal complementarity sensitivity

The model tested three normalized concave functions (`sqrt`, `log`, `exp`) and
gamma values 0, 0.05, 0.10, 0.25, 0.50, and 1.00 for all seven formations.

- At gamma=0.10, all four-defender formations retained their Stage 4.2 player
  sets under every concave function.
- The 3-5-2, 3-4-3, and 5-3-2 each changed one center back at gamma=0.10.
- Across functions at gamma=0.10, the average role-adjusted-quality loss was
  0.00010 to 0.00182 on a roughly 8.4--8.7 outfield total.
- At gamma=0.25, the mean number of changed players ranged from 1.0 (`exp`) to
  2.14 (`sqrt` and `log`). Larger gamma values produced larger quality-coverage
  tradeoffs.

These are sensitivity results, not evidence that any gamma value represents
true football value.

## External descriptive check

Spain 2026 and Real Madrid 2026-27 model-selected 4-3-3 XIs were compared with
3,000 formation-valid global XIs matched within 0.25 mean OVR.

- Spain ranked at approximately the 22nd--31st percentiles, depending on the
  concave function.
- Real Madrid ranked at approximately the 51st--54th percentiles.

The first external check therefore does not support the claim that higher
static-attribute coverage is a defining feature of elite real rosters. The
metric remains an exploratory extension and should not be used as validated
performance evidence.

## Stage 3.5 accuracy discrepancy

- 74.6947%: the class-balanced 34-feature logistic regression in
  `stage3_5C_logistic_34features_functional.py`.
- 77.2211%: the earlier unweighted 34-feature logistic regression in
  `stage3_5_logistic_functional_positions.py`.

Stages 3.6B and 3.6C use `class_weight="balanced"`, so 74.69% is the
internally consistent accuracy to report for the model that generates the
role-suitability probabilities used by Stage 4.
