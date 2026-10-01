# Stage 4 Updated Notes: Role-Aware Soccer Lineup Selection

## Important version note

These notes document the **current seven-formation research pipeline** used for the Sloan analysis. They replace the older Stage 4 notes based on three formations, a fixed 23-player squad, and match-state substitution plans. Those older experiments can remain in the repository as an archived exploratory branch, but they should not be mixed with the present Stage 4.1–4.4C results.

The current Stage 4 research question is:

> How does optimal soccer lineup selection change when moving from overall player quality to functional role suitability and, experimentally, team-level attribute coverage?

The seven formations are 4-3-3, 4-2-3-1, 4-4-2, 4-1-2-1-2, 3-5-2, 3-4-3, and 5-3-2.

---

## Stage 4.1 — OVR-only lineup baseline

### Purpose

Stage 4.1 establishes the simplest comparison method:

> Choose the highest-OVR eligible players for the required formation slots.

This answers the mentor's baseline question directly: what team would we obtain by choosing the “best” player at each position using only EAFC overall rating?

### Method

- Eligibility comes only from each player's official primary and alternative EAFC positions.
- Learned role-suitability probabilities are not used.
- The Hungarian assignment algorithm fills all ten outfield slots simultaneously.
- A player can fill only one slot in a lineup.
- Goalkeepers are handled separately.

Joint assignment is necessary because a player may be officially eligible for multiple slots but cannot occupy two positions at once.

### Results

| Formation | Mean XI OVR | Total XI OVR |
|---|---:|---:|
| 4-3-3 | 89.36 | 983 |
| 4-2-3-1 | 89.36 | 983 |
| 4-4-2 | 89.45 | 984 |
| 4-1-2-1-2 | 89.27 | 982 |
| 3-5-2 | 89.55 | 985 |
| 3-4-3 | 89.55 | 985 |
| 5-3-2 | 89.00 | 979 |

### Interpretation

The baseline produces extremely high-rated teams, but OVR alone does not measure how strongly a player's detailed skill profile resembles the functional demands of the assigned role.

### How this result can be used

- It is the primary control condition for the paper.
- It makes the later role-adjusted changes interpretable.
- It prevents us from claiming that the optimized team is different without comparing it with the obvious “all-star” selection rule.

---

## Stage 4.2 — Role-adjusted quality

### Purpose

Stage 4.2 asks whether lineup selection changes when overall quality is discounted for imperfect functional-role fit.

### Method

For player \(i\) and role \(r\), the selection score is

\[
\operatorname{Score}_{ir}
=Q_i\left[1-\lambda(1-S_{ir})\right],
\]

where

- \(Q_i=\operatorname{OVR}_i/100\),
- \(S_{ir}\) is the learned role-suitability probability, and
- \(\lambda\) controls the strength of the role-fit penalty.

The multiplicative form is important. Role suitability can preserve or discount a player's quality, but it cannot make a low-OVR player elite merely because the player has a “pure” positional profile.

The analysis tests \(\lambda\in\{0.10,0.25,0.40\}\) and uses \(\lambda=0.25\) as the reference setting. Eligibility continues to come only from official EAFC positions; the learned model does not invent position eligibility.

### Reference-setting results

At \(\lambda=0.25\), role adjustment changes between four and six starters in every formation.

| Formation | Players changed | Jaccard overlap | OVR-only mean | Role-adjusted mean | Mean OVR change |
|---|---:|---:|---:|---:|---:|
| 4-3-3 | 4 | 0.47 | 89.36 | 88.82 | -0.55 |
| 4-2-3-1 | 4 | 0.47 | 89.36 | 88.36 | -1.00 |
| 4-4-2 | 5 | 0.38 | 89.45 | 87.82 | -1.64 |
| 4-1-2-1-2 | 4 | 0.47 | 89.27 | 88.55 | -0.73 |
| 3-5-2 | 6 | 0.29 | 89.55 | 88.45 | -1.09 |
| 3-4-3 | 6 | 0.29 | 89.55 | 88.82 | -0.73 |
| 5-3-2 | 5 | 0.38 | 89.00 | 88.45 | -0.55 |

### Main finding

Role-aware selection produces a **meaningfully different lineup** while accepting a **small and controllable OVR cost**. Across the seven formations, four to six of eleven players change, while mean OVR falls by only 0.55–1.64 points.

This supports the claim that “highest OVR” and “best role-adjusted lineup” are not equivalent selection rules.

### What this does not prove

The results do not establish that the role-adjusted teams would win more matches. They demonstrate a selection difference and quantify its cost; match superiority would require outcome-linked validation.

### How this result can be used

- This is the strongest result for the Sloan abstract.
- The 4–6 changed players, 0.29–0.47 Jaccard overlap, and 0.55–1.64 OVR cost provide a concise quantitative story.
- The \(\lambda\) sweep shows that the conclusion is not based on one unexamined penalty value.

---

## Stage 4.3 — Formation robustness and selection margins

### Purpose

Stage 4.3 asks two related questions:

1. Which players continue to be selected when the formation changes?
2. How strongly does the optimizer prefer each selected player over the next-best lineup available without that player?

### Method

- Count each player's appearances across the seven formations.
- Compare OVR-only and reference role-adjusted robustness.
- Examine stability across the tested \(\lambda\) values.
- Re-optimize after removing each selected player to calculate a player-removal margin.

The removal margin measures the reduction in the globally optimized lineup score, rather than merely comparing a selected player with the next player listed for the same slot.

### Results: universal robust core

Six players appear in all seven role-adjusted formations:

- Achraf Hakimi
- Alisson
- Erling Haaland
- Gabriel
- Nuno Mendes
- Virgil van Dijk

Using the broader pre-specified robustness threshold of at least 70% of formations adds:

- Pedri: 6/7
- Vitinha: 6/7
- Moisés Caicedo: 5/7

Role adjustment changes the identity of robust players. For example, Pedri moves from 0/7 OVR-only formations to 6/7 role-adjusted formations, while Alessandro Bastoni falls from 3/7 to 0/7.

### Results: removal margins

| Formation | Mean removal margin | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|
| 3-4-3 | 0.01968 | 0.01351 | 0.00023 | 0.05158 |
| 3-5-2 | 0.01733 | 0.02092 | 0.00023 | 0.03979 |
| 4-1-2-1-2 | 0.01953 | 0.02174 | 0.00356 | 0.03979 |
| 4-2-3-1 | 0.01893 | 0.01351 | 0.00314 | 0.05158 |
| 4-3-3 | 0.02169 | 0.02174 | 0.00356 | 0.05158 |
| 4-4-2 | 0.01829 | 0.02092 | 0.00356 | 0.03979 |
| 5-3-2 | 0.01733 | 0.02092 | 0.00023 | 0.03979 |

### Interpretation

Repeated selection and indispensability are different ideas. A player may appear in every optimized formation while still having a close replacement. The small margins show that several elite-player choices are only weakly dominant over the next-best feasible alternative.

### How this result can be used

- It adds nuance to the “robust core” result.
- It prevents overclaiming that universally selected players are irreplaceable.
- It is useful for discussing roster depth: a small margin implies that an injury or unavailability may have little effect on the optimized score.

---

## Stage 4.4A — Team attribute coverage diagnostic

### Purpose

Stage 4.4A describes the PAC, SHO, PAS, DRI, DEF, and PHY profiles produced by the existing OVR-only and role-adjusted lineups.

### Method

- Estimate each functional role's empirical six-attribute profile from players officially labeled in that role.
- Average the required slot profiles to construct a formation-specific demand profile.
- Compare each selected lineup's mean attributes with that empirical benchmark.

This stage is diagnostic only. It does not change the selected players or collapse the six attributes into a single validated performance score.

### Main result

Elite optimized lineups exceed the average formation-demand profile on nearly every attribute. This revealed that minimizing distance to an average demand vector would be a poor optimization objective: it could penalize a lineup simply for being substantially better than average.

### How this result can be used

- Use the plots to describe the attribute shape of different formations and lineups.
- Use the result to justify rejecting “distance to average” as the complementarity objective.
- Do not present closeness to the empirical mean as evidence of better team construction.

---

## Stage 4.4B — Attribute-complementarity extension **(new work)**

### What was added

Stage 4.4B incorporates a team-level coverage term directly into the lineup optimization:

\[
J_f(X)=\sum_i \operatorname{RoleAdjustedQuality}_i+\gamma C_f(X).
\]

The coverage term is

\[
C_f(X)=\sum_a w_{fa}\,g\!\left(\sum_i x_{ia}\right),
\]

where

- player attributes are converted to percentiles within the outfield candidate pool,
- \(w_{fa}\) is a formation-specific attribute weight derived from the empirical Stage 4.4A demand profile,
- \(g\) is a concave diminishing-returns function, and
- \(\gamma\) controls the quality–coverage tradeoff.

Three normalized concave functions were tested:

- square root,
- logarithmic, and
- saturating exponential.

The sensitivity grid used \(\gamma\in\{0,0.05,0.10,0.25,0.50,1.00\}\) for all seven formations, producing 126 mixed-integer optimization runs.

### Critical validation

At \(\gamma=0\), Stage 4.4B must reduce exactly to Stage 4.2. All seven formations passed both checks:

- identical selected player sets, and
- identical role-adjusted-quality totals.

This confirms that any later lineup changes come from the new coverage term rather than an accidental change in eligibility or scoring.

### Results

At \(\gamma=0.10\):

- All four-defender formations retain their Stage 4.2 player sets for all three concave functions.
- The 3-5-2, 3-4-3, and 5-3-2 each change one center back.
- Average role-adjusted-quality loss across functions is only 0.00010–0.00182 on outfield totals of approximately 8.4–8.7.

At \(\gamma=0.25\):

- The exponential function changes an average of 1.0 player per formation.
- The square-root and logarithmic functions change an average of 2.14 players.
- Mean quality loss ranges from approximately 0.006 to 0.021, depending on the function.

Larger \(\gamma\) values produce progressively greater coverage gains and greater role-adjusted-quality losses, as intended.

### Interpretation

The model successfully exposes a quality–coverage frontier. Low coverage weights leave most lineups unchanged; stronger weights select somewhat different players in exchange for more balanced attribute coverage.

However, there is no empirical basis yet for declaring one \(\gamma\) or one concave function to be the correct football model. These settings form a sensitivity analysis, not a validated ranking of teams.

### How this result can be used

- Use it in the longer paper as an exploratory extension.
- Use the \(\gamma=0\) reproduction test as evidence of correct implementation.
- Use the tradeoff curves to show how much individual quality must be sacrificed to obtain greater attribute coverage.
- Do not use it as proof that complementarity-aware teams perform better in real matches.

---

## Stage 4.4C — External roster sanity check **(new work)**

### What was added

Stage 4.4C tests whether the Stage 4.4B coverage metric assigns unusually high scores to two elite roster constructions already available in the project:

- Spain 2026 model-selected 4-3-3, and
- Real Madrid 2026–27 model-selected 4-3-3.

Each lineup was compared with 3,000 randomly generated, formation-valid global XIs whose mean OVR was within 0.25 points of the elite lineup. The comparison was repeated for all three concave functions.

This is a descriptive external sanity check, not match-outcome validation.

### Results

| Roster | Function | Coverage percentile among OVR-matched random XIs |
|---|---|---:|
| Spain 2026 | Square root | 30.97 |
| Spain 2026 | Logarithmic | 29.17 |
| Spain 2026 | Exponential | 21.77 |
| Real Madrid 2026–27 | Square root | 50.53 |
| Real Madrid 2026–27 | Logarithmic | 50.93 |
| Real Madrid 2026–27 | Exponential | 53.63 |

### Interpretation

Spain scores below the middle of the matched-random distribution, while Real Madrid scores approximately at its center. The pattern is consistent across the three concave functions.

Therefore, this first external check does **not** support the claim that high static-attribute coverage, as currently defined, is a distinguishing feature of elite real-world roster construction. This is a useful negative result: it prevents the project from presenting an internally convenient metric as externally validated.

Possible explanations include:

- Six broad EAFC attributes may be too coarse to capture true complementarity.
- Team complementarity may depend on pairwise interactions, tactics, and event data rather than marginal attribute totals.
- The model-selected Spain and Real Madrid lineups are not direct observations of actual starting-XI choices across many matches.
- OVR-matched global teams can combine elite specialists unrealistically because transfer, nationality, salary, and availability constraints are absent.

### How this result can be used

- Report it honestly in development notes or a limitations section.
- Use it to motivate a stronger complementarity definition for the full paper.
- Do not include it as positive evidence in the Sloan abstract.
- Do not claim that Stage 4.4B has been externally validated.

---

## Stage 3.5 accuracy discrepancy resolved

Two 34-feature logistic-regression accuracies appear in the repository:

- **74.69%**: the class-balanced model in `stage3_5C_logistic_34features_functional.py`.
- **77.22%**: an earlier unweighted model in `stage3_5_logistic_functional_positions.py`.

Stages 3.6B and 3.6C use `class_weight="balanced"`, so their learned role probabilities—and therefore the current Stage 4 pipeline—are based on the class-balanced formulation. The internally consistent accuracy to report for this pipeline is **74.69%**.

---

## Current paper-level conclusions

### Supported by completed results

1. OVR-only and role-adjusted optimization are not equivalent: role adjustment changes four to six starters across every tested formation.
2. The change costs only 0.55–1.64 points of mean OVR.
3. Several players remain robust across formations, but small removal margins show that persistence does not imply indispensability.
4. The complementarity extension correctly produces a tunable quality–coverage tradeoff.

### Not supported yet

1. Role-adjusted lineups win more matches.
2. Higher static-attribute coverage produces better teams.
3. One value of \(\lambda\), \(\gamma\), or one concave function is the true football weighting.
4. Six broad EAFC attributes capture real player chemistry.

### Recommended use for the Sloan abstract

Center the abstract on Stages 4.1–4.3:

- seven-formation OVR baseline,
- role-adjusted selection,
- four to six changed starters,
- small OVR cost,
- cross-formation robustness, and
- nuanced removal-margin result.

Treat Stage 4.4B as exploratory future work and omit Stage 4.4C unless space is needed for an unusually explicit limitation. Exclude the uncalibrated Stage 5 match simulations from the core evidence.
