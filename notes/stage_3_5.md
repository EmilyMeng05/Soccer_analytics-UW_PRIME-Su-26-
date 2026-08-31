# Stage 3.5 — Functional Position Validation

## Goal

The goal of Stage 3.5 was to test whether soccer positions can be represented more cleanly using **functional roles** rather than the original exact EAFC position labels, and then investigate whether **Preferred Foot** helps explain remaining left/right positional ambiguity.

Earlier experiments showed that several left/right position pairs had very similar player attribute profiles:

- Left Winger and Right Winger
- Left Midfielder and Right Midfielder
- Left Back and Right Back

This suggested that player skill attributes may describe **what functional role a player performs** more strongly than **which side of the field they perform it on**.

Stage 3.5 therefore asks three main questions:

1. Does merging left/right positions into broader functional roles improve position prediction?
2. Do 34 detailed player attributes provide more positional information than the six broad EAFC attributes?
3. Does Preferred Foot explain some of the remaining difficulty in distinguishing left- and right-sided positions?

---

# Functional Position Labels

The original 11 exact position labels were reduced to 8 functional roles.

The following positions were merged:

```text
Left Winger + Right Winger → Winger
Left Midfielder + Right Midfielder → Wide Midfielder
Left Back + Right Back → Fullback
```

The remaining positions were kept unchanged:

```text
Attacking Midfielder
Center Back
Central Midfielder
Defensive Midfielder
Striker
```

The final eight functional roles are:

```text
Attacking Midfielder
Center Back
Central Midfielder
Defensive Midfielder
Fullback
Striker
Wide Midfielder
Winger
```

---

# Dataset

The six-feature experiments used:

```text
data/processed/cleaned_eafc26_outfield_players.csv
```

The 34-feature experiments used:

```text
data/processed/eafc26_outfield_full_features.csv
```

Both contained **11,872 outfield players**.

Goalkeepers were excluded because their attributes and responsibilities differ substantially from outfield roles.

## Functional Position Distribution

| Functional Position | Players |
|---|---:|
| Center Back | 2,516 |
| Fullback | 2,026 |
| Striker | 1,805 |
| Wide Midfielder | 1,583 |
| Central Midfielder | 1,569 |
| Defensive Midfielder | 1,090 |
| Attacking Midfielder | 767 |
| Winger | 516 |

The classes remain imbalanced, so Macro F1 is important in addition to accuracy.

---

# Experimental Design

| Stage | Model | Features | Labels |
|---|---|---:|---:|
| 3.5A | Logistic Regression | 6 broad | 8 functional |
| 3.5B | Random Forest | 6 broad | 8 functional |
| 3.5C | Logistic Regression | 34 detailed | 8 functional |
| 3.5D | Random Forest | 34 detailed | 8 functional |
| 3.5E | Logistic Regression | 34 detailed vs. 34 detailed + Preferred Foot | 11 exact |

Stages 3.5A–D test **functional-role prediction**.

Stage 3.5E returns to the original 11 exact positions and tests whether **Preferred Foot specifically helps resolve left/right positional distinctions**.

---

# Evaluation Metrics

## Top-1 Accuracy

The percentage of players where the model's highest-ranked predicted role matches the official role.

## Macro F1

F1 is calculated separately for each class and averaged equally across classes.

This is particularly useful because the dataset is imbalanced.

## Weighted F1

F1 weighted by the number of players in each class.

## Top-2 Accuracy

The prediction is counted as correct if the official role appears among the two highest-ranked predictions.

## Top-3 Accuracy

The prediction is counted as correct if the official role appears among the three highest-ranked predictions.

Top-k metrics are especially relevant because many soccer players may plausibly resemble more than one role.

---

# Stage 3.5A — Logistic Regression, 6 Broad Features

## Method

Features:

```text
PAC
SHO
PAS
DRI
DEF
PHY
```

The features were standardized with `StandardScaler`.

Logistic Regression used balanced class weights and an 80/20 stratified train/test split.

## Results

```text
Top-1 Accuracy: 0.6695
Macro F1:       0.6032
Weighted F1:    0.6737
Top-2 Accuracy: 0.8627
Top-3 Accuracy: 0.9554
```

Five-fold stratified cross-validation:

| Metric | Mean | Std |
|---|---:|---:|
| Accuracy | 0.6688 | 0.0039 |
| Macro F1 | 0.6031 | 0.0069 |
| Weighted F1 | 0.6736 | 0.0039 |

## Interpretation

Using only six broad attributes, Logistic Regression correctly predicts the player's functional role about **66.95%** of the time.

The cross-validation accuracy of **66.88%** is almost identical, suggesting that performance is stable rather than dependent on one favorable train/test split.

Top-2 accuracy reaches **86.27%** and Top-3 reaches **95.54%**.

This suggests that even when the model's first prediction is incorrect, the official role is usually among a small group of plausible roles.

---

# Stage 3.5B — Random Forest, 6 Broad Features

## Method

The same six broad features were used with:

```text
RandomForestClassifier
n_estimators = 300
class_weight = "balanced"
random_state = 42
```

Scaling was not required.

## Results

```text
Top-1 Accuracy: 0.6699
Macro F1:       0.5903
Weighted F1:    0.6694
Top-2 Accuracy: 0.8632
Top-3 Accuracy: 0.9516
```

## Feature Importance

| Feature | Importance |
|---|---:|
| DEF | 0.2383 |
| SHO | 0.1784 |
| PAC | 0.1565 |
| PAS | 0.1493 |
| PHY | 0.1433 |
| DRI | 0.1341 |

## Interpretation

Random Forest and Logistic Regression have almost identical Top-1 accuracy:

```text
Logistic Regression: 66.95%
Random Forest:        66.99%
```

However, Logistic Regression has a higher Macro F1:

```text
Logistic Regression: 0.6032
Random Forest:        0.5903
```

Therefore, increasing classifier complexity does not provide a meaningful improvement when only six broad attributes are available.

`DEF` is the most influential Random Forest feature, consistent with defensive ability strongly separating defensive and attacking functional roles.

---

# Stage 3.5C — Logistic Regression, 34 Detailed Features

## Method

The original six broad features were retained:

```text
PAC
SHO
PAS
DRI
DEF
PHY
```

Twenty-eight detailed attributes were added:

```text
Acceleration
Sprint Speed
Positioning
Finishing
Shot Power
Long Shots
Volleys
Penalties
Vision
Crossing
Free Kick Accuracy
Short Passing
Long Passing
Curve
Agility
Balance
Reactions
Ball Control
Dribbling
Composure
Interceptions
Heading Accuracy
Standing Tackle
Sliding Tackle
Jumping
Stamina
Strength
Aggression
```

This produced **34 total features**.

## Results

```text
Top-1 Accuracy: 0.7469
Macro F1:       0.6831
Weighted F1:    0.7495
Top-2 Accuracy: 0.9192
Top-3 Accuracy: 0.9756
```

## Interpretation

Adding detailed attributes substantially improves performance.

Compared with Stage 3.5A:

```text
Top-1:    66.95% → 74.69%
Macro F1: 60.32% → 68.31%
Top-2:    86.27% → 91.92%
Top-3:    95.54% → 97.56%
```

The six aggregate ratings therefore hide positional information that remains available in the underlying detailed attributes.

For example, two players can have similar `PAS` ratings while differing substantially in:

```text
Vision
Crossing
Short Passing
Long Passing
Curve
```

These detailed differences can help identify functional roles.

---

# Stage 3.5D — Random Forest, 34 Detailed Features

## Method

The same 34 features used in Stage 3.5C were used with the Random Forest classifier.

## Results

```text
Top-1 Accuracy: 0.7562
Macro F1:       0.6613
Weighted F1:    0.7479
Top-2 Accuracy: 0.9154
Top-3 Accuracy: 0.9718
```

## Top 15 Feature Importances

| Feature | Importance |
|---|---:|
| Vision | 0.0621 |
| DEF | 0.0573 |
| Heading Accuracy | 0.0505 |
| Sliding Tackle | 0.0461 |
| Interceptions | 0.0460 |
| Standing Tackle | 0.0433 |
| Finishing | 0.0417 |
| Positioning | 0.0414 |
| Long Passing | 0.0385 |
| SHO | 0.0342 |
| Jumping | 0.0319 |
| PAC | 0.0317 |
| Crossing | 0.0314 |
| Strength | 0.0305 |
| Sprint Speed | 0.0286 |

## Interpretation

Random Forest has the highest Top-1 accuracy at **75.62%**.

However, the 34-feature Logistic Regression model performs better on:

- Macro F1
- Weighted F1
- Top-2 accuracy
- Top-3 accuracy

Therefore, neither model completely dominates.

Random Forest is slightly stronger for a single hard prediction, while Logistic Regression is stronger for balanced class performance and ranked alternative roles.

---

# Overall Stage 3.5A–D Results

| Experiment | Model | Features | Top-1 | Macro F1 | Weighted F1 | Top-2 | Top-3 |
|---|---|---:|---:|---:|---:|---:|---:|
| 3.5A | Logistic Regression | 6 | 0.6695 | 0.6032 | 0.6737 | 0.8627 | 0.9554 |
| 3.5B | Random Forest | 6 | 0.6699 | 0.5903 | 0.6694 | 0.8632 | 0.9516 |
| 3.5C | Logistic Regression | 34 | 0.7469 | **0.6831** | **0.7495** | **0.9192** | **0.9756** |
| 3.5D | Random Forest | 34 | **0.7562** | 0.6613 | 0.7479 | 0.9154 | 0.9718 |

---

# Main Finding 1 — Detailed Attributes Matter More Than Model Complexity

The largest improvement comes from moving from six broad attributes to 34 detailed attributes.

For Logistic Regression:

```text
6 features:  0.6695
34 features: 0.7469
```

This is an improvement of approximately **7.7 percentage points**.

For Random Forest:

```text
6 features:  0.6699
34 features: 0.7562
```

This is an improvement of approximately **8.6 percentage points**.

In comparison, switching between Logistic Regression and Random Forest while keeping the same feature representation changes performance much less.

This suggests:

> **Better player representation contributes more to functional-role prediction than simply increasing classifier complexity.**

---

# Main Finding 2 — Functional Roles Are More Appropriate Than Exact Left/Right Labels

Earlier exact-position models struggled especially with:

```text
Left Winger vs Right Winger
Left Midfielder vs Right Midfielder
Left Back vs Right Back
```

Stage 3.4 also showed that these left/right counterparts had very similar skill fingerprints.

The Stage 3.5A–D results support the interpretation that player attributes describe:

> **what type of role a player performs**

more strongly than:

> **which side of the field they perform it on.**

This motivated a more targeted laterality experiment.

---

# Stage 3.5E — Laterality Validation with Preferred Foot

## Goal

Stage 3.5E tests whether **Preferred Foot** helps explain why the model struggles with left/right position labels.

Instead of using the 8 functional roles, this experiment returns to the original **11 exact positions**.

Two controlled Logistic Regression models were compared:

```text
E1:
34 skill attributes
→ 11 exact positions

E2:
34 skill attributes + Preferred Foot
→ 11 exact positions
```

The same model type, same train/test split, and same skill features were used in both conditions.

This makes Preferred Foot the main changed variable.

---

## Preferred Foot Distribution

```text
Right: 8,708
Left:  3,164
```

All **11,872 outfield players** were used.

---

## Exact Position Distribution

| Position | Players |
|---|---:|
| Center Back | 2,516 |
| Striker | 1,805 |
| Central Midfielder | 1,569 |
| Defensive Midfielder | 1,090 |
| Right Back | 1,045 |
| Left Back | 981 |
| Left Midfielder | 816 |
| Right Midfielder | 767 |
| Attacking Midfielder | 767 |
| Left Winger | 263 |
| Right Winger | 253 |

---

## Stage 3.5E Overall Results

| Experiment | Top-1 | Macro F1 | Weighted F1 | Top-2 | Top-3 |
|---|---:|---:|---:|---:|---:|
| 34 Skills | 0.6307 | 0.4971 | 0.6367 | 0.8404 | 0.9166 |
| 34 Skills + Preferred Foot | **0.6977** | **0.5690** | **0.7011** | **0.8640** | **0.9347** |

Adding Preferred Foot improved:

```text
Top-1 Accuracy: +0.0669
Macro F1:       +0.0718
```

This is approximately a **6.7 percentage-point increase in exact-position accuracy** from adding one laterality variable.

---

# Stage 3.5E Cross-Validation

Five-fold stratified cross-validation produced:

| Experiment | Accuracy Mean | Accuracy Std | Macro F1 Mean | Macro F1 Std | Weighted F1 Mean | Weighted F1 Std |
|---|---:|---:|---:|---:|---:|---:|
| 34 Skills | 0.6421 | 0.0053 | 0.5042 | 0.0073 | 0.6455 | 0.0043 |
| 34 Skills + Preferred Foot | **0.7022** | 0.0099 | **0.5757** | 0.0133 | **0.7024** | 0.0088 |

The improvement therefore persists across multiple train/test splits and is not limited to the random_state=42 test set.

---

# Left/Right Confusion Analysis

The main purpose of Stage 3.5E was not simply improving overall accuracy.

The key question was:

> Does Preferred Foot reduce confusion between left/right counterparts?

## Results

| Pair | Skills Only Confusion | With Preferred Foot | Reduction |
|---|---:|---:|---:|
| Left Back ↔ Right Back | 159 | **14** | **91.2%** |
| Left Midfielder ↔ Right Midfielder | 65 | 52 | 20.0% |
| Left Winger ↔ Right Winger | 24 | 21 | 12.5% |
| **All three pairs** | **248** | **87** | **64.9%** |

Overall left/right confusion dropped from:

```text
248 → 87
```

which is a **64.92% aggregate reduction**.

However, most of this improvement comes specifically from the fullback pair.

---

# Fullback Result

The strongest Stage 3.5E result is the Left Back / Right Back distinction.

Without Preferred Foot:

```text
LB ↔ RB confusions = 159
```

With Preferred Foot:

```text
LB ↔ RB confusions = 14
```

This is a **91.2% reduction**.

The individual F1 scores also increased dramatically:

```text
Left Back:
0.5124 → 0.8894

Right Back:
0.4884 → 0.8164
```

This suggests that the 34 skill attributes were already reasonably good at identifying:

> **This player has a fullback-like skill profile.**

However, those skill attributes alone were poor at deciding:

> **Is this player specifically a Left Back or Right Back?**

Preferred Foot provides substantial information for this second question.

---

# Wide Midfielder Result

The improvement is much smaller for Left Midfielder and Right Midfielder.

```text
LM ↔ RM confusion:
65 → 52

Reduction:
20.0%
```

F1 scores:

```text
Left Midfielder:
0.2267 → 0.2460

Right Midfielder:
0.2437 → 0.2267
```

Preferred Foot slightly improves Left Midfielder performance but slightly decreases Right Midfielder F1.

Therefore, Preferred Foot does **not** strongly resolve the LM/RM distinction.

---

# Winger Result

The winger distinction remains difficult.

```text
LW ↔ RW confusion:
24 → 21

Reduction:
12.5%
```

F1 scores:

```text
Left Winger:
0.1656 → 0.2333

Right Winger:
0.1457 → 0.1299
```

Left Winger improves somewhat, but Right Winger decreases slightly.

Preferred Foot therefore does **not** fully explain the difficulty of distinguishing left and right wingers.

---

# Important Interpretation of Stage 3.5E

It would be too strong to conclude:

> "Preferred Foot explains left/right positions."

The results instead support a more careful interpretation:

> **Preferred Foot contains substantial laterality information, but its usefulness is role-dependent. It almost resolves the Left Back / Right Back distinction, while Left/Right Midfielders and especially Left/Right Wingers remain difficult to distinguish.**

The aggregate **64.92% reduction in lateral confusion** should therefore be interpreted carefully because most of the improvement comes from fullbacks.

Out of the 161 total lateral errors removed:

```text
145 came from LB/RB
13 came from LM/RM
3 came from LW/RW
```

Therefore, roughly 90% of the reduction comes from the fullback distinction.

---

# Functional Roles vs Exact Positions

A useful comparison now becomes possible.

The best 8-role functional models achieved:

```text
Logistic, 34 features:
74.69% Top-1

Random Forest, 34 features:
75.62% Top-1
```

The 11-position exact model achieved:

```text
34 skills:
63.07%

34 skills + Preferred Foot:
69.77%
```

Even after adding Preferred Foot, exact-position accuracy remains lower than the functional-role models.

This suggests that the earlier exact-position difficulty was **not only caused by missing footedness information**.

Some left/right distinctions, especially wide attacking and midfield positions, remain difficult to recover from the current player attributes.

---

# Main Finding 3 — Role and Side Can Be Treated as Related but Distinct Problems

Taken together, Stage 3.4, Stage 3.5A–D, and Stage 3.5E support a useful conceptual framework:

```text
SKILL PROFILE
    ↓
Functional Role
Fullback / Winger / CM / etc.

        +

LATERALITY INFORMATION
Preferred Foot
    ↓
Additional information about
Left / Right assignment
```

However, laterality does not affect every role equally.

A more precise conclusion is:

> **Skill attributes primarily describe functional role, while laterality provides additional side-specific information, especially for fullbacks.**

This is more defensible than treating every exact EAFC position as a completely separate skill archetype.

---

# Main Finding 4 — Multiple Plausible Roles Still Matter

For Stage 3.5C:

```text
Top-1 = 74.69%
Top-2 = 91.92%
Top-3 = 97.56%
```

The official functional role therefore appears among the Logistic Regression model's top three predictions for nearly 98% of players.

This motivates moving away from:

```text
Player → one rigid position
```

toward:

```text
Player → several plausible functional roles
```

This provides the bridge from classification to role suitability.

---

# Main Finding 5 — Logistic Regression vs Random Forest

Random Forest with 34 features has the highest Top-1 functional-role accuracy:

```text
75.62%
```

Logistic Regression with 34 features has better:

```text
Macro F1
Weighted F1
Top-2 Accuracy
Top-3 Accuracy
```

Because later stages care about ranked alternative roles rather than only hard classification, **Stage 3.5C remains the leading candidate for downstream role analysis**.

Stage 3.5D remains an important comparison model.

---

# Important Limitation

High Top-2 or Top-3 accuracy does **not automatically demonstrate player versatility**.

A model may assign meaningful probability to several roles because:

1. the player is genuinely versatile, or
2. the model is uncertain.

These are not equivalent.

Therefore, classifier probability spread or entropy should not automatically be interpreted as versatility.

This needs independent validation.

---

# Next Step — Stage 3.6

Stage 3.6 should move from functional-position classification toward **Role Suitability and Secondary-Role Validation**.

The main candidate model is:

```text
Stage 3.5C
Logistic Regression
34 detailed attributes
8 functional roles
```

The dataset also contains:

```text
Alternative positions
```

This provides an important validation opportunity.

Because Stage 3.5 trained on primary positions, Stage 3.6 can test whether the model's second- and third-ranked functional roles correspond to EAFC's officially listed alternative positions.

For example:

```text
Primary:
Central Midfielder

Alternatives:
Attacking Midfielder
Defensive Midfielder

Model:
1. Central Midfielder
2. Attacking Midfielder
3. Defensive Midfielder
```

If this pattern occurs systematically, then ranked model predictions may capture meaningful player flexibility rather than simply classifier uncertainty.

Potential evaluation metrics include:

```text
Alternative-position Hit@1
Alternative-position Hit@2
Alternative-position Hit@3
Recall@K
```

After validating secondary roles, Stage 3.6 can begin constructing a more defensible role-suitability score.

---

# Stage 3.5 Conclusion

Stage 3.5 provides several important results.

First, moving from six broad ratings to 34 detailed player attributes substantially improves functional-role prediction.

Second, the improvement from richer features is larger than the improvement from changing classifier complexity.

Third, the 8 functional-role representation performs better than the original 11 exact-position representation.

Fourth, adding Preferred Foot improves exact-position prediction from approximately **63.1% to 69.8%** and reduces aggregate left/right confusion by **64.9%**.

However, this laterality effect is highly role-dependent:

```text
LB/RB confusion reduction: 91.2%
LM/RM confusion reduction: 20.0%
LW/RW confusion reduction: 12.5%
```

Therefore, the most defensible interpretation is:

> **Detailed technical, physical, attacking, and defensive attributes provide a strong representation of functional soccer role. Preferred Foot adds meaningful side-specific information, particularly for fullbacks, but does not fully explain all left/right positional distinctions. This supports separating functional-role representation from exact side assignment rather than treating every left/right position as a completely distinct player archetype.**

The high Top-2 and Top-3 functional-role accuracy also motivates representing players through several plausible roles rather than one rigid position.

Stage 3.6 will build on this result by validating whether ranked secondary-role predictions recover officially listed alternative positions before those predictions are used as evidence of player versatility or role suitability.
