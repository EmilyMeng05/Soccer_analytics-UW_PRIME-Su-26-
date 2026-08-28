# Stage 3.5 --- Functional Position Validation

## Goal

The goal of Stage 3.5 was to test whether soccer positions can be
represented more cleanly using **functional roles** rather than the
original exact EAFC position labels.

Earlier experiments showed that some left/right position pairs had very
similar player attribute profiles:

-   Left Winger and Right Winger
-   Left Midfielder and Right Midfielder
-   Left Back and Right Back

This suggested that player skill attributes may describe **what
functional role a player performs** more strongly than **which side of
the field they perform it on**.

Stage 3.5 therefore asks:

1.  Does merging left/right positions into broader functional roles
    improve position prediction?
2.  Do 34 detailed player attributes provide more positional information
    than the six broad EAFC attributes?

------------------------------------------------------------------------

## Functional Position Labels

The original 11 position labels were reduced to 8 functional roles:

-   Left Winger + Right Winger → **Winger**
-   Left Midfielder + Right Midfielder → **Wide Midfielder**
-   Left Back + Right Back → **Fullback**
-   Attacking Midfielder → unchanged
-   Center Back → unchanged
-   Central Midfielder → unchanged
-   Defensive Midfielder → unchanged
-   Striker → unchanged

The final eight functional roles are:

``` text
Attacking Midfielder
Center Back
Central Midfielder
Defensive Midfielder
Fullback
Striker
Wide Midfielder
Winger
```

------------------------------------------------------------------------

## Dataset

The six-feature experiments used:

``` text
data/processed/cleaned_eafc26_outfield_players.csv
```

The 34-feature experiments used:

``` text
data/processed/eafc26_outfield_full_features.csv
```

Both contained **11,872 outfield players** after preprocessing.

Goalkeepers were excluded because their attributes and responsibilities
differ substantially from outfield roles.

### Functional Position Distribution

  Functional Position      Players
  ---------------------- ---------
  Center Back                2,516
  Fullback                   2,026
  Striker                    1,805
  Wide Midfielder            1,583
  Central Midfielder         1,569
  Defensive Midfielder       1,090
  Attacking Midfielder         767
  Winger                       516

The classes remain imbalanced, so Macro F1 is important in addition to
accuracy.

------------------------------------------------------------------------

## Experimental Design

  Stage   Model                      Features         Labels
  ------- --------------------- ------------- --------------
  3.5A    Logistic Regression         6 broad   8 functional
  3.5B    Random Forest               6 broad   8 functional
  3.5C    Logistic Regression     34 detailed   8 functional
  3.5D    Random Forest           34 detailed   8 functional

This separates the effect of **model complexity** from the effect of
**feature representation**.

### Evaluation Metrics

-   **Top-1 Accuracy:** official functional role equals the
    highest-ranked prediction.
-   **Macro F1:** F1 calculated for each class and averaged equally.
-   **Weighted F1:** F1 weighted by class size.
-   **Top-2 Accuracy:** official role is among the top two predictions.
-   **Top-3 Accuracy:** official role is among the top three
    predictions.

Top-k metrics are useful because players may plausibly resemble more
than one functional role.

------------------------------------------------------------------------

# Stage 3.5A --- Logistic Regression, 6 Broad Features

## Method

Features:

``` text
PAC, SHO, PAS, DRI, DEF, PHY
```

The features were standardized with `StandardScaler`. Logistic
Regression used balanced class weights and an 80/20 stratified
train/test split.

## Results

``` text
Top-1 Accuracy: 0.6695
Macro F1:       0.6032
Weighted F1:    0.6737
Top-2 Accuracy: 0.8627
Top-3 Accuracy: 0.9554
```

Five-fold stratified cross-validation:

  Metric            Mean      Std
  ------------- -------- --------
  Accuracy        0.6688   0.0039
  Macro F1        0.6031   0.0069
  Weighted F1     0.6736   0.0039

## Interpretation

Using only six broad attributes, Logistic Regression correctly predicts
the functional role about **66.95%** of the time.

The cross-validation accuracy of 66.88% is almost identical, suggesting
stable performance across splits.

Top-2 accuracy reaches **86.27%** and Top-3 reaches **95.54%**. Even
when the first prediction is wrong, the official role is usually among a
small set of plausible roles.

------------------------------------------------------------------------

# Stage 3.5B --- Random Forest, 6 Broad Features

## Method

The same six broad features were used with:

``` text
RandomForestClassifier
n_estimators = 300
class_weight = "balanced"
random_state = 42
```

Scaling was not required.

## Results

``` text
Top-1 Accuracy: 0.6699
Macro F1:       0.5903
Weighted F1:    0.6694
Top-2 Accuracy: 0.8632
Top-3 Accuracy: 0.9516
```

## Feature Importance

  Feature     Importance
  --------- ------------
  DEF             0.2383
  SHO             0.1784
  PAC             0.1565
  PAS             0.1493
  PHY             0.1433
  DRI             0.1341

## Interpretation

Random Forest and Logistic Regression have almost identical Top-1
accuracy:

``` text
Logistic Regression: 66.95%
Random Forest:        66.99%
```

However, Logistic Regression has a higher Macro F1 (0.6032 vs. 0.5903).

Therefore, increasing classifier complexity does not provide a
meaningful improvement when only six broad attributes are available.

`DEF` is the most influential Random Forest feature, consistent with
defensive ability strongly separating defensive and attacking functional
roles.

------------------------------------------------------------------------

# Stage 3.5C --- Logistic Regression, 34 Detailed Features

## Method

The original six broad features were retained:

``` text
PAC, SHO, PAS, DRI, DEF, PHY
```

Twenty-eight detailed features were added:

``` text
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

``` text
Top-1 Accuracy: 0.7469
Macro F1:       0.6831
Weighted F1:    0.7495
Top-2 Accuracy: 0.9192
Top-3 Accuracy: 0.9756
```

## Interpretation

Adding detailed attributes substantially improves performance.

Compared with Stage 3.5A:

``` text
Top-1:   66.95% → 74.69%
Macro F1: 60.32% → 68.31%
Top-2:   86.27% → 91.92%
Top-3:   95.54% → 97.56%
```

The six aggregate ratings therefore hide positional information that
remains available in the underlying detailed attributes.

For example, two players can have similar `PAS` values while differing
in Vision, Crossing, Short Passing, Long Passing, and Curve. These
distinctions can help identify different functional roles.

------------------------------------------------------------------------

# Stage 3.5D --- Random Forest, 34 Detailed Features

## Method

The same 34 features were used with the Random Forest classifier.

## Results

``` text
Top-1 Accuracy: 0.7562
Macro F1:       0.6613
Weighted F1:    0.7479
Top-2 Accuracy: 0.9154
Top-3 Accuracy: 0.9718
```

## Top 15 Feature Importances

  Feature              Importance
  ------------------ ------------
  Vision                   0.0621
  DEF                      0.0573
  Heading Accuracy         0.0505
  Sliding Tackle           0.0461
  Interceptions            0.0460
  Standing Tackle          0.0433
  Finishing                0.0417
  Positioning              0.0414
  Long Passing             0.0385
  SHO                      0.0342
  Jumping                  0.0319
  PAC                      0.0317
  Crossing                 0.0314
  Strength                 0.0305
  Sprint Speed             0.0286

## Interpretation

Random Forest has the highest Top-1 accuracy at **75.62%**.

However, the 34-feature Logistic Regression model has better:

-   Macro F1
-   Weighted F1
-   Top-2 accuracy
-   Top-3 accuracy

Therefore, neither model completely dominates.

Random Forest is slightly stronger for a single hard prediction, while
Logistic Regression is stronger for balanced class performance and
ranked alternative roles.

------------------------------------------------------------------------

# Overall Stage 3.5 Results

  -----------------------------------------------------------------------------------------------------
  Experiment   Model          Features        Top-1     Macro F1  Weighted F1        Top-2        Top-3
  ------------ ------------ ---------- ------------ ------------ ------------ ------------ ------------
  3.5A         Logistic              6       0.6695       0.6032       0.6737       0.8627       0.9554
               Regression                                                                  

  3.5B         Random                6       0.6699       0.5903       0.6694       0.8632       0.9516
               Forest                                                                      

  3.5C         Logistic             34       0.7469   **0.6831**   **0.7495**   **0.9192**   **0.9756**
               Regression                                                                  

  3.5D         Random               34   **0.7562**       0.6613       0.7479       0.9154       0.9718
               Forest                                                                      
  -----------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

# Main Findings

## 1. Detailed attributes matter more than model complexity

Moving from six to 34 features produces a much larger improvement than
switching from Logistic Regression to Random Forest.

Logistic Regression:

``` text
6 features:  0.6695
34 features: 0.7469
```

Random Forest:

``` text
6 features:  0.6699
34 features: 0.7562
```

This suggests:

> **Better player representation contributes more to functional-role
> prediction than simply increasing classifier complexity.**

------------------------------------------------------------------------

## 2. Functional roles are more appropriate than exact left/right labels

Earlier exact-position models struggled with left/right counterparts,
while Stage 3.4 showed that these positions have very similar attribute
fingerprints.

The results support the interpretation that player attributes describe
**what type of role a player performs** more strongly than **which side
of the field they perform it on**.

------------------------------------------------------------------------

## 3. Multiple plausible roles matter

For Stage 3.5C:

``` text
Top-1 = 74.69%
Top-2 = 91.92%
Top-3 = 97.56%
```

This motivates moving away from:

``` text
Player → one rigid position
```

toward:

``` text
Player → several plausible functional roles
```

This provides the bridge from classification to role suitability.

------------------------------------------------------------------------

## 4. Logistic Regression vs. Random Forest

Random Forest with 34 features has the best Top-1 accuracy.

Logistic Regression with 34 features has the best Macro F1, Weighted F1,
Top-2, and Top-3 performance.

Because later stages care about ranked alternative roles rather than
only hard classification, **Stage 3.5C is currently the leading
candidate for downstream role analysis**.

Stage 3.5D remains an important comparison model.

------------------------------------------------------------------------

# Important Limitation

High Top-2 or Top-3 accuracy does **not automatically demonstrate player
versatility**.

A model may assign substantial probability to several positions because:

1.  the player is genuinely versatile, or
2.  the model is uncertain.

These are not equivalent.

Therefore, diffuse classifier probabilities should not automatically be
interpreted as a versatility score.

------------------------------------------------------------------------

# Next Step --- Stage 3.6

Stage 3.6 should move from functional-position classification toward
**Role Suitability**.

The main model candidate is:

``` text
Stage 3.5C
Logistic Regression
34 detailed features
8 functional roles
```

Two possible sources of role suitability should be compared:

1.  Model probability for each functional role
2.  Similarity between the player's attribute profile and the Stage 3.4
    role fingerprint

The exact formula should be evaluated rather than chosen arbitrarily.

## Validation Using Alternative Positions

The dataset contains `Alternative positions`.

Because Stage 3.5 was trained using each player's primary position, the
alternative-position field provides an independent secondary target for
testing whether ranked predictions capture meaningful flexibility.

For example:

``` text
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

If secondary model predictions repeatedly recover official alternative
positions, this would provide stronger evidence that the model is
capturing positional flexibility rather than only uncertainty.

Possible metrics include alternative-position Hit/Recall@K.

------------------------------------------------------------------------

# Stage 3.5 Conclusion

Stage 3.5 provides evidence that functional-role prediction improves
substantially when player representation becomes more detailed.

The strongest models achieve approximately:

``` text
75% Top-1 accuracy
92% Top-2 accuracy
97% Top-3 accuracy
```

The major improvement comes from moving from six broad attributes to 34
detailed attributes rather than simply switching to a more complex
classifier.

> **Functional soccer roles can be learned substantially better using
> detailed player attributes than using aggregate ratings alone. The
> improvement from six to 34 features suggests that functional position
> is encoded in the underlying technical, physical, attacking, and
> defensive profile of a player. The high Top-2 and Top-3 performance
> further motivates representing players through several plausible
> functional roles rather than one rigid position.**

Stage 3.6 will build on this result by developing and validating a
player-role suitability framework before moving to team-level lineup
construction.
