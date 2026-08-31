# Stage 3.6 --- Validating Player Flexibility and Building a Multi-Role Representation

## Goal

The original goal of Stage 3.6 was to add two potential flexible
positions to each player using Stage 3.5 model probabilities. However, a
secondary model probability can reflect classifier uncertainty rather
than genuine player versatility.

Therefore, Stage 3.6 first validates secondary-role predictions against
the official `Alternative positions` in the EAFC dataset. The final goal
is to determine whether learned player attributes contain meaningful
information about secondary roles and to build a representation that
separates **official positional eligibility** from **learned role
suitability**.

Stage 3.6 is divided into:

-   **3.6A:** Analyze official alternative positions.
-   **3.6B:** Validate learned secondary-role predictions against
    official alternatives and a frequency baseline.
-   **3.6C:** Build a validated multi-role representation for Stage 4.

------------------------------------------------------------------------

# Stage 3.6A --- Alternative Position Analysis

## Motivation

The full outfield dataset contains an `Alternative positions` field in
addition to primary `Position`. Examples include Salah RM → RW, Mbappé
ST → LW/LM, Rodri CDM → CM, Bellingham CAM → CM, Vitinha CM → CDM/CAM,
and Valverde CM → CDM/RB.

The alternative-position field provides an independent signal of player
flexibility that was not used as the target in Stage 3.5C.

The alternative-position abbreviations were normalized and mapped into
the same eight functional roles used in Stage 3.5:

1.  Attacking Midfielder
2.  Center Back
3.  Central Midfielder
4.  Defensive Midfielder
5.  Fullback
6.  Striker
7.  Wide Midfielder
8.  Winger

## Exact versus functional flexibility

Exact alternatives do not always imply a new functional role. For
example, Left Back → Right Back is an exact-position change, but both
map to `Fullback`. Stage 3.6 therefore distinguishes exact positional
flexibility from functional-role flexibility.

## Results

The dataset contains **11,872 outfield players**.

-   At least one exact alternative: **8,699 (73.27%)**
-   At least one different functional alternative: **8,685 (73.16%)**
-   Alternatives only within the same functional role: **14 (0.12%)**

### Exact alternative count distribution

    Number of alternatives   Players
  ------------------------ ---------
                         0     3,173
                         1     4,146
                         2     3,557
                         3       996

### Functional alternative count distribution

    Number of functional alternatives   Players
  ----------------------------------- ---------
                                    0     3,187
                                    1     4,965
                                    2     3,298
                                    3       422

## Most common functional transitions

  Primary role           Alternative role         Count
  ---------------------- ---------------------- -------
  Fullback               Wide Midfielder          1,812
  Wide Midfielder        Winger                   1,427
  Defensive Midfielder   Central Midfielder       1,067
  Central Midfielder     Defensive Midfielder       998
  Attacking Midfielder   Central Midfielder         726
  Central Midfielder     Attacking Midfielder       652
  Center Back            Fullback                   531
  Winger                 Wide Midfielder            516
  Striker                Attacking Midfielder       417
  Striker                Wide Midfielder            360
  Striker                Winger                     355
  Fullback               Center Back                327

## Interpretation

Positional flexibility is common: about **73%** of outfield players have
an alternative belonging to a different functional role.

However, alternative roles are highly structured. Common pathways such
as Fullback → Wide Midfielder, Wide Midfielder → Winger, DM ↔ CM, and
CAM ↔ CM suggest that primary position alone may already contain
substantial information about likely secondary roles.

This motivates Stage 3.6B:

> Does a player's individual attribute profile identify secondary roles
> better than what can already be inferred from primary functional role?

------------------------------------------------------------------------

# Stage 3.6B --- Secondary Role Validation

## Goal

Stage 3.5C produces probabilities across eight functional roles using
Logistic Regression and 34 player attributes. Rather than automatically
interpreting the highest non-primary probabilities as flexible
positions, Stage 3.6B tests whether those rankings recover official EAFC
alternative roles.

## Method

The validation uses the Stage 3.5C model:

-   Logistic Regression
-   34 player attributes
-   8 functional roles
-   balanced class weights
-   standardized features

Five-fold stratified cross-validation generates **out-of-fold
predictions**. For every player, the model producing their validation
probabilities was not trained on that player.

The player's primary functional role is removed from the ranking, and
the remaining roles are evaluated against their official alternative
functional roles.

## Frequency baseline

A baseline is built using training data only. For each primary
functional role, it ranks alternative roles according to how frequently
they occur among training players with that primary role.

This tests whether individual attributes provide information beyond
simply knowing the player's primary role.

## Evaluation sample

Only players with at least one different official functional alternative
are evaluated:

**8,685 players**

## Metrics

-   **Hit@K:** whether at least one official alternative occurs within
    the top K predictions.
-   **Recall@K:** proportion of official alternatives recovered within
    the top K predictions.
-   **MRR:** how highly the first correct alternative appears.

## Overall results

  -------------------------------------------------------------------------------------------------------
  Method              Hit@1     Recall@1        Hit@2     Recall@2        Hit@3     Recall@3          MRR
  ------------ ------------ ------------ ------------ ------------ ------------ ------------ ------------
  34-feature         0.6974       0.5246       0.8750       0.7460       0.9606       0.8889       0.8242
  model                                                                                      

  Frequency      **0.8629**   **0.6741**   **0.9523**   **0.8335**   **0.9929**   **0.9257**   **0.9228**
  baseline                                                                                   
  -------------------------------------------------------------------------------------------------------

For Hit@1:

``` text
Model    = 69.74%
Baseline = 86.29%
Difference = -16.55 percentage points
```

The hypothesis that individual attributes generally predict official
secondary roles better than primary position alone is therefore **not
supported overall**.

## Performance by primary functional role

  ----------------------------------------------------------------------------------------------------------------------
  Primary Role   Players  Model Hit@1   Baseline  Model Hit@2   Baseline  Model Hit@3   Baseline    Model MRR   Baseline
                                           Hit@1                   Hit@2                   Hit@3                     MRR
  ------------ --------- ------------ ---------- ------------ ---------- ------------ ---------- ------------ ----------
  Attacking          767       0.5750     0.9465       0.7849     0.9765       0.9465     1.0000       0.7472     0.9694
  Midfielder                                                                                                  

  Center Back        672       0.5938     0.7902   **0.9360**     0.7917   **0.9985**     0.9940       0.7861     0.8597

  Central          1,414   **0.7546**     0.7058       0.9194     0.9526       0.9625     0.9887   **0.8602**     0.8435
  Midfielder                                                                                                  

  Defensive        1,085       0.7742     0.9834       0.9419     0.9945       0.9972     0.9982       0.8771     0.9906
  Midfielder                                                                                                  

  Fullback         1,932       0.5756     0.9379       0.7257     0.9922       0.9022     0.9928       0.7334     0.9670

  Striker            716   **0.6425**     0.5824       0.8925     0.9246       0.9972     0.9986   **0.8030**     0.7784

  Wide             1,583       0.8181     0.9015   **0.9450**     0.9280       0.9672     0.9842       0.8962     0.9374
  Midfielder                                                                                                  

  Winger             516       0.8585     1.0000       0.9864     1.0000       0.9981     1.0000       0.9268     1.0000
  ----------------------------------------------------------------------------------------------------------------------

## Where individual attributes help

### Central Midfielder

The model beats the baseline at Hit@1 and MRR:

``` text
Model Hit@1    = 75.46%
Baseline Hit@1 = 70.58%

Model MRR      = 0.8602
Baseline MRR   = 0.8435
```

Central Midfielders have several possible alternative pathways.
Individual attributes can therefore help distinguish whether a
particular CM is more defensive, attacking, wide-oriented, or suited to
another role.

### Striker

The model also beats the baseline at Hit@1 and MRR:

``` text
Model Hit@1    = 64.25%
Baseline Hit@1 = 58.24%

Model MRR      = 0.8030
Baseline MRR   = 0.7784
```

Strikers have multiple secondary pathways including Attacking
Midfielder, Wide Midfielder, and Winger.

### Wide Midfielder

The baseline wins at Hit@1, but the model slightly wins at Hit@2:

``` text
Hit@1:
Model    = 81.81%
Baseline = 90.15%

Hit@2:
Model    = 94.50%
Baseline = 92.80%
```

This suggests attributes may help rank multiple plausible alternatives
even when they do not improve the most common first prediction.

## Main finding

> **Primary position explains much of the structure in official
> alternative positions. Individual player attributes do not outperform
> this information overall, but they provide additional value for some
> heterogeneous roles, particularly Central Midfielder and Striker.**

------------------------------------------------------------------------

# Stage 3.6C --- Validated Multi-Role Player Representation

## Motivation

Because Stage 3.6B showed that model probabilities do not independently
predict official alternatives better overall, the model should not
decide whether a player is eligible to play another role.

Instead, Stage 3.6C separates:

### Official eligibility

Which roles is the player officially recognized as being able to play?

``` text
Primary Position + Official Alternative Positions → Eligibility
```

### Learned role suitability

How strongly does the player's 34-attribute profile resemble each
functional role?

``` text
34-feature functional-role model → Suitability
```

Thus:

\[ oxed{ ext{Eligibility}(i,r)} \]

and

\[ oxed{ ext{Suitability}(i,r)} \]

remain separate.

No arbitrary probability threshold is used.

## Eligibility distribution

    Number of eligible functional roles   Players
  ------------------------------------- ---------
                                      1     3,187
                                      2     4,965
                                      3     3,298
                                      4       422

Most players are officially eligible for more than one functional role,
giving Stage 4 meaningful flexibility while keeping eligibility grounded
in the source data.

------------------------------------------------------------------------

# Example Multi-Role Profiles

## Jude Bellingham

Official: - Primary: Attacking Midfielder - Alternative: Central
Midfielder

Learned suitability:

  Role                     Suitability
  ---------------------- -------------
  Central Midfielder        **0.6064**
  Attacking Midfielder          0.3794
  Defensive Midfielder          0.0112

The learned model sees Bellingham as more CM-like than CAM-like, even
though CAM is his official primary role.

## Federico Valverde

Official: - Primary: Central Midfielder - Alternatives: Defensive
Midfielder, Fullback

Learned suitability:

  Role                     Suitability
  ---------------------- -------------
  Central Midfielder        **0.6911**
  Defensive Midfielder          0.2756
  Fullback                      0.0111

EAFC recognizes Fullback as an eligible alternative, but the learned
profile is much more strongly CM/DM-like. This demonstrates why
eligibility and suitability should remain separate.

## Achraf Hakimi

Official: - Primary: Fullback - Alternative: Wide Midfielder

Learned suitability:

  Role                Suitability
  ----------------- -------------
  Fullback             **0.8338**
  Wide Midfielder          0.1114
  Winger                   0.0218

## Mohamed Salah

Official: - Primary: Wide Midfielder - Alternative: Winger

Learned suitability:

  Role                     Suitability
  ---------------------- -------------
  Winger                    **0.6369**
  Attacking Midfielder          0.1658
  Wide Midfielder               0.1640

The model sees Salah's attributes as substantially more Winger-like than
Wide-Midfielder-like.

## Kylian Mbappé

Official: - Primary: Striker - Alternatives: Winger, Wide Midfielder

Learned suitability:

  Role                     Suitability
  ---------------------- -------------
  Striker                   **0.5587**
  Winger                        0.3156
  Wide Midfielder               0.0671
  Attacking Midfielder          0.0587

## Rodri

Official: - Primary: Defensive Midfielder - Alternative: Central
Midfielder

Learned suitability:

  Role                     Suitability
  ---------------------- -------------
  Defensive Midfielder      **0.7348**
  Central Midfielder            0.2606
  Center Back                   0.0028

Rodri provides a clean example of official multi-role eligibility
combined with a learned profile that strongly favors one of those roles.

------------------------------------------------------------------------

# Stage 3.6C Outputs

## Player-level representation

``` text
data/processed/eafc26_players_multirole_representation.csv
```

This preserves the player-level dataset while adding:

-   Primary Functional Role
-   Official Alternative Functional Roles
-   Official Eligible Functional Roles
-   Number of Official Eligible Functional Roles
-   suitability probability for each of the eight functional roles
-   learned role rankings
-   Primary Functional Role Suitability
-   Best Official Alternative Functional Role
-   Best Official Alternative Role Suitability
-   Second Official Alternative Functional Role
-   Second Official Alternative Role Suitability

## Player × Role representation

``` text
results/stage_3_6/C_multirole_representation/
player_role_eligibility_and_suitability.csv
```

Each row represents one player × one functional role and records:

-   player ID and name
-   OVR
-   original position
-   functional role
-   whether it is the primary functional role
-   whether it is an official alternative functional role
-   whether the player is officially eligible
-   learned role-suitability probability

This long-format dataset is the main bridge into Stage 4 lineup
optimization.

------------------------------------------------------------------------

# Important Interpretation of Model Probabilities

The model probabilities should **not** be interpreted as literal
probabilities that a player can successfully play a position.

For example:

``` text
Role Suitability Probability - Fullback = 0.70
```

does not mean that the player has a 70% probability of successfully
playing Fullback.

The probabilities come from a classifier trained to distinguish EAFC
functional-position labels. They are therefore interpreted as **relative
learned role-suitability signals** describing how strongly a player's
attribute profile resembles each functional role.

Stage 3.6B demonstrates why these probabilities should not independently
determine eligibility.

------------------------------------------------------------------------

# Why the Original Flexible-Position Method Was Changed

The original Stage 3.6 design would:

1.  remove the player's primary role,
2.  take the two highest remaining model probabilities,
3.  label them `Flexible Position 1` and `Flexible Position 2`,
4.  potentially use a fixed threshold such as 0.10.

This approach was not retained.

A secondary probability may reflect classifier uncertainty rather than
genuine versatility, and a threshold such as 0.10 would be arbitrary
without validation.

More importantly, Stage 3.6B showed that official alternative positions
are predicted substantially better overall from primary-position
structure than from individual model probabilities.

The final design therefore uses:

``` text
Official primary + alternative positions
                ↓
            ELIGIBILITY

34-feature functional-role model
                ↓
            SUITABILITY
```

The model ranks suitability; it does not invent eligibility.

------------------------------------------------------------------------

# Main Conclusions

### 1. Positional flexibility is common

**73.16%** of the 11,872 outfield players have at least one official
alternative belonging to a different functional role.

### 2. Alternative roles are highly structured

Common transitions include Fullback → Wide Midfielder, Wide Midfielder →
Winger, DM ↔ CM, and CAM ↔ CM.

Primary role therefore contains substantial information about likely
secondary roles.

### 3. Individual attributes do not predict secondary roles better overall

At Hit@1:

``` text
Model    = 69.74%
Baseline = 86.29%
```

Therefore, Stage 3.6 does not support the claim that model-generated
secondary probabilities independently discover player versatility.

### 4. Attributes still provide role-specific information

The model beats the baseline at Hit@1 and MRR for Central Midfielder and
Striker, while also slightly beating the baseline at Hit@2 for Wide
Midfielder.

Attributes therefore appear most useful when a primary role has more
heterogeneous secondary-role pathways.

### 5. Eligibility and suitability should remain separate

The final player representation distinguishes:

\[ oxed{ ext{Official Eligibility}(i,r)} \]

from:

\[ oxed{ ext{Learned Suitability}(i,r)} \]

This avoids equating classifier uncertainty with genuine player
versatility.

------------------------------------------------------------------------

# Connection to Stage 4

Stage 3.6 completes the player-level role representation.

For every player, the project now has:

-   overall quality information,
-   detailed technical and physical attributes,
-   primary position,
-   official alternative positions,
-   functional-role eligibility,
-   learned functional-role suitability.

Stage 4 can therefore move from:

> **What kind of player is this?**

to the team-level question:

> **Does selecting players according to role suitability, positional
> flexibility, and eventually complementarity produce a meaningfully
> different team from simply selecting the best individual player at
> each position?**

The Stage 4 optimizer can use **official eligibility as a constraint**
and **learned suitability as one component of the lineup objective**.
