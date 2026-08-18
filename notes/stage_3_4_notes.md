# Stage 3.4 — Learning Position Profiles

## Goal

Stage 3.3 tested whether a player's official EAFC position could be predicted using the six broad attributes:

- PAC — Pace
- SHO — Shooting
- PAS — Passing
- DRI — Dribbling
- DEF — Defending
- PHY — Physicality

Both Logistic Regression and Random Forest showed that these attributes contain useful positional information.

However, the models struggled with several closely related positions, especially:

- Left Winger vs Right Winger
- Left Midfielder vs Right Midfielder
- Left Back vs Right Back
- Attacking Midfielder vs Central Midfielder
- Central Midfielder vs Defensive Midfielder

Instead of immediately trying another model, Stage 3.4 asks:

> What attributes actually characterize each soccer position?

The goal is to learn a data-driven "fingerprint" for each position.


## Why Do This Before Training More Models?

Stage 3.3 showed that simply using a more complicated model does not necessarily solve the position prediction problem.

The Logistic Regression model achieved approximately 61% accuracy, while the Random Forest model achieved approximately 57%.

This suggests that the main limitation may not be the model itself.

Instead, the problem may be that:

1. The six broad EAFC attributes do not contain enough detail.
2. Some EAFC position labels may be extremely similar based on player skills.

Therefore, Stage 3.4 focuses on understanding the features and position labels before continuing model experimentation.


## Dataset

Unlike the earlier stages, Stage 3.4 uses the full-feature processed dataset:

`eafc26_outfield_full_features.csv`

The dataset contains approximately:

`11,872 outfield players`

and:

`59 columns`

The purpose of using the full dataset is to preserve more detailed information about each player's abilities.


## Core Attributes

The original six attributes are still included:

- PAC
- SHO
- PAS
- DRI
- DEF
- PHY

Keeping these attributes allows us to compare Stage 3.4 with the earlier experiments.


## Detailed Attributes

Stage 3.4 also identifies detailed numerical attributes available in the dataset.

The current analysis found 28 additional attributes:

### Pace

- Acceleration
- Sprint Speed

### Shooting

- Positioning
- Finishing
- Shot Power
- Long Shots
- Volleys
- Penalties

### Passing

- Vision
- Crossing
- Free Kick Accuracy
- Short Passing
- Long Passing
- Curve

### Dribbling

- Agility
- Balance
- Reactions
- Ball Control
- Dribbling
- Composure

### Defending

- Interceptions
- Heading Accuracy
- Standing Tackle
- Sliding Tackle

### Physical

- Jumping
- Stamina
- Strength
- Aggression

Therefore, Stage 3.4 currently analyzes:

`6 broad attributes + 28 detailed attributes = 34 features`


## Position Profiles

For each official EAFC position, we calculate the average value of every selected attribute.

For example:

`Average Center Back = average attributes of all Center Backs`

`Average Striker = average attributes of all Strikers`

These averages create a basic profile of what players at each position tend to look like.


## Standardized Position Fingerprints

Raw EAFC ratings are useful, but they can be difficult to compare across attributes.

Therefore, the attributes are also standardized using `StandardScaler`.

Each attribute is transformed relative to the overall outfield-player population.

A standardized score:

`> 0`

means the position tends to be above the average outfield player for that attribute.

A score:

`< 0`

means the position tends to be below average.

A score near:

`0`

means the position is relatively close to the overall average.

The average standardized profile for each position becomes its:

**Position Fingerprint**


## Example Position Fingerprints

The richer attributes produce intuitive positional patterns.


### Center Back

Some of the strongest characteristics include:

- Strength
- DEF
- Heading Accuracy
- Sliding Tackle
- Standing Tackle

This suggests that center backs are primarily distinguished by defensive and physical attributes.


### Striker

Some of the strongest characteristics include:

- Penalties
- Finishing
- Volleys
- SHO
- Positioning

This produces a very different fingerprint from Center Back and reflects the striker's scoring role.


### Defensive Midfielder

Important characteristics include:

- Interceptions
- Long Passing
- DEF
- Standing Tackle
- Sliding Tackle

This is interesting because Defensive Midfielders combine defensive skills with passing ability.


### Central Midfielder

Important characteristics include:

- Long Passing
- Crossing
- PAS
- Short Passing
- Free Kick Accuracy

This suggests that passing and ball distribution are particularly important for Central Midfielders.


### Attacking Midfielder

Important characteristics include:

- Free Kick Accuracy
- Crossing
- PAS
- Curve
- DRI

The Attacking Midfielder profile therefore appears more creative and attacking than the Central or Defensive Midfielder profiles.


## Which Attributes Distinguish Positions the Most?

We also calculate how much each attribute varies across the average position profiles.

If an attribute is almost identical across every position, it is probably not very useful for distinguishing positions.

If an attribute changes substantially between positions, it may contain important positional information.

Some of the attributes with the greatest variation were approximately:

| Attribute | Variation |
|---|---:|
| DEF | 0.759 |
| Sliding Tackle | 0.743 |
| Interceptions | 0.736 |
| Standing Tackle | 0.730 |
| Finishing | 0.526 |
| SHO | 0.471 |
| Volleys | 0.395 |
| PHY | 0.377 |
| Heading Accuracy | 0.376 |
| Positioning | 0.374 |
| Long Shots | 0.373 |
| Acceleration | 0.362 |

One important observation is that several detailed attributes vary strongly between positions.

This suggests that the richer attributes may contain information that is hidden when we only use PAC, SHO, PAS, DRI, DEF, and PHY.


## Important Discovery: Left vs Right Positions

One of the most important observations from Stage 3.4 is how similar left-side and right-side positions are.


### Left Midfielder

Strong characteristics include:

- Acceleration
- PAC
- Agility
- Sprint Speed
- Finishing


### Right Midfielder

Strong characteristics include:

- Acceleration
- PAC
- Agility
- Sprint Speed
- Vision


### Left Winger

Strong characteristics include:

- Acceleration
- Agility
- PAC
- Finishing
- Sprint Speed


### Right Winger

Strong characteristics include:

- Acceleration
- PAC
- Finishing
- Agility
- Sprint Speed


## Interpretation of the Left/Right Problem

This helps explain one of the major problems from Stage 3.3.

A player's skill attributes may tell us that they resemble a winger, but they may contain very little information about whether that player belongs on the left or right side.

For example:

`PAC = 85`

does not inherently mean:

`Left Winger`

or:

`Right Winger`

The same applies to attributes such as:

- Finishing
- Dribbling
- Acceleration
- Passing
- Defending

These describe the player's functional abilities more than the side of the field where the player is positioned.

Therefore, predicting exact left/right positions may create an unnecessary classification problem.


## Functional Positions

Based on this observation, the next experiment will remove the left/right distinction.

The following positions will be combined:

`Left Winger + Right Winger → Winger`

`Left Midfielder + Right Midfielder → Wide Midfielder`

`Left Back + Right Back → Fullback`

The remaining positions stay separate:

- Attacking Midfielder
- Central Midfielder
- Defensive Midfielder
- Center Back
- Striker

This changes the prediction problem from:

`11 exact EAFC positions`

to:

`8 functional positions`


## Why Keep Winger and Wide Midfielder Separate?

Although left and right versions appear extremely similar, we should not immediately combine every similar position.

For example:

`Winger`

and:

`Wide Midfielder`

may still represent different functional roles.

The next model should determine whether the richer attributes contain enough information to distinguish them.

If the model still cannot distinguish them, that becomes another useful experimental result rather than something we assume beforehand.


## Main Takeaway

Stage 3.4 shows that detailed EAFC attributes provide a much richer description of positional roles than the six broad attributes alone.

It also reveals that some of the difficulty in Stage 3.3 may come from the definition of the target labels rather than the machine learning models.

In particular, left and right versions of the same role have extremely similar attribute profiles.

Therefore, the next experiment should focus on **functional positions rather than exact left/right positions**.


## Next Experiment

We will compare several conditions.

### Existing Baselines

1. Logistic Regression
   - 6 attributes
   - 11 exact positions

2. Random Forest
   - 6 attributes
   - 11 exact positions


### Functional Position Experiments

3. Logistic Regression
   - 6 attributes
   - 8 functional positions

4. Random Forest
   - 6 attributes
   - 8 functional positions

5. Logistic Regression
   - 34 attributes
   - 8 functional positions

6. Random Forest
   - 34 attributes
   - 8 functional positions


## Research Questions

This experiment allows us to answer two main questions:

### Question 1

Does removing the left/right distinction improve position prediction?

This tests whether some of the previous model errors were caused by asking the models to distinguish positions that have nearly identical skill profiles.


### Question 2

Do detailed player attributes improve position prediction?

This tests whether information such as:

- Finishing
- Vision
- Long Passing
- Interceptions
- Standing Tackle
- Acceleration
- Strength
- Positioning

provides useful positional information beyond the six broad EAFC ratings.


## Long-Term Connection to Team Building

The final goal is not simply to classify players correctly.

Once we identify a useful position model, we can examine the probability that a player fits several functional positions.

For example:

`Player A`

`Central Midfielder        0.45`

`Attacking Midfielder      0.30`

`Defensive Midfielder      0.15`

Rather than restricting Player A to their official position, the model could identify several plausible roles based on their attributes.

This will eventually allow the formation-building stage to consider positional versatility when selecting players.