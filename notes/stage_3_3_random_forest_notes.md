# Stage 3.3B — Random Forest Position Prediction

## Goal

Stage 3.3A used Logistic Regression to predict player positions from the six broad EAFC attributes.

However, Logistic Regression may struggle when the relationship between attributes and positions is nonlinear.

Stage 3.3B therefore asks:

> Can Random Forest better learn the relationship between player attributes and official EAFC positions?

The same features and target labels are used so the two models can be compared fairly.


## Input Features

The Random Forest model uses:

- PAC
- SHO
- PAS
- DRI
- DEF
- PHY

Therefore:

`X = [PAC, SHO, PAS, DRI, DEF, PHY]`


## Target

The target is the player's official EAFC position.

The experiment uses the same 11 classes as Logistic Regression.


## Why Random Forest?

Random Forest combines many decision trees.

Unlike Logistic Regression, it can naturally capture nonlinear relationships and interactions between player attributes.

For example, the meaning of high pace may depend on whether a player also has:

- high defending
- high shooting
- high dribbling

Random Forest can potentially learn these interactions without manually defining them.


## Model

The baseline model uses:

`n_estimators = 300`

`random_state = 42`

`class_weight = "balanced"`

`n_jobs = -1`

Class weighting is included because some positions contain substantially more players than others.

Unlike Logistic Regression, Random Forest does not require feature standardization.


## Train-Test Split

The same structure as Logistic Regression is used:

- 80% training
- 20% testing
- random_state = 42
- stratified by position

Using the same split structure makes the comparison between models more meaningful.


## Results

Overall Random Forest test accuracy:

`0.5672`

or approximately:

`56.7%`


## Strong Position Performance

Random Forest performed especially well for positions with distinctive attribute profiles.

### Center Back

Approximately:

`F1 = 0.89`

Center backs have a relatively distinctive combination of:

- high defending
- high physicality
- lower shooting


### Striker

Approximately:

`F1 = 0.87`

Strikers also have a relatively distinctive offensive profile, especially in shooting.


## Weak Position Performance

The model struggled with several similar or smaller position classes.

Examples:

### Left Winger

Approximately:

`F1 = 0.05`

### Right Winger

Approximately:

`F1 = 0.03`

The model also had difficulty separating related roles such as:

- LW vs LM
- RW vs RM
- CAM vs CM
- CM vs CDM


## Feature Importance

Random Forest provides feature importance values.

Previous results:

| Feature | Importance |
|---|---:|
| DEF | 0.214 |
| SHO | 0.171 |
| PAC | 0.162 |
| PHY | 0.156 |
| PAS | 0.152 |
| DRI | 0.144 |

DEF was the most useful individual feature for separating positions overall.

However, feature importance does NOT mean that DEF defines one particular position.

It means that DEF was especially useful across the decision trees when distinguishing between position classes.


## Position Probabilities

As with Logistic Regression, Random Forest generates probabilities across all possible positions.

For each player we save:

- PredictedPosition1
- PositionProbability1
- PredictedPosition2
- PositionProbability2
- PredictedPosition3
- PositionProbability3

These probabilities are more useful for positional flexibility than simply storing the top prediction.


## Model Disagreements

We also examine players whose predicted position differs from their official EAFC position.

These cases may help identify players with unusual or versatile attribute profiles.

However, disagreement does NOT automatically mean that Random Forest has discovered a better position.

The model is still trained from EAFC's official position labels.


## Logistic Regression vs Random Forest

Using the same six attributes:

| Model | Approx. Accuracy |
|---|---:|
| Logistic Regression | ~61% |
| Random Forest | ~57% |

Random Forest did not outperform Logistic Regression in the initial experiment.

This is an important result.

A more complicated model does not automatically produce better predictions.

One possible explanation is that the main limitation is not model complexity.

Instead, the six broad EAFC attributes may simply not contain enough information to distinguish certain detailed positions.


## Left/Right Position Problem

The particularly poor performance for winger classes helped reveal an important issue.

Attributes such as:

- pace
- shooting
- passing
- dribbling
- defending
- physicality

may distinguish:

`Winger vs Center Back`

very well.

But they may contain little information for distinguishing:

`Left Winger vs Right Winger`

This suggests that exact left/right labels may introduce unnecessary difficulty into the prediction problem.


## Main Takeaway

Random Forest confirms that the six EAFC attributes contain strong information about broad positional roles.

However, increasing model complexity did not solve the confusion between closely related positions.

This suggests that the next improvement should focus on the **representation of positions and player features**, rather than simply trying increasingly complicated models.


## Next Experiment

Later experiments will investigate:

1. Removing left/right distinctions.

For example:

`Left Winger + Right Winger → Winger`

`Left Midfielder + Right Midfielder → Wide Midfielder`

`Left Back + Right Back → Fullback`

This reduces the original 11 labels to 8 functional positions.


2. Adding richer EAFC attributes.

Instead of only:

`PAC, SHO, PAS, DRI, DEF, PHY`

we can use detailed attributes such as:

- Acceleration
- Finishing
- Positioning
- Vision
- Short Passing
- Long Passing
- Ball Control
- Interceptions
- Standing Tackle
- Stamina
- Strength
- Aggression

This will allow us to test whether detailed player information improves functional-position prediction.