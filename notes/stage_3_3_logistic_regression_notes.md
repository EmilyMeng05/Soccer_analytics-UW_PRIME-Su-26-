# Stage 3.3A — Logistic Regression Position Prediction

## Goal

Earlier stages showed that players with different official positions can have similar attribute profiles.

This raises an important question:

> Can a player's likely position be learned directly from their attributes?

Stage 3.3 begins exploring positional flexibility.

Instead of assuming that a player must remain in their listed EAFC position, we train a model to learn the relationship between player attributes and official positions.


## Input Features

The baseline model uses the same six EAFC attributes:

- PAC
- SHO
- PAS
- DRI
- DEF
- PHY

Therefore:

`X = [PAC, SHO, PAS, DRI, DEF, PHY]`


## Target

The target variable is the player's official EAFC position.

The original experiment contains 11 position classes:

- Attacking Midfielder
- Center Back
- Central Midfielder
- Defensive Midfielder
- Left Back
- Left Midfielder
- Left Winger
- Right Back
- Right Midfielder
- Right Winger
- Striker


## Why Logistic Regression?

Logistic Regression provides a relatively simple and interpretable baseline.

It can:

1. predict multiple position classes
2. produce probabilities for each possible position
3. show whether the six broad attributes contain enough information to distinguish positions

The features are standardized using `StandardScaler` before training.


## Train-Test Split

The dataset is divided into:

- 80% training
- 20% testing

The split uses:

`random_state = 42`

and:

`stratify = y`

Stratification helps preserve the relative frequency of each position in both datasets.


## Evaluation

The model is evaluated using:

- accuracy
- precision
- recall
- F1-score
- confusion matrix

Overall accuracy is useful, but position-level F1 scores are especially important because some positions are much more common than others.


## Result

The six-feature Logistic Regression model achieved approximately:

`61% test accuracy`

This indicates that the six broad EAFC attributes contain substantial information about player position.

However, they are not sufficient to perfectly distinguish all positional roles.


## Position Probabilities

The model is not only used to return one predicted position.

For every player, probabilities are calculated across all possible positions.

The three most likely positions are saved as:

- PredictedPosition1
- PositionProbability1
- PredictedPosition2
- PositionProbability2
- PredictedPosition3
- PositionProbability3


## Why Top-3 Positions Matter

The top-3 predictions are important for the project's positional flexibility idea.

For example, a hypothetical player could receive:

`Central Midfielder = 0.45`

`Attacking Midfielder = 0.32`

`Defensive Midfielder = 0.15`

Instead of saying:

"This player is only a Central Midfielder"

we can say:

"The player's attributes resemble several midfield roles."

This may eventually allow players to be considered for positions outside their official EAFC label.


## Important Limitation

The model is trained using EAFC's existing position labels.

Therefore, it is NOT discovering a player's objectively correct or ideal position.

It is learning:

> Which combinations of attributes are typically associated with each existing EAFC position?

A disagreement between the model and EAFC position therefore does not automatically mean either one is wrong.


## Left/Right Position Problem

One important issue is that the six attributes may not contain enough information to distinguish:

- Left Winger vs Right Winger
- Left Midfielder vs Right Midfielder
- Left Back vs Right Back

A player's pace, shooting, passing, defending, etc. describe their playing ability but may not tell us which side of the field they play on.

This becomes an important motivation for later functional-position experiments.


## Main Takeaway

Logistic Regression provides evidence that player attributes contain meaningful positional information.

However, the six broad attributes appear better suited for recognizing general soccer roles than every exact EAFC position.

The model therefore serves as a baseline for later experiments using functional position labels and richer player attributes.