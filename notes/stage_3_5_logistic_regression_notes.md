# Stage 3.5A — Logistic Regression for Functional Positions

## Goal

Stage 3.3 tested whether Logistic Regression could predict a player's exact EAFC position using six broad attributes. However, Stage 3.4 showed that left-side and right-side versions of the same role have extremely similar attribute profiles.

Stage 3.4.2 therefore combined the original 11 positions into eight functional positions:

- Attacking Midfielder
- Center Back
- Central Midfielder
- Defensive Midfielder
- Full Back
- Striker
- Wide Midfielder
- Winger

Stage 3.5A asks:

> How accurately can Logistic Regression predict these eight functional positions, and do detailed EAFC attributes improve its predictions?


## Functional-Position Mapping

The following left/right positions were combined:

`Left Back + Right Back → Full Back`

`Left Midfielder + Right Midfielder → Wide Midfielder`

`Left Winger + Right Winger → Winger`

The other five positions remained separate.

This removes a distinction that the attribute data did not describe well while preserving meaningful differences between broader soccer roles.


## Dataset

The model uses the functional-position dataset created in Stage 3.4.2:

`eafc26_functional_positions.csv`

The dataset contains:

`11,872 outfield players`

The data was divided into:

- 80% training data
- 20% testing data

The split used `random_state=42` and `stratify=y`, ensuring that each functional position had approximately the same proportion in the training and testing sets.


## Why Use Logistic Regression?

Logistic Regression provides a simple and interpretable baseline for multiclass classification.

It is useful for this project because it:

- Predicts one functional position for each player.
- Produces a probability for every possible position.
- Allows us to examine a player's second and third most likely roles.
- Provides a direct comparison with the original Stage 3.3 model.

Because the attributes use different numerical distributions, `StandardScaler` is applied before training. The scaler is fitted only to the training data to avoid test-data leakage.


## Feature Experiments

Two feature representations were compared.

### Six-Feature Model

The first model used the six broad EAFC attributes:

- PAC
- SHO
- PAS
- DRI
- DEF
- PHY

This provides a direct comparison with Stage 3.3.


### Rich-Feature Model

The second model used:

`6 broad attributes + 28 detailed attributes = 34 features`

The detailed features include abilities such as:

- Acceleration
- Finishing
- Vision
- Short Passing
- Long Passing
- Ball Control
- Interceptions
- Standing Tackle
- Strength
- Positioning

This experiment tests whether detailed player information distinguishes roles more effectively than the six broad ratings alone.


## Evaluation Measurements

### Top-1 Accuracy

Top-1 accuracy checks whether the model's first predicted position matches the player's functional-position label.

For example, if the actual position is `Attacking Midfielder`, the prediction is counted as correct only when Attacking Midfielder is the model's first choice.


### Top-2 Accuracy

Top-2 accuracy checks whether the actual position appears among the model's two most likely predictions.

This is useful because a player may reasonably resemble two different functional roles.


### Top-3 Accuracy

Top-3 accuracy checks whether the actual position appears among the model's three most likely predictions.

Top-2 and top-3 accuracy are especially relevant to this project because the final goal involves positional versatility rather than forcing every player into exactly one role.


## Results

| Feature Set | Features | Top-1 Accuracy | Top-2 Accuracy | Top-3 Accuracy |
|---|---:|---:|---:|---:|
| Broad attributes | 6 | 69.31% | 87.96% | 96.13% |
| Rich attributes | 34 | **77.22%** | **93.39%** | **98.27%** |

Adding the detailed attributes increased top-1 accuracy by:

`77.22% − 69.31% = 7.91 percentage points`

This shows that the detailed attributes contain important positional information that is hidden inside the six broad ratings.


## Rich-Feature Classification Results

| Functional Position | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Attacking Midfielder | 0.640 | 0.474 | 0.545 |
| Center Back | 0.927 | 0.913 | 0.920 |
| Central Midfielder | 0.674 | 0.783 | 0.725 |
| Defensive Midfielder | 0.674 | 0.587 | 0.627 |
| Full Back | 0.839 | 0.916 | 0.876 |
| Striker | 0.875 | 0.895 | 0.885 |
| Wide Midfielder | 0.590 | 0.722 | 0.650 |
| Winger | 0.417 | 0.049 | 0.087 |

The rich-feature model achieved:

- Macro F1-score: 0.664
- Weighted F1-score: 0.758


## Strongest Position Predictions

The strongest results were produced for:

- Center Back
- Striker
- Full Back

These roles have distinctive attribute profiles.

Center Backs are strongly associated with defensive and physical attributes, while Strikers are strongly associated with finishing, shooting, and positioning. Full Backs combine defensive ability with pace and stamina.


## Midfield Position Results

The model improved its predictions for the three central midfield roles, although some overlap remained.

For example:

- Central Midfielder vs Attacking Midfielder
- Central Midfielder vs Defensive Midfielder
- Defensive Midfielder vs Full Back

This overlap is reasonable because midfield players often share passing, dribbling, defensive, and physical abilities.

The probability distribution is therefore more informative than only examining the first prediction.


## Main Limitation: Wingers

Winger remained the most difficult position.

The rich-feature Logistic Regression model achieved only:

`Winger recall = 4.9%`

This means the model correctly chose Winger as its first prediction for approximately 5% of the players labeled as Wingers.

Many Wingers were instead predicted as:

- Wide Midfielder
- Attacking Midfielder
- Striker

This suggests that Wingers have substantial attribute overlap with other attacking wide roles. Their attributes may describe how they attack more clearly than whether EAFC labels them as a Winger or Wide Midfielder.


## Example Player Interpretations

### Jude Bellingham

The model predicted approximately:

`Central Midfielder       79.4%`

`Attacking Midfielder     19.3%`

Although his EAFC label was Attacking Midfielder, his detailed attributes more strongly resembled the Central Midfielder profile. His official role still appeared as the second prediction.


### Trent Alexander-Arnold

The model predicted approximately:

`Full Back                53.1%`

`Central Midfielder       32.8%`

`Defensive Midfielder     13.1%`

This probability distribution reflects his ability to contribute both as a Full Back and as a central passing player.


### Lionel Messi

The model predicted Attacking Midfielder ahead of Winger.

This disagreement does not necessarily mean the model failed. It suggests that Messi's current attribute profile resembles a central creative attacker more than the average Winger in the dataset.


## Interpretation

The results answer both Stage 3.5 research questions.

### Did Functional Positions Help?

Yes. Logistic Regression increased from approximately 61% accuracy for 11 exact positions in Stage 3.3 to 69.31% using the same six attributes and eight functional positions.

This supports the Stage 3.4 conclusion that left/right distinctions were creating unnecessary classification difficulty.


### Did Detailed Attributes Help?

Yes. Expanding from six to 34 attributes increased accuracy from 69.31% to 77.22%.

The detailed attributes improved the model's ability to separate functional roles, especially Center Back, Full Back, Striker, and the central midfield positions.


## Main Takeaway

The 34-feature Logistic Regression model produced the strongest overall Stage 3.5 results:

- Top-1 accuracy: 77.22%
- Top-2 accuracy: 93.39%
- Top-3 accuracy: 98.27%
- Weighted F1-score: 75.8%

Although top-1 accuracy did not reach exactly 80%, the top-2 and top-3 results are strong enough for the project's purpose.

The model identifies the official functional position among its first two predictions for more than 93% of test players. This makes its probability distribution useful for studying positional versatility.


## Connection to Lineup Construction

After evaluation, a final rich-feature Logistic Regression model is trained using all available players.

The model saves:

- A probability for every functional position.
- Each player's three most likely functional positions.
- The probability associated with each of those positions.

The main output is:

`all_player_functional_position_probabilities.csv`

This file can support the lineup stage by allowing a player to fill a role based on attribute similarity rather than only their official EAFC position.


## Decision

The rich-feature Logistic Regression model will be used as the main functional-position model for the next stage because it achieved the highest top-1, top-2, top-3, and weighted F1 results.

