# Stage 3.5B — Random Forest for Functional Positions

## Goal

Stage 3.5B performs the same functional-position experiments as Stage 3.5A but replaces Logistic Regression with Random Forest.

The main question is:

> Can a nonlinear model identify relationships between player attributes and functional positions more accurately than Logistic Regression?

Random Forest is tested using both the six broad attributes and the richer set of 34 attributes.


## Why Use Random Forest?

Logistic Regression learns relatively simple decision boundaries between positions.

Random Forest combines many decision trees and can learn more complicated patterns, such as:

- A player having both high pace and high defending.
- A midfielder combining long passing, interceptions, and stamina.
- A forward combining finishing, positioning, dribbling, and strength.

Random Forest therefore tests whether nonlinear feature interactions improve functional-position prediction.


## Functional Positions

The model predicts the same eight functional positions used in the Logistic Regression experiment:

- Attacking Midfielder
- Center Back
- Central Midfielder
- Defensive Midfielder
- Full Back
- Striker
- Wide Midfielder
- Winger

The left/right combinations remain:

`Left Back + Right Back → Full Back`

`Left Midfielder + Right Midfielder → Wide Midfielder`

`Left Winger + Right Winger → Winger`


## Dataset and Split

The model uses:

`eafc26_functional_positions.csv`

The dataset contains approximately:

`11,872 outfield players`

The same stratified 80/20 train/test split and `random_state=42` are used for both model types. Therefore, Logistic Regression and Random Forest are evaluated using comparable training and testing data.


## Random Forest Settings

The model uses:

- 500 decision trees
- `min_samples_leaf=2`
- `max_features="sqrt"`
- `random_state=42`

Using many trees produces more stable predictions. Requiring at least two samples in each leaf provides mild protection against overfitting.

Random Forest does not require `StandardScaler` because decision trees split individual attributes using thresholds. Changing the scale of an attribute does not change the ordering of its values.


## Feature Experiments

### Six-Feature Model

The first model uses:

- PAC
- SHO
- PAS
- DRI
- DEF
- PHY


### Rich-Feature Model

The second model uses:

`6 broad attributes + 28 detailed attributes = 34 features`

This makes it possible to test whether the detailed attributes improve Random Forest in the same way that they improved Logistic Regression.


## Evaluation Measurements

The Random Forest model is evaluated using:

- Top-1 accuracy
- Top-2 accuracy
- Top-3 accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Top-1 accuracy checks only the model's first prediction. Top-2 and top-3 accuracy check whether the official functional position appears among the model's first two or three predictions.


## Results

| Feature Set | Features | Top-1 Accuracy | Top-2 Accuracy | Top-3 Accuracy |
|---|---:|---:|---:|---:|
| Broad attributes | 6 | 68.00% | 86.65% | 95.33% |
| Rich attributes | 34 | **75.20%** | **91.45%** | **97.73%** |

Adding the detailed attributes increased top-1 accuracy by:

`75.20% − 68.00% = 7.20 percentage points`

This confirms that the detailed attributes provide useful positional information for both model types.


## Rich-Feature Classification Results

| Functional Position | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Attacking Midfielder | 0.659 | 0.351 | 0.458 |
| Center Back | 0.907 | 0.915 | 0.911 |
| Central Midfielder | 0.611 | 0.825 | 0.702 |
| Defensive Midfielder | 0.647 | 0.495 | 0.561 |
| Full Back | 0.805 | 0.889 | 0.845 |
| Striker | 0.872 | 0.903 | 0.887 |
| Wide Midfielder | 0.589 | 0.681 | 0.632 |
| Winger | 0.429 | 0.029 | 0.055 |

The rich-feature model achieved:

- Macro F1-score: 0.631
- Weighted F1-score: 0.733


## Strongest Position Predictions

The Random Forest performed especially well for:

- Center Back
- Striker
- Full Back

These are also among the strongest positions for Logistic Regression, suggesting that their attribute profiles are genuinely more distinct in the dataset.

The model also achieved strong recall for Central Midfielder, correctly identifying approximately 82.5% of the Central Midfielders in the test set.


## Main Limitation: Wingers

Winger was again the most difficult role.

The rich-feature Random Forest achieved:

`Winger recall = 2.9%`

This is even lower than the Logistic Regression Winger recall of 4.9%.

Both models frequently assign Wingers to related attacking roles, especially Wide Midfielder and Attacking Midfielder. Because both algorithms struggle with this distinction, the limitation likely comes from overlapping role definitions rather than only from the choice of model.


## Feature Importance

Random Forest provides a feature-importance value for each attribute.

Feature importance estimates how much an attribute contributed to separating the functional positions across the decision trees.

A high importance does not mean that the attribute causes a player to have a position. It means that the model frequently found the attribute useful when making splits.

The saved feature-importance files can help identify whether detailed attributes such as Finishing, Interceptions, Standing Tackle, or Positioning contributed more strongly than the broad EAFC ratings.


## Comparison With Logistic Regression

### Six-Feature Comparison

| Model | Top-1 | Top-2 | Top-3 |
|---|---:|---:|---:|
| Logistic Regression | **69.31%** | **87.96%** | **96.13%** |
| Random Forest | 68.00% | 86.65% | 95.33% |


### Rich-Feature Comparison

| Model | Top-1 | Top-2 | Top-3 | Weighted F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | **77.22%** | **93.39%** | **98.27%** | **75.8%** |
| Random Forest | 75.20% | 91.45% | 97.73% | 73.3% |

Logistic Regression performed slightly better for both feature representations and across every reported top-k accuracy measurement.

This means that the additional nonlinear complexity of Random Forest did not improve the overall prediction task.


## Interpretation

The Random Forest experiment supports three conclusions.

### Functional Positions Are Easier to Predict

The six-feature Random Forest improved from approximately 57% accuracy for 11 exact positions in Stage 3.3 to 68.00% for eight functional positions.

This supports combining left/right versions of similar roles.


### Detailed Attributes Are Valuable

Increasing the representation from six to 34 attributes improved Random Forest accuracy from 68.00% to 75.20%.

Therefore, the richer player representation benefits both Logistic Regression and Random Forest.


### A More Complicated Model Is Not Automatically Better

Random Forest can learn nonlinear relationships, but it still performed slightly worse than Logistic Regression.

This suggests that the remaining errors are influenced more by overlapping position labels than by an inability to learn complicated attribute patterns.


## Main Takeaway

The rich-feature Random Forest produced useful results:

- Top-1 accuracy: 75.20%
- Top-2 accuracy: 91.45%
- Top-3 accuracy: 97.73%
- Weighted F1-score: 73.3%

Its top-2 and top-3 accuracy demonstrate that the model usually places the official role near the top of its probability ranking.

However, Logistic Regression performed slightly better overall and produced stronger results for the project’s positional-versatility goal.


## Decision

Random Forest will remain an important comparison model, but it will not be the primary model used for lineup construction.

The selected position model is:

`34-feature Logistic Regression`

The Logistic Regression probability output will therefore be used in the next stage to evaluate player-role fit and positional versatility.

