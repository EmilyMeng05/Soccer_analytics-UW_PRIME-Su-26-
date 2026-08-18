from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# STAGE 3.3A: LOGISTIC REGRESSION POSITION PREDICTION
#
# prepare_dataset.py has already cleaned the EAFC26 dataset.
#
# In this stage, I want to ask:
#
# Can a player's likely position be learned directly from their
# six main EAFC attributes?
#
# The model is given:
#
# PAC
# SHO
# PAS
# DRI
# DEF
# PHY
#
# and tries to predict the player's official EAFC position.
#
# The goal is not only to predict one position.
#
# I also want to keep the probability distribution across positions
# because this may help identify players whose attribute profiles
# resemble multiple positions.


# Define project paths.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA = PROJECT_ROOT / "data"

RESULTS = (
    PROJECT_ROOT
    / "results"
    / "stage_3_3"
    / "logistic_regression"
)

RESULTS.mkdir(
    parents=True,
    exist_ok=True
)


# Read the cleaned outfield player dataset created by prepare_dataset.py.

df = pd.read_csv(
    DATA
    / "processed"
    / "cleaned_eafc26_outfield_players.csv"
)

print(
    "Dataset shape:",
    df.shape
)


# Show all columns when printing player predictions.

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)


# Use the same six-dimensional player representation that has been used
# throughout the earlier stages of the project.

features = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY"
]


# Input variables.

X = df[
    features
]


# Target variable.

y = df[
    "Position"
]


# Split the dataset into training and testing sets.
#
# The model learns from the training players.
#
# The test players are held out so that we can evaluate whether the
# learned relationship generalizes to unseen players.
#
# stratify=y keeps the position distribution similar in both sets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Standardize the player attributes.
#
# The scaler is fit only on the training data to avoid using information
# from the test set during model training.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# Train the Logistic Regression model.
#
# Logistic Regression is useful because:
#
# 1. It supports multiple position classes.
# 2. It produces probabilities for each possible position.
# 3. It provides a simple and interpretable baseline model.

model = LogisticRegression(
    max_iter=3000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


# Predict positions for the held-out test players.

y_pred = model.predict(
    X_test_scaled
)


# Calculate overall accuracy.

accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "\nLogistic Regression accuracy:"
)

print(
    round(
        accuracy,
        4
    )
)


# Save the overall model evaluation.

model_evaluation = pd.DataFrame({
    "Metric": [
        "Accuracy"
    ],
    "Value": [
        accuracy
    ]
})


model_evaluation.to_csv(
    RESULTS
    / "model_evaluation.csv",
    index=False
)


# Calculate precision, recall, and F1-score for each position.
#
# This is important because some positions contain many more players
# than others.

report = classification_report(
    y_test,
    y_pred,
    zero_division=0,
    output_dict=True
)


report_df = pd.DataFrame(
    report
).transpose()


print(
    "\nClassification report:"
)

print(
    report_df.round(
        3
    )
)


report_df.to_csv(
    RESULTS
    / "classification_report.csv"
)


# Create the confusion matrix.
#
# This helps us understand which positions the model tends to confuse.
#
# Some confusing pairs may reflect genuine similarities:
#
# CAM vs CM
#
# CM vs CDM
#
# LW vs LM
#
# RW vs RM

position_labels = model.classes_


confusion = confusion_matrix(
    y_test,
    y_pred,
    labels=position_labels
)


confusion_df = pd.DataFrame(
    confusion,
    index=position_labels,
    columns=position_labels
)


print(
    "\nConfusion matrix:"
)

print(
    confusion_df
)


confusion_df.to_csv(
    RESULTS
    / "position_prediction_confusion_matrix.csv"
)


# The train/test model above is used only to evaluate whether Logistic
# Regression can learn meaningful relationships between attributes
# and positions.
#
# Now train a final model using the complete dataset.
#
# This final model is used only to generate position probabilities
# for exploratory analysis and future stages.

final_scaler = StandardScaler()


X_all_scaled = final_scaler.fit_transform(
    df[
        features
    ]
)


final_model = LogisticRegression(
    max_iter=3000,
    random_state=42
)


final_model.fit(
    X_all_scaled,
    y
)


# Calculate the probability that every player resembles each possible
# EAFC position.

position_probabilities = (
    final_model.predict_proba(
        X_all_scaled
    )
)


# Convert the probability matrix into a DataFrame.
#
# Each column represents one possible position.

probability_columns = [
    f"Prob_{position}"
    for position in final_model.classes_
]


probability_df = pd.DataFrame(
    position_probabilities,
    columns=probability_columns,
    index=df.index
)


# Add the probability columns to the player dataset.

results = pd.concat(
    [
        df,
        probability_df
    ],
    axis=1
)


# Find each player's three most likely positions.
#
# The complete probability distribution is still saved.
#
# The top three simply make the results easier to interpret.

def get_top_positions(
    probabilities,
    classes,
    top_n=3
):

    order = (
        probabilities
        .argsort()[::-1][:top_n]
    )


    positions = [
        classes[i]
        for i in order
    ]


    scores = [
        probabilities[i]
        for i in order
    ]


    return (
        positions,
        scores
    )


top_1_positions = []
top_1_probabilities = []

top_2_positions = []
top_2_probabilities = []

top_3_positions = []
top_3_probabilities = []


for probabilities in position_probabilities:

    positions, scores = get_top_positions(
        probabilities,
        final_model.classes_,
        top_n=3
    )


    top_1_positions.append(
        positions[0]
    )

    top_1_probabilities.append(
        scores[0]
    )


    top_2_positions.append(
        positions[1]
    )

    top_2_probabilities.append(
        scores[1]
    )


    top_3_positions.append(
        positions[2]
    )

    top_3_probabilities.append(
        scores[2]
    )


# Add the three most likely positions to the results.

results[
    "PredictedPosition1"
] = top_1_positions

results[
    "PositionProbability1"
] = top_1_probabilities


results[
    "PredictedPosition2"
] = top_2_positions

results[
    "PositionProbability2"
] = top_2_probabilities


results[
    "PredictedPosition3"
] = top_3_positions

results[
    "PositionProbability3"
] = top_3_probabilities


# Check whether the first predicted position matches the official EAFC position.
#
# A disagreement does NOT automatically mean the model is wrong.
#
# It may indicate that the player's attribute profile resembles
# another positional group.

results[
    "OfficialPositionMatch"
] = (
    results[
        "PredictedPosition1"
    ]
    == results[
        "Position"
    ]
)


# Columns used when printing example predictions.

example_columns = [
    "Name",
    "Position",
    "PredictedPosition1",
    "PositionProbability1",
    "PredictedPosition2",
    "PositionProbability2",
    "PredictedPosition3",
    "PositionProbability3"
]


print(
    "\nExample predicted positions:"
)

print(
    results[
        example_columns
    ]
    .head(25)
)


# Create a function for inspecting individual players.

def show_player_positions(
    player_name
):

    player = results[
        results[
            "Name"
        ]
        .str.lower()
        == player_name.lower()
    ]


    if len(player) == 0:

        print(
            f"\nPlayer '{player_name}' not found."
        )

        return


    print(
        f"\nLogistic Regression position predictions for {player_name}:"
    )


    print(
        player[
            example_columns
        ]
    )


# Inspect several interesting players.

show_player_positions(
    "Jude Bellingham"
)

show_player_positions(
    "Federico Valverde"
)

show_player_positions(
    "Achraf Hakimi"
)

show_player_positions(
    "Lionel Messi"
)

show_player_positions(
    "Trent Alexander-Arnold"
)


# Find highly rated players whose first predicted position differs
# from their official EAFC position.
#
# These players may be useful examples when studying positional flexibility.

position_disagreements = results[
    results[
        "OfficialPositionMatch"
    ]
    == False
].copy()


position_disagreements = (
    position_disagreements
    .sort_values(
        "OVR",
        ascending=False
    )
)


print(
    "\nHighest-rated players whose Logistic Regression prediction "
    "differs from their listed position:"
)


print(
    position_disagreements[
        example_columns
    ]
    .head(30)
)


# Save the complete position-probability dataset.
#
# This contains:
#
# player information
#
# PAC SHO PAS DRI DEF PHY
#
# full probability distribution across positions
#
# top three predicted positions
#
# probability for each predicted position
#
# official-position match indicator

results.to_csv(
    RESULTS
    / "player_position_predictions.csv",
    index=False
)


# Save disagreement examples separately.

position_disagreements.to_csv(
    RESULTS
    / "position_prediction_disagreements.csv",
    index=False
)


# Save a small model summary.
#
# This will make it easier to compare Logistic Regression with
# Random Forest and future rich-feature models.

model_summary = pd.DataFrame({
    "Model": [
        "Logistic Regression"
    ],
    "Features": [
        "PAC, SHO, PAS, DRI, DEF, PHY"
    ],
    "Accuracy": [
        accuracy
    ]
})


model_summary.to_csv(
    RESULTS
    / "model_summary.csv",
    index=False
)


# Stage 3.3A does NOT claim that the model has discovered a player's
# true or ideal soccer position.
#
# The official EAFC position is used as the training label.
#
# Therefore, the model learns which combinations of the six EAFC
# attributes are associated with those existing position labels.
#
# The classification accuracy tells us how well these six broad
# attributes distinguish official positions.
#
# The probability distribution is useful for exploring positional
# flexibility.
#
# For example:
#
# CM  = 0.45
# CAM = 0.35
# CDM = 0.15
#
# suggests a more flexible profile than:
#
# CB = 0.95
#
# Stage 3.4 will investigate the actual attribute profiles associated
# with each position and eventually introduce richer player attributes.


print(
    "\nStage 3.3A Logistic Regression position prediction complete!"
)