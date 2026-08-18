from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# STAGE 3.3B: RANDOM FOREST POSITION PREDICTION
#
# Stage 3.3A used Logistic Regression to test whether the six main EAFC
# attributes can predict a player's official position.
#
# In this stage, I want to test whether Random Forest can capture more
# complicated relationships between:
#
# PAC
# SHO
# PAS
# DRI
# DEF
# PHY
#
# and the player's listed EAFC position.
#
# I will use the same six attributes and the same train-test split structure
# as Logistic Regression so the two models can be compared fairly.
#
# The model is not meant to discover a player's true or ideal position.
#
# Instead, it learns which attribute profiles are associated with the
# existing EAFC position labels.


# Define project paths.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA = PROJECT_ROOT / "data"

RANDOM_FOREST_RESULTS = (
    PROJECT_ROOT
    / "results"
    / "stage_3_3"
    / "random_forest"
)

RANDOM_FOREST_RESULTS.mkdir(
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


# Use the same six-dimensional player representation as Logistic Regression.

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
# stratify=y keeps the relative position distribution similar
# in both sets.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Random Forest does not require feature standardization.
#
# It creates many decision trees and combines their predictions.
#
# class_weight="balanced" helps reduce the effect of position imbalance.
#
# n_estimators=300 means the forest contains 300 trees.

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# Train the evaluation model.

model.fit(
    X_train,
    y_train
)


# Predict positions for the held-out test players.

y_pred = model.predict(
    X_test
)


# Calculate overall accuracy.

accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "\nRandom Forest accuracy:"
)

print(
    round(
        accuracy,
        4
    )
)


# Save overall model evaluation.

model_evaluation = pd.DataFrame({
    "Metric": [
        "Accuracy"
    ],
    "Value": [
        accuracy
    ]
})


model_evaluation.to_csv(
    RANDOM_FOREST_RESULTS
    / "model_evaluation.csv",
    index=False
)


# Calculate precision, recall, and F1-score for every position.
#
# Position-level performance is important because overall accuracy
# can hide poor performance on smaller classes such as LW or RW.

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
    RANDOM_FOREST_RESULTS
    / "classification_report.csv"
)


# Create the confusion matrix.
#
# This helps identify positions the model frequently confuses.
#
# Examples:
#
# LW vs LM
# RW vs RM
# CAM vs CM
# CM vs CDM

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
    RANDOM_FOREST_RESULTS
    / "position_prediction_confusion_matrix.csv"
)


# Random Forest provides feature importance values.
#
# These indicate how useful each of the six attributes is for
# distinguishing positions across the entire model.
#
# Feature importance does NOT mean that one feature defines a specific
# position by itself.

feature_importance = pd.DataFrame({
    "Feature":
        features,

    "Importance":
        model.feature_importances_
})


feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)


print(
    "\nFeature importance:"
)

print(
    feature_importance
)


feature_importance.to_csv(
    RANDOM_FOREST_RESULTS
    / "feature_importance.csv",
    index=False
)


# The train-test model above is used only to evaluate the method.
#
# Now train a final Random Forest using the complete dataset.
#
# This final model will generate the position-probability profiles
# that can be used in later exploratory stages.

final_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


final_model.fit(
    X,
    y
)


# Calculate the probability that every player resembles each possible
# EAFC position.

position_probabilities = (
    final_model.predict_proba(
        X
    )
)


# Convert all position probabilities into columns.

probability_columns = [
    f"Prob_{position}"
    for position in final_model.classes_
]


probability_df = pd.DataFrame(
    position_probabilities,
    columns=probability_columns,
    index=df.index
)


# Combine the original player information with the probability results.

results = pd.concat(
    [
        df,
        probability_df
    ],
    axis=1
)


# Find the three most likely positions for every player.

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


# Add the three most likely positions.

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
# A disagreement is not automatically an error.
#
# It may identify players whose attribute profiles resemble several
# possible roles.

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


# Columns used for displaying predictions.

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


# Function for inspecting individual players.

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
        f"\nRandom Forest position predictions for {player_name}:"
    )


    print(
        player[
            example_columns
        ]
    )


# Inspect several versatile or interesting players.

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
# These disagreements may be useful when investigating positional flexibility.

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
    "\nHighest-rated players whose Random Forest prediction "
    "differs from their listed position:"
)


print(
    position_disagreements[
        example_columns
    ]
    .head(30)
)


# Save the complete Random Forest position-probability results.

results.to_csv(
    RANDOM_FOREST_RESULTS
    / "player_position_predictions_random_forest.csv",
    index=False
)


# Save disagreement examples separately.

position_disagreements.to_csv(
    RANDOM_FOREST_RESULTS
    / "position_prediction_disagreements.csv",
    index=False
)


# Save a summary of this model.
#
# This will later allow direct comparison with Logistic Regression
# and richer-feature models.

model_summary = pd.DataFrame({
    "Model": [
        "Random Forest"
    ],
    "Features": [
        "PAC, SHO, PAS, DRI, DEF, PHY"
    ],
    "Accuracy": [
        accuracy
    ]
})


model_summary.to_csv(
    RANDOM_FOREST_RESULTS
    / "model_summary.csv",
    index=False
)


# Stage 3.3B does NOT determine a player's true or ideal soccer position.
#
# It learns relationships between the six EAFC attributes and the
# existing EAFC position labels.
#
# The main comparison is:
#
# Logistic Regression
#
# vs
#
# Random Forest
#
# using exactly the same six attributes.
#
# We want to compare:
#
# overall accuracy
#
# position-level F1 scores
#
# confusion between similar positions
#
# top-three position probabilities
#
# feature importance
#
# These results provide a baseline before Stage 3.4 investigates richer
# player attributes and position fingerprints.


print(
    "\nStage 3.3B Random Forest position prediction complete!"
)