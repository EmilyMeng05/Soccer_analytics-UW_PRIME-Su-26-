from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# STAGE 3.5A: LOGISTIC REGRESSION FUNCTIONAL-POSITION PREDICTION
#
# Stage 3.3 used Logistic Regression to predict 11 exact EAFC
# positions using the six broad EAFC attributes.
#
# Stage 3.4.2 combined similar left/right positions into eight
# functional positions.
#
# This stage performs two Logistic Regression experiments:
#
# Experiment 1:
# Use the six broad EAFC attributes.
#
# Experiment 2:
# Use the six broad attributes and 28 detailed attributes.
#
# We compare top-1, top-2, and top-3 accuracy because a player may
# reasonably resemble more than one functional position.


# Define the project paths.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
INPUT_FILE = DATA / "processed" / "eafc26_functional_positions.csv"

RESULTS = (
    PROJECT_ROOT
    / "results"
    / "stage_3_5"
    / "logistic_regression"
)

RESULTS.mkdir(parents=True, exist_ok=True)


# Check that Stage 3.4.2 created the required dataset.

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Could not find the functional-position dataset.\n"
        "Run stage3_4_2_functional_positions.py first.\n\n"
        f"Expected file:\n{INPUT_FILE}"
    )


# Read the functional-position dataset.

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)


# Define the six broad EAFC attributes.
#
# These are the same features used during Stage 3.3.

broad_features = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY"
]


# Define the 28 detailed EAFC attributes found during Stage 3.4.

detailed_features = [
    "Acceleration",
    "Sprint Speed",
    "Positioning",
    "Finishing",
    "Shot Power",
    "Long Shots",
    "Volleys",
    "Penalties",
    "Vision",
    "Crossing",
    "Free Kick Accuracy",
    "Short Passing",
    "Long Passing",
    "Curve",
    "Agility",
    "Balance",
    "Reactions",
    "Ball Control",
    "Dribbling",
    "Composure",
    "Interceptions",
    "Heading Accuracy",
    "Standing Tackle",
    "Sliding Tackle",
    "Jumping",
    "Stamina",
    "Strength",
    "Aggression"
]


# Combine the six broad and 28 detailed attributes.

rich_features = broad_features + detailed_features

print("\nNumber of broad features:", len(broad_features))
print("Number of rich features:", len(rich_features))


# Check that all required columns are available.

required_columns = rich_features + ["Functional Position"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following required columns are missing:\n"
        f"{missing_columns}"
    )


# Calculate top-k accuracy.
#
# Top-1 accuracy:
# The model's first prediction must be correct.
#
# Top-2 or top-3 accuracy:
# The correct role must appear among the model's two or three most
# likely predictions.

def calculate_top_k_accuracy(
    y_true,
    probabilities,
    classes,
    top_k
):
    top_indices = np.argsort(
        probabilities,
        axis=1
    )[:, -top_k:]

    top_positions = classes[top_indices]

    correct = [
        actual_position in predicted_positions
        for actual_position, predicted_positions
        in zip(y_true, top_positions)
    ]

    return np.mean(correct)


# Save the confusion matrix as a CSV file and an image.
#
# Rows represent actual positions.
# Columns represent predicted positions.

def save_confusion_matrix(
    y_true,
    y_pred,
    position_labels,
    experiment_name
):
    confusion = confusion_matrix(
        y_true,
        y_pred,
        labels=position_labels
    )

    confusion_df = pd.DataFrame(
        confusion,
        index=position_labels,
        columns=position_labels
    )

    confusion_df.to_csv(
        RESULTS / f"{experiment_name}_confusion_matrix.csv"
    )

    figure, axis = plt.subplots(figsize=(11, 9))

    image = axis.imshow(
        confusion,
        cmap="Blues"
    )

    axis.set_xticks(range(len(position_labels)))
    axis.set_xticklabels(
        position_labels,
        rotation=45,
        ha="right"
    )

    axis.set_yticks(range(len(position_labels)))
    axis.set_yticklabels(position_labels)

    axis.set_xlabel("Predicted functional position")
    axis.set_ylabel("Actual functional position")

    axis.set_title(
        "Logistic Regression: "
        + experiment_name.replace("_", " ").title()
    )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()

    figure.savefig(
        RESULTS / f"{experiment_name}_confusion_matrix.png",
        dpi=200
    )

    plt.close(figure)


# Train and evaluate Logistic Regression using one feature set.
#
# This function is called once with six features and once with all
# 34 features.

def run_logistic_regression_experiment(
    experiment_name,
    features
):
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION:", experiment_name.upper())
    print("=" * 70)

    # Remove players who are missing any required feature or target.

    experiment_df = df.dropna(
        subset=features + ["Functional Position"]
    ).copy()

    print("\nPlayers used:", len(experiment_df))
    print("Number of features:", len(features))

    # Define the model inputs and prediction target.

    X = experiment_df[features]
    y = experiment_df["Functional Position"]

    # Split the dataset into training and testing sets.
    #
    # The model learns from 80% of the players.
    #
    # The remaining 20% are used to measure how well the model
    # generalizes to unseen players.
    #
    # stratify=y keeps the proportion of each role similar in both
    # datasets.

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create a pipeline that first standardizes the attributes and
    # then trains Logistic Regression.
    #
    # The scaler is fit only using the training data, preventing
    # information from the test data from leaking into training.

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=3000,
                solver="lbfgs",
                random_state=42
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    # Predict one role and the complete role-probability distribution
    # for every test player.

    y_pred = model.predict(X_test)
    y_probabilities = model.predict_proba(X_test)

    position_labels = (
        model.named_steps[
            "classifier"
        ].classes_
    )

    # Calculate top-1, top-2, and top-3 accuracy.

    top_1_accuracy = accuracy_score(
        y_test,
        y_pred
    )

    top_2_accuracy = calculate_top_k_accuracy(
        y_test.to_numpy(),
        y_probabilities,
        position_labels,
        top_k=2
    )

    top_3_accuracy = calculate_top_k_accuracy(
        y_test.to_numpy(),
        y_probabilities,
        position_labels,
        top_k=3
    )

    print("\nTop-1 accuracy:", round(top_1_accuracy, 4))
    print("Top-2 accuracy:", round(top_2_accuracy, 4))
    print("Top-3 accuracy:", round(top_3_accuracy, 4))

    # Create the classification report.
    #
    # This shows precision, recall, and F1-score for each role.

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    print("\nClassification report:")
    print(report_df.round(3))

    report_df.to_csv(
        RESULTS / f"{experiment_name}_classification_report.csv"
    )

    # Save the confusion matrix.

    save_confusion_matrix(
        y_test,
        y_pred,
        position_labels,
        experiment_name
    )

    # Return the main measurements so the two feature experiments
    # can be compared.

    return {
        "Model": "Logistic Regression",
        "Feature Set": experiment_name,
        "Number of Features": len(features),
        "Top-1 Accuracy": top_1_accuracy,
        "Top-2 Accuracy": top_2_accuracy,
        "Top-3 Accuracy": top_3_accuracy
    }


# Run the six-feature experiment.

six_feature_results = run_logistic_regression_experiment(
    experiment_name="six_features",
    features=broad_features
)


# Run the 34-feature experiment.

rich_feature_results = run_logistic_regression_experiment(
    experiment_name="rich_features",
    features=rich_features
)


# Combine and save the experiment results.

model_comparison = pd.DataFrame([
    six_feature_results,
    rich_feature_results
])

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION MODEL COMPARISON")
print("=" * 70)
print(model_comparison.round(4))

model_comparison.to_csv(
    RESULTS / "model_comparison.csv",
    index=False
)


# The train/test models above are used for evaluation.
#
# We now train one final rich-feature model using all available
# players.
#
# This model is not used to calculate test accuracy.
#
# It is used to create role probabilities for every player so the
# probabilities can support lineup construction.

final_df = df.dropna(
    subset=rich_features + ["Functional Position"]
).copy()

X_all = final_df[rich_features]
y_all = final_df["Functional Position"]


# Create and train the final Logistic Regression model.

final_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "classifier",
        LogisticRegression(
            C=1.0,
            max_iter=3000,
            solver="lbfgs",
            random_state=42
        )
    )
])

final_model.fit(
    X_all,
    y_all
)


# Calculate the complete probability distribution across all eight
# functional positions for every player.

all_probabilities = final_model.predict_proba(X_all)

functional_positions = (
    final_model.named_steps[
        "classifier"
    ].classes_
)


# Begin the result with the complete player dataset.

player_predictions = final_df.copy()


# Add one probability column for each functional position.

for position_index, position_name in enumerate(
    functional_positions
):
    player_predictions[
        f"Prob_{position_name}"
    ] = all_probabilities[:, position_index]


# Sort every player's position probabilities from highest to lowest.

ranked_position_indices = np.argsort(
    all_probabilities,
    axis=1
)[:, ::-1]


# Save the top three functional positions and their probabilities.

for rank in range(3):
    player_predictions[
        f"PredictedFunctionalPosition{rank + 1}"
    ] = functional_positions[
        ranked_position_indices[:, rank]
    ]

    player_predictions[
        f"FunctionalPositionProbability{rank + 1}"
    ] = all_probabilities[
        np.arange(len(player_predictions)),
        ranked_position_indices[:, rank]
    ]


# Check whether the first predicted functional position matches the
# position derived from the player's official EAFC label.
#
# A disagreement may indicate that the player's attributes resemble
# another functional role.

player_predictions[
    "OfficialFunctionalPositionMatch"
] = (
    player_predictions[
        "PredictedFunctionalPosition1"
    ]
    == player_predictions[
        "Functional Position"
    ]
)


# Define the columns used when inspecting example players.

example_columns = [
    "Name",
    "Position",
    "Functional Position",
    "PredictedFunctionalPosition1",
    "FunctionalPositionProbability1",
    "PredictedFunctionalPosition2",
    "FunctionalPositionProbability2",
    "PredictedFunctionalPosition3",
    "FunctionalPositionProbability3"
]


print("\nExample functional-position predictions:")

print(
    player_predictions[
        example_columns
    ].head(25)
)


# Create a function for inspecting individual players.

def show_player_positions(player_name):
    player = player_predictions[
        player_predictions["Name"].str.lower()
        == player_name.lower()
    ]

    if len(player) == 0:
        print(f"\nPlayer '{player_name}' not found.")
        return

    print(
        f"\nLogistic Regression functional-position predictions "
        f"for {player_name}:"
    )

    print(player[example_columns])


# Inspect several interesting players.

show_player_positions("Jude Bellingham")
show_player_positions("Federico Valverde")
show_player_positions("Achraf Hakimi")
show_player_positions("Lionel Messi")
show_player_positions("Trent Alexander-Arnold")


# Find highly rated players whose first predicted position differs
# from their listed functional position.

position_disagreements = player_predictions[
    player_predictions[
        "OfficialFunctionalPositionMatch"
    ]
    == False
].copy()

position_disagreements = position_disagreements.sort_values(
    "OVR",
    ascending=False
)

print(
    "\nHighest-rated players whose predicted functional position "
    "differs from their listed functional position:"
)

print(
    position_disagreements[
        example_columns
    ].head(30)
)


# Save the full player-probability dataset.

player_predictions.to_csv(
    RESULTS / "all_player_functional_position_probabilities.csv",
    index=False
)


# Save the disagreement examples separately.

position_disagreements.to_csv(
    RESULTS / "functional_position_disagreements.csv",
    index=False
)


print("\nResults saved to:")
print(RESULTS)

print(
    "\nStage 3.5A Logistic Regression functional-position "
    "prediction complete!"
)