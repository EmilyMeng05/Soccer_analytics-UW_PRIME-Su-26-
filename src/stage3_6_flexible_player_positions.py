from pathlib import Path

import numpy as np
import pandas as pd


# STAGE 3.6: ADD FLEXIBLE POSITIONS TO THE PLAYER DATASET
#
# Stage 3.5 compared Logistic Regression and Random Forest for
# predicting eight functional player roles.
#
# The 34-feature Logistic Regression model achieved the strongest
# results:
#
# Top-1 accuracy = 77.22%
# Top-2 accuracy = 93.39%
# Top-3 accuracy = 98.27%
#
# Therefore, this stage uses the Logistic Regression probabilities
# to add two possible flexible positions to every player.
#
# The final dataset keeps the player's original EAFC Position column.
#
# It does not keep the temporary Functional Position column used
# during model training.
#
# Four new columns are added:
#
# Flexible Position 1
# Flexible Position 1 Probability
# Flexible Position 2
# Flexible Position 2 Probability
#
# The player's original position is excluded when selecting the two
# flexible positions.
#
# For example, if a player is originally a Right Back, Full Back is
# considered their existing role and will not be selected as one of
# the two additional flexible positions.


# Define the project paths.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"
RESULTS = PROJECT_ROOT / "results"


# Read the complete player-probability dataset created by the
# 34-feature Logistic Regression model in Stage 3.5.

INPUT_FILE = (
    RESULTS
    / "stage_3_5"
    / "logistic_regression"
    / "all_player_functional_position_probabilities.csv"
)


# Save the expanded player dataset inside data/processed so Stage 4
# can use it as its central input.

OUTPUT_FILE = (
    DATA
    / "processed"
    / "eafc26_players_with_flexible_positions.csv"
)


# Check that the Stage 3.5 probability file exists.

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Could not find the Logistic Regression probability file.\n"
        "Run stage3_5_logistic_functional_positions.py first.\n\n"
        f"Expected file:\n{INPUT_FILE}"
    )


# Read the Stage 3.5 results.

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# Check that the original EAFC Position column is available.

if "Position" not in df.columns:
    raise ValueError(
        "The input dataset must contain the original 'Position' column."
    )


# Map each original EAFC position to the broader role used during
# Stage 3.5.
#
# This mapping is used only while deciding which model prediction
# represents the player's existing role.
#
# The mapped role is not retained as a column in the final dataset.

position_to_model_role = {
    "Center Back": "Center Back",
    "Left Back": "Full Back",
    "Right Back": "Full Back",
    "Defensive Midfielder": "Defensive Midfielder",
    "Central Midfielder": "Central Midfielder",
    "Attacking Midfielder": "Attacking Midfielder",
    "Left Midfielder": "Wide Midfielder",
    "Right Midfielder": "Wide Midfielder",
    "Left Winger": "Winger",
    "Right Winger": "Winger",
    "Striker": "Striker"
}


# Create a temporary Series containing each player's existing model
# role.
#
# For example:
#
# Right Back -> Full Back
# Left Winger -> Winger
#
# This Series is temporary and will not be added to the output.

existing_model_roles = df["Position"].map(
    position_to_model_role
)


# Check whether any original positions were not included in the map.

unmapped_positions = sorted(
    df.loc[
        existing_model_roles.isna(),
        "Position"
    ]
    .dropna()
    .unique()
)

if unmapped_positions:
    raise ValueError(
        "The following positions were not included in the role map:\n"
        f"{unmapped_positions}"
    )


# Find the columns containing the probabilities for all eight model
# roles.
#
# Stage 3.5 saved these columns using names such as:
#
# Prob_Center Back
# Prob_Central Midfielder
# Prob_Striker

probability_columns = [
    column
    for column in df.columns
    if column.startswith("Prob_")
]


if not probability_columns:
    raise ValueError(
        "No position-probability columns were found.\n"
        "Expected columns beginning with 'Prob_'."
    )


print("\nProbability columns found:")

for column in probability_columns:
    print(column)


# Store the model-role name associated with every probability column.
#
# For example:
#
# Prob_Striker -> Striker

probability_column_to_role = {
    column: column.replace("Prob_", "", 1)
    for column in probability_columns
}


# Create lists for the two flexible positions and their probabilities.

flexible_position_1 = []
flexible_position_1_probability = []

flexible_position_2 = []
flexible_position_2_probability = []


# Examine the complete probability distribution for every player.

for row_index, row in df.iterrows():
    existing_role = existing_model_roles.loc[row_index]

    alternative_roles = []

    # Examine the probability associated with every possible role.

    for probability_column in probability_columns:
        role = probability_column_to_role[
            probability_column
        ]

        probability = row[
            probability_column
        ]

        # Exclude the role corresponding to the player's original
        # EAFC position.
        #
        # This ensures that the new columns represent additional
        # positions rather than repeating the player's current role.

        if role == existing_role:
            continue

        alternative_roles.append(
            (
                role,
                probability
            )
        )

    # Sort the alternative roles from highest to lowest probability.

    alternative_roles.sort(
        key=lambda role_and_probability: role_and_probability[1],
        reverse=True
    )

    # Save the two most likely non-official roles.

    first_role, first_probability = alternative_roles[0]
    second_role, second_probability = alternative_roles[1]

    flexible_position_1.append(
        first_role
    )

    flexible_position_1_probability.append(
        first_probability
    )

    flexible_position_2.append(
        second_role
    )

    flexible_position_2_probability.append(
        second_probability
    )


# Add the flexible positions and probabilities to the dataset.

df[
    "Flexible Position 1"
] = flexible_position_1

df[
    "Flexible Position 1 Probability"
] = flexible_position_1_probability

df[
    "Flexible Position 2"
] = flexible_position_2

df[
    "Flexible Position 2 Probability"
] = flexible_position_2_probability


# Add indicators showing whether each flexible position has reasonably
# strong model support.
#
# A probability of at least 0.10 is used as the initial threshold.
#
# This prevents Stage 4 from treating a position with an extremely
# small probability as strong evidence of flexibility.
#
# The probability values are still saved, so this threshold can be
# changed later without rerunning the position model.

FLEXIBILITY_THRESHOLD = 0.10

df[
    "Flexible Position 1 Supported"
] = (
    df[
        "Flexible Position 1 Probability"
    ]
    >= FLEXIBILITY_THRESHOLD
)

df[
    "Flexible Position 2 Supported"
] = (
    df[
        "Flexible Position 2 Probability"
    ]
    >= FLEXIBILITY_THRESHOLD
)


# Remove the temporary Functional Position column.
#
# The final dataset keeps only the original EAFC Position column.

if "Functional Position" in df.columns:
    df = df.drop(
        columns=[
            "Functional Position"
        ]
    )


# Remove the earlier Stage 3.5 prediction-summary columns.
#
# These are no longer necessary because Stage 3.6 has created the two
# alternative-position columns in the desired format.
#
# The complete probability columns are temporarily retained so they
# can be used during Stage 4 when calculating player-role suitability.

old_prediction_columns = [
    column
    for column in df.columns
    if column.startswith("PredictedFunctionalPosition")
    or column.startswith("FunctionalPositionProbability")
    or column == "OfficialFunctionalPositionMatch"
]


df = df.drop(
    columns=old_prediction_columns,
    errors="ignore"
)


# Arrange the new flexible-position columns directly after the
# original Position column.

flexible_columns = [
    "Flexible Position 1",
    "Flexible Position 1 Probability",
    "Flexible Position 1 Supported",
    "Flexible Position 2",
    "Flexible Position 2 Probability",
    "Flexible Position 2 Supported"
]


remaining_columns = [
    column
    for column in df.columns
    if column not in flexible_columns
]


position_index = remaining_columns.index(
    "Position"
)


ordered_columns = (
    remaining_columns[
        :position_index + 1
    ]
    + flexible_columns
    + remaining_columns[
        position_index + 1:
    ]
)


df = df[
    ordered_columns
]


# Print example results.

example_columns = [
    "Name",
    "Position",
    "Flexible Position 1",
    "Flexible Position 1 Probability",
    "Flexible Position 1 Supported",
    "Flexible Position 2",
    "Flexible Position 2 Probability",
    "Flexible Position 2 Supported"
]


print("\nExample flexible-position results:")

print(
    df[
        example_columns
    ]
    .head(25)
)


# Create a function for inspecting individual players.

def show_player_flexible_positions(player_name):
    player = df[
        df[
            "Name"
        ]
        .str.lower()
        == player_name.lower()
    ]

    if len(player) == 0:
        print(f"\nPlayer '{player_name}' not found.")
        return

    print(
        f"\nFlexible-position predictions for {player_name}:"
    )

    print(
        player[
            example_columns
        ]
    )


# Inspect several example players.

show_player_flexible_positions("Jude Bellingham")
show_player_flexible_positions("Federico Valverde")
show_player_flexible_positions("Achraf Hakimi")
show_player_flexible_positions("Lionel Messi")
show_player_flexible_positions("Trent Alexander-Arnold")


# Print the number of supported alternative positions.

print("\nFlexible Position 1 support counts:")

print(
    df[
        "Flexible Position 1 Supported"
    ]
    .value_counts()
)


print("\nFlexible Position 2 support counts:")

print(
    df[
        "Flexible Position 2 Supported"
    ]
    .value_counts()
)


# Save the expanded player dataset.

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nExpanded player dataset shape:", df.shape)

print("\nExpanded player dataset saved to:")

print(OUTPUT_FILE)

print("\nStage 3.6 flexible-position expansion complete!")