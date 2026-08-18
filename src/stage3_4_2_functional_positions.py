from pathlib import Path

import pandas as pd


# STAGE 3.4.2: CREATE FUNCTIONAL POSITION LABELS
#
# The previous position-prediction models tried to predict 11 exact
# EAFC positions.
#
# However, Stage 3.4 showed that the attribute profiles for the left
# and right versions of several positions were extremely similar.
#
# For example:
#
# Left Back and Right Back
# Left Midfielder and Right Midfielder
# Left Winger and Right Winger
#
# The six broad attributes and the 28 detailed attributes do not
# strongly describe which side of the field a player uses.
#
# Therefore, this stage combines those left/right positions into
# broader functional roles.
#
# This changes the prediction problem from 11 exact positions to
# eight functional positions.
#
# The player attributes are NOT changed.
#
# We only create a new target column named:
#
# Functional Position
#
# The resulting dataset will be used by the Logistic Regression and
# Random Forest models in Stage 3.5.


# Define the main project directory.
#
# This script should be stored inside:
#
# src/
#
# Therefore, moving up one directory gives the main project folder.

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Define the data directories.

DATA = PROJECT_ROOT / "data"

PROCESSED_DATA = DATA / "processed"


# Input dataset created during prepare_dataset.py.
#
# This dataset contains:
#
# player information
# the six broad EAFC attributes
# the 28 detailed EAFC attributes
# the original EAFC position

INPUT_FILE = (
    PROCESSED_DATA
    / "eafc26_outfield_full_features.csv"
)


# Define the new output file.

OUTPUT_FILE = (
    PROCESSED_DATA
    / "eafc26_functional_positions.csv"
)


# Check that the input dataset exists before continuing.

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Could not find the input dataset:\n{INPUT_FILE}"
    )


# Read the full-feature outfield-player dataset.

df = pd.read_csv(
    INPUT_FILE
)


print(
    "Original dataset shape:",
    df.shape
)


# Make sure the dataset contains the original Position column.

if "Position" not in df.columns:

    raise ValueError(
        "The input dataset must contain a 'Position' column."
    )


# Map the 11 original EAFC positions to eight functional positions.
#
# Three pairs of left/right positions are combined:
#
# Left Back + Right Back
#     -> Full Back
#
# Left Midfielder + Right Midfielder
#     -> Wide Midfielder
#
# Left Winger + Right Winger
#     -> Winger
#
# Positions without a left/right distinction keep their original name.

position_to_functional_position = {

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


# Create the new functional-position target.
#
# The original Position column is kept.
#
# Keeping both columns allows us to compare:
#
# original exact EAFC position
#
# versus
#
# broader functional position

df[
    "Functional Position"
] = df[
    "Position"
].map(
    position_to_functional_position
)


# Check whether any original positions were not included in the mapping.

unmapped_positions = (
    df.loc[
        df[
            "Functional Position"
        ].isna(),
        "Position"
    ]
    .dropna()
    .unique()
)


if len(unmapped_positions) > 0:

    print(
        "\nUnmapped positions:"
    )

    print(
        sorted(
            unmapped_positions
        )
    )


# Remove players whose positions were not part of the functional-role map.
#
# Goalkeepers should already have been removed during preprocessing.
#
# However, this step protects the new dataset in case an unexpected
# position appears.

functional_df = (
    df.dropna(
        subset=[
            "Functional Position"
        ]
    )
    .copy()
)


# Print the number of players in each original position.

print(
    "\nPlayers per original position:"
)

print(
    functional_df[
        "Position"
    ]
    .value_counts()
)


# Print the number of players in each new functional position.

print(
    "\nPlayers per functional position:"
)

print(
    functional_df[
        "Functional Position"
    ]
    .value_counts()
)


# Count the number of functional positions.
#
# We expect this value to be eight.

number_of_functional_positions = (
    functional_df[
        "Functional Position"
    ]
    .nunique()
)


print(
    "\nNumber of functional positions:",
    number_of_functional_positions
)


# Save the new dataset.
#
# Both Stage 3.5 models will read this exact file.
#
# This ensures that Logistic Regression and Random Forest use the
# same players and the same target labels.

PROCESSED_DATA.mkdir(
    parents=True,
    exist_ok=True
)


functional_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    "\nNew dataset shape:",
    functional_df.shape
)

print(
    "\nStage 3.4.2 functional-position dataset complete!"
)