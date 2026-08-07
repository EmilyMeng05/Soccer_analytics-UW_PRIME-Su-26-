import os
import pandas as pd


# DATA PREPARATION
#
# This file is responsible only for cleaning and preparing the raw EAFC data.
#
# Instead of repeating the same filtering and position-cleaning steps in
# every analysis file, I want to create one standardized dataset that all
# later stages can use.
#
# The general workflow is:
#
# Raw EAFC26 data
#
# ↓
#
# Select useful columns
#
# ↓
#
# Standardize position names
#
# ↓
#
# Remove very low-rated players
#
# ↓
#
# Separate goalkeepers and outfield players
#
# ↓
#
# Save cleaned datasets
#
# Later, this same process can be applied to previous EAFC seasons.


# Create folders if they do not already exist.

os.makedirs(
    "processed",
    exist_ok=True
)

os.makedirs(
    "results",
    exist_ok=True
)


# Read the raw EAFC26 dataset.

df = pd.read_csv(
    "EAFC26-Men.csv"
)

print(
    "Original dataset shape:",
    df.shape
)


# Display all available columns.
#
# This is especially useful now because the original dataset contains
# many more variables than the six aggregate EAFC attributes.
#
# Later, Stage 3.3 can investigate whether some of these additional
# variables improve position prediction.

print("\nAvailable columns:")

for column in df.columns:
    print(column)


# Keep the columns currently needed throughout the project.
#
# I am intentionally keeping Alternative positions because it may become
# especially useful for the positional versatility analysis later.

important_columns = [
    "ID",
    "Name",
    "OVR",
    "Position",
    "Alternative positions",
    "Age",
    "Nation",
    "League",
    "Team",
    "play style",
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY"
]


# Check whether all required columns actually exist.
#
# This prevents confusing errors later if a column name changes
# between datasets.

missing_columns = [
    column
    for column in important_columns
    if column not in df.columns
]

if missing_columns:

    print(
        "\nWarning: the following expected columns are missing:"
    )

    print(
        missing_columns
    )


# Only select columns that actually exist.

available_columns = [
    column
    for column in important_columns
    if column in df.columns
]

cleaned = df[
    available_columns
].copy()


# Standardize the main position names.
#
# Using full names makes later tables and visualizations easier to understand.

position_map = {

    "GK":
        "Goalkeeper",

    "CB":
        "Center Back",

    "LB":
        "Left Back",

    "RB":
        "Right Back",

    "LWB":
        "Left Wing Back",

    "RWB":
        "Right Wing Back",

    "CDM":
        "Defensive Midfielder",

    "CM":
        "Central Midfielder",

    "CAM":
        "Attacking Midfielder",

    "LM":
        "Left Midfielder",

    "RM":
        "Right Midfielder",

    "LW":
        "Left Winger",

    "RW":
        "Right Winger",

    "CF":
        "Center Forward",

    "ST":
        "Striker"
}


cleaned["Position"] = (
    cleaned["Position"]
    .replace(
        position_map
    )
)


# Remove duplicate player rows.
#
# ID should identify individual players, so if duplicated IDs occur,
# keep only the first record.

if "ID" in cleaned.columns:

    duplicate_count = cleaned[
        "ID"
    ].duplicated().sum()

    print(
        "\nDuplicate player IDs:",
        duplicate_count
    )

    cleaned = cleaned.drop_duplicates(
        subset="ID",
        keep="first"
    )


# Remove players with missing values in the six core EAFC attributes.
#
# These attributes are required for:
#
# PCA
# clustering
# attacking/defensive contribution
# position prediction
#
# Therefore, players without these values cannot be used in the
# current modeling pipeline.

core_features = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY"
]


before_missing_filter = len(
    cleaned
)

cleaned = cleaned.dropna(
    subset=core_features
)

after_missing_filter = len(
    cleaned
)

print(
    "\nPlayers removed because of missing core attributes:",
    before_missing_filter - after_missing_filter
)


# Convert the six core features and OVR to numeric values.
#
# If any invalid text exists, pandas will convert it to NaN.

numeric_columns = [
    "OVR",
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY"
]


for column in numeric_columns:

    if column in cleaned.columns:

        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce"
        )


# Remove any rows that became missing after numeric conversion.

cleaned = cleaned.dropna(
    subset=numeric_columns
)


# Apply the same first-round quality filter used earlier in the project.
#
# OVR > 60 is NOT being used to define elite players.
#
# It simply removes very low-rated players while keeping a large
# candidate pool for analysis.

cleaned = cleaned[
    cleaned["OVR"] > 60
].copy()


print(
    "\nPlayers after OVR filtering:",
    cleaned.shape
)


# Separate goalkeepers from outfield players.
#
# Goalkeepers have a very different attribute system and should not be
# analyzed using PAC, SHO, PAS, DRI, DEF, and PHY in the same way.

goalkeepers = cleaned[
    cleaned["Position"]
    == "Goalkeeper"
].copy()


outfield = cleaned[
    cleaned["Position"]
    != "Goalkeeper"
].copy()


print(
    "\nGoalkeepers:",
    len(goalkeepers)
)

print(
    "Outfield players:",
    len(outfield)
)


# Check the distribution of positions.
#
# This is useful for Stage 3.3 because supervised learning models may
# struggle when some positions contain far fewer players than others.

print(
    "\nOutfield position counts:"
)

print(
    outfield[
        "Position"
    ]
    .value_counts()
)


# Check missing values after cleaning.

print(
    "\nMissing values after cleaning:"
)

print(
    cleaned.isna().sum()
)


# Save the complete cleaned dataset.

cleaned.to_csv(
    "processed/cleaned_eafc26_players.csv",
    index=False
)


# Save the outfield-player dataset.
#
# Most of the current project will use this file.

outfield.to_csv(
    "processed/cleaned_eafc26_outfield_players.csv",
    index=False
)


# Save goalkeepers separately.
#
# Goalkeepers can receive their own analysis later.

goalkeepers.to_csv(
    "processed/cleaned_eafc26_goalkeepers.csv",
    index=False
)


# Save the position counts separately.

position_counts = (
    outfield[
        "Position"
    ]
    .value_counts()
    .rename_axis(
        "Position"
    )
    .reset_index(
        name="PlayerCount"
    )
)


position_counts.to_csv(
    "processed/eafc26_position_counts.csv",
    index=False
)


# Save a summary of the six main EAFC attributes.

attribute_summary = (
    outfield[
        core_features
    ]
    .describe()
    .round(2)
)


attribute_summary.to_csv(
    "processed/eafc26_attribute_summary.csv"
)


# Later we will probably want to use more detailed EAFC attributes.
#
# Instead of deleting the original 59-column dataset, save a filtered
# full-feature version as well.
#
# This keeps all available variables for future machine-learning experiments.

full_feature_data = df.copy()


# Standardize position names in the full dataset too.

full_feature_data[
    "Position"
] = (
    full_feature_data[
        "Position"
    ]
    .replace(
        position_map
    )
)


# Apply the same OVR cutoff.

full_feature_data = full_feature_data[
    full_feature_data[
        "OVR"
    ] > 60
].copy()


# Remove goalkeepers for the detailed outfield feature dataset.

full_feature_outfield = full_feature_data[
    full_feature_data[
        "Position"
    ] != "Goalkeeper"
].copy()


full_feature_outfield.to_csv(
    "processed/eafc26_outfield_full_features.csv",
    index=False
)


print(
    "\nSaved:"
)

print(
    "processed/cleaned_eafc26_players.csv"
)

print(
    "processed/cleaned_eafc26_outfield_players.csv"
)

print(
    "processed/cleaned_eafc26_goalkeepers.csv"
)

print(
    "processed/eafc26_position_counts.csv"
)

print(
    "processed/eafc26_attribute_summary.csv"
)

print(
    "processed/eafc26_outfield_full_features.csv"
)


print(
    "\nDataset preparation complete!"
)