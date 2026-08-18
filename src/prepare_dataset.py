from pathlib import Path

import pandas as pd


# DATA PREPARATION
#
# This is the only file that should read the raw EAFC26 dataset.
#
# Its job is to:
#
# 1. Read the raw 59-column EAFC26 dataset.
# 2. Standardize position names.
# 3. Apply the OVR > 60 candidate filter.
# 4. Separate goalkeepers and outfield players.
# 5. Save a simple six-attribute dataset.
# 6. Save a full-feature dataset for richer Stage 3 experiments.
#
# All later stages should read one of the files created here instead
# of cleaning the raw data again.


# Define project paths.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA = PROJECT_ROOT / "data"

RAW_DATA = DATA / "unprocessed"

PROCESSED_DATA = DATA / "processed"


# Create the processed-data folder if necessary.

PROCESSED_DATA.mkdir(
    parents=True,
    exist_ok=True
)


# Read the original EAFC26 dataset.
#
# This should contain all original columns.

df = pd.read_csv(
    RAW_DATA / "EAFC26-Men.csv"
)

print(
    "Original dataset shape:",
    df.shape
)

print(
    "\nOriginal number of columns:",
    len(df.columns)
)


# Print all available raw columns.
#
# This will be especially useful when we begin using richer attributes.

print(
    "\nAvailable raw columns:"
)

for column in df.columns:

    print(column)


# Standardize the main EAFC position labels.

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


df["Position"] = (
    df["Position"]
    .replace(
        position_map
    )
)


# Remove duplicate player IDs.

if "ID" in df.columns:

    duplicate_count = (
        df["ID"]
        .duplicated()
        .sum()
    )

    print(
        "\nDuplicate IDs:",
        duplicate_count
    )

    df = df.drop_duplicates(
        subset="ID",
        keep="first"
    )


# Convert the main numerical attributes to numeric values.

core_features = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY"
]

numeric_columns = [
    "OVR",
    *core_features
]


for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Apply the same candidate filter used throughout the project.
#
# This is not intended to define an elite player.
#
# It simply removes very low-rated players while preserving a large
# candidate pool.

df = df[
    df["OVR"] > 60
].copy()


print(
    "\nPlayers after OVR filtering:",
    df.shape
)


# Separate goalkeepers and outfield players.

goalkeepers_full = df[
    df["Position"]
    == "Goalkeeper"
].copy()


outfield_full = df[
    df["Position"]
    != "Goalkeeper"
].copy()


# Outfield players need the six main attributes.
#
# Remove players missing any of these values.

outfield_full = outfield_full.dropna(
    subset=core_features
).copy()


print(
    "\nOutfield players:",
    len(outfield_full)
)

print(
    "Goalkeepers:",
    len(goalkeepers_full)
)


# Create the simplified dataset used by the earlier project stages.
#
# This contains:
#
# player information
# +
# PAC SHO PAS DRI DEF PHY

basic_columns = [
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


available_basic_columns = [
    column
    for column in basic_columns
    if column in outfield_full.columns
]


outfield_basic = outfield_full[
    available_basic_columns
].copy()


# Save the simplified outfield dataset.
#
# Stages 1 through 3.3 currently use this representation.

outfield_basic.to_csv(
    PROCESSED_DATA
    / "cleaned_eafc26_outfield_players.csv",
    index=False
)


# Save the goalkeepers separately.

goalkeepers_full.to_csv(
    PROCESSED_DATA
    / "cleaned_eafc26_goalkeepers.csv",
    index=False
)


# Save the complete filtered player dataset.

df.to_csv(
    PROCESSED_DATA
    / "cleaned_eafc26_players.csv",
    index=False
)


# MOST IMPORTANT FOR STAGE 3.4:
#
# Save ALL original columns for outfield players.
#
# Unlike cleaned_eafc26_outfield_players.csv, this file should still contain
# most of the original ~59 EAFC columns.
#
# Stage 3.4 can use this file to investigate richer attributes.

outfield_full.to_csv(
    PROCESSED_DATA
    / "eafc26_outfield_full_features.csv",
    index=False
)


# Save position counts.

position_counts = (
    outfield_basic["Position"]
    .value_counts()
    .rename_axis(
        "Position"
    )
    .reset_index(
        name="PlayerCount"
    )
)


position_counts.to_csv(
    PROCESSED_DATA
    / "eafc26_position_counts.csv",
    index=False
)


# Save a basic summary of the six main attributes.

attribute_summary = (
    outfield_basic[
        core_features
    ]
    .describe()
    .round(2)
)


attribute_summary.to_csv(
    PROCESSED_DATA
    / "eafc26_attribute_summary.csv"
)


print(
    "\nSaved datasets:"
)

print(
    PROCESSED_DATA
    / "cleaned_eafc26_outfield_players.csv"
)

print(
    PROCESSED_DATA
    / "cleaned_eafc26_goalkeepers.csv"
)

print(
    PROCESSED_DATA
    / "cleaned_eafc26_players.csv"
)

print(
    PROCESSED_DATA
    / "eafc26_outfield_full_features.csv"
)


print(
    "\nBasic outfield shape:",
    outfield_basic.shape
)

print(
    "Full-feature outfield shape:",
    outfield_full.shape
)


print(
    "\nDataset preparation complete!"
)