from pathlib import Path

import numpy as np
import pandas as pd


# STAGE 4.1: PLAYER-ROLE SUITABILITY
#
# Stage 3.6 expanded the outfield-player dataset by adding two
# potential flexible positions for every player.
#
# Stage 4.1 asks:
#
# How suitable is every player for each possible lineup role?
#
# A player's role-suitability score combines:
#
# 1. Overall player quality
# 2. Attributes that are important for the role
# 3. Evidence that the player can perform that role
#
# The final score is:
#
# Player-Role Suitability =
#     35% Overall Rating
#     + 40% Role-Specific Attribute Score
#     + 25% Position Fit
#
# The score is calculated on a scale from 0 to 100.
#
# Official positions receive a Position Fit score of 100.
#
# Flexible positions use the position probability created by the
# 34-feature Logistic Regression model.
#
# A player is eligible for a flexible position only when the Stage 3.6
# Supported column is True.


# Define the project paths.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"

RESULTS = (
    PROJECT_ROOT
    / "results"
    / "stage_4_1"
)

RESULTS.mkdir(parents=True, exist_ok=True)

INPUT_FILE = (
    DATA
    / "processed"
    / "eafc26_players_with_flexible_positions.csv"
)

OUTPUT_FILE = (
    DATA
    / "processed"
    / "eafc26_player_role_suitability.csv"
)


# Check that Stage 3.6 created the required dataset.

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Could not find the expanded outfield-player dataset.\n"
        "Run stage3_6_flexible_player_positions.py first.\n\n"
        f"Expected file:\n{INPUT_FILE}"
    )


# Read the expanded outfield-player dataset.

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# Check that the columns created during Stage 3.6 are available.

required_position_columns = [
    "Position",
    "Flexible Position 1",
    "Flexible Position 1 Probability",
    "Flexible Position 1 Supported",
    "Flexible Position 2",
    "Flexible Position 2 Probability",
    "Flexible Position 2 Supported",
    "OVR"
]

missing_position_columns = [
    column
    for column in required_position_columns
    if column not in df.columns
]

if missing_position_columns:
    raise ValueError(
        "The following required columns are missing:\n"
        f"{missing_position_columns}"
    )


# Convert OVR and flexible-position probabilities to numeric values.

numeric_columns = [
    "OVR",
    "Flexible Position 1 Probability",
    "Flexible Position 2 Probability"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# Convert the Supported columns into reliable Boolean values.
#
# This handles both actual Boolean values and text such as:
#
# "True"
# "False"

def convert_to_boolean(value):
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() == "true"


df[
    "Flexible Position 1 Supported"
] = df[
    "Flexible Position 1 Supported"
].apply(convert_to_boolean)

df[
    "Flexible Position 2 Supported"
] = df[
    "Flexible Position 2 Supported"
].apply(convert_to_boolean)


# Map the original EAFC positions to the broader role groups used
# during the functional-position experiment.
#
# This mapping does not replace the player's original Position column.
#
# It is used only to determine whether a role matches the player's
# official EAFC position.

position_to_role = {
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


# Create a temporary role group for every player's original position.

df[
    "Original Role Group"
] = df[
    "Position"
].map(position_to_role)


# Check whether any original positions were not included in the map.

unmapped_positions = sorted(
    df.loc[
        df[
            "Original Role Group"
        ].isna(),
        "Position"
    ]
    .dropna()
    .unique()
)

if unmapped_positions:
    raise ValueError(
        "The following original positions were not mapped:\n"
        f"{unmapped_positions}"
    )


# Define the attributes that are important for each role.
#
# The weights within each role add to 1.
#
# These weights are an initial soccer-based definition rather than
# values learned from match outcomes.
#
# Later stages can test how sensitive the selected lineups are to
# changes in these weights.

role_attribute_weights = {
    "Center Back": {
        "Interceptions": 0.20,
        "Heading Accuracy": 0.15,
        "Standing Tackle": 0.20,
        "Sliding Tackle": 0.15,
        "Strength": 0.15,
        "Aggression": 0.10,
        "Jumping": 0.05
    },

    "Full Back": {
        "Acceleration": 0.15,
        "Sprint Speed": 0.15,
        "Crossing": 0.15,
        "Stamina": 0.20,
        "Interceptions": 0.12,
        "Standing Tackle": 0.13,
        "Short Passing": 0.10
    },

    "Defensive Midfielder": {
        "Interceptions": 0.20,
        "Standing Tackle": 0.15,
        "Long Passing": 0.15,
        "Short Passing": 0.15,
        "Stamina": 0.15,
        "Aggression": 0.10,
        "Composure": 0.10
    },

    "Central Midfielder": {
        "Short Passing": 0.20,
        "Long Passing": 0.18,
        "Vision": 0.17,
        "Ball Control": 0.15,
        "Composure": 0.10,
        "Stamina": 0.12,
        "Reactions": 0.08
    },

    "Attacking Midfielder": {
        "Vision": 0.20,
        "Dribbling": 0.15,
        "Ball Control": 0.15,
        "Short Passing": 0.15,
        "Positioning": 0.15,
        "Long Shots": 0.10,
        "Curve": 0.10
    },

    "Wide Midfielder": {
        "Acceleration": 0.13,
        "Sprint Speed": 0.12,
        "Crossing": 0.18,
        "Dribbling": 0.12,
        "Short Passing": 0.12,
        "Stamina": 0.15,
        "Vision": 0.10,
        "Ball Control": 0.08
    },

    "Winger": {
        "Acceleration": 0.15,
        "Sprint Speed": 0.15,
        "Dribbling": 0.15,
        "Crossing": 0.15,
        "Positioning": 0.15,
        "Finishing": 0.12,
        "Ball Control": 0.08,
        "Agility": 0.05
    },

    "Striker": {
        "Finishing": 0.22,
        "Positioning": 0.20,
        "Shot Power": 0.12,
        "Heading Accuracy": 0.12,
        "Reactions": 0.10,
        "Strength": 0.10,
        "Volleys": 0.08,
        "Composure": 0.06
    }
}


# Check that all role-specific attributes exist in the dataset.

all_role_attributes = sorted({
    attribute
    for role_weights in role_attribute_weights.values()
    for attribute in role_weights
})

missing_role_attributes = [
    attribute
    for attribute in all_role_attributes
    if attribute not in df.columns
]

if missing_role_attributes:
    raise ValueError(
        "The following role-specific attributes are missing:\n"
        f"{missing_role_attributes}"
    )


# Convert all role-specific attributes to numeric values.

for attribute in all_role_attributes:
    df[attribute] = pd.to_numeric(
        df[attribute],
        errors="coerce"
    )


# Define the contribution of each part of the final score.

OVERALL_WEIGHT = 0.35
ATTRIBUTE_WEIGHT = 0.40
POSITION_FIT_WEIGHT = 0.25


# Calculate the weighted attribute score for one player and one role.

def calculate_role_attribute_score(row, role):
    attribute_weights = role_attribute_weights[role]

    weighted_values = [
        row[attribute] * weight
        for attribute, weight
        in attribute_weights.items()
    ]

    if any(
        pd.isna(value)
        for value in weighted_values
    ):
        return np.nan

    return sum(weighted_values)


# Determine whether a player is eligible for a role and calculate
# their Position Fit score.
#
# Official role:
# Position Fit = 100
#
# Supported flexible role:
# Position Fit = model probability × 100
#
# Unsupported role:
# The player is not eligible.

def calculate_position_fit(row, role):
    if row["Original Role Group"] == role:
        return {
            "Eligible": True,
            "Position Fit Score": 100.0,
            "Role Source": "Official Position"
        }

    flexible_1_matches = (
        row["Flexible Position 1"] == role
        and row["Flexible Position 1 Supported"]
    )

    flexible_2_matches = (
        row["Flexible Position 2"] == role
        and row["Flexible Position 2 Supported"]
    )

    flexible_options = []

    if flexible_1_matches:
        flexible_options.append({
            "Probability":
                row[
                    "Flexible Position 1 Probability"
                ],
            "Source":
                "Flexible Position 1"
        })

    if flexible_2_matches:
        flexible_options.append({
            "Probability":
                row[
                    "Flexible Position 2 Probability"
                ],
            "Source":
                "Flexible Position 2"
        })

    if not flexible_options:
        return {
            "Eligible": False,
            "Position Fit Score": np.nan,
            "Role Source": "Not Eligible"
        }

    best_flexible_option = max(
        flexible_options,
        key=lambda option: option["Probability"]
    )

    return {
        "Eligible": True,
        "Position Fit Score":
            best_flexible_option[
                "Probability"
            ] * 100,
        "Role Source":
            best_flexible_option[
                "Source"
            ]
    }


# Create a long-format dataset.
#
# Every row represents one eligible player-role combination.
#
# A flexible player may therefore appear several times:
#
# once for their official role
#
# and once for each supported flexible role.

role_suitability_rows = []


# Use whichever identifying columns are available.

identity_columns = [
    column
    for column in [
        "ID",
        "Name",
        "Position",
        "Team",
        "League",
        "Nation",
        "OVR",
        "Preferred foot",
        "Alternative positions",
        "Age",
        "Height",
        "Weight"
    ]
    if column in df.columns
]


for player_index, row in df.iterrows():
    for role in role_attribute_weights:
        position_fit = calculate_position_fit(
            row,
            role
        )

        if not position_fit["Eligible"]:
            continue

        role_attribute_score = (
            calculate_role_attribute_score(
                row,
                role
            )
        )

        if pd.isna(role_attribute_score):
            continue

        player_role_score = (
            OVERALL_WEIGHT
            * row["OVR"]
            + ATTRIBUTE_WEIGHT
            * role_attribute_score
            + POSITION_FIT_WEIGHT
            * position_fit[
                "Position Fit Score"
            ]
        )

        result_row = {
            column: row[column]
            for column in identity_columns
        }

        result_row.update({
            "Original Role Group":
                row[
                    "Original Role Group"
                ],
            "Evaluated Role":
                role,
            "Role Source":
                position_fit[
                    "Role Source"
                ],
            "Position Fit Score":
                position_fit[
                    "Position Fit Score"
                ],
            "Role Attribute Score":
                role_attribute_score,
            "Player Role Suitability":
                player_role_score
        })

        role_suitability_rows.append(
            result_row
        )


role_suitability = pd.DataFrame(
    role_suitability_rows
)


# Sort by role and suitability score.

role_suitability = role_suitability.sort_values(
    [
        "Evaluated Role",
        "Player Role Suitability"
    ],
    ascending=[
        True,
        False
    ]
).reset_index(drop=True)


print(
    "\nNumber of eligible player-role combinations:",
    len(role_suitability)
)


print(
    "\nEligible combinations by role:"
)

print(
    role_suitability[
        "Evaluated Role"
    ].value_counts()
)


print(
    "\nEligible combinations by role source:"
)

print(
    role_suitability[
        "Role Source"
    ].value_counts()
)


# Print the top 10 candidates for every role.

top_role_tables = []

for role in role_attribute_weights:
    role_results = role_suitability[
        role_suitability[
            "Evaluated Role"
        ]
        == role
    ].copy()

    top_role_results = role_results.head(25)

    top_role_tables.append(
        top_role_results
    )

    print("\n" + role.upper())
    print("-" * len(role))
    print(
        top_role_results[
            [
                column
                for column in [
                    "Name",
                    "Position",
                    "Team",
                    "OVR",
                    "Role Source",
                    "Position Fit Score",
                    "Role Attribute Score",
                    "Player Role Suitability"
                ]
                if column
                in top_role_results.columns
            ]
        ]
        .head(10)
        .round(2)
    )

    safe_role_name = (
        role.lower()
        .replace(
            " ",
            "_"
        )
    )

    top_role_results.to_csv(
        RESULTS
        / f"top_{safe_role_name}_candidates.csv",
        index=False
    )


# Save a smaller file containing only the top 25 candidates for every
# possible role.

top_role_candidates = pd.concat(
    top_role_tables,
    ignore_index=True
)

top_role_candidates.to_csv(
    RESULTS
    / "top_candidates_by_role.csv",
    index=False
)


# Create a wide-format suitability matrix.
#
# Each row represents one player.
#
# Each column represents one possible role.
#
# Missing values mean that the player is not currently eligible for
# that role.

player_identifier = (
    "ID"
    if "ID" in role_suitability.columns
    else "Name"
)


suitability_matrix = role_suitability.pivot_table(
    index=player_identifier,
    columns="Evaluated Role",
    values="Player Role Suitability",
    aggfunc="max"
)


suitability_matrix = suitability_matrix.reset_index()


# Rename the role columns so they are easier to identify.

suitability_matrix = suitability_matrix.rename(
    columns={
        role:
            f"Suitability: {role}"
        for role in role_attribute_weights
    }
)


suitability_matrix.to_csv(
    RESULTS
    / "player_role_suitability_matrix.csv",
    index=False
)


# Save the complete long-format role-suitability dataset.
#
# This is the main file that the lineup optimizer will use.

role_suitability.to_csv(
    OUTPUT_FILE,
    index=False
)


role_suitability.to_csv(
    RESULTS
    / "player_role_suitability.csv",
    index=False
)


# Save the role-specific attribute definitions.
#
# This makes the scoring choices transparent and easier to explain in
# the Stage 4.1 notes.

role_definition_rows = []

for role, attribute_weights in role_attribute_weights.items():
    for attribute, weight in attribute_weights.items():
        role_definition_rows.append({
            "Role": role,
            "Attribute": attribute,
            "Attribute Weight": weight
        })


role_definitions = pd.DataFrame(
    role_definition_rows
)


role_definitions.to_csv(
    RESULTS
    / "role_attribute_definitions.csv",
    index=False
)


# Save a summary of the three formations.
#
# The exact midfield arrangement of a 4-3-3 can vary.
#
# Stage 4.2 may evaluate more than one three-midfielder combination
# rather than assuming that every 4-3-3 must use exactly one DM and
# two CMs.

formation_summary = pd.DataFrame([
    {
        "Formation": "4-3-3",
        "Goalkeepers": 1,
        "Defenders": 4,
        "Midfielders": 3,
        "Forwards": 3,
        "Possible Role Structure":
            "2 Full Backs, 2 Center Backs, "
            "3 central midfield roles, 2 Wingers, 1 Striker"
    },
    {
        "Formation": "4-2-3-1",
        "Goalkeepers": 1,
        "Defenders": 4,
        "Midfielders": 5,
        "Forwards": 1,
        "Possible Role Structure":
            "2 Full Backs, 2 Center Backs, "
            "2 Defensive Midfielders, 1 Attacking Midfielder, "
            "2 wide attackers, 1 Striker"
    },
    {
        "Formation": "4-4-2",
        "Goalkeepers": 1,
        "Defenders": 4,
        "Midfielders": 4,
        "Forwards": 2,
        "Possible Role Structure":
            "2 Full Backs, 2 Center Backs, "
            "2 Central Midfielders, 2 Wide Midfielders, "
            "2 Strikers"
    }
])


formation_summary.to_csv(
    RESULTS
    / "formation_summary.csv",
    index=False
)


print("\nPlayer-role suitability dataset saved to:")

print(OUTPUT_FILE)

print("\nStage 4.1 results saved to:")

print(RESULTS)

print("\nStage 4.1 player-role suitability complete!")