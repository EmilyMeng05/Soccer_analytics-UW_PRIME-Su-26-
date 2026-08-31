from pathlib import Path
import ast
import re

import numpy as np
import pandas as pd


# ============================================================
# STAGE 3.6A — OFFICIAL ALTERNATIVE POSITION ANALYSIS
# ============================================================
#
# Goal:
#
# Understand the "Alternative positions" field before using it
# as validation for model-predicted secondary roles.
#
# Questions:
#
# 1. How many players have official alternative positions?
# 2. How many alternatives does each player have?
# 3. Which primary -> alternative transitions are common?
# 4. After mapping exact positions to our eight functional roles,
#    which alternatives represent genuinely different roles?
# 5. How often are alternatives only left/right variants of the
#    player's existing functional role?
#
# Output:
#
# A normalized player-level validation dataset that Stage 3.6B
# can use directly.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA = PROJECT_ROOT / "data"
RESULTS = PROJECT_ROOT / "results"

INPUT_FILE = (
    DATA
    / "processed"
    / "eafc26_outfield_full_features.csv"
)

RESULTS_DIR = (
    RESULTS
    / "stage_3_6"
    / "A_alternative_position_analysis"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NORMALIZED_OUTPUT = (
    RESULTS_DIR
    / "normalized_alternative_positions.csv"
)


# ============================================================
# POSITION MAPPING
# ============================================================

# ============================================================
# POSITION NORMALIZATION
# ============================================================
#
# The primary Position column uses full names:
#
#   Right Winger
#   Central Midfielder
#   Left Back
#
# But the Alternative positions column uses EAFC abbreviations:
#
#   RW
#   CM
#   LB
#
# We therefore normalize abbreviations to the same full-name
# format before mapping to functional roles.
# ============================================================

# ============================================================
# POSITION NORMALIZATION
# ============================================================
#
# The primary Position column uses full names:
#
#   Right Winger
#   Central Midfielder
#   Left Back
#
# But the Alternative positions column uses EAFC abbreviations:
#
#   RW
#   CM
#   LB
#
# We therefore normalize abbreviations to the same full-name
# format before mapping to functional roles.
# ============================================================

POSITION_ABBREVIATION_TO_FULL = {
    "GK": "Goalkeeper",
    "CB": "Center Back",
    "LB": "Left Back",
    "RB": "Right Back",
    "CDM": "Defensive Midfielder",
    "CM": "Central Midfielder",
    "CAM": "Attacking Midfielder",
    "LM": "Left Midfielder",
    "RM": "Right Midfielder",
    "LW": "Left Winger",
    "RW": "Right Winger",
    "ST": "Striker",
}


POSITION_TO_FUNCTIONAL = {
    "Center Back": "Center Back",
    "Left Back": "Fullback",
    "Right Back": "Fullback",
    "Defensive Midfielder": "Defensive Midfielder",
    "Central Midfielder": "Central Midfielder",
    "Attacking Midfielder": "Attacking Midfielder",
    "Left Midfielder": "Wide Midfielder",
    "Right Midfielder": "Wide Midfielder",
    "Left Winger": "Winger",
    "Right Winger": "Winger",
    "Striker": "Striker",
}

FUNCTIONAL_ROLES = [
    "Attacking Midfielder",
    "Center Back",
    "Central Midfielder",
    "Defensive Midfielder",
    "Fullback",
    "Striker",
    "Wide Midfielder",
    "Winger",
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("STAGE 3.6A — ALTERNATIVE POSITION ANALYSIS")
print("=" * 70)

print(f"\nLoaded dataset: {df.shape}")


required_columns = [
    "Name",
    "Position",
    "Alternative positions",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# INSPECT RAW FORMAT
# ============================================================

print("\nExample raw Alternative positions values:")

print(
    df[
        [
            "Name",
            "Position",
            "Alternative positions",
        ]
    ]
    .head(20)
    .to_string(index=False)
)


# ============================================================
# POSITION PARSER
# ============================================================
#
# This parser intentionally handles several possible CSV formats:
#
# "Left Back, Right Back"
#
# "Left Back; Right Back"
#
# "['Left Back', 'Right Back']"
#
# "[Left Back, Right Back]"
#
# Missing values are returned as an empty list.
#
# ============================================================

def parse_alternative_positions(value):

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    if text.lower() in {
        "nan",
        "none",
        "null",
        "n/a",
    }:
        return []

    # --------------------------------------------------------
    # Try Python-list format first.
    #
    # Example:
    # ['RW', 'CAM']
    # --------------------------------------------------------

    if (
        text.startswith("[")
        and text.endswith("]")
    ):

        try:

            parsed = ast.literal_eval(
                text
            )

            if isinstance(parsed, list):

                raw_positions = [
                    str(position).strip()
                    for position in parsed
                    if str(position).strip()
                ]

            else:

                raw_positions = []

        except (
            ValueError,
            SyntaxError,
        ):

            raw_positions = []

    else:

        # ----------------------------------------------------
        # Fallback for comma / semicolon separated text.
        # ----------------------------------------------------

        raw_positions = re.split(
            r"\s*[,;/|]\s*",
            text,
        )

        raw_positions = [
            position
            .strip()
            .strip("'")
            .strip('"')
            .strip()

            for position in raw_positions

            if position.strip()
        ]


    # --------------------------------------------------------
    # Normalize EAFC abbreviations to full position names.
    # --------------------------------------------------------

    normalized_positions = []

    for position in raw_positions:

        full_position = (
            POSITION_ABBREVIATION_TO_FULL
            .get(
                position,
                position,
            )
        )

        if full_position:
            normalized_positions.append(
                full_position
            )


    return normalized_positions


# ============================================================
# PARSE ALTERNATIVES
# ============================================================

df[
    "Alternative Position List"
] = (
    df[
        "Alternative positions"
    ]
    .apply(parse_alternative_positions)
)


# ============================================================
# CHECK ALL POSITION VALUES
# ============================================================

observed_primary_positions = set(
    df["Position"]
    .dropna()
    .unique()
)

observed_alternative_positions = set()

for positions in df[
    "Alternative Position List"
]:
    observed_alternative_positions.update(
        positions
    )


unknown_primary = sorted(
    observed_primary_positions
    - set(POSITION_TO_FUNCTIONAL)
)

unknown_alternatives = sorted(
    observed_alternative_positions
    - set(POSITION_TO_FUNCTIONAL)
)


print("\nUnknown primary positions:")
print(unknown_primary)

print("\nUnknown alternative positions:")
print(unknown_alternatives)


if unknown_primary:
    raise ValueError(
        "Some primary positions are not in POSITION_TO_FUNCTIONAL."
    )


if unknown_alternatives:
    raise ValueError(
        "\nSome alternative positions were not recognized.\n"
        "Inspect the printed values before continuing.\n\n"
        f"{unknown_alternatives}"
    )


# ============================================================
# PRIMARY FUNCTIONAL ROLE
# ============================================================

df[
    "Primary Functional Role"
] = (
    df["Position"]
    .map(POSITION_TO_FUNCTIONAL)
)


# ============================================================
# MAP ALTERNATIVES TO FUNCTIONAL ROLES
# ============================================================

def map_to_functional_roles(position_list):

    roles = []

    for position in position_list:

        role = POSITION_TO_FUNCTIONAL[
            position
        ]

        if role not in roles:
            roles.append(role)

    return roles


df[
    "Alternative Functional Role List Raw"
] = (
    df[
        "Alternative Position List"
    ]
    .apply(map_to_functional_roles)
)


# ============================================================
# REMOVE PRIMARY FUNCTIONAL ROLE
# ============================================================
#
# Example:
#
# Primary:
# Left Back -> Fullback
#
# Alternative:
# Right Back -> Fullback
#
# This is an official alternative exact position, but it does
# NOT represent a new functional role.
#
# We therefore distinguish:
#
# Exact-position flexibility
# vs.
# Functional-role flexibility
#
# ============================================================

def remove_primary_role(row):

    primary_role = row[
        "Primary Functional Role"
    ]

    alternative_roles = row[
        "Alternative Functional Role List Raw"
    ]

    return [
        role
        for role in alternative_roles
        if role != primary_role
    ]


df[
    "Alternative Functional Role List"
] = df.apply(
    remove_primary_role,
    axis=1,
)


# ============================================================
# SUMMARY COUNTS
# ============================================================

df[
    "Number of Exact Alternatives"
] = (
    df[
        "Alternative Position List"
    ]
    .apply(len)
)

df[
    "Number of Functional Alternatives"
] = (
    df[
        "Alternative Functional Role List"
    ]
    .apply(len)
)

df[
    "Has Exact Alternative"
] = (
    df[
        "Number of Exact Alternatives"
    ]
    > 0
)

df[
    "Has Functional Alternative"
] = (
    df[
        "Number of Functional Alternatives"
    ]
    > 0
)

df[
    "Only Same Functional Role Alternatives"
] = (
    df[
        "Has Exact Alternative"
    ]
    & ~df[
        "Has Functional Alternative"
    ]
)


# ============================================================
# PRINT SUMMARY
# ============================================================

n_players = len(df)

n_exact = int(
    df[
        "Has Exact Alternative"
    ].sum()
)

n_functional = int(
    df[
        "Has Functional Alternative"
    ].sum()
)

n_same_role_only = int(
    df[
        "Only Same Functional Role Alternatives"
    ].sum()
)


print("\n" + "=" * 70)
print("ALTERNATIVE POSITION SUMMARY")
print("=" * 70)

print(
    f"\nTotal players: {n_players}"
)

print(
    f"Players with >=1 exact alternative: "
    f"{n_exact} ({n_exact / n_players:.2%})"
)

print(
    f"Players with >=1 different functional alternative: "
    f"{n_functional} ({n_functional / n_players:.2%})"
)

print(
    f"Players whose alternatives stay within their "
    f"primary functional role: "
    f"{n_same_role_only} ({n_same_role_only / n_players:.2%})"
)


# ============================================================
# ALTERNATIVE COUNT DISTRIBUTIONS
# ============================================================

exact_count_distribution = (
    df[
        "Number of Exact Alternatives"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "Number of Exact Alternatives"
    )
    .reset_index(
        name="Players"
    )
)

functional_count_distribution = (
    df[
        "Number of Functional Alternatives"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "Number of Functional Alternatives"
    )
    .reset_index(
        name="Players"
    )
)


print("\nExact alternative count distribution:")

print(
    exact_count_distribution
    .to_string(index=False)
)


print("\nFunctional alternative count distribution:")

print(
    functional_count_distribution
    .to_string(index=False)
)


# ============================================================
# EXACT TRANSITIONS
# ============================================================

exact_transition_rows = []

for _, row in df.iterrows():

    for alternative_position in row[
        "Alternative Position List"
    ]:

        exact_transition_rows.append(
            {
                "Primary Position":
                    row["Position"],

                "Alternative Position":
                    alternative_position,
            }
        )


exact_transitions = pd.DataFrame(
    exact_transition_rows
)


if len(exact_transitions) > 0:

    exact_transition_counts = (
        exact_transitions
        .value_counts(
            [
                "Primary Position",
                "Alternative Position",
            ]
        )
        .reset_index(
            name="Count"
        )
        .sort_values(
            "Count",
            ascending=False,
        )
    )

else:

    exact_transition_counts = pd.DataFrame(
        columns=[
            "Primary Position",
            "Alternative Position",
            "Count",
        ]
    )


# ============================================================
# FUNCTIONAL TRANSITIONS
# ============================================================

functional_transition_rows = []

for _, row in df.iterrows():

    for alternative_role in row[
        "Alternative Functional Role List"
    ]:

        functional_transition_rows.append(
            {
                "Primary Functional Role":
                    row[
                        "Primary Functional Role"
                    ],

                "Alternative Functional Role":
                    alternative_role,
            }
        )


functional_transitions = pd.DataFrame(
    functional_transition_rows
)


if len(functional_transitions) > 0:

    functional_transition_counts = (
        functional_transitions
        .value_counts(
            [
                "Primary Functional Role",
                "Alternative Functional Role",
            ]
        )
        .reset_index(
            name="Count"
        )
        .sort_values(
            "Count",
            ascending=False,
        )
    )

else:

    functional_transition_counts = pd.DataFrame(
        columns=[
            "Primary Functional Role",
            "Alternative Functional Role",
            "Count",
        ]
    )


print("\nMost common functional transitions:")

print(
    functional_transition_counts
    .head(30)
    .to_string(index=False)
)


# ============================================================
# SAVE NORMALIZED PLAYER DATA
# ============================================================
#
# Lists are converted to "|" separated strings so the CSV is
# easy to read and Stage 3.6B can parse them reliably.
#
# ============================================================

output_df = df.copy()

list_columns = [
    "Alternative Position List",
    "Alternative Functional Role List Raw",
    "Alternative Functional Role List",
]

for column in list_columns:

    output_df[column] = (
        output_df[column]
        .apply(
            lambda values:
            "|".join(values)
        )
    )


output_df.to_csv(
    NORMALIZED_OUTPUT,
    index=False,
)


exact_count_distribution.to_csv(
    RESULTS_DIR
    / "exact_alternative_count_distribution.csv",
    index=False,
)

functional_count_distribution.to_csv(
    RESULTS_DIR
    / "functional_alternative_count_distribution.csv",
    index=False,
)

exact_transition_counts.to_csv(
    RESULTS_DIR
    / "exact_position_transition_counts.csv",
    index=False,
)

functional_transition_counts.to_csv(
    RESULTS_DIR
    / "functional_role_transition_counts.csv",
    index=False,
)


summary_df = pd.DataFrame(
    [
        {
            "Total Players":
                n_players,

            "Players With Exact Alternative":
                n_exact,

            "Exact Alternative Rate":
                n_exact / n_players,

            "Players With Functional Alternative":
                n_functional,

            "Functional Alternative Rate":
                n_functional / n_players,

            "Same Functional Role Only":
                n_same_role_only,

            "Same Functional Role Only Rate":
                n_same_role_only / n_players,
        }
    ]
)


summary_df.to_csv(
    RESULTS_DIR
    / "alternative_position_summary.csv",
    index=False,
)


print("\nResults saved to:")
print(RESULTS_DIR)

print(
    "\nStage 3.6A alternative-position "
    "analysis complete!"
)