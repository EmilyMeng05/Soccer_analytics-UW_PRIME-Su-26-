from pathlib import Path
import ast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# STAGE 3.6C — VALIDATED MULTI-ROLE PLAYER REPRESENTATION
# ============================================================
#
# PURPOSE
# -------
# Stage 3.6A showed that official EAFC alternative positions are
# available for most players.
#
# Stage 3.6B showed that primary position alone is a very strong
# predictor of official alternative roles. The 34-feature model
# did NOT beat that baseline overall, although it added useful
# information for some heterogeneous roles such as Central
# Midfielder and Striker.
#
# Therefore, Stage 3.6C does NOT claim that the model discovers
# a player's "true" flexible positions.
#
# Instead, it keeps two concepts separate:
#
#   1. OFFICIAL ELIGIBILITY
#      Which functional roles does EAFC explicitly list for the
#      player through primary + alternative positions?
#
#   2. LEARNED ROLE SUITABILITY
#      How strongly does the player's 34-attribute profile
#      resemble each of the eight functional roles?
#
# This creates the player representation that Stage 4 can use:
#
#   Official eligibility -> where the player is allowed to play
#   Learned probability  -> how well the profile fits that role
#
# IMPORTANT
# ---------
# No arbitrary probability threshold is used here.
#
# The model probabilities are retained as continuous role-
# suitability signals. They should not be interpreted as proof
# of versatility by themselves.
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

OUTPUT_FILE = (
    DATA
    / "processed"
    / "eafc26_players_multirole_representation.csv"
)

RESULTS_DIR = (
    RESULTS
    / "stage_3_6"
    / "C_multirole_representation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42


# ============================================================
# POSITION NORMALIZATION
# ============================================================
#
# Primary Position uses full names.
# Alternative positions uses EAFC abbreviations.
# ============================================================

POSITION_ABBREVIATION_TO_FULL = {
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
# STAGE 3.5C FEATURES
# ============================================================

CORE_FEATURES = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY",
]

DETAILED_FEATURES = [
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
    "Aggression",
]

FEATURES = CORE_FEATURES + DETAILED_FEATURES


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 78)
print("STAGE 3.6C — VALIDATED MULTI-ROLE PLAYER REPRESENTATION")
print("=" * 78)

print(f"\nLoaded dataset: {df.shape}")


required_columns = (
    FEATURES
    + [
        "Name",
        "Position",
        "Alternative positions",
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# PARSE OFFICIAL ALTERNATIVE POSITIONS
# ============================================================

def parse_alternative_positions(value):

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text or text.lower() in {
        "nan",
        "none",
        "null",
        "n/a",
    }:
        return []

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            raw_positions = [
                str(position).strip()
                for position in parsed
                if str(position).strip()
            ]
        else:
            raw_positions = []

    except (ValueError, SyntaxError):
        raw_positions = []

    normalized_positions = []

    for position in raw_positions:

        full_position = (
            POSITION_ABBREVIATION_TO_FULL
            .get(position, position)
        )

        if full_position not in POSITION_TO_FUNCTIONAL:
            raise ValueError(
                "Unrecognized alternative position: "
                f"{position}"
            )

        if full_position not in normalized_positions:
            normalized_positions.append(
                full_position
            )

    return normalized_positions


df[
    "Official Alternative Position List"
] = (
    df[
        "Alternative positions"
    ]
    .apply(parse_alternative_positions)
)


# ============================================================
# PRIMARY FUNCTIONAL ROLE
# ============================================================

df[
    "Primary Functional Role"
] = (
    df[
        "Position"
    ]
    .map(POSITION_TO_FUNCTIONAL)
)


unmapped_primary = sorted(
    df.loc[
        df[
            "Primary Functional Role"
        ].isna(),
        "Position",
    ]
    .dropna()
    .unique()
)


if unmapped_primary:
    raise ValueError(
        "Unmapped primary positions:\n"
        f"{unmapped_primary}"
    )


# ============================================================
# OFFICIAL ALTERNATIVE FUNCTIONAL ROLES
# ============================================================

def get_alternative_functional_roles(row):

    primary_role = row[
        "Primary Functional Role"
    ]

    functional_roles = []

    for position in row[
        "Official Alternative Position List"
    ]:

        role = POSITION_TO_FUNCTIONAL[
            position
        ]

        # Do not count a left/right version of the same
        # functional role as a new functional role.
        if role == primary_role:
            continue

        if role not in functional_roles:
            functional_roles.append(
                role
            )

    return functional_roles


df[
    "Official Alternative Functional Roles"
] = df.apply(
    get_alternative_functional_roles,
    axis=1,
)


# ============================================================
# COMPLETE OFFICIAL FUNCTIONAL ELIGIBILITY
# ============================================================
#
# Eligibility includes:
#
#   primary functional role
#   +
#   any different official alternative functional roles
#
# ============================================================

def build_eligible_role_list(row):

    roles = [
        row[
            "Primary Functional Role"
        ]
    ]

    for role in row[
        "Official Alternative Functional Roles"
    ]:

        if role not in roles:
            roles.append(role)

    return roles


df[
    "Official Eligible Functional Roles"
] = df.apply(
    build_eligible_role_list,
    axis=1,
)


df[
    "Number Official Eligible Functional Roles"
] = (
    df[
        "Official Eligible Functional Roles"
    ]
    .apply(len)
)


# ============================================================
# PREPARE 34-FEATURE MODEL
# ============================================================

for feature in FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce",
    )


missing_feature_rows = (
    df[
        FEATURES
    ]
    .isna()
    .any(axis=1)
)


if missing_feature_rows.any():

    raise ValueError(
        f"{missing_feature_rows.sum()} players contain "
        "missing Stage 3.5C model features."
    )


X = df[
    FEATURES
].copy()

y_text = df[
    "Primary Functional Role"
].copy()


label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    y_text
)

class_names = (
    label_encoder
    .classes_
)


print("\nFunctional roles learned by model:")

for role in class_names:
    print(f"  {role}")


# ============================================================
# TRAIN FINAL STAGE 3.5C MODEL
# ============================================================
#
# Stage 3.6B already performed out-of-fold validation.
#
# Here we deliberately retrain on the complete dataset because
# these probabilities are for downstream Stage 4 representation,
# not for estimating validation performance.
# ============================================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


model = LogisticRegression(
    max_iter=5000,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)

model.fit(
    X_scaled,
    y,
)


probabilities = model.predict_proba(
    X_scaled
)


# ============================================================
# SAVE COMPLETE LEARNED ROLE SUITABILITY PROFILE
# ============================================================

for class_index, role in enumerate(
    class_names
):

    df[
        f"Role Suitability Probability - {role}"
    ] = probabilities[
        :,
        class_index,
    ]


# ============================================================
# RANK ALL FUNCTIONAL ROLES BY LEARNED SUITABILITY
# ============================================================
#
# These rankings are descriptive model outputs.
#
# They do NOT automatically define eligibility.
# ============================================================

learned_rank_1 = []
learned_rank_1_probability = []

learned_rank_2 = []
learned_rank_2_probability = []

learned_rank_3 = []
learned_rank_3_probability = []


for row_index, row in df.iterrows():

    role_scores = []

    for role in class_names:

        probability = row[
            f"Role Suitability Probability - {role}"
        ]

        role_scores.append(
            (
                role,
                probability,
            )
        )

    role_scores.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    learned_rank_1.append(
        role_scores[0][0]
    )

    learned_rank_1_probability.append(
        role_scores[0][1]
    )

    learned_rank_2.append(
        role_scores[1][0]
    )

    learned_rank_2_probability.append(
        role_scores[1][1]
    )

    learned_rank_3.append(
        role_scores[2][0]
    )

    learned_rank_3_probability.append(
        role_scores[2][1]
    )


df[
    "Learned Role Rank 1"
] = learned_rank_1

df[
    "Learned Role Rank 1 Probability"
] = learned_rank_1_probability

df[
    "Learned Role Rank 2"
] = learned_rank_2

df[
    "Learned Role Rank 2 Probability"
] = learned_rank_2_probability

df[
    "Learned Role Rank 3"
] = learned_rank_3

df[
    "Learned Role Rank 3 Probability"
] = learned_rank_3_probability


# ============================================================
# SUITABILITY WITHIN OFFICIAL ELIGIBLE ROLES
# ============================================================
#
# This is the key Stage 4 bridge.
#
# We first use EAFC primary + alternative positions to determine
# which functional roles are officially eligible.
#
# Then we use the learned model probability to rank ONLY those
# eligible roles.
#
# Example:
#
# Official eligibility:
#   CM, DM, Fullback
#
# Learned probabilities:
#   CM       0.50
#   DM       0.24
#   Fullback 0.07
#
# Stage 4 can therefore distinguish:
#
#   eligibility from suitability.
# ============================================================

best_eligible_role = []
best_eligible_probability = []

second_eligible_role = []
second_eligible_probability = []


for row_index, row in df.iterrows():

    eligible_roles = row[
        "Official Eligible Functional Roles"
    ]

    eligible_scores = []

    for role in eligible_roles:

        probability = row[
            f"Role Suitability Probability - {role}"
        ]

        eligible_scores.append(
            (
                role,
                probability,
            )
        )

    eligible_scores.sort(
        key=lambda item: item[1],
        reverse=True,
    )


    # Best officially eligible functional role.
    best_eligible_role.append(
        eligible_scores[0][0]
    )

    best_eligible_probability.append(
        eligible_scores[0][1]
    )


    # Some players have only one eligible functional role.
    if len(eligible_scores) >= 2:

        second_eligible_role.append(
            eligible_scores[1][0]
        )

        second_eligible_probability.append(
            eligible_scores[1][1]
        )

    else:

        second_eligible_role.append(
            np.nan
        )

        second_eligible_probability.append(
            np.nan
        )


df[
    "Best Eligible Functional Role"
] = best_eligible_role

df[
    "Best Eligible Functional Role Probability"
] = best_eligible_probability

df[
    "Second Eligible Functional Role"
] = second_eligible_role

df[
    "Second Eligible Functional Role Probability"
] = second_eligible_probability


# ============================================================
# PRIMARY-ROLE SUITABILITY
# ============================================================

primary_role_suitability = []


for row_index, row in df.iterrows():

    primary_role = row[
        "Primary Functional Role"
    ]

    probability = row[
        f"Role Suitability Probability - {primary_role}"
    ]

    primary_role_suitability.append(
        probability
    )


df[
    "Primary Functional Role Suitability"
] = primary_role_suitability


# ============================================================
# BEST OFFICIAL ALTERNATIVE ROLE
# ============================================================
#
# This is useful for Stage 4 when a player must move away from
# their primary role.
#
# Unlike the old Stage 3.6, this does NOT invent alternatives
# from model probabilities.
#
# The candidate alternatives come from EAFC eligibility first.
# The model only ranks them.
# ============================================================

best_official_alternative_role = []
best_official_alternative_probability = []

second_official_alternative_role = []
second_official_alternative_probability = []


for row_index, row in df.iterrows():

    alternative_roles = row[
        "Official Alternative Functional Roles"
    ]

    scores = []

    for role in alternative_roles:

        probability = row[
            f"Role Suitability Probability - {role}"
        ]

        scores.append(
            (
                role,
                probability,
            )
        )

    scores.sort(
        key=lambda item: item[1],
        reverse=True,
    )


    if len(scores) >= 1:

        best_official_alternative_role.append(
            scores[0][0]
        )

        best_official_alternative_probability.append(
            scores[0][1]
        )

    else:

        best_official_alternative_role.append(
            np.nan
        )

        best_official_alternative_probability.append(
            np.nan
        )


    if len(scores) >= 2:

        second_official_alternative_role.append(
            scores[1][0]
        )

        second_official_alternative_probability.append(
            scores[1][1]
        )

    else:

        second_official_alternative_role.append(
            np.nan
        )

        second_official_alternative_probability.append(
            np.nan
        )


df[
    "Best Official Alternative Functional Role"
] = best_official_alternative_role

df[
    "Best Official Alternative Role Suitability"
] = best_official_alternative_probability

df[
    "Second Official Alternative Functional Role"
] = second_official_alternative_role

df[
    "Second Official Alternative Role Suitability"
] = second_official_alternative_probability


# ============================================================
# PLAYER-ROLE LONG FORMAT
# ============================================================
#
# This file will be especially useful for Stage 4 optimization.
#
# Each row represents:
#
#   one player
#   x
#   one functional role
#
# and records:
#
#   Is this the player's primary role?
#   Is this an official alternative role?
#   Is the player officially eligible?
#   What is the learned role-suitability probability?
#
# ============================================================

player_role_rows = []


for row_index, row in df.iterrows():

    primary_role = row[
        "Primary Functional Role"
    ]

    alternative_roles = set(
        row[
            "Official Alternative Functional Roles"
        ]
    )

    eligible_roles = set(
        row[
            "Official Eligible Functional Roles"
        ]
    )


    for role in FUNCTIONAL_ROLES:

        player_role_rows.append(
            {
                "ID":
                    row["ID"]
                    if "ID" in df.columns
                    else row_index,

                "Name":
                    row["Name"],

                "OVR":
                    row["OVR"]
                    if "OVR" in df.columns
                    else np.nan,

                "Original Position":
                    row["Position"],

                "Functional Role":
                    role,

                "Is Primary Functional Role":
                    role == primary_role,

                "Is Official Alternative Functional Role":
                    role in alternative_roles,

                "Is Officially Eligible":
                    role in eligible_roles,

                "Role Suitability Probability":
                    row[
                        f"Role Suitability Probability - {role}"
                    ],
            }
        )


player_role_df = pd.DataFrame(
    player_role_rows
)


# ============================================================
# CONVERT LIST COLUMNS TO READABLE CSV STRINGS
# ============================================================

df[
    "Official Alternative Position List"
] = (
    df[
        "Official Alternative Position List"
    ]
    .apply(
        lambda values:
        "|".join(values)
    )
)


df[
    "Official Alternative Functional Roles"
] = (
    df[
        "Official Alternative Functional Roles"
    ]
    .apply(
        lambda values:
        "|".join(values)
    )
)


df[
    "Official Eligible Functional Roles"
] = (
    df[
        "Official Eligible Functional Roles"
    ]
    .apply(
        lambda values:
        "|".join(values)
    )
)


# ============================================================
# EXAMPLE OUTPUT
# ============================================================

example_columns = [
    "Name",
    "Position",
    "Primary Functional Role",
    "Official Alternative Functional Roles",
    "Official Eligible Functional Roles",
    "Primary Functional Role Suitability",
    "Best Official Alternative Functional Role",
    "Best Official Alternative Role Suitability",
    "Learned Role Rank 1",
    "Learned Role Rank 1 Probability",
    "Learned Role Rank 2",
    "Learned Role Rank 2 Probability",
]


print("\n" + "=" * 78)
print("EXAMPLE MULTI-ROLE PLAYER REPRESENTATIONS")
print("=" * 78)


print(
    df[
        example_columns
    ]
    .head(25)
    .to_string(index=False)
)


# ============================================================
# INDIVIDUAL PLAYER INSPECTION
# ============================================================

def show_player(player_name):

    player = df[
        df[
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


    probability_columns = [
        f"Role Suitability Probability - {role}"
        for role in FUNCTIONAL_ROLES
    ]


    print(
        "\n"
        + "-" * 78
    )

    print(
        f"MULTI-ROLE PROFILE: {player_name}"
    )

    print(
        "-" * 78
    )


    print(
        player[
            example_columns
            + probability_columns
        ]
        .to_string(index=False)
    )


for player_name in [
    "Jude Bellingham",
    "Federico Valverde",
    "Achraf Hakimi",
    "Mohamed Salah",
    "Kylian Mbappé",
    "Rodri",
]:

    show_player(
        player_name
    )


# ============================================================
# SUMMARY
# ============================================================

eligibility_distribution = (
    df[
        "Number Official Eligible Functional Roles"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "Number Eligible Functional Roles"
    )
    .reset_index(
        name="Players"
    )
)


print(
    "\nOfficial functional-role eligibility distribution:"
)

print(
    eligibility_distribution
    .to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)


df.to_csv(
    OUTPUT_FILE,
    index=False,
)


player_role_df.to_csv(
    RESULTS_DIR
    / "player_role_eligibility_and_suitability.csv",
    index=False,
)


df[
    example_columns
].to_csv(
    RESULTS_DIR
    / "player_multirole_summary.csv",
    index=False,
)


eligibility_distribution.to_csv(
    RESULTS_DIR
    / "functional_role_eligibility_distribution.csv",
    index=False,
)


print("\n" + "=" * 78)
print("STAGE 3.6C COMPLETE")
print("=" * 78)

print(
    "\nMain player dataset saved to:"
)

print(
    OUTPUT_FILE
)

print(
    "\nStage 4 player-role table saved to:"
)

print(
    RESULTS_DIR
    / "player_role_eligibility_and_suitability.csv"
)

print(
    "\nIMPORTANT:"
)

print(
    "Official eligibility and learned role suitability "
    "remain separate."
)

print(
    "No arbitrary probability threshold was used."
)