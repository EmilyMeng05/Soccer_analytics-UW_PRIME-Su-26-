import ast

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from stage4_config import (
    FORMATIONS,
    GOALKEEPER_FILE,
    MULTIROLE_FILE,
    PROJECT_ROOT,
)


# ============================================================
# STAGE 4.2 — ROLE-ADJUSTED QUALITY LINEUPS
# ============================================================
#
# MOTIVATION
# ----------
#
# The first Stage 4.2 experiment used an additive score:
#
#     (1 - w) * Quality + w * RoleFit
#
# That experiment revealed an important problem:
#
# A low-OVR player with a very "pure" role profile can receive a
# very high Stage 3.6C role-suitability probability and therefore
# outrank a much stronger player.
#
# But Stage 3.6C probabilities answer:
#
#     "How strongly does this player's attribute profile resemble
#      this functional role?"
#
# They do NOT directly answer:
#
#     "How good is this player at this role?"
#
# Therefore, this revised Stage 4.2 treats learned role suitability
# as a bounded compatibility adjustment to player quality rather
# than as a second interchangeable quality score.
#
#
# ROLE-ADJUSTED QUALITY
# ---------------------
#
# Let:
#
#     Q_i  = OVR_i / 100
#     S_ir = learned Stage 3.6C role-suitability probability
#
# Then:
#
#     Score_ir =
#         Q_i * [1 - lambda * (1 - S_ir)]
#
# Equivalent form:
#
#     Score_ir =
#         Q_i * [(1 - lambda) + lambda * S_ir]
#
#
# Interpretation:
#
# lambda = 0
#     Pure OVR. Role fit has no influence.
#
# lambda > 0
#     A player loses some value when their learned profile fits the
#     required role poorly.
#
# IMPORTANT:
#
# High role suitability can preserve a strong player's quality,
# but cannot create elite quality from a weak OVR.
#
#
# SENSITIVITY ANALYSIS
# --------------------
#
# We do NOT claim that one lambda is the true football value.
#
# We test:
#
#     0.10
#     0.25
#     0.40
#
# and use lambda = 0.25 as the reference setting for Stage 4.3.
#
#
# ELIGIBILITY
# -----------
#
# Eligibility still comes only from official EAFC information:
#
#     primary Position
#     +
#     official Alternative positions
#
# Learned suitability never creates eligibility.
#
#
# WING-BACK NOTE
# --------------
#
# Stage 3 never trained a separate Wingback class.
#
# For LWB / RWB slots, this script uses a conservative proxy:
#
#     max(Fullback suitability, Wide Midfielder suitability)
#
# rather than the simple mean used in the earlier experiment.
#
# This avoids artificially cutting a strong fullback's fit roughly
# in half simply because they do not also resemble a wide midfielder.
#
# This remains an explicit proxy and should be discussed as a
# limitation / sensitivity choice.
# ============================================================


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage_4_2_role_adjusted_quality"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ROLE_ADJUSTMENT_STRENGTHS = [
    0.10,
    0.25,
    0.40,
]

REFERENCE_LAMBDA = 0.25


# ============================================================
# DATA PARSING
# ============================================================

def parse_pipe_or_list(value):
    """Parse Stage 3.6C exact-position lists saved in CSV form."""

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    if "|" in text:
        return [
            item.strip()
            for item in text.split("|")
            if item.strip()
        ]

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    except (ValueError, SyntaxError):
        pass

    return [text]


def load_players():
    """Load the Stage 3.6C multi-role representation."""

    if not MULTIROLE_FILE.exists():
        raise FileNotFoundError(
            "Could not find Stage 3.6C output:\n"
            f"{MULTIROLE_FILE}\n\n"
            "Run stage3_6C_multirole_representation.py first."
        )

    df = pd.read_csv(
        MULTIROLE_FILE
    )

    required = {
        "Name",
        "OVR",
        "Position",
        "Official Alternative Position List",
    }

    required_roles = sorted(
        {
            role
            for formation_slots in FORMATIONS.values()
            for slot_definition in formation_slots
            for role in slot_definition["suitability_roles"]
        }
    )

    probability_columns = [
        f"Role Suitability Probability - {role}"
        for role in required_roles
    ]

    required.update(
        probability_columns
    )

    missing = sorted(
        required.difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Stage 3.6C output is missing required columns:\n"
            f"{missing}"
        )

    df["OVR"] = pd.to_numeric(
        df["OVR"],
        errors="coerce",
    )

    for column in probability_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[
        [
            "OVR",
            *probability_columns,
        ]
    ].isna().any().any():
        raise ValueError(
            "Missing OVR or role-suitability values were found."
        )

    df[
        "Official Alternative Position List Parsed"
    ] = df[
        "Official Alternative Position List"
    ].apply(
        parse_pipe_or_list
    )

    df[
        "Exact Eligible Positions"
    ] = df.apply(
        lambda row:
        set(
            [row["Position"]]
            + row[
                "Official Alternative Position List Parsed"
            ]
        ),
        axis=1,
    )

    if "ID" in df.columns:
        df["Player Key"] = (
            df["ID"]
            .astype(str)
        )
    else:
        df["Player Key"] = (
            df["Name"]
            .astype(str)
        )

    return df


def load_goalkeepers():
    """Load Stage 3.7 goalkeeper data."""

    if not GOALKEEPER_FILE.exists():
        raise FileNotFoundError(
            "Could not find Stage 3.7 goalkeeper output:\n"
            f"{GOALKEEPER_FILE}\n\n"
            "Run stage3_7_goalkeeper_analysis.py first."
        )

    df = pd.read_csv(
        GOALKEEPER_FILE
    )

    required = {
        "Name",
        "OVR",
        "GoalkeeperSkill",
    }

    missing = sorted(
        required.difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Stage 3.7 output is missing required columns:\n"
            f"{missing}"
        )

    for column in [
        "OVR",
        "GoalkeeperSkill",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df.dropna(
        subset=[
            "Name",
            "OVR",
            "GoalkeeperSkill",
        ]
    ).copy()


# ============================================================
# ROLE-FIT CALCULATION
# ============================================================

def calculate_slot_role_fit(
    row,
    slot_definition,
):
    """
    Return the Stage 3.6C learned suitability signal for one slot.

    Normal slots:
        use the one matching functional-role probability.

    Wing-back-like slots:
        use max(Fullback, Wide Midfielder), because Stage 3 did not
        contain a dedicated Wingback class.
    """

    role_scores = []

    for role in slot_definition[
        "suitability_roles"
    ]:

        column = (
            "Role Suitability Probability - "
            f"{role}"
        )

        role_scores.append(
            float(
                row[column]
            )
        )

    if len(role_scores) == 1:
        return role_scores[0]

    return max(
        role_scores
    )


# ============================================================
# SCORING
# ============================================================

def role_adjusted_quality(
    ovr,
    role_fit,
    adjustment_strength,
):
    """
    Apply a bounded role-fit penalty to OVR-based quality.

    Score =
        Q * [1 - lambda * (1 - S)]

    where:
        Q = OVR / 100
        S = learned role suitability
    """

    quality = (
        float(ovr)
        / 100.0
    )

    score = (
        quality
        * (
            1.0
            - adjustment_strength
            * (
                1.0
                - role_fit
            )
        )
    )

    return quality, score


# ============================================================
# LINEUP OPTIMIZATION
# ============================================================

def optimize_lineup(
    players,
    formation_name,
    formation_slots,
    adjustment_strength,
):
    """Find the best unique-player assignment for one formation."""

    player_keys = (
        players[
            "Player Key"
        ]
        .drop_duplicates()
        .tolist()
    )

    key_to_column = {
        key: index
        for index, key
        in enumerate(
            player_keys
        )
    }

    score_matrix = np.full(
        (
            len(formation_slots),
            len(player_keys),
        ),
        -1_000_000.0,
    )

    row_lookup = {}
    fit_lookup = {}
    quality_lookup = {}

    for slot_index, slot_definition in enumerate(
        formation_slots
    ):

        accepted_positions = (
            slot_definition[
                "accepted_positions"
            ]
        )

        for row_index, row in players.iterrows():

            if not (
                accepted_positions
                & row[
                    "Exact Eligible Positions"
                ]
            ):
                continue

            player_key = row[
                "Player Key"
            ]

            player_column = (
                key_to_column[
                    player_key
                ]
            )

            role_fit = (
                calculate_slot_role_fit(
                    row,
                    slot_definition,
                )
            )

            quality, selection_score = (
                role_adjusted_quality(
                    row["OVR"],
                    role_fit,
                    adjustment_strength,
                )
            )

            if selection_score > score_matrix[
                slot_index,
                player_column,
            ]:

                score_matrix[
                    slot_index,
                    player_column,
                ] = selection_score

                row_lookup[
                    (
                        slot_index,
                        player_column,
                    )
                ] = row_index

                fit_lookup[
                    (
                        slot_index,
                        player_column,
                    )
                ] = role_fit

                quality_lookup[
                    (
                        slot_index,
                        player_column,
                    )
                ] = quality

    slot_indices, player_indices = (
        linear_sum_assignment(
            score_matrix,
            maximize=True,
        )
    )

    selected_rows = []

    for slot_index, player_index in zip(
        slot_indices,
        player_indices,
    ):

        if score_matrix[
            slot_index,
            player_index,
        ] < 0:
            raise ValueError(
                "No valid assignment exists for "
                f"{formation_name}, slot "
                f"{formation_slots[slot_index]['slot']}."
            )

        source_row = players.loc[
            row_lookup[
                (
                    slot_index,
                    player_index,
                )
            ]
        ]

        slot_definition = (
            formation_slots[
                slot_index
            ]
        )

        role_fit = (
            fit_lookup[
                (
                    slot_index,
                    player_index,
                )
            ]
        )

        quality = (
            quality_lookup[
                (
                    slot_index,
                    player_index,
                )
            ]
        )

        selected_rows.append(
            {
                "Method":
                    "Role-Adjusted Quality",

                "Role Adjustment Strength":
                    adjustment_strength,

                "Formation":
                    formation_name,

                "Formation Slot":
                    slot_definition[
                        "slot"
                    ],

                "Name":
                    source_row[
                        "Name"
                    ],

                "Player Key":
                    source_row[
                        "Player Key"
                    ],

                "Primary Position":
                    source_row[
                        "Position"
                    ],

                "OVR":
                    float(
                        source_row[
                            "OVR"
                        ]
                    ),

                "Quality Score":
                    quality,

                "Role Fit Score":
                    role_fit,

                "Role Fit Penalty":
                    (
                        adjustment_strength
                        * (
                            1.0
                            - role_fit
                        )
                    ),

                "Selection Score":
                    score_matrix[
                        slot_index,
                        player_index,
                    ],

                "Accepted Exact Positions":
                    "|".join(
                        sorted(
                            slot_definition[
                                "accepted_positions"
                            ]
                        )
                    ),

                "Suitability Roles":
                    "|".join(
                        slot_definition[
                            "suitability_roles"
                        ]
                    ),

                "Player Exact Eligible Positions":
                    "|".join(
                        sorted(
                            source_row[
                                "Exact Eligible Positions"
                            ]
                        )
                    ),
            }
        )

    return pd.DataFrame(
        selected_rows
    )


# ============================================================
# GOALKEEPER SELECTION
# ============================================================

def select_goalkeeper(
    goalkeepers,
    formation_name,
    adjustment_strength,
):
    """
    Apply the same bounded quality-adjustment idea to goalkeepers.

    OVR is the base quality measure.
    GoalkeeperSkill / 100 is the style-independent goalkeeper fit proxy.
    """

    candidates = (
        goalkeepers
        .copy()
    )

    candidates[
        "Quality Score"
    ] = (
        candidates[
            "OVR"
        ]
        / 100.0
    )

    candidates[
        "Role Fit Score"
    ] = (
        candidates[
            "GoalkeeperSkill"
        ]
        / 100.0
    )

    candidates[
        "Selection Score"
    ] = (
        candidates[
            "Quality Score"
        ]
        * (
            1.0
            - adjustment_strength
            * (
                1.0
                - candidates[
                    "Role Fit Score"
                ]
            )
        )
    )

    row = (
        candidates
        .sort_values(
            [
                "Selection Score",
                "OVR",
            ],
            ascending=False,
        )
        .iloc[0]
    )

    return pd.DataFrame(
        [
            {
                "Method":
                    "Role-Adjusted Quality",

                "Role Adjustment Strength":
                    adjustment_strength,

                "Formation":
                    formation_name,

                "Formation Slot":
                    "GK",

                "Name":
                    row[
                        "Name"
                    ],

                "Player Key":
                    (
                        str(
                            row["ID"]
                        )
                        if "ID"
                        in row.index
                        else row[
                            "Name"
                        ]
                    ),

                "Primary Position":
                    row.get(
                        "Position",
                        "Goalkeeper",
                    ),

                "OVR":
                    float(
                        row[
                            "OVR"
                        ]
                    ),

                "Quality Score":
                    float(
                        row[
                            "Quality Score"
                        ]
                    ),

                "Role Fit Score":
                    float(
                        row[
                            "Role Fit Score"
                        ]
                    ),

                "Role Fit Penalty":
                    (
                        adjustment_strength
                        * (
                            1.0
                            - float(
                                row[
                                    "Role Fit Score"
                                ]
                            )
                        )
                    ),

                "Selection Score":
                    float(
                        row[
                            "Selection Score"
                        ]
                    ),

                "Accepted Exact Positions":
                    "Goalkeeper",

                "Suitability Roles":
                    "GoalkeeperSkill",

                "Player Exact Eligible Positions":
                    "Goalkeeper",
            }
        ]
    )


# ============================================================
# SUMMARY
# ============================================================

def summarize_lineup(
    lineup,
    formation_name,
    adjustment_strength,
):
    return {
        "Method":
            "Role-Adjusted Quality",

        "Role Adjustment Strength":
            adjustment_strength,

        "Formation":
            formation_name,

        "Average OVR":
            lineup[
                "OVR"
            ].mean(),

        "Total OVR":
            lineup[
                "OVR"
            ].sum(),

        "Average Role Fit":
            lineup[
                "Role Fit Score"
            ].mean(),

        "Average Role Fit Penalty":
            lineup[
                "Role Fit Penalty"
            ].mean(),

        "Average Selection Score":
            lineup[
                "Selection Score"
            ].mean(),
    }


# ============================================================
# RUN
# ============================================================

def main():

    players = load_players()

    goalkeepers = load_goalkeepers()

    all_lineups = []

    summaries = []

    for adjustment_strength in (
        ROLE_ADJUSTMENT_STRENGTHS
    ):

        for formation_name, formation_slots in (
            FORMATIONS.items()
        ):

            outfield = optimize_lineup(
                players,
                formation_name,
                formation_slots,
                adjustment_strength,
            )

            goalkeeper = select_goalkeeper(
                goalkeepers,
                formation_name,
                adjustment_strength,
            )

            lineup = pd.concat(
                [
                    goalkeeper,
                    outfield,
                ],
                ignore_index=True,
            )

            all_lineups.append(
                lineup
            )

            summaries.append(
                summarize_lineup(
                    lineup,
                    formation_name,
                    adjustment_strength,
                )
            )

            print(
                "\n"
                + "=" * 78
            )

            print(
                "ROLE-ADJUSTED QUALITY — "
                f"{formation_name} — "
                f"lambda={adjustment_strength:.2f}"
            )

            print(
                "=" * 78
            )

            print(
                lineup[
                    [
                        "Formation Slot",
                        "Name",
                        "OVR",
                        "Role Fit Score",
                        "Role Fit Penalty",
                        "Selection Score",
                    ]
                ]
                .round(
                    4
                )
                .to_string(
                    index=False
                )
            )

    all_lineups_df = pd.concat(
        all_lineups,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(
        summaries
    )

    all_lineups_df.to_csv(
        RESULTS_DIR
        / "all_role_adjusted_quality_lineups.csv",
        index=False,
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "role_adjusted_quality_summary.csv",
        index=False,
    )

    reference_lineups = (
        all_lineups_df[
            np.isclose(
                all_lineups_df[
                    "Role Adjustment Strength"
                ],
                REFERENCE_LAMBDA,
            )
        ]
        .copy()
    )

    reference_lineups.to_csv(
        RESULTS_DIR
        / (
            "reference_role_adjusted_quality_"
            f"lambda_{REFERENCE_LAMBDA:.2f}.csv"
        ),
        index=False,
    )

    print(
        "\nStage 4.2 complete."
    )

    print(
        "\nReference setting:"
    )

    print(
        f"lambda = {REFERENCE_LAMBDA:.2f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Stage 3.6C role probabilities are being used "
        "as bounded compatibility adjustments, not as direct "
        "measures of player quality."
    )

    print(
        "The lambda grid is a sensitivity analysis, not a claim "
        "that one value is the true football weighting."
    )


if __name__ == "__main__":
    main()
