import ast

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from stage4_config import (
    FORMATIONS,
    GOALKEEPER_FILE,
    MULTIROLE_FILE,
    PROJECT_ROOT,
    REFERENCE_ROLE_FIT_WEIGHT,
    ROLE_FIT_WEIGHTS,
)


# ============================================================
# STAGE 4.2 — QUALITY + LEARNED ROLE FIT
# ============================================================
#
# Stage 4.1 asks:
#
#   What happens if we choose the highest-OVR eligible player
#   for every slot?
#
# Stage 4.2 adds one new piece of information:
#
#   Stage 3.6C learned role suitability.
#
# Eligibility still comes ONLY from official EAFC primary +
# alternative positions.
#
# For one eligible player-slot pair:
#
#   Quality = OVR / 100
#
#   RoleFit = Stage 3.6C suitability probability
#
#   SelectionScore =
#       (1 - w) * Quality
#       + w * RoleFit
#
# Instead of claiming one arbitrary w is correct, this script runs
# several values of w and reports how selections change.
#
# w = 0 would reproduce quality-only scoring, but Stage 4.1 remains
# the clean primary OVR baseline because it uses exact positions
# directly and is easier to explain.
# ============================================================


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage_4_2_role_fit"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATA PREPARATION
# ============================================================

def parse_pipe_or_list(value):

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
        parsed = ast.literal_eval(
            text
        )

        if isinstance(
            parsed,
            list,
        ):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

    except (
        ValueError,
        SyntaxError,
    ):
        pass

    return [
        text
    ]


def load_players():

    if not MULTIROLE_FILE.exists():
        raise FileNotFoundError(
            "Stage 3.6C output was not found:\n"
            f"{MULTIROLE_FILE}"
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

    # Every functional role in Stage 4 must also have the probability
    # generated in Stage 3.6C.
    required_roles = sorted(
        {
            role
            for formation_slots
            in FORMATIONS.values()
            for slot_definition
            in formation_slots
            for role
            in slot_definition[
                "suitability_roles"
            ]
        }
    )

    probability_columns = [
        (
            "Role Suitability Probability - "
            f"{role}"
        )
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
            "Stage 3.6C output is missing columns:\n"
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
            "Missing OVR or role-suitability "
            "values were found."
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
            [
                row["Position"]
            ]
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

    if not GOALKEEPER_FILE.exists():
        raise FileNotFoundError(
            "Stage 3.7 output was not found:\n"
            f"{GOALKEEPER_FILE}"
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
            "Stage 3.7 output is missing columns:\n"
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
# ROLE FIT
# ============================================================

def calculate_slot_role_fit(
    row,
    slot_definition,
):
    """Return Stage 3.6C learned fit for one formation slot."""

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
                row[
                    column
                ]
            )
        )

    # Normal slots have one role.
    #
    # Wing-back-like slots have:
    #
    #   Fullback
    #   Wide Midfielder
    #
    # and currently use their simple mean.
    return float(
        np.mean(
            role_scores
        )
    )


def optimize_role_fit_lineup(
    players,
    formation_name,
    formation_slots,
    role_fit_weight,
):
    """Optimize one lineup under one quality-vs-role-fit weight."""

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

    for slot_index, slot_definition in enumerate(
        formation_slots
    ):

        accepted_positions = slot_definition[
            "accepted_positions"
        ]

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

            player_column = key_to_column[
                player_key
            ]

            quality = (
                float(
                    row["OVR"]
                )
                / 100.0
            )

            role_fit = calculate_slot_role_fit(
                row,
                slot_definition,
            )

            selection_score = (
                (
                    1.0
                    - role_fit_weight
                )
                * quality
                + role_fit_weight
                * role_fit
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
                "No valid role-fit assignment exists "
                f"for {formation_name}, "
                f"slot "
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

        slot_definition = formation_slots[
            slot_index
        ]

        role_fit = fit_lookup[
            (
                slot_index,
                player_index,
            )
        ]

        selected_rows.append(
            {
                "Method":
                    "Quality + Role Fit",

                "Role Fit Weight":
                    role_fit_weight,

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
                    float(
                        source_row[
                            "OVR"
                        ]
                    )
                    / 100.0,

                "Role Fit Score":
                    role_fit,

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
            }
        )

    return pd.DataFrame(
        selected_rows
    )


def select_goalkeeper(
    goalkeepers,
    formation_name,
    role_fit_weight,
):

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
        (
            1.0
            - role_fit_weight
        )
        * candidates[
            "Quality Score"
        ]
        + role_fit_weight
        * candidates[
            "Role Fit Score"
        ]
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
                    "Quality + Role Fit",

                "Role Fit Weight":
                    role_fit_weight,

                "Formation":
                    formation_name,

                "Formation Slot":
                    "GK",

                "Name":
                    row["Name"],

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
                        row["OVR"]
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
            }
        ]
    )


# ============================================================
# RUN
# ============================================================

def main():

    players = load_players()

    goalkeepers = load_goalkeepers()

    all_lineups = []

    summaries = []

    for role_fit_weight in ROLE_FIT_WEIGHTS:

        for formation_name, formation_slots in (
            FORMATIONS.items()
        ):

            outfield = optimize_role_fit_lineup(
                players,
                formation_name,
                formation_slots,
                role_fit_weight,
            )

            goalkeeper = select_goalkeeper(
                goalkeepers,
                formation_name,
                role_fit_weight,
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
                {
                    "Method":
                        "Quality + Role Fit",

                    "Role Fit Weight":
                        role_fit_weight,

                    "Formation":
                        formation_name,

                    "Average OVR":
                        lineup[
                            "OVR"
                        ].mean(),

                    "Average Role Fit":
                        lineup[
                            "Role Fit Score"
                        ].mean(),

                    "Average Selection Score":
                        lineup[
                            "Selection Score"
                        ].mean(),
                }
            )

            print(
                "\n"
                + "=" * 72
            )

            print(
                "QUALITY + ROLE FIT — "
                f"{formation_name} — "
                f"w={role_fit_weight:.2f}"
            )

            print(
                "=" * 72
            )

            print(
                lineup[
                    [
                        "Formation Slot",
                        "Name",
                        "OVR",
                        "Role Fit Score",
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
        / "all_role_fit_lineups.csv",
        index=False,
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "role_fit_summary.csv",
        index=False,
    )

    reference_lineups = (
        all_lineups_df[
            np.isclose(
                all_lineups_df[
                    "Role Fit Weight"
                ],
                REFERENCE_ROLE_FIT_WEIGHT,
            )
        ]
        .copy()
    )

    reference_lineups.to_csv(
        RESULTS_DIR
        / (
            "reference_role_fit_lineups_"
            f"w_{REFERENCE_ROLE_FIT_WEIGHT:.2f}.csv"
        ),
        index=False,
    )

    print(
        "\nStage 4.2 complete."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The 0.50 weight is a reference setting, "
        "not a claim that 50/50 is the true football value."
    )

    print(
        "The 0.25/0.50/0.75 grid is retained "
        "for sensitivity analysis."
    )


if __name__ == "__main__":
    main()
