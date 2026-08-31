from pathlib import Path
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
# STAGE 4.1 — OVR BASELINE
# ============================================================
#
# Research question:
#
# What lineup do we obtain if we simply select the highest-OVR
# eligible player for every required position?
#
# This is the project's primary baseline:
#
#     "Choose the best player at each position."
#
# Eligibility comes ONLY from official EAFC information:
#
#   primary Position
#   +
#   official Alternative positions
#
# Learned role-suitability probabilities are NOT used here.
#
# We solve the lineup as one assignment problem instead of calling
# max() independently for each slot because the same flexible player
# may be eligible for more than one position but cannot occupy two
# positions simultaneously.
# ============================================================


RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage_4_1_ovr_baseline"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HELPERS
# ============================================================

def parse_pipe_or_list(value):
    """Parse a Stage 3.6C exact-position list from the saved CSV."""

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    # Stage 3.6C currently saves this as a pipe-separated string.
    if "|" in text:
        return [
            item.strip()
            for item in text.split("|")
            if item.strip()
        ]

    # This fallback makes the script robust if an older output
    # still contains a Python-list-like string.
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


def load_outfield_players():
    """Load Stage 3.6C and reconstruct exact official eligibility."""

    if not MULTIROLE_FILE.exists():
        raise FileNotFoundError(
            "Stage 3.6C output was not found:\n"
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

    if df["OVR"].isna().any():
        raise ValueError(
            "Some outfield players have missing OVR values."
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

    # ID is safer than Name when available.
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
            "Stage 3.7 goalkeeper output was not found:\n"
            f"{GOALKEEPER_FILE}\n\n"
            "Run stage3_7_goalkeeper_analysis.py first."
        )

    df = pd.read_csv(
        GOALKEEPER_FILE
    )

    required = {
        "Name",
        "OVR",
    }

    missing = sorted(
        required.difference(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Goalkeeper output is missing columns:\n"
            f"{missing}"
        )

    df["OVR"] = pd.to_numeric(
        df["OVR"],
        errors="coerce",
    )

    return df.dropna(
        subset=[
            "OVR",
            "Name",
        ]
    ).copy()


def optimize_ovr_lineup(
    players,
    formation_name,
    formation_slots,
):
    """Maximize total OVR while enforcing exact slot eligibility."""

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

    for slot_index, slot_definition in enumerate(
        formation_slots
    ):

        accepted_positions = slot_definition[
            "accepted_positions"
        ]

        for row_index, row in players.iterrows():

            eligible_positions = row[
                "Exact Eligible Positions"
            ]

            if not (
                accepted_positions
                & eligible_positions
            ):
                continue

            player_key = row[
                "Player Key"
            ]

            player_column = key_to_column[
                player_key
            ]

            score = float(
                row["OVR"]
            )

            if score > score_matrix[
                slot_index,
                player_column,
            ]:
                score_matrix[
                    slot_index,
                    player_column,
                ] = score

                row_lookup[
                    (
                        slot_index,
                        player_column,
                    )
                ] = row_index

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
                "No valid OVR-baseline assignment exists for "
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

        slot_definition = formation_slots[
            slot_index
        ]

        selected_rows.append(
            {
                "Method":
                    "OVR Baseline",

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

                "Accepted Exact Positions":
                    "|".join(
                        sorted(
                            slot_definition[
                                "accepted_positions"
                            ]
                        )
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


def select_ovr_goalkeeper(
    goalkeepers,
    formation_name,
):
    """Select the highest-OVR goalkeeper."""

    row = (
        goalkeepers
        .sort_values(
            [
                "OVR",
                "GoalkeeperSkill",
            ]
            if "GoalkeeperSkill"
            in goalkeepers.columns
            else [
                "OVR"
            ],
            ascending=False,
        )
        .iloc[0]
    )

    return pd.DataFrame(
        [
            {
                "Method":
                    "OVR Baseline",

                "Formation":
                    formation_name,

                "Formation Slot":
                    "GK",

                "Name":
                    row["Name"],

                "Player Key":
                    (
                        str(row["ID"])
                        if "ID" in row.index
                        else row["Name"]
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

                "Accepted Exact Positions":
                    "Goalkeeper",

                "Player Exact Eligible Positions":
                    "Goalkeeper",
            }
        ]
    )


def summarize_lineup(
    formation_name,
    lineup,
):
    return {
        "Method":
            "OVR Baseline",

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
    }


# ============================================================
# RUN
# ============================================================

def main():

    players = load_outfield_players()

    goalkeepers = load_goalkeepers()

    all_lineups = []

    summaries = []

    for formation_name, formation_slots in (
        FORMATIONS.items()
    ):

        outfield_lineup = optimize_ovr_lineup(
            players,
            formation_name,
            formation_slots,
        )

        goalkeeper = select_ovr_goalkeeper(
            goalkeepers,
            formation_name,
        )

        lineup = pd.concat(
            [
                goalkeeper,
                outfield_lineup,
            ],
            ignore_index=True,
        )

        all_lineups.append(
            lineup
        )

        summaries.append(
            summarize_lineup(
                formation_name,
                lineup,
            )
        )

        safe_name = (
            formation_name
            .replace(
                "-",
                "_",
            )
        )

        lineup.to_csv(
            RESULTS_DIR
            / (
                f"ovr_baseline_"
                f"{safe_name}.csv"
            ),
            index=False,
        )

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"OVR BASELINE — "
            f"{formation_name}"
        )

        print(
            "=" * 70
        )

        print(
            lineup[
                [
                    "Formation Slot",
                    "Name",
                    "Primary Position",
                    "OVR",
                ]
            ]
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
        / "all_ovr_baseline_lineups.csv",
        index=False,
    )

    summary_df.to_csv(
        RESULTS_DIR
        / "ovr_baseline_summary.csv",
        index=False,
    )

    print(
        "\nStage 4.1 complete."
    )

    print(
        "\nPrimary baseline:"
    )

    print(
        "Highest OVR official-eligible "
        "player at each formation slot."
    )


if __name__ == "__main__":
    main()
