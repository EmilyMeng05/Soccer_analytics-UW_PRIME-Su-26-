import numpy as np
import pandas as pd

from stage4_config import (
    FORMATIONS,
    GOALKEEPER_FILE,
    MULTIROLE_FILE,
    PROJECT_ROOT,
)


# ============================================================
# STAGE 4.3 — FORMATION ROBUSTNESS (REVISED)
# ============================================================
#
# PURPOSE
# -------
#
# Stage 4.1:
#     OVR baseline
#     = "best officially eligible player at each position"
#
# Stage 4.2:
#     role-adjusted quality
#     = OVR quality preserved, then mildly penalized when a player's
#       learned Stage 3.6C role profile fits the assigned role poorly.
#
# Stage 4.3 asks:
#
#     1. Which players remain selected across many formations?
#
#     2. How different is the role-adjusted model from the OVR-only
#        baseline?
#
#     3. Are those selections stable when lambda changes?
#
#     4. Which players gain or lose tactical robustness after role fit
#        is introduced?
#
#
# IMPORTANT
# ---------
#
# This stage does NOT claim that one formation is objectively better.
# The seven formations are treated as a tactical stress-test set.
#
# This stage also does NOT yet model teammate complementarity.
#
#
# REFERENCE MODEL
# ---------------
#
# Stage 4.2 tested:
#
#     lambda = 0.10
#     lambda = 0.25
#     lambda = 0.40
#
# We use lambda = 0.25 as the reference setting because it gives role
# fit meaningful influence without allowing the learned role classifier
# to dominate overall player quality.
#
# The other lambda values remain part of a sensitivity analysis.
# ============================================================


# ============================================================
# FILE PATHS
# ============================================================

BASELINE_FILE = (
    PROJECT_ROOT
    / "results"
    / "stage_4_1_ovr_baseline"
    / "all_ovr_baseline_lineups.csv"
)

ROLE_ADJUSTED_FILE = (
    PROJECT_ROOT
    / "results"
    / "stage_4_2_role_adjusted_quality"
    / "all_role_adjusted_quality_lineups.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage_4_3_formation_robustness"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


REFERENCE_LAMBDA = 0.25


# ============================================================
# LOAD + VALIDATE
# ============================================================

def load_inputs():
    """Load Stage 4.1 and revised Stage 4.2 outputs."""

    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            "Could not find the Stage 4.1 OVR baseline output:\n"
            f"{BASELINE_FILE}\n\n"
            "Run stage4_1_ovr_baseline.py first."
        )

    if not ROLE_ADJUSTED_FILE.exists():
        raise FileNotFoundError(
            "Could not find the revised Stage 4.2 output:\n"
            f"{ROLE_ADJUSTED_FILE}\n\n"
            "Run stage4_2_role_adjusted_quality.py first."
        )

    baseline = pd.read_csv(
        BASELINE_FILE
    )

    role_adjusted = pd.read_csv(
        ROLE_ADJUSTED_FILE
    )

    required_baseline = {
        "Formation",
        "Formation Slot",
        "Name",
        "OVR",
    }

    required_role_adjusted = {
        "Formation",
        "Formation Slot",
        "Name",
        "OVR",
        "Role Adjustment Strength",
        "Role Fit Score",
        "Selection Score",
    }

    missing_baseline = sorted(
        required_baseline.difference(
            baseline.columns
        )
    )

    missing_role_adjusted = sorted(
        required_role_adjusted.difference(
            role_adjusted.columns
        )
    )

    if missing_baseline:
        raise ValueError(
            "Stage 4.1 output is missing columns:\n"
            f"{missing_baseline}"
        )

    if missing_role_adjusted:
        raise ValueError(
            "Stage 4.2 output is missing columns:\n"
            f"{missing_role_adjusted}"
        )

    role_adjusted[
        "Role Adjustment Strength"
    ] = pd.to_numeric(
        role_adjusted[
            "Role Adjustment Strength"
        ],
        errors="coerce",
    )

    if role_adjusted[
        "Role Adjustment Strength"
    ].isna().any():
        raise ValueError(
            "Some Stage 4.2 rows have missing lambda values."
        )

    return baseline, role_adjusted


# ============================================================
# BASIC ROBUSTNESS
# ============================================================

def calculate_formation_robustness(
    lineup_data,
    method_label,
):
    """
    Calculate how often each player appears across the benchmark formations.

    Formation Robustness =
        number of distinct formations selecting player
        ------------------------------------------------
        total benchmark formations
    """

    n_formations = len(
        FORMATIONS
    )

    appearances = (
        lineup_data[
            [
                "Name",
                "Formation",
            ]
        ]
        .drop_duplicates()
    )

    robustness = (
        appearances
        .groupby(
            "Name"
        )[
            "Formation"
        ]
        .agg(
            Formations_Selected=lambda values:
                "|".join(
                    sorted(
                        values
                    )
                ),
            Number_of_Formations="nunique",
        )
        .reset_index()
    )

    robustness[
        "Formation Robustness"
    ] = (
        robustness[
            "Number_of_Formations"
        ]
        / n_formations
    )

    robustness[
        "Method"
    ] = method_label

    return (
        robustness
        .rename(
            columns={
                "Formations_Selected":
                    "Formations Selected",

                "Number_of_Formations":
                    "Number of Formations",
            }
        )
        .sort_values(
            [
                "Formation Robustness",
                "Number of Formations",
                "Name",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# BASELINE VS ROLE-ADJUSTED COMPARISON
# ============================================================

def compare_lineups_by_formation(
    baseline,
    reference,
):
    """
    Compare OVR-only and reference role-adjusted lineups formation by formation.

    We report:
        - number of shared players
        - Jaccard overlap
        - which players leave
        - which players enter
        - mean OVR change
        - mean role-fit score of role-adjusted lineup
    """

    rows = []

    for formation_name in FORMATIONS:

        baseline_formation = (
            baseline[
                baseline[
                    "Formation"
                ]
                == formation_name
            ]
            .copy()
        )

        reference_formation = (
            reference[
                reference[
                    "Formation"
                ]
                == formation_name
            ]
            .copy()
        )

        baseline_names = set(
            baseline_formation[
                "Name"
            ]
        )

        reference_names = set(
            reference_formation[
                "Name"
            ]
        )

        shared = (
            baseline_names
            & reference_names
        )

        union = (
            baseline_names
            | reference_names
        )

        baseline_only = sorted(
            baseline_names
            - reference_names
        )

        reference_only = sorted(
            reference_names
            - baseline_names
        )

        rows.append(
            {
                "Formation":
                    formation_name,

                "Shared Players":
                    len(
                        shared
                    ),

                "Players Changed":
                    len(
                        baseline_only
                    ),

                "Jaccard Overlap":
                    (
                        len(
                            shared
                        )
                        / len(
                            union
                        )
                        if union
                        else np.nan
                    ),

                "OVR Baseline Only":
                    "|".join(
                        baseline_only
                    ),

                "Role-Adjusted Only":
                    "|".join(
                        reference_only
                    ),

                "OVR Baseline Mean OVR":
                    baseline_formation[
                        "OVR"
                    ].mean(),

                "Role-Adjusted Mean OVR":
                    reference_formation[
                        "OVR"
                    ].mean(),

                "Mean OVR Change":
                    (
                        reference_formation[
                            "OVR"
                        ].mean()
                        - baseline_formation[
                            "OVR"
                        ].mean()
                    ),

                "Role-Adjusted Mean Role Fit":
                    reference_formation[
                        "Role Fit Score"
                    ].mean(),

                "Role-Adjusted Mean Selection Score":
                    reference_formation[
                        "Selection Score"
                    ].mean(),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PLAYER-LEVEL ROBUSTNESS CHANGE
# ============================================================

def compare_player_robustness(
    baseline_robustness,
    reference_robustness,
):
    """
    Compare cross-formation robustness for each player.

    Positive change:
        player becomes more tactically persistent under role adjustment.

    Negative change:
        player loses appearances after role fit is introduced.
    """

    baseline_part = (
        baseline_robustness[
            [
                "Name",
                "Number of Formations",
                "Formation Robustness",
            ]
        ]
        .rename(
            columns={
                "Number of Formations":
                    "OVR Number of Formations",

                "Formation Robustness":
                    "OVR Formation Robustness",
            }
        )
    )

    reference_part = (
        reference_robustness[
            [
                "Name",
                "Number of Formations",
                "Formation Robustness",
            ]
        ]
        .rename(
            columns={
                "Number of Formations":
                    "Role-Adjusted Number of Formations",

                "Formation Robustness":
                    "Role-Adjusted Formation Robustness",
            }
        )
    )

    comparison = (
        baseline_part
        .merge(
            reference_part,
            on="Name",
            how="outer",
        )
        .fillna(
            0
        )
    )

    comparison[
        "Change in Number of Formations"
    ] = (
        comparison[
            "Role-Adjusted Number of Formations"
        ]
        - comparison[
            "OVR Number of Formations"
        ]
    )

    comparison[
        "Change in Formation Robustness"
    ] = (
        comparison[
            "Role-Adjusted Formation Robustness"
        ]
        - comparison[
            "OVR Formation Robustness"
        ]
    )

    comparison[
        "Robustness Direction"
    ] = np.select(
        [
            comparison[
                "Change in Formation Robustness"
            ] > 0,

            comparison[
                "Change in Formation Robustness"
            ] < 0,
        ],
        [
            "Gained Robustness",
            "Lost Robustness",
        ],
        default="Unchanged",
    )

    return (
        comparison
        .sort_values(
            [
                "Change in Formation Robustness",
                "Role-Adjusted Formation Robustness",
                "Name",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# LAMBDA SENSITIVITY
# ============================================================

def calculate_lambda_stability(
    role_adjusted,
):
    """
    For each formation-player pair, count how many lambda values select
    that player.

    Lambda Stability =
        number of tested lambda values selecting player
        -----------------------------------------------
        total tested lambda values
    """

    tested_lambdas = sorted(
        role_adjusted[
            "Role Adjustment Strength"
        ]
        .dropna()
        .unique()
    )

    n_lambdas = len(
        tested_lambdas
    )

    appearances = (
        role_adjusted[
            [
                "Formation",
                "Name",
                "Role Adjustment Strength",
            ]
        ]
        .drop_duplicates()
    )

    stability = (
        appearances
        .groupby(
            [
                "Formation",
                "Name",
            ]
        )[
            "Role Adjustment Strength"
        ]
        .agg(
            Lambdas_Selected=lambda values:
                "|".join(
                    f"{value:.2f}"
                    for value
                    in sorted(
                        values
                    )
                ),
            Number_of_Lambdas="nunique",
        )
        .reset_index()
    )

    stability[
        "Lambda Stability"
    ] = (
        stability[
            "Number_of_Lambdas"
        ]
        / n_lambdas
    )

    stability = (
        stability
        .rename(
            columns={
                "Lambdas_Selected":
                    "Lambdas Selected",

                "Number_of_Lambdas":
                    "Number of Lambdas",
            }
        )
        .sort_values(
            [
                "Formation",
                "Lambda Stability",
                "Name",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return stability


def summarize_lambda_lineup_overlap(
    role_adjusted,
):
    """
    Compare complete lineups pairwise across lambda settings
    within each formation.

    This tells us whether the *team* is stable under reasonable
    role-adjustment choices.
    """

    lambdas = sorted(
        role_adjusted[
            "Role Adjustment Strength"
        ]
        .dropna()
        .unique()
    )

    rows = []

    for formation_name in FORMATIONS:

        formation_data = (
            role_adjusted[
                role_adjusted[
                    "Formation"
                ]
                == formation_name
            ]
        )

        for index_a in range(
            len(
                lambdas
            )
        ):

            for index_b in range(
                index_a + 1,
                len(
                    lambdas
                ),
            ):

                lambda_a = lambdas[
                    index_a
                ]

                lambda_b = lambdas[
                    index_b
                ]

                names_a = set(
                    formation_data.loc[
                        np.isclose(
                            formation_data[
                                "Role Adjustment Strength"
                            ],
                            lambda_a,
                        ),
                        "Name",
                    ]
                )

                names_b = set(
                    formation_data.loc[
                        np.isclose(
                            formation_data[
                                "Role Adjustment Strength"
                            ],
                            lambda_b,
                        ),
                        "Name",
                    ]
                )

                shared = (
                    names_a
                    & names_b
                )

                union = (
                    names_a
                    | names_b
                )

                rows.append(
                    {
                        "Formation":
                            formation_name,

                        "Lambda A":
                            lambda_a,

                        "Lambda B":
                            lambda_b,

                        "Shared Players":
                            len(
                                shared
                            ),

                        "Jaccard Overlap":
                            (
                                len(
                                    shared
                                )
                                / len(
                                    union
                                )
                                if union
                                else np.nan
                            ),

                        "Only Lambda A":
                            "|".join(
                                sorted(
                                    names_a
                                    - names_b
                                )
                            ),

                        "Only Lambda B":
                            "|".join(
                                sorted(
                                    names_b
                                    - names_a
                                )
                            ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# ROBUST CORE
# ============================================================

def identify_robust_core(
    reference_robustness,
    minimum_formation_share=0.70,
):
    """
    Define a descriptive robust core.

    The threshold is NOT optimized and is not a football law.

    A 0.70 threshold means:
        player appears in at least 70% of benchmark formations.

    With seven formations, this corresponds to at least five.
    """

    core = (
        reference_robustness[
            reference_robustness[
                "Formation Robustness"
            ]
            >= minimum_formation_share
        ]
        .copy()
    )

    core[
        "Core Threshold"
    ] = minimum_formation_share

    return (
        core
        .sort_values(
            [
                "Formation Robustness",
                "Name",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SLOT-SPECIFIC CHANGES
# ============================================================

def compare_slots(
    baseline,
    reference,
):
    """
    Compare the exact selected player in each formation slot.

    This is useful for interpretation:
        Which tactical positions are most affected by role adjustment?
    """

    left = (
        baseline[
            [
                "Formation",
                "Formation Slot",
                "Name",
                "OVR",
            ]
        ]
        .rename(
            columns={
                "Name":
                    "OVR Baseline Player",

                "OVR":
                    "OVR Baseline Player OVR",
            }
        )
    )

    right = (
        reference[
            [
                "Formation",
                "Formation Slot",
                "Name",
                "OVR",
                "Role Fit Score",
                "Selection Score",
            ]
        ]
        .rename(
            columns={
                "Name":
                    "Role-Adjusted Player",

                "OVR":
                    "Role-Adjusted Player OVR",

                "Role Fit Score":
                    "Role-Adjusted Role Fit",

                "Selection Score":
                    "Role-Adjusted Selection Score",
            }
        )
    )

    comparison = (
        left
        .merge(
            right,
            on=[
                "Formation",
                "Formation Slot",
            ],
            how="inner",
        )
    )

    comparison[
        "Same Player"
    ] = (
        comparison[
            "OVR Baseline Player"
        ]
        == comparison[
            "Role-Adjusted Player"
        ]
    )

    comparison[
        "OVR Change"
    ] = (
        comparison[
            "Role-Adjusted Player OVR"
        ]
        - comparison[
            "OVR Baseline Player OVR"
        ]
    )

    return (
        comparison
        .sort_values(
            [
                "Formation",
                "Formation Slot",
            ]
        )
        .reset_index(
            drop=True
        )
    )



# ============================================================
# SELECTION-MARGIN ANALYSIS
# ============================================================

def parse_pipe_or_list(value):
    """Parse exact alternative-position lists saved by Stage 3.6C."""
    import ast

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    if "|" in text:
        return [item.strip() for item in text.split("|") if item.strip()]

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (ValueError, SyntaxError):
        pass

    return [text]


def load_margin_candidate_data():
    """
    Load the same player data used by revised Stage 4.2.

    We recompute candidate scores here because the Stage 4.2 lineup file
    contains only selected players, not every runner-up candidate.
    """
    players = pd.read_csv(MULTIROLE_FILE)
    goalkeepers = pd.read_csv(GOALKEEPER_FILE)

    players["OVR"] = pd.to_numeric(players["OVR"], errors="coerce")

    role_columns = [
        column
        for column in players.columns
        if column.startswith("Role Suitability Probability - ")
    ]

    for column in role_columns:
        players[column] = pd.to_numeric(players[column], errors="coerce")

    players["Official Alternative Position List Parsed"] = (
        players["Official Alternative Position List"]
        .apply(parse_pipe_or_list)
    )

    players["Exact Eligible Positions"] = players.apply(
        lambda row: set(
            [row["Position"]]
            + row["Official Alternative Position List Parsed"]
        ),
        axis=1,
    )

    if "ID" in players.columns:
        players["Player Key"] = players["ID"].astype(str)
    else:
        players["Player Key"] = players["Name"].astype(str)

    for column in ["OVR", "GoalkeeperSkill"]:
        goalkeepers[column] = pd.to_numeric(
            goalkeepers[column],
            errors="coerce",
        )

    return players, goalkeepers


def calculate_slot_role_fit(row, slot_definition):
    """
    Match revised Stage 4.2 exactly.

    Normal slot:
        one functional-role probability.

    LWB/RWB proxy:
        max(Fullback, Wide Midfielder).
    """
    scores = []

    for role in slot_definition["suitability_roles"]:
        column = f"Role Suitability Probability - {role}"
        scores.append(float(row[column]))

    if len(scores) == 1:
        return scores[0]

    return max(scores)


def candidate_selection_score(ovr, role_fit, adjustment_strength):
    """Same role-adjusted-quality equation used in Stage 4.2."""
    quality = float(ovr) / 100.0

    return quality * (
        1.0
        - adjustment_strength
        * (1.0 - role_fit)
    )


def build_outfield_candidate_table(
    players,
    formation_name,
    formation_slots,
    adjustment_strength,
):
    """
    Build all officially eligible candidate-slot scores for one formation.

    IMPORTANT:
    These are raw candidate scores before the global unique-player
    assignment constraint is applied.
    """
    rows = []

    for slot_definition in formation_slots:
        accepted_positions = slot_definition["accepted_positions"]

        for _, player in players.iterrows():
            if not (
                accepted_positions
                & player["Exact Eligible Positions"]
            ):
                continue

            role_fit = calculate_slot_role_fit(
                player,
                slot_definition,
            )

            score = candidate_selection_score(
                player["OVR"],
                role_fit,
                adjustment_strength,
            )

            rows.append(
                {
                    "Formation": formation_name,
                    "Formation Slot": slot_definition["slot"],
                    "Name": player["Name"],
                    "Player Key": player["Player Key"],
                    "OVR": float(player["OVR"]),
                    "Role Fit Score": role_fit,
                    "Candidate Selection Score": score,
                }
            )

    return pd.DataFrame(rows)


def build_goalkeeper_candidate_table(
    goalkeepers,
    formation_name,
    adjustment_strength,
):
    """Build goalkeeper candidate scores using the Stage 4.2 GK equation."""
    candidates = goalkeepers.copy()

    candidates["Role Fit Score"] = (
        candidates["GoalkeeperSkill"] / 100.0
    )

    candidates["Candidate Selection Score"] = (
        (candidates["OVR"] / 100.0)
        * (
            1.0
            - adjustment_strength
            * (1.0 - candidates["Role Fit Score"])
        )
    )

    if "ID" in candidates.columns:
        candidates["Player Key"] = candidates["ID"].astype(str)
    else:
        candidates["Player Key"] = candidates["Name"].astype(str)

    candidates["Formation"] = formation_name
    candidates["Formation Slot"] = "GK"

    return candidates[
        [
            "Formation",
            "Formation Slot",
            "Name",
            "Player Key",
            "OVR",
            "Role Fit Score",
            "Candidate Selection Score",
        ]
    ].copy()


def calculate_reference_player_removal_margins(
    players,
    goalkeepers,
    reference_lineups,
    adjustment_strength,
):
    """
    Measure how strongly each selected player beats alternatives.

    Two margins are reported:

    1. Naive Runner-Up Margin
       Selected score minus the highest-scoring OTHER player for that
       individual slot.

       This is easy to interpret, but it ignores the fact that the same
       player cannot occupy two slots simultaneously.

    2. Player Removal Margin
       Re-optimize the COMPLETE formation after removing the selected
       player from the entire formation candidate pool, then compare
       total lineup score.

       This is the more faithful margin because Stage 4.2 uses a global
       assignment problem with unique-player constraints and duplicate
       role labels such as CM1/CM2 or ST1/ST2.

    A small player-removal margin means the selected lineup has a close
    alternative even if robustness counts make the difference look large.
    """
    from scipy.optimize import linear_sum_assignment

    rows = []

    for formation_name, formation_slots in FORMATIONS.items():
        selected = (
            reference_lineups[
                reference_lineups["Formation"] == formation_name
            ]
            .copy()
        )

        outfield_candidates = build_outfield_candidate_table(
            players,
            formation_name,
            formation_slots,
            adjustment_strength,
        )

        goalkeeper_candidates = build_goalkeeper_candidate_table(
            goalkeepers,
            formation_name,
            adjustment_strength,
        )

        # --------------------------
        # Goalkeeper margin
        # --------------------------
        selected_gk = selected[
            selected["Formation Slot"] == "GK"
        ].iloc[0]

        gk_ranked = (
            goalkeeper_candidates
            .sort_values(
                ["Candidate Selection Score", "OVR", "Name"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )

        gk_alternatives = gk_ranked[
            gk_ranked["Name"] != selected_gk["Name"]
        ]

        runner_up = (
            gk_alternatives.iloc[0]
            if not gk_alternatives.empty
            else None
        )

        selected_score = float(selected_gk["Selection Score"])

        rows.append(
            {
                "Formation": formation_name,
                "Formation Slot": "GK",
                "Selected Player": selected_gk["Name"],
                "Selected OVR": selected_gk["OVR"],
                "Selected Role Fit": selected_gk["Role Fit Score"],
                "Selected Score": selected_score,
                "Naive Runner-Up": (
                    runner_up["Name"] if runner_up is not None else np.nan
                ),
                "Naive Runner-Up OVR": (
                    runner_up["OVR"] if runner_up is not None else np.nan
                ),
                "Naive Runner-Up Role Fit": (
                    runner_up["Role Fit Score"]
                    if runner_up is not None else np.nan
                ),
                "Naive Runner-Up Score": (
                    runner_up["Candidate Selection Score"]
                    if runner_up is not None else np.nan
                ),
                "Naive Runner-Up Margin": (
                    selected_score
                    - float(runner_up["Candidate Selection Score"])
                    if runner_up is not None else np.nan
                ),
                # GK is independent of outfield assignment, so the
                # player-removal margin equals the direct runner-up margin.
                "Player Removal Margin": (
                    selected_score
                    - float(runner_up["Candidate Selection Score"])
                    if runner_up is not None else np.nan
                ),
            }
        )

        # --------------------------
        # Outfield global assignment
        # --------------------------
        player_keys = (
            players["Player Key"]
            .drop_duplicates()
            .tolist()
        )

        key_to_col = {
            key: idx
            for idx, key in enumerate(player_keys)
        }

        n_slots = len(formation_slots)
        n_players = len(player_keys)

        score_matrix = np.full(
            (n_slots, n_players),
            -1_000_000.0,
        )

        name_matrix = {}

        for slot_idx, slot_definition in enumerate(formation_slots):
            slot_name = slot_definition["slot"]

            slot_candidates = outfield_candidates[
                outfield_candidates["Formation Slot"] == slot_name
            ]

            for _, candidate in slot_candidates.iterrows():
                col = key_to_col[candidate["Player Key"]]
                score = float(candidate["Candidate Selection Score"])

                if score > score_matrix[slot_idx, col]:
                    score_matrix[slot_idx, col] = score
                    name_matrix[(slot_idx, col)] = candidate["Name"]

        base_rows, base_cols = linear_sum_assignment(
            score_matrix,
            maximize=True,
        )

        base_total = float(
            score_matrix[base_rows, base_cols].sum()
        )

        for slot_idx, slot_definition in enumerate(formation_slots):
            slot_name = slot_definition["slot"]

            selected_row = selected[
                selected["Formation Slot"] == slot_name
            ].iloc[0]

            selected_name = selected_row["Name"]
            selected_score = float(selected_row["Selection Score"])

            slot_candidates = (
                outfield_candidates[
                    outfield_candidates["Formation Slot"] == slot_name
                ]
                .sort_values(
                    ["Candidate Selection Score", "OVR", "Name"],
                    ascending=[False, False, True],
                )
                .copy()
            )

            naive_alternatives = slot_candidates[
                slot_candidates["Name"] != selected_name
            ]

            naive_runner_up = (
                naive_alternatives.iloc[0]
                if not naive_alternatives.empty
                else None
            )

            # Remove the selected player from the ENTIRE formation,
            # not just from this one slot. This avoids artificial zero
            # margins caused by equivalent slots such as CM1/CM2 or ST1/ST2.
            selected_key = str(selected_row["Player Key"])
            selected_col = key_to_col[selected_key]

            counterfactual_matrix = score_matrix.copy()
            counterfactual_matrix[:, selected_col] = -1_000_000.0

            alt_rows, alt_cols = linear_sum_assignment(
                counterfactual_matrix,
                maximize=True,
            )

            alt_total = float(
                counterfactual_matrix[
                    alt_rows,
                    alt_cols,
                ].sum()
            )

            player_removal_margin = (
                base_total - alt_total
            )

            # Record which player now occupies the selected player's
            # original slot after the whole XI is re-optimized.
            replacement_player = name_matrix.get(
                (slot_idx, alt_cols[list(alt_rows).index(slot_idx)]),
                np.nan,
            )

            rows.append(
                {
                    "Formation": formation_name,
                    "Formation Slot": slot_name,
                    "Selected Player": selected_name,
                    "Selected OVR": selected_row["OVR"],
                    "Selected Role Fit": selected_row["Role Fit Score"],
                    "Selected Score": selected_score,
                    "Naive Runner-Up": (
                        naive_runner_up["Name"]
                        if naive_runner_up is not None else np.nan
                    ),
                    "Naive Runner-Up OVR": (
                        naive_runner_up["OVR"]
                        if naive_runner_up is not None else np.nan
                    ),
                    "Naive Runner-Up Role Fit": (
                        naive_runner_up["Role Fit Score"]
                        if naive_runner_up is not None else np.nan
                    ),
                    "Naive Runner-Up Score": (
                        naive_runner_up["Candidate Selection Score"]
                        if naive_runner_up is not None else np.nan
                    ),
                    "Naive Runner-Up Margin": (
                        selected_score
                        - float(
                            naive_runner_up["Candidate Selection Score"]
                        )
                        if naive_runner_up is not None else np.nan
                    ),
                    "Replacement Player After Reoptimization":
                        replacement_player,
                    "Player Removal Margin":
                        player_removal_margin,
                }
            )

    return pd.DataFrame(rows)


def summarize_margin_strength(player_removal_margins):
    """
    Formation-level descriptive summary of how decisive selections are.

    We intentionally report continuous margins rather than inventing
    arbitrary 'close' versus 'decisive' thresholds.
    """
    return (
        player_removal_margins
        .groupby("Formation")
        .agg(
            Mean_Player_Removal_Margin=(
                "Player Removal Margin",
                "mean",
            ),
            Median_Player_Removal_Margin=(
                "Player Removal Margin",
                "median",
            ),
            Minimum_Player_Removal_Margin=(
                "Player Removal Margin",
                "min",
            ),
            Maximum_Player_Removal_Margin=(
                "Player Removal Margin",
                "max",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "Mean_Player_Removal_Margin":
                    "Mean Player Removal Margin",
                "Median_Player_Removal_Margin":
                    "Median Player Removal Margin",
                "Minimum_Player_Removal_Margin":
                    "Minimum Player Removal Margin",
                "Maximum_Player_Removal_Margin":
                    "Maximum Player Removal Margin",
            }
        )
    )



# ============================================================
# RUN
# ============================================================

def main():

    baseline, role_adjusted = (
        load_inputs()
    )

    reference = (
        role_adjusted[
            np.isclose(
                role_adjusted[
                    "Role Adjustment Strength"
                ],
                REFERENCE_LAMBDA,
            )
        ]
        .copy()
    )

    if reference.empty:
        raise ValueError(
            "No Stage 4.2 rows were found for "
            f"lambda={REFERENCE_LAMBDA:.2f}."
        )

    # --------------------------------------------------------
    # 1. Cross-formation robustness
    # --------------------------------------------------------

    baseline_robustness = (
        calculate_formation_robustness(
            baseline,
            "OVR Baseline",
        )
    )

    reference_robustness = (
        calculate_formation_robustness(
            reference,
            (
                "Role-Adjusted Quality "
                f"(lambda={REFERENCE_LAMBDA:.2f})"
            ),
        )
    )

    # --------------------------------------------------------
    # 2. OVR vs role-adjusted comparison
    # --------------------------------------------------------

    formation_comparison = (
        compare_lineups_by_formation(
            baseline,
            reference,
        )
    )

    player_robustness_change = (
        compare_player_robustness(
            baseline_robustness,
            reference_robustness,
        )
    )

    slot_comparison = (
        compare_slots(
            baseline,
            reference,
        )
    )

    # --------------------------------------------------------
    # 3. Sensitivity to lambda
    # --------------------------------------------------------

    lambda_stability = (
        calculate_lambda_stability(
            role_adjusted
        )
    )

    lambda_lineup_overlap = (
        summarize_lambda_lineup_overlap(
            role_adjusted
        )
    )

    # --------------------------------------------------------
    # 4. Descriptive tactical core
    # --------------------------------------------------------

    robust_core = (
        identify_robust_core(
            reference_robustness,
            minimum_formation_share=0.70,
        )
    )

    # --------------------------------------------------------
    # 5. Selection margins
    # --------------------------------------------------------

    margin_players, margin_goalkeepers = (
        load_margin_candidate_data()
    )

    player_removal_margins = (
        calculate_reference_player_removal_margins(
            margin_players,
            margin_goalkeepers,
            reference,
            REFERENCE_LAMBDA,
        )
    )

    margin_summary = (
        summarize_margin_strength(
            player_removal_margins
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    baseline_robustness.to_csv(
        RESULTS_DIR
        / "ovr_baseline_formation_robustness.csv",
        index=False,
    )

    reference_robustness.to_csv(
        RESULTS_DIR
        / "reference_role_adjusted_formation_robustness.csv",
        index=False,
    )

    formation_comparison.to_csv(
        RESULTS_DIR
        / "baseline_vs_role_adjusted_by_formation.csv",
        index=False,
    )

    player_robustness_change.to_csv(
        RESULTS_DIR
        / "player_robustness_change.csv",
        index=False,
    )

    slot_comparison.to_csv(
        RESULTS_DIR
        / "slot_level_baseline_vs_role_adjusted.csv",
        index=False,
    )

    lambda_stability.to_csv(
        RESULTS_DIR
        / "lambda_player_stability.csv",
        index=False,
    )

    lambda_lineup_overlap.to_csv(
        RESULTS_DIR
        / "lambda_lineup_overlap.csv",
        index=False,
    )

    robust_core.to_csv(
        RESULTS_DIR
        / "reference_robust_core.csv",
        index=False,
    )

    player_removal_margins.to_csv(
        RESULTS_DIR
        / "reference_player_removal_margins.csv",
        index=False,
    )

    margin_summary.to_csv(
        RESULTS_DIR
        / "reference_player_removal_margin_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # PRINT INTERPRETABLE RESULTS
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 80
    )

    print(
        "REFERENCE ROLE-ADJUSTED FORMATION ROBUSTNESS "
        f"(lambda={REFERENCE_LAMBDA:.2f})"
    )

    print(
        "=" * 80
    )

    print(
        reference_robustness[
            [
                "Name",
                "Number of Formations",
                "Formation Robustness",
                "Formations Selected",
            ]
        ]
        .head(
            30
        )
        .round(
            {
                "Formation Robustness":
                    3,
            }
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "DESCRIPTIVE ROBUST CORE "
        "(selected in >= 70% of formations)"
    )

    print(
        "=" * 80
    )

    print(
        robust_core[
            [
                "Name",
                "Number of Formations",
                "Formation Robustness",
            ]
        ]
        .round(
            {
                "Formation Robustness":
                    3,
            }
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "OVR BASELINE VS ROLE-ADJUSTED LINEUPS"
    )

    print(
        "=" * 80
    )

    print(
        formation_comparison[
            [
                "Formation",
                "Shared Players",
                "Players Changed",
                "Jaccard Overlap",
                "Mean OVR Change",
                "Role-Adjusted Mean Role Fit",
            ]
        ]
        .round(
            {
                "Jaccard Overlap":
                    3,

                "Mean OVR Change":
                    3,

                "Role-Adjusted Mean Role Fit":
                    3,
            }
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BIGGEST ROBUSTNESS GAINS"
    )

    print(
        "=" * 80
    )

    print(
        player_robustness_change[
            player_robustness_change[
                "Change in Formation Robustness"
            ]
            > 0
        ][
            [
                "Name",
                "OVR Formation Robustness",
                "Role-Adjusted Formation Robustness",
                "Change in Formation Robustness",
            ]
        ]
        .head(
            15
        )
        .round(
            3
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "BIGGEST ROBUSTNESS LOSSES"
    )

    print(
        "=" * 80
    )

    print(
        player_robustness_change[
            player_robustness_change[
                "Change in Formation Robustness"
            ]
            < 0
        ][
            [
                "Name",
                "OVR Formation Robustness",
                "Role-Adjusted Formation Robustness",
                "Change in Formation Robustness",
            ]
        ]
        .sort_values(
            "Change in Formation Robustness"
        )
        .head(
            15
        )
        .round(
            3
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "LAMBDA LINEUP STABILITY"
    )

    print(
        "=" * 80
    )

    print(
        lambda_lineup_overlap[
            [
                "Formation",
                "Lambda A",
                "Lambda B",
                "Shared Players",
                "Jaccard Overlap",
            ]
        ]
        .round(
            {
                "Jaccard Overlap":
                    3,
            }
        )
        .to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "SMALLEST PLAYER-REMOVAL MARGINS "
        f"(lambda={REFERENCE_LAMBDA:.2f})"
    )

    print(
        "=" * 80
    )

    print(
        player_removal_margins[
            [
                "Formation",
                "Formation Slot",
                "Selected Player",
                "Selected Score",
                "Naive Runner-Up",
                "Naive Runner-Up Score",
                "Naive Runner-Up Margin",
                "Replacement Player After Reoptimization",
                "Player Removal Margin",
            ]
        ]
        .sort_values(
            "Player Removal Margin"
        )
        .head(30)
        .round(5)
        .to_string(index=False)
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FORMATION-LEVEL PLAYER-REMOVAL MARGIN SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        margin_summary
        .round(5)
        .to_string(index=False)
    )

    print(
        "\nStage 4.3 complete."
    )

    print(
        "\nInterpretation:"
    )

    print(
        "Formation robustness measures how often a player survives "
        "changes in tactical structure."
    )

    print(
        "It is not yet a measure of teammate complementarity or "
        "real-world match performance."
    )


if __name__ == "__main__":
    main()
