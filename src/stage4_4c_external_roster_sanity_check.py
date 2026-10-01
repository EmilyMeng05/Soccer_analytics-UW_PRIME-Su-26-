"""Stage 4.4C: external descriptive sanity check for attribute coverage.

Scores two elite model-selected 4-3-3 XIs and compares them with random,
formation-valid global XIs matched on mean OVR.  This is not match-outcome
validation; it only asks whether real elite roster constructions look unusual
under the Stage 4.4B static-attribute coverage metric.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from stage4_config import FORMATIONS, PROJECT_ROOT
from stage4_2_role_adjusted_quality import load_players
from stage4_4b_attribute_complementarity import (
    ATTRIBUTES,
    FUNCTIONS,
    add_percentiles,
    concave_value,
    derive_formation_weights,
)


N_ACCEPTED = 3000
OVR_TOLERANCE = 0.25
OVR_WINDOW_BELOW = 3.0
OVR_WINDOW_ABOVE = 5.0
RANDOM_SEED = 20260930
RESULTS_DIR = PROJECT_ROOT / "results" / "stage_4_4c_external_roster_sanity_check"
ROSTERS = {
    "Spain 2026": PROJECT_ROOT / "results" / "stage_5_2_spain"
    / "spain_2026_model_selected_4_3_3.csv",
    "Real Madrid 2026-27": PROJECT_ROOT / "results" / "stage_5_3_real_madrid"
    / "real_madrid_model_selected_4_3_3.csv",
}


def coverage_score(lineup, weights, function_name):
    totals = {
        attribute: lineup[f"{attribute} Percentile"].sum()
        for attribute in ATTRIBUTES
    }
    return 10.0 * sum(
        float(weights[f"{attribute} Weight"])
        * float(concave_value(function_name, totals[attribute] / 10.0))
        for attribute in ATTRIBUTES
    )


def attach_percentiles(lineup, population):
    result = lineup.copy()
    for attribute in ATTRIBUTES:
        values = population[attribute].to_numpy()
        result[f"{attribute} Percentile"] = result[attribute].apply(
            lambda x: (values <= float(x)).mean()
        )
    return result


def build_random_candidate_matrix(players, target_ovr):
    slots = FORMATIONS["4-3-3"]
    eligible_players = players[
        players["OVR"].between(
            target_ovr - OVR_WINDOW_BELOW,
            target_ovr + OVR_WINDOW_ABOVE,
        )
    ].copy().reset_index(drop=True)
    matrix = np.zeros((len(slots), len(eligible_players)), dtype=bool)
    for slot_index, slot in enumerate(slots):
        matrix[slot_index] = eligible_players["Exact Eligible Positions"].apply(
            lambda positions: bool(slot["accepted_positions"] & positions)
        ).to_numpy()
    if (~matrix.any(axis=1)).any():
        raise ValueError("OVR window leaves at least one formation slot empty.")
    return eligible_players, matrix


def sample_matched_lineups(players, target_ovr, weights, rng):
    candidates, eligibility = build_random_candidate_matrix(players, target_ovr)
    accepted = []
    attempts = 0
    max_attempts = 500000
    while len(accepted) < N_ACCEPTED and attempts < max_attempts:
        attempts += 1
        random_scores = rng.random(eligibility.shape)
        random_scores[~eligibility] = -1e9
        slot_indices, player_indices = linear_sum_assignment(random_scores, maximize=True)
        if (random_scores[slot_indices, player_indices] < 0).any():
            continue
        lineup = candidates.iloc[player_indices]
        mean_ovr = lineup["OVR"].mean()
        if abs(mean_ovr - target_ovr) > OVR_TOLERANCE:
            continue
        record = {"Mean OVR": mean_ovr}
        for function_name in FUNCTIONS:
            record[f"Coverage {function_name}"] = coverage_score(
                lineup, weights, function_name
            )
        accepted.append(record)
    if len(accepted) < N_ACCEPTED:
        raise RuntimeError(
            f"Only obtained {len(accepted)} matched lineups in {attempts} attempts."
        )
    return pd.DataFrame(accepted), attempts


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    players = add_percentiles(load_players())
    weights_table = derive_formation_weights(players)
    weights = weights_table[weights_table["Formation"] == "4-3-3"].iloc[0]
    rng = np.random.default_rng(RANDOM_SEED)
    distribution_rows, summary_rows = [], []

    for roster_name, path in ROSTERS.items():
        lineup = attach_percentiles(pd.read_csv(path), players)
        target_ovr = lineup["OVR"].mean()
        random_lineups, attempts = sample_matched_lineups(
            players, target_ovr, weights, rng
        )
        random_lineups.insert(0, "Roster", roster_name)
        distribution_rows.append(random_lineups)
        for function_name in FUNCTIONS:
            actual = coverage_score(lineup, weights, function_name)
            random_values = random_lineups[f"Coverage {function_name}"]
            percentile = 100.0 * (
                (random_values < actual).sum()
                + 0.5 * (random_values == actual).sum()
            ) / len(random_values)
            summary_rows.append({
                "Roster": roster_name,
                "Complementarity Function": function_name,
                "Roster Mean OVR": target_ovr,
                "Roster Coverage": actual,
                "Random Mean Coverage": random_values.mean(),
                "Random SD Coverage": random_values.std(ddof=1),
                "Coverage Z Score": (actual - random_values.mean()) / random_values.std(ddof=1),
                "Percentile Among OVR-Matched Random XIs": percentile,
                "Random XIs": len(random_values),
                "Sampling Attempts": attempts,
                "OVR Tolerance": OVR_TOLERANCE,
                "OVR Window Below": OVR_WINDOW_BELOW,
                "OVR Window Above": OVR_WINDOW_ABOVE,
            })

    pd.concat(distribution_rows, ignore_index=True).to_csv(
        RESULTS_DIR / "matched_random_xi_distributions.csv", index=False
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "external_roster_coverage_summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print("\nInterpretation: descriptive static-attribute check, not outcome validation.")


if __name__ == "__main__":
    main()
