"""Stage 4.4B: joint role quality and team attribute complementarity.

The model maximizes, for each formation,

    sum_i RoleAdjustedQuality_i + gamma * C_f(X)

where C_f is a weighted, concave function of the selected outfield
players' percentile-normalized PAC/SHO/PAS/DRI/DEF/PHY totals.

Formation weights are derived from the empirical demand profiles produced
by Stage 4.4A.  A piecewise-linear MILP representation keeps the assignment
problem exact at the chosen breakpoints.  gamma=0 is automatically checked
against the Stage 4.2 reference lineups.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from stage4_config import FORMATIONS, PROJECT_ROOT
from stage4_2_role_adjusted_quality import (
    REFERENCE_LAMBDA,
    calculate_slot_role_fit,
    load_players,
    role_adjusted_quality,
)


ATTRIBUTES = ["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]
GAMMAS = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]
FUNCTIONS = ["sqrt", "log", "exp"]
N_SEGMENTS = 20

DEMAND_FILE = (
    PROJECT_ROOT / "results" / "stage_4_4a_team_attribute_coverage"
    / "formation_attribute_demand_profiles.csv"
)
REFERENCE_FILE = (
    PROJECT_ROOT / "results" / "stage_4_2_role_adjusted_quality"
    / "reference_role_adjusted_quality_lambda_0.25.csv"
)
RESULTS_DIR = PROJECT_ROOT / "results" / "stage_4_4b_attribute_complementarity"


def concave_value(name, x):
    """Normalized concave functions on [0, 1]."""
    x = np.asarray(x, dtype=float)
    if name == "sqrt":
        return np.sqrt(x)
    if name == "log":
        return np.log1p(4.0 * x) / np.log(5.0)
    if name == "exp":
        return (1.0 - np.exp(-3.0 * x)) / (1.0 - np.exp(-3.0))
    raise ValueError(f"Unknown complementarity function: {name}")


def validate_inputs(players):
    missing = sorted(set(ATTRIBUTES).difference(players.columns))
    if missing:
        raise ValueError(f"Player data is missing broad attributes: {missing}")
    if not DEMAND_FILE.exists():
        raise FileNotFoundError(f"Run Stage 4.4A first: {DEMAND_FILE}")
    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Run Stage 4.2 first: {REFERENCE_FILE}")


def add_percentiles(players):
    """Percentile-normalize attributes over the outfield candidate pool."""
    result = players.copy()
    for attribute in ATTRIBUTES:
        result[attribute] = pd.to_numeric(result[attribute], errors="coerce")
        if result[attribute].isna().any():
            raise ValueError(f"Missing values found in {attribute}.")
        result[f"{attribute} Percentile"] = result[attribute].rank(
            method="average", pct=True
        )
    return result


def derive_formation_weights(players):
    """Convert Stage 4.4A demand values to population percentiles and normalize."""
    demand = pd.read_csv(DEMAND_FILE)
    rows = []
    for _, row in demand.iterrows():
        raw = []
        for attribute in ATTRIBUTES:
            value = float(row[attribute])
            raw.append(float((players[attribute] <= value).mean()))
        raw = np.asarray(raw)
        weights = raw / raw.sum()
        record = {"Formation": row["Formation"]}
        for attribute, demand_percentile, weight in zip(ATTRIBUTES, raw, weights):
            record[f"{attribute} Demand Percentile"] = demand_percentile
            record[f"{attribute} Weight"] = weight
        rows.append(record)
    return pd.DataFrame(rows)


def prepare_candidates(players, slots):
    """Create one eligible player-slot row for every official assignment."""
    rows = []
    for slot_index, slot_definition in enumerate(slots):
        for player_index, player in players.iterrows():
            if not (slot_definition["accepted_positions"] & player["Exact Eligible Positions"]):
                continue
            fit = calculate_slot_role_fit(player, slot_definition)
            _, quality = role_adjusted_quality(player["OVR"], fit, REFERENCE_LAMBDA)
            record = {
                "slot_index": slot_index,
                "Formation Slot": slot_definition["slot"],
                "player_index": player_index,
                "Player Key": player["Player Key"],
                "Name": player["Name"],
                "Position": player["Position"],
                "OVR": float(player["OVR"]),
                "Role Fit": fit,
                "Role Adjusted Quality": quality,
            }
            for attribute in ATTRIBUTES:
                record[attribute] = float(player[attribute])
                record[f"{attribute} Percentile"] = float(player[f"{attribute} Percentile"])
            rows.append(record)
    return pd.DataFrame(rows)


def optimize_formation(candidates, slots, weights, gamma, function_name):
    """Solve one formation with binary assignments and concave coverage segments."""
    n_x = len(candidates)
    breaks = np.linspace(0.0, 10.0, N_SEGMENTS + 1)
    widths = np.diff(breaks)
    n_z = len(ATTRIBUTES) * N_SEGMENTS
    n_vars = n_x + n_z

    objective = np.zeros(n_vars)
    objective[:n_x] = -candidates["Role Adjusted Quality"].to_numpy()

    # C_f is scaled to [0, 10], comparable with ten per-player quality terms.
    for a_index, attribute in enumerate(ATTRIBUTES):
        values = 10.0 * concave_value(function_name, breaks / 10.0)
        slopes = np.diff(values) / widths
        weight = float(weights[f"{attribute} Weight"])
        start = n_x + a_index * N_SEGMENTS
        objective[start:start + N_SEGMENTS] = -gamma * weight * slopes

    matrix_rows, matrix_cols, matrix_values = [], [], []
    lower, upper = [], []
    row_number = 0

    def add(coefficients, lo, hi):
        nonlocal row_number
        for column, value in coefficients.items():
            matrix_rows.append(row_number)
            matrix_cols.append(column)
            matrix_values.append(value)
        lower.append(lo)
        upper.append(hi)
        row_number += 1

    for slot_index in range(len(slots)):
        indices = candidates.index[candidates["slot_index"] == slot_index]
        add({int(i): 1.0 for i in indices}, 1.0, 1.0)

    for _, group in candidates.groupby("Player Key"):
        add({int(i): 1.0 for i in group.index}, 0.0, 1.0)

    for a_index, attribute in enumerate(ATTRIBUTES):
        coefficients = {
            int(i): float(value)
            for i, value in candidates[f"{attribute} Percentile"].items()
        }
        start = n_x + a_index * N_SEGMENTS
        for segment in range(N_SEGMENTS):
            coefficients[start + segment] = -1.0
        add(coefficients, 0.0, 0.0)

    matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_cols)), shape=(row_number, n_vars)
    ).tocsr()
    constraints = LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    variable_lower = np.zeros(n_vars)
    variable_upper = np.ones(n_vars)
    for a_index in range(len(ATTRIBUTES)):
        start = n_x + a_index * N_SEGMENTS
        variable_upper[start:start + N_SEGMENTS] = widths
    integrality = np.zeros(n_vars)
    integrality[:n_x] = 1

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=constraints,
        options={"time_limit": 180, "mip_rel_gap": 1e-9, "disp": False},
    )
    if result.x is None:
        raise RuntimeError(f"MILP failed: {result.message}")

    selected = candidates.loc[np.flatnonzero(result.x[:n_x] > 0.5)].copy()
    selected = selected.sort_values("slot_index").reset_index(drop=True)
    totals = {
        attribute: selected[f"{attribute} Percentile"].sum()
        for attribute in ATTRIBUTES
    }
    coverage = 10.0 * sum(
        float(weights[f"{attribute} Weight"])
        * float(concave_value(function_name, totals[attribute] / 10.0))
        for attribute in ATTRIBUTES
    )
    quality_total = selected["Role Adjusted Quality"].sum()
    return selected, quality_total, coverage, quality_total + gamma * coverage, result.message


def validate_gamma_zero(lineups, reference):
    """Require exact player sets and scores at gamma=0 for every formation."""
    rows = []
    all_pass = True
    for formation in FORMATIONS:
        actual = lineups[(formation, "sqrt", 0.0)]
        expected = reference[reference["Formation"] == formation]
        actual_names = set(actual["Name"])
        expected_names = set(expected["Name"])
        names_match = actual_names == expected_names
        actual_score = actual["Role Adjusted Quality"].sum()
        expected_score = expected["Selection Score"].sum()
        score_match = bool(np.isclose(actual_score, expected_score, atol=1e-9))
        passed = names_match and score_match
        all_pass &= passed
        rows.append({
            "Formation": formation,
            "Player Set Match": names_match,
            "Score Match": score_match,
            "Actual Quality Total": actual_score,
            "Reference Quality Total": expected_score,
            "Missing From Stage 4.4B": "|".join(sorted(expected_names - actual_names)),
            "Added By Stage 4.4B": "|".join(sorted(actual_names - expected_names)),
            "Passed": passed,
        })
    checks = pd.DataFrame(rows)
    if not all_pass:
        raise AssertionError("gamma=0 did not reproduce Stage 4.2. See validation CSV.")
    return checks


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    players = load_players()
    validate_inputs(players)
    players = add_percentiles(players)
    weights = derive_formation_weights(players)
    reference = pd.read_csv(REFERENCE_FILE)
    reference = reference[reference["Formation Slot"] != "GK"].copy()

    all_lineups = {}
    lineup_rows, summary_rows = [], []
    for formation, slots in FORMATIONS.items():
        candidates = prepare_candidates(players, slots).reset_index(drop=True)
        formation_weights = weights[weights["Formation"] == formation].iloc[0]
        for function_name in FUNCTIONS:
            for gamma in GAMMAS:
                selected, quality, coverage, objective, status = optimize_formation(
                    candidates, slots, formation_weights, gamma, function_name
                )
                all_lineups[(formation, function_name, gamma)] = selected
                selected.insert(0, "Formation", formation)
                selected.insert(1, "Complementarity Function", function_name)
                selected.insert(2, "Gamma", gamma)
                lineup_rows.append(selected)
                summary_rows.append({
                    "Formation": formation,
                    "Complementarity Function": function_name,
                    "Gamma": gamma,
                    "Quality Total": quality,
                    "Mean OVR": selected["OVR"].mean(),
                    "Coverage Score": coverage,
                    "Joint Objective": objective,
                    "Solver Status": status,
                    "Players": "|".join(selected["Name"]),
                })
                print(formation, function_name, gamma, round(quality, 5), round(coverage, 5))

    checks = validate_gamma_zero(all_lineups, reference)
    all_lineups_df = pd.concat(lineup_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    # Compare every setting with its gamma=0 lineup within formation/function.
    comparisons = []
    for _, row in summary.iterrows():
        baseline = all_lineups[(row["Formation"], row["Complementarity Function"], 0.0)]
        current = all_lineups[(row["Formation"], row["Complementarity Function"], row["Gamma"])]
        base_names, current_names = set(baseline["Name"]), set(current["Name"])
        comparisons.append({
            "Formation": row["Formation"],
            "Complementarity Function": row["Complementarity Function"],
            "Gamma": row["Gamma"],
            "Players Changed": len(base_names - current_names),
            "Jaccard vs Gamma 0": len(base_names & current_names) / len(base_names | current_names),
            "Removed": "|".join(sorted(base_names - current_names)),
            "Added": "|".join(sorted(current_names - base_names)),
        })

    weights.to_csv(RESULTS_DIR / "formation_attribute_weights.csv", index=False)
    all_lineups_df.to_csv(RESULTS_DIR / "all_complementarity_lineups.csv", index=False)
    summary.to_csv(RESULTS_DIR / "complementarity_sensitivity_summary.csv", index=False)
    pd.DataFrame(comparisons).to_csv(
        RESULTS_DIR / "lineup_changes_vs_gamma_zero.csv", index=False
    )
    checks.to_csv(RESULTS_DIR / "gamma_zero_reproduction_check.csv", index=False)
    print(f"\nAll gamma=0 checks passed. Results saved to:\n{RESULTS_DIR}")


if __name__ == "__main__":
    main()
