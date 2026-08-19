"""Stage 5.1: scenario-based Monte Carlo simulation for the optimized squad.

This first simulation compares two strategies against a neutral reference
opponent with strength matched to our default XI:

1. No tactical changes: keep the default 4-3-3 for all 90 minutes.
2. Situation-aware: at halftime, use the legal Stage 4.4 leading, tied, or
   trailing lineup according to the simulated score.

The simulation is deliberately transparent. EAFC attributes are converted to
team attack and defense ratings, then a Poisson goal model simulates the first
and second halves. These are model-based scenarios—not real-world match
predictions. Later Stage 5 files can replace the reference opponent with Spain,
Real Madrid, Bayern Munich, Arsenal, PSG, and Manchester City.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STAGE_4_4_DIR = PROJECT_ROOT / "results" / "stage_4_4"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage_5_1"

STARTING_LINEUP_FILE = STAGE_4_4_DIR / "default_starting_4_3_3.csv"
ENDING_LINEUPS_FILE = STAGE_4_4_DIR / "scenario_ending_lineups.csv"
COMPLIANCE_FILE = STAGE_4_4_DIR / "rule_compliance_checks.csv"
GOALKEEPER_FILE = PROCESSED_DIR / "eafc26_goalkeepers_with_styles.csv"

STARTING_GOALKEEPER = "Alisson"
N_SIMULATIONS = 50_000
RANDOM_SEED = 42
DECISION_MINUTE = 45
STATE_RANDOM_SEED_OFFSETS = {
    "Leading": 100,
    "Tied": 200,
    "Trailing": 300,
}

# Average goals for an equally matched team over a full match. This value is a
# transparent calibration assumption and can later be estimated from match data.
BASE_EXPECTED_GOALS = 1.35
RATING_SCALE = 20.0


# Role weights reflect how strongly each position contributes to attacking and
# defensive phases. They are explicit modeling assumptions, not learned values.
ROLE_WEIGHTS = {
    "Center Back": (0.35, 1.50),
    "Full Back": (0.60, 1.20),
    "Defensive Midfielder": (0.55, 1.40),
    "Central Midfielder": (0.85, 1.00),
    "Attacking Midfielder": (1.15, 0.60),
    "Winger": (1.30, 0.50),
    "Wide Midfielder": (1.10, 0.70),
    "Striker": (1.40, 0.35),
}

# Formation effects prevent a whole-team average from treating structure as
# irrelevant. A 4-2-3-1 receives a defensive-structure bonus for its double
# pivot; a 4-4-2 gains attack but accepts more defensive exposure.
FORMATION_ADJUSTMENTS = {
    "4-3-3": {"Attack": 0.0, "Defense": 0.0},
    "4-2-3-1": {"Attack": -0.5, "Defense": 2.0},
    "4-4-2": {"Attack": 2.0, "Defense": -1.5},
}

REQUIRED_LINEUP_COLUMNS = {
    "Formation",
    "Assigned Role",
    "Name",
    "Selection Score",
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY",
}


def find_column(data, possible_names):
    """Return the first available column from a list of alternatives."""
    for column in possible_names:
        if column in data.columns:
            return column
    raise KeyError(f"None of these columns were found: {possible_names}")


def load_inputs():
    """Load Stage 4.4 lineups and stop if any plan failed its legality audit."""
    required_files = [
        STARTING_LINEUP_FILE,
        ENDING_LINEUPS_FILE,
        COMPLIANCE_FILE,
        GOALKEEPER_FILE,
    ]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Run the corrected Stage 4.4 script before Stage 5.1. Missing:\n"
            + "\n".join(missing_files)
        )

    starting = pd.read_csv(STARTING_LINEUP_FILE)
    endings = pd.read_csv(ENDING_LINEUPS_FILE)
    compliance = pd.read_csv(COMPLIANCE_FILE)
    goalkeepers = pd.read_csv(GOALKEEPER_FILE)

    for label, data in [("starting lineup", starting), ("ending lineups", endings)]:
        missing = sorted(REQUIRED_LINEUP_COLUMNS.difference(data.columns))
        if missing:
            raise ValueError(f"The {label} file is missing columns: {missing}")

    if "Scenario" not in endings.columns:
        raise ValueError("The ending-lineup file must contain a Scenario column.")

    legal_values = compliance["Lineup Legal"].astype(str).str.upper()
    if not legal_values.eq("YES").all():
        failed = compliance.loc[~legal_values.eq("YES"), "Scenario"].tolist()
        raise ValueError(
            "Stage 5.1 will not simulate illegal Stage 4.4 plans: "
            + ", ".join(failed)
        )

    expected_scenarios = {"Leading", "Tied", "Trailing"}
    actual_scenarios = set(endings["Scenario"].unique())
    if actual_scenarios != expected_scenarios:
        raise ValueError(
            f"Expected scenarios {sorted(expected_scenarios)}, found {sorted(actual_scenarios)}."
        )

    return starting, endings, compliance, goalkeepers


def goalkeeper_rating(goalkeepers):
    """Read Alisson's protection/skill rating from the goalkeeper dataset."""
    name_column = find_column(goalkeepers, ["Name", "Player Name", "LongName"])
    protection_column = find_column(
        goalkeepers,
        ["Protection Contribution", "ProtectionContribution", "Protection Score"],
    )
    goalkeeper_row = goalkeepers[goalkeepers[name_column].eq(STARTING_GOALKEEPER)]
    if goalkeeper_row.empty:
        raise ValueError(f"Could not find {STARTING_GOALKEEPER} in the goalkeeper file.")
    return float(goalkeeper_row.iloc[0][protection_column])


def calculate_team_ratings(lineup, goalkeeper_protection):
    """Convert one legal outfield lineup into role-weighted team ratings."""
    lineup = lineup.copy()
    unknown_roles = sorted(set(lineup["Assigned Role"]).difference(ROLE_WEIGHTS))
    if unknown_roles:
        raise ValueError(f"No simulation role weights exist for: {unknown_roles}")

    lineup["Player Attack"] = (
        0.30 * lineup["SHO"]
        + 0.25 * lineup["PAC"]
        + 0.25 * lineup["DRI"]
        + 0.20 * lineup["PAS"]
    )
    lineup["Player Defense"] = (
        0.55 * lineup["DEF"]
        + 0.25 * lineup["PHY"]
        + 0.10 * lineup["PAC"]
        + 0.10 * lineup["PAS"]
    )
    lineup["Attack Weight"] = lineup["Assigned Role"].map(
        lambda role: ROLE_WEIGHTS[role][0]
    )
    lineup["Defense Weight"] = lineup["Assigned Role"].map(
        lambda role: ROLE_WEIGHTS[role][1]
    )

    weighted_attack = np.average(
        lineup["Player Attack"], weights=lineup["Attack Weight"]
    )
    weighted_defense = np.average(
        lineup["Player Defense"], weights=lineup["Defense Weight"]
    )

    # Suitability rewards coherent assignments without overwhelming the raw
    # football attributes. Goalkeeper protection affects only team defense.
    average_suitability = lineup["Selection Score"].mean()
    attack = 0.85 * weighted_attack + 0.15 * average_suitability
    defense = 0.72 * weighted_defense + 0.18 * goalkeeper_protection + 0.10 * average_suitability

    formation = lineup["Formation"].iloc[0]
    adjustment = FORMATION_ADJUSTMENTS[formation]
    attack += adjustment["Attack"]
    defense += adjustment["Defense"]

    return {
        "Formation": formation,
        "Attack Rating": float(attack),
        "Defense Rating": float(defense),
        "Raw Weighted Attack": float(weighted_attack),
        "Raw Weighted Defense": float(weighted_defense),
        "Average Suitability": float(average_suitability),
        "Formation Attack Adjustment": adjustment["Attack"],
        "Formation Defense Adjustment": adjustment["Defense"],
    }


def expected_goals(attack_rating, opponent_defense, minute_fraction):
    """Convert a rating matchup and match fraction into Poisson expected goals."""
    full_match_rate = BASE_EXPECTED_GOALS * np.exp(
        (attack_rating - opponent_defense) / RATING_SCALE
    )
    return float(full_match_rate * minute_fraction)


def simulate_strategy(
    strategy_name,
    starting_ratings,
    scenario_ratings,
    opponent_attack,
    opponent_defense,
    rng,
):
    """Simulate one strategy for many matches using halftime as the decision point."""
    first_period_fraction = DECISION_MINUTE / 90.0
    second_period_fraction = 1.0 - first_period_fraction

    our_first_lambda = expected_goals(
        starting_ratings["Attack Rating"], opponent_defense, first_period_fraction
    )
    opponent_first_lambda = expected_goals(
        opponent_attack, starting_ratings["Defense Rating"], first_period_fraction
    )
    our_first_goals = rng.poisson(our_first_lambda, N_SIMULATIONS)
    opponent_first_goals = rng.poisson(opponent_first_lambda, N_SIMULATIONS)

    states = np.where(
        our_first_goals > opponent_first_goals,
        "Leading",
        np.where(our_first_goals < opponent_first_goals, "Trailing", "Tied"),
    )

    our_second_goals = np.zeros(N_SIMULATIONS, dtype=int)
    opponent_second_goals = np.zeros(N_SIMULATIONS, dtype=int)

    for state in ["Leading", "Tied", "Trailing"]:
        mask = states == state
        count = int(mask.sum())
        if strategy_name == "Situation-aware Stage 4.4":
            ratings = scenario_ratings[state]
        else:
            ratings = starting_ratings

        our_lambda = expected_goals(
            ratings["Attack Rating"], opponent_defense, second_period_fraction
        )
        opponent_lambda = expected_goals(
            opponent_attack, ratings["Defense Rating"], second_period_fraction
        )

        # Use the same state-specific random stream for both strategies. When
        # two strategies use identical ratings—as they do while tied—their
        # simulated outcomes are therefore exactly identical. For changed
        # ratings, common random numbers reduce comparison noise.
        state_rng = np.random.default_rng(
            RANDOM_SEED + STATE_RANDOM_SEED_OFFSETS[state]
        )
        our_second_goals[mask] = state_rng.poisson(our_lambda, count)
        opponent_second_goals[mask] = state_rng.poisson(opponent_lambda, count)

    our_total = our_first_goals + our_second_goals
    opponent_total = opponent_first_goals + opponent_second_goals
    wins = our_total > opponent_total
    draws = our_total == opponent_total
    losses = our_total < opponent_total

    win_rate = wins.mean()
    standard_error = np.sqrt(win_rate * (1.0 - win_rate) / N_SIMULATIONS)
    overall = {
        "Strategy": strategy_name,
        "Simulations": N_SIMULATIONS,
        "Win Rate": win_rate,
        "Draw Rate": draws.mean(),
        "Loss Rate": losses.mean(),
        "Average Goals For": our_total.mean(),
        "Average Goals Against": opponent_total.mean(),
        "Average Goal Difference": (our_total - opponent_total).mean(),
        "Win Rate 95% CI Lower": max(0.0, win_rate - 1.96 * standard_error),
        "Win Rate 95% CI Upper": min(1.0, win_rate + 1.96 * standard_error),
        "Leading at Decision": (states == "Leading").mean(),
        "Tied at Decision": (states == "Tied").mean(),
        "Trailing at Decision": (states == "Trailing").mean(),
    }

    state_rows = []
    for state in ["Leading", "Tied", "Trailing"]:
        mask = states == state
        state_count = int(mask.sum())
        state_wins = wins[mask]
        state_draws = draws[mask]
        state_losses = losses[mask]
        state_our_goals = our_total[mask]
        state_opponent_goals = opponent_total[mask]

        if state == "Leading":
            tactical_objective = "Preserve the lead"
            objective_success = state_wins.mean()
            success_label = "Lead Preserved as Win"
        elif state == "Tied":
            tactical_objective = "Convert the tied match into a win"
            objective_success = state_wins.mean()
            success_label = "Tie Converted to Win"
        else:
            tactical_objective = "Recover at least a draw"
            objective_success = (state_wins | state_draws).mean()
            success_label = "Defeat Avoided"

        state_rows.append({
            "Strategy": strategy_name,
            "State at Decision": state,
            "Tactical Objective": tactical_objective,
            "Matches in State": state_count,
            "Share of Simulations": state_count / N_SIMULATIONS,
            "Final Win Rate": state_wins.mean(),
            "Final Draw Rate": state_draws.mean(),
            "Final Loss Rate": state_losses.mean(),
            "Average Final Goals For": state_our_goals.mean(),
            "Average Final Goals Against": state_opponent_goals.mean(),
            "Average Final Goal Difference": (
                state_our_goals - state_opponent_goals
            ).mean(),
            "Objective Success Label": success_label,
            "Objective Success Rate": objective_success,
        })

    return overall, state_rows


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    starting, endings, compliance, goalkeepers = load_inputs()
    goalkeeper_protection = goalkeeper_rating(goalkeepers)

    starting_ratings = calculate_team_ratings(starting, goalkeeper_protection)
    scenario_ratings = {}
    rating_rows = [{"Scenario": "Default Starting XI", **starting_ratings}]
    for scenario in ["Leading", "Tied", "Trailing"]:
        lineup = endings[endings["Scenario"].eq(scenario)].copy()
        ratings = calculate_team_ratings(lineup, goalkeeper_protection)
        scenario_ratings[scenario] = ratings
        rating_rows.append({"Scenario": scenario, **ratings})

    ratings_table = pd.DataFrame(rating_rows)

    # The neutral reference opponent is strength-matched to our starting XI.
    # This isolates the effect of tactical strategy. It is not a named team.
    opponent_attack = starting_ratings["Attack Rating"]
    opponent_defense = starting_ratings["Defense Rating"]

    results = []
    state_results = []
    for strategy in [
        "No tactical changes",
        "Situation-aware Stage 4.4",
    ]:
        # Reusing the seed gives both strategies the same first-half
        # random match histories. This reduces noise in the strategy comparison.
        rng = np.random.default_rng(RANDOM_SEED)
        overall_result, strategy_state_rows = simulate_strategy(
            strategy,
            starting_ratings,
            scenario_ratings,
            opponent_attack,
            opponent_defense,
            rng,
        )
        results.append(overall_result)
        state_results.extend(strategy_state_rows)
    results_table = pd.DataFrame(results)
    state_results_table = pd.DataFrame(state_results)

    print("STAGE 5.1: MATCH SIMULATION ENGINE")
    print("==================================")
    print(f"Simulations per strategy: {N_SIMULATIONS:,}")
    print(f"Decision point: halftime ({DECISION_MINUTE} minutes)")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Base expected goals per equally matched team: {BASE_EXPECTED_GOALS:.2f}")
    print("Opponent: neutral reference opponent matched to the default XI")

    print("\nTEAM RATINGS")
    print("------------")
    print(ratings_table.round(2).to_string(index=False))

    display = results_table.copy()
    percentage_columns = [
        "Win Rate", "Draw Rate", "Loss Rate",
        "Win Rate 95% CI Lower", "Win Rate 95% CI Upper",
        "Leading at Decision", "Tied at Decision", "Trailing at Decision",
    ]
    display[percentage_columns] = (100 * display[percentage_columns]).round(2)
    numeric_columns = ["Average Goals For", "Average Goals Against", "Average Goal Difference"]
    display[numeric_columns] = display[numeric_columns].round(3)

    print("\nSIMULATION RESULTS (PERCENTAGES SHOWN AS 0-100)")
    print("------------------------------------------------")
    print(display.to_string(index=False))

    state_display = state_results_table.copy()
    state_percentage_columns = [
        "Share of Simulations",
        "Final Win Rate",
        "Final Draw Rate",
        "Final Loss Rate",
        "Objective Success Rate",
    ]
    state_display[state_percentage_columns] = (
        100 * state_display[state_percentage_columns]
    ).round(2)
    state_goal_columns = [
        "Average Final Goals For",
        "Average Final Goals Against",
        "Average Final Goal Difference",
    ]
    state_display[state_goal_columns] = state_display[state_goal_columns].round(3)

    print("\nRESULTS BY SCORE STATE AT HALFTIME")
    print("----------------------------------")
    print(state_display.to_string(index=False))

    # Directly compare adaptive and unchanged outcomes inside each score state.
    state_comparison_rows = []
    indexed_states = state_results_table.set_index(["Strategy", "State at Decision"])
    for state in ["Leading", "Tied", "Trailing"]:
        unchanged_state = indexed_states.loc[("No tactical changes", state)]
        adaptive_state = indexed_states.loc[("Situation-aware Stage 4.4", state)]
        state_comparison_rows.append({
            "State at Decision": state,
            "Adaptive Win-Rate Difference": (
                adaptive_state["Final Win Rate"] - unchanged_state["Final Win Rate"]
            ),
            "Adaptive Draw-Rate Difference": (
                adaptive_state["Final Draw Rate"] - unchanged_state["Final Draw Rate"]
            ),
            "Adaptive Loss-Rate Difference": (
                adaptive_state["Final Loss Rate"] - unchanged_state["Final Loss Rate"]
            ),
            "Adaptive Goal-Difference Change": (
                adaptive_state["Average Final Goal Difference"]
                - unchanged_state["Average Final Goal Difference"]
            ),
            "Adaptive Objective-Success Difference": (
                adaptive_state["Objective Success Rate"]
                - unchanged_state["Objective Success Rate"]
            ),
        })
    state_comparison = pd.DataFrame(state_comparison_rows)
    comparison_display = state_comparison.copy()
    comparison_percentage_columns = [
        "Adaptive Win-Rate Difference",
        "Adaptive Draw-Rate Difference",
        "Adaptive Loss-Rate Difference",
        "Adaptive Objective-Success Difference",
    ]
    comparison_display[comparison_percentage_columns] = (
        100 * comparison_display[comparison_percentage_columns]
    ).round(2)
    comparison_display["Adaptive Goal-Difference Change"] = (
        comparison_display["Adaptive Goal-Difference Change"].round(3)
    )

    print("\nADAPTIVE MINUS UNCHANGED, BY HALFTIME STATE")
    print("---------------------------------------------")
    print(comparison_display.to_string(index=False))

    adaptive = results_table.set_index("Strategy").loc["Situation-aware Stage 4.4"]
    unchanged = results_table.set_index("Strategy").loc["No tactical changes"]
    print("\nSTRATEGY DIFFERENCE")
    print("-------------------")
    print(
        "Adaptive minus unchanged win rate: "
        f"{100 * (adaptive['Win Rate'] - unchanged['Win Rate']):+.2f} percentage points"
    )
    print(
        "Adaptive minus unchanged goal difference: "
        f"{adaptive['Average Goal Difference'] - unchanged['Average Goal Difference']:+.3f} goals per match"
    )

    ratings_table.to_csv(RESULTS_DIR / "lineup_simulation_ratings.csv", index=False)
    results_table.to_csv(RESULTS_DIR / "strategy_simulation_results.csv", index=False)
    state_results_table.to_csv(
        RESULTS_DIR / "simulation_results_by_halftime_state.csv", index=False
    )
    state_comparison.to_csv(
        RESULTS_DIR / "adaptive_strategy_difference_by_state.csv", index=False
    )
    compliance.to_csv(RESULTS_DIR / "stage4_4_compliance_input.csv", index=False)

    assumptions = pd.DataFrame([
        {"Assumption": "Simulation count", "Value": N_SIMULATIONS},
        {"Assumption": "Random seed", "Value": RANDOM_SEED},
        {"Assumption": "Decision minute", "Value": DECISION_MINUTE},
        {"Assumption": "Base expected goals", "Value": BASE_EXPECTED_GOALS},
        {"Assumption": "Rating scale", "Value": RATING_SCALE},
        {"Assumption": "Opponent", "Value": "Reference opponent matched to default XI"},
        {"Assumption": "Goal model", "Value": "Independent Poisson scoring in the first and second halves"},
    ])
    assumptions.to_csv(RESULTS_DIR / "simulation_assumptions.csv", index=False)

    print(f"\nStage 5.1 results saved to:\n{RESULTS_DIR}")
    print("\nInterpret results as simulated performance under stated assumptions,")
    print("not as real-world predicted win probabilities.")


if __name__ == "__main__":
    main()