"""Stage 5.3: compare our optimized team with Real Madrid's 2026-27 squad.

The current Real Madrid roster is matched to EAFC26, then the strongest
balanced 4-3-3 is selected under the same role, wide-attacker, goalkeeper, and
simulation rules used for Spain. Real Madrid remains tactically fixed in this
first benchmark so the value of our own halftime strategy can be isolated.

The club's official player page describes the current list as provisional.
Accordingly, this script labels it a provisional roster and reports every
unmatched name rather than treating the roster as permanently final.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from archive.stage4_4_starting_xi_substitution_plans import optimize_lineup
from stage5_1_match_simulation_engine import (
    N_SIMULATIONS,
    RANDOM_SEED,
    calculate_team_ratings,
    simulate_strategy,
)
from stage5_2_spain_2026_comparison import (
    build_name_index,
    format_percentages,
    load_our_ratings,
    normalize_name,
    prepare_spain_role_data,
    select_spain_goalkeeper,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage_5_3_real_madrid"

ROLE_FILE = PROCESSED_DIR / "eafc26_player_role_suitability.csv"
FEATURE_FILE = PROCESSED_DIR / "eafc26_outfield_full_features.csv"
GOALKEEPER_FILE = PROCESSED_DIR / "eafc26_goalkeepers_with_styles.csv"

REAL_MADRID_FORMATION = "4-3-3"


# Current 2026-27 provisional first-team group. Candidate aliases resolve
# official display names against common EAFC spellings.
REAL_MADRID_ROSTER_ALIASES = {
    # Goalkeepers
    "Thibaut Courtois": ["Thibaut Courtois", "Courtois"],
    "Andriy Lunin": ["Andriy Lunin", "Lunin"],
    "Sergio Mestre": ["Sergio Mestre"],

    # Defenders
    "Raúl Asencio": ["Raúl Asencio", "Raul Asencio"],
    "Éder Militão": ["Éder Militão", "Eder Militão", "Éder Militão"],
    "Dean Huijsen": ["Dean Huijsen", "Huijsen"],
    "Trent Alexander-Arnold": ["Trent Alexander-Arnold", "Trent"],
    "Ibrahima Konaté": ["Ibrahima Konaté", "Ibrahima Konate"],
    "Marc Cucurella": ["Marc Cucurella", "Cucurella"],
    "Álvaro Carreras": ["Álvaro Carreras", "Alvaro Carreras", "Á. Carreras"],
    "Antonio Rüdiger": ["Antonio Rüdiger", "Antonio Rudiger", "Rüdiger"],
    "Ferland Mendy": ["Ferland Mendy", "F. Mendy"],
    "Denzel Dumfries": ["Denzel Dumfries", "Dumfries"],

    # Midfielders
    "Jude Bellingham": ["Jude Bellingham", "Bellingham"],
    "Eduardo Camavinga": ["Eduardo Camavinga", "Camavinga"],
    "Federico Valverde": ["Federico Valverde", "Valverde"],
    "Aurélien Tchouaméni": ["Aurélien Tchouaméni", "Aurelien Tchouameni", "Tchouaméni"],
    "Arda Güler": ["Arda Güler", "Arda Guler"],
    "Bernardo Silva": ["Bernardo Silva"],
    "Thiago Pitarch": ["Thiago Pitarch"],

    # Forwards
    "Vini Jr.": ["Vini Jr.", "Vini Jr", "Vinícius Júnior"],
    "Endrick": ["Endrick"],
    "Kylian Mbappé": ["Kylian Mbappé", "Kylian Mbappe", "Mbappé"],
    "Rodrygo": ["Rodrygo"],
    "Brahim Díaz": ["Brahim Díaz", "Brahim Diaz", "Brahim"],
    "Carlos Espí": ["Carlos Espí", "Carlos Espi", "C. Espí"],
    "Yan Diomande": ["Yan Diomande", "Diomande"],
}

REAL_MADRID_GOALKEEPERS = {
    "Thibaut Courtois", "Andriy Lunin", "Sergio Mestre"
}


def resolve_real_madrid_roster(role_data, goalkeeper_data):
    """Match provisional roster names to exact EAFC display names."""
    goalkeeper_name_column = next(
        column for column in ["Name", "Player Name", "LongName"]
        if column in goalkeeper_data.columns
    )
    all_names = set(role_data["Name"].dropna()).union(
        goalkeeper_data[goalkeeper_name_column].dropna()
    )
    name_index = build_name_index(all_names)

    rows = []
    for official_name, candidates in REAL_MADRID_ROSTER_ALIASES.items():
        matches = []
        for candidate in candidates:
            matches.extend(name_index.get(normalize_name(candidate), []))
        matches = list(dict.fromkeys(matches))
        if len(matches) == 1:
            status = "Matched"
            eafc_name = matches[0]
        elif len(matches) == 0:
            status = "Missing from EAFC"
            eafc_name = pd.NA
        else:
            status = "Ambiguous"
            eafc_name = ", ".join(matches)
        rows.append({
            "Official Roster Name": official_name,
            "EAFC Name": eafc_name,
            "Roster Position Group": (
                "Goalkeeper" if official_name in REAL_MADRID_GOALKEEPERS
                else "Outfield"
            ),
            "Match Status": status,
        })
    return pd.DataFrame(rows), goalkeeper_name_column


def create_role_coverage(real_madrid_roles):
    """Count distinct eligible players for every modeled functional role."""
    coverage = (
        real_madrid_roles.groupby("Evaluated Role")["Name"]
        .nunique()
        .rename("Eligible Players")
        .reset_index()
        .sort_values(["Eligible Players", "Evaluated Role"])
    )
    return coverage


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    required_files = [ROLE_FILE, FEATURE_FILE, GOALKEEPER_FILE]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    role_data = pd.read_csv(ROLE_FILE)
    feature_data = pd.read_csv(FEATURE_FILE)
    goalkeeper_data = pd.read_csv(GOALKEEPER_FILE)

    roster_matches, goalkeeper_name_column = resolve_real_madrid_roster(
        role_data, goalkeeper_data
    )
    unresolved = roster_matches[~roster_matches["Match Status"].eq("Matched")]

    print("STAGE 5.3: REAL MADRID 2026-27 COMPARISON")
    print("==========================================")
    print("Roster status: provisional")
    print(f"Provisional roster size: {len(roster_matches)}")
    print(f"EAFC-matched players: {(roster_matches['Match Status'] == 'Matched').sum()}")
    if not unresolved.empty:
        print("\nUNRESOLVED ROSTER NAMES")
        print(unresolved.to_string(index=False))

    # This generic preparation helper merges raw EAFC attributes and applies
    # the corrected Winger/Wide Midfielder position-fit rule from Stage 5.2.
    real_madrid_roles = prepare_spain_role_data(
        role_data, feature_data, roster_matches
    )
    role_coverage = create_role_coverage(real_madrid_roles)

    real_madrid_lineup = optimize_lineup(
        real_madrid_roles,
        REAL_MADRID_FORMATION,
        "Balanced Score",
    )
    goalkeeper, goalkeeper_protection, goalkeeper_candidates = (
        select_spain_goalkeeper(
            goalkeeper_data, roster_matches, goalkeeper_name_column
        )
    )
    real_madrid_ratings = calculate_team_ratings(
        real_madrid_lineup, goalkeeper_protection
    )
    _, _, our_starting_ratings, our_scenario_ratings = load_our_ratings(
        goalkeeper_data
    )

    print("\nREAL MADRID MODEL-SELECTED STARTING XI")
    print("--------------------------------------")
    print(
        "Wide-attacker rule: an existing Winger profile is required; dual "
        "Winger/Wide Midfielder players may use the stronger position fit."
    )
    print(f"GK  Goalkeeper              {goalkeeper}")
    print(real_madrid_lineup[[
        "Formation Slot", "Assigned Role", "Name", "Official Position",
        "Role Source", "OVR", "Selection Score", "Tactical Score",
    ]].round(2).to_string(index=False))

    print("\nREAL MADRID ROLE COVERAGE")
    print("-------------------------")
    print(role_coverage.to_string(index=False))

    rating_comparison = pd.DataFrame([
        {"Team": "Our optimized team", **our_starting_ratings},
        {"Team": "Real Madrid 2026-27", **real_madrid_ratings},
    ])
    print("\nSTARTING-LINEUP RATING COMPARISON")
    print("---------------------------------")
    print(rating_comparison.round(2).to_string(index=False))

    overall_results = []
    state_results = []
    for strategy in ["No tactical changes", "Situation-aware Stage 4.4"]:
        rng = np.random.default_rng(RANDOM_SEED)
        overall, states = simulate_strategy(
            strategy,
            our_starting_ratings,
            our_scenario_ratings,
            real_madrid_ratings["Attack Rating"],
            real_madrid_ratings["Defense Rating"],
            rng,
        )
        overall["Opponent"] = "Real Madrid 2026-27"
        for state in states:
            state["Opponent"] = "Real Madrid 2026-27"
        overall_results.append(overall)
        state_results.extend(states)

    overall_table = pd.DataFrame(overall_results)
    states_table = pd.DataFrame(state_results)
    display = format_percentages(overall_table, [
        "Win Rate", "Draw Rate", "Loss Rate",
        "Leading at Decision", "Tied at Decision", "Trailing at Decision",
    ])
    print("\nSIMULATION AGAINST FIXED REAL MADRID LINEUP")
    print("---------------------------------------------")
    print(display[[
        "Strategy", "Win Rate", "Draw Rate", "Loss Rate",
        "Average Goals For", "Average Goals Against", "Average Goal Difference",
        "Leading at Decision", "Tied at Decision", "Trailing at Decision",
    ]].round(3).to_string(index=False))

    print("\nLIMITATIONS")
    print("Real Madrid remains tactically fixed in this benchmark.")
    print("The roster is provisional, and unmatched players receive no invented scores.")
    print("Results describe the EAFC/Poisson model, not real match probabilities.")

    roster_matches.to_csv(
        RESULTS_DIR / "real_madrid_roster_name_matches.csv", index=False
    )
    real_madrid_lineup.to_csv(
        RESULTS_DIR / "real_madrid_model_selected_4_3_3.csv", index=False
    )
    role_coverage.to_csv(
        RESULTS_DIR / "real_madrid_role_coverage.csv", index=False
    )
    rating_comparison.to_csv(
        RESULTS_DIR / "our_team_vs_real_madrid_ratings.csv", index=False
    )
    overall_table.to_csv(
        RESULTS_DIR / "our_team_vs_real_madrid_simulation.csv", index=False
    )
    states_table.to_csv(
        RESULTS_DIR / "our_team_vs_real_madrid_by_halftime_state.csv", index=False
    )
    goalkeeper_candidates.to_csv(
        RESULTS_DIR / "real_madrid_goalkeeper_candidates.csv", index=False
    )

    print(f"\nStage 5.3 results saved to:\n{RESULTS_DIR}")
    print(f"Simulations per strategy: {N_SIMULATIONS:,}")


if __name__ == "__main__":
    main()