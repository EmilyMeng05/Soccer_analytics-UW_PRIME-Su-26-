"""Stage 5.2: compare our optimized squad with 2026 World Cup winner Spain.

Spain's official 26-player tournament roster is matched to EAFC26. The script
then builds Spain's strongest balanced 4-3-3 with the same role-suitability and
assignment rules used for our team. Finally, it simulates our unchanged and
situation-aware strategies against Spain's fixed lineup.

Important: this is a model comparison, not a recreation of the World Cup final.
Spain is held tactically fixed in this first benchmark because we have not yet
modeled Spain-specific halftime substitutions. That limitation is printed and
saved with the results.
"""

from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

from archive.stage4_4_starting_xi_substitution_plans import add_tactical_scores, optimize_lineup
from stage5_1_match_simulation_engine import (
    N_SIMULATIONS,
    RANDOM_SEED,
    calculate_team_ratings,
    simulate_strategy,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STAGE_4_4_DIR = PROJECT_ROOT / "results" / "stage_4_4"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage_5_2_spain"

ROLE_FILE = PROCESSED_DIR / "eafc26_player_role_suitability.csv"
FEATURE_FILE = PROCESSED_DIR / "eafc26_outfield_full_features.csv"
GOALKEEPER_FILE = PROCESSED_DIR / "eafc26_goalkeepers_with_styles.csv"
OUR_STARTING_FILE = STAGE_4_4_DIR / "default_starting_4_3_3.csv"
OUR_ENDINGS_FILE = STAGE_4_4_DIR / "scenario_ending_lineups.csv"
OUR_COMPLIANCE_FILE = STAGE_4_4_DIR / "rule_compliance_checks.csv"

EAFC_ATTRIBUTES = ["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]
SPAIN_FORMATION = "4-3-3"


# Official RFEF 2026 World Cup squad. Candidate names handle differences
# between federation display names and EAFC naming conventions.
SPAIN_ROSTER_ALIASES = {
    "David Raya": ["David Raya"],
    "Marc Pubill": ["Marc Pubill"],
    "Álex Grimaldo": ["Grimaldo", "Álex Grimaldo", "Alejandro Grimaldo"],
    "Eric García": ["Eric García", "Eric Garcia"],
    "Marcos Llorente": ["Marcos Llorente"],
    "Mikel Merino": ["Mikel Merino"],
    "Ferran Torres": ["Ferran Torres"],
    "Fabián Ruiz": ["Fabián", "Fabián Ruiz", "Fabian Ruiz"],
    "Gavi": ["Gavi"],
    "Dani Olmo": ["Dani Olmo"],
    "Yeremy Pino": ["Yeremy Pino", "Yeremy"],
    "Pedro Porro": ["Pedro Porro"],
    "Joan García": ["Joan García", "Joan Garcia"],
    "Aymeric Laporte": ["Aymeric Laporte", "Laporte"],
    "Álex Baena": ["Álex Baena", "Alex Baena"],
    "Rodri": ["Rodri"],
    "Nico Williams": ["Nico Williams", "Williams Jr."],
    "Martín Zubimendi": ["Martín Zubimendi", "Martin Zubimendi", "Zubimendi"],
    "Lamine Yamal": ["Lamine Yamal"],
    "Pedri": ["Pedri"],
    "Mikel Oyarzabal": ["Mikel Oyarzabal", "Oyarzabal"],
    "Pau Cubarsí": ["Pau Cubarsí", "Pau Cubarsi", "Cubarsí"],
    "Unai Simón": ["Unai Simón", "Unai Simon"],
    "Marc Cucurella": ["Marc Cucurella", "Cucurella"],
    "Víctor Muñoz": ["Víctor Muñoz", "Victor Muñoz"],
    "Borja Iglesias": ["Borja Iglesias"],
}

SPAIN_GOALKEEPERS = {"David Raya", "Joan García", "Unai Simón"}


def normalize_name(value):
    """Normalize accents, spacing, punctuation, and capitalization for matching."""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(character for character in value if not unicodedata.combining(character))
    return "".join(character.lower() for character in value if character.isalnum())


def build_name_index(names):
    """Map a normalized name to every corresponding EAFC display name."""
    index = {}
    for name in pd.Series(list(names)).dropna().drop_duplicates():
        index.setdefault(normalize_name(name), []).append(name)
    return index


def resolve_spain_roster(role_data, goalkeeper_data):
    """Resolve official roster names without silently inventing missing players."""
    goalkeeper_name_column = next(
        column for column in ["Name", "Player Name", "LongName"]
        if column in goalkeeper_data.columns
    )
    all_names = set(role_data["Name"].dropna()).union(
        goalkeeper_data[goalkeeper_name_column].dropna()
    )
    name_index = build_name_index(all_names)

    rows = []
    for official_name, candidates in SPAIN_ROSTER_ALIASES.items():
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
                "Goalkeeper" if official_name in SPAIN_GOALKEEPERS else "Outfield"
            ),
            "Match Status": status,
        })
    return pd.DataFrame(rows), goalkeeper_name_column


def prepare_spain_role_data(role_data, feature_data, roster_matches):
    """Attach real EAFC attributes to all matched Spanish outfield role rows."""
    matched_names = roster_matches.loc[
        roster_matches["Roster Position Group"].eq("Outfield")
        & roster_matches["Match Status"].eq("Matched"),
        "EAFC Name",
    ].tolist()
    spain_roles = role_data[role_data["Name"].isin(matched_names)].copy()
    spain_roles = (
        spain_roles.sort_values("Player Role Suitability", ascending=False)
        .drop_duplicates(["Name", "Evaluated Role"])
    )

    if "OVR" in feature_data.columns:
        feature_data = feature_data.sort_values("OVR", ascending=False)
    features = feature_data.drop_duplicates("Name")[["Name", *EAFC_ATTRIBUTES]]
    existing_attributes = [column for column in EAFC_ATTRIBUTES if column in spain_roles.columns]
    if existing_attributes:
        spain_roles = spain_roles.drop(columns=existing_attributes)
    spain_roles = spain_roles.merge(features, on="Name", how="left", validate="many_to_one")

    missing_attributes = spain_roles.loc[
        spain_roles[EAFC_ATTRIBUTES].isna().any(axis=1), "Name"
    ].unique()
    if len(missing_attributes):
        raise ValueError(
            "Missing EAFC attributes for matched Spain players: "
            + ", ".join(sorted(missing_attributes))
        )
    # Correct only the position-label penalty for players who already possess
    # BOTH a modeled Winger row and a Wide Midfielder row. The Winger-specific
    # Role Attribute Score is retained. This helps a genuine wide attacker such
    # as Lamine Yamal without converting every wide midfielder (for example,
    # Grimaldo) into a full-score front-three candidate.
    wide_midfielder_fit = (
        spain_roles[spain_roles["Evaluated Role"].eq("Wide Midfielder")]
        .groupby("Name")["Position Fit Score"]
        .max()
    )
    winger_mask = spain_roles["Evaluated Role"].eq("Winger")
    dual_role_mask = winger_mask & spain_roles["Name"].isin(wide_midfielder_fit.index)
    for row_index in spain_roles.index[dual_role_mask]:
        player = spain_roles.at[row_index, "Name"]
        accepted_fit = max(
            float(spain_roles.at[row_index, "Position Fit Score"]),
            float(wide_midfielder_fit.loc[player]),
        )
        spain_roles.at[row_index, "Position Fit Score"] = accepted_fit
        spain_roles.at[row_index, "Player Role Suitability"] = (
            0.40 * float(spain_roles.at[row_index, "Role Attribute Score"])
            + 0.35 * float(spain_roles.at[row_index, "OVR"])
            + 0.25 * accepted_fit
        )
        spain_roles.at[row_index, "Role Source"] = (
            "Winger Attributes + Wide-Midfielder Position Fit"
        )

    return add_tactical_scores(spain_roles)


def select_spain_goalkeeper(goalkeeper_data, roster_matches, name_column):
    """Select Spain's strongest modeled protection goalkeeper from its roster."""
    protection_column = next(
        column for column in [
            "Protection Contribution", "ProtectionContribution", "Protection Score"
        ] if column in goalkeeper_data.columns
    )
    matched_goalkeepers = roster_matches.loc[
        roster_matches["Roster Position Group"].eq("Goalkeeper")
        & roster_matches["Match Status"].eq("Matched"),
        "EAFC Name",
    ].tolist()
    candidates = goalkeeper_data[goalkeeper_data[name_column].isin(matched_goalkeepers)].copy()
    candidates = candidates.sort_values(protection_column, ascending=False).drop_duplicates(name_column)
    if candidates.empty:
        raise ValueError("No Spain goalkeeper could be matched to the EAFC goalkeeper file.")
    selected = candidates.iloc[0]
    return selected[name_column], float(selected[protection_column]), candidates


def load_our_ratings(goalkeeper_data):
    """Rebuild our Stage 4.4 lineup ratings for the Spain simulation."""
    starting = pd.read_csv(OUR_STARTING_FILE)
    endings = pd.read_csv(OUR_ENDINGS_FILE)
    compliance = pd.read_csv(OUR_COMPLIANCE_FILE)
    if not compliance["Lineup Legal"].astype(str).str.upper().eq("YES").all():
        raise ValueError("At least one Stage 4.4 tactical plan is illegal.")

    name_column = next(
        column for column in ["Name", "Player Name", "LongName"]
        if column in goalkeeper_data.columns
    )
    protection_column = next(
        column for column in [
            "Protection Contribution", "ProtectionContribution", "Protection Score"
        ] if column in goalkeeper_data.columns
    )
    alisson = goalkeeper_data[goalkeeper_data[name_column].eq("Alisson")]
    if alisson.empty:
        raise ValueError("Alisson was not found in the goalkeeper dataset.")
    protection = float(alisson.iloc[0][protection_column])

    starting_ratings = calculate_team_ratings(starting, protection)
    scenario_ratings = {
        scenario: calculate_team_ratings(
            endings[endings["Scenario"].eq(scenario)].copy(), protection
        )
        for scenario in ["Leading", "Tied", "Trailing"]
    }
    return starting, endings, starting_ratings, scenario_ratings


def format_percentages(data, columns):
    """Convert probability columns to readable percentages for printing."""
    output = data.copy()
    output[columns] = (100 * output[columns]).round(2)
    return output


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    required_files = [
        ROLE_FILE, FEATURE_FILE, GOALKEEPER_FILE,
        OUR_STARTING_FILE, OUR_ENDINGS_FILE, OUR_COMPLIANCE_FILE,
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

    role_data = pd.read_csv(ROLE_FILE)
    feature_data = pd.read_csv(FEATURE_FILE)
    goalkeeper_data = pd.read_csv(GOALKEEPER_FILE)

    roster_matches, goalkeeper_name_column = resolve_spain_roster(
        role_data, goalkeeper_data
    )
    unresolved = roster_matches[~roster_matches["Match Status"].eq("Matched")]
    print("STAGE 5.2: SPAIN 2026 WORLD CUP CHAMPION COMPARISON")
    print("====================================================")
    print(f"Official Spain roster size: {len(roster_matches)}")
    print(f"EAFC-matched players: {(roster_matches['Match Status'] == 'Matched').sum()}")
    if not unresolved.empty:
        print("\nUNRESOLVED ROSTER NAMES")
        print(unresolved.to_string(index=False))

    spain_roles = prepare_spain_role_data(role_data, feature_data, roster_matches)
    spain_lineup = optimize_lineup(
        spain_roles,
        SPAIN_FORMATION,
        "Balanced Score",
    )
    spain_goalkeeper, spain_goalkeeper_protection, goalkeeper_candidates = (
        select_spain_goalkeeper(
            goalkeeper_data, roster_matches, goalkeeper_name_column
        )
    )
    spain_ratings = calculate_team_ratings(
        spain_lineup, spain_goalkeeper_protection
    )

    our_starting, our_endings, our_starting_ratings, our_scenario_ratings = (
        load_our_ratings(goalkeeper_data)
    )

    print("\nSPAIN MODEL-SELECTED STARTING XI")
    print("--------------------------------")
    print(
        "Wide-attacker rule: players must have a modeled Winger row; "
        "dual Winger/Wide Midfielder players may use the stronger position fit."
    )
    print(f"GK  Goalkeeper              {spain_goalkeeper}")
    print(spain_lineup[[
        "Formation Slot", "Assigned Role", "Name", "Official Position",
        "Role Source", "OVR", "Selection Score", "Tactical Score",
    ]].round(2).to_string(index=False))

    rating_comparison = pd.DataFrame([
        {"Team": "Our optimized team", **our_starting_ratings},
        {"Team": "Spain 2026", **spain_ratings},
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
            spain_ratings["Attack Rating"],
            spain_ratings["Defense Rating"],
            rng,
        )
        overall["Opponent"] = "Spain 2026"
        for state in states:
            state["Opponent"] = "Spain 2026"
        overall_results.append(overall)
        state_results.extend(states)

    overall_table = pd.DataFrame(overall_results)
    states_table = pd.DataFrame(state_results)
    display = format_percentages(overall_table, [
        "Win Rate", "Draw Rate", "Loss Rate",
        "Leading at Decision", "Tied at Decision", "Trailing at Decision",
    ])
    print("\nSIMULATION AGAINST FIXED SPAIN LINEUP")
    print("-------------------------------------")
    print(display[[
        "Strategy", "Win Rate", "Draw Rate", "Loss Rate",
        "Average Goals For", "Average Goals Against", "Average Goal Difference",
        "Leading at Decision", "Tied at Decision", "Trailing at Decision",
    ]].round(3).to_string(index=False))

    print("\nLIMITATION")
    print("Spain remains tactically fixed in this first benchmark. The simulation")
    print("does not claim Spain's real coaching staff would make no halftime changes.")

    roster_matches.to_csv(RESULTS_DIR / "spain_2026_roster_name_matches.csv", index=False)
    spain_lineup.to_csv(RESULTS_DIR / "spain_2026_model_selected_4_3_3.csv", index=False)
    rating_comparison.to_csv(RESULTS_DIR / "our_team_vs_spain_ratings.csv", index=False)
    overall_table.to_csv(RESULTS_DIR / "our_team_vs_spain_simulation.csv", index=False)
    states_table.to_csv(
        RESULTS_DIR / "our_team_vs_spain_simulation_by_halftime_state.csv",
        index=False,
    )
    goalkeeper_candidates.to_csv(
        RESULTS_DIR / "spain_goalkeeper_candidates.csv", index=False
    )
    pd.DataFrame([{
        "Limitation": (
            "Spain is held tactically fixed; results are scenario outputs under "
            "EAFC-based rating and Poisson-model assumptions, not real predictions."
        )
    }]).to_csv(RESULTS_DIR / "comparison_limitations.csv", index=False)

    print(f"\nStage 5.2 results saved to:\n{RESULTS_DIR}")
    print(f"Simulations per strategy: {N_SIMULATIONS:,}")


if __name__ == "__main__":
    main()