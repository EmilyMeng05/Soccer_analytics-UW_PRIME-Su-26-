"""Stage 4.4: choose a default XI and create legal substitution plans.

This stage uses only the fixed 23-player squad selected in Stage 4.3. It does
not simulate match outcomes yet. Instead, it creates one balanced starting XI
and three post-kickoff tactical plans for when the team is leading, tied, or
trailing.

Rules enforced by this script:
    * exactly 11 players are on the field, including exactly one goalkeeper;
    * Alisson remains the goalkeeper during ordinary tactical changes;
    * at most five players are substituted;
    * all planned tactical changes are made together at halftime;
    * a substituted player does not return;
    * only members of the Stage 4.3 squad may play;
    * every required formation role is filled by one unique eligible player.

The scenario weights are transparent modeling assumptions. They are not
learned from match-event, fatigue, injury, or opponent data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


# ---------------------------------------------------------------------------
# Project paths and main settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage_4_4"

PLAYER_ROLE_FILE = PROCESSED_DIR / "eafc26_player_role_suitability.csv"
PLAYER_FEATURE_FILE = PROCESSED_DIR / "eafc26_outfield_full_features.csv"
EAFC_ATTRIBUTES = ["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]

# The first path is the corrected Stage 4.3 filename. The second is retained
# only so the error message can explain when an older local file was found.
SQUAD_FILE_CANDIDATES = [
    PROCESSED_DIR / "eafc26_fixed_flexible_23_player_squad.csv",
    PROCESSED_DIR / "eafc26_fixed_flexible_18_player_squad.csv",
]

STARTING_GOALKEEPER = "Alisson"
DEFAULT_FORMATION = "4-3-3"
MAX_SUBSTITUTIONS = 5
MAX_SUBSTITUTION_WINDOWS = 3
TACTICAL_DECISION_TIME = "Halftime (after 45 minutes)"


# The formations contain ten outfield slots. The goalkeeper is added later.
FORMATIONS = {
    "4-3-3": [
        ("CB1", "Center Back"),
        ("CB2", "Center Back"),
        ("FB1", "Full Back"),
        ("FB2", "Full Back"),
        ("DM", "Defensive Midfielder"),
        ("CM1", "Central Midfielder"),
        ("CM2", "Central Midfielder"),
        ("W1", "Winger"),
        ("W2", "Winger"),
        ("ST", "Striker"),
    ],
    "4-2-3-1": [
        ("CB1", "Center Back"),
        ("CB2", "Center Back"),
        ("FB1", "Full Back"),
        ("FB2", "Full Back"),
        ("DM1", "Defensive Midfielder"),
        ("DM2", "Defensive Midfielder"),
        ("AM", "Attacking Midfielder"),
        ("W1", "Winger"),
        ("W2", "Winger"),
        ("ST", "Striker"),
    ],
    "4-4-2": [
        ("CB1", "Center Back"),
        ("CB2", "Center Back"),
        ("FB1", "Full Back"),
        ("FB2", "Full Back"),
        ("CM1", "Central Midfielder"),
        ("CM2", "Central Midfielder"),
        ("WM1", "Wide Midfielder"),
        ("WM2", "Wide Midfielder"),
        ("ST1", "Striker"),
        ("ST2", "Striker"),
    ],
}


# Each situation has a chosen ending shape and a transparent scoring formula.
# The weights sum to one. Suitability protects positional fit; the remaining
# terms express the tactical objective for that situation.
SCENARIOS = {
    "Leading": {
        "Ending Formation": "4-2-3-1",
        "Description": "Protect the lead through defense, physical strength, and possession.",
        "Maximum Tactical Substitutions": 2,
        "Substitution Penalty": 1.50,
        "Weights": {
            "Suitability": 0.55,
            "Defense": 0.20,
            "Physical": 0.10,
            "Passing": 0.10,
            "OVR": 0.05,
        },
    },
    "Tied": {
        "Ending Formation": "4-3-3",
        "Description": "Remain balanced while improving the strongest available XI.",
        "Maximum Tactical Substitutions": 2,
        "Substitution Penalty": 2.00,
        "Weights": {
            "Suitability": 0.70,
            "Attack": 0.10,
            "Defense": 0.10,
            "OVR": 0.10,
        },
    },
    "Trailing": {
        "Ending Formation": "4-4-2",
        "Description": "Add a second striker and emphasize pace, shooting, passing, and dribbling.",
        "Maximum Tactical Substitutions": 3,
        "Substitution Penalty": 1.00,
        "Weights": {
            "Suitability": 0.50,
            "Attack": 0.35,
            "OVR": 0.15,
        },
    },
}


REQUIRED_ROLE_COLUMNS = {
    "Name",
    "Position",
    "OVR",
    "Evaluated Role",
    "Role Source",
    "Position Fit Score",
    "Role Attribute Score",
    "Player Role Suitability",
}


def locate_squad_file():
    """Find the corrected Stage 4.3 squad file and reject an 18-player input."""
    for path in SQUAD_FILE_CANDIDATES:
        if path.exists():
            return path
    expected = "\n".join(str(path) for path in SQUAD_FILE_CANDIDATES)
    raise FileNotFoundError(
        "Could not find a Stage 4.3 squad file. Checked:\n" + expected
    )


def load_and_validate_data():
    """Load the fixed squad and role scores, then verify Stage 4.4 inputs."""
    squad_file = locate_squad_file()
    if not PLAYER_ROLE_FILE.exists():
        raise FileNotFoundError(f"Could not find player-role data: {PLAYER_ROLE_FILE}")

    squad = pd.read_csv(squad_file)
    role_data = pd.read_csv(PLAYER_ROLE_FILE)

    if "Name" not in squad.columns or "Squad Group" not in squad.columns:
        raise ValueError("The Stage 4.3 squad file must contain Name and Squad Group.")

    missing = sorted(REQUIRED_ROLE_COLUMNS.difference(role_data.columns))
    if missing:
        raise ValueError(f"The player-role file is missing columns: {missing}")

    squad_names = squad["Name"].dropna().drop_duplicates().tolist()
    goalkeeper_names = squad.loc[
        squad["Squad Group"].eq("Goalkeeper"), "Name"
    ].dropna().drop_duplicates().tolist()
    outfield_names = squad.loc[
        squad["Squad Group"].eq("Outfield"), "Name"
    ].dropna().drop_duplicates().tolist()

    if len(squad_names) != 23 or len(goalkeeper_names) != 3 or len(outfield_names) != 20:
        raise ValueError(
            "Stage 4.4 requires the corrected 23-player Stage 4.3 squad: "
            "3 goalkeepers and 20 outfield players. "
            f"Found {len(squad_names)} total, {len(goalkeeper_names)} goalkeepers, "
            f"and {len(outfield_names)} outfield players in {squad_file.name}."
        )

    if STARTING_GOALKEEPER not in goalkeeper_names:
        raise ValueError(
            f"{STARTING_GOALKEEPER} is not one of the selected Stage 4.3 goalkeepers."
        )

    role_data = role_data[role_data["Name"].isin(outfield_names)].copy()
    role_data = (
        role_data.sort_values("Player Role Suitability", ascending=False)
        .drop_duplicates(["Name", "Evaluated Role"])
        .reset_index(drop=True)
    )

    missing_outfield = sorted(set(outfield_names).difference(role_data["Name"]))
    if missing_outfield:
        raise ValueError(
            "These selected outfield players were missing from the role file: "
            + ", ".join(missing_outfield)
        )

    # Stage 4.1's role-suitability output may not retain the six raw EAFC
    # attributes. Tactical scenario scoring must use the real attributes—not
    # duplicate Role Attribute Score as a fallback—so merge them from the
    # full-feature outfield dataset whenever necessary.
    missing_attributes = [column for column in EAFC_ATTRIBUTES if column not in role_data.columns]
    if missing_attributes:
        if not PLAYER_FEATURE_FILE.exists():
            raise FileNotFoundError(
                "The player-role file does not contain PAC/SHO/PAS/DRI/DEF/PHY, "
                f"and the feature file could not be found: {PLAYER_FEATURE_FILE}"
            )
        feature_data = pd.read_csv(PLAYER_FEATURE_FILE)
        required_features = {"Name", *EAFC_ATTRIBUTES}
        missing_feature_columns = sorted(required_features.difference(feature_data.columns))
        if missing_feature_columns:
            raise ValueError(
                f"The full-feature file is missing columns: {missing_feature_columns}"
            )

        # If duplicate names exist, prefer the highest-OVR record when OVR is
        # available. This matches the elite-player records used by Stage 4.3.
        if "OVR" in feature_data.columns:
            feature_data = feature_data.sort_values("OVR", ascending=False)
        feature_data = feature_data.drop_duplicates("Name")[["Name", *EAFC_ATTRIBUTES]]
        role_data = role_data.merge(feature_data, on="Name", how="left", validate="many_to_one")

    for column in EAFC_ATTRIBUTES:
        role_data[column] = pd.to_numeric(role_data[column], errors="coerce")
    missing_attribute_players = sorted(
        role_data.loc[role_data[EAFC_ATTRIBUTES].isna().any(axis=1), "Name"].unique()
    )
    if missing_attribute_players:
        raise ValueError(
            "Real EAFC attributes could not be resolved for these squad players: "
            + ", ".join(missing_attribute_players)
        )

    print(f"Squad file: {squad_file.name}")
    print(f"Squad size: {len(squad_names)} ({len(goalkeeper_names)} GK + {len(outfield_names)} outfield)")
    print(f"Squad player-role combinations: {len(role_data)}")
    print(f"Tactical attributes loaded from: {PLAYER_FEATURE_FILE.name}")
    return squad, role_data, set(squad_names), goalkeeper_names


def numeric_value(row, column, fallback):
    """Return a numeric attribute or a documented fallback when unavailable."""
    if column in row.index and pd.notna(row[column]):
        return float(row[column])
    return float(fallback)


def add_tactical_scores(role_data):
    """Create attack, defense, and situation-specific assignment scores."""
    scored = role_data.copy()

    # These columns were validated and, when needed, merged from the Stage 3.4
    # full-feature dataset. Never replace them with a neutral fallback: doing
    # so would make attack, defense, physical, and passing summaries identical.
    for column in EAFC_ATTRIBUTES:
        scored[column] = pd.to_numeric(scored[column], errors="raise")

    scored["Attack Score"] = scored[["PAC", "SHO", "PAS", "DRI"]].mean(axis=1)
    scored["Defense Score"] = scored["DEF"]
    scored["Physical Score"] = scored["PHY"]
    scored["Passing Score"] = scored["PAS"]

    scored["Balanced Score"] = (
        0.80 * scored["Player Role Suitability"]
        + 0.10 * scored["OVR"]
        + 0.05 * scored["Attack Score"]
        + 0.05 * scored["Defense Score"]
    )

    for scenario_name, settings in SCENARIOS.items():
        weights = settings["Weights"]
        score = weights.get("Suitability", 0.0) * scored["Player Role Suitability"]
        score += weights.get("Attack", 0.0) * scored["Attack Score"]
        score += weights.get("Defense", 0.0) * scored["Defense Score"]
        score += weights.get("Physical", 0.0) * scored["Physical Score"]
        score += weights.get("Passing", 0.0) * scored["Passing Score"]
        score += weights.get("OVR", 0.0) * scored["OVR"]
        scored[f"{scenario_name} Score"] = score

    return scored


def optimize_lineup(
    role_data,
    formation_name,
    score_column,
    starting_names=None,
    max_new_players=None,
    substitution_penalty=0.0,
):
    """Optimize a unique player-to-slot assignment with an optional sub limit."""
    slots = FORMATIONS[formation_name]
    required_roles = {role for _, role in slots}
    candidates = role_data[role_data["Evaluated Role"].isin(required_roles)].copy()
    players = sorted(candidates["Name"].unique())
    row_lookup = candidates.set_index(["Name", "Evaluated Role"])

    variable_count = 0
    assignment_variables = {}
    entrant_variables = {}

    for slot_index, (_, role) in enumerate(slots):
        eligible = candidates.loc[candidates["Evaluated Role"].eq(role), "Name"].unique()
        for player in eligible:
            assignment_variables[(slot_index, player)] = variable_count
            variable_count += 1

    if starting_names is not None:
        starting_names = set(starting_names)
        for player in players:
            if player not in starting_names:
                entrant_variables[player] = variable_count
                variable_count += 1

    objective = np.zeros(variable_count)
    for (slot_index, player), variable in assignment_variables.items():
        role = slots[slot_index][1]
        objective[variable] = -float(row_lookup.loc[(player, role), score_column])

    # A substitution must create enough tactical value to justify disrupting
    # the current XI. Without this penalty, the optimizer uses every permitted
    # change for even a tiny numerical gain.
    for variable in entrant_variables.values():
        objective[variable] = float(substitution_penalty)

    matrix_rows = []
    matrix_columns = []
    matrix_values = []
    lower_bounds = []
    upper_bounds = []
    constraint_index = 0

    def add_constraint(coefficients, lower, upper):
        nonlocal constraint_index
        for variable, coefficient in coefficients.items():
            matrix_rows.append(constraint_index)
            matrix_columns.append(variable)
            matrix_values.append(coefficient)
        lower_bounds.append(lower)
        upper_bounds.append(upper)
        constraint_index += 1

    # Fill every slot exactly once.
    for slot_index in range(len(slots)):
        coefficients = {
            variable: 1.0
            for (item_slot, _), variable in assignment_variables.items()
            if item_slot == slot_index
        }
        if not coefficients:
            raise ValueError(f"No eligible candidates exist for {formation_name} slot {slots[slot_index][0]}.")
        add_constraint(coefficients, 1.0, 1.0)

    # Each player can fill at most one slot.
    for player in players:
        coefficients = {
            variable: 1.0
            for (_, item_player), variable in assignment_variables.items()
            if item_player == player
        }
        if coefficients:
            add_constraint(coefficients, 0.0, 1.0)

    # Link non-starter assignments to entrant variables and limit substitutes.
    if entrant_variables:
        for player, entrant_variable in entrant_variables.items():
            coefficients = {
                variable: 1.0
                for (_, item_player), variable in assignment_variables.items()
                if item_player == player
            }
            coefficients[entrant_variable] = -1.0
            add_constraint(coefficients, 0.0, 0.0)
        add_constraint(
            {variable: 1.0 for variable in entrant_variables.values()},
            0.0,
            float(max_new_players),
        )

    constraint_matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(constraint_index, variable_count),
    ).tocsr()

    result = milp(
        c=objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=LinearConstraint(
            constraint_matrix,
            np.array(lower_bounds),
            np.array(upper_bounds),
        ),
        options={"time_limit": 60, "mip_rel_gap": 0.0001, "disp": False},
    )
    if result.x is None:
        raise RuntimeError(f"Could not optimize the {formation_name} lineup: {result.message}")

    rows = []
    for slot_index, (slot_name, role) in enumerate(slots):
        selected_player = next(
            player
            for (item_slot, player), variable in assignment_variables.items()
            if item_slot == slot_index and result.x[variable] > 0.5
        )
        source = row_lookup.loc[(selected_player, role)]
        rows.append({
            "Formation": formation_name,
            "Formation Slot": slot_name,
            "Assigned Role": role,
            "Name": selected_player,
            "Official Position": source["Position"],
            "Role Source": source["Role Source"],
            "OVR": float(source["OVR"]),
            "Position Fit Score": float(source["Position Fit Score"]),
            "Role Attribute Score": float(source["Role Attribute Score"]),
            "Selection Score": float(source["Player Role Suitability"]),
            "Tactical Score": float(source[score_column]),
            "PAC": numeric_value(source, "PAC", source["Role Attribute Score"]),
            "SHO": numeric_value(source, "SHO", source["Role Attribute Score"]),
            "PAS": numeric_value(source, "PAS", source["Role Attribute Score"]),
            "DRI": numeric_value(source, "DRI", source["Role Attribute Score"]),
            "DEF": numeric_value(source, "DEF", source["Role Attribute Score"]),
            "PHY": numeric_value(source, "PHY", source["Role Attribute Score"]),
        })
    return pd.DataFrame(rows)


def create_substitution_plan(starting_lineup, ending_lineup, scenario_name):
    """Describe player changes and retained-player role changes."""
    starting = starting_lineup.set_index("Name")
    ending = ending_lineup.set_index("Name")
    starting_names = set(starting.index)
    ending_names = set(ending.index)

    leaving = sorted(starting_names.difference(ending_names))
    entering = sorted(ending_names.difference(starting_names))
    retained = sorted(starting_names.intersection(ending_names))

    # Pair changes for a readable plan. A pair is a roster transaction, not a
    # claim that the entering player directly inherits the departing role.
    change_rows = []
    for change_number, (player_out, player_in) in enumerate(zip(leaving, entering), start=1):
        change_rows.append({
            "Scenario": scenario_name,
            "Substitution Window": "Halftime (not an in-play opportunity)",
            "Suggested Time": TACTICAL_DECISION_TIME,
            "Change Number": change_number,
            "Player Out": player_out,
            "Player Out Role": starting.loc[player_out, "Assigned Role"],
            "Player In": player_in,
            "Player In Ending Role": ending.loc[player_in, "Assigned Role"],
        })

    role_changes = []
    for player in retained:
        old_role = starting.loc[player, "Assigned Role"]
        new_role = ending.loc[player, "Assigned Role"]
        if old_role != new_role:
            role_changes.append({
                "Scenario": scenario_name,
                "Name": player,
                "Starting Role": old_role,
                "Ending Role": new_role,
            })

    substitutions = pd.DataFrame(change_rows)
    roles = pd.DataFrame(role_changes)
    return substitutions, roles, leaving, entering, retained


def check_rule_compliance(
    scenario_name,
    squad_names,
    goalkeeper_names,
    starting_lineup,
    ending_lineup,
    leaving,
    entering,
):
    """Return an explicit legality audit for one tactical plan."""
    starting_on_field = set(starting_lineup["Name"]) | {STARTING_GOALKEEPER}
    ending_on_field = set(ending_lineup["Name"]) | {STARTING_GOALKEEPER}
    substitutions_used = len(entering)
    # Halftime substitutions still count toward the five-player maximum, but
    # they do not use one of the three in-play substitution opportunities.
    windows_used = 0

    checks = {
        "11 players start": len(starting_on_field) == 11,
        "11 players finish plan": len(ending_on_field) == 11,
        "Exactly one goalkeeper starts": len(starting_on_field.intersection(goalkeeper_names)) == 1,
        "Exactly one goalkeeper finishes plan": len(ending_on_field.intersection(goalkeeper_names)) == 1,
        "Alisson remains goalkeeper": STARTING_GOALKEEPER in starting_on_field and STARTING_GOALKEEPER in ending_on_field,
        "Maximum five substitutions": substitutions_used <= MAX_SUBSTITUTIONS,
        "Maximum three substitution windows": windows_used <= MAX_SUBSTITUTION_WINDOWS,
        "Substituted players do not return": set(leaving).isdisjoint(ending_on_field),
        "Entering players were not already on field": set(entering).isdisjoint(starting_on_field),
        "All players belong to 23-player squad": starting_on_field.union(ending_on_field).issubset(squad_names),
        "Ten unique outfield roles filled": ending_lineup["Name"].nunique() == 10 and len(ending_lineup) == 10,
    }
    return {
        "Scenario": scenario_name,
        "Starting Formation": DEFAULT_FORMATION,
        "Ending Formation": ending_lineup["Formation"].iloc[0],
        "Players on Field": len(ending_on_field),
        "Goalkeepers on Field": len(ending_on_field.intersection(goalkeeper_names)),
        "Substitutions Used": substitutions_used,
        "Substitution Windows Used": windows_used,
        "Halftime Window Used": "Yes" if substitutions_used else "No",
        "Removed Player Returned": "No" if checks["Substituted players do not return"] else "Yes",
        "All Individual Checks Passed": all(checks.values()),
        "Lineup Legal": "YES" if all(checks.values()) else "NO",
        "Failed Checks": "None" if all(checks.values()) else "; ".join(
            label for label, passed in checks.items() if not passed
        ),
    }


def lineup_summary(scenario_name, lineup):
    """Summarize tactical qualities for the starting or ending XI."""
    return {
        "Scenario": scenario_name,
        "Formation": lineup["Formation"].iloc[0],
        "Average Role Suitability": round(lineup["Selection Score"].mean(), 2),
        "Average Tactical Score": round(lineup["Tactical Score"].mean(), 2),
        "Average Attack": round(lineup[["PAC", "SHO", "PAS", "DRI"]].mean(axis=1).mean(), 2),
        "Average Defense": round(lineup["DEF"].mean(), 2),
        "Average Physical": round(lineup["PHY"].mean(), 2),
        "Average Passing": round(lineup["PAS"].mean(), 2),
        "Flexible-Role Assignments": int((lineup["Role Source"] != "Official Position").sum()),
    }


def print_lineup(title, lineup):
    """Print the most useful lineup columns."""
    print(f"\n{title}")
    print("-" * len(title))
    print(f"GK  Goalkeeper              {STARTING_GOALKEEPER}")
    print(lineup[[
        "Formation Slot",
        "Assigned Role",
        "Name",
        "Official Position",
        "Role Source",
        "OVR",
        "Selection Score",
        "Tactical Score",
    ]].round(2).to_string(index=False))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    squad, role_data, squad_names, goalkeeper_names = load_and_validate_data()
    scored_roles = add_tactical_scores(role_data)

    # Optimize one balanced default XI. This same team starts every scenario,
    # so differences later are responses to match state rather than hindsight.
    starting_lineup = optimize_lineup(
        scored_roles,
        DEFAULT_FORMATION,
        "Balanced Score",
    )
    starting_names = set(starting_lineup["Name"])

    print("\nSTAGE 4.4 DEFAULT STARTING XI")
    print("================================")
    print_lineup(f"Balanced {DEFAULT_FORMATION}", starting_lineup)

    all_ending_lineups = []
    all_substitutions = []
    all_role_changes = []
    compliance_rows = []
    summary_rows = [lineup_summary("Default Starting XI", starting_lineup)]

    for scenario_name, settings in SCENARIOS.items():
        ending_formation = settings["Ending Formation"]
        ending_lineup = optimize_lineup(
            scored_roles,
            ending_formation,
            f"{scenario_name} Score",
            starting_names=starting_names,
            max_new_players=settings["Maximum Tactical Substitutions"],
            substitution_penalty=settings["Substitution Penalty"],
        )
        ending_lineup.insert(0, "Scenario", scenario_name)
        all_ending_lineups.append(ending_lineup)

        substitutions, role_changes, leaving, entering, retained = create_substitution_plan(
            starting_lineup, ending_lineup, scenario_name
        )
        if not substitutions.empty:
            all_substitutions.append(substitutions)
        if not role_changes.empty:
            all_role_changes.append(role_changes)

        compliance_rows.append(check_rule_compliance(
            scenario_name,
            squad_names,
            set(goalkeeper_names),
            starting_lineup,
            ending_lineup,
            leaving,
            entering,
        ))
        summary_rows.append(lineup_summary(scenario_name, ending_lineup))

        print(f"\n\n{scenario_name.upper()} SCENARIO")
        print("=" * (len(scenario_name) + 9))
        print(settings["Description"])
        print(f"Starting shape: {DEFAULT_FORMATION} | Ending shape: {ending_formation}")
        print(
            "Tactical substitution cap: "
            f"{settings['Maximum Tactical Substitutions']} "
            f"(match-law maximum remains {MAX_SUBSTITUTIONS})"
        )
        print(f"Decision time: {TACTICAL_DECISION_TIME}")
        print(
            f"Players retained: {len(retained)} | Substitutions: {len(entering)} "
            "| In-play substitution windows: 0"
        )
        print(f"Players out: {', '.join(leaving) if leaving else 'None'}")
        print(f"Players in:  {', '.join(entering) if entering else 'None'}")
        print_lineup(f"{scenario_name} ending XI ({ending_formation})", ending_lineup)
        if not role_changes.empty:
            print("\nRetained players changing roles:")
            print(role_changes.to_string(index=False))

    ending_lineups = pd.concat(all_ending_lineups, ignore_index=True)
    substitutions_output = (
        pd.concat(all_substitutions, ignore_index=True)
        if all_substitutions
        else pd.DataFrame(columns=[
            "Scenario", "Substitution Window", "Suggested Time", "Change Number",
            "Player Out", "Player Out Role", "Player In", "Player In Ending Role",
        ])
    )
    role_changes_output = (
        pd.concat(all_role_changes, ignore_index=True)
        if all_role_changes
        else pd.DataFrame(columns=["Scenario", "Name", "Starting Role", "Ending Role"])
    )
    compliance = pd.DataFrame(compliance_rows)
    summaries = pd.DataFrame(summary_rows)

    print("\n\nRULE COMPLIANCE CHECK")
    print("=====================")
    print(compliance.to_string(index=False))
    if not compliance["All Individual Checks Passed"].all():
        raise RuntimeError("At least one tactical plan failed the rule-compliance check.")

    print("\nTACTICAL SUMMARY")
    print("================")
    print(summaries.to_string(index=False))

    starting_lineup.to_csv(RESULTS_DIR / "default_starting_4_3_3.csv", index=False)
    ending_lineups.to_csv(RESULTS_DIR / "scenario_ending_lineups.csv", index=False)
    substitutions_output.to_csv(RESULTS_DIR / "rules_compliant_substitution_plans.csv", index=False)
    role_changes_output.to_csv(RESULTS_DIR / "retained_player_role_changes.csv", index=False)
    compliance.to_csv(RESULTS_DIR / "rule_compliance_checks.csv", index=False)
    summaries.to_csv(RESULTS_DIR / "tactical_lineup_summary.csv", index=False)

    print(f"\nStage 4.4 results saved to:\n{RESULTS_DIR}")
    print("\nStage 4.4 starting XI and legal substitution plans complete!")


if __name__ == "__main__":
    main()