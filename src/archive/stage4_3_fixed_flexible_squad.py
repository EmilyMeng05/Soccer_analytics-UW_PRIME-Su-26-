import numpy as np
import pandas as pd
from itertools import combinations, permutations
from pathlib import Path
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

# Stage 4.3 selects one fixed 18-player squad for all three formations:
# 16 outfield players and 2 goalkeepers. The optimization rewards both strong
# lineups and players who can remain on the field when the formation changes.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYER_FILE = PROJECT_ROOT / "data" / "processed" / "eafc26_player_role_suitability.csv"
GOALKEEPER_FILE = PROJECT_ROOT / "data" / "processed" / "eafc26_goalkeepers_with_styles.csv"
BASELINE_FILE = PROJECT_ROOT / "data" / "processed" / "eafc26_best_independent_formation_lineups.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage_4_3"

OUTFIELD_SQUAD_SIZE = 20
GOALKEEPER_SQUAD_SIZE = 3
TOP_PLAYERS_PER_ROLE = 40
TOP_FLEXIBLE_PLAYERS = 100
RETENTION_BONUS = 2.0
RESERVE_VALUE_WEIGHT = 0.01

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

def find_column(data, possible_names):
    """Return the first available column from a list of possible names."""
    for column in possible_names:
        if column in data.columns:
            return column
    raise KeyError(f"None of these columns were found: {possible_names}")

def validate_player_data(data):
    """Check that the Stage 4.1 player-role output has the required columns."""
    required = {
        "Name", "Position", "OVR", "Evaluated Role", "Role Source",
        "Position Fit Score", "Role Attribute Score", "Player Role Suitability",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing player columns: {missing}")

def prepare_role_data(player_data):
    """Keep each player's strongest row for each eligible role."""
    role_data = player_data.sort_values(
        "Player Role Suitability", ascending=False
    ).drop_duplicates(["Name", "Evaluated Role"])
    return role_data.reset_index(drop=True)

def create_candidate_pool(role_data):
    """Reduce the optimization to elite specialists and useful flexible players."""
    candidate_names = set()
    for _, group in role_data.groupby("Evaluated Role"):
        candidate_names.update(group.nlargest(
            TOP_PLAYERS_PER_ROLE, "Player Role Suitability"
        )["Name"])

    # Flexible players receive credit for having multiple supported roles, but
    # their suitability scores still determine whether they make the squad.
    profiles = role_data.groupby("Name").agg(
        Best_Score=("Player Role Suitability", "max"),
        Average_Score=("Player Role Suitability", "mean"),
        Eligible_Roles=("Evaluated Role", "nunique"),
    ).reset_index()
    profiles["Flexibility Value"] = (
        profiles["Best_Score"]
        + 0.50 * profiles["Average_Score"]
        + 1.50 * (profiles["Eligible_Roles"] - 1)
    )
    flexible = profiles[profiles["Eligible_Roles"] >= 2].nlargest(
        TOP_FLEXIBLE_PLAYERS, "Flexibility Value"
    )
    candidate_names.update(flexible["Name"])

    # Keep the Stage 4.2 baseline players so the fixed-squad result can always
    # be compared fairly with the independent best lineups.
    if BASELINE_FILE.exists():
        baseline = pd.read_csv(BASELINE_FILE)
        candidate_names.update(baseline["Name"].dropna())

    candidate_data = role_data[role_data["Name"].isin(candidate_names)].copy()
    print(f"Optimization candidate players: {candidate_data['Name'].nunique()}")
    print(f"Candidate player-role combinations: {len(candidate_data)}")
    return candidate_data.reset_index(drop=True)

def build_and_solve_model(candidate_data):
    """Jointly choose the fixed squad and the three formation lineups."""
    players = sorted(candidate_data["Name"].unique())
    role_lookup = candidate_data.set_index(["Name", "Evaluated Role"])
    formation_names = list(FORMATIONS)
    formation_pairs = list(combinations(formation_names, 2))

    variable_count = 0
    x_variables = {}
    y_variables = {}
    z_variables = {}

    # x = player is assigned to a particular formation slot.
    for formation, slots in FORMATIONS.items():
        for slot_index, (_, role) in enumerate(slots):
            eligible_players = candidate_data.loc[
                candidate_data["Evaluated Role"] == role, "Name"
            ].unique()
            for player in eligible_players:
                x_variables[(formation, slot_index, player)] = variable_count
                variable_count += 1

    # y = player belongs to the fixed 16-player outfield squad.
    for player in players:
        y_variables[player] = variable_count
        variable_count += 1

    # z = player starts in both formations in a pair. Maximizing z reduces the
    # number of substitutions required when switching between formations.
    for formation_pair in formation_pairs:
        for player in players:
            z_variables[(formation_pair, player)] = variable_count
            variable_count += 1

    objective = np.zeros(variable_count)
    for (formation, slot_index, player), variable in x_variables.items():
        role = FORMATIONS[formation][slot_index][1]
        score = role_lookup.loc[(player, role), "Player Role Suitability"]
        objective[variable] = -float(score)

    player_profiles = candidate_data.groupby("Name").agg(
        Best_Score=("Player Role Suitability", "max"),
        Eligible_Roles=("Evaluated Role", "nunique"),
    )
    for player, variable in y_variables.items():
        reserve_value = (
            player_profiles.loc[player, "Best_Score"]
            + player_profiles.loc[player, "Eligible_Roles"] - 1
        )
        objective[variable] = -RESERVE_VALUE_WEIGHT * reserve_value

    for variable in z_variables.values():
        objective[variable] = -RETENTION_BONUS

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

    # Every formation slot must contain exactly one eligible player.
    for formation, slots in FORMATIONS.items():
        for slot_index in range(len(slots)):
            coefficients = {
                variable: 1.0
                for (item_formation, item_slot, _), variable in x_variables.items()
                if item_formation == formation and item_slot == slot_index
            }
            add_constraint(coefficients, 1.0, 1.0)

    # A player can fill no more than one slot in the same formation.
    for formation in formation_names:
        for player in players:
            coefficients = {
                variable: 1.0
                for (item_formation, _, item_player), variable in x_variables.items()
                if item_formation == formation and item_player == player
            }
            if coefficients:
                add_constraint(coefficients, 0.0, 1.0)

    # A player can start only when selected for the fixed squad.
    for (_, _, player), x_variable in x_variables.items():
        add_constraint({x_variable: 1.0, y_variables[player]: -1.0}, -np.inf, 0.0)

    # Select exactly 16 outfield players; the two goalkeepers are added later.
    add_constraint({variable: 1.0 for variable in y_variables.values()},
                   OUTFIELD_SQUAD_SIZE, OUTFIELD_SQUAD_SIZE)

    # A retained-player variable can equal one only if that player starts in
    # both formations. Because retention has a positive reward, the solver will
    # activate it whenever both appearances occur.
    for formation_pair in formation_pairs:
        first_formation, second_formation = formation_pair
        for player in players:
            z_variable = z_variables[(formation_pair, player)]
            first_appearance = {
                variable: -1.0
                for (formation, _, item_player), variable in x_variables.items()
                if formation == first_formation and item_player == player
            }
            second_appearance = {
                variable: -1.0
                for (formation, _, item_player), variable in x_variables.items()
                if formation == second_formation and item_player == player
            }
            first_appearance[z_variable] = 1.0
            second_appearance[z_variable] = 1.0
            add_constraint(first_appearance, -np.inf, 0.0)
            add_constraint(second_appearance, -np.inf, 0.0)

    constraint_matrix = coo_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(constraint_index, variable_count),
    ).tocsr()
    constraints = LinearConstraint(
        constraint_matrix,
        np.array(lower_bounds),
        np.array(upper_bounds),
    )
    result = milp(
        c=objective,
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.ones(variable_count)),
        constraints=constraints,
        options={"time_limit": 120, "mip_rel_gap": 0.001, "disp": False},
    )
    if result.x is None:
        raise RuntimeError(f"The fixed-squad optimization failed: {result.message}")
    print(f"Optimization status: {result.message}")
    return result.x, x_variables, y_variables, role_lookup

def extract_outfield_lineups(solution, x_variables, role_lookup):
    """Convert the selected assignment variables into readable lineups."""
    lineups = {}
    for formation, slots in FORMATIONS.items():
        rows = []
        for slot_index, (slot_name, role) in enumerate(slots):
            selected_player = None
            for (item_formation, item_slot, player), variable in x_variables.items():
                if item_formation == formation and item_slot == slot_index and solution[variable] > 0.5:
                    selected_player = player
                    break
            if selected_player is None:
                raise RuntimeError(f"No player was selected for {formation} {slot_name}.")
            source_row = role_lookup.loc[(selected_player, role)]
            rows.append({
                "Formation": formation,
                "Lineup Status": "Starter",
                "Formation Slot": slot_name,
                "Assigned Role": role,
                "Name": selected_player,
                "Official Position": source_row["Position"],
                "Role Source": source_row["Role Source"],
                "OVR": source_row["OVR"],
                "Position Fit Score": source_row["Position Fit Score"],
                "Role Attribute Score": source_row["Role Attribute Score"],
                "Selection Score": source_row["Player Role Suitability"],
            })
        lineups[formation] = pd.DataFrame(rows)
    return lineups

def select_goalkeepers(goalkeeper_data):
    """Select the best overall starting goalkeeper and one backup."""
    name_column = find_column(goalkeeper_data, ["Name", "Player Name", "LongName"])
    position_column = find_column(goalkeeper_data, ["Position", "Best Position"])
    skill_column = find_column(
        goalkeeper_data, ["Goalkeeper Skill", "GoalkeeperSkill", "Goalkeeper Skill Score"]
    )
    protection_column = find_column(
        goalkeeper_data,
        ["Protection Contribution", "ProtectionContribution", "Protection Score"],
    )
    distribution_column = find_column(
        goalkeeper_data,
        ["Distribution Contribution", "DistributionContribution", "Distribution Score"],
    )
    style_column = find_column(
        goalkeeper_data, ["Goalkeeper Style", "GoalkeeperStyle", "Style"]
    )
    ranked = goalkeeper_data.sort_values(
        [skill_column, protection_column, distribution_column], ascending=False
    ).drop_duplicates(name_column).head(GOALKEEPER_SQUAD_SIZE)
    rows = []
    for rank, (_, row) in enumerate(ranked.iterrows()):
        rows.append({
            "Squad Group": "Goalkeeper",
            "Squad Status": "Starting Goalkeeper" if rank == 0 else "Backup Goalkeeper",
            "Name": row[name_column],
            "Official Position": row[position_column],
            "OVR": row["OVR"] if "OVR" in goalkeeper_data.columns else pd.NA,
            "Goalkeeper Skill": row[skill_column],
            "Protection Contribution": row[protection_column],
            "Distribution Contribution": row[distribution_column],
            "Goalkeeper Style": row[style_column],
        })
    return pd.DataFrame(rows)

def create_squad_roster(solution, y_variables, candidate_data, lineups, goalkeepers):
    """Create one table containing all 18 selected squad members."""
    selected_names = [player for player, variable in y_variables.items() if solution[variable] > 0.5]
    rows = []
    for player in selected_names:
        player_roles = candidate_data[candidate_data["Name"] == player].sort_values(
            "Player Role Suitability", ascending=False
        )
        best_row = player_roles.iloc[0]
        starting_formations = [
            formation for formation, lineup in lineups.items()
            if player in set(lineup["Name"])
        ]
        flexible_roles = player_roles.loc[
            player_roles["Role Source"] != "Official Position", "Evaluated Role"
        ].tolist()
        rows.append({
            "Squad Group": "Outfield",
            "Squad Status": "Starter in at least one formation" if starting_formations else "Reserve",
            "Name": player,
            "Official Position": best_row["Position"],
            "OVR": best_row["OVR"],
            "Best Suitability Score": best_row["Player Role Suitability"],
            "Eligible Roles": ", ".join(player_roles["Evaluated Role"]),
            "Flexible Roles": ", ".join(flexible_roles) if flexible_roles else "None",
            "Number of Eligible Roles": player_roles["Evaluated Role"].nunique(),
            "Number of Starting Formations": len(starting_formations),
            "Starting Formations": ", ".join(starting_formations) if starting_formations else "None",
        })
    outfield_roster = pd.DataFrame(rows).sort_values(
        ["Number of Starting Formations", "Best Suitability Score"], ascending=False
    )
    goalkeeper_roster = goalkeepers.copy()
    for column in outfield_roster.columns:
        if column not in goalkeeper_roster.columns:
            goalkeeper_roster[column] = pd.NA
    goalkeeper_roster = goalkeeper_roster[outfield_roster.columns]
    return pd.concat([goalkeeper_roster, outfield_roster], ignore_index=True)

def add_goalkeepers_to_lineups(outfield_lineups, goalkeepers):
    """Add the same starting and backup goalkeepers to each fixed-squad lineup."""
    combined = []
    starter = goalkeepers.iloc[0]
    backup = goalkeepers.iloc[1]
    for formation, outfield in outfield_lineups.items():
        goalkeeper_rows = []
        for status, slot, goalkeeper in [
            ("Starter", "GK", starter),
            ("Backup", "Backup GK", backup),
        ]:
            goalkeeper_rows.append({
                "Formation": formation,
                "Lineup Status": status,
                "Formation Slot": slot,
                "Assigned Role": "Goalkeeper",
                "Name": goalkeeper["Name"],
                "Official Position": goalkeeper["Official Position"],
                "Role Source": "Official Position",
                "OVR": goalkeeper["OVR"],
                "Position Fit Score": 100.0,
                "Role Attribute Score": goalkeeper["Goalkeeper Skill"],
                "Selection Score": goalkeeper["Goalkeeper Skill"],
            })
        formation_lineup = pd.concat([
            pd.DataFrame([goalkeeper_rows[0]]),
            outfield,
            pd.DataFrame([goalkeeper_rows[1]]),
        ], ignore_index=True)
        combined.append(formation_lineup)
    return pd.concat(combined, ignore_index=True)

def create_transition_summary(lineups):
    """Measure substitutions and role changes for all six directed switches."""
    rows = []
    for starting_formation, ending_formation in permutations(FORMATIONS, 2):
        starting_lineup = lineups[starting_formation].set_index("Name")
        ending_lineup = lineups[ending_formation].set_index("Name")
        retained = sorted(set(starting_lineup.index).intersection(ending_lineup.index))
        leaving = sorted(set(starting_lineup.index).difference(ending_lineup.index))
        entering = sorted(set(ending_lineup.index).difference(starting_lineup.index))
        role_changes = []
        for player in retained:
            old_role = starting_lineup.loc[player, "Assigned Role"]
            new_role = ending_lineup.loc[player, "Assigned Role"]
            if old_role != new_role:
                role_changes.append(f"{player}: {old_role} -> {new_role}")
        rows.append({
            "Starting Formation": starting_formation,
            "Ending Formation": ending_formation,
            "Outfield Players Retained": len(retained),
            "Substitutions Required": len(entering),
            "Players Changing Roles": len(role_changes),
            "Retained Players": ", ".join(retained),
            "Players Leaving": ", ".join(leaving) if leaving else "None",
            "Players Entering": ", ".join(entering) if entering else "None",
            "Role Changes": "; ".join(role_changes) if role_changes else "None",
        })
    return pd.DataFrame(rows)

def create_formation_summary(lineups, goalkeepers):
    """Compare fixed-squad lineup quality with the independent Stage 4.2 baseline."""
    baseline_totals = {}
    if BASELINE_FILE.exists():
        baseline = pd.read_csv(BASELINE_FILE)
        baseline_outfield = baseline[
            (baseline["Lineup Status"] == "Starter")
            & (baseline["Assigned Role"] != "Goalkeeper")
        ]
        baseline_totals = baseline_outfield.groupby("Formation")["Selection Score"].sum().to_dict()
    goalkeeper_score = goalkeepers.iloc[0]["Goalkeeper Skill"]
    rows = []
    for formation, lineup in lineups.items():
        outfield_total = lineup["Selection Score"].sum()
        baseline_total = baseline_totals.get(formation, np.nan)
        flexible_starters = (lineup["Role Source"] != "Official Position").sum()
        rows.append({
            "Formation": formation,
            "Fixed-Squad Outfield Total": round(outfield_total, 2),
            "Fixed-Squad Average Outfield Suitability": round(outfield_total / 10, 2),
            "Independent Baseline Outfield Total": round(baseline_total, 2) if pd.notna(baseline_total) else np.nan,
            "Score Loss from Fixed Squad": round(baseline_total - outfield_total, 2) if pd.notna(baseline_total) else np.nan,
            "Starting XI Average Score": round((outfield_total + goalkeeper_score) / 11, 2),
            "Official-Role Outfield Starters": int(10 - flexible_starters),
            "Flexible-Role Outfield Starters": int(flexible_starters),
        })
    return pd.DataFrame(rows)

def print_results(squad, lineups, goalkeepers, formation_summary, transition_summary):
    """Print the fixed squad, its three lineups, and switching measurements."""
    print("\nFIXED 23-PLAYER SQUAD")
    print("---------------------")
    print(squad[[
        "Squad Group", "Squad Status", "Name", "Official Position", "OVR",
        "Eligible Roles", "Number of Starting Formations"
    ]].to_string(index=False))
    for formation, lineup in lineups.items():
        print(f"\n{formation} FIXED-SQUAD LINEUP")
        print("-" * (len(formation) + 20))
        print(f"Starting goalkeeper: {goalkeepers.iloc[0]['Name']}")
        print(lineup[[
            "Formation Slot", "Assigned Role", "Name", "Official Position",
            "Role Source", "OVR", "Selection Score"
        ]].to_string(index=False))
        print(f"Backup goalkeeper: {goalkeepers.iloc[1]['Name']}")
    print("\nFORMATION QUALITY COMPARISON")
    print("----------------------------")
    print(formation_summary.to_string(index=False))
    print("\nFORMATION SWITCHING SUMMARY")
    print("---------------------------")
    print(transition_summary[[
        "Starting Formation", "Ending Formation", "Outfield Players Retained",
        "Substitutions Required", "Players Changing Roles", "Role Changes"
    ]].to_string(index=False))

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    player_data = pd.read_csv(PLAYER_FILE)
    goalkeeper_data = pd.read_csv(GOALKEEPER_FILE)
    validate_player_data(player_data)
    print(f"Player-role dataset shape: {player_data.shape}")
    print(f"Goalkeeper dataset shape: {goalkeeper_data.shape}")

    role_data = prepare_role_data(player_data)
    candidate_data = create_candidate_pool(role_data)
    solution, x_variables, y_variables, role_lookup = build_and_solve_model(candidate_data)
    outfield_lineups = extract_outfield_lineups(solution, x_variables, role_lookup)
    goalkeepers = select_goalkeepers(goalkeeper_data)
    squad = create_squad_roster(
        solution, y_variables, candidate_data, outfield_lineups, goalkeepers
    )
    combined_lineups = add_goalkeepers_to_lineups(outfield_lineups, goalkeepers)
    transition_summary = create_transition_summary(outfield_lineups)
    formation_summary = create_formation_summary(outfield_lineups, goalkeepers)
    print_results(squad, outfield_lineups, goalkeepers, formation_summary, transition_summary)

    squad_output = PROCESSED_DIR / "eafc26_fixed_flexible_18_player_squad.csv"
    lineup_output = PROCESSED_DIR / "eafc26_fixed_squad_formation_lineups.csv"
    squad.to_csv(squad_output, index=False)
    combined_lineups.to_csv(lineup_output, index=False)
    formation_summary.to_csv(RESULTS_DIR / "fixed_squad_formation_summary.csv", index=False)
    transition_summary.to_csv(RESULTS_DIR / "fixed_squad_transition_summary.csv", index=False)
    for formation, lineup in outfield_lineups.items():
        safe_name = formation.replace("-", "_")
        lineup.to_csv(RESULTS_DIR / f"fixed_squad_{safe_name}_outfield_lineup.csv", index=False)

    print(f"\nFixed 23-player squad saved to:\n{squad_output}")
    print(f"\nFixed-squad formation lineups saved to:\n{lineup_output}")
    print(f"\nStage 4.3 results saved to:\n{RESULTS_DIR}")
    print("\nStage 4.3 fixed flexible squad complete!")

if __name__ == "__main__":
    main()