import pandas as pd
from pathlib import Path
from scipy.optimize import linear_sum_assignment

# Stage 4.2 creates the strongest independent lineup for each formation.
# These lineups are a baseline: every formation may choose a different group
# of players. Stage 4.3 will later impose one fixed squad across all formations.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLAYER_FILE = PROJECT_ROOT / "data" / "processed" / "eafc26_player_role_suitability.csv"
GOALKEEPER_FILE = PROJECT_ROOT / "data" / "processed" / "eafc26_goalkeepers_with_styles.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage_4_2"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Formation notation describes the ten outfield players. Every lineup also
# receives one starting goalkeeper, producing the real-life starting XI.
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
    """Check that the Stage 4.1 output contains the fields used here."""
    required = {
        "Name",
        "Position",
        "OVR",
        "Evaluated Role",
        "Role Source",
        "Position Fit Score",
        "Role Attribute Score",
        "Player Role Suitability",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing player columns: {missing}")

def optimize_outfield_lineup(player_data, formation_name, role_slots):
    """Find the highest-scoring valid assignment of ten unique outfield players."""
    required_roles = {role for _, role in role_slots}
    candidates = player_data[player_data["Evaluated Role"].isin(required_roles)].copy()

    # One column is created for every candidate player. Each matrix row is a
    # formation slot. A very negative value marks a player as ineligible.
    player_names = candidates["Name"].drop_duplicates().tolist()
    score_matrix = pd.DataFrame(-1_000_000.0, index=range(len(role_slots)), columns=player_names)
    row_lookup = {}

    for slot_index, (_, required_role) in enumerate(role_slots):
        role_candidates = candidates[candidates["Evaluated Role"] == required_role]
        for row_index, row in role_candidates.iterrows():
            name = row["Name"]
            score = row["Player Role Suitability"]
            if score > score_matrix.loc[slot_index, name]:
                score_matrix.loc[slot_index, name] = score
                row_lookup[(slot_index, name)] = row_index

    # The Hungarian assignment algorithm maximizes the total score while
    # preventing the same player from filling two positions in one lineup.
    slot_indices, player_indices = linear_sum_assignment(score_matrix.to_numpy(), maximize=True)
    selected_rows = []

    for slot_index, player_index in zip(slot_indices, player_indices):
        name = player_names[player_index]
        if score_matrix.iloc[slot_index, player_index] < 0:
            raise ValueError(f"No eligible assignment exists for {formation_name}.")
        source_row = candidates.loc[row_lookup[(slot_index, name)]]
        slot_name, required_role = role_slots[slot_index]
        selected_rows.append({
            "Formation": formation_name,
            "Lineup Status": "Starter",
            "Formation Slot": slot_name,
            "Assigned Role": required_role,
            "Name": source_row["Name"],
            "Official Position": source_row["Position"],
            "Role Source": source_row["Role Source"],
            "OVR": source_row["OVR"],
            "Position Fit Score": source_row["Position Fit Score"],
            "Role Attribute Score": source_row["Role Attribute Score"],
            "Selection Score": source_row["Player Role Suitability"],
        })

    slot_order = {slot: index for index, (slot, _) in enumerate(role_slots)}
    lineup = pd.DataFrame(selected_rows)
    lineup["Slot Order"] = lineup["Formation Slot"].map(slot_order)
    return lineup.sort_values("Slot Order").drop(columns="Slot Order").reset_index(drop=True)

def select_goalkeepers(goalkeeper_data, formation_name):
    """Select one starting goalkeeper and one backup using overall GK skill."""
    name_column = find_column(goalkeeper_data, ["Name", "Player Name", "LongName"])
    position_column = find_column(goalkeeper_data, ["Position", "Best Position"])
    skill_column = find_column(
        goalkeeper_data,
        ["Goalkeeper Skill", "GoalkeeperSkill", "Goalkeeper Skill Score"],
    )
    protection_column = find_column(
        goalkeeper_data,
        ["Protection Contribution", "ProtectionContribution", "Protection Score"],
    )
    distribution_column = find_column(
        goalkeeper_data,
        ["Distribution Contribution", "DistributionContribution", "Distribution Score"],
    )
    style_column = find_column(goalkeeper_data, ["Goalkeeper Style", "GoalkeeperStyle", "Style"])

    ranked = goalkeeper_data.sort_values(
        [skill_column, protection_column, distribution_column],
        ascending=False,
    ).drop_duplicates(name_column).head(2)

    goalkeeper_rows = []
    for rank, (_, row) in enumerate(ranked.iterrows()):
        goalkeeper_rows.append({
            "Formation": formation_name,
            "Lineup Status": "Starter" if rank == 0 else "Backup",
            "Formation Slot": "GK" if rank == 0 else "Backup GK",
            "Assigned Role": "Goalkeeper",
            "Name": row[name_column],
            "Official Position": row[position_column],
            "Role Source": "Official Position",
            "OVR": row["OVR"] if "OVR" in goalkeeper_data.columns else pd.NA,
            "Position Fit Score": 100.0,
            "Role Attribute Score": row[skill_column],
            "Selection Score": row[skill_column],
            "Protection Contribution": row[protection_column],
            "Distribution Contribution": row[distribution_column],
            "Goalkeeper Style": row[style_column],
        })
    return pd.DataFrame(goalkeeper_rows)

def summarize_formation(formation_name, outfield_lineup, goalkeepers):
    """Create one summary row describing the optimized formation."""
    starter_goalkeeper = goalkeepers[goalkeepers["Lineup Status"] == "Starter"].iloc[0]
    backup_goalkeeper = goalkeepers[goalkeepers["Lineup Status"] == "Backup"].iloc[0]
    outfield_total = outfield_lineup["Selection Score"].sum()
    lineup_total = outfield_total + starter_goalkeeper["Selection Score"]
    flexible_starters = (outfield_lineup["Role Source"] != "Official Position").sum()
    return {
        "Formation": formation_name,
        "Starting Goalkeeper": starter_goalkeeper["Name"],
        "Backup Goalkeeper": backup_goalkeeper["Name"],
        "Outfield Suitability Total": round(outfield_total, 2),
        "Average Outfield Suitability": round(outfield_lineup["Selection Score"].mean(), 2),
        "Starting XI Total Score": round(lineup_total, 2),
        "Starting XI Average Score": round(lineup_total / 11, 2),
        "Official-Role Outfield Starters": int(10 - flexible_starters),
        "Flexible-Role Outfield Starters": int(flexible_starters),
    }

def create_player_overlap(lineups):
    """Show which starting outfield players appear in multiple formations."""
    appearances = []
    for formation, lineup in lineups.items():
        for name in lineup["Name"]:
            appearances.append({"Name": name, "Formation": formation})
    appearances = pd.DataFrame(appearances)
    overlap = appearances.groupby("Name")["Formation"].agg(list).reset_index()
    overlap["Number of Formations"] = overlap["Formation"].str.len()
    overlap["Formations"] = overlap["Formation"].str.join(", ")
    return overlap.drop(columns="Formation").sort_values(
        ["Number of Formations", "Name"], ascending=[False, True]
    ).reset_index(drop=True)

def print_formation(formation_name, outfield_lineup, goalkeepers, summary):
    """Print the selected starting XI, backup goalkeeper, and formation scores."""
    print(f"\n{formation_name}")
    print("-" * len(formation_name))
    starter_goalkeeper = goalkeepers[goalkeepers["Lineup Status"] == "Starter"]
    backup_goalkeeper = goalkeepers[goalkeepers["Lineup Status"] == "Backup"]
    print("\nStarting goalkeeper:")
    print(starter_goalkeeper[[
        "Name", "Selection Score", "Protection Contribution",
        "Distribution Contribution", "Goalkeeper Style"
    ]].to_string(index=False))
    print("\nOutfield starters:")
    print(outfield_lineup[[
        "Formation Slot", "Assigned Role", "Name", "Official Position",
        "Role Source", "OVR", "Selection Score"
    ]].to_string(index=False))
    print("\nBackup goalkeeper:")
    print(backup_goalkeeper[[
        "Name", "Selection Score", "Protection Contribution",
        "Distribution Contribution", "Goalkeeper Style"
    ]].to_string(index=False))
    print(f"\nAverage outfield suitability: {summary['Average Outfield Suitability']:.2f}")
    print(f"Starting XI average score: {summary['Starting XI Average Score']:.2f}")
    print(f"Flexible-role outfield starters: {summary['Flexible-Role Outfield Starters']}")

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    player_data = pd.read_csv(PLAYER_FILE)
    goalkeeper_data = pd.read_csv(GOALKEEPER_FILE)
    validate_player_data(player_data)

    print(f"Player-role dataset shape: {player_data.shape}")
    print(f"Goalkeeper dataset shape: {goalkeeper_data.shape}")

    all_lineup_rows = []
    all_summaries = []
    outfield_lineups = {}

    for formation_name, role_slots in FORMATIONS.items():
        outfield_lineup = optimize_outfield_lineup(player_data, formation_name, role_slots)
        goalkeepers = select_goalkeepers(goalkeeper_data, formation_name)
        summary = summarize_formation(formation_name, outfield_lineup, goalkeepers)
        outfield_lineups[formation_name] = outfield_lineup
        all_lineup_rows.extend([goalkeepers.iloc[[0]], outfield_lineup, goalkeepers.iloc[[1]]])
        all_summaries.append(summary)
        print_formation(formation_name, outfield_lineup, goalkeepers, summary)

        safe_name = formation_name.replace("-", "_")
        formation_output = pd.concat(
            [goalkeepers.iloc[[0]], outfield_lineup, goalkeepers.iloc[[1]]],
            ignore_index=True,
        )
        formation_output.to_csv(RESULTS_DIR / f"best_{safe_name}_lineup.csv", index=False)

    combined_lineups = pd.concat(all_lineup_rows, ignore_index=True)
    summary_data = pd.DataFrame(all_summaries)
    overlap_data = create_player_overlap(outfield_lineups)

    combined_output = PROCESSED_DIR / "eafc26_best_independent_formation_lineups.csv"
    combined_lineups.to_csv(combined_output, index=False)
    summary_data.to_csv(RESULTS_DIR / "formation_lineup_summary.csv", index=False)
    overlap_data.to_csv(RESULTS_DIR / "starting_outfield_player_overlap.csv", index=False)

    print("\nFORMATION COMPARISON")
    print("--------------------")
    print(summary_data.to_string(index=False))
    print("\nPlayers selected in more than one formation:")
    print(overlap_data[overlap_data["Number of Formations"] > 1].to_string(index=False))
    print(f"\nCombined lineup dataset saved to:\n{combined_output}")
    print(f"\nStage 4.2 results saved to:\n{RESULTS_DIR}")
    print("\nStage 4.2 independent formation lineups complete!")

if __name__ == "__main__":
    main()