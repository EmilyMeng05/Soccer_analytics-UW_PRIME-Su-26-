"""Stage 4.5: compare the optimized 4-3-3 with independent role specialists.

Run from the project with:
    python3 src/stage4_5_ovr_baseline_visualization.py

The baseline selects the player with the highest position-specific Role
Attribute Score for every conventional 4-3-3 slot, while requiring unique
players. This is more informative than an OVR-only baseline because different
roles emphasize different football attributes. Final suitability breaks ties.
"""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
STAGE44_DIR = PROJECT_ROOT / "results" / "stage_4_4"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage_4_5"

ROLE_FILE = PROCESSED_DIR / "eafc26_player_role_suitability.csv"
GK_FILE = PROCESSED_DIR / "eafc26_goalkeepers_with_styles.csv"
OPTIMIZED_FILE = STAGE44_DIR / "default_starting_4_3_3.csv"

# Conventional positions make "best player at each position" unambiguous.
# Each tuple is: formation slot, accepted official positions, evaluated role.
OVR_BASELINE_SLOTS = [
    ("CB1", {"Center Back"}, "Center Back"),
    ("CB2", {"Center Back"}, "Center Back"),
    ("LB", {"Left Back"}, "Full Back"),
    ("RB", {"Right Back"}, "Full Back"),
    ("DM", {"Defensive Midfielder"}, "Defensive Midfielder"),
    ("CM1", {"Central Midfielder"}, "Central Midfielder"),
    ("CM2", {"Central Midfielder"}, "Central Midfielder"),
    ("LW", {"Left Winger"}, "Winger"),
    ("RW", {"Right Winger"}, "Winger"),
    ("ST", {"Striker"}, "Striker"),
]


def find_column(data, choices):
    for column in choices:
        if column in data.columns:
            return column
    raise KeyError(f"Could not find any of these columns: {choices}")


def load_inputs():
    missing = [path for path in (ROLE_FILE, GK_FILE, OPTIMIZED_FILE) if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Run Stages 4.1 and 4.4 first. Missing:\n{formatted}")

    roles = pd.read_csv(ROLE_FILE)
    goalkeepers = pd.read_csv(GK_FILE)
    optimized = pd.read_csv(OPTIMIZED_FILE)
    return roles, goalkeepers, optimized


def select_role_specialist_baseline(roles):
    """Maximize position-specific attribute grades with unique players."""
    candidates = roles.copy()
    candidates = candidates.sort_values(
        ["OVR", "Player Role Suitability"], ascending=False
    ).drop_duplicates(["Name", "Position", "Evaluated Role"])
    player_names = candidates["Name"].drop_duplicates().tolist()
    player_index = {name: index for index, name in enumerate(player_names)}

    # Role Attribute Score dominates. Final suitability only resolves ties.
    scores = np.full((len(OVR_BASELINE_SLOTS), len(player_names)), -1_000_000.0)
    row_lookup = {}
    for slot_i, (_, accepted_positions, role) in enumerate(OVR_BASELINE_SLOTS):
        eligible = candidates[
            candidates["Position"].isin(accepted_positions)
            & (candidates["Evaluated Role"] == role)
        ]
        for row_i, row in eligible.iterrows():
            player_i = player_index[row["Name"]]
            score = (
                1000.0 * float(row["Role Attribute Score"])
                + float(row["Player Role Suitability"])
            )
            if score > scores[slot_i, player_i]:
                scores[slot_i, player_i] = score
                row_lookup[(slot_i, player_i)] = row_i

    slot_indices, player_indices = linear_sum_assignment(scores, maximize=True)
    output = []
    for slot_i, player_i in zip(slot_indices, player_indices):
        if scores[slot_i, player_i] < 0:
            raise ValueError(f"No eligible role-specialist assignment for {OVR_BASELINE_SLOTS[slot_i][0]}")
        row = candidates.loc[row_lookup[(slot_i, player_i)]]
        slot, _, role = OVR_BASELINE_SLOTS[slot_i]
        output.append({
            "Formation Slot": slot,
            "Assigned Role": role,
            "Name": row["Name"],
            "Official Position": row["Position"],
            "OVR": float(row["OVR"]),
            "Position Fit Score": float(row["Position Fit Score"]),
            "Role Attribute Score": float(row["Role Attribute Score"]),
            "Selection Score": float(row["Player Role Suitability"]),
        })
    order = {slot: i for i, (slot, _, _) in enumerate(OVR_BASELINE_SLOTS)}
    result = pd.DataFrame(output)
    return result.sort_values("Formation Slot", key=lambda s: s.map(order)).reset_index(drop=True)


def select_role_specialist_goalkeeper(goalkeepers):
    name_col = find_column(goalkeepers, ["Name", "Player Name", "LongName"])
    position_col = find_column(goalkeepers, ["Position", "Best Position"])
    skill_col = find_column(
        goalkeepers,
        ["Goalkeeper Skill", "GoalkeeperSkill", "Goalkeeper Skill Score"],
    )
    ranked = goalkeepers.sort_values(skill_col, ascending=False).drop_duplicates(name_col)
    row = ranked.iloc[0]
    return {
        "Formation Slot": "GK",
        "Assigned Role": "Goalkeeper",
        "Name": row[name_col],
        "Official Position": row[position_col],
        "OVR": float(row["OVR"]),
        "Position Fit Score": 100.0,
        "Role Attribute Score": float(row[skill_col]),
        "Selection Score": float(row[skill_col]),
    }


def prepare_optimized(optimized, goalkeepers):
    required = ["Formation Slot", "Assigned Role", "Name", "Official Position", "OVR"]
    missing = [column for column in required if column not in optimized.columns]
    if missing:
        raise ValueError(f"Stage 4.4 lineup is missing columns: {missing}")
    result = optimized.copy()
    if "Selection Score" not in result.columns and "Player Role Suitability" in result.columns:
        result["Selection Score"] = result["Player Role Suitability"]
    if "Position Fit Score" not in result.columns:
        result["Position Fit Score"] = np.nan
    if "Role Attribute Score" not in result.columns:
        result["Role Attribute Score"] = np.nan

    if not (result["Formation Slot"] == "GK").any():
        # Stage 4.4 prints the goalkeeper separately, so recover Alisson's row.
        name_col = find_column(goalkeepers, ["Name", "Player Name", "LongName"])
        position_col = find_column(goalkeepers, ["Position", "Best Position"])
        alisson = goalkeepers[goalkeepers[name_col] == "Alisson"]
        if alisson.empty:
            raise ValueError("Alisson was not found in the goalkeeper file.")
        row = alisson.iloc[0]
        skill_col = find_column(
            goalkeepers,
            ["Goalkeeper Skill", "GoalkeeperSkill", "Goalkeeper Skill Score"],
        )
        gk = pd.DataFrame([{
            "Formation Slot": "GK", "Assigned Role": "Goalkeeper",
            "Name": "Alisson", "Official Position": row[position_col],
            "OVR": float(row["OVR"]), "Position Fit Score": 100.0,
            "Role Attribute Score": float(row[skill_col]),
            "Selection Score": float(row[skill_col]),
        }])
        result = pd.concat([gk, result], ignore_index=True)
    return result


def summarize(label, lineup):
    outfield = lineup[lineup["Assigned Role"] != "Goalkeeper"]
    return {
        "Team": label,
        "Average OVR": lineup["OVR"].mean(),
        "Average Role Suitability": outfield["Selection Score"].mean(),
        "Average Role Attribute Score": outfield["Role Attribute Score"].mean(),
        "Average Position Fit": outfield["Position Fit Score"].mean(),
    }


def comparison_rows(optimized, baseline):
    role_order = ["Goalkeeper", "Center Back", "Full Back", "Defensive Midfielder",
                  "Central Midfielder", "Winger", "Striker"]
    rows = []
    for role in role_order:
        a = optimized[optimized["Assigned Role"] == role].sort_values("Name")
        b = baseline[baseline["Assigned Role"] == role].sort_values("Name")
        a_names, b_names = set(a["Name"]), set(b["Name"])
        rows.append({
            "Role": role,
            "Optimized Players": ", ".join(a["Name"]),
            "Independent Specialists": ", ".join(b["Name"]),
            "Optimized Role Grade": a["Role Attribute Score"].mean(),
            "Specialist Role Grade": b["Role Attribute Score"].mean(),
            "Same Selection": a_names == b_names,
        })
    return pd.DataFrame(rows)


def bar(value, maximum, css_class):
    width = 0 if pd.isna(value) else max(0, min(100, value / maximum * 100))
    label = "—" if pd.isna(value) else f"{value:.2f}"
    return f'<div class="bar-track"><div class="bar {css_class}" style="width:{width:.2f}%"></div></div><strong>{label}</strong>'


def build_html(summary, differences):
    optimized = summary.iloc[0]
    baseline = summary.iloc[1]
    metrics = [
        ("Average Role Attribute Score", 100),
        ("Average Role Suitability", 100),
        ("Average Position Fit", 100),
    ]
    metric_html = []
    for metric, maximum in metrics:
        metric_html.append(f"""
        <section class="metric">
          <h2>{escape(metric)}</h2>
          <div class="bar-row"><span>Optimized</span>{bar(optimized[metric], maximum, 'optimized')}</div>
          <div class="bar-row"><span>Specialists</span>{bar(baseline[metric], maximum, 'baseline')}</div>
        </section>""")

    shared = int(differences["Same Selection"].sum())
    changed = len(differences) - shared
    table_rows = []
    for _, row in differences.iterrows():
        status = "Same" if row["Same Selection"] else "Different"
        table_rows.append(
            f"<tr><td>{escape(row['Role'])}</td><td>{escape(row['Optimized Players'])}</td>"
            f"<td>{escape(row['Independent Specialists'])}</td>"
            f"<td>{row['Optimized Role Grade']:.2f}</td><td>{row['Specialist Role Grade']:.2f}</td>"
            f"<td><span class='pill'>{status}</span></td></tr>"
        )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Optimized Team vs Independent Position Specialists</title>
<style>
:root{{--bg:#f7f7f4;--panel:#fff;--text:#17211b;--muted:#637068;--line:#dfe5e0;--opt:#207a53;--base:#68717a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:16px/1.45 system-ui,-apple-system,sans-serif}}
main{{max-width:1050px;margin:auto;padding:34px 22px}} h1{{font-size:clamp(26px,4vw,42px);margin:0 0 6px}} .subtitle{{color:var(--muted);margin:0 0 28px}}
.headline{{display:grid;grid-template-columns:repeat(2,1fr);gap:14px;margin-bottom:26px}} .card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}}
.big{{font-size:30px;font-weight:700}} .small{{color:var(--muted)}} .charts{{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;margin-bottom:28px}}
.metric h2{{font-size:16px;margin:0 0 10px}} .bar-row{{display:grid;grid-template-columns:90px 1fr 55px;gap:10px;align-items:center;margin:9px 0}}
.bar-track{{height:14px;background:var(--line);border-radius:20px;overflow:hidden}} .bar{{height:100%}} .optimized{{background:var(--opt)}} .baseline{{background:var(--base)}}
table{{width:100%;border-collapse:collapse;background:var(--panel)}} th,td{{padding:12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:var(--muted);font-size:13px}}
.table-wrap{{overflow-x:auto;border:1px solid var(--line);border-radius:14px}} .pill{{font-size:13px;color:var(--muted)}} .note{{margin-top:18px;color:var(--muted)}}
@media(max-width:700px){{.headline,.charts{{grid-template-columns:1fr}} .bar-row{{grid-template-columns:82px 1fr 50px}}}}
</style></head><body><main>
<h1>Optimized 4-3-3 vs independent position specialists</h1>
<p class="subtitle">Specialists maximize the position-specific attribute grade. The optimized XI also considers OVR, position fit, formation flexibility, and squad continuity.</p>
<div class="headline">
 <div class="card"><div class="big">{shared} of {len(differences)}</div><div class="small">role groups select the same player set</div></div>
 <div class="card"><div class="big">{changed}</div><div class="small">role groups change when specialists are chosen independently</div></div>
</div>
<div class="charts">{''.join(metric_html)}</div>
<div class="table-wrap"><table><thead><tr><th>Role</th><th>Optimized method</th><th>Independent specialists</th><th>Optimized grade</th><th>Specialist grade</th><th>Result</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<p class="note">The role grade is computed before OVR and position fit are added. The optimized method uses final suitability = 40% role grade + 35% OVR + 25% position fit. Role-level rows compare player sets, so CB1/CB2 ordering does not create a false difference.</p>
</main></body></html>"""


def main():
    roles, goalkeepers, optimized_raw = load_inputs()
    optimized = prepare_optimized(optimized_raw, goalkeepers)
    baseline_outfield = select_role_specialist_baseline(roles)
    baseline = pd.concat(
        [pd.DataFrame([select_role_specialist_goalkeeper(goalkeepers)]), baseline_outfield],
        ignore_index=True,
    )

    summary = pd.DataFrame([
        summarize("Optimized role-suitability XI", optimized),
        summarize("Independent position-specialist XI", baseline),
    ])
    differences = comparison_rows(optimized, baseline)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(RESULTS_DIR / "role_specialist_4_3_3_lineup.csv", index=False)
    summary.to_csv(RESULTS_DIR / "optimized_vs_role_specialist_summary.csv", index=False)
    differences.to_csv(RESULTS_DIR / "optimized_vs_role_specialist_differences.csv", index=False)
    html_path = RESULTS_DIR / "optimized_vs_role_specialist_visualization.html"
    html_path.write_text(build_html(summary, differences), encoding="utf-8")

    print("STAGE 4.5: OPTIMIZED TEAM VS POSITION SPECIALISTS")
    print("=" * 55)
    print("\nTEAM SCORE COMPARISON")
    print(summary.round(2).to_string(index=False))
    print("\nPLAYER DIFFERENCES BY ROLE")
    print(differences.to_string(index=False))
    print(f"\nVisualization saved to:\n{html_path}")


if __name__ == "__main__":
    main()