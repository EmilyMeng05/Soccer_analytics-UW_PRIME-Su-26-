import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


# STAGE 4.5: ROBUSTNESS SHIFT FIGURE
#
# Core argument this figure exists to make:
#   Picking the highest-OVR player at each slot is NOT the same as picking
#   the player who actually fits the team. Some of the highest-rated
#   players in the entire dataset (Mbappé, Dembélé, Rodri) are selected in
#   every single formation under pure-OVR selection, but DROP OUT ENTIRELY
#   once role fit is taken into account. Meanwhile players invisible to the
#   OVR-only method (Pedri, Caicedo) become near-constant picks.
#
# This script reads the already-computed Stage 4.3 robustness-change table
# and turns it into a single, presentation-ready slopegraph.


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_IN = (
    PROJECT_ROOT
    / "results"
    / "stage_4_3_formation_robustness"
    / "player_robustness_change.csv"
)

RESULTS_OUT = PROJECT_ROOT / "graph"
RESULTS_OUT.mkdir(parents=True, exist_ok=True)

N_FORMATIONS = 7  # 4-3-3, 4-2-3-1, 4-4-2, 4-1-2-1-2, 3-5-2, 3-4-3, 5-3-2


# ---------------------------------------------------------------
# Load real results (no hardcoded numbers)
# ---------------------------------------------------------------

df = pd.read_csv(RESULTS_IN)

df = df.rename(
    columns={
        "Name": "name",
        "OVR Formation Robustness": "ovr_robustness",
        "Role-Adjusted Formation Robustness": "role_robustness",
    }
)[["name", "ovr_robustness", "role_robustness"]]


# ---------------------------------------------------------------
# Group players who share an identical (ovr, role) trajectory,
# so the chart shows one clean line instead of overlapping duplicates
# ---------------------------------------------------------------

grouped = (
    df.groupby(["ovr_robustness", "role_robustness"])["name"]
    .apply(list)
    .reset_index()
)

lines = [
    (", ".join(row["name"]), row["ovr_robustness"], row["role_robustness"], row["name"])
    for _, row in grouped.iterrows()
]


# ---------------------------------------------------------------
# Plot
# ---------------------------------------------------------------

COLOR_GAIN = "#1b8a7a"
COLOR_LOSS = "#d1495b"
COLOR_UNCHANGED = "#6c757d"

BIG_MOVE_THRESHOLD = 4 / N_FORMATIONS  # highlight swings of 4+ formations

fig, ax = plt.subplots(figsize=(10, 9))

for label, o, r, names in lines:
    diff = r - o
    is_big_move = abs(diff) >= BIG_MOVE_THRESHOLD - 1e-9
    color = COLOR_GAIN if r > o else (COLOR_LOSS if r < o else COLOR_UNCHANGED)
    ax.plot(
        [0, 1], [o, r],
        color=color,
        marker="o",
        markersize=6.5 if is_big_move else 4,
        linewidth=3.2 if is_big_move else 1.4,
        alpha=1.0 if is_big_move else 0.5,
        zorder=4 if is_big_move else 2,
    )


def place_labels(x_pos, use_ovr_side, ha, skip_zero):
    """Place non-overlapping name labels on one side of the chart."""
    ordered = sorted(lines, key=lambda t: -(t[1] if use_ovr_side else t[2]))
    used_y = []
    for label, o, r, names in ordered:
        y = o if use_ovr_side else r
        if skip_zero and y == 0:
            continue
        y_adj = y
        for prev_y in used_y:
            if abs(y_adj - prev_y) < 0.05:
                y_adj = prev_y - 0.05
        used_y.append(y_adj)

        diff = r - o
        is_big_move = abs(diff) >= BIG_MOVE_THRESHOLD - 1e-9
        ax.text(
            x_pos, y_adj, label,
            va="center", ha=ha,
            fontsize=9.6 if is_big_move else 8.4,
            fontweight="bold" if is_big_move else "normal",
        )


# Left side: skip players who start at 0 (never selected under OVR-only) —
# they're the "gainers," and their meaningful value is where they land, shown on the right.
place_labels(-0.04, use_ovr_side=True, ha="right", skip_zero=True)
# Right side: show everyone, including the dramatic drop-to-zero cluster — that IS the headline.
place_labels(1.04, use_ovr_side=False, ha="left", skip_zero=False)

ax.set_xlim(-1.75, 2.5)
ax.set_ylim(-0.32, 1.25)
ax.set_xticks([])
ax.text(0, -0.24, "OVR-Only\nBaseline", ha="center", va="top", fontsize=12, fontweight="bold")
ax.text(1, -0.24, "Role-Adjusted\n(\u03bb = 0.25)", ha="center", va="top", fontsize=12, fontweight="bold")
ax.axhline(0, color="#ddd", lw=0.8, zorder=0)
ax.set_ylabel(f"Formation robustness (fraction of {N_FORMATIONS} formations selected)", fontsize=10.5)
for spine in ["top", "right", "bottom"]:
    ax.spines[spine].set_visible(False)
ax.set_title(
    "Who Stays, Who Goes: Formation Robustness Before and After Role Fit",
    fontsize=13.5, fontweight="bold", pad=16,
)

# ---------------------------------------------------------------
# Callout: put the paper's main argument directly on the figure
# ---------------------------------------------------------------
ax.annotate(
    "Highest-rated \u2260 best fit:\nsome of the top-rated players\nin the dataset are selected in\nALL 7 formations by OVR alone\u2014\nbut 0 formations once role fit\nis considered.",
    xy=(1, 0), xycoords="data",
    xytext=(1.85, 0.42), textcoords="data",
    fontsize=9.3, ha="left", va="center",
    bbox=dict(boxstyle="round,pad=0.5", fc="#fff3f3", ec=COLOR_LOSS, lw=1.2),
    arrowprops=dict(arrowstyle="->", color=COLOR_LOSS, lw=1.4, connectionstyle="arc3,rad=-0.25"),
)

legend_elems = [
    Line2D([0], [0], color=COLOR_GAIN, lw=3, label="Gained robustness"),
    Line2D([0], [0], color=COLOR_LOSS, lw=3, label="Lost robustness"),
    Line2D([0], [0], color=COLOR_UNCHANGED, lw=3, label="Unchanged (robust core)"),
]
ax.legend(
    handles=legend_elems, loc="lower center", bbox_to_anchor=(0.5, -0.17),
    ncol=3, frameon=False, fontsize=10.5,
)

plt.tight_layout()

png_path = RESULTS_OUT / "fig_robustness_shift.png"
pdf_path = RESULTS_OUT / "fig_robustness_shift.pdf"
plt.savefig(png_path, dpi=220, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")