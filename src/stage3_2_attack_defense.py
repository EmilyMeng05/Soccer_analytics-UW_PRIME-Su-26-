import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path


# STAGE 3.2: ATTACKING VS DEFENSIVE CONTRIBUTION
#
# Stage 3.1 used K-Means clustering to discover natural player archetypes.
#
# In this stage, I want to ask a different question:
#
# How much does each player's attribute profile lean toward attacking,
# defending, or contributing strongly to both?
#
# Instead of clustering players, I will reduce the six main EAFC attributes
# into two easier-to-interpret measurements:
#
# Attacking Contribution
# Defensive Contribution
#
# Every player can then be represented as a point in a two-dimensional
# attacking-versus-defensive space.


# Read the cleaned outfield player dataset created in Stage 1.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA = PROJECT_ROOT / "data"

RESULTS = (
    PROJECT_ROOT
    / "results"
    / "stage_3_2"
)

RESULTS.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    DATA
    / "processed"
    / "cleaned_eafc26_outfield_players.csv"
)


# For attacking contribution, I will begin with four EAFC attributes:
df["AttackingContribution"] = (
    df["PAC"]
    + df["SHO"]
    + df["PAS"]
    + df["DRI"]
) / 4


# For defensive contribution, I will begin with:
#
# DEF = Defending
# PHY = Physicality
#
# These are also given equal weight for the first version.
#
# Physicality does not belong exclusively to defending, so this definition
# could be adjusted later if necessary.

df["DefensiveContribution"] = (
    df["DEF"]
    + df["PHY"]
) / 2


# OVR will continue to represent overall player quality.
#
# This allows us to distinguish between players who are mathematically
# balanced and players who are both balanced and highly rated.

df["OverallQuality"] = df["OVR"]


# Measure the difference between attacking and defensive contribution.
#
# A smaller BalanceGap means the player's attacking and defensive
# contribution scores are more similar.
#
# However, a small gap does NOT automatically mean the player is strong.
#
# For example:
#
# Attack = 50
# Defense = 50
#
# is perfectly balanced, but the player may still be relatively weak.

df["BalanceGap"] = abs(
    df["AttackingContribution"]
    - df["DefensiveContribution"]
)


# Calculate dataset averages.
#
# These averages provide a simple reference point for deciding whether
# a player's attacking or defensive contribution is relatively high.

attack_mean = df["AttackingContribution"].mean()
defense_mean = df["DefensiveContribution"].mean()

print("\nAverage attacking contribution:")
print(round(attack_mean, 2))

print("\nAverage defensive contribution:")
print(round(defense_mean, 2))


# Divide players into broad contribution categories.
#
# Players with high attacking and defensive contribution are above
# the dataset average in both dimensions.
#
# This does NOT mean they are necessarily perfectly balanced.
# It simply means they contribute relatively strongly on both sides.

high_both = df[
    (df["AttackingContribution"] >= attack_mean)
    &
    (df["DefensiveContribution"] >= defense_mean)
].copy()


# Players with high attacking contribution have above-average attacking
# contribution but below-average defensive contribution.

attack_focused = df[
    (df["AttackingContribution"] >= attack_mean)
    &
    (df["DefensiveContribution"] < defense_mean)
].copy()


# Players with high defensive contribution have above-average defensive
# contribution but below-average attacking contribution.

defense_focused = df[
    (df["DefensiveContribution"] >= defense_mean)
    &
    (df["AttackingContribution"] < attack_mean)
].copy()


# Players below both averages are not necessarily bad players.
#
# They simply fall below the current dataset averages under this particular
# attack-defense representation.

lower_both = df[
    (df["AttackingContribution"] < attack_mean)
    &
    (df["DefensiveContribution"] < defense_mean)
].copy()


# Print the highest-rated players who have high attacking
# and defensive contribution.

top_high_both = (
    high_both
    .sort_values(
        "OVR",
        ascending=False
    )
    [
        [
            "Name",
            "Position",
            "Team",
            "OVR",
            "AttackingContribution",
            "DefensiveContribution",
            "BalanceGap"
        ]
    ]
    .head(25)
)

print(
    "\nPlayers with High Attacking and Defensive Contribution:"
)

print(top_high_both)


# Print the players with the strongest attacking contribution.

top_attack = (
    attack_focused
    .sort_values(
        "AttackingContribution",
        ascending=False
    )
    [
        [
            "Name",
            "Position",
            "Team",
            "OVR",
            "AttackingContribution",
            "DefensiveContribution"
        ]
    ]
    .head(20)
)

print(
    "\nPlayers with High Attacking Contribution:"
)

print(top_attack)


# Print the players with the strongest defensive contribution.

top_defense = (
    defense_focused
    .sort_values(
        "DefensiveContribution",
        ascending=False
    )
    [
        [
            "Name",
            "Position",
            "Team",
            "OVR",
            "AttackingContribution",
            "DefensiveContribution"
        ]
    ]
    .head(20)
)

print(
    "\nPlayers with High Defensive Contribution:"
)

print(top_defense)


# I also want to investigate players whose attacking and defensive
# contributions are especially similar.
#
# This is a separate concept from having high contribution in both areas.
#
# To avoid highlighting very low-rated players simply because their
# two scores happen to match, I will only consider players with OVR >= 75.

balanced_quality_players = df[
    df["OVR"] >= 75
].copy()

smallest_gap = (
    balanced_quality_players
    .sort_values(
        "BalanceGap",
        ascending=True
    )
    [
        [
            "Name",
            "Position",
            "Team",
            "OVR",
            "AttackingContribution",
            "DefensiveContribution",
            "BalanceGap"
        ]
    ]
    .head(25)
)

print(
    "\nPlayers with the Smallest Attack-Defense Difference:"
)

print(smallest_gap)


# Create the main static visualization.
#
# Every player appears as a point.
#
# The vertical line represents average attacking contribution.
# The horizontal line represents average defensive contribution.
#
# These two lines divide the graph into four broad regions.

plt.figure(figsize=(10,8))

plt.scatter(
    df["AttackingContribution"],
    df["DefensiveContribution"],
    alpha=0.25,
    s=12
)

plt.axvline(
    attack_mean,
    linestyle="--"
)

plt.axhline(
    defense_mean,
    linestyle="--"
)

plt.xlabel("Attacking Contribution")
plt.ylabel("Defensive Contribution")

plt.title(
    "Attacking vs Defensive Contribution of EAFC26 Players"
)

plt.tight_layout()

plt.savefig(
    RESULTS
    / "attack_defense_player_map.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# For a clearer visualization, I also want to highlight only a few
# recognizable players.
#
# Instead of labeling thousands of players, I will highlight the
# five highest-rated players from three important categories:
#
# High Attacking Contribution
# High Defensive Contribution
# High Attacking and Defensive Contribution
#
# The rest of the players remain visible in the background.


# Select the five highest-rated players from each category.

top_5_attack = (
    attack_focused
    .sort_values(
        "OVR",
        ascending=False
    )
    .head(5)
)

top_5_defense = (
    defense_focused
    .sort_values(
        "OVR",
        ascending=False
    )
    .head(5)
)


# For the high-attacking-and-defensive group, I also want to avoid
# highlighting players who technically qualify but lean very strongly
# toward only one side.
#
# Therefore, among these players, I will first prefer players with
# relatively smaller attack-defense differences.
#
# I still require strong overall quality by filtering to OVR >= 80.

high_both_for_highlight = high_both[
    high_both["OVR"] >= 80
].copy()

top_5_both = (
    high_both_for_highlight
    .sort_values(
        ["OVR", "BalanceGap"],
        ascending=[False, True]
    )
    .head(5)
)


# Create a cleaner static visualization.
#
# All players appear lightly in the background.
# Only the selected top players are strongly highlighted and labeled.

plt.figure(figsize=(11,8))

plt.scatter(
    df["AttackingContribution"],
    df["DefensiveContribution"],
    alpha=0.15,
    s=10,
    label="All Players"
)


# Highlight players with high attacking contribution.

plt.scatter(
    top_5_attack["AttackingContribution"],
    top_5_attack["DefensiveContribution"],
    s=80,
    label="High Attacking Contribution"
)


# Highlight players with high defensive contribution.

plt.scatter(
    top_5_defense["AttackingContribution"],
    top_5_defense["DefensiveContribution"],
    s=80,
    label="High Defensive Contribution"
)


# Highlight players with high attacking and defensive contribution.

plt.scatter(
    top_5_both["AttackingContribution"],
    top_5_both["DefensiveContribution"],
    s=80,
    label="High Attacking and Defensive Contribution"
)


# Label the five highlighted attacking players.

for _, row in top_5_attack.iterrows():

    plt.annotate(
        row["Name"],
        (
            row["AttackingContribution"],
            row["DefensiveContribution"]
        ),
        xytext=(5,5),
        textcoords="offset points",
        fontsize=8
    )


# Label the five highlighted defensive players.

for _, row in top_5_defense.iterrows():

    plt.annotate(
        row["Name"],
        (
            row["AttackingContribution"],
            row["DefensiveContribution"]
        ),
        xytext=(5,5),
        textcoords="offset points",
        fontsize=8
    )


# Label the five players with high attacking and defensive contribution.

for _, row in top_5_both.iterrows():

    plt.annotate(
        row["Name"],
        (
            row["AttackingContribution"],
            row["DefensiveContribution"]
        ),
        xytext=(5,5),
        textcoords="offset points",
        fontsize=8
    )


plt.axvline(
    attack_mean,
    linestyle="--"
)

plt.axhline(
    defense_mean,
    linestyle="--"
)

plt.xlabel("Attacking Contribution")
plt.ylabel("Defensive Contribution")

plt.title(
    "Attacking vs Defensive Contribution with Selected Players"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULTS
    / "attack_defense_highlighted_players.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Now create an interactive version of the visualization.
#
# This graph keeps every player in the dataset.
#
# A viewer can hover over any point and see:
#
# Name
# Position
# Team
# Overall Rating
# Attacking Contribution
# Defensive Contribution
#
# Only the selected top players receive permanent visible labels.

fig = px.scatter(
    df,
    x="AttackingContribution",
    y="DefensiveContribution",
    hover_name="Name",
    hover_data={
        "Position": True,
        "Team": True,
        "OVR": True,
        "AttackingContribution": ":.2f",
        "DefensiveContribution": ":.2f"
    },
    opacity=0.20,
    title="Interactive Attacking vs Defensive Contribution of EAFC26 Players"
)


# Make background points smaller.

fig.update_traces(
    marker=dict(
        size=5
    )
)


# Add the average attacking contribution reference line.

fig.add_vline(
    x=attack_mean,
    line_dash="dash",
    annotation_text="Average Attacking Contribution"
)


# Add the average defensive contribution reference line.

fig.add_hline(
    y=defense_mean,
    line_dash="dash",
    annotation_text="Average Defensive Contribution"
)


# Add the top five attacking players.

fig.add_trace(
    go.Scatter(
        x=top_5_attack["AttackingContribution"],
        y=top_5_attack["DefensiveContribution"],
        mode="markers+text",
        text=top_5_attack["Name"],
        textposition="top center",
        name="High Attacking Contribution",
        marker=dict(
            size=12
        ),
        customdata=top_5_attack[
            [
                "Position",
                "Team",
                "OVR"
            ]
        ],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Position: %{customdata[0]}<br>"
            "Team: %{customdata[1]}<br>"
            "OVR: %{customdata[2]}<br>"
            "Attack: %{x:.2f}<br>"
            "Defense: %{y:.2f}"
            "<extra></extra>"
        )
    )
)


# Add the top five defensive players.

fig.add_trace(
    go.Scatter(
        x=top_5_defense["AttackingContribution"],
        y=top_5_defense["DefensiveContribution"],
        mode="markers+text",
        text=top_5_defense["Name"],
        textposition="top center",
        name="High Defensive Contribution",
        marker=dict(
            size=12
        ),
        customdata=top_5_defense[
            [
                "Position",
                "Team",
                "OVR"
            ]
        ],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Position: %{customdata[0]}<br>"
            "Team: %{customdata[1]}<br>"
            "OVR: %{customdata[2]}<br>"
            "Attack: %{x:.2f}<br>"
            "Defense: %{y:.2f}"
            "<extra></extra>"
        )
    )
)


# Add the top five players who have high attacking
# and defensive contribution.

fig.add_trace(
    go.Scatter(
        x=top_5_both["AttackingContribution"],
        y=top_5_both["DefensiveContribution"],
        mode="markers+text",
        text=top_5_both["Name"],
        textposition="top center",
        name="High Attacking and Defensive Contribution",
        marker=dict(
            size=12
        ),
        customdata=top_5_both[
            [
                "Position",
                "Team",
                "OVR",
                "BalanceGap"
            ]
        ],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Position: %{customdata[0]}<br>"
            "Team: %{customdata[1]}<br>"
            "OVR: %{customdata[2]}<br>"
            "Attack: %{x:.2f}<br>"
            "Defense: %{y:.2f}<br>"
            "Attack-Defense Difference: %{customdata[3]:.2f}"
            "<extra></extra>"
        )
    )
)


# Give the interactive graph clearer labels.

fig.update_layout(
    xaxis_title="Attacking Contribution",
    yaxis_title="Defensive Contribution",
    legend_title="Highlighted Player Groups"
)


# Save the interactive graph as an HTML file.
#
# This file preserves hovering, zooming, and panning.
#
# Someone viewing the project can open this file in their browser
# and explore individual players.

fig.write_html(
    RESULTS
    / "interactive_attack_defense_player_map.html"
)


# Automatically open the interactive graph when the script runs.

fig.show()

# Save the complete Stage 3.2 dataset.
#
# Every player now has:
#
# Attacking Contribution
# Defensive Contribution
# Overall Quality
# Balance Gap

df.to_csv(
    RESULTS
    / "player_balance_results.csv",
    index=False
)


# Save useful subsets separately.

top_high_both.to_csv(
    RESULTS
    / "high_attacking_and_defensive_players.csv",
    index=False
)


top_attack.to_csv(
    RESULTS
    / "high_attacking_players.csv",
    index=False
)


top_defense.to_csv(
    RESULTS
    / "high_defensive_players.csv",
    index=False
)


smallest_gap.to_csv(
    RESULTS
    / "smallest_attack_defense_gap.csv",
    index=False
)

print("stage 3_2 complete")