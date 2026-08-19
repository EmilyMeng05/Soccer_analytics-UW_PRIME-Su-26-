from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


# STAGE 3.7: GOALKEEPER STYLE ANALYSIS
#
# Goalkeepers were excluded from the outfield-player experiments
# because they use a different set of attributes.
#
# This stage analyzes:
#
# GK Diving
# GK Handling
# GK Kicking
# GK Positioning
# GK Reflexes
#
# PCA is used to visualize goalkeeper attribute profiles.
#
# We also create two interpretable measurements:
#
# Protection Contribution
# Distribution Contribution
#
# This allows us to identify goalkeepers who may complement different
# types of outfield lineups.


# Define the project paths.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA = PROJECT_ROOT / "data"

RESULTS = (
    PROJECT_ROOT
    / "results"
    / "stage_3_7"
)

RESULTS.mkdir(parents=True, exist_ok=True)

INPUT_FILE = (
    DATA
    / "processed"
    / "cleaned_eafc26_goalkeepers.csv"
)

OUTPUT_FILE = (
    DATA
    / "processed"
    / "eafc26_goalkeepers_with_styles.csv"
)


# Check that the goalkeeper dataset exists.

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        "Could not find the cleaned goalkeeper dataset.\n\n"
        f"Expected file:\n{INPUT_FILE}"
    )


# Read the cleaned goalkeeper dataset.

df = pd.read_csv(INPUT_FILE)

print("Dataset shape:", df.shape)


# Define the five goalkeeper attributes.

features = [
    "GK Diving",
    "GK Handling",
    "GK Kicking",
    "GK Positioning",
    "GK Reflexes"
]


# Check that every required attribute is available.

missing_columns = [
    column
    for column in features
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The following goalkeeper attributes are missing:\n"
        f"{missing_columns}"
    )


# Convert the goalkeeper attributes to numeric values.

for feature in features:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )

if "OVR" in df.columns:
    df["OVR"] = pd.to_numeric(
        df["OVR"],
        errors="coerce"
    )


# Remove goalkeepers who are missing one or more required attributes.

original_count = len(df)

df = df.dropna(
    subset=features
).copy()

removed_count = original_count - len(df)

print(
    "Goalkeepers removed because of missing attributes:",
    removed_count
)

print(
    "Goalkeepers remaining:",
    len(df)
)


# Calculate Protection Contribution.
#
# Protection Contribution measures the goalkeeper's ability to stop
# shots, control dangerous situations, and protect the goal.
#
# The four protection attributes receive equal weight.

df[
    "ProtectionContribution"
] = df[
    [
        "GK Diving",
        "GK Handling",
        "GK Positioning",
        "GK Reflexes"
    ]
].mean(axis=1)


# Calculate Distribution Contribution.
#
# GK Kicking describes the goalkeeper's ability to move the ball
# forward through passes, clearances, and goal kicks.
#
# It does not mean that the goalkeeper takes attacking shots.

df[
    "DistributionContribution"
] = df[
    "GK Kicking"
]


# Calculate overall Goalkeeper Skill using all five attributes.
#
# OVR remains separate because it is already EAFC's overall
# evaluation of the player.

df[
    "GoalkeeperSkill"
] = df[
    features
].mean(axis=1)


# Calculate the difference between protection and distribution.
#
# A smaller StyleGap indicates a more balanced goalkeeper profile.
#
# However, a small difference does not automatically mean that the
# goalkeeper has high overall ability.

df[
    "StyleGap"
] = abs(
    df["ProtectionContribution"]
    - df["DistributionContribution"]
)


# Calculate the dataset averages.
#
# These values divide the contribution graph into four goalkeeper
# style regions.

protection_mean = df[
    "ProtectionContribution"
].mean()

distribution_mean = df[
    "DistributionContribution"
].mean()

print("\nAverage protection contribution:")
print(round(protection_mean, 2))

print("\nAverage distribution contribution:")
print(round(distribution_mean, 2))


# Assign broad goalkeeper styles based on the average contribution
# values.

def assign_goalkeeper_style(row):
    high_protection = (
        row["ProtectionContribution"]
        >= protection_mean
    )

    high_distribution = (
        row["DistributionContribution"]
        >= distribution_mean
    )

    if high_protection and high_distribution:
        return "High Protection and Distribution"

    if high_protection:
        return "Protection Focused"

    if high_distribution:
        return "Distribution Focused"

    return "Lower Both"


df[
    "GoalkeeperStyle"
] = df.apply(
    assign_goalkeeper_style,
    axis=1
)


print("\nGoalkeepers per style:")

print(
    df[
        "GoalkeeperStyle"
    ].value_counts()
)


# Standardize the five goalkeeper attributes.
#
# Standardization prevents an attribute with a larger numerical spread
# from dominating PCA and similarity calculations.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df[
        features
    ]
)


# Fit PCA using all five possible components.
#
# This allows us to inspect how much variation is explained by every
# principal component.

pca_full = PCA()

pca_full.fit(
    X_scaled
)


explained_variance = pd.DataFrame({
    "PrincipalComponent": [
        f"PC{component_number}"
        for component_number in range(
            1,
            len(features) + 1
        )
    ],
    "ExplainedVarianceRatio":
        pca_full.explained_variance_ratio_,
    "CumulativeExplainedVariance":
        pca_full.explained_variance_ratio_.cumsum()
})


print("\nPCA explained variance:")

print(
    explained_variance.round(4)
)


explained_variance.to_csv(
    RESULTS / "goalkeeper_pca_explained_variance.csv",
    index=False
)


# Use two PCA components to create a visualization.
#
# PCA is only reducing the goalkeeper profiles for visualization.
#
# The similarity calculations later in the script still use all five
# standardized goalkeeper attributes.

pca = PCA(
    n_components=2
)

pca_components = pca.fit_transform(
    X_scaled
)

df["PC1"] = pca_components[:, 0]
df["PC2"] = pca_components[:, 1]


# Calculate the PCA loadings.
#
# The loadings show how strongly each goalkeeper attribute contributes
# to PC1 and PC2.
#
# We should inspect these values before deciding what the two
# components represent.

pca_loadings = pd.DataFrame(
    pca.components_.T,
    index=features,
    columns=[
        "PC1",
        "PC2"
    ]
)


print("\nPCA loadings:")

print(
    pca_loadings.round(3)
)


pca_loadings.to_csv(
    RESULTS / "goalkeeper_pca_loadings.csv"
)


pc1_variance = pca.explained_variance_ratio_[0]
pc2_variance = pca.explained_variance_ratio_[1]

print("\nPC1 explained variance:")
print(round(pc1_variance, 4))

print("\nPC2 explained variance:")
print(round(pc2_variance, 4))

print("\nPC1 and PC2 combined explained variance:")
print(
    round(
        pc1_variance + pc2_variance,
        4
    )
)


# Define the columns displayed in goalkeeper result tables.

display_columns = [
    column
    for column in [
        "Name",
        "Position",
        "Team",
        "League",
        "OVR",
        "GK Diving",
        "GK Handling",
        "GK Kicking",
        "GK Positioning",
        "GK Reflexes",
        "ProtectionContribution",
        "DistributionContribution",
        "GoalkeeperSkill",
        "StyleGap",
        "GoalkeeperStyle",
        "PC1",
        "PC2"
    ]
    if column in df.columns
]


pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    220
)


# Find the strongest overall goalkeepers.

top_overall_goalkeepers = (
    df.sort_values(
        [
            "GoalkeeperSkill",
            "OVR"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(25)
)


print("\nTop Goalkeepers by Overall Goalkeeper Skill:")

print(
    top_overall_goalkeepers[
        display_columns
    ].round(2)
)


# Find the strongest protection-focused goalkeepers.

protection_focused = df[
    df[
        "GoalkeeperStyle"
    ]
    == "Protection Focused"
].copy()


top_protection_goalkeepers = (
    protection_focused
    .sort_values(
        [
            "ProtectionContribution",
            "OVR"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(25)
)


print("\nTop Protection-Focused Goalkeepers:")

print(
    top_protection_goalkeepers[
        display_columns
    ].round(2)
)


# Find the strongest distribution-focused goalkeepers.

distribution_focused = df[
    df[
        "GoalkeeperStyle"
    ]
    == "Distribution Focused"
].copy()


top_distribution_goalkeepers = (
    distribution_focused
    .sort_values(
        [
            "DistributionContribution",
            "OVR"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(25)
)


print("\nTop Distribution-Focused Goalkeepers:")

print(
    top_distribution_goalkeepers[
        display_columns
    ].round(2)
)


# Find goalkeepers who are above average in both dimensions.

high_both_goalkeepers = df[
    df[
        "GoalkeeperStyle"
    ]
    == "High Protection and Distribution"
].copy()


top_high_both_goalkeepers = (
    high_both_goalkeepers
    .sort_values(
        [
            "GoalkeeperSkill",
            "StyleGap",
            "OVR"
        ],
        ascending=[
            False,
            True,
            False
        ]
    )
    .head(25)
)


print(
    "\nGoalkeepers with High Protection and Distribution:"
)

print(
    top_high_both_goalkeepers[
        display_columns
    ].round(2)
)


# Find highly rated goalkeepers with balanced protection and
# distribution scores.

balanced_quality_goalkeepers = df[
    df[
        "OVR"
    ]
    >= 75
].copy()


top_balanced_goalkeepers = (
    balanced_quality_goalkeepers
    .sort_values(
        [
            "StyleGap",
            "GoalkeeperSkill"
        ],
        ascending=[
            True,
            False
        ]
    )
    .head(25)
)


print("\nHighly Rated Goalkeepers with the Smallest Style Gap:")

print(
    top_balanced_goalkeepers[
        display_columns
    ].round(2)
)


# Save the goalkeeper ranking tables.

top_overall_goalkeepers[
    display_columns
].to_csv(
    RESULTS / "top_goalkeepers_overall.csv",
    index=False
)

top_protection_goalkeepers[
    display_columns
].to_csv(
    RESULTS / "top_goalkeepers_protection.csv",
    index=False
)

top_distribution_goalkeepers[
    display_columns
].to_csv(
    RESULTS / "top_goalkeepers_distribution.csv",
    index=False
)

top_high_both_goalkeepers[
    display_columns
].to_csv(
    RESULTS / "top_goalkeepers_high_both.csv",
    index=False
)

top_balanced_goalkeepers[
    display_columns
].to_csv(
    RESULTS / "top_goalkeepers_balanced.csv",
    index=False
)


# Create a static Protection-versus-Distribution graph.
#
# The vertical line represents average Distribution Contribution.
#
# The horizontal line represents average Protection Contribution.

plt.figure(
    figsize=(10, 8)
)

plt.scatter(
    df["DistributionContribution"],
    df["ProtectionContribution"],
    alpha=0.30,
    s=18
)

plt.axvline(
    distribution_mean,
    linestyle="--",
    color="gray"
)

plt.axhline(
    protection_mean,
    linestyle="--",
    color="gray"
)

plt.xlabel(
    "Distribution Contribution"
)

plt.ylabel(
    "Protection Contribution"
)

plt.title(
    "Goalkeeper Protection vs Distribution"
)

plt.tight_layout()

plt.savefig(
    RESULTS / "goalkeeper_protection_distribution_map.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Select the top five goalkeepers from the three main categories.
#
# These players will receive permanent labels on the interactive
# website.

top_5_protection = (
    protection_focused
    .sort_values(
        [
            "ProtectionContribution",
            "OVR"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(5)
)


top_5_distribution = (
    distribution_focused
    .sort_values(
        [
            "DistributionContribution",
            "OVR"
        ],
        ascending=[
            False,
            False
        ]
    )
    .head(5)
)


top_5_both = (
    high_both_goalkeepers
    .sort_values(
        [
            "GoalkeeperSkill",
            "StyleGap"
        ],
        ascending=[
            False,
            True
        ]
    )
    .head(5)
)


# Create the interactive goalkeeper website.
#
# Every goalkeeper appears as a point.
#
# Hovering over a goalkeeper shows their identity, team, OVR,
# attributes, contribution scores, and goalkeeper style.
#
# The graph can be zoomed, moved, and filtered using the legend.

fig = px.scatter(
    df,
    x="DistributionContribution",
    y="ProtectionContribution",
    color="GoalkeeperStyle",
    hover_name="Name",
    hover_data={
        "Position": True,
        "Team": True,
        "League": True,
        "OVR": True,
        "GK Diving": True,
        "GK Handling": True,
        "GK Kicking": True,
        "GK Positioning": True,
        "GK Reflexes": True,
        "DistributionContribution": ":.2f",
        "ProtectionContribution": ":.2f",
        "GoalkeeperSkill": ":.2f",
        "StyleGap": ":.2f",
        "GoalkeeperStyle": False
    },
    opacity=0.35,
    color_discrete_map={
        "High Protection and Distribution": "#2ca02c",
        "Protection Focused": "#1f77b4",
        "Distribution Focused": "#ff7f0e",
        "Lower Both": "#a7a7a7"
    },
    title=(
        "Interactive Goalkeeper Protection "
        "vs Distribution"
    )
)


# Make the background goalkeeper points smaller.

fig.update_traces(
    marker=dict(
        size=7
    )
)


# Add the average Distribution Contribution line.

fig.add_vline(
    x=distribution_mean,
    line_dash="dash",
    line_color="black",
    annotation_text="Average Distribution",
    annotation_position="bottom right"
)


# Add the average Protection Contribution line.

fig.add_hline(
    y=protection_mean,
    line_dash="dash",
    line_color="black",
    annotation_text="Average Protection",
    annotation_position="top left"
)


# Add labels describing the four goalkeeper regions.

x_min = df[
    "DistributionContribution"
].min()

x_max = df[
    "DistributionContribution"
].max()

y_min = df[
    "ProtectionContribution"
].min()

y_max = df[
    "ProtectionContribution"
].max()


fig.add_annotation(
    x=x_max,
    y=y_max,
    text="High Protection and Distribution",
    showarrow=False,
    xanchor="right",
    yanchor="top",
    bgcolor="rgba(255,255,255,0.75)"
)


fig.add_annotation(
    x=x_min,
    y=y_max,
    text="Protection Focused",
    showarrow=False,
    xanchor="left",
    yanchor="top",
    bgcolor="rgba(255,255,255,0.75)"
)


fig.add_annotation(
    x=x_max,
    y=y_min,
    text="Distribution Focused",
    showarrow=False,
    xanchor="right",
    yanchor="bottom",
    bgcolor="rgba(255,255,255,0.75)"
)


fig.add_annotation(
    x=x_min,
    y=y_min,
    text="Below Average in Both",
    showarrow=False,
    xanchor="left",
    yanchor="bottom",
    bgcolor="rgba(255,255,255,0.75)"
)


# Add the five highlighted protection-focused goalkeepers.

fig.add_trace(
    go.Scatter(
        x=top_5_protection[
            "DistributionContribution"
        ],
        y=top_5_protection[
            "ProtectionContribution"
        ],
        mode="markers+text",
        text=top_5_protection[
            "Name"
        ],
        textposition="top center",
        name="Top Protection Goalkeepers",
        marker=dict(
            size=13,
            color="#0047AB",
            line=dict(
                width=1,
                color="white"
            )
        ),
        customdata=top_5_protection[
            [
                "Team",
                "OVR",
                "GoalkeeperSkill",
                "GoalkeeperStyle"
            ]
        ],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Team: %{customdata[0]}<br>"
            "OVR: %{customdata[1]}<br>"
            "Goalkeeper Skill: %{customdata[2]:.2f}<br>"
            "Style: %{customdata[3]}<br>"
            "Distribution: %{x:.2f}<br>"
            "Protection: %{y:.2f}"
            "<extra></extra>"
        )
    )
)


# Add the five highlighted distribution-focused goalkeepers.

fig.add_trace(
    go.Scatter(
        x=top_5_distribution[
            "DistributionContribution"
        ],
        y=top_5_distribution[
            "ProtectionContribution"
        ],
        mode="markers+text",
        text=top_5_distribution[
            "Name"
        ],
        textposition="top center",
        name="Top Distribution Goalkeepers",
        marker=dict(
            size=13,
            color="#ff7f0e",
            line=dict(
                width=1,
                color="white"
            )
        ),
        customdata=top_5_distribution[
            [
                "Team",
                "OVR",
                "GoalkeeperSkill",
                "GoalkeeperStyle"
            ]
        ],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Team: %{customdata[0]}<br>"
            "OVR: %{customdata[1]}<br>"
            "Goalkeeper Skill: %{customdata[2]:.2f}<br>"
            "Style: %{customdata[3]}<br>"
            "Distribution: %{x:.2f}<br>"
            "Protection: %{y:.2f}"
            "<extra></extra>"
        )
    )
)


# Add the five highlighted goalkeepers who are strong in both
# protection and distribution.

fig.add_trace(
    go.Scatter(
        x=top_5_both[
            "DistributionContribution"
        ],
        y=top_5_both[
            "ProtectionContribution"
        ],
        mode="markers+text",
        text=top_5_both[
            "Name"
        ],
        textposition="top center",
        name="Top All-Around Goalkeepers",
        marker=dict(
            size=14,
            color="#2ca02c",
            symbol="diamond",
            line=dict(
                width=1,
                color="white"
            )
        ),
        customdata=top_5_both[
            [
                "Team",
                "OVR",
                "GoalkeeperSkill",
                "StyleGap"
            ]
        ],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Team: %{customdata[0]}<br>"
            "OVR: %{customdata[1]}<br>"
            "Goalkeeper Skill: %{customdata[2]:.2f}<br>"
            "Style Gap: %{customdata[3]:.2f}<br>"
            "Distribution: %{x:.2f}<br>"
            "Protection: %{y:.2f}"
            "<extra></extra>"
        )
    )
)


# Improve the website layout.

fig.update_layout(
    xaxis_title="Distribution Contribution",
    yaxis_title="Protection Contribution",
    legend_title="Goalkeeper Style",
    template="plotly_white",
    width=1200,
    height=800
)


# Save the interactive goalkeeper website.
#
# Opening this HTML file in a browser preserves:
#
# hovering
# zooming
# panning
# category filtering
# permanent labels for selected goalkeepers

fig.write_html(
    RESULTS
    / "interactive_goalkeeper_protection_distribution.html"
)


# Automatically open the website when the script runs.

fig.show()


# Create a static PCA graph.
#
# Goalkeepers close together have similar standardized profiles.

plt.figure(
    figsize=(10, 8)
)

plt.scatter(
    df["PC1"],
    df["PC2"],
    alpha=0.30,
    s=18
)

plt.xlabel(
    f"PC1 ({pc1_variance:.1%} explained variance)"
)

plt.ylabel(
    f"PC2 ({pc2_variance:.1%} explained variance)"
)

plt.title(
    "PCA Map of EAFC26 Goalkeeper Styles"
)

plt.tight_layout()

plt.savefig(
    RESULTS / "goalkeeper_pca_map.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Create an interactive PCA website.

pca_fig = px.scatter(
    df,
    x="PC1",
    y="PC2",
    color="GoalkeeperStyle",
    hover_name="Name",
    hover_data={
        "Team": True,
        "League": True,
        "OVR": True,
        "GK Diving": True,
        "GK Handling": True,
        "GK Kicking": True,
        "GK Positioning": True,
        "GK Reflexes": True,
        "ProtectionContribution": ":.2f",
        "DistributionContribution": ":.2f",
        "GoalkeeperSkill": ":.2f",
        "GoalkeeperStyle": False
    },
    color_discrete_map={
        "High Protection and Distribution": "#2ca02c",
        "Protection Focused": "#1f77b4",
        "Distribution Focused": "#ff7f0e",
        "Lower Both": "#a7a7a7"
    },
    title="Interactive PCA Map of EAFC26 Goalkeepers"
)


pca_fig.update_layout(
    xaxis_title=(
        f"PC1 ({pc1_variance:.1%} explained variance)"
    ),
    yaxis_title=(
        f"PC2 ({pc2_variance:.1%} explained variance)"
    ),
    legend_title="Goalkeeper Style",
    template="plotly_white",
    width=1200,
    height=800
)


pca_fig.write_html(
    RESULTS
    / "interactive_goalkeeper_pca_map.html"
)


# Use all five standardized attributes to find similar goalkeepers.
#
# PCA is not used for the similarity calculation because the first
# two components do not preserve all available information.

number_of_neighbors = min(
    6,
    len(df)
)

nearest_neighbors = NearestNeighbors(
    n_neighbors=number_of_neighbors
)

nearest_neighbors.fit(
    X_scaled
)


def show_similar_goalkeepers(goalkeeper_name):
    matching_positions = df.index[
        df[
            "Name"
        ]
        .astype(str)
        .str.lower()
        == goalkeeper_name.lower()
    ].tolist()

    if not matching_positions:
        print(
            f"\nGoalkeeper '{goalkeeper_name}' not found."
        )

        return

    goalkeeper_position = matching_positions[0]

    distances, neighbor_positions = (
        nearest_neighbors.kneighbors(
            X_scaled[
                goalkeeper_position
            ].reshape(
                1,
                -1
            )
        )
    )

    similarity_results = df.iloc[
        neighbor_positions[0]
    ].copy()

    similarity_results[
        "AttributeDistance"
    ] = distances[0]

    similarity_results = similarity_results[
        similarity_results[
            "Name"
        ]
        .astype(str)
        .str.lower()
        != goalkeeper_name.lower()
    ]

    similarity_columns = [
        column
        for column in [
            "Name",
            "Team",
            "OVR",
            "ProtectionContribution",
            "DistributionContribution",
            "GoalkeeperSkill",
            "GoalkeeperStyle",
            "AttributeDistance"
        ]
        if column in similarity_results.columns
    ]

    print(
        f"\nGoalkeepers most similar to {goalkeeper_name}:"
    )

    print(
        similarity_results[
            similarity_columns
        ].round(3)
    )


# Inspect several recognizable goalkeepers.

show_similar_goalkeepers(
    "Gianluigi Donnarumma"
)

show_similar_goalkeepers(
    "Alisson"
)

show_similar_goalkeepers(
    "Thibaut Courtois"
)

show_similar_goalkeepers(
    "Ederson"
)


# Save the expanded goalkeeper dataset.
#
# Stage 4 can use this file to select a goalkeeper who complements
# the attacking and defensive attributes of the chosen outfield team.

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nExpanded goalkeeper dataset saved to:")

print(OUTPUT_FILE)

print("\nInteractive goalkeeper website saved to:")

print(
    RESULTS
    / "interactive_goalkeeper_protection_distribution.html"
)

print("\nStage 3.7 goalkeeper analysis complete!")