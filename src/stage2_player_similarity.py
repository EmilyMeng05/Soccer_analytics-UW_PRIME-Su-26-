from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# STAGE 2: PLAYER SIMILARITY AND PCA
#
# prepare_dataset.py has already cleaned the EAFC26 data.
#
# Stage 2 therefore starts directly from the cleaned outfield-player dataset.
#
# The goal of this stage is to understand whether the six main EAFC
# attributes can be simplified into a smaller number of meaningful dimensions
# and whether players with similar attribute profiles appear close together.


# Define project paths.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA = PROJECT_ROOT / "data"

RESULTS = (
    PROJECT_ROOT
    / "results"
    / "stage_2"
)

RESULTS.mkdir(
    parents=True,
    exist_ok=True
)


# Read the cleaned outfield player dataset created by prepare_dataset.py.

df = pd.read_csv(
    DATA
    / "processed"
    / "cleaned_eafc26_outfield_players.csv"
)

print(
    "Dataset shape:",
    df.shape
)


# Use the six main EAFC attributes to represent every outfield player.
#
# Mathematically:
#
# player = [PAC, SHO, PAS, DRI, DEF, PHY]
#
# At this stage, the goal is not to decide whether one player is better
# than another.
#
# Instead, I want to understand how these six attributes relate to each
# other and whether players with similar profiles appear close together.

features = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY"
]


# Standardize the six attributes.
#
# PCA and distance-based methods can be influenced by differences in
# feature scale and variation.
#
# Standardization transforms each attribute so that it has approximately:
#
# mean = 0
# standard deviation = 1
#
# This allows every attribute to contribute more fairly.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df[features]
)


# Run PCA using all six possible components first.
#
# Since we begin with six attributes, PCA can create up to six
# principal components.
#
# The explained variance tells us how much of the original variation
# in player attributes is captured by each component.

pca_full = PCA()

pca_full.fit(
    X_scaled
)

explained_variance = (
    pca_full.explained_variance_ratio_
)


print(
    "\nExplained variance by principal component:"
)

for i, variance in enumerate(
    explained_variance
):

    print(
        f"PC{i + 1}: "
        f"{variance:.3f} "
        f"({variance * 100:.1f}%)"
    )


# Save the explained variance values.

explained_variance_df = pd.DataFrame({
    "PrincipalComponent": [
        f"PC{i + 1}"
        for i in range(
            len(explained_variance)
        )
    ],
    "ExplainedVarianceRatio":
        explained_variance
})


explained_variance_df.to_csv(
    RESULTS
    / "pca_explained_variance.csv",
    index=False
)


# Plot the variance explained by every principal component.

plt.figure(
    figsize=(8,5)
)

plt.bar(
    range(
        1,
        len(explained_variance) + 1
    ),
    explained_variance
)

plt.xlabel(
    "Principal Component"
)

plt.ylabel(
    "Explained Variance Ratio"
)

plt.title(
    "Variance Explained by Each Principal Component"
)

plt.tight_layout()

plt.savefig(
    RESULTS
    / "pca_explained_variance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Reduce the six-dimensional player representation to two dimensions.
#
# PCA is NOT clustering players.
#
# Instead, it creates new variables that summarize the largest patterns
# of variation in the original six attributes.
#
# PC1 and PC2 are used mainly for interpretation and visualization.

pca = PCA(
    n_components=2
)

components = pca.fit_transform(
    X_scaled
)

df["PC1"] = (
    components[:, 0]
)

df["PC2"] = (
    components[:, 1]
)


# Examine PCA loadings.
#
# Loadings describe how strongly each original EAFC attribute contributes
# to each principal component.
#
# Large positive or negative values indicate a stronger relationship
# between the original attribute and the principal component.

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[
        "PC1",
        "PC2"
    ],
    index=features
)


print(
    "\nPCA loadings:"
)

print(
    loadings
)


loadings.to_csv(
    RESULTS
    / "pca_loadings.csv"
)


# Calculate the total amount of variance explained by PC1 and PC2.

two_component_variance = (
    pca.explained_variance_ratio_.sum()
)

print(
    "\nVariance explained by PC1 and PC2:"
)

print(
    f"{two_component_variance:.3f} "
    f"({two_component_variance * 100:.1f}%)"
)


# Plot players in the two-dimensional PCA space.
#
# Every point represents one player.
#
# Players with similar combinations of EAFC attributes should appear
# closer together.
#
# The points are colored by official position only to help interpret
# whether traditional soccer positions naturally appear in different
# parts of the feature space.

plt.figure(
    figsize=(11,8)
)

positions = (
    df["Position"]
    .unique()
)


for position in positions:

    subset = df[
        df["Position"]
        == position
    ]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        alpha=0.5,
        label=position,
        s=15
    )


plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "Player Similarity Based on EAFC26 Attributes"
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    RESULTS
    / "PCA_Player_Map.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# PCA is useful for visualization, but PC1 and PC2 do not preserve
# all of the information from the original six-dimensional player space.
#
# Therefore, player similarity will be calculated using all six
# standardized attributes rather than only PC1 and PC2.

neighbors = NearestNeighbors(
    n_neighbors=6,
    metric="euclidean"
)

neighbors.fit(
    X_scaled
)


# Find players with the most similar six-attribute profiles.
#
# The first nearest neighbor will normally be the player themselves,
# so that result is removed.

def find_similar_players(
    player_name
):

    matches = df[
        df["Name"]
        .str.lower()
        == player_name.lower()
    ]


    if len(matches) == 0:

        print(
            f"\nPlayer '{player_name}' not found."
        )

        return None


    player_index = (
        matches.index[0]
    )


    distances, indices = (
        neighbors.kneighbors(
            X_scaled[
                player_index
            ].reshape(
                1,
                -1
            )
        )
    )


    similar_players = (
        df.iloc[
            indices[0][1:]
        ][
            [
                "Name",
                "Position",
                "Team",
                "OVR",
                "PAC",
                "SHO",
                "PAS",
                "DRI",
                "DEF",
                "PHY"
            ]
        ]
        .copy()
    )


    similar_players[
        "Distance"
    ] = (
        distances[0][1:]
    )


    print(
        f"\nPlayers most similar to {player_name}:"
    )

    print(
        similar_players
    )


    return similar_players


# Test the similarity function with Lionel Messi.
#
# This can be changed to any player in the dataset.

messi_similar_players = (
    find_similar_players(
        "Lionel Messi"
    )
)


# Save the example similarity results.

if messi_similar_players is not None:

    messi_similar_players.to_csv(
        RESULTS
        / "lionel_messi_similar_players.csv",
        index=False
    )


# Save the PCA coordinates for every player.
#
# These coordinates can be reused later for visualization and clustering.

df.to_csv(
    RESULTS
    / "player_similarity_results.csv",
    index=False
)


# Stage 2 focuses on understanding the structure of the six EAFC attributes.
#
# Main questions:
#
# Can the six-dimensional player representation be simplified?
#
# What do the first principal components appear to represent?
#
# Do players with similar EAFC profiles appear close together?
#
# Stage 3.1 will use the same six-dimensional player representation
# to investigate whether players naturally divide into different archetypes.

print(
    "\nStage 2 player similarity analysis complete!"
)