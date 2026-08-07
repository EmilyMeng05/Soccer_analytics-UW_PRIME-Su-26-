import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# Read the cleaned outfield player dataset from Stage 1.
# Since the data has already been cleaned, we do not need to repeat
# the filtering and goalkeeper separation steps.

df = pd.read_csv("cleaned_eafc26_outfield_players.csv")

print("Dataset shape:", df.shape)


# We will continue using the six main EAFC attributes to represent each player.
#
# player = [PAC, SHO, PAS, DRI, DEF, PHY]
#
# At this stage, the goal is no longer to decide whether a player is good.
# Instead, we want to understand how players differ from each other.

features = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY"
]

# The six attributes have different distributions and ranges.
# Before calculating distances or running PCA, we standardize them so that
# one attribute does not dominate the analysis simply because it has
# a larger spread.
#
# After standardization, each feature will approximately have
# mean = 0 and standard deviation = 1.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(df[features])


# Before reducing the data to two dimensions, I want to see how much
# information each principal component explains.
#
# Since we started with six attributes, PCA can produce up to six
# principal components.

pca_full = PCA()

pca_full.fit(X_scaled)

explained_variance = pca_full.explained_variance_ratio_

print("\nExplained variance by principal component:")

for i, variance in enumerate(explained_variance):
    print(
        f"PC{i + 1}: {variance:.3f} "
        f"({variance * 100:.1f}%)"
    )


# Plot the amount of variance explained by each principal component.
#
# This allows us to decide whether reducing the data to two dimensions
# still preserves enough information about the original players.

plt.figure(figsize=(8,5))

plt.bar(
    range(1, len(explained_variance) + 1),
    explained_variance
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("Variance Explained by Each Principal Component")

plt.tight_layout()

plt.savefig(
    "pca_explained_variance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Now reduce the six-dimensional player representation into two dimensions.
#
# PCA is NOT grouping players by overall rating.
# Instead, it creates two new variables, PC1 and PC2, that capture
# the largest patterns of variation across PAC, SHO, PAS, DRI, DEF, and PHY.

pca = PCA(n_components=2)

components = pca.fit_transform(X_scaled)

df["PC1"] = components[:, 0]
df["PC2"] = components[:, 1]


# To understand what PC1 and PC2 actually represent, we look at the PCA loadings.
#
# A large positive or negative loading means that the original attribute
# contributes strongly to that principal component.

loadings = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2"],
    index=features
)

print("\nPCA loadings:")
print(loadings)


# Plot players in the two-dimensional PCA space.
#
# Each point represents one player.
# Players with similar combinations of EAFC attributes should appear
# closer together on the graph.
#
# We color the players by their traditional soccer position so that we
# can check whether positions naturally emerge from the attribute data.

plt.figure(figsize=(11,8))

positions = df["Position"].unique()

for position in positions:

    subset = df[df["Position"] == position]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        alpha=0.5,
        label=position,
        s=15
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.title("Player Similarity Based on EAFC26 Attributes")

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "PCA_Player_Map.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# PCA is useful for visualization, but we do not want to calculate
# player similarity using only PC1 and PC2 because some information
# is lost when reducing six dimensions to two.
#
# Instead, player similarity will be calculated using the original
# six standardized attributes.

neighbors = NearestNeighbors(
    n_neighbors=6,
    metric="euclidean"
)

neighbors.fit(X_scaled)


# This function allows us to enter a player's name and find the players
# with the most similar EAFC attribute profiles.
#
# The first returned player is normally the player themselves,
# so we exclude that result.

def find_similar_players(player_name):

    matches = df[
        df["Name"].str.lower() == player_name.lower()
    ]

    if len(matches) == 0:

        print("Player not found.")
        return

    player_index = matches.index[0]

    distances, indices = neighbors.kneighbors(
        X_scaled[player_index].reshape(1, -1)
    )

    results = df.iloc[indices[0][1:]][
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
    ].copy()

    results["Distance"] = distances[0][1:]

    print(
        f"\nPlayers most similar to {player_name}:"
    )

    print(results)

    return results


# Test the similarity function with a player from the dataset.
# You can replace this name with any player you are interested in.

find_similar_players("Lionel Messi")


# Save the PCA coordinates so that we can reuse them later
# when we begin clustering players into different archetypes.

df.to_csv(
    "player_similarity_results.csv",
    index=False
)

print("\nStage 2 initial analysis complete!")



# Dataset shape: (11872, 16)

# Explained variance by principal component:
# PC1: 0.484 (48.4%)
# PC2: 0.248 (24.8%)
# PC3: 0.121 (12.1%)
# PC4: 0.112 (11.2%)
# PC5: 0.020 (2.0%)
# PC6: 0.015 (1.5%)

# since PC1 and PC2 has the highest amount we pick those two

# PCA loadings:
#           PC1       PC2
# PAC  0.361919 -0.174162
# SHO  0.511179  0.048507
# PAS  0.444461  0.460897
# DRI  0.540490  0.224083
# DEF -0.272101  0.628252
# PHY -0.209753  0.556753

# Notice PC1 seems to focus on the previous 4 attributes which are mostly associated with attacking
# while PC2 focus on the last four attributes and are more associated with defensing

# Players most similar to Lionel Messi:
#                  Name              Position             Team  OVR  PAC  SHO  PAS  DRI  DEF  PHY  Distance
# 47       Paulo Dybala  Attacking Midfielder          AS Roma   86   80   85   84   87   41   64  0.650135
# 115      Riyad Mahrez      Right Midfielder          Al Ahli   84   78   80   81   88   39   63  0.741222
# 159        Iago Aspas          Right Winger            Celta   83   77   84   80   84   35   62  0.953613
# 163  Leandro Trossard           Left Winger          Arsenal   83   80   81   80   85   30   60  1.018354
# 217    Ángel Di María          Right Winger  Rosario Central   82   73   80   85   85   43   62  1.065817

# Stage 2 initial analysis complete!