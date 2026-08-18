from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# STAGE 3.1: DISCOVERING PLAYER ARCHETYPES
#
# Stage 2 showed that players with similar EAFC attribute profiles
# appear close together mathematically.
#
# Stage 3.1 asks a different question:
#
# Can the data naturally divide players into different player archetypes?
#
# I do NOT want to assume how many archetypes exist beforehand.
#
# Instead, I will test several possible numbers of clusters and compare:
#
# inertia
# silhouette score
# cluster sizes
# average attributes
# position distributions
# recognizable players
#
# The goal is not necessarily to find one "perfect" value of K.
#
# Instead, I want to understand what kinds of player groupings naturally
# appear at different levels of detail.


# Save graphs without automatically opening them.

matplotlib.use("Agg")


# Define project paths.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA = PROJECT_ROOT / "data"

RESULTS = (
    PROJECT_ROOT
    / "results"
    / "stage_3_1"
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


# Continue using the same six-dimensional player representation
# from Stages 1 and 2.
#
# player = [PAC, SHO, PAS, DRI, DEF, PHY]

features = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY"
]


# Standardize the six attributes before clustering.
#
# K-Means uses distances between players.
#
# Standardization prevents an attribute with a larger spread
# from dominating the clustering process.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df[features]
)


# Test several possible numbers of clusters.
#
# K represents the number of player groups that K-Means will create.

k_values = range(
    2,
    11
)

inertias = []

silhouette_scores = []


# Evaluate every possible K using:
#
# Inertia
#
# Lower inertia means players are closer to their assigned cluster centers.
#
# However, inertia always decreases as K increases, so the smallest
# inertia is NOT automatically the best solution.
#
# The Elbow Method looks for the point where adding another cluster
# produces only a relatively small improvement.
#
# Silhouette Score
#
# Higher silhouette scores indicate that players tend to fit their
# own cluster better than neighboring clusters.

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        X_scaled
    )

    inertias.append(
        kmeans.inertia_
    )

    silhouette_scores.append(
        silhouette_score(
            X_scaled,
            labels
        )
    )


# Print clustering evaluation results.

print(
    "\nClustering evaluation:"
)

for k, inertia, silhouette in zip(
    k_values,
    inertias,
    silhouette_scores
):

    print(
        f"K = {k}: "
        f"Inertia = {inertia:.2f}, "
        f"Silhouette = {silhouette:.3f}"
    )


# Save clustering evaluation results.

evaluation = pd.DataFrame({
    "K":
        list(k_values),

    "Inertia":
        inertias,

    "SilhouetteScore":
        silhouette_scores
})


evaluation.to_csv(
    RESULTS
    / "clustering_evaluation.csv",
    index=False
)


# Plot the Elbow Method.

plt.figure(
    figsize=(8,5)
)

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Inertia"
)

plt.title(
    "Elbow Method for Player Archetypes"
)

plt.xticks(
    list(k_values)
)

plt.tight_layout()

plt.savefig(
    RESULTS
    / "clustering_elbow_method.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Plot silhouette score for each K.

plt.figure(
    figsize=(8,5)
)

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.title(
    "Silhouette Score by Number of Player Archetypes"
)

plt.xticks(
    list(k_values)
)

plt.tight_layout()

plt.savefig(
    RESULTS
    / "clustering_silhouette_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Find the K with the highest silhouette score.
#
# This is useful mathematical information, but it does NOT automatically
# mean that this K is the best soccer interpretation.

best_index = (
    silhouette_scores.index(
        max(
            silhouette_scores
        )
    )
)

best_k = (
    list(k_values)[
        best_index
    ]
)

print(
    "\nHighest silhouette score occurs at K =",
    best_k
)


# The earlier experiment showed that K = 2 creates a broad split
# between more defensive players and more attacking players.
#
# That is meaningful, but it may be too general if we want to discover
# more detailed player archetypes.
#
# Therefore, I will examine several candidate values of K in more detail.

candidate_k_values = [
    2,
    3,
    4,
    5,
    6
]


# Use PCA only for visualization.
#
# K-Means itself still uses all six standardized EAFC attributes.
#
# PCA reduces the six-dimensional player space to two dimensions
# so we can draw the clusters.

pca = PCA(
    n_components=2
)

pca_components = pca.fit_transform(
    X_scaled
)

df["PC1"] = (
    pca_components[:, 0]
)

df["PC2"] = (
    pca_components[:, 1]
)


# Run separate clustering experiments for each candidate K.
#
# Every K gets its own result folder:
#
# results/
#     stage_3_1/
#         k2/
#         k3/
#         k4/
#         k5/
#         k6/
#
# This keeps the experiments organized and prevents files
# from different K values from being mixed together.

for k in candidate_k_values:

    print(
        "\n======================================"
    )

    print(
        f"CLUSTERING EXPERIMENT: K = {k}"
    )

    print(
        "======================================"
    )


    # Create a result folder for the current K.

    K_RESULTS = (
        RESULTS
        / f"k{k}"
    )

    K_RESULTS.mkdir(
        parents=True,
        exist_ok=True
    )


    # Run K-Means.

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(
        X_scaled
    )


    # Create a separate copy for this clustering experiment.

    experiment_df = df.copy()

    experiment_df[
        "Cluster"
    ] = cluster_labels


    # Count the number of players in every cluster.
    #
    # Very small clusters could indicate that K is becoming too large.
    #
    # Very large clusters could indicate that the grouping is still broad.

    cluster_counts = (
        experiment_df[
            "Cluster"
        ]
        .value_counts()
        .sort_index()
    )


    print(
        "\nPlayers in each cluster:"
    )

    print(
        cluster_counts
    )


    cluster_counts.to_csv(
        K_RESULTS
        / "cluster_counts.csv"
    )


    # Calculate the average EAFC attributes inside each cluster.
    #
    # These averages help us interpret what each group represents.
    #
    # We should not name the clusters before examining these results.

    cluster_means = (
        experiment_df
        .groupby(
            "Cluster"
        )[features]
        .mean()
        .round(2)
    )


    print(
        "\nAverage attributes by cluster:"
    )

    print(
        cluster_means
    )


    cluster_means.to_csv(
        K_RESULTS
        / "cluster_attribute_means.csv"
    )


    # Examine the traditional position distribution inside each cluster.
    #
    # This helps answer:
    #
    # Does clustering simply recreate soccer positions?
    #
    # Or do multiple positions share similar attribute-based archetypes?

    position_counts = pd.crosstab(
        experiment_df[
            "Cluster"
        ],
        experiment_df[
            "Position"
        ]
    )


    print(
        "\nPosition counts by cluster:"
    )

    print(
        position_counts
    )


    position_counts.to_csv(
        K_RESULTS
        / "cluster_position_counts.csv"
    )


    # Calculate position percentages inside each cluster.
    #
    # Percentages are easier to compare when cluster sizes differ.

    position_percentages = pd.crosstab(
        experiment_df[
            "Cluster"
        ],
        experiment_df[
            "Position"
        ],
        normalize="index"
    ).round(
        3
    )


    print(
        "\nPosition percentages by cluster:"
    )

    print(
        position_percentages
    )


    position_percentages.to_csv(
        K_RESULTS
        / "cluster_position_percentages.csv"
    )


    # Look at the highest-rated players inside every cluster.
    #
    # Cluster averages provide the mathematical interpretation,
    # while recognizable players help us decide whether that
    # interpretation makes sense from a soccer perspective.

    top_players_all_clusters = []


    for cluster in sorted(
        experiment_df[
            "Cluster"
        ].unique()
    ):

        print(
            f"\nTop players in Cluster {cluster}:"
        )


        top_players = (
            experiment_df[
                experiment_df[
                    "Cluster"
                ]
                == cluster
            ]
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
                    "PAC",
                    "SHO",
                    "PAS",
                    "DRI",
                    "DEF",
                    "PHY"
                ]
            ]
            .head(10)
            .copy()
        )


        top_players[
            "Cluster"
        ] = cluster


        print(
            top_players
        )


        top_players_all_clusters.append(
            top_players
        )


    # Save all top players for this K in one CSV.

    top_players_df = pd.concat(
        top_players_all_clusters,
        ignore_index=True
    )


    top_players_df.to_csv(
        K_RESULTS
        / "top_players_by_cluster.csv",
        index=False
    )


    # Visualize the clusters using the two PCA coordinates.
    #
    # PCA is only being used for the graph.
    #
    # K-Means itself was performed using all six standardized attributes.

    plt.figure(
        figsize=(10,8)
    )


    clusters = sorted(
        experiment_df[
            "Cluster"
        ].unique()
    )


    for cluster in clusters:

        subset = experiment_df[
            experiment_df[
                "Cluster"
            ]
            == cluster
        ]


        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            alpha=0.5,
            label=f"Cluster {cluster}",
            s=15
        )


    plt.xlabel(
        "Principal Component 1"
    )

    plt.ylabel(
        "Principal Component 2"
    )

    plt.title(
        f"EAFC26 Player Archetypes: K = {k}"
    )

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()


    plt.savefig(
        K_RESULTS
        / "player_clusters.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # Save every player's cluster assignment.
    #
    # The different K solutions remain separate because we have not
    # decided that one representation is automatically correct.

    experiment_df.to_csv(
        K_RESULTS
        / "player_clusters.csv",
        index=False
    )


# Stage 3.1 does NOT conclude that one clustering solution is correct.
#
# Instead, it produces several candidate descriptions of player archetypes.
#
# These solutions can be compared using:
#
# silhouette score
# elbow method
# cluster sizes
# average attributes
# position distributions
# recognizable players
# interpretability
#
# K = 2 may reveal the strongest broad split in the dataset.
#
# Larger values of K may reveal more detailed player styles.
#
# Stage 3.2 asks a different question:
#
# "How much does each player contribute offensively and defensively?"
#
# Clustering asks:
#
# "What groups naturally exist?"
#
# Stage 3.2 asks:
#
# "Where does each individual player lie on an attack-defense spectrum?"
#
# Keeping these questions separate prevents us from forcing a
# "balanced player" cluster into the data if K-Means does not
# naturally produce one.


print(
    "\nStage 3.1 clustering experiments complete!"
)