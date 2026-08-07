import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# just save the graph, dont show me
matplotlib.use("Agg")

# STAGE 3.1: DISCOVERING PLAYER ARCHETYPES
#
# In Stage 2, we showed that players with similar EAFC attribute profiles
# tend to appear close together mathematically.
#
# Now I want to ask a slightly different question:
#
# Can the data naturally divide players into different playing archetypes?
#
# Importantly, I do NOT want to assume how many playing styles exist.
# I also do not want to assume that one clustering solution is automatically
# correct just because it has the best numerical score.
#
# Instead, this stage will generate several candidate clustering solutions
# and compare them mathematically and from a soccer interpretation perspective.


# Read the cleaned outfield player dataset created in Stage 1.

df = pd.read_csv("cleaned_eafc26_outfield_players.csv")

# Continue using the same six-dimensional player representation:
features = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY"
]


# Standardize the six attributes before clustering.
scaler = StandardScaler()

X_scaled = scaler.fit_transform(df[features])


# We do not know how many natural player archetypes exist.
#
# Instead of immediately choosing something like K = 5, I will test
# several possible values.
#
# K represents the number of clusters that K-Means tries to create.

k_values = range(2, 11)

inertias = []
silhouette_scores = []


# For every possible value of K, run K-Means and calculate two measurements:
#
# 1. Inertia
#    Measures how tightly players are grouped around their cluster centers.
#    Lower values mean players are closer to the center of their assigned group.
#
#    However, inertia ALWAYS decreases as K increases.
#    Therefore, we cannot simply choose the K with the lowest inertia.
#
#    Instead, the Elbow Method looks for the point where adding additional
#    clusters stops producing a large improvement.
#
# 2. Silhouette Score
#    Measures how well each player fits inside their own cluster compared
#    with neighboring clusters.
#
#    Higher silhouette scores generally represent clearer separation.
#
#    However, the clustering with the highest silhouette score is not
#    automatically the most useful soccer interpretation.

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertias.append(
        kmeans.inertia_
    )

    silhouette_scores.append(
        silhouette_score(
            X_scaled,
            labels
        )
    )


# Print the results so we can compare different values of K.

print("\nClustering evaluation:")

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


# Save the clustering evaluation results.
#
# This will make it easier to record the results in PROJECT_NOTES.md
# and compare the different clustering experiments later.

evaluation = pd.DataFrame({
    "K": list(k_values),
    "Inertia": inertias,
    "SilhouetteScore": silhouette_scores
})

evaluation.to_csv(
    "clustering_evaluation.csv",
    index=False
)


# Plot the Elbow Method.
#
# Again, we are NOT looking for the lowest inertia.
#
# Since inertia decreases whenever more clusters are added, the lowest
# value will almost always occur at the largest K tested.
#
# Instead, we are looking for a bend in the curve where adding more
# clusters begins producing smaller improvements.

plt.figure(figsize=(8,5))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Player Archetypes")

plt.xticks(list(k_values))

plt.tight_layout()

plt.savefig(
    "clustering_elbow_method.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Plot silhouette score for each K.
#
# A higher silhouette score generally indicates more clearly separated
# clusters.
#
# In the initial experiment, K = 2 produced the highest silhouette score.
# However, that solution mainly separated defensive players from the rest
# of the dataset.
#
# Since our goal is to discover more detailed player archetypes, we will
# examine several larger values of K rather than automatically accepting K = 2.

plt.figure(figsize=(8,5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Number of Player Archetypes")

plt.xticks(list(k_values))

plt.tight_layout()

plt.savefig(
    "clustering_silhouette_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# Print the mathematically best K according to silhouette score.
#
# This is useful information, but this is NOT yet our final choice.

best_index = silhouette_scores.index(
    max(silhouette_scores)
)

best_k = list(k_values)[best_index]

print(
    "\nHighest silhouette score occurs at K =",
    best_k
)


# The first experiment showed that K = 2 mainly creates a broad
# attack-versus-defense split.
#
# That is meaningful, but it may be too general for our goal of discovering
# player archetypes.
#
# Therefore, I want to examine several candidate clustering solutions
# in more detail.
#
# K = 2 gives us the broadest structure.
#
# K = 3 is interesting because soccer naturally includes players who may
# contribute strongly to both attacking and defending.
#
# K = 4, 5, and 6 may reveal more specialized styles.
#
# We will not assume in advance which of these is the correct answer.

candidate_k_values = [
    2,
    3,
    4,
    5,
    6
]


# PCA will be used only for visualization.
#
# The actual K-Means clustering still uses all six standardized attributes.
#
# PCA converts the six-dimensional player representation into two dimensions
# so that we can visualize where the clusters appear relative to each other.

pca = PCA(
    n_components=2
)

pca_components = pca.fit_transform(
    X_scaled
)

df["PC1"] = pca_components[:, 0]
df["PC2"] = pca_components[:, 1]


# Now run separate clustering experiments for each candidate K.
#
# For every experiment, we will:
#
# 1. Assign every player to a cluster.
# 2. Count the number of players in each cluster.
# 3. Calculate average EAFC attributes for every cluster.
# 4. Examine position distributions.
# 5. Print recognizable high-rated players.
# 6. Create a PCA visualization.
# 7. Save the results.
#
# This gives us enough information to compare the different clustering
# solutions later without making an immediate decision.

for k in candidate_k_values:

    print("\n")
    print("======================================")
    print(f"CLUSTERING EXPERIMENT: K = {k}")
    print("======================================")


    # Run K-Means using the current number of clusters.

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    cluster_labels = kmeans.fit_predict(
        X_scaled
    )


    # Create a temporary copy of the player dataset.
    #
    # This prevents one clustering experiment from overwriting another.

    experiment_df = df.copy()

    experiment_df["Cluster"] = cluster_labels


    # Count how many players belong to each cluster.
    #
    # Extremely small clusters might suggest that K is becoming too large,
    # while very large clusters may indicate that the grouping is too broad.

    cluster_counts = (
        experiment_df["Cluster"]
        .value_counts()
        .sort_index()
    )

    print("\nPlayers in each cluster:")

    print(cluster_counts)


    # Calculate the mean EAFC attributes inside every cluster.
    #
    # These averages help us understand what each group represents.
    #
    # For example, a cluster with high PAC, SHO, and DRI but low DEF
    # might represent attacking players.
    #
    # We should NOT name the clusters before looking at these results.

    cluster_means = (
        experiment_df
        .groupby("Cluster")[features]
        .mean()
        .round(2)
    )

    print("\nAverage attributes by cluster:")

    print(cluster_means)


    # Save the cluster means for this K.
    #
    # Having separate files will make it much easier to compare
    # K = 2, 3, 4, 5, and 6 later.

    cluster_means.to_csv(
        f"cluster_attribute_means_k{k}.csv"
    )


    # Examine which traditional soccer positions appear inside each cluster.
    #
    # One of the main research questions is whether the clustering simply
    # recreates traditional positions or discovers styles that cross
    # positional boundaries.

    position_counts = pd.crosstab(
        experiment_df["Cluster"],
        experiment_df["Position"]
    )

    print("\nPosition counts by cluster:")

    print(position_counts)

    position_counts.to_csv(
        f"cluster_position_counts_k{k}.csv"
    )


    # Percentages are more useful than raw counts when clusters have
    # different sizes.
    #
    # For example, if a cluster is:
    #
    # 40% CAM
    # 25% CM
    # 20% RW
    # 15% CF
    #
    # that might represent a playing style shared across several positions.

    position_percentages = pd.crosstab(
        experiment_df["Cluster"],
        experiment_df["Position"],
        normalize="index"
    ).round(3)

    print("\nPosition percentages by cluster:")

    print(position_percentages)

    position_percentages.to_csv(
        f"cluster_position_percentages_k{k}.csv"
    )


    # Look at the highest-rated players inside every cluster.
    #
    # Cluster averages give us the mathematical description of a group,
    # while recognizable players help us understand whether that description
    # makes sense from a soccer perspective.
    #
    # The interpretation should come AFTER seeing both pieces of information.

    for cluster in sorted(
        experiment_df["Cluster"].unique()
    ):

        print(
            f"\nTop players in Cluster {cluster}:"
        )

        top_players = (
            experiment_df[
                experiment_df["Cluster"] == cluster
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
        )

        print(top_players)


    # Visualize the current clustering solution using PCA.
    #
    # Again, PCA is only being used to draw the graph.
    # K-Means was performed using all six standardized attributes.

    plt.figure(figsize=(10,8))

    clusters = sorted(
        experiment_df["Cluster"].unique()
    )

    for cluster in clusters:

        subset = experiment_df[
            experiment_df["Cluster"] == cluster
        ]

        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            alpha=0.5,
            label=f"Cluster {cluster}",
            s=15
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.title(
        f"EAFC26 Player Archetypes: K = {k}"
    )

    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    plt.tight_layout()

    plt.savefig(
        f"player_clusters_k{k}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # Save every player's cluster assignment for the current experiment.
    #
    # We are intentionally keeping the different K solutions separate
    # because we have not yet decided which representation is most useful.

    experiment_df.to_csv(
        f"player_clusters_k{k}.csv",
        index=False
    )


# Stage 3.1 does NOT conclude that one clustering solution is correct.
#
# Instead, the goal is to produce several candidate descriptions of
# player archetypes.
#
# We will compare them using:
#
# - silhouette score
# - elbow method
# - cluster sizes
# - average EAFC attributes
# - position distributions
# - recognizable players
# - interpretability
#
# K = 2 may reveal the strongest overall split in the dataset, while
# larger K values may reveal more detailed playing archetypes.
#
# Separately, Stage 3.2 will investigate attacking-versus-defensive balance.
#
# This distinction matters because clustering asks:
#
# "What groups naturally exist?"
#
# while balance analysis asks:
#
# "Which players are strong in both attacking and defending?"
#
# We do not want to force a balanced-player group into K-Means if the data
# does not naturally create one.

print("\nStage 3.1 clustering experiments complete!")