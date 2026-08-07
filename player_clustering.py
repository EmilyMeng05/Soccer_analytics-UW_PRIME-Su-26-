import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# Read the cleaned outfield player dataset from Stage 1.
# We use the same six EAFC attributes so that Stage 3 builds directly
# on the player representation created earlier.

df = pd.read_csv("cleaned_eafc26_outfield_players.csv")

# Here are all the features 
features = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY"
]

# Before clustering, standardize all six attributes.
scaler = StandardScaler()

X_scaled = scaler.fit_transform(df[features])


# We do not want to decide the number of player archetypes in advance.
# Instead, I will test several possible values of K and compare how well
# each clustering solution performs.
#
# I will test K values from 2 through 10.

k_values = range(2, 11)

inertias = []
silhouette_scores = []


# For every possible value of K, run K-Means and record two measurements.
#
# Inertia measures how tightly players are grouped around their cluster centers.
# Lower inertia is better, but inertia always decreases when K increases.
#
# Silhouette score measures whether players are closer to their own cluster
# than to other clusters. Higher silhouette scores are better.

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    inertias.append(kmeans.inertia_)

    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )


# Print the silhouette score for each possible number of clusters.
#
# This gives us a numerical way to compare different choices of K.

print("\nSilhouette scores:")

for k, score in zip(k_values, silhouette_scores):

    print(
        f"K = {k}: {score:.3f}"
    )


# Plot the Elbow Method.
#
# We look for the point where adding more clusters stops producing
# a large improvement in inertia.
#
# This graph should not be used alone to choose K, but it gives us
# another perspective on the structure of the data.

plt.figure(figsize=(8,5))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Player Clustering")

plt.tight_layout()

plt.savefig(
    "clustering_elbow_method.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Plot silhouette score for each value of K.
#
# The value of K with the highest silhouette score provides one possible
# candidate for the number of natural player groups in the dataset.

plt.figure(figsize=(8,5))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score by Number of Clusters")

plt.tight_layout()

plt.savefig(
    "clustering_silhouette_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Find the K with the highest silhouette score.
#
# This is only a starting point.
# We will also examine whether the resulting clusters make sense
# from a soccer interpretation perspective.

best_index = silhouette_scores.index(
    max(silhouette_scores)
)

best_k = list(k_values)[best_index]

print(
    "\nBest K based on silhouette score:",
    best_k
)


# Run the final K-Means model using the best K found above.
#
# Every player now receives a cluster label.
#
# The cluster numbers themselves do not have any meaning yet.
# For example, Cluster 0 is not automatically "strikers" or "playmakers."
# We will interpret each cluster after examining its attributes.

kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)


# Count how many players belong to each cluster.
#
# This helps us determine whether the clustering creates reasonably sized
# groups or whether some clusters contain very few players.

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

print("\nPlayers in each cluster:")
print(cluster_counts)


# Calculate the average EAFC attributes for every cluster.
#
# These cluster averages are the main tool we will use to interpret
# what kind of player each cluster represents.

cluster_means = (
    df
    .groupby("Cluster")[features]
    .mean()
    .round(2)
)

print("\nAverage attributes by cluster:")
print(cluster_means)


# Save the cluster averages separately.
#
# This table will make it easier to compare the different archetypes
# and eventually assign meaningful soccer descriptions to them.

cluster_means.to_csv(
    "cluster_attribute_means.csv"
)


# Check which traditional positions appear inside each cluster.
#
# This is especially important because we want to know whether the model
# is simply recreating soccer positions or discovering broader playing styles
# that may include players from multiple positions.

position_cluster_table = pd.crosstab(
    df["Cluster"],
    df["Position"]
)

print("\nPosition distribution by cluster:")
print(position_cluster_table)

position_cluster_table.to_csv(
    "cluster_position_distribution.csv"
)


# It is also useful to look at position proportions instead of only counts.
#
# For example, a cluster could contain 500 midfielders and 300 attackers,
# but percentages make it easier to understand the overall composition.

position_cluster_percent = pd.crosstab(
    df["Cluster"],
    df["Position"],
    normalize="index"
).round(3)

print("\nPosition percentage by cluster:")
print(position_cluster_percent)

position_cluster_percent.to_csv(
    "cluster_position_percentages.csv"
)


# To help interpret the clusters, I also want to see some of the
# highest-rated players inside each group.
#
# Famous or recognizable players can make it easier to understand
# what a cluster represents.

for cluster in sorted(df["Cluster"].unique()):

    print(
        f"\nTop players in Cluster {cluster}:"
    )

    top_players = (
        df[df["Cluster"] == cluster]
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


# PCA is only being used here for visualization.
#
# The actual clustering was performed using all six standardized attributes,
# not the two PCA dimensions.
#
# Reducing the data to two dimensions allows us to create a visual map
# showing where the discovered clusters appear.

pca = PCA(n_components=2)

pca_components = pca.fit_transform(X_scaled)

df["PC1"] = pca_components[:, 0]
df["PC2"] = pca_components[:, 1]


# Plot players in PCA space and color them based on their discovered cluster.
#
# If the clusters are meaningful, we should see at least some separation
# between the groups in this visualization.

plt.figure(figsize=(10,8))

clusters = sorted(
    df["Cluster"].unique()
)

for cluster in clusters:

    subset = df[
        df["Cluster"] == cluster
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
    "EAFC26 Player Archetypes Discovered by K-Means"
)

plt.legend(
    bbox_to_anchor=(1.05, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "player_clusters_pca.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Save the final dataset.
#
# Each player now has a cluster assignment that represents their
# discovered player archetype.
#
# We are intentionally not claiming yet that players from the same
# or different clusters work better together.
#
# That question will be tested later when we begin building teams.

df.to_csv(
    "player_clusters.csv",
    index=False
)

print("\nStage 3 initial clustering complete!")