# Stage 3.1 — Player Clustering

## Goal

The goal of Stage 3.1 is to investigate whether players naturally separate into different groups based on their EAFC26 attributes.

Instead of using the player's listed position, this stage uses unsupervised learning to ask:

> If we only look at player attributes, what types of players naturally appear in the dataset?

Each outfield player is represented using the six main EAFC attributes:

- PAC — Pace
- SHO — Shooting
- PAS — Passing
- DRI — Dribbling
- DEF — Defending
- PHY — Physicality

Therefore, each player can be represented as:

`[PAC, SHO, PAS, DRI, DEF, PHY]`


## Why Clustering?

Traditional soccer positions already divide players into categories such as striker, midfielder, and center back.

However, these labels may hide similarities between players who officially play different positions.

For example, two players with different listed positions may still have very similar attacking, defensive, or physical profiles.

K-Means clustering allows us to group players based only on their attributes without telling the algorithm their official position.


## Data Preparation

The cleaned outfield player dataset from `prepare_dataset.py` is used.

Before clustering, the six attributes are standardized using `StandardScaler`.

This is important because clustering depends on distances between players.

After standardization, each attribute contributes on a comparable scale.


## Choosing the Number of Clusters

K-Means clustering was tested for:

`K = 2 through K = 10`

Two measurements were examined:

### Inertia

Inertia measures how far players are from the center of their assigned cluster.

Lower inertia means players are grouped more tightly.

However, inertia almost always decreases when more clusters are added, so it cannot determine the best K by itself.


### Silhouette Score

The silhouette score measures both:

1. How similar players are to their own cluster.
2. How different they are from other clusters.

A higher silhouette score indicates clearer separation between clusters.


## Results

| K | Inertia | Silhouette |
|---|---:|---:|
| 2 | 48375.04 | 0.294 |
| 3 | 37760.49 | 0.269 |
| 4 | 32677.16 | 0.235 |
| 5 | 29053.39 | 0.234 |
| 6 | 26642.85 | 0.226 |
| 7 | 24622.49 | 0.221 |
| 8 | 22964.35 | 0.210 |
| 9 | 21681.82 | 0.203 |
| 10 | 20755.85 | 0.198 |

The highest silhouette score occurred at:

`K = 2`


## K = 2 Cluster Profiles

### Cluster 0

Average attributes:

| Attribute | Average |
|---|---:|
| PAC | 61.85 |
| SHO | 39.64 |
| PAS | 53.08 |
| DRI | 56.67 |
| DEF | 64.81 |
| PHY | 71.53 |

Number of players:

`3783`

This cluster has relatively high defending and physicality but much lower shooting.

It therefore appears to represent a more **defense-oriented player profile**.

Center backs make up a particularly large portion of this cluster.


### Cluster 1

Average attributes:

| Attribute | Average |
|---|---:|
| PAC | 72.75 |
| SHO | 62.42 |
| PAS | 63.68 |
| DRI | 69.37 |
| DEF | 49.56 |
| PHY | 65.36 |

Number of players:

`8089`

This cluster has higher pace, shooting, passing, and dribbling but lower defending.

It therefore appears to represent a more **attack-oriented player profile**.


## Interpretation

The strongest natural separation in the six-attribute dataset appears to be approximately:

`Defense-oriented players ↔ Attack-oriented players`

However, this does NOT mean that soccer players only belong to two meaningful categories.

K = 2 simply produced the clearest statistical separation according to the silhouette score.

There may still be meaningful subgroups within these broad clusters, such as:

- defensive midfielders
- box-to-box midfielders
- creative midfielders
- fullbacks
- wingers
- strikers

Therefore, clustering with larger values of K can still be useful for exploratory analysis even though K = 2 has the highest silhouette score.


## Relationship to PCA

Stage 2 used PCA to reduce the six-dimensional player representation into two dimensions for visualization.

Stage 3.1 uses the original standardized six attributes for K-Means clustering.

PCA coordinates are useful for visualizing the resulting clusters, but clustering itself should not depend only on PC1 and PC2 because reducing the data to two dimensions removes some information.


## Main Takeaway

The six EAFC attributes contain a strong broad attacking-versus-defending structure.

However, clustering alone does not provide enough information to identify more specific soccer roles.

This motivates Stage 3.2, where attacking and defensive ability are represented explicitly rather than relying entirely on unsupervised clusters.