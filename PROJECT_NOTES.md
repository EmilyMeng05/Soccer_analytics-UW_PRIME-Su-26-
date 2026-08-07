# Stage 1: Representing Players

The first stage of this project focuses on building a clean mathematical representation for every player. Before performing any machine learning or optimization, I wanted to make sure the dataset was well understood and suitable for analysis.

For this stage, I used the EAFC26 men's player ratings dataset as the primary source of player attributes.

## Dataset Overview

- Original dataset: **16,228 players**
- After applying a first-round quality filter (**OVR > 60**): **13,146 players**

The OVR > 60 cutoff is **not** intended to define elite players. Instead, it serves as a simple filtering step that removes very low-rated players while still keeping a large candidate pool for future team selection.

## Separating Goalkeepers

Since goalkeepers use a completely different rating system from outfield players, I decided to analyze them separately.

Current player counts:

- **Outfield players:** 11,872
- **Goalkeepers:** 1,274

The remainder of this project will initially focus on outfield players represented using the six primary EAFC attributes.

## Player Representation

For every outfield player, I define the following feature vector:

```text
Player = [PAC, SHO, PAS, DRI, DEF, PHY]
```

where

- **PAC** – Pace
- **SHO** – Shooting
- **PAS** – Passing
- **DRI** – Dribbling
- **DEF** – Defending
- **PHY** – Physicality

This six-dimensional representation will become the foundation for the similarity analysis, PCA, clustering, and team optimization stages later in the project.

---

## Initial Observations

### Overall Ratings Across Positions

One interesting observation is that the average overall rating is surprisingly similar across different positions.

| Position | Average OVR |
|-----------|------------:|
| Attacking Midfielder | 69.47 |
| Defensive Midfielder | 69.09 |
| Right Winger | 68.91 |
| Left Winger | 68.73 |
| Right Midfielder | 68.65 |
| Left Midfielder | 68.58 |
| Center Back | 68.52 |
| Central Midfielder | 68.41 |
| Goalkeeper | 68.39 |
| Striker | 68.37 |
| Left Back | 68.18 |
| Right Back | 67.97 |

The difference between the highest and lowest average rating is only **1.5 overall rating points**, suggesting that EAFC26's overall rating system is relatively balanced across different positions.


### Missing Values

The dataset is generally very clean.

Only one important variable contains missing values:

| Variable | Missing |
|----------|--------:|
| Alternative positions | 3,173 |

All other player attributes used in this project contain **no missing values**.

The missing alternative positions most likely indicate players who naturally play only one position.

### Attribute Summary

The median player attributes are:

| Attribute | Median |
|-----------|-------:|
| Pace | 70 |
| Shooting | 58 |
| Passing | 61 |
| Dribbling | 66 |
| Defending | 60 |
| Physicality | 68 |

One interesting observation is that **shooting has the lowest median value**, which makes sense because the dataset contains defenders, midfielders, and fullbacks in addition to attackers.


### Attribute Variation

The largest variation occurs in **Defending**, while **Passing** varies the least.

This also aligns with intuition. Defenders generally have very high defending ratings, while attackers often have very low defending ratings, creating a wide spread across the dataset. Passing, on the other hand, tends to be more balanced across different positions.


## Conclusions

Stage 1 successfully produced a clean player representation for future analysis.

Key takeaways include:

- The dataset is clean with almost no missing values.
- Goalkeepers should be analyzed separately from outfield players.
- Overall ratings are fairly balanced across positions.
- Defending is the attribute with the greatest variation across players.
- The six primary EAFC attributes provide a strong starting point for representing player ability.

The cleaned dataset generated in this stage will be used throughout the remainder of the project.

## Questions for the Next Stage

Now that every player has been represented as a six-dimensional feature vector, several questions naturally arise:

- Which attributes explain the largest differences between players?
- Can players be grouped into natural playing styles?
- How should player similarity be mathematically defined?
- Do traditional positions emerge naturally from the data, or do new player archetypes appear?

These questions will be explored in **Stage 2: Player Similarity and Principal Component Analysis (PCA).**


# Stage 2: Player Similarity

After creating a clean mathematical representation for every player in Stage 1, the next question naturally becomes:

> **How can we mathematically define when two players are similar?**

Instead of relying on subjective opinions, I wanted to use the six primary EAFC26 attributes to compare players directly. Every outfield player is represented by the feature vector

```text
Player = [PAC, SHO, PAS, DRI, DEF, PHY]
```

where each attribute describes a different aspect of the player's abilities.

Before comparing players, I standardized each attribute so that every feature contributes equally to the analysis. Without standardization, attributes with larger variation would dominate the similarity calculation.

## Principal Component Analysis (PCA)

To better understand the overall structure of the dataset, I performed Principal Component Analysis (PCA).

The six-dimensional player representation was reduced to two principal components for visualization.

### Explained Variance

| Principal Component | Variance Explained |
|--------------------|-------------------:|
| PC1 | 48.4% |
| PC2 | 24.8% |
| PC3 | 12.1% |
| PC4 | 11.2% |
| PC5 | 2.0% |
| PC6 | 1.5% |

The first two principal components explain approximately **73.2%** of the total variance in the original six-dimensional dataset.

This means that a two-dimensional visualization preserves most of the important information while making it much easier to interpret player relationships.

## Interpreting the Principal Components

The PCA loadings provide insight into what each principal component represents.

### PC1

| Attribute | Loading |
|-----------|---------:|
| PAC | +0.362 |
| SHO | +0.511 |
| PAS | +0.444 |
| DRI | +0.540 |
| DEF | -0.272 |
| PHY | -0.210 |

The first principal component appears to represent an **attacking and technical ability axis**.

Players with high pace, shooting, passing, and dribbling tend to have larger PC1 values, while players with stronger defensive and physical attributes tend to lie in the opposite direction.

### PC2

| Attribute | Loading |
|-----------|---------:|
| PAC | -0.174 |
| SHO | +0.049 |
| PAS | +0.461 |
| DRI | +0.224 |
| DEF | +0.628 |
| PHY | +0.557 |

The second principal component appears to capture a combination of **defensive ability, physicality, and ball distribution**.

This suggests that PC2 helps distinguish players who contribute more defensively while still maintaining strong passing ability.

## Player Similarity

Although PCA provides an intuitive visualization of the data, I did **not** use the two principal components to calculate player similarity.

Instead, similarity was computed using the original six standardized attributes. This preserves all of the information contained in the player representation.

I used the Euclidean distance between standardized player vectors to identify the nearest neighbors for each player.

## Initial Validation

As an initial sanity check, I searched for players most similar to **Lionel Messi**.

The five closest players returned by the model were:

1. Paulo Dybala
2. Riyad Mahrez
3. Iago Aspas
4. Leandro Trossard
5. Ángel Di María

This result closely matches my soccer intuition.

Rather than returning random high-rated players, the model identified technically gifted, creative attacking players who possess similar attribute profiles to Messi, so it indeed works!

This provides encouraging evidence that the six-dimensional player representation captures meaningful similarities between playing styles.

## Conclusions

Stage 2 demonstrates that the six primary EAFC26 attributes already contain a significant amount of information about player style.

Some key observations include:

- The first two principal components explain over **73%** of the variation in player attributes.
- PC1 appears to describe attacking and technical ability.
- PC2 appears to describe defensive and physical contribution.
- Similarity computed from standardized player attributes produces results that align well with soccer intuition.

These findings suggest that the player representation developed in Stage 1 is suitable for identifying similar players and can serve as the foundation for clustering player archetypes.

## Questions for the Next Stage

Now that player similarity has been established, the next question becomes:

- Can players be grouped into natural playing archetypes?
- How many distinct player styles exist in the dataset?
- Do these clusters simply reproduce traditional positions, or do they reveal new styles of play?
- Can player clusters help build more balanced soccer teams by reducing redundancy between players?

These questions will be explored in **Stage 3: Player Clustering and Playing Archetypes**.