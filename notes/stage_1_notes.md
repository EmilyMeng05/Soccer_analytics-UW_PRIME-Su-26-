# Stage 1: Representing Players

**See data_process.py**

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