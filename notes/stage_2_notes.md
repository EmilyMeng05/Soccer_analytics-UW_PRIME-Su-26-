# Stage 2: Understanding the Player Feature Space

**See player_similarity.py**

## Goal

Stage 1 created a mathematical representation for every player using the six
main EAFC attributes:

- Pace (PAC)
- Shooting (SHO)
- Passing (PAS)
- Dribbling (DRI)
- Defending (DEF)
- Physicality (PHY)

The goal of Stage 2 is to better understand this six-dimensional player space
before attempting to discover player archetypes.

Instead of immediately grouping players, I first wanted to answer several
questions:

- Which attributes tend to vary together?
- Can the six-dimensional player representation be simplified?
- Does the mathematical representation actually capture player similarity?


## Principal Component Analysis (PCA)

Since each player is represented using six attributes, it is difficult to
visualize the player space directly.

Principal Component Analysis (PCA) was used to reduce the six-dimensional
representation into two principal components while preserving as much variation
in the data as possible.

The first two principal components explained approximately:

- PC1: 48.4% of the total variance
- PC2: 24.8% of the total variance

Together, these two components explain approximately **73.2%** of the variation
among players.

This means that although each player is originally represented by six
attributes, much of the variation can already be understood using only two
dimensions.

## Interpreting the Principal Components

Examining the PCA loadings suggests that the principal components capture two
major directions in the player space.

### Principal Component 1 (PC1)

PC1 places large positive weights on:

- Dribbling
- Shooting
- Passing
- Pace

while Defending and Physicality contribute negatively.

This suggests that PC1 mainly represents a player's **attacking and technical
ability**.

Players with larger PC1 values tend to be more attack-oriented.

### Principal Component 2 (PC2)

PC2 places the largest positive weights on:

- Defending
- Physicality

with Passing and Dribbling contributing moderately.

This suggests that PC2 primarily captures a player's **defensive and physical
profile**.

Players with larger PC2 values tend to contribute more defensively.

## Player Similarity

After constructing the player space, I wanted to test whether nearby players
were actually similar from a soccer perspective.

Using Euclidean distance in the six-dimensional attribute space, I searched for
players most similar to Lionel Messi.

The nearest players included:

- Paulo Dybala
- Riyad Mahrez
- Iago Aspas
- Leandro Trossard
- Ángel Di María

This result agrees well with common soccer intuition, as these players share
many of Messi's technical characteristics, including strong dribbling,
creativity, passing ability, and attacking play.

Although they are not identical players, the similarity search suggests that
the chosen mathematical representation captures meaningful relationships between
players.

## Main Findings

Several important observations were made during Stage 2.

- The six EAFC attributes provide a meaningful mathematical representation of
  players.

- PCA reduces the six-dimensional player space to two dimensions while still
  preserving over 70% of the total variation.

- The first principal component appears to represent attacking and technical
  ability.

- The second principal component appears to represent defensive and physical
  ability.

- Similar-player searches produce intuitive soccer comparisons, providing
  confidence that the representation captures meaningful playing styles.

## Questions for the Next Stage

Now that player similarity has been established, the next question becomes:

- Can players be grouped into natural playing archetypes?
- How many distinct player styles exist in the dataset?
- Do these clusters simply reproduce traditional positions, or do they reveal new styles of play?
- Can player clusters help build more balanced soccer teams by reducing redundancy between players?

These questions will be explored in **Stage 3: Player Clustering and Playing Archetypes**.