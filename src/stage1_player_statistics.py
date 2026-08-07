import pandas as pd
import matplotlib.pyplot as plt

# Read the EAFC26 player dataset.
# This will be the main dataset we use to represent individual players.
df = pd.read_csv("EAFC26-Men.csv")

# Get a sense of the number of data point we have
print("Original dataset shape:", df.shape)


# Before filtering players, I want to understand what the overall rating
# distribution looks like across the entire EAFC26 dataset.
# This will help us decide what a reasonable first-round cutoff might be.

plt.figure(figsize=(8,5))

plt.hist(df["OVR"], bins=20)

plt.xlabel("Overall Rating")
plt.ylabel("Number of Players")
plt.title("Distribution of Player Overall Ratings")

plt.tight_layout()

plt.savefig(
    "before_rating_num_players.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# We do not necessarily want every player in the EAFC26 database to be
# considered when we eventually build the best team.
# For now, I am using OVR > 60 as a first-round exclusion criterion.
#
# This is not meant to define an "elite" player.
# Instead, it gives us a reasonably large candidate pool while removing
# players with very low overall ratings.
#
# We can always experiment with this cutoff later and see whether changing
# it affects our clustering or final team selection.

players = df[df["OVR"] > 60].copy()

print("Players after filtering:", players.shape)


# The original dataset uses abbreviated position names such as ST, CM, and CB.
# I want to replace these with full names so that future graphs and tables
# are easier to read and understand.

position_map = {
    "GK":"Goalkeeper",
    "CB":"Center Back",
    "LB":"Left Back",
    "RB":"Right Back",
    "LWB":"Left Wing Back",
    "RWB":"Right Wing Back",
    "CDM":"Defensive Midfielder",
    "CM":"Central Midfielder",
    "CAM":"Attacking Midfielder",
    "LM":"Left Midfielder",
    "RM":"Right Midfielder",
    "LW":"Left Winger",
    "RW":"Right Winger",
    "CF":"Center Forward",
    "ST":"Striker"
}

players["Position"] = players["Position"].replace(position_map)


# Before comparing players, it is useful to understand how many players
# we have for each position.
#
# If one position has significantly more players than another, this may
# affect some of our later analysis and clustering results.

players["Position"].value_counts().plot(
    kind="bar",
    figsize=(10,6)
)

plt.ylabel("Number of Players")
plt.title("Player Position Distribution")

plt.tight_layout()

plt.show()


# I also want to check whether some positions naturally have higher
# overall ratings than others.
#
# This could become important later because comparing a striker directly
# with a center back using only overall rating may not always be meaningful.

avg_rating = (
    players
    .groupby("Position")["OVR"]
    .mean()
    .round(2)
    .sort_values(ascending=False)
)

print(avg_rating)

avg_rating.to_csv("average_rating_by_position.csv")


# Goalkeepers have a completely different rating system from outfield players.
#
# Outfield players are mainly represented using PAC, SHO, PAS, DRI, DEF,
# and PHY, while goalkeepers have attributes such as diving, handling,
# kicking, positioning, and reflexes.
#
# Since we eventually want to compare players based on similar attribute
# profiles, I will separate goalkeepers from the other players for now.
# We can build a separate goalkeeper analysis later.

goalkeepers = players[
    players["Position"] == "Goalkeeper"
].copy()

outfield_players = players[
    players["Position"] != "Goalkeeper"
].copy()

print("Outfield players:", len(outfield_players))
print("Goalkeepers:", len(goalkeepers))


# For the first version of the project, I want to represent each outfield
# player using the six main EAFC attributes.
#
# Mathematically, each player can be thought of as a point in a
# six-dimensional space:
#
# player = [PAC, SHO, PAS, DRI, DEF, PHY]
#
# This representation will later allow us to measure player similarity,
# perform PCA, and cluster players into different playing archetypes.

features = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY"
]


# In addition to the numerical attributes, I want to keep identifying
# information that will be useful when interpreting the results.
#
# For example, after clustering players, we will want to know their
# position, team, league, and play style rather than only seeing numbers.

player_columns = [
    "ID",
    "Name",
    "OVR",
    "Position",
    "Alternative positions",
    "Age",
    "Nation",
    "League",
    "Team",
    "play style"
]

player_features = outfield_players[
    player_columns + features
].copy()


# Before doing any machine learning, I want to check for missing values.
#
# Missing values can cause problems later when we standardize the features,
# run PCA, or perform clustering.
#
# If there are missing values, we should investigate why they are missing
# instead of immediately deleting the players.

print("\nMissing values")

print(player_features.isnull().sum())


# It is also useful to look at the basic distribution of each attribute.
#
# This gives us a sense of the average player and how much variation exists
# in attributes such as pace, shooting, defending, and physicality.

print("\nAttribute summary")

print(player_features[features].describe())


# At this point, we have created the basic mathematical representation
# of each outfield player.
#
# I will save this cleaned dataset so that future stages of the project
# can start from the same player representation instead of repeatedly
# cleaning the raw EAFC26 dataset.

player_features.to_csv(
    "cleaned_eafc26_outfield_players.csv",
    index=False
)


# I will also save the goalkeepers separately so that we can analyze them
# later using their own goalkeeper-specific attributes.

goalkeepers.to_csv(
    "cleaned_eafc26_goalkeepers.csv",
    index=False
)

print("\nStage 1 complete!")


# PRINTED RESULTS

# Original dataset shape: (16228, 59)
# Players after filtering: (13146, 59)

# Position
# Attacking Midfielder    69.47
# Defensive Midfielder    69.09
# Right Winger            68.91
# Left Winger             68.73
# Right Midfielder        68.65
# Left Midfielder         68.58
# Center Back             68.52
# Central Midfielder      68.41
# Goalkeeper              68.39
# Striker                 68.37
# Left Back               68.18
# Right Back              67.97
# Name: OVR, dtype: float64
# Outfield players: 11872
# Goalkeepers: 1274

# Missing values
# ID                          0
# Name                        0
# OVR                         0
# Position                    0
# Alternative positions    3173
# Age                         0
# Nation                      0
# League                      0
# Team                        0
# play style                  0
# PAC                         0
# SHO                         0
# PAS                         0
# DRI                         0
# DEF                         0
# PHY                         0
# dtype: int64

# Attribute summary
#                 PAC           SHO           PAS           DRI           DEF           PHY
# count  11872.000000  11872.000000  11872.000000  11872.000000  11872.000000  11872.000000
# mean      69.277460     55.160967     60.304245     65.320586     54.420822     67.329346
# std       10.922569     13.721748      8.797093      8.447751     15.997529      8.730246
# min       30.000000     21.000000     31.000000     32.000000     17.000000     35.000000
# 25%       63.000000     45.000000     55.000000     61.000000     40.000000     62.000000
# 50%       70.000000     58.000000     61.000000     66.000000     60.000000     68.000000
# 75%       77.000000     65.000000     66.000000     71.000000     66.000000     74.000000
# max       97.000000     92.000000     92.000000     93.000000     90.000000     91.000000

# Stage 1 complete!