from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# STAGE 1: PLAYER STATISTICS
#
# the goal of this stage is to understand the player population
# that will be used throughout the rest of the project.

# Define project paths.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA = PROJECT_ROOT / "data"

RESULTS = (
    PROJECT_ROOT
    / "results"
    / "stage_1"
)

RESULTS.mkdir(
    parents=True,
    exist_ok=True
)


# Read the cleaned datasets created by prepare_dataset.py.
#
# cleaned_eafc26_players contains both goalkeepers and outfield players.
#
# cleaned_eafc26_outfield_players contains only players who can be
# represented using the six main EAFC outfield attributes.

players = pd.read_csv(
    DATA
    / "processed"
    / "cleaned_eafc26_players.csv"
)

outfield_players = pd.read_csv(
    DATA
    / "processed"
    / "cleaned_eafc26_outfield_players.csv"
)

goalkeepers = pd.read_csv(
    DATA
    / "processed"
    / "cleaned_eafc26_goalkeepers.csv"
)


print(
    "Total players in candidate pool:",
    len(players)
)

print(
    "Outfield players:",
    len(outfield_players)
)

print(
    "Goalkeepers:",
    len(goalkeepers)
)


# Look at the overall-rating distribution of the candidate player pool.
#
# Since prepare_dataset.py already applied OVR > 60, this graph describes
# the distribution AFTER the first-round candidate filter.

plt.figure(
    figsize=(8,5)
)

plt.hist(
    players["OVR"],
    bins=20
)

plt.xlabel(
    "Overall Rating"
)

plt.ylabel(
    "Number of Players"
)

plt.title(
    "Distribution of Player Overall Ratings After OVR Filter"
)

plt.tight_layout()

plt.savefig(
    RESULTS
    / "rating_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Examine the number of players at each position.
#
# Position imbalance may become important later because some positions
# contain many more examples than others.

position_counts = (
    players["Position"]
    .value_counts()
)


print(
    "\nPlayer counts by position:"
)

print(
    position_counts
)


plt.figure(
    figsize=(10,6)
)

position_counts.plot(
    kind="bar"
)

plt.xlabel(
    "Position"
)

plt.ylabel(
    "Number of Players"
)

plt.title(
    "Player Position Distribution"
)

plt.tight_layout()

plt.savefig(
    RESULTS
    / "position_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# Calculate average overall rating by position.
#
# This helps us check whether some positions have systematically
# higher or lower ratings within the candidate population.

avg_rating = (
    players
    .groupby(
        "Position"
    )["OVR"]
    .mean()
    .round(2)
    .sort_values(
        ascending=False
    )
)


print(
    "\nAverage overall rating by position:"
)

print(
    avg_rating
)


avg_rating.to_csv(
    RESULTS
    / "average_rating_by_position.csv"
)


# The six main EAFC outfield attributes form the basic mathematical
# representation used throughout the first several stages of the project.
#
# Each outfield player can be represented as:
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


# Check missing values in the cleaned outfield dataset.
#
# Ideally, the core six attributes should contain no missing values
# because prepare_dataset.py already checks them.

print(
    "\nMissing values in outfield player dataset:"
)

print(
    outfield_players.isnull().sum()
)


# Examine the basic distributions of the six core attributes.

attribute_summary = (
    outfield_players[
        features
    ]
    .describe()
)


print(
    "\nAttribute summary:"
)

print(
    attribute_summary
)


attribute_summary.to_csv(
    RESULTS
    / "attribute_summary.csv"
)


# Calculate correlations between the six player attributes.
#
# This gives an early indication that some attributes may describe
# related aspects of player ability.
#
# Stage 2 will investigate this structure more formally using PCA.

correlation = (
    outfield_players[
        features
    ]
    .corr()
    .round(3)
)


print(
    "\nAttribute correlations:"
)

print(
    correlation
)


correlation.to_csv(
    RESULTS
    / "attribute_correlations.csv"
)


# Stage 1 provides a descriptive overview of the cleaned EAFC26
# candidate population.
#
# The main outputs are:
#
# player rating distribution
#
# position distribution
#
# average rating by position
#
# six-attribute summary
#
# attribute correlations
#
# Stage 2 will then move beyond descriptive statistics and investigate
# whether the six-dimensional player representation can be simplified
# into a smaller number of meaningful dimensions.


print(
    "\nStage 1 player statistics complete!"
)