from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# STAGE 3.4: LEARNING POSITION PROFILES
#
# Stage 3.3 tested whether a player's official position could be predicted
# using the six broad EAFC attributes:
#
# PAC
# SHO
# PAS
# DRI
# DEF
# PHY
#
# Logistic Regression and Random Forest were both able to recognize some
# broad positional patterns, especially Center Back and Striker.
#
# However, the models struggled with positions that have very similar
# six-attribute profiles, such as:
#
# LW vs LM
# RW vs RM
# CAM vs CM
# CM vs CDM
#
# Stage 3.4 therefore asks a different question:
#
# What actually characterizes each soccer position?
#
# Instead of immediately trying another model, I want to examine the
# average attribute profile associated with every official EAFC position.
#
# I will use:
#
# 1. The six broad EAFC attributes.
# 2. Any detailed attributes available in the full-feature dataset.
#
# These average profiles will become "position fingerprints."
#
# Later, we can compare individual players with these fingerprints to
# explore which positions their attribute profiles resemble.


# Define project paths.

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA = PROJECT_ROOT / "data"

RESULTS = (
    PROJECT_ROOT
    / "results"
    / "stage_3_4"
    / "position_profiles"
)

RESULTS.mkdir(
    parents=True,
    exist_ok=True
)


# Read the full-feature outfield dataset created by prepare_dataset.py.
#
# This file should preserve most of the original EAFC26 columns rather than
# only the six broad attributes.

df = pd.read_csv(
    DATA
    / "processed"
    / "eafc26_outfield_full_features.csv"
)


print(
    "Dataset shape:",
    df.shape
)

print(
    "\nNumber of columns:",
    len(df.columns)
)


# Print all available columns.
#
# This helps verify that prepare_dataset.py successfully preserved the
# richer EAFC attributes.

print(
    "\nAvailable columns:"
)

for column in df.columns:

    print(column)


# Keep the original six broad EAFC attributes.
#
# These connect Stage 3.4 directly with the earlier analyses.

core_features = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY"
]


# Candidate detailed attributes.
#
# Different EAFC datasets may use slightly different names for the same
# concept.
#
# Therefore, this list includes several common naming variations.
#
# The script will automatically keep only the attributes that actually
# exist in the dataset.

rich_feature_candidates = [

    # Pace-related attributes

    "Acceleration",
    "Sprint Speed",
    "SprintSpeed",


    # Shooting-related attributes

    "Positioning",
    "Finishing",
    "Shot Power",
    "ShotPower",
    "Long Shots",
    "LongShots",
    "Volleys",
    "Penalties",


    # Passing-related attributes

    "Vision",
    "Crossing",
    "Free Kick Accuracy",
    "FKAccuracy",
    "Short Passing",
    "ShortPassing",
    "Long Passing",
    "LongPassing",
    "Curve",


    # Dribbling-related attributes

    "Agility",
    "Balance",
    "Reactions",
    "Ball Control",
    "BallControl",
    "Dribbling",
    "Composure",


    # Defensive attributes

    "Interceptions",
    "Heading Accuracy",
    "HeadingAccuracy",
    "Defensive Awareness",
    "DefensiveAwareness",
    "Standing Tackle",
    "StandingTackle",
    "Sliding Tackle",
    "SlidingTackle",


    # Physical attributes

    "Jumping",
    "Stamina",
    "Strength",
    "Aggression"
]


# Keep only detailed attributes that actually exist.

rich_features = [

    feature

    for feature in rich_feature_candidates

    if feature in df.columns
]


print(
    "\nNumber of detailed attributes found:",
    len(rich_features)
)

print(
    "\nDetailed attributes found:"
)

print(
    rich_features
)


# Combine the six broad attributes with the detailed attributes.
#
# dict.fromkeys removes duplicate feature names while preserving order.

features = list(
    dict.fromkeys(
        core_features
        + rich_features
    )
)


print(
    "\nTotal number of features being used:",
    len(features)
)

print(
    "\nFeatures being used:"
)

print(
    features
)


# Make sure the selected features are numeric.
#
# Invalid entries will be converted to NaN.

for feature in features:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce"
    )


# Remove players missing their official position.

df = df.dropna(
    subset=[
        "Position"
    ]
).copy()


# Remove players missing any of the selected numerical features.
#
# This makes sure every player used in the fingerprint calculation has
# complete data for the chosen feature set.

before_missing_filter = len(
    df
)

df = df.dropna(
    subset=features
).copy()

after_missing_filter = len(
    df
)


print(
    "\nPlayers removed because of missing selected attributes:",
    before_missing_filter - after_missing_filter
)

print(
    "Players remaining:",
    len(df)
)


# Count players in each official position.
#
# The sample size for each position matters because very small groups may
# produce less stable position averages.

position_counts = (
    df[
        "Position"
    ]
    .value_counts()
    .sort_values(
        ascending=False
    )
)


print(
    "\nPlayers per position:"
)

print(
    position_counts
)


position_counts.to_csv(
    RESULTS
    / "position_counts.csv"
)


# Calculate the raw average attribute profile for every position.
#
# This gives the first version of the position fingerprint.
#
# For example, Center Backs may naturally show:
#
# high DEF
# high PHY
# lower SHO
#
# while Strikers may show:
#
# high SHO
# lower DEF
#
# These patterns are learned directly from the player data.

position_profiles = (
    df
    .groupby(
        "Position"
    )[features]
    .mean()
    .round(2)
)


print(
    "\nAverage attribute profile by position:"
)

print(
    position_profiles
)


position_profiles.to_csv(
    RESULTS
    / "position_attribute_profiles.csv"
)


# Standardize player attributes.
#
# Raw averages tell us the absolute EAFC values.
#
# Standardized values tell us whether a position is unusually strong or
# weak in a particular attribute relative to the average outfield player.
#
# A standardized score around:
#
# 0
#
# means the position is close to the overall player average.
#
# Positive values mean the position tends to be above average.
#
# Negative values mean the position tends to be below average.

scaler = StandardScaler()

scaled_player_features = scaler.fit_transform(
    df[
        features
    ]
)


scaled_df = pd.DataFrame(
    scaled_player_features,
    columns=features,
    index=df.index
)


scaled_df[
    "Position"
] = df[
    "Position"
]


# Calculate the mean standardized score for each position.
#
# These values form the standardized position fingerprints.

position_zscores = (
    scaled_df
    .groupby(
        "Position"
    )[features]
    .mean()
    .round(3)
)


print(
    "\nStandardized position fingerprints:"
)

print(
    position_zscores
)


position_zscores.to_csv(
    RESULTS
    / "position_fingerprints_zscore.csv"
)


# Identify the strongest characteristics for every position.
#
# The five largest standardized scores represent the attributes where
# players at that position tend to be most above the average outfield player.

top_attributes_rows = []


for position in position_zscores.index:

    position_values = (
        position_zscores
        .loc[
            position
        ]
        .sort_values(
            ascending=False
        )
    )


    top_features = (
        position_values
        .head(
            5
        )
    )


    print(
        f"\nTop characteristics for {position}:"
    )

    print(
        top_features
    )


    for rank, (
        feature,
        score
    ) in enumerate(
        top_features.items(),
        start=1
    ):

        top_attributes_rows.append({

            "Position":
                position,

            "Rank":
                rank,

            "Feature":
                feature,

            "StandardizedScore":
                score

        })


top_attributes_df = pd.DataFrame(
    top_attributes_rows
)


top_attributes_df.to_csv(
    RESULTS
    / "top_attributes_by_position.csv",
    index=False
)


# Identify the weakest characteristics for every position.
#
# Low attributes can also help distinguish positions.
#
# For example:
#
# low SHO may help distinguish defenders
#
# while low DEF may help distinguish attackers.

bottom_attributes_rows = []


for position in position_zscores.index:

    position_values = (
        position_zscores
        .loc[
            position
        ]
        .sort_values(
            ascending=True
        )
    )


    bottom_features = (
        position_values
        .head(
            5
        )
    )


    print(
        f"\nLowest characteristics for {position}:"
    )

    print(
        bottom_features
    )


    for rank, (
        feature,
        score
    ) in enumerate(
        bottom_features.items(),
        start=1
    ):

        bottom_attributes_rows.append({

            "Position":
                position,

            "Rank":
                rank,

            "Feature":
                feature,

            "StandardizedScore":
                score

        })


bottom_attributes_df = pd.DataFrame(
    bottom_attributes_rows
)


bottom_attributes_df.to_csv(
    RESULTS
    / "lowest_attributes_by_position.csv",
    index=False
)


# Determine which attributes vary the most between positions.
#
# If one attribute has almost the same average for every position,
# it probably does not help distinguish positions very well.
#
# If an attribute changes substantially between positions,
# it may be especially informative for positional identity.

between_position_variation = (
    position_zscores
    .var()
    .sort_values(
        ascending=False
    )
)


print(
    "\nAttributes that vary most between positions:"
)

print(
    between_position_variation
)


between_position_variation.to_csv(
    RESULTS
    / "attribute_variation_between_positions.csv"
)


# Plot the attributes that vary most between positions.
#
# Once richer attributes are available, this graph may reveal which
# detailed skills are especially useful for distinguishing tactical roles.

top_variable_features = (
    between_position_variation
    .head(
        15
    )
)


plt.figure(
    figsize=(10,7)
)


top_variable_features.sort_values().plot(
    kind="barh"
)


plt.xlabel(
    "Variance Across Position Profiles"
)

plt.ylabel(
    "Attribute"
)

plt.title(
    "Attributes that Most Distinguish EAFC26 Positions"
)


plt.tight_layout()


plt.savefig(
    RESULTS
    / "most_position_distinguishing_attributes.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# Create a heatmap-style visualization of the standardized position profiles.
#
# Each row represents a position.
#
# Each column represents an attribute.
#
# Positive values indicate above-average characteristics.
#
# Negative values indicate below-average characteristics.

plt.figure(
    figsize=(
        max(
            12,
            len(features) * 0.55
        ),
        8
    )
)


plt.imshow(
    position_zscores,
    aspect="auto"
)


plt.colorbar(
    label="Standardized Score"
)


plt.xticks(
    range(
        len(features)
    ),
    features,
    rotation=90
)


plt.yticks(
    range(
        len(
            position_zscores.index
        )
    ),
    position_zscores.index
)


plt.xlabel(
    "Player Attribute"
)

plt.ylabel(
    "Official Position"
)

plt.title(
    "EAFC26 Position Fingerprints"
)


plt.tight_layout()


plt.savefig(
    RESULTS
    / "position_fingerprint_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()


# Save a six-attribute-only profile.
#
# This makes it possible to directly compare:
#
# the original six broad attributes
#
# vs
#
# the richer position representation.

core_position_profiles = (
    df
    .groupby(
        "Position"
    )[core_features]
    .mean()
    .round(2)
)


core_position_profiles.to_csv(
    RESULTS
    / "six_attribute_position_profiles.csv"
)


# Save the rich feature list separately.
#
# This makes it easy to know exactly which attributes were included in
# this Stage 3.4 experiment.

feature_list_df = pd.DataFrame({
    "Feature":
        features,

    "FeatureType":
        [
            "Core"
            if feature in core_features
            else "Rich"
            for feature in features
        ]
})


feature_list_df.to_csv(
    RESULTS
    / "features_used.csv",
    index=False
)


# Stage 3.4 does NOT yet predict alternative positions.
#
# Instead, this stage focuses on understanding what distinguishes each
# official position mathematically.
#
# Main outputs:
#
# 1. Player count by position.
#
# 2. Average raw attribute profile by position.
#
# 3. Standardized position fingerprints.
#
# 4. Strongest characteristics for every position.
#
# 5. Weakest characteristics for every position.
#
# 6. Attributes that vary most strongly between positions.
#
# 7. Comparison between the original six-feature representation and
#    the richer feature representation.
#
# The next experiment can use these rich features with:
#
# Logistic Regression
#
# and
#
# Random Forest
#
# Then we can compare:
#
# Six-feature Logistic Regression
# Six-feature Random Forest
# Rich-feature Logistic Regression
# Rich-feature Random Forest
#
# Later, the position fingerprints can also be used to compare individual
# players against multiple possible positions.


print(
    "\nStage 3.4 position profile analysis complete!"
)