import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# STAGE 3.3: LEARNING POSSIBLE POSITIONS FROM PLAYER ATTRIBUTES
#
# Up to this point, the EAFC position column has mostly been used as
# descriptive information.
#
# However, many soccer players are capable of playing multiple roles.
#
# If we force every player to stay in their listed EAFC position,
# we may lose interesting possibilities when building teams later.
#
# In this stage, I want to ask:
#
# Can a player's likely positions be learned directly from their
# six main EAFC attributes?
#
# The model will NOT be told that strikers should have high shooting
# or that center backs should have high defending.
#
# Instead, it will learn those relationships from players whose
# positions are already known.


# Read the cleaned outfield player dataset from Stage 1.
df = pd.read_csv("cleaned_eafc26_outfield_players.csv")

# Use the same six-dimensional player representation that has been used
# throughout the project.

features = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY"
]


# The target variable is the player's listed EAFC position.
#
# This makes Stage 3.3 a supervised learning problem:
#
# inputs:
# PAC, SHO, PAS, DRI, DEF, PHY
#
# output:
# Position

X = df[features]
y = df["Position"]


# Split the player dataset into training and testing sets.
#
# The model will learn the relationship between player attributes and
# positions using the training set.
#
# The test set contains players the model has not seen during training.
#
# This allows us to evaluate whether the learned relationships generalize
# to unseen players.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Standardize the player attributes.
#
# This keeps the scale of each feature comparable and is especially useful
# for Logistic Regression.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Start with multinomial Logistic Regression.
#
# Logistic Regression is useful here because:
#
# 1. It can predict multiple classes.
# 2. It produces probabilities for every possible position.
# 3. It is easier to interpret than many more complicated models.
#
# The most important output for this project is not only the single
# predicted position.
#
# We want the full probability distribution because that tells us
# whether a player may reasonably fit several positions.

model = LogisticRegression(
    max_iter=3000,
    random_state=42
)

model.fit(
    X_train_scaled,
    y_train
)


# Evaluate the model using players it did not see during training.

y_pred = model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel accuracy:")
print(round(accuracy, 4))


# Print a classification report.
#
# Precision, recall, and F1-score are especially useful here because
# some positions occur much more frequently than others.

print("\nClassification report:")

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# Create a confusion matrix.
#
# This will help us understand which positions the model tends
# to confuse with one another.
#
# Those mistakes may actually be interesting.
#
# For example:
#
# CM being confused with CDM or CAM
#
# LW being confused with LM
#
# RW being confused with RM
#
# could reflect genuine positional similarity rather than completely
# incorrect predictions.

position_labels = model.classes_

confusion = confusion_matrix(
    y_test,
    y_pred,
    labels=position_labels
)

confusion_df = pd.DataFrame(
    confusion,
    index=position_labels,
    columns=position_labels
)

print("\nConfusion matrix:")
print(confusion_df)

confusion_df.to_csv(
    "position_prediction_confusion_matrix.csv"
)


# Now train the same model representation on every player.
#
# The model will generate a probability for every possible position.
#
# For example, a player might receive:
#
# CM = 0.40
# CAM = 0.32
# CDM = 0.18
# RM = 0.07
# ST = 0.03
#
# These probabilities give us a continuous measure of positional fit
# instead of forcing every player into one position.

X_all_scaled = scaler.transform(
    df[features]
)

position_probabilities = model.predict_proba(
    X_all_scaled
)


# Convert the probability matrix into a DataFrame.
#
# Each column represents one possible position.

probability_columns = [
    f"Prob_{position}"
    for position in model.classes_
]

probability_df = pd.DataFrame(
    position_probabilities,
    columns=probability_columns,
    index=df.index
)


# Add all position probabilities back to the player dataset.

results = pd.concat(
    [
        df,
        probability_df
    ],
    axis=1
)


# Find each player's three most likely positions.
#
# The top three positions make the results easier for people to read.
#
# However, the full probability distribution will still be saved because
# Stage 3.4 and Stage 4 may need more than only the top three choices.

def get_top_positions(probabilities, classes, top_n=3):

    order = probabilities.argsort()[::-1][:top_n]

    positions = [
        classes[i]
        for i in order
    ]

    scores = [
        probabilities[i]
        for i in order
    ]

    return positions, scores


top_1_positions = []
top_1_probabilities = []

top_2_positions = []
top_2_probabilities = []

top_3_positions = []
top_3_probabilities = []


for probabilities in position_probabilities:

    positions, scores = get_top_positions(
        probabilities,
        model.classes_,
        top_n=3
    )

    top_1_positions.append(
        positions[0]
    )

    top_1_probabilities.append(
        scores[0]
    )

    top_2_positions.append(
        positions[1]
    )

    top_2_probabilities.append(
        scores[1]
    )

    top_3_positions.append(
        positions[2]
    )

    top_3_probabilities.append(
        scores[2]
    )


results["PredictedPosition1"] = top_1_positions
results["PositionProbability1"] = top_1_probabilities

results["PredictedPosition2"] = top_2_positions
results["PositionProbability2"] = top_2_probabilities

results["PredictedPosition3"] = top_3_positions
results["PositionProbability3"] = top_3_probabilities


# Compare the model's most likely position with the listed EAFC position.
#
# If they are different, that does NOT automatically mean the model is wrong.
#
# These disagreements may identify players whose attribute profiles resemble
# another tactical position.

results["OfficialPositionMatch"] = (
    results["PredictedPosition1"]
    == results["Position"]
)


# Print some example results.

example_columns = [
    "Name",
    "Position",
    "PredictedPosition1",
    "PositionProbability1",
    "PredictedPosition2",
    "PositionProbability2",
    "PredictedPosition3",
    "PositionProbability3"
]

print("\nExample predicted positions:")

print(
    results[
        example_columns
    ]
    .head(25)
)


# Create a function that allows us to inspect individual players.
#
# This will be useful for players such as:
#
# Jude Bellingham
# Federico Valverde
# Achraf Hakimi
# Lionel Messi
#
# where we are especially interested in positional flexibility.

def show_player_positions(player_name):

    player = results[
        results["Name"].str.lower()
        == player_name.lower()
    ]

    if len(player) == 0:

        print(
            f"\nPlayer '{player_name}' not found."
        )

        return


    print(
        f"\nPosition predictions for {player_name}:"
    )

    print(
        player[
            [
                "Name",
                "Position",
                "PredictedPosition1",
                "PositionProbability1",
                "PredictedPosition2",
                "PositionProbability2",
                "PredictedPosition3",
                "PositionProbability3"
            ]
        ]
    )


# Test several interesting players.

show_player_positions(
    "Jude Bellingham"
)

show_player_positions(
    "Federico Valverde"
)

show_player_positions(
    "Achraf Hakimi"
)

show_player_positions(
    "Lionel Messi"
)


# We can also look for players whose model-predicted position differs
# from their listed EAFC position.
#
# These players may be especially interesting for the role-based
# team-building part of the project.

position_disagreements = results[
    results["OfficialPositionMatch"] == False
].copy()

position_disagreements = (
    position_disagreements
    .sort_values(
        "OVR",
        ascending=False
    )
)


print("\nHighest-rated players whose predicted position differs from their listed position:")

print(
    position_disagreements[
        example_columns
    ]
    .head(30)
)


# Save the complete results.
#
# This file includes:
#
# original player information
#
# six EAFC attributes
#
# full probability distribution across positions
#
# top three predicted positions
#
# probability for each predicted position
#
# whether the model's first prediction matches the official EAFC position
#
# Stage 3.4 can use this file to visualize positional flexibility.

results.to_csv(
    "player_position_predictions.csv",
    index=False
)


# Save the position disagreements separately because these players
# may provide some of the most interesting examples in the project.

position_disagreements.to_csv(
    "position_prediction_disagreements.csv",
    index=False
)


# Stage 3.3 does NOT claim that the model has discovered a player's
# true position.
#
# The official EAFC position is being used as the training label,
# so the model is learning which attribute profiles are associated
# with those existing position labels.
#
# The useful result is the probability distribution.
#
# A player with:
#
# CM  = 0.42
# CAM = 0.35
# CDM = 0.15
#
# appears much more positionally flexible than a player with:
#
# CB = 0.95
#
# This provides a mathematical foundation for exploring positional
# versatility without allowing every player to play every role equally.
#
# Stage 3.4 can use these probabilities together with the attacking
# and defensive contribution scores from Stage 3.2 to build role maps.

print("\nStage 3.3 position prediction complete!")