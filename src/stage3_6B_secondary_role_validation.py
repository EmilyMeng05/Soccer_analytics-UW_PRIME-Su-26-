from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler


# ============================================================
# STAGE 3.6B — SECONDARY ROLE VALIDATION
# ============================================================
#
# Goal:
#
# Test whether Stage 3.5C's secondary functional-role
# predictions correspond to EAFC's official Alternative
# positions.
#
# IMPORTANT:
#
# We use OUT-OF-FOLD predictions.
#
# Each player's probabilities come from a model that was not
# trained on that player.
#
# This avoids evaluating flexibility using in-sample
# probabilities.
#
#
# MODEL:
#
# Logistic Regression
# 34 Stage 3.5C features
# 8 functional roles
#
#
# BASELINE:
#
# Given only the player's primary functional role, predict the
# most common alternative functional roles observed among the
# TRAINING players with that same primary role.
#
#
# MAIN QUESTION:
#
# Does the player's individual attribute profile recover
# official secondary roles better than primary-position
# frequency alone?
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA = PROJECT_ROOT / "data"
RESULTS = PROJECT_ROOT / "results"

INPUT_FILE = (
    RESULTS
    / "stage_3_6"
    / "A_alternative_position_analysis"
    / "normalized_alternative_positions.csv"
)

RESULTS_DIR = (
    RESULTS
    / "stage_3_6"
    / "B_secondary_role_validation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5


# ============================================================
# STAGE 3.5C FEATURES
# ============================================================

CORE_FEATURES = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY",
]

DETAILED_FEATURES = [
    "Acceleration",
    "Sprint Speed",
    "Positioning",
    "Finishing",
    "Shot Power",
    "Long Shots",
    "Volleys",
    "Penalties",
    "Vision",
    "Crossing",
    "Free Kick Accuracy",
    "Short Passing",
    "Long Passing",
    "Curve",
    "Agility",
    "Balance",
    "Reactions",
    "Ball Control",
    "Dribbling",
    "Composure",
    "Interceptions",
    "Heading Accuracy",
    "Standing Tackle",
    "Sliding Tackle",
    "Jumping",
    "Stamina",
    "Strength",
    "Aggression",
]

FEATURES = (
    CORE_FEATURES
    + DETAILED_FEATURES
)


# ============================================================
# LOAD NORMALIZED DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

print("=" * 70)
print("STAGE 3.6B — SECONDARY ROLE VALIDATION")
print("=" * 70)

print(f"\nLoaded dataset: {df.shape}")


required_columns = (
    FEATURES
    + [
        "Name",
        "Position",
        "Primary Functional Role",
        "Alternative Functional Role List",
    ]
)

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# PARSE FUNCTIONAL ALTERNATIVE LIST
# ============================================================

def parse_pipe_list(value):

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    return [
        item.strip()
        for item in text.split("|")
        if item.strip()
    ]


df[
    "Official Alternative Roles"
] = (
    df[
        "Alternative Functional Role List"
    ]
    .apply(parse_pipe_list)
)


# ============================================================
# NUMERIC FEATURES
# ============================================================

for feature in FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce",
    )


before = len(df)

df = df.dropna(
    subset=(
        FEATURES
        + [
            "Primary Functional Role",
        ]
    )
).reset_index(drop=True)

after = len(df)

print(
    f"\nDropped {before - after} players "
    f"with missing model features."
)

print(
    f"Players remaining: {after}"
)


# ============================================================
# ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df[
        "Primary Functional Role"
    ]
)

class_names = label_encoder.classes_

n_classes = len(class_names)

print("\nFunctional role classes:")

for index, role in enumerate(class_names):
    print(f"{index}: {role}")


# ============================================================
# STORAGE FOR OOF RESULTS
# ============================================================

oof_probabilities = np.zeros(
    (
        len(df),
        n_classes,
    ),
    dtype=float,
)

oof_fold = np.zeros(
    len(df),
    dtype=int,
)

baseline_rankings = [
    None
    for _ in range(len(df))
]


# ============================================================
# CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)


X = df[FEATURES].copy()


# ============================================================
# FREQUENCY BASELINE HELPER
# ============================================================
#
# For every primary role, calculate how often each OTHER
# functional role appears as an official alternative among
# training players.
#
# This baseline has no access to the player's attributes.
#
# ============================================================

def build_frequency_baseline(
    training_df,
):

    baseline = {}

    for primary_role in class_names:

        subset = training_df[
            training_df[
                "Primary Functional Role"
            ]
            == primary_role
        ]

        counts = {
            role: 0
            for role in class_names
            if role != primary_role
        }

        for alternatives in subset[
            "Official Alternative Roles"
        ]:

            for alternative_role in alternatives:

                if (
                    alternative_role
                    != primary_role
                    and alternative_role
                    in counts
                ):
                    counts[
                        alternative_role
                    ] += 1

        ranking = sorted(
            counts,
            key=lambda role: (
                -counts[role],
                role,
            ),
        )

        baseline[
            primary_role
        ] = ranking

    return baseline


# ============================================================
# RUN OOF TRAINING
# ============================================================

for fold_number, (
    train_indices,
    test_indices,
) in enumerate(
    cv.split(X, y),
    start=1,
):

    print(
        f"\nRunning fold "
        f"{fold_number}/{N_SPLITS}..."
    )

    X_train = X.iloc[
        train_indices
    ].copy()

    X_test = X.iloc[
        test_indices
    ].copy()

    y_train = y[
        train_indices
    ]


    # --------------------------------------------------------
    # Scale using TRAINING DATA ONLY.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )


    # --------------------------------------------------------
    # Stage 3.5C model.
    # --------------------------------------------------------

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train_scaled,
        y_train,
    )


    # --------------------------------------------------------
    # OOF probabilities.
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        X_test_scaled
    )

    oof_probabilities[
        test_indices
    ] = probabilities

    oof_fold[
        test_indices
    ] = fold_number


    # --------------------------------------------------------
    # Build baseline using TRAINING DATA ONLY.
    # --------------------------------------------------------

    training_df = df.iloc[
        train_indices
    ].copy()

    baseline = build_frequency_baseline(
        training_df
    )


    for test_index in test_indices:

        primary_role = df.loc[
            test_index,
            "Primary Functional Role",
        ]

        baseline_rankings[
            test_index
        ] = baseline[
            primary_role
        ]


# ============================================================
# CREATE MODEL SECONDARY RANKINGS
# ============================================================

model_rankings = []

model_secondary_probabilities = []


for player_index in range(
    len(df)
):

    primary_role = df.loc[
        player_index,
        "Primary Functional Role",
    ]

    role_probability_pairs = []

    for class_index, role in enumerate(
        class_names
    ):

        if role == primary_role:
            continue

        role_probability_pairs.append(
            (
                role,
                oof_probabilities[
                    player_index,
                    class_index,
                ],
            )
        )

    role_probability_pairs.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    model_rankings.append(
        [
            role
            for role, probability
            in role_probability_pairs
        ]
    )

    model_secondary_probabilities.append(
        [
            probability
            for role, probability
            in role_probability_pairs
        ]
    )


# ============================================================
# EVALUATION HELPERS
# ============================================================

def hit_at_k(
    ranking,
    official_alternatives,
    k,
):

    if not official_alternatives:
        return np.nan

    predicted = set(
        ranking[:k]
    )

    official = set(
        official_alternatives
    )

    return int(
        len(
            predicted
            & official
        )
        > 0
    )


def recall_at_k(
    ranking,
    official_alternatives,
    k,
):

    if not official_alternatives:
        return np.nan

    predicted = set(
        ranking[:k]
    )

    official = set(
        official_alternatives
    )

    return (
        len(
            predicted
            & official
        )
        / len(official)
    )


def reciprocal_rank(
    ranking,
    official_alternatives,
):

    if not official_alternatives:
        return np.nan

    official = set(
        official_alternatives
    )

    for rank, role in enumerate(
        ranking,
        start=1,
    ):

        if role in official:
            return 1 / rank

    return 0.0


# ============================================================
# PLAYER-LEVEL VALIDATION
# ============================================================

validation_rows = []


for i in range(
    len(df)
):

    official = df.loc[
        i,
        "Official Alternative Roles",
    ]

    model_ranking = model_rankings[
        i
    ]

    baseline_ranking = baseline_rankings[
        i
    ]

    secondary_probabilities = (
        model_secondary_probabilities[
            i
        ]
    )


    row = {
        "Name":
            df.loc[i, "Name"],

        "Position":
            df.loc[i, "Position"],

        "Primary Functional Role":
            df.loc[
                i,
                "Primary Functional Role",
            ],

        "Official Alternative Roles":
            "|".join(official),

        "Number Official Functional Alternatives":
            len(official),

        "Fold":
            oof_fold[i],
    }


    # --------------------------------------------------------
    # Save model ranking.
    # --------------------------------------------------------

    for rank in range(
        min(
            7,
            len(model_ranking),
        )
    ):

        row[
            f"Model Secondary Role {rank + 1}"
        ] = model_ranking[
            rank
        ]

        row[
            f"Model Secondary Probability {rank + 1}"
        ] = secondary_probabilities[
            rank
        ]


    # --------------------------------------------------------
    # Save baseline ranking.
    # --------------------------------------------------------

    for rank in range(
        min(
            7,
            len(baseline_ranking),
        )
    ):

        row[
            f"Baseline Secondary Role {rank + 1}"
        ] = baseline_ranking[
            rank
        ]


    # --------------------------------------------------------
    # Evaluation metrics.
    # --------------------------------------------------------

    for k in [
        1,
        2,
        3,
    ]:

        row[
            f"Model Hit@{k}"
        ] = hit_at_k(
            model_ranking,
            official,
            k,
        )

        row[
            f"Baseline Hit@{k}"
        ] = hit_at_k(
            baseline_ranking,
            official,
            k,
        )

        row[
            f"Model Recall@{k}"
        ] = recall_at_k(
            model_ranking,
            official,
            k,
        )

        row[
            f"Baseline Recall@{k}"
        ] = recall_at_k(
            baseline_ranking,
            official,
            k,
        )


    row[
        "Model Reciprocal Rank"
    ] = reciprocal_rank(
        model_ranking,
        official,
    )

    row[
        "Baseline Reciprocal Rank"
    ] = reciprocal_rank(
        baseline_ranking,
        official,
    )


    validation_rows.append(
        row
    )


validation_df = pd.DataFrame(
    validation_rows
)


# ============================================================
# EVALUATION SUBSET
# ============================================================
#
# Only players with at least one DIFFERENT functional
# alternative can evaluate functional-role flexibility.
#
# ============================================================

evaluation_df = validation_df[
    validation_df[
        "Number Official Functional Alternatives"
    ]
    > 0
].copy()


print(
    "\nPlayers with at least one "
    "functional alternative:"
)

print(
    len(evaluation_df)
)


# ============================================================
# OVERALL METRIC SUMMARY
# ============================================================

summary_rows = []


for method in [
    "Model",
    "Baseline",
]:

    row = {
        "Method": method,
        "Players Evaluated":
            len(evaluation_df),
    }

    for k in [
        1,
        2,
        3,
    ]:

        row[
            f"Hit@{k}"
        ] = (
            evaluation_df[
                f"{method} Hit@{k}"
            ]
            .mean()
        )

        row[
            f"Recall@{k}"
        ] = (
            evaluation_df[
                f"{method} Recall@{k}"
            ]
            .mean()
        )

    row[
        "MRR"
    ] = (
        evaluation_df[
            f"{method} Reciprocal Rank"
        ]
        .mean()
    )

    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)


print("\n" + "=" * 70)
print("SECONDARY ROLE VALIDATION")
print("=" * 70)

print(
    summary_df
    .to_string(index=False)
)


# ============================================================
# IMPROVEMENT OVER BASELINE
# ============================================================

model_summary = (
    summary_df[
        summary_df[
            "Method"
        ]
        == "Model"
    ]
    .iloc[0]
)

baseline_summary = (
    summary_df[
        summary_df[
            "Method"
        ]
        == "Baseline"
    ]
    .iloc[0]
)


improvement = {
    "Players Evaluated":
        len(evaluation_df)
}


for metric in [
    "Hit@1",
    "Hit@2",
    "Hit@3",
    "Recall@1",
    "Recall@2",
    "Recall@3",
    "MRR",
]:

    improvement[
        f"{metric} Model"
    ] = model_summary[
        metric
    ]

    improvement[
        f"{metric} Baseline"
    ] = baseline_summary[
        metric
    ]

    improvement[
        f"{metric} Difference"
    ] = (
        model_summary[
            metric
        ]
        - baseline_summary[
            metric
        ]
    )


improvement_df = pd.DataFrame(
    [improvement]
)


# ============================================================
# PERFORMANCE BY PRIMARY ROLE
# ============================================================

role_rows = []


for primary_role, group in (
    evaluation_df
    .groupby(
        "Primary Functional Role"
    )
):

    role_row = {
        "Primary Functional Role":
            primary_role,

        "Players":
            len(group),
    }

    for method in [
        "Model",
        "Baseline",
    ]:

        for k in [
            1,
            2,
            3,
        ]:

            role_row[
                f"{method} Hit@{k}"
            ] = (
                group[
                    f"{method} Hit@{k}"
                ]
                .mean()
            )

        role_row[
            f"{method} MRR"
        ] = (
            group[
                f"{method} Reciprocal Rank"
            ]
            .mean()
        )

    role_rows.append(
        role_row
    )


role_summary_df = pd.DataFrame(
    role_rows
)


print("\nPerformance by primary functional role:")

print(
    role_summary_df
    .to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

validation_df.to_csv(
    RESULTS_DIR
    / "player_secondary_role_validation.csv",
    index=False,
)

evaluation_df.to_csv(
    RESULTS_DIR
    / "players_with_functional_alternatives.csv",
    index=False,
)

summary_df.to_csv(
    RESULTS_DIR
    / "validation_summary.csv",
    index=False,
)

improvement_df.to_csv(
    RESULTS_DIR
    / "model_vs_baseline_improvement.csv",
    index=False,
)

role_summary_df.to_csv(
    RESULTS_DIR
    / "validation_by_primary_role.csv",
    index=False,
)


print("\nResults saved to:")
print(RESULTS_DIR)

print(
    "\nStage 3.6B secondary-role "
    "validation complete!"
)