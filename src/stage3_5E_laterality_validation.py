from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    StandardScaler,
)


# ============================================================
# STAGE 3.5E
# LATERALITY VALIDATION
# ============================================================
#
# Research question:
#
# Does Preferred Foot help explain why models struggle to
# distinguish left- and right-sided soccer positions?
#
# Controlled comparison:
#
#   E1:
#   34 skill attributes
#       ->
#   11 exact positions
#
#   E2:
#   34 skill attributes + Preferred foot
#       ->
#   11 exact positions
#
# IMPORTANT:
#
# We intentionally use the SAME model for both experiments.
#
# That way, if performance changes, the difference can be
# attributed to Preferred foot rather than to switching models.
#
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "eafc26_outfield_full_features.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage_3_5"
    / "E_laterality_validation"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20


# ============================================================
# 34 FEATURES FROM STAGE 3.5C
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

SKILL_FEATURES = CORE_FEATURES + DETAILED_FEATURES


# ============================================================
# EXACT 11 POSITIONS
# ============================================================

EXACT_POSITIONS = [
    "Attacking Midfielder",
    "Center Back",
    "Central Midfielder",
    "Defensive Midfielder",
    "Left Back",
    "Left Midfielder",
    "Left Winger",
    "Right Back",
    "Right Midfielder",
    "Right Winger",
    "Striker",
]


# ============================================================
# LEFT / RIGHT POSITION PAIRS
# ============================================================

LATERAL_PAIRS = [
    ("Left Back", "Right Back"),
    ("Left Midfielder", "Right Midfielder"),
    ("Left Winger", "Right Winger"),
]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print("=" * 70)
print("STAGE 3.5E — LATERALITY VALIDATION")
print("=" * 70)

print(f"\nLoaded dataset: {df.shape}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    SKILL_FEATURES
    + [
        "Position",
        "Preferred foot",
    ]
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nThe following required columns are missing:\n"
        + "\n".join(missing_columns)
    )


print(
    f"\nSkill features available: {len(SKILL_FEATURES)}"
)

print(
    "\nPreferred foot values:"
)

print(
    df["Preferred foot"]
    .value_counts(dropna=False)
)


# ============================================================
# KEEP ONLY THE ORIGINAL 11 OUTfield POSITIONS
# ============================================================

df = df[
    df["Position"].isin(EXACT_POSITIONS)
].copy()


# ============================================================
# CONVERT NUMERIC FEATURES
# ============================================================

for feature in SKILL_FEATURES:

    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce",
    )


# ============================================================
# DROP MISSING VALUES
# ============================================================

df = df.dropna(
    subset=(
        SKILL_FEATURES
        + [
            "Position",
            "Preferred foot",
        ]
    )
).copy()


print(
    f"\nPlayers used in Stage 3.5E: {len(df)}"
)


print(
    "\nExact position counts:"
)

print(
    df["Position"]
    .value_counts()
)


# ============================================================
# ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df["Position"]
)

class_names = label_encoder.classes_

n_classes = len(class_names)


print(
    f"\nNumber of exact position classes: {n_classes}"
)

print("\nClass order:")

for i, position in enumerate(class_names):

    print(
        f"{i}: {position}"
    )


# ============================================================
# SAVE DATASET SUMMARY
# ============================================================

position_counts = (
    df["Position"]
    .value_counts()
    .rename_axis("Position")
    .reset_index(name="Count")
)

position_counts.to_csv(
    RESULTS_DIR
    / "exact_position_counts.csv",
    index=False,
)


foot_counts = (
    df["Preferred foot"]
    .value_counts()
    .rename_axis("Preferred foot")
    .reset_index(name="Count")
)

foot_counts.to_csv(
    RESULTS_DIR
    / "preferred_foot_counts.csv",
    index=False,
)


# ============================================================
# CREATE ONE SHARED TRAIN / TEST SPLIT
# ============================================================
#
# Critical:
#
# E1 and E2 must use exactly the same players in training and
# testing.
#
# Otherwise differences could simply come from a different split.
#
# ============================================================

train_indices, test_indices = train_test_split(
    np.arange(len(df)),
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y,
)


y_train = y[train_indices]
y_test = y[test_indices]


# ============================================================
# HELPER:
# CREATE LOGISTIC REGRESSION MODEL
# ============================================================

def create_logistic_pipeline(
    numeric_features,
    categorical_features=None,
):

    if categorical_features is None:
        categorical_features = []

    transformers = [
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        )
    ]

    if categorical_features:

        transformers.append(
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                model,
            ),
        ]
    )

    return pipeline


# ============================================================
# HELPER:
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    experiment_name,
    model,
    X,
):

    X_train = X.iloc[
        train_indices
    ].copy()

    X_test = X.iloc[
        test_indices
    ].copy()


    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    y_proba = model.predict_proba(
        X_test
    )


    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
    )

    top2_accuracy = top_k_accuracy_score(
        y_test,
        y_proba,
        k=2,
        labels=np.arange(n_classes),
    )

    top3_accuracy = top_k_accuracy_score(
        y_test,
        y_proba,
        k=3,
        labels=np.arange(n_classes),
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(experiment_name)
    print("=" * 70)

    print(
        f"Top-1 Accuracy: {accuracy:.4f}"
    )

    print(
        f"Macro F1:       {macro_f1:.4f}"
    )

    print(
        f"Weighted F1:    {weighted_f1:.4f}"
    )

    print(
        f"Top-2 Accuracy: {top2_accuracy:.4f}"
    )

    print(
        f"Top-3 Accuracy: {top3_accuracy:.4f}"
    )


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(n_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(n_classes),
    )

    cm_df = pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    safe_name = (
        experiment_name
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )

    report_df.to_csv(
        RESULTS_DIR
        / f"{safe_name}_classification_report.csv"
    )

    cm_df.to_csv(
        RESULTS_DIR
        / f"{safe_name}_confusion_matrix.csv"
    )


    # --------------------------------------------------------
    # RETURN EVERYTHING NEEDED LATER
    # --------------------------------------------------------

    metrics = {
        "Experiment": experiment_name,
        "Top-1 Accuracy": accuracy,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Top-2 Accuracy": top2_accuracy,
        "Top-3 Accuracy": top3_accuracy,
    }

    return {
        "model": model,
        "metrics": metrics,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": cm,
        "confusion_matrix_df": cm_df,
        "classification_report": report_df,
    }


# ============================================================
# EXPERIMENT E1
#
# 34 SKILL FEATURES ONLY
# ============================================================

X_skills = df[
    SKILL_FEATURES
].copy()


model_skills = create_logistic_pipeline(
    numeric_features=SKILL_FEATURES,
)


results_skills = evaluate_model(
    experiment_name="E1 34 Skills",
    model=model_skills,
    X=X_skills,
)


# ============================================================
# EXPERIMENT E2
#
# 34 SKILL FEATURES + PREFERRED FOOT
# ============================================================

FEATURES_WITH_FOOT = (
    SKILL_FEATURES
    + ["Preferred foot"]
)

X_skills_foot = df[
    FEATURES_WITH_FOOT
].copy()


model_skills_foot = create_logistic_pipeline(
    numeric_features=SKILL_FEATURES,
    categorical_features=[
        "Preferred foot"
    ],
)


results_foot = evaluate_model(
    experiment_name="E2 34 Skills + Preferred Foot",
    model=model_skills_foot,
    X=X_skills_foot,
)


# ============================================================
# OVERALL MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    [
        results_skills["metrics"],
        results_foot["metrics"],
    ]
)


comparison_df.to_csv(
    RESULTS_DIR
    / "overall_model_comparison.csv",
    index=False,
)


print("\n")
print("=" * 70)
print("OVERALL COMPARISON")
print("=" * 70)

print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# LATERAL CONFUSION ANALYSIS
# ============================================================
#
# We specifically care about:
#
#   LB <-> RB
#   LM <-> RM
#   LW <-> RW
#
# Example:
#
# True Left Back predicted Right Back
# True Right Back predicted Left Back
#
# These counts are summed into one pairwise confusion measure.
#
# ============================================================

def get_pair_confusion(
    confusion_matrix_array,
    position_a,
    position_b,
):

    a_idx = np.where(
        class_names == position_a
    )[0][0]

    b_idx = np.where(
        class_names == position_b
    )[0][0]


    a_to_b = (
        confusion_matrix_array[
            a_idx,
            b_idx,
        ]
    )

    b_to_a = (
        confusion_matrix_array[
            b_idx,
            a_idx,
        ]
    )

    total_confusion = (
        a_to_b
        + b_to_a
    )


    return (
        a_to_b,
        b_to_a,
        total_confusion,
    )


laterality_results = []


for left_position, right_position in LATERAL_PAIRS:

    (
        left_to_right_skills,
        right_to_left_skills,
        total_skills,
    ) = get_pair_confusion(
        results_skills[
            "confusion_matrix"
        ],
        left_position,
        right_position,
    )


    (
        left_to_right_foot,
        right_to_left_foot,
        total_foot,
    ) = get_pair_confusion(
        results_foot[
            "confusion_matrix"
        ],
        left_position,
        right_position,
    )


    if total_skills > 0:

        reduction = (
            (
                total_skills
                - total_foot
            )
            / total_skills
        )

    else:

        reduction = np.nan


    laterality_results.append(
        {
            "Pair": (
                f"{left_position} <-> "
                f"{right_position}"
            ),

            "Skills Only Left->Right":
                left_to_right_skills,

            "Skills Only Right->Left":
                right_to_left_skills,

            "Skills Only Total Confusion":
                total_skills,

            "With Foot Left->Right":
                left_to_right_foot,

            "With Foot Right->Left":
                right_to_left_foot,

            "With Foot Total Confusion":
                total_foot,

            "Confusion Reduction":
                reduction,
        }
    )


laterality_df = pd.DataFrame(
    laterality_results
)


laterality_df.to_csv(
    RESULTS_DIR
    / "left_right_confusion_comparison.csv",
    index=False,
)


print("\n")
print("=" * 70)
print("LEFT / RIGHT CONFUSION ANALYSIS")
print("=" * 70)

print(
    laterality_df.to_string(
        index=False
    )
)


# ============================================================
# AGGREGATE LATERAL CONFUSION
# ============================================================

total_lateral_skills = (
    laterality_df[
        "Skills Only Total Confusion"
    ].sum()
)

total_lateral_foot = (
    laterality_df[
        "With Foot Total Confusion"
    ].sum()
)


if total_lateral_skills > 0:

    overall_reduction = (
        (
            total_lateral_skills
            - total_lateral_foot
        )
        / total_lateral_skills
    )

else:

    overall_reduction = np.nan


aggregate_lateral_df = pd.DataFrame(
    [
        {
            "Skills Only Lateral Confusions":
                total_lateral_skills,

            "With Preferred Foot Lateral Confusions":
                total_lateral_foot,

            "Overall Lateral Confusion Reduction":
                overall_reduction,
        }
    ]
)


aggregate_lateral_df.to_csv(
    RESULTS_DIR
    / "aggregate_lateral_confusion.csv",
    index=False,
)


print("\n")
print("=" * 70)
print("AGGREGATE LATERAL CONFUSION")
print("=" * 70)

print(
    f"Skills-only left/right confusions: "
    f"{total_lateral_skills}"
)

print(
    f"With Preferred Foot:              "
    f"{total_lateral_foot}"
)

print(
    f"Relative reduction:               "
    f"{overall_reduction:.2%}"
)


# ============================================================
# POSITION-SPECIFIC F1 COMPARISON
# ============================================================
#
# This tells us whether LB/RB/LM/RM/LW/RW themselves improve,
# rather than only checking total model accuracy.
#
# ============================================================

side_positions = [
    "Left Back",
    "Right Back",
    "Left Midfielder",
    "Right Midfielder",
    "Left Winger",
    "Right Winger",
]


side_position_results = []


for position in side_positions:

    skills_report = (
        results_skills[
            "classification_report"
        ]
    )

    foot_report = (
        results_foot[
            "classification_report"
        ]
    )


    skills_f1 = (
        skills_report.loc[
            position,
            "f1-score",
        ]
    )

    foot_f1 = (
        foot_report.loc[
            position,
            "f1-score",
        ]
    )


    side_position_results.append(
        {
            "Position":
                position,

            "Skills Only F1":
                skills_f1,

            "With Preferred Foot F1":
                foot_f1,

            "F1 Change":
                foot_f1
                - skills_f1,
        }
    )


side_f1_df = pd.DataFrame(
    side_position_results
)


side_f1_df.to_csv(
    RESULTS_DIR
    / "side_position_f1_comparison.csv",
    index=False,
)


print("\n")
print("=" * 70)
print("SIDE POSITION F1 COMPARISON")
print("=" * 70)

print(
    side_f1_df.to_string(
        index=False
    )
)


# ============================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================
#
# We also test whether the overall difference persists across
# multiple splits rather than depending on random_state=42.
#
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)


scoring = {
    "accuracy": "accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
}


print("\n")
print("=" * 70)
print("RUNNING 5-FOLD CROSS-VALIDATION")
print("=" * 70)


# ------------------------------------------------------------
# E1 CV
# ------------------------------------------------------------

cv_skills_model = create_logistic_pipeline(
    numeric_features=SKILL_FEATURES,
)


cv_skills = cross_validate(
    cv_skills_model,
    X_skills,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
)


# ------------------------------------------------------------
# E2 CV
# ------------------------------------------------------------

cv_foot_model = create_logistic_pipeline(
    numeric_features=SKILL_FEATURES,
    categorical_features=[
        "Preferred foot"
    ],
)


cv_foot = cross_validate(
    cv_foot_model,
    X_skills_foot,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
)


# ============================================================
# CV SUMMARY
# ============================================================

cv_summary = pd.DataFrame(
    [
        {
            "Experiment":
                "E1 34 Skills",

            "Accuracy Mean":
                cv_skills[
                    "test_accuracy"
                ].mean(),

            "Accuracy Std":
                cv_skills[
                    "test_accuracy"
                ].std(),

            "Macro F1 Mean":
                cv_skills[
                    "test_macro_f1"
                ].mean(),

            "Macro F1 Std":
                cv_skills[
                    "test_macro_f1"
                ].std(),

            "Weighted F1 Mean":
                cv_skills[
                    "test_weighted_f1"
                ].mean(),

            "Weighted F1 Std":
                cv_skills[
                    "test_weighted_f1"
                ].std(),
        },

        {
            "Experiment":
                "E2 34 Skills + Preferred Foot",

            "Accuracy Mean":
                cv_foot[
                    "test_accuracy"
                ].mean(),

            "Accuracy Std":
                cv_foot[
                    "test_accuracy"
                ].std(),

            "Macro F1 Mean":
                cv_foot[
                    "test_macro_f1"
                ].mean(),

            "Macro F1 Std":
                cv_foot[
                    "test_macro_f1"
                ].std(),

            "Weighted F1 Mean":
                cv_foot[
                    "test_weighted_f1"
                ].mean(),

            "Weighted F1 Std":
                cv_foot[
                    "test_weighted_f1"
                ].std(),
        },
    ]
)


cv_summary.to_csv(
    RESULTS_DIR
    / "cross_validation_summary.csv",
    index=False,
)


print("\n")
print("=" * 70)
print("5-FOLD CV RESULTS")
print("=" * 70)

print(
    cv_summary.to_string(
        index=False
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

accuracy_change = (
    results_foot[
        "metrics"
    ]["Top-1 Accuracy"]
    -
    results_skills[
        "metrics"
    ]["Top-1 Accuracy"]
)


macro_f1_change = (
    results_foot[
        "metrics"
    ]["Macro F1"]
    -
    results_skills[
        "metrics"
    ]["Macro F1"]
)


summary = pd.DataFrame(
    [
        {
            "Accuracy Change":
                accuracy_change,

            "Macro F1 Change":
                macro_f1_change,

            "Lateral Confusion Before":
                total_lateral_skills,

            "Lateral Confusion After":
                total_lateral_foot,

            "Lateral Confusion Reduction":
                overall_reduction,
        }
    ]
)


summary.to_csv(
    RESULTS_DIR
    / "stage3_5E_summary.csv",
    index=False,
)


print("\n")
print("=" * 70)
print("STAGE 3.5E SUMMARY")
print("=" * 70)

print(
    f"Accuracy change after Preferred Foot: "
    f"{accuracy_change:+.4f}"
)

print(
    f"Macro F1 change after Preferred Foot: "
    f"{macro_f1_change:+.4f}"
)

print(
    f"Left/right confusion reduction: "
    f"{overall_reduction:.2%}"
)


print("\nResults saved to:")

print(
    RESULTS_DIR
)