from pathlib import Path

import numpy as np
import pandas as pd

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
from sklearn.preprocessing import LabelEncoder, StandardScaler


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
    / "C_logistic_34features"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURES
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

FEATURES = CORE_FEATURES + DETAILED_FEATURES

RANDOM_STATE = 42


FUNCTIONAL_POSITION_MAP = {
    "Left Winger": "Winger",
    "Right Winger": "Winger",

    "Left Midfielder": "Wide Midfielder",
    "Right Midfielder": "Wide Midfielder",

    "Left Back": "Fullback",
    "Right Back": "Fullback",

    "Attacking Midfielder": "Attacking Midfielder",
    "Central Midfielder": "Central Midfielder",
    "Defensive Midfielder": "Defensive Midfielder",
    "Center Back": "Center Back",
    "Striker": "Striker",
}


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

print(f"Loaded dataset: {df.shape}")


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        "Missing required features:\n"
        + "\n".join(missing_features)
    )

print(
    f"\nNumber of features used: {len(FEATURES)}"
)


# ============================================================
# CREATE FUNCTIONAL POSITIONS
# ============================================================

df["Functional Position"] = df["Position"].map(
    FUNCTIONAL_POSITION_MAP
)

df = df.dropna(
    subset=FEATURES + ["Functional Position"]
).copy()


# Convert explicitly to numeric
for feature in FEATURES:
    df[feature] = pd.to_numeric(
        df[feature],
        errors="coerce",
    )

df = df.dropna(
    subset=FEATURES
).copy()


print(f"Players remaining: {len(df)}")

print("\nFunctional position counts:")
print(df["Functional Position"].value_counts())


# ============================================================
# FEATURES / TARGET
# ============================================================

X = df[FEATURES].copy()

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(
    df["Functional Position"]
)

class_names = label_encoder.classes_


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)


# ============================================================
# LOGISTIC MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)


model.fit(
    X_train,
    y_train,
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_proba = model.predict_proba(X_test)


# ============================================================
# METRICS
# ============================================================

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
    labels=np.arange(len(class_names)),
)

top3_accuracy = top_k_accuracy_score(
    y_test,
    y_proba,
    k=3,
    labels=np.arange(len(class_names)),
)


print("\n==============================")
print("STAGE 3.5C RESULTS")
print("==============================")

print(f"Top-1 Accuracy: {accuracy:.4f}")
print(f"Macro F1:       {macro_f1:.4f}")
print(f"Weighted F1:    {weighted_f1:.4f}")
print(f"Top-2 Accuracy: {top2_accuracy:.4f}")
print(f"Top-3 Accuracy: {top3_accuracy:.4f}")


# ============================================================
# SAVE EVALUATION
# ============================================================

pd.DataFrame(
    [
        {
            "Experiment": "3.5C",
            "Model": "Logistic Regression",
            "Features": len(FEATURES),
            "Position Labels": 8,
            "Top-1 Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "Top-2 Accuracy": top2_accuracy,
            "Top-3 Accuracy": top3_accuracy,
        }
    ]
).to_csv(
    RESULTS_DIR / "model_evaluation.csv",
    index=False,
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

pd.DataFrame(
    report
).transpose().to_csv(
    RESULTS_DIR / "classification_report.csv"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=np.arange(len(class_names)),
)

pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names,
).to_csv(
    RESULTS_DIR / "confusion_matrix.csv"
)


# ============================================================
# CROSS-VALIDATION
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

cv_results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring,
    n_jobs=-1,
)

cv_summary = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Macro F1",
            "Weighted F1",
        ],
        "Mean": [
            cv_results["test_accuracy"].mean(),
            cv_results["test_macro_f1"].mean(),
            cv_results["test_weighted_f1"].mean(),
        ],
        "Std": [
            cv_results["test_accuracy"].std(),
            cv_results["test_macro_f1"].std(),
            cv_results["test_weighted_f1"].std(),
        ],
    }
)

cv_summary.to_csv(
    RESULTS_DIR / "cross_validation_summary.csv",
    index=False,
)


# ============================================================
# FINAL MODEL
# ============================================================

final_model = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler(),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

final_model.fit(
    X,
    y,
)


# ============================================================
# FULL PLAYER PREDICTIONS
# ============================================================

all_proba = final_model.predict_proba(X)

top3_indices = np.argsort(
    all_proba,
    axis=1,
)[:, -3:][:, ::-1]

predictions = df[
    [
        "ID",
        "Name",
        "OVR",
        "Position",
        "Functional Position",
    ]
].copy()

for rank in range(3):

    idx = top3_indices[:, rank]

    predictions[
        f"Predicted Functional Position {rank + 1}"
    ] = label_encoder.inverse_transform(idx)

    predictions[
        f"Probability {rank + 1}"
    ] = all_proba[
        np.arange(len(df)),
        idx,
    ]


predictions.to_csv(
    RESULTS_DIR / "player_functional_position_predictions.csv",
    index=False,
)


# ============================================================
# SAVE FEATURE LIST
# ============================================================

pd.DataFrame(
    {
        "Feature": FEATURES
    }
).to_csv(
    RESULTS_DIR / "features_used.csv",
    index=False,
)


print(
    f"\nSaved Stage 3.5C results to:\n{RESULTS_DIR}"
)