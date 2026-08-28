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
    / "cleaned_eafc26_outfield_players.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage_3_5"
    / "A_logistic_6features"
)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

FEATURES = [
    "PAC",
    "SHO",
    "PAS",
    "DRI",
    "DEF",
    "PHY",
]

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
# CREATE FUNCTIONAL POSITION LABELS
# ============================================================

df["Functional Position"] = df["Position"].map(FUNCTIONAL_POSITION_MAP)

missing_positions = df.loc[
    df["Functional Position"].isna(),
    "Position"
].dropna().unique()

if len(missing_positions) > 0:
    print("\nUnmapped positions found:")
    print(missing_positions)

df = df.dropna(
    subset=FEATURES + ["Functional Position"]
).copy()


print("\nFunctional position counts:")
print(df["Functional Position"].value_counts())


# ============================================================
# SAVE POSITION COUNTS
# ============================================================

position_counts = (
    df["Functional Position"]
    .value_counts()
    .rename_axis("Functional Position")
    .reset_index(name="Count")
)

position_counts.to_csv(
    RESULTS_DIR / "functional_position_counts.csv",
    index=False,
)


# ============================================================
# FEATURES / TARGET
# ============================================================

X = df[FEATURES].copy()
y_text = df["Functional Position"].copy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)

class_names = label_encoder.classes_

print("\nClasses:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")


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
# MODEL
# ============================================================

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
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


# ============================================================
# TRAIN
# ============================================================

model.fit(X_train, y_train)


# ============================================================
# TEST SET PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)


# ============================================================
# TEST SET METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

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
print("STAGE 3.5A RESULTS")
print("==============================")

print(f"Top-1 Accuracy: {accuracy:.4f}")
print(f"Macro F1:       {macro_f1:.4f}")
print(f"Weighted F1:    {weighted_f1:.4f}")
print(f"Top-2 Accuracy: {top2_accuracy:.4f}")
print(f"Top-3 Accuracy: {top3_accuracy:.4f}")


# ============================================================
# SAVE MAIN METRICS
# ============================================================

metrics_df = pd.DataFrame(
    [
        {
            "Experiment": "3.5A",
            "Model": "Logistic Regression",
            "Features": 6,
            "Position Labels": 8,
            "Top-1 Accuracy": accuracy,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "Top-2 Accuracy": top2_accuracy,
            "Top-3 Accuracy": top3_accuracy,
        }
    ]
)

metrics_df.to_csv(
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

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
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

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names,
)

cm_df.to_csv(
    RESULTS_DIR / "confusion_matrix.csv"
)


# ============================================================
# 5-FOLD STRATIFIED CROSS-VALIDATION
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


print("\n5-Fold CV:")
print(cv_summary.to_string(index=False))


# ============================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================

final_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
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

final_model.fit(X, y)


# ============================================================
# FULL-DATA POSITION PROBABILITIES
# ============================================================

all_proba = final_model.predict_proba(X)

top3_indices = np.argsort(all_proba, axis=1)[:, -3:][:, ::-1]

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

    predictions[f"Predicted Functional Position {rank + 1}"] = (
        label_encoder.inverse_transform(idx)
    )

    predictions[f"Probability {rank + 1}"] = (
        all_proba[
            np.arange(len(df)),
            idx,
        ]
    )


predictions.to_csv(
    RESULTS_DIR / "player_functional_position_predictions.csv",
    index=False,
)


print(
    f"\nSaved Stage 3.5A results to:\n{RESULTS_DIR}"
)