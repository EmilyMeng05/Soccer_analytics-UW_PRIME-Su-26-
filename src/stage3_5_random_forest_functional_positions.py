"""Stage 3.5: Random Forest prediction of eight functional roles."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# STAGE 3.5B: RANDOM FOREST FUNCTIONAL-ROLE PREDICTION
#
# This script performs the same two experiments as the Logistic Regression
# script, but uses Random Forest. Random Forest can learn nonlinear patterns
# and interactions between attributes that Logistic Regression may miss.
#
# Keeping the split and feature sets consistent makes the two model types
# directly comparable.


# Define the input and output paths.

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "eafc26_functional_positions.csv"
RESULTS_DIR = PROJECT_ROOT / "results" / "stage_3_5" / "random_forest"

# Experiment 1 uses the original six broad EAFC attributes.

BROAD_FEATURES = ["PAC", "SHO", "PAS", "DRI", "DEF", "PHY"]


# Experiment 2 uses all 34 broad and detailed outfield attributes.

DETAILED_FEATURES = [
    "Acceleration", "Sprint Speed", "Positioning", "Finishing", "Shot Power",
    "Long Shots", "Volleys", "Penalties", "Vision", "Crossing",
    "Free Kick Accuracy", "Short Passing", "Long Passing", "Curve", "Agility",
    "Balance", "Reactions", "Ball Control", "Dribbling", "Composure",
    "Interceptions", "Heading Accuracy", "Standing Tackle", "Sliding Tackle",
    "Jumping", "Stamina", "Strength", "Aggression",
]
RICH_FEATURES = BROAD_FEATURES + DETAILED_FEATURES


def top_k_accuracy(y_true, probabilities, classes, k):
    """Return the fraction of true roles appearing among the top k choices."""

    # Top-2 and top-3 accuracy are especially useful for identifying players
    # whose profiles make them suitable for multiple lineup roles.
    top_indices = np.argsort(probabilities, axis=1)[:, -k:]
    top_labels = classes[top_indices]
    return np.mean([truth in predictions for truth, predictions in zip(y_true, top_labels)])


def build_model():
    """Create the same Random Forest configuration for every experiment."""

    # 500 trees give stable results without making the experiment excessively
    # complicated. min_samples_leaf=2 provides mild regularization, while
    # n_jobs=-1 lets the computer use all available processor cores.
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )


def save_confusion_matrix(y_true, y_pred, labels, path, title):
    """Create a plot showing which functional roles are confused."""

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Greens")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted role")
    ax.set_ylabel("Actual role")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_experiment(players, feature_name, features):
    """Train and evaluate Random Forest for one feature set."""

    # Select the model inputs and the eight-role prediction target.
    data = players.dropna(subset=features + ["Functional Position"]).copy()
    X = data[features]
    y = data["Functional Position"]
    # Use the same stratified 80/20 split and random seed as Logistic
    # Regression so differences are caused by the models, not different data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = build_model()
    model.fit(X_train, y_train)
    # Generate both the single predicted role and the probability distribution
    # across all functional roles.
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    classes = model.classes_

    # Compare strict top-1 accuracy with the more flexible top-2 and top-3
    # measurements that will matter during lineup construction.
    scores = {
        "feature_set": feature_name,
        "number_of_features": len(features),
        "top_1_accuracy": accuracy_score(y_test, predictions),
        "top_2_accuracy": top_k_accuracy(y_test.to_numpy(), probabilities, classes, 2),
        "top_3_accuracy": top_k_accuracy(y_test.to_numpy(), probabilities, classes, 3),
    }
    # Save per-role precision, recall, and F1-score in addition to accuracy.
    report = classification_report(y_test, predictions, digits=4, zero_division=0)
    (RESULTS_DIR / f"{feature_name}_classification_report.txt").write_text(report, encoding="utf-8")
    save_confusion_matrix(
        y_test,
        predictions,
        classes,
        RESULTS_DIR / f"{feature_name}_confusion_matrix.png",
        f"Random Forest: {feature_name.replace('_', ' ').title()}",
    )

    # Random Forest supplies feature-importance values. These show which
    # attributes contributed most strongly to separating functional roles.
    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(RESULTS_DIR / f"{feature_name}_feature_importance.csv", index=False)

    print(f"\n{'=' * 60}\nRANDOM FOREST: {feature_name.upper()}\n{'=' * 60}")
    print(f"Top-1 accuracy: {scores['top_1_accuracy']:.4f}")
    print(f"Top-2 accuracy: {scores['top_2_accuracy']:.4f}")
    print(f"Top-3 accuracy: {scores['top_3_accuracy']:.4f}\n")
    print(report)
    return scores


def save_all_player_probabilities(players):
    """Train on all players and save role probabilities for lineup analysis."""

    # Evaluation is completed using held-out data above. We then fit a final
    # rich-feature model to the full dataset to create exploratory predictions
    # for every player in the future lineup candidate pool.
    data = players.dropna(subset=RICH_FEATURES + ["Functional Position"]).copy()
    model = build_model()
    model.fit(data[RICH_FEATURES], data["Functional Position"])
    probabilities = model.predict_proba(data[RICH_FEATURES])
    classes = model.classes_

    # Keep identifying columns and add the probability for every role.
    identity_columns = [c for c in ["ID", "Name", "OVR", "Position", "Functional Position"] if c in data]
    output = data[identity_columns].reset_index(drop=True)
    for index, role in enumerate(classes):
        output[f"Probability: {role}"] = probabilities[:, index]

    # Save each player's three most likely roles for easier inspection while
    # preserving all eight probability columns in the same output file.
    ranked = np.argsort(probabilities, axis=1)[:, ::-1]
    for rank in range(3):
        output[f"Predicted Role {rank + 1}"] = classes[ranked[:, rank]]
        output[f"Role {rank + 1} Probability"] = probabilities[
            np.arange(len(output)), ranked[:, rank]
        ]

    output.to_csv(RESULTS_DIR / "all_player_role_probabilities.csv", index=False)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    players = pd.read_csv(DATA_PATH)
    # Run the six-feature and rich-feature experiments and save their metrics
    # side by side for a direct comparison.
    experiments = [
        run_experiment(players, "six_features", BROAD_FEATURES),
        run_experiment(players, "rich_features", RICH_FEATURES),
    ]
    pd.DataFrame(experiments).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    # Save full-dataset probabilities from the rich-feature Random Forest for
    # later lineup construction and positional-versatility analysis.
    save_all_player_probabilities(players)
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()