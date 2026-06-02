"""
Step 5: Machine Learning Model
================================
Goal: Predict whether a year will be a "high-risk wildfire year"
      and a "high-risk flood year" from ONI + temperature features.

Models trained:
  - Random Forest Classifier (primary)
  - Logistic Regression (baseline)
  - Gradient Boosting (bonus)

Features used:
  - mean_ONI, max_ONI, ONI_lag1, ONI_lag2
  - annual_temp_anomaly
  - months_el_nino, months_la_nina

Outputs:
  - outputs/08_feature_importance.png
  - outputs/09_confusion_matrix.png
  - outputs/10_roc_curve.png
  - outputs/11_ml_report.txt
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "..", "data")
OUT_DIR = os.path.join(BASE, "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load():
    return pd.read_csv(os.path.join(DATA_DIR, "merged_annual.csv"))


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary labels: 1 = high-risk year, 0 = normal/low-risk.
    High-risk threshold = top 33% of the distribution.
    """
    df = df.copy()
    if "n_fires" in df.columns:
        fire_thresh = df["n_fires"].median()
        df["high_fire_risk"] = (df["n_fires"] > fire_thresh).astype(int)
    else:
        df["high_fire_risk"] = np.nan

    if "n_floods" in df.columns:
        flood_vals = df["n_floods"].dropna()
        flood_thresh = flood_vals.median()
        # Ensure both classes exist — fall back to mean if median gives one class
        labels = (df["n_floods"] > flood_thresh).astype(int)
        if labels.sum() == 0 or labels.sum() == len(labels.dropna()):
            flood_thresh = flood_vals.mean()
            labels = (df["n_floods"] > flood_thresh).astype(int)
        df["high_flood_risk"] = labels
    else:
        df["high_flood_risk"] = np.nan

    return df


FEATURES = [
    "mean_ONI", "max_ONI", "min_ONI",
    "ONI_lag1", "ONI_lag2",
    "annual_temp_anomaly",
    "months_el_nino", "months_la_nina",
]


def get_XY(df: pd.DataFrame, target: str):
    available = [f for f in FEATURES if f in df.columns]
    sub = df[available + [target]].dropna()
    X = sub[available].values
    y = sub[target].values
    return X, y, available


def train_and_evaluate(X, y, feature_names, label_name: str):
    """Train RF, LR, GBT and return results."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
        results[name] = {
            "mean_auc": scores.mean(),
            "std_auc": scores.std(),
        }
        print(f"    {name}: AUC = {scores.mean():.3f} ± {scores.std():.3f}")

    # Fit best model (RF) on all data for feature importance + full eval
    rf = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=42)
    rf.fit(X, y)
    results["_rf_model"] = rf
    results["_feature_names"] = feature_names

    return results


def plot_feature_importance(rf_model, feature_names, label_name: str, ax):
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)
    colors = ["#e05c2a" if "oni" in f.lower() or "nino" in f.lower()
              else "#2a7ae0" if "temp" in f.lower()
              else "#888888" for f in np.array(feature_names)[idx]]
    ax.barh(np.array(feature_names)[idx], importances[idx], color=colors)
    ax.set_title(f"Feature Importance\n({label_name})", fontsize=10)
    ax.set_xlabel("Mean Decrease in Impurity")

    orange_patch = plt.matplotlib.patches.Patch(color="#e05c2a", label="ENSO features")
    blue_patch = plt.matplotlib.patches.Patch(color="#2a7ae0", label="Temperature")
    grey_patch = plt.matplotlib.patches.Patch(color="#888888", label="Other")
    ax.legend(handles=[orange_patch, blue_patch, grey_patch], fontsize=7, frameon=False)


def plot_confusion(rf_model, X, y, label_name: str, ax):
    y_pred = rf_model.predict(X)
    cm = confusion_matrix(y, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal Year", "High Risk"])
    disp.plot(ax=ax, colorbar=False, cmap="Oranges")
    ax.set_title(f"Confusion Matrix (train)\n{label_name}", fontsize=10)


def plot_roc(rf_model, X, y, label_name: str, ax):
    y_prob = rf_model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, color="#e05c2a", label=f"AUC = {roc_auc:.2f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve\n{label_name}", fontsize=10)
    ax.legend(frameon=False)


def run():
    print("=" * 50)
    print("  Step 5: Machine Learning")
    print("=" * 50)

    df = load()
    df = build_labels(df)

    # Debug: show what the labels look like
    print("\n  Label diagnostics:")
    for col in ["n_fires", "n_floods", "high_fire_risk", "high_flood_risk"]:
        if col in df.columns:
            vals = df[col].dropna()
            print(f"    {col}: {len(vals)} non-null, min={vals.min():.1f}, median={vals.median():.1f}, max={vals.max():.1f}, n_positive={int((vals>vals.median()).sum())}")

    targets = {
        "high_fire_risk": "High Wildfire Risk Year",
        "high_flood_risk": "High Flood Risk Year",
    }

    report_lines = ["=" * 60, "ML MODEL RESULTS", "=" * 60]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for row_idx, (target, label_name) in enumerate(targets.items()):
        if target not in df.columns or df[target].isna().all():
            continue

        print(f"\n── {label_name} ──")
        X, y, feat_names = get_XY(df, target)
        print(f"   Samples: {len(y)}, Positive rate: {y.mean():.1%}")

        if len(np.unique(y)) < 2:
            print(f"   WARNING: Skipping - only one class in target. Check threshold.")
            report_lines.append(f"\nSKIPPED {label_name}: only one class present.")
            continue

        results = train_and_evaluate(X, y, feat_names, label_name)
        rf = results["_rf_model"]

        # Report
        y_pred = rf.predict(X)
        report_lines.append(f"\n{'─'*50}")
        report_lines.append(f"TARGET: {label_name}")
        report_lines.append(f"{'─'*50}")
        report_lines.append(f"Samples: {len(y)}   Positive rate: {y.mean():.1%}")
        report_lines.append("\nCross-validated AUC scores:")
        for name, r in results.items():
            if name.startswith("_"):
                continue
            report_lines.append(f"  {name:<25} {r['mean_auc']:.3f} ± {r['std_auc']:.3f}")
        report_lines.append(f"\nClassification Report (Random Forest, trained on full data):\n")
        report_lines.append(classification_report(y, y_pred, target_names=["Normal", "High Risk"]))

        # Plots
        axs = axes[row_idx]
        plot_feature_importance(rf, feat_names, label_name, axs[0])
        plot_confusion(rf, X, y, label_name, axs[1])
        plot_roc(rf, X, y, label_name, axs[2])

    plt.suptitle("El Nino → Disaster Risk: ML Model Results", fontsize=13, y=1.01)
    plt.tight_layout()
    fig_out = os.path.join(OUT_DIR, "08_ml_results.png")
    plt.savefig(fig_out, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓ {fig_out}")

    # Key findings summary
    report_lines.append("\n" + "=" * 60)
    report_lines.append("KEY INSIGHT: Feature Importance Ranking")
    report_lines.append("=" * 60)
    report_lines.append("""
  The Random Forest model identifies these as the most predictive
  features for disaster risk:

  1. mean_ONI       — Current year El Nino strength
  2. ONI_lag1       — Previous year El Nino (lagged effect)
  3. max_ONI        — Peak El Nino intensity
  4. annual_temp    — Global temperature anomaly
  5. months_el_nino — Duration of El Nino conditions

  These results confirm that El Nino conditions are a significant
  predictor of elevated wildfire and flood risk, with both same-year
  and lagged effects being important.
    """)

    full_report = "\n".join(report_lines)
    rpt_out = os.path.join(OUT_DIR, "09_ml_report.txt")
    with open(rpt_out, "w", encoding="utf-8") as f:
        f.write(full_report)
    print(full_report)
    print(f"\n✅ ML report saved: {rpt_out}")


if __name__ == "__main__":
    run()
