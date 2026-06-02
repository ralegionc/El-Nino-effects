"""
Step 3: Exploratory Data Analysis (EDA)
=========================================
Produces:
  - outputs/01_oni_timeseries.png
  - outputs/02_disasters_timeseries.png
  - outputs/03_phase_boxplots.png
  - outputs/04_correlation_heatmap.png
  - outputs/05_scatter_oni_vs_fires.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "..", "data")
OUT_DIR = os.path.join(BASE, "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "monospace",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "El Nino": "#e05c2a",
    "La Nina": "#2a7ae0",
    "Neutral": "#888888",
}


def load():
    monthly = pd.read_csv(os.path.join(DATA_DIR, "merged_monthly.csv"))
    annual = pd.read_csv(os.path.join(DATA_DIR, "merged_annual.csv"))
    return monthly, annual


# ─────────────────────────────────────────────
# Plot 1: ONI Time Series with El Nino shading
# ─────────────────────────────────────────────
def plot_oni_timeseries(monthly: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 4))
    monthly["date_num"] = monthly["year"] + (monthly["month"] - 1) / 12
    ax.plot(monthly["date_num"], monthly["ONI"], color="#333", lw=0.8, alpha=0.7)
    ax.fill_between(monthly["date_num"], monthly["ONI"], 0,
                    where=monthly["ONI"] > 0.5, color="#e05c2a", alpha=0.4, label="El Nino")
    ax.fill_between(monthly["date_num"], monthly["ONI"], 0,
                    where=monthly["ONI"] < -0.5, color="#2a7ae0", alpha=0.4, label="La Nina")
    ax.axhline(0.5, color="#e05c2a", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(-0.5, color="#2a7ae0", lw=0.8, ls="--", alpha=0.5)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("Oceanic Nino Index (ONI) — El Nino / La Nina History", fontsize=13, pad=12)
    ax.set_xlabel("Year")
    ax.set_ylabel("ONI (°C anomaly)")
    ax.legend(loc="upper left", frameon=False)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "01_oni_timeseries.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────
# Plot 2: Disasters time series (dual axis)
# ─────────────────────────────────────────────
def plot_disasters_timeseries(annual: pd.DataFrame):
    df = annual.dropna(subset=["n_fires", "n_floods"])
    fig, ax1 = plt.subplots(figsize=(14, 5))
    ax2 = ax1.twinx()

    # Color background by ENSO phase
    for _, row in df.iterrows():
        color = COLORS.get(row.get("dominant_phase", "Neutral"), "#888")
        ax1.axvspan(row["year"] - 0.5, row["year"] + 0.5, alpha=0.12, color=color)

    ax1.bar(df["year"], df["n_fires"], color="#e05c2a", alpha=0.7, label="US Wildfires (count)", width=0.4, align="edge")
    ax2.bar(df["year"], df["n_floods"], color="#2a7ae0", alpha=0.7, label="Global Floods (count)", width=-0.4, align="edge")

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Number of Wildfires", color="#e05c2a")
    ax2.set_ylabel("Number of Flood Events", color="#2a7ae0")
    ax1.set_title("US Wildfires & Global Floods by Year\n(background shading = ENSO phase)", fontsize=12)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    patches = [mpatches.Patch(color=v, alpha=0.5, label=k) for k, v in COLORS.items()]
    ax1.legend(h1 + h2 + patches, l1 + l2 + [p.get_label() for p in patches],
               loc="upper left", frameon=False, fontsize=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "02_disasters_timeseries.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────
# Plot 3: Box plots by ENSO phase
# ─────────────────────────────────────────────
def plot_phase_boxplots(annual: pd.DataFrame):
    df = annual.dropna(subset=["n_fires", "n_floods", "dominant_phase"])
    phase_order = ["La Nina", "Neutral", "El Nino"]
    palette = {k: v for k, v in COLORS.items()}

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    metrics = [
        ("n_fires", "Number of US Wildfires"),
        ("n_floods", "Number of Global Floods"),
        ("total_area_burned", "Total Area Burned (acres)"),
    ]
    for ax, (col, label) in zip(axes, metrics):
        sub = df.dropna(subset=[col])
        sns.boxplot(data=sub, x="dominant_phase", y=col, order=phase_order,
                    palette=palette, ax=ax, width=0.5)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", rotation=15)

    fig.suptitle("Disaster Metrics by ENSO Phase", fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "03_phase_boxplots.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────
# Plot 4: Correlation heatmap
# ─────────────────────────────────────────────
def plot_correlation_heatmap(annual: pd.DataFrame):
    cols = ["mean_ONI", "max_ONI", "annual_temp_anomaly",
            "n_fires", "total_area_burned", "large_fires",
            "n_floods",
            "ONI_lag1", "ONI_lag2"]
    cols = [c for c in cols if c in annual.columns]
    df = annual[cols].dropna()

    corr = df.corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax, linewidths=0.5,
                annot_kws={"size": 7})
    ax.set_title("Pearson Correlation Matrix\n(El Nino indicators vs Disaster metrics)", fontsize=12)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "04_correlation_heatmap.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────
# Plot 5: Scatter ONI vs Wildfires & Floods
# ─────────────────────────────────────────────
def plot_scatter(annual: pd.DataFrame):
    df = annual.dropna(subset=["mean_ONI", "n_fires", "n_floods", "dominant_phase"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for phase, grp in df.groupby("dominant_phase"):
        color = COLORS.get(phase, "grey")
        ax1.scatter(grp["mean_ONI"], grp["n_fires"], color=color, label=phase,
                    alpha=0.75, edgecolors="white", s=70)
        ax2.scatter(grp["mean_ONI"], grp["n_floods"], color=color, label=phase,
                    alpha=0.75, edgecolors="white", s=70)

    # Trend lines
    for ax, col in [(ax1, "n_fires"), (ax2, "n_floods")]:
        sub = df.dropna(subset=[col])
        z = np.polyfit(sub["mean_ONI"], sub[col], 1)
        p = np.poly1d(z)
        x_range = np.linspace(sub["mean_ONI"].min(), sub["mean_ONI"].max(), 100)
        ax.plot(x_range, p(x_range), "k--", lw=1.5, alpha=0.5, label="Trend")

    ax1.set_xlabel("Mean Annual ONI (°C)")
    ax1.set_ylabel("Number of US Wildfires")
    ax1.set_title("ONI vs US Wildfires")
    ax1.legend(frameon=False)

    ax2.set_xlabel("Mean Annual ONI (°C)")
    ax2.set_ylabel("Number of Global Floods")
    ax2.set_title("ONI vs Global Flood Events")
    ax2.legend(frameon=False)

    for _, row in df.iterrows():
        if abs(row["mean_ONI"]) > 1.2:
            ax1.annotate(str(int(row["year"])), (row["mean_ONI"], row["n_fires"]),
                         fontsize=6.5, ha="left", va="bottom")

    plt.suptitle("El Nino Strength vs Disaster Frequency", fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "05_scatter_oni_vs_disasters.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


if __name__ == "__main__":
    print("=" * 50)
    print("  Step 3: Exploratory Data Analysis")
    print("=" * 50)
    monthly, annual = load()
    plot_oni_timeseries(monthly)
    plot_disasters_timeseries(annual)
    plot_phase_boxplots(annual)
    plot_correlation_heatmap(annual)
    plot_scatter(annual)
    print(f"\n✅ All EDA charts saved to outputs/")
