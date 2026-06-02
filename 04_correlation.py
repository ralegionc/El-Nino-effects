"""
Step 4: Correlation & Statistical Tests
=========================================
Tests whether El Nino significantly affects wildfires and floods.

Tests performed:
  1. Pearson & Spearman correlation: ONI vs disaster metrics
  2. Lag correlation: ONI(t-k) vs fires/floods(t), k = 0..3 years
  3. One-way ANOVA: disaster metrics across ENSO phases
  4. Pairwise t-tests (El Nino vs Neutral, La Nina vs Neutral)

Outputs:
  - outputs/06_lag_correlation.png
  - outputs/07_anova_results.txt
  - outputs/08_statistical_summary.txt
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "..", "data")
OUT_DIR = os.path.join(BASE, "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


def load():
    return pd.read_csv(os.path.join(DATA_DIR, "merged_annual.csv"))


# ─────────────────────────────────────────────
# 1. Pearson & Spearman correlations
# ─────────────────────────────────────────────
def correlations(df: pd.DataFrame) -> str:
    targets = {
        "n_fires": "US Wildfire Count",
        "total_area_burned": "Total Area Burned",
        "n_floods": "Global Flood Count",
    }
    oni_vars = ["mean_ONI", "max_ONI", "ONI_lag1", "ONI_lag2"]
    lines = ["=" * 60, "CORRELATION: ONI vs Disaster Metrics", "=" * 60]

    for oni_var in oni_vars:
        lines.append(f"\n  ONI variable: {oni_var}")
        lines.append(f"  {'Metric':<25} {'Pearson r':>10} {'p-value':>10} {'Spearman r':>12} {'p-value':>10}")
        lines.append("  " + "-" * 70)
        for col, label in targets.items():
            sub = df[[oni_var, col]].dropna()
            if len(sub) < 5:
                continue
            pr, pp = stats.pearsonr(sub[oni_var], sub[col])
            sr, sp = stats.spearmanr(sub[oni_var], sub[col])
            sig_p = "***" if pp < 0.001 else "**" if pp < 0.01 else "*" if pp < 0.05 else ""
            sig_s = "***" if sp < 0.001 else "**" if sp < 0.01 else "*" if sp < 0.05 else ""
            lines.append(f"  {label:<25} {pr:>+9.3f}{sig_p:>3} {pp:>9.3f}   {sr:>+10.3f}{sig_s:>3} {sp:>9.3f}")

    lines.append("\n  Significance: * p<0.05  ** p<0.01  *** p<0.001")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# 2. Lag correlation plot
# ─────────────────────────────────────────────
def lag_correlation_plot(df: pd.DataFrame):
    df = df.sort_values("year").reset_index(drop=True)
    targets = {
        "n_fires": ("US Wildfire Count", "#e05c2a"),
        "n_floods": ("Global Flood Count", "#2a7ae0"),
        "total_area_burned": ("Area Burned (acres)", "#f0a830"),
    }
    lags = list(range(0, 5))

    fig, ax = plt.subplots(figsize=(9, 5))
    for col, (label, color) in targets.items():
        corrs = []
        for lag in lags:
            if lag == 0:
                sub = df[["mean_ONI", col]].dropna()
                r, _ = stats.pearsonr(sub["mean_ONI"], sub[col])
            else:
                shifted = df[["mean_ONI", col]].copy()
                shifted["mean_ONI_lag"] = shifted["mean_ONI"].shift(lag)
                sub = shifted[["mean_ONI_lag", col]].dropna()
                if len(sub) < 5:
                    r = np.nan
                else:
                    r, _ = stats.pearsonr(sub["mean_ONI_lag"], sub[col])
            corrs.append(r)
        ax.plot(lags, corrs, marker="o", label=label, color=color, lw=2)

    ax.axhline(0, color="black", lw=0.7)
    ax.axhline(0.3, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.axhline(-0.3, color="gray", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("Lag (years): ONI measured N years before disaster")
    ax.set_ylabel("Pearson Correlation (r)")
    ax.set_title("Lag Correlation: ONI(t−k) vs Disaster Metrics(t)\n(does last year's El Nino predict this year's disasters?)")
    ax.legend(frameon=False)
    ax.set_xticks(lags)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "06_lag_correlation.png")
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out}")


# ─────────────────────────────────────────────
# 3. ANOVA across ENSO phases
# ─────────────────────────────────────────────
def anova_tests(df: pd.DataFrame) -> str:
    targets = {
        "n_fires": "US Wildfire Count",
        "total_area_burned": "Area Burned (acres)",
        "n_floods": "Global Flood Count",
    }
    lines = ["\n" + "=" * 60, "ONE-WAY ANOVA: El Nino Phase vs Disaster Metrics", "=" * 60]
    lines.append(f"\n  {'Metric':<25} {'F-stat':>10} {'p-value':>12}  {'Significant?':>14}")
    lines.append("  " + "-" * 65)

    for col, label in targets.items():
        sub = df[["dominant_phase", col]].dropna()
        el_nino = sub[sub["dominant_phase"] == "El Nino"][col]
        neutral  = sub[sub["dominant_phase"] == "Neutral"][col]
        la_nina  = sub[sub["dominant_phase"] == "La Nina"][col]
        groups = [g for g in [el_nino, neutral, la_nina] if len(g) >= 2]
        if len(groups) < 2:
            lines.append(f"  {label:<25} {'N/A (insufficient data)':>40}")
            continue
        f, p = stats.f_oneway(*groups)
        sig = "YES ***" if p < 0.001 else "YES **" if p < 0.01 else "YES *" if p < 0.05 else "No"
        n_en, n_ne, n_ln = len(el_nino), len(neutral), len(la_nina)
        lines.append(f"  {label:<25} {f:>10.3f} {p:>12.4f}  {sig:>14}  (n: EN={n_en}, Ne={n_ne}, LN={n_ln})")

    lines.append("\n  ENSO groups: El Nino / Neutral / La Nina")
    lines.append("  H0: all groups have equal means")

    # Pairwise t-tests
    lines.append("\n" + "=" * 60)
    lines.append("PAIRWISE t-TESTS (El Nino vs each other phase)")
    lines.append("=" * 60)
    for col, label in targets.items():
        lines.append(f"\n  {label}:")
        sub = df[["dominant_phase", col]].dropna()
        g_en = sub[sub["dominant_phase"] == "El Nino"][col]
        g_ne = sub[sub["dominant_phase"] == "Neutral"][col]
        g_ln = sub[sub["dominant_phase"] == "La Nina"][col]
        for phase_name, g2 in [("Neutral", g_ne), ("La Nina", g_ln)]:
            g1 = g_en
            if len(g1) < 2 or len(g2) < 2:
                lines.append(f"    El Nino vs {phase_name}: insufficient data (n={len(g1)} vs {len(g2)})")
                continue
            t, p = stats.ttest_ind(g1, g2)
            direction = "HIGHER" if g1.mean() > g2.mean() else "LOWER"
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            lines.append(
                f"    El Nino vs {phase_name}: t={t:.2f}, p={p:.4f} {sig}"
                f"  | El Nino mean={g1.mean():.1f}, {phase_name} mean={g2.mean():.1f} -> El Nino is {direction}"
            )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# 4. Group statistics table
# ─────────────────────────────────────────────
def group_stats(df: pd.DataFrame) -> str:
    cols = ["n_fires", "total_area_burned", "n_floods"]
    cols = [c for c in cols if c in df.columns]
    tbl = df.groupby("dominant_phase")[cols].agg(["mean", "std", "count"])
    lines = ["\n" + "=" * 60, "MEAN +/- STD BY ENSO PHASE", "=" * 60]
    lines.append(tbl.to_string())
    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 50)
    print("  Step 4: Statistical Tests")
    print("=" * 50)
    annual = load()

    lag_correlation_plot(annual)

    report = []
    report.append(correlations(annual))
    report.append(anova_tests(annual))
    report.append(group_stats(annual))
    full_report = "\n".join(report)

    stats_out = os.path.join(OUT_DIR, "07_statistical_report.txt")
    with open(stats_out, "w", encoding="utf-8") as f:
        f.write(full_report)

    print(full_report)
    print(f"\n✅ Statistical report saved: {stats_out}")
