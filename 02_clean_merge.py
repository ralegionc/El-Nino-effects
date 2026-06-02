"""
Step 2: Clean & Merge Datasets
================================
Reads all raw CSVs from data/, cleans them, and produces:
  - data/merged_annual.csv   → one row per year with all indicators
  - data/merged_monthly.csv  → one row per year-month
"""

import os
import pandas as pd
import numpy as np

BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "..", "data")


def label_enso_phase(oni: float) -> str:
    """Classify a single ONI value into El Nino / Neutral / La Nina."""
    if oni >= 0.5:
        return "El Nino"
    elif oni <= -0.5:
        return "La Nina"
    else:
        return "Neutral"


def label_enso_strength(oni: float) -> str:
    """Further classify El Nino / La Nina by strength."""
    if oni >= 1.5:
        return "Strong El Nino"
    elif oni >= 0.5:
        return "Weak/Mod El Nino"
    elif oni <= -1.5:
        return "Strong La Nina"
    elif oni <= -0.5:
        return "Weak/Mod La Nina"
    else:
        return "Neutral"


# ─────────────────────────────────────────────
# Load & clean ONI
# ─────────────────────────────────────────────
def load_oni() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "oni.csv"))
    df = df[["year", "month", "ONI"]].dropna()
    df["enso_phase"] = df["ONI"].apply(label_enso_phase)
    df["enso_strength"] = df["ONI"].apply(label_enso_strength)
    print(f"ONI: {len(df)} monthly rows, {df['year'].min()}–{df['year'].max()}")
    return df


# ─────────────────────────────────────────────
# Load & clean temperature anomaly
# ─────────────────────────────────────────────
def load_temp() -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_DIR, "nasa_temp.csv"))
    df = df[["year", "month", "temp_anomaly"]].dropna()
    print(f"NASA Temp: {len(df)} monthly rows")
    return df


# ─────────────────────────────────────────────
# Load & clean wildfires
# ─────────────────────────────────────────────
def load_wildfires() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "US_wildfires.csv")
    df = pd.read_csv(path, low_memory=False)

    # Standardize column names (handles both real Kaggle data and synthetic)
    df.columns = [c.upper().strip() for c in df.columns]
    if "FIRE_YEAR" not in df.columns and "YEAR" in df.columns:
        df = df.rename(columns={"YEAR": "FIRE_YEAR"})
    if "FIRE_SIZE" not in df.columns and "FIRE_SIZE_ACRES" in df.columns:
        df = df.rename(columns={"FIRE_SIZE_ACRES": "FIRE_SIZE"})

    df = df[["FIRE_YEAR", "STATE", "FIRE_SIZE"]].dropna()
    df = df.rename(columns={"FIRE_YEAR": "year", "STATE": "state",
                             "FIRE_SIZE": "fire_size_acres"})
    df["year"] = df["year"].astype(int)
    df["fire_size_acres"] = pd.to_numeric(df["fire_size_acres"], errors="coerce")
    df = df.dropna()

    # Annual aggregation
    annual = df.groupby("year").agg(
        n_fires=("fire_size_acres", "count"),
        total_area_burned=("fire_size_acres", "sum"),
        mean_fire_size=("fire_size_acres", "mean"),
        large_fires=("fire_size_acres", lambda x: (x > 1000).sum()),  # >1000 acres
    ).reset_index()
    print(f"Wildfires: {len(df):,} raw records → {len(annual)} annual rows")
    return annual


# ─────────────────────────────────────────────
# Load & clean floods (Our World in Data / EM-DAT)
# ─────────────────────────────────────────────
def load_floods() -> pd.DataFrame:
    """
    Reads floods_clean.csv produced by 01_load_data.py.
    Already annual-level data: year, n_floods (and optionally flood_deaths).
    """
    # Prefer the cleaned output from step 1; fall back to raw floods.csv
    clean_path = os.path.join(DATA_DIR, "floods_clean.csv")
    raw_path   = os.path.join(DATA_DIR, "floods.csv")
    path = clean_path if os.path.exists(clean_path) else raw_path

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # Ensure n_floods column exists
    if "n_floods" not in df.columns:
        flood_col = next((c for c in df.columns if "flood" in c.lower()), None)
        if flood_col:
            df = df.rename(columns={flood_col: "n_floods"})
        else:
            raise ValueError(f"Cannot find flood count column. Columns: {list(df.columns)}")

    df["n_floods"] = pd.to_numeric(df["n_floods"], errors="coerce")
    df = df.dropna(subset=["n_floods"])
    df = df[df["year"].between(1950, 2025)]

    print(f"Floods: {len(df)} annual rows | {df.year.min()}–{df.year.max()}")
    return df


# ─────────────────────────────────────────────
# Build annual ONI summary
# ─────────────────────────────────────────────
def annual_oni(df_oni: pd.DataFrame) -> pd.DataFrame:
    """Compute yearly mean ONI and dominant ENSO phase using numeric thresholds only."""
    df_oni = df_oni.copy()
    df_oni["is_el_nino"] = (df_oni["ONI"] >= 0.5).astype(int)
    df_oni["is_la_nina"] = (df_oni["ONI"] <= -0.5).astype(int)

    agg = df_oni.groupby("year").agg(
        mean_ONI=("ONI", "mean"),
        max_ONI=("ONI", "max"),
        min_ONI=("ONI", "min"),
        months_el_nino=("is_el_nino", "sum"),
        months_la_nina=("is_la_nina", "sum"),
    ).reset_index()

    def classify(r):
        if r.months_el_nino > 3 and r.months_el_nino >= r.months_la_nina:
            return "El Nino"
        elif r.months_la_nina > 3:
            return "La Nina"
        else:
            return "Neutral"

    agg["dominant_phase"] = agg.apply(classify, axis=1)
    print("Phase counts:", agg["dominant_phase"].value_counts().to_dict())
    return agg


# ─────────────────────────────────────────────
# Main merge
# ─────────────────────────────────────────────
def build_merged():
    print("\n" + "=" * 50)
    print("  Step 2: Cleaning & Merging Datasets")
    print("=" * 50 + "\n")

    oni = load_oni()
    temp = load_temp()
    fires = load_wildfires()
    floods = load_floods()

    # ── Monthly merge (ONI + temp) ──
    monthly = pd.merge(oni, temp, on=["year", "month"], how="inner")
    out_monthly = os.path.join(DATA_DIR, "merged_monthly.csv")
    monthly.to_csv(out_monthly, index=False)
    print(f"\n✓ Monthly data: {len(monthly)} rows → {out_monthly}")

    # ── Annual merge ──
    oni_annual = annual_oni(oni)
    temp_annual = temp.groupby("year")["temp_anomaly"].mean().reset_index()
    temp_annual.columns = ["year", "annual_temp_anomaly"]

    merged = oni_annual.merge(temp_annual, on="year", how="inner")
    merged = merged.merge(fires, on="year", how="left")
    merged = merged.merge(floods, on="year", how="left")

    # Lag features: ONI from previous year (disasters may lag)
    merged = merged.sort_values("year")
    merged["ONI_lag1"] = merged["mean_ONI"].shift(1)
    merged["ONI_lag2"] = merged["mean_ONI"].shift(2)

    out_annual = os.path.join(DATA_DIR, "merged_annual.csv")
    merged.to_csv(out_annual, index=False)
    print(f"✓ Annual data: {len(merged)} rows → {out_annual}")
    print(f"\nColumns: {list(merged.columns)}")
    print("\nSample:\n", merged.tail(5).to_string())

    return monthly, merged


if __name__ == "__main__":
    build_merged()
