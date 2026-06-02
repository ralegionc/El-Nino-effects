"""
Step 1: Load Real Data (Local Files Only)
==========================================
NO synthetic data. NO HTTP downloads.
This script validates and parses the 4 real datasets you downloaded manually.

─────────────────────────────────────────────────────────────
REQUIRED FILES — place all in the data/ folder before running:
─────────────────────────────────────────────────────────────

1. data/oni.ascii.txt
   └─ URL: https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt
   └─ What: NOAA Oceanic Nino Index, 1950–present (seasonal, plain text)

2. data/nasa_temp_raw.csv
   └─ URL: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
   └─ What: NASA GISS monthly global surface temperature anomaly, 1880–present

3. data/floods.csv
   └─ URL: https://ourworldindata.org/grapher/number-of-natural-disaster-events.csv?v=1&csvType=full&useColumnShortNames=false
   └─ What: EM-DAT global disaster event counts by type & year (Our World in Data, no account needed)
   └─ Save the file as: data/floods.csv

4. data/US_wildfires.sqlite   ← PREFERRED (official USDA source, no account needed)
   └─ URL: https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6
            Click "FPA_FOD_20221014.sqlite" to download (~900 MB)
   └─ What: 2.3M US wildfire records 1992–2020 (USDA FPA FOD 6th Edition, public domain)

   OR alternatively:

   data/US_wildfires.sqlite   ← Kaggle mirror (free account needed)
   └─ URL: https://www.kaggle.com/datasets/behroozsohrabi/us-wildfire-records-6th-edition
   └─ Download the .sqlite file and save as: data/US_wildfires.sqlite

   OR if you prefer a CSV:
   data/US_wildfires.csv
   └─ URL: https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires
            (older 4th edition, 1992-2015, still valid for this analysis)
   └─ Export the SQLite "Fires" table to CSV, or use the pre-exported CSV if available

─────────────────────────────────────────────────────────────
Run:  python src/01_load_data.py
─────────────────────────────────────────────────────────────
"""

import os
import sys
import pandas as pd
import io

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

REQUIRED = {
    "oni.ascii.txt":    "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
    "nasa_temp_raw.csv":"https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
    "floods.csv":       "https://ourworldindata.org/grapher/number-of-natural-disaster-events.csv?v=1&csvType=full&useColumnShortNames=false",
}

WILDFIRE_SQLITE = os.path.join(DATA_DIR, "US_wildfires.sqlite")
WILDFIRE_CSV    = os.path.join(DATA_DIR, "US_wildfires.csv")


def check_files():
    """Verify all required files exist before doing any work."""
    print("\nChecking for required data files...\n")
    missing = []
    for fname, url in REQUIRED.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  ✓  {fname}  ({size_kb:.1f} KB)")
        else:
            print(f"  ✗  MISSING: {fname}")
            print(f"       Download from: {url}")
            missing.append(fname)

    # Wildfire: accept SQLite (official) OR CSV (Kaggle export)
    if os.path.exists(WILDFIRE_SQLITE):
        size_mb = os.path.getsize(WILDFIRE_SQLITE) / 1024 / 1024
        print(f"  ✓  US_wildfires.sqlite  ({size_mb:.0f} MB)  [official USDA]")
    elif os.path.exists(WILDFIRE_CSV):
        size_mb = os.path.getsize(WILDFIRE_CSV) / 1024 / 1024
        print(f"  ✓  US_wildfires.csv  ({size_mb:.0f} MB)  [CSV version]")
    else:
        print("  ✗  MISSING: US wildfire data (need one of):")
        print("       Option A — Official USDA SQLite (~900 MB, no account needed):")
        print("         https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6")
        print("         Download FPA_FOD_20221014.sqlite → save as data/US_wildfires.sqlite")
        print("       Option B — Kaggle mirror (free account needed):")
        print("         https://www.kaggle.com/datasets/behroozsohrabi/us-wildfire-records-6th-edition")
        print("         Save as data/US_wildfires.sqlite")
        print("       Option C — Older CSV edition (Kaggle, 1992-2015):")
        print("         https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires")
        print("         Export SQLite Fires table → save as data/US_wildfires.csv")
        missing.append("US_wildfires")

    if missing:
        print(f"\n❌ {len(missing)} file(s) missing. Download them and re-run.\n")
        sys.exit(1)
    print("\n✅ All files present. Parsing...\n")


# ─────────────────────────────────────────────
# 1. Parse ONI (NOAA plain-text format)
# ─────────────────────────────────────────────
def parse_oni() -> pd.DataFrame:
    """
    NOAA ONI format (oni.ascii.txt):
      SEAS  YR   ANOM
      DJF  1950  -1.53
      JFM  1950  -1.34
      ...
    Seasons map to the middle month of each 3-month window.
    """
    path = os.path.join(DATA_DIR, "oni.ascii.txt")
    season_to_month = {
        "DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4,
        "AMJ": 5, "MJJ": 6, "JJA": 7, "JAS": 8,
        "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12,
    }
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.upper().startswith("SEAS"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            season, year, anom = parts[0], parts[1], parts[2]
            if season not in season_to_month:
                continue
            try:
                rows.append({
                    "year":   int(year),
                    "month":  season_to_month[season],
                    "season": season,
                    "ONI":    float(anom),
                })
            except ValueError:
                continue

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("ONI file parsed to 0 rows — check the file format.")

    def phase(v):
        return "El Nino" if v >= 0.5 else ("La Nina" if v <= -0.5 else "Neutral")

    def strength(v):
        if v >= 1.5:   return "Strong El Nino"
        if v >= 0.5:   return "Weak/Mod El Nino"
        if v <= -1.5:  return "Strong La Nina"
        if v <= -0.5:  return "Weak/Mod La Nina"
        return "Neutral"

    df["enso_phase"]    = df["ONI"].apply(phase)
    df["enso_strength"] = df["ONI"].apply(strength)

    out = os.path.join(DATA_DIR, "oni.csv")
    df.to_csv(out, index=False)
    print(f"  ✓ ONI:        {len(df):>5} rows | {df.year.min()}–{df.year.max()} → oni.csv")
    return df


# ─────────────────────────────────────────────
# 2. Parse NASA GISS temperature
# ─────────────────────────────────────────────
def parse_nasa_temp() -> pd.DataFrame:
    """
    NASA GISS CSV has a 1-row header then:
      Year, Jan, Feb, ..., Dec, J-D, D-N, DJF, MAM, JJA, SON
    Temperature anomaly in hundredths of a degree (older files) or
    in degrees C directly. We use the monthly columns Jan–Dec.
    Missing values are marked as ***.
    """
    path = os.path.join(DATA_DIR, "nasa_temp_raw.csv")
    df_raw = pd.read_csv(path, skiprows=1, na_values=["***", "****", "-999"])

    month_cols = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    year_col = df_raw.columns[0]  # Usually "Year"

    keep = [c for c in month_cols if c in df_raw.columns]
    df_raw = df_raw[[year_col] + keep].copy()
    df_raw = df_raw.rename(columns={year_col: "Year"})

    # Melt to long format
    df_long = df_raw.melt(id_vars="Year", var_name="month_name", value_name="temp_anomaly")
    month_map = {m: i + 1 for i, m in enumerate(month_cols)}
    df_long["month"] = df_long["month_name"].map(month_map)
    df_long = df_long.rename(columns={"Year": "year"})
    df_long = df_long.dropna(subset=["temp_anomaly"])
    df_long["year"]         = df_long["year"].astype(int)
    df_long["temp_anomaly"] = df_long["temp_anomaly"].astype(float)

    # NASA reports in °C already (GISTEMP v4)
    # Values should be roughly -1 to +2; if max > 10 they're in hundredths → divide by 100
    if df_long["temp_anomaly"].abs().max() > 10:
        df_long["temp_anomaly"] /= 100.0

    out = df_long[["year", "month", "temp_anomaly"]]
    out.to_csv(os.path.join(DATA_DIR, "nasa_temp.csv"), index=False)
    print(f"  ✓ NASA GISS:  {len(out):>5} rows | {out.year.min()}–{out.year.max()} → nasa_temp.csv")
    return out


# ─────────────────────────────────────────────
# 3. Parse Our World in Data / EM-DAT floods
# ─────────────────────────────────────────────
def parse_floods() -> pd.DataFrame:
    """
    Our World in Data publishes EM-DAT disaster counts as a clean CSV.

    Download from:
      https://ourworldindata.org/grapher/number-of-natural-disaster-events.csv?v=1&csvType=full&useColumnShortNames=false
    Save as: data/floods.csv

    Format (columns vary slightly by version but always contain):
      Entity (country or "World"), Code (ISO), Year, Flood, ... (other disaster types)

    We filter to the "World" aggregate row for each year and extract flood counts.
    """
    path = os.path.join(DATA_DIR, "floods.csv")
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip() for c in df.columns]

    # Find year column
    year_col = next((c for c in df.columns if c.lower() == "year"), None)
    if year_col is None:
        raise ValueError(f"No 'Year' column found. Columns: {list(df.columns)}")

    # Find entity/country column
    entity_col = next(
        (c for c in df.columns if c.lower() in ["entity", "country", "location"]), None
    )

    # Find flood/disaster count column
    # OWID uses "Flood" on the by-type chart, "Disasters" on the total events chart
    flood_col = next(
        (c for c in df.columns
         if c.lower() in ["flood", "floods", "disasters", "number of disasters",
                          "number of events"]
         or "flood" in c.lower() or "disaster" in c.lower()), None
    )
    if flood_col is None:
        raise ValueError(
            f"No flood/disaster column found. Available: {list(df.columns)}. "
            "Expected a column named 'Flood', 'Floods', or 'Disasters'."
        )

    # Use World aggregate if available, else sum all countries per year
    if entity_col and "World" in df[entity_col].values:
        df_world = df[df[entity_col] == "World"].copy()
    elif entity_col:
        # Sum across all countries per year
        df_world = df.groupby(year_col)[flood_col].sum().reset_index()
        df_world.columns = [year_col, flood_col]
    else:
        df_world = df.copy()

    df_world = df_world[[year_col, flood_col]].copy()
    df_world.columns = ["year", "n_floods"]
    df_world["year"]    = pd.to_numeric(df_world["year"], errors="coerce")
    df_world["n_floods"] = pd.to_numeric(df_world["n_floods"], errors="coerce")
    df_world = df_world.dropna()
    df_world = df_world[df_world["year"].between(1950, 2025)]
    df_world["year"] = df_world["year"].astype(int)
    df_world = df_world.sort_values("year").reset_index(drop=True)

    # Also pull deaths if available
    deaths_col = next(
        (c for c in df.columns if "death" in c.lower() or "dead" in c.lower()), None
    )
    if deaths_col and entity_col and "World" in df[entity_col].values:
        deaths = df[df[entity_col] == "World"][[year_col, deaths_col]].copy()
        deaths.columns = ["year", "flood_deaths"]
        deaths["year"] = pd.to_numeric(deaths["year"], errors="coerce").astype("Int64")
        df_world = df_world.merge(deaths, on="year", how="left")

    out_path = os.path.join(DATA_DIR, "floods_clean.csv")
    df_world.to_csv(out_path, index=False)
    print(f"  ✓ Floods:     {len(df_world):>5} rows | {df_world.year.min()}–{df_world.year.max()} → floods_clean.csv")
    return df_world


# ─────────────────────────────────────────────
# 4. Parse US Wildfires (Kaggle CSV)
# ─────────────────────────────────────────────
def parse_wildfires() -> pd.DataFrame:
    """
    Supports two input formats:

    A) data/US_wildfires.sqlite  — Official USDA FPA FOD (6th ed., 1992-2020)
       Table: "Fires"
       Key columns: FIRE_YEAR, STATE, FIRE_SIZE, DISCOVERY_DOY, STAT_CAUSE_DESCR

    B) data/US_wildfires.csv — CSV export of the same SQLite table
       Same column names apply.

    We select the SQLite if present, else fall back to CSV.
    """
    import sqlite3

    want_cols = ["FIRE_YEAR", "FIRE_SIZE", "STATE", "DISCOVERY_DOY", "STAT_CAUSE_DESCR"]

    if os.path.exists(WILDFIRE_SQLITE):
        print("  Reading wildfires from SQLite (this may take ~30s for 2.3M rows)...")
        con = sqlite3.connect(WILDFIRE_SQLITE)
        # Only pull columns we need to keep memory low
        available = [r[1] for r in con.execute("PRAGMA table_info(Fires)").fetchall()]
        select = [c for c in want_cols if c in available]
        if not select:
            raise ValueError(
                f"No expected columns found in SQLite Fires table.\n"
                f"Available: {available[:20]}..."
            )
        query = f"SELECT {', '.join(select)} FROM Fires"
        df = pd.read_sql_query(query, con)
        con.close()
    elif os.path.exists(WILDFIRE_CSV):
        print("  Reading wildfires from CSV...")
        df = pd.read_csv(WILDFIRE_CSV, low_memory=False,
                         usecols=lambda c: c.upper().strip() in want_cols)
    else:
        raise FileNotFoundError(
            "No wildfire file found. Expected data/US_wildfires.sqlite or data/US_wildfires.csv"
        )

    df.columns = [c.strip().upper() for c in df.columns]

    # Rename to standard names
    df = df.rename(columns={
        "FIRE_YEAR":        "year",
        "FIRE_SIZE":        "fire_size_acres",
        "STATE":            "state",
        "DISCOVERY_DOY":    "discovery_doy",
        "STAT_CAUSE_DESCR": "cause",
    })

    df["year"]           = pd.to_numeric(df["year"], errors="coerce")
    df["fire_size_acres"] = pd.to_numeric(df["fire_size_acres"], errors="coerce")
    df = df.dropna(subset=["year", "fire_size_acres"])
    df = df[df["year"].between(1992, 2024)]
    df["year"] = df["year"].astype(int)

    out = df.to_csv(os.path.join(DATA_DIR, "US_wildfires_clean.csv"), index=False)
    print(f"  ✓ Wildfires:  {len(df):>7,} rows | {df.year.min()}–{df.year.max()} → US_wildfires_clean.csv")
    return df


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  El Nino Impact Project — Step 1: Load Real Data")
    print("=" * 55)

    check_files()

    errors = []
    for name, fn in [
        ("ONI",         parse_oni),
        ("NASA Temp",   parse_nasa_temp),
        ("Floods",      parse_floods),
        ("Wildfires",   parse_wildfires),
    ]:
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            errors.append(name)

    print()
    if errors:
        print(f"❌ {len(errors)} dataset(s) failed: {errors}")
        print("   Fix the issues above and re-run.")
        sys.exit(1)
    else:
        print("✅ All 4 real datasets parsed successfully.")
        print("   Next: python src/02_clean_merge.py")
