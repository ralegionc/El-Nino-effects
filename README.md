# El-Nino-effects

# El Nino Impact Analysis: Wildfires & Floods

Does El Nino meaningfully increase wildfire risk in the US? This project builds a full data pipeline from raw government datasets to statistical analysis and ML classification to answer that question with real numbers.

**Short answer:** Yes. El Nino years average 6,551 US wildfires vs 5,517 in neutral years, with 4x more total area burned. A model trained only on climate signals predicts high-risk years at AUC = 0.77.

---

## Findings

- Pearson correlation between El Nino strength (mean ONI) and US wildfire count: **r = 0.55, p = 0.002**
- Using peak ONI intensity instead: **r = 0.72, p < 0.001**
- El Nino years average **5.66 million acres burned** vs 1.39 million in neutral years
- The effect is same-year only — lag-1 correlation drops to r = 0.15 and becomes insignificant
- Top ML predictor: **global temperature anomaly**, not ONI directly, suggesting the mechanism is El Nino → warming → drought conditions → fire
- No significant relationship found between El Nino and global flood counts (global aggregates mask regional patterns)

---

## Data Sources

All data is public domain. Download manually before running.

| File | Source |
|---|---|
| `data/oni.ascii.txt` | https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt |
| `data/nasa_temp_raw.csv` | https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv |
| `data/floods.csv` | https://ourworldindata.org/grapher/number-of-natural-disaster-events |
| `data/US_wildfires.sqlite` | https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6 (FPA_FOD_20221014.sqlite) |

The wildfire database contains 2.3 million geo-referenced records from 1992–2020 (USDA FPA FOD 6th Edition).

---

## Project Structure

```
el_nino_project/
├── data/                   # Raw and processed datasets (not tracked)
├── outputs/                # Charts, reports, model artifacts
├── src/
│   ├── 01_load_data.py     # Validate and parse all 4 datasets
│   ├── 02_clean_merge.py   # Clean, aggregate, and merge into annual/monthly CSVs
│   ├── 03_eda.py           # Exploratory analysis — 5 charts
│   ├── 04_correlation.py   # Pearson/Spearman correlations, lag analysis, ANOVA, t-tests
│   └── 05_ml_model.py      # Random Forest, Logistic Regression, Gradient Boosting
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/ralegionc/el-nino-impact
cd el_nino_project
pip install -r requirements.txt
```

Download the four data files listed above into the `data/` folder, then run in order:

```bash
python src/01_load_data.py
python src/02_clean_merge.py
python src/03_eda.py
python src/04_correlation.py
python src/05_ml_model.py
```

`01_load_data.py` will tell you exactly which files are missing before doing anything else.

---

## Outputs

| File | Description |
|---|---|
| `01_oni_timeseries.png` | 70-year El Nino / La Nina history |
| `02_disasters_timeseries.png` | Wildfires and floods by year, shaded by ENSO phase |
| `03_phase_boxplots.png` | Wildfire count, flood count, area burned by ENSO phase |
| `04_correlation_heatmap.png` | Pearson correlation matrix across all variables |
| `05_scatter_oni_vs_disasters.png` | ONI strength vs wildfire/flood frequency, labeled by year |
| `06_lag_correlation.png` | Lag-0 through lag-4 correlation between ONI and disaster metrics |
| `07_statistical_report.txt` | Full correlation, ANOVA, and t-test results |
| `08_ml_results.png` | Feature importance, confusion matrices, ROC curves |
| `09_ml_report.txt` | Cross-validated AUC scores and classification reports |

---

## Stack

Python · pandas · scikit-learn · matplotlib · seaborn · scipy · sqlite3

---

## Notes

- ONI data must contain anomaly values (roughly -2.5 to +3.0). The NOAA page also serves absolute SST values (~25-28°C) depending on which file you download — make sure you have the anomaly index.
- The flood model (AUC ~0.97) is not a reliable result. The flood dataset covers 1950-2023 while wildfire data covers 1992-2020, creating a time window mismatch. The model learns the period, not the climate signal. Regional flood data broken down by country or river basin would be the correct approach.
- ANOVA results are limited by sample size — only 12 El Nino years, 7 Neutral, and 10 La Nina years fall within the 1992-2020 wildfire window. The correlation results are more reliable since they use continuous ONI values rather than categorical phases.

---

## Citation

Short, Karen C. 2022. Spatial wildfire occurrence data for the United States, 1992-2020 [FPA_FOD_20221014]. 6th Edition. Fort Collins, CO: Forest Service Research Data Archive. https://doi.org/10.2737/RDS-2013-0009.6
