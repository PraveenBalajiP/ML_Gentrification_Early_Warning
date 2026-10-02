# Early Warning System for Gentrification in California - Replication Kit

Replicates *Pattabi (2018), "A Proximity-Based Early Warning System for Gentrification in California"*.
Pipeline: Census tract data -> difference features + spatial features -> Random Forest / XGBoost /
L1-logit / majority-vote ensemble -> Tables 1-3 and Figures 1-3 analogues.

## 0. Quick start (Windows PowerShell, from the project folder)

```powershell
python -m pip install -r requirements.txt
# 1) download the 14 zips (section 2) and drop them, still zipped, into  data\downloads\
python src\run_all.py            # runs steps 1-4;  add --fast for a ~1-minute smoke test
```

Or step by step:
```powershell
python src\01_validate_data.py     # unzips + renames + validates + fetches tract shapefile
python src\02_build_dataset.py     # builds data\processed\dataset.csv
python src\03_train_evaluate.py    # models + results   (--fast | --full optional)
python src\04_figures.py           # optional figures
```
No Census API key is needed (nothing calls the API).

## 1. Project layout
```
data/downloads/   <- you put the 14 Census .zip files here
data/raw/         <- auto-filled: YYYY_TABLE.csv + tiger_tracts/ (shapefile)
data/processed/   <- dataset.csv, feature_labels.csv
outputs/          <- results.csv, feature_importance.png, confusion_matrices.txt, best_params.json, figures
src/              <- config.py  common.py  01..04 scripts  run_all.py
```

## 2. Datasets to download (14 files, all free, ACS 5-year, **all census tracts in California**)

| # | Year | Table | Content | Link |
|---|------|-------|---------|------|
| 1 | 2010 | `B25085` | Price asked, vacant for-sale units | [open](https://data.census.gov/table/ACSDT5Y2010.B25085?g=040XX00US06$1400000) |
| 2 | 2010 | `B25105` | Median monthly housing costs | [open](https://data.census.gov/table/ACSDT5Y2010.B25105?g=040XX00US06$1400000) |
| 3 | 2010 | `DP03` | Selected economic characteristics | [open](https://data.census.gov/table/ACSDP5Y2010.DP03?g=040XX00US06$1400000) |
| 4 | 2010 | `S2502` | Demographic characteristics of occupied housing units | [open](https://data.census.gov/table/ACSST5Y2010.S2502?g=040XX00US06$1400000) |
| 5 | 2010 | `S2503` | Financial characteristics of occupied housing units (income bins) | [open](https://data.census.gov/table/ACSST5Y2010.S2503?g=040XX00US06$1400000) |
| 6 | 2011 | `B25085` | Price asked, vacant for-sale units | [open](https://data.census.gov/table/ACSDT5Y2011.B25085?g=040XX00US06$1400000) |
| 7 | 2011 | `B25105` | Median monthly housing costs | [open](https://data.census.gov/table/ACSDT5Y2011.B25105?g=040XX00US06$1400000) |
| 8 | 2011 | `DP03` | Selected economic characteristics | [open](https://data.census.gov/table/ACSDP5Y2011.DP03?g=040XX00US06$1400000) |
| 9 | 2011 | `S2502` | Demographic characteristics of occupied housing units | [open](https://data.census.gov/table/ACSST5Y2011.S2502?g=040XX00US06$1400000) |
| 10 | 2011 | `S2503` | Financial characteristics of occupied housing units (income bins) | [open](https://data.census.gov/table/ACSST5Y2011.S2503?g=040XX00US06$1400000) |
| 11 | 2012 | `B25105` | Median monthly housing costs | [open](https://data.census.gov/table/ACSDT5Y2012.B25105?g=040XX00US06$1400000) |
| 12 | 2012 | `S2503` | Financial characteristics of occupied housing units (income bins) | [open](https://data.census.gov/table/ACSST5Y2012.S2503?g=040XX00US06$1400000) |
| 13 | 2016 | `B25105` | Median monthly housing costs | [open](https://data.census.gov/table/ACSDT5Y2016.B25105?g=040XX00US06$1400000) |
| 14 | 2016 | `S2503` | Financial characteristics of occupied housing units (income bins) | [open](https://data.census.gov/table/ACSST5Y2016.S2503?g=040XX00US06$1400000) |

**For each link on data.census.gov:**
1. Open the link. Check the page header says *ACS 5-Year Estimates* for the right year, and the geography is
   **All Census Tracts within California**. (If the link doesn't pre-select it: click *Geos* -> *Tract* ->
   *California* -> *All Census Tracts within California*.)
2. Click **Download** -> choose the **CSV** option -> *Download* (the file may take a moment to be built).
3. Move the downloaded **.zip** into `data/downloads/` - **don't unzip, don't rename**. The script finds the
   `...-Data.csv` inside each zip and names it `YYYY_TABLE.csv` itself.
4. Repeat for all 14, then run `python src\01_validate_data.py`. It lists anything missing, with the link.

*Tract shapefile* (queen-contiguity graph): downloaded automatically by step 1 from
https://www2.census.gov/geo/tiger/TIGER2016/TRACT/tl_2016_06_tract.zip. If your network blocks it, open that link in a browser and drop the zip into `data/downloads/`.

Why 14 files? The paper names four tables, but the responses and spatial features also need `B25105` (median monthly housing cost) for
all four years and `S2503` (income bins) for 2012 and 2016 as well as 2010-2011.

## 3. What the code does (and how it maps to the paper)

| Paper | Implementation |
|---|---|
| Features: ~150 variables from S2502, S2503, B25085, DP03, **difference 2010 -> 2011** | `02`: every estimate column of these tables, 2011 minus 2010; drops columns >30% missing, constant, or duplicated |
| Response 1: change in monthly housing cost, inflation-adjusted, relabelled 1/0 | `B25105_001E`: 2016 cost (in 2012 dollars via CPI-U) minus 2012 cost; label 1 if > 0 |
| Response 2: Hellinger distance of income distribution to an all-affluent baseline, 2012 vs 2016 | 11 income bins of `S2503`; H(2012) - H(2016) > 0 (became more affluent) -> label 1 |
| 4 engineered features: first-order spatial lag + Local Moran's I of both responses over 2010-2011 | queen-contiguity graph of all CA tracts (unweighted, undirected); lag = mean of neighbours; Moran's I_i = Z_i / (sum Z^2 / n) * sum of neighbours' Z, with mean/variance from the **training** tracts as the paper states |
| 90% / 5% / 5% split (7,262 / 397 / 397) | fixed random split (seed 42) stored in `dataset.csv` column `split` |
| Random Forest (tuned n, p), XGBoost (tuned learning rate, depth, lambda), L1-logit (tuned C), majority-vote ensemble | `03`; two-stage grid search selecting on **validation accuracy** (coarse grid -> refined grid around winner) |
| Metrics: accuracy, precision, recall, no-information rate | on the held-out **test** set; no-info rate = majority-class share |
| Table 2 hyper-parameters (RF n=125, p=18; XGB 0.25 / 4 / 25; C=0.005) | also run with exactly these settings -> `results.csv` column `Hyperparameters = paper Table 2` |
| Table 3 confusion matrix (RF, income response) | `outputs/confusion_matrices.txt` |
| Figure 3/5 XGBoost importances | `outputs/feature_importance.png`, `xgb_top_features.csv` |
| Figure 1 / Figure 2 | `04_figures.py` -> `figure1_income_shift.png`, `figure2_tract_map.png` |

Run times (standard mode): a few minutes on a normal laptop; `--fast` ~1 min; `--full` (paper's complete grids) longest.

## 4. What to expect
Paper: cost response ~0.62-0.64 test accuracy vs 0.53 no-info (XGBoost / ensemble best); income response: nothing beats
no-info, high recall + low precision ("trigger-happy"); the two responses are nearly uncorrelated (rho = 0.06);
Local Moran's I features rank at the top of the importances. **Your numbers will differ** (different data pull, exact
variable set, split) - the *pattern* is what to compare. Step 2 prints the dataset summary (tracts, split sizes, class balance, rho)
so you can compare those with the paper first.

## 5. Honest list of deviations from the paper (put these in your write-up)
- American FactFinder no longer exists; data come from data.census.gov (same ACS tables).
- The paper's code isn't public in the PDFs: label thresholds (cost change > 0, H-change > 0), the split seed, grid
  ranges and the use of feature scaling for the LASSO are my reconstructions.
- The paper chose the pivot year (2012) by validation over {2012..2015}; this kit fixes 2012.
- The paper's ~150 features vs. whatever the four tables give after cleaning (step 2 prints the count).
- ACS 5-year windows overlap heavily (2010 = 2006-10, 2011 = 2007-11), so 2010->2011 differences are noisy - inherent to the design.
- Missing values are median-imputed inside each model pipeline (fit on training data only).

## 6. Troubleshooting
| Symptom | Fix |
|---|---|
| `MISS ...` in step 1 | download that table (link printed) and drop the zip into `data/downloads/` |
| `BAD ... no GEO_ID` / `no ACS estimate columns` | you downloaded a different file type; re-download using **Download -> CSV** and keep the zip |
| warning: far fewer than ~8,000 tracts | geography wasn't "All Census Tracts within California"; re-download |
| `income-bin columns not found` | open `data/raw/2012_S2503.csv`, look at row 1-2 for the "Less than $5,000" ... "$150,000 or more" columns and put their IDs in `INCOME_VARS` in `config.py` |
| geopandas install fails | `conda install -c conda-forge geopandas libpysal pyogrio` |
| step 3 too slow | use `--fast`, or lower the grids at the top of `03_train_evaluate.py` |
