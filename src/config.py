"""Configuration for the California gentrification early-warning replication
(Pattabi 2018).  Everything tunable lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS = ROOT / "data" / "downloads"   # <- drop the zip files from data.census.gov here
RAW = ROOT / "data" / "raw"               # <- 01_validate_data.py puts YYYY_TABLE.csv here
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
for _p in (DOWNLOADS, RAW, PROC, OUT):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- study design (paper Sec. II)
STATE_FIPS = "06"                  # California
OBS_YEARS = (2010, 2011)           # features = change between these ACS 5-year vintages
PIVOT_YEAR, END_YEAR = 2012, 2016  # responses = change between these vintages

FEATURE_TABLES = ("S2502", "S2503", "B25085", "DP03")   # the four tables named in the paper
COST_TABLE = "B25105"              # median monthly housing costs (single variable B25105_001E)
INCOME_TABLE = "S2503"             # contains the 11 income bins of the paper's Table 1
HOUSING_COST_VAR = "B25105_001E"

# Fallback IDs for the 11 income bins (the code first tries to find them by label text)
INCOME_VARS = [f"S2503_C01_{i:03d}E" for i in range(2, 13)]
INCOME_BIN_KEYS = ["lessthan5000", "5000to9999", "10000to14999", "15000to19999",
                   "20000to24999", "25000to34999", "35000to49999", "50000to74999",
                   "75000to99999", "100000to149999", "150000ormore"]
INCOME_BIN_LABELS = ["<$5k", "$5-10k", "$10-15k", "$15-20k", "$20-25k", "$25-35k",
                     "$35-50k", "$50-75k", "$75-100k", "$100-150k", "$150k+"]

# Every (year, table) file the pipeline needs -> 14 downloads
REQUIRED = sorted(
    {(y, t) for y in OBS_YEARS for t in FEATURE_TABLES + (COST_TABLE,)}
    | {(y, t) for y in (PIVOT_YEAR, END_YEAR) for t in (INCOME_TABLE, COST_TABLE)}
)

# CPI-U annual averages: express housing costs in constant dollars
CPI = {2010: 218.056, 2011: 224.939, 2012: 229.594, 2016: 240.007}

# ---------------------------------------------------------------- data handling
SEED = 42
HOLDOUT_FRAC = 0.05        # 90% train / 5% validation / 5% test (paper Sec. II)
MIN_NONMISSING = 0.70      # drop feature columns with >30% missing values
TIGER_URL = "https://www2.census.gov/geo/tiger/TIGER2016/TRACT/tl_2016_06_tract.zip"

# ---------------------------------------------------------------- paper's final hyper-parameters (Table 2)
PAPER_PARAMS = {
    "RandomForest": {"n_estimators": 125, "max_features": 18},
    "XGBoost": {"learning_rate": 0.25, "max_depth": 4, "reg_lambda": 25},
    "L1Logit": {"C": 0.005},
}


def census_url(year, table):
    """data.census.gov link for a table (pre-filtered to all CA tracts)."""
    kind = "ACSDP5Y" if table.startswith("DP") else "ACSST5Y" if table.startswith("S") else "ACSDT5Y"
    return f"https://data.census.gov/table/{kind}{year}.{table}?g=040XX00US06$1400000"
