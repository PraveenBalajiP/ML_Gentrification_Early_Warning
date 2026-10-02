"""Shared helpers: robust reading of Census files + the paper's income / housing-cost measures."""
import re
from functools import lru_cache
import numpy as np
import pandas as pd
from config import *

# ACS estimate columns: B25105_001E, S2503_C01_002E, DP03_0052E ... (margins "M", percent "PE", annotations excluded)
EST_RE = re.compile(r"^[A-Z]+\d+_(?:C\d+_)?\d+E$")


def find_file(year, table):
    f = RAW / f"{year}_{table}.csv"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found. Put the Census zip files in {DOWNLOADS} "
                                f"and run:  python src/01_validate_data.py")
    return f


def _numeric(col):
    # "3,500+" -> 3500 ; "250,000+" -> 250000 ; "(X)", "-", "N", "**" -> NaN
    return pd.to_numeric(col.astype(str).str.replace(r"[,+$%\s]", "", regex=True), errors="coerce")


@lru_cache(maxsize=None)
def read_table(year, table):
    """Return (numeric estimate columns indexed by 11-digit tract GEOID, {column: label text}).
    Handles data.census.gov '-Data.csv' files (2nd row = descriptive labels) and API-style files."""
    raw = pd.read_csv(find_file(year, table), dtype=str, low_memory=False, encoding="utf-8-sig")
    raw.columns = [str(c).strip() for c in raw.columns]
    labels = {}
    geo_col = next((c for c in raw.columns if c.upper() in ("GEO_ID", "GEOID")), None)
    if geo_col is not None:
        if len(raw) and not re.search(r"\d{9}", str(raw[geo_col].iloc[0])):
            labels = {c: str(v) for c, v in raw.iloc[0].items()}   # descriptive row, not a tract
            raw = raw.iloc[1:]
        geoid = raw[geo_col].astype(str).str.extract(r"(\d{11})$", expand=False)
    elif all(c in raw.columns for c in ("state", "county", "tract")):
        geoid = raw["state"].str.zfill(2) + raw["county"].str.zfill(3) + raw["tract"].str.zfill(6)
    else:
        raise ValueError(f"{year}_{table}.csv has no GEO_ID / GEOID column - is this the '-Data.csv' file?")
    raw = raw.assign(GEOID=geoid)
    raw = raw[raw["GEOID"].notna() & raw["GEOID"].str.startswith(STATE_FIPS)]
    raw = raw.drop_duplicates("GEOID").set_index("GEOID")
    est = [c for c in raw.columns if EST_RE.match(c)]
    if not est:
        raise ValueError(f"{year}_{table}.csv has no ACS estimate columns (e.g. {table}_C01_001E). "
                         f"Download the table again and use the file ending in '-Data.csv'.")
    num = raw[est].apply(_numeric)
    num = num.mask(num < -1e6)                      # Census sentinels (-666666666, -999999999 ...)
    return num, {c: labels.get(c, "") for c in est}


def feature_columns(year, table):
    """Return Census estimate columns and exclude margin-of-error columns."""
    df, labels = read_table(year, table)

    keep = [
        c for c in df.columns
        if EST_RE.match(c)
    ]
    return df[keep], {c: labels.get(c, "") for c in keep}

def real_cost(series, year, base):
    """Express `year` dollars in `base`-year dollars (CPI-U)."""
    return series * CPI[base] / CPI[year]


def housing_cost(year):
    df, _ = read_table(year, COST_TABLE)
    if HOUSING_COST_VAR not in df.columns:
        raise ValueError(f"{COST_TABLE} {year}: column {HOUSING_COST_VAR} not found. Columns: {list(df.columns)[:6]}")
    return df[HOUSING_COST_VAR].rename("housing_cost")


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


norm_text = _norm


def income_columns(year):
    """The 11 household-income bin columns of S2503 (total occupied units, estimates)."""
    df, labels = read_table(year, INCOME_TABLE)
    if any(labels.values()):                                  # preferred: find by label text
        c01 = [c for c in df.columns if "_C01_" in c and labels.get(c, "").lower().startswith("estimate")]
        found = [next((c for c in c01 if _norm(labels[c].split("!!")[-1]) == key), None)
                 for key in INCOME_BIN_KEYS]
        if all(found):
            return found
    missing = [c for c in INCOME_VARS if c not in df.columns]
    if missing:
        raise ValueError(f"S2503 {year}: income-bin columns not found ({missing[:3]} ...). "
                         f"Edit INCOME_VARS in config.py to the right column IDs.")
    return list(INCOME_VARS)


def income_bins(year):
    df, _ = read_table(year, INCOME_TABLE)
    b = df[income_columns(year)].copy()
    b.columns = INCOME_BIN_LABELS
    return b


def income_H(year):
    """Hellinger distance between a tract's income distribution and a baseline where all
    households are in the top bin (paper Sec. II).  Low = affluent, high = low-income."""
    b = income_bins(year).dropna()
    b = b[b.sum(axis=1) > 0]
    P = b.div(b.sum(axis=1), axis=0)
    q = np.zeros(P.shape[1]); q[-1] = 1.0
    h = np.sqrt(((np.sqrt(P.values) - np.sqrt(q)) ** 2).sum(axis=1) / 2.0)
    return pd.Series(h, index=P.index, name="H")
