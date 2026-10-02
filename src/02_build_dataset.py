"""Step 2: build the paper-aligned California tract dataset.

 * Features  : 2011-minus-2010 change of every estimate in S2502, S2503, B25085, DP03
 * Responses : (1) real change in median monthly housing cost 2012->2016
               (2) change in Hellinger distance of the income distribution 2012->2016
 * Spatial   : first-order lag and Local Moran's I of the 2010->2011 change of each response,
               on a queen-contiguity graph of all California tracts (4 engineered features)
 * Split     : 90% train / 5% validation / 5% test
"""
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import scipy.sparse as sp
from config import *
from common import *

warnings.filterwarnings("ignore", category=FutureWarning)


def queen_adjacency(gdf):
    """Unweighted, undirected queen-contiguity matrix in the row order of gdf."""
    try:
        from libpysal.weights import Queen
        A = Queen.from_dataframe(gdf.reset_index(drop=True), silence_warnings=True).sparse.tocsr().astype(float)
        if A.shape[0] == len(gdf):
            return A
    except Exception as e:
        print(f"  (libpysal route failed: {e}; using geopandas spatial index instead)")
    left, right = gdf.sindex.query(gdf.geometry.values, predicate="intersects")
    keep = left != right
    A = sp.coo_matrix((np.ones(keep.sum()), (left[keep], right[keep])), shape=(len(gdf),) * 2).tocsr()
    return ((A + A.T) > 0).astype(float)


def main():
    t0, t1 = OBS_YEARS
    # ------------------------------------------------------------------ features
    parts, label_map = [], {}
    print("Building difference features ...")
    for table in FEATURE_TABLES:
        a, la = feature_columns(t0, table)
        b, lb = feature_columns(t1, table)
        lab = lb
        shared = a.columns.intersection(b.columns)
        # Census occasionally renumbers rows between vintages: only difference columns whose
        # row label (last segment) is the same in both years.
        common_cols = [c for c in shared if not (la.get(c) and lb.get(c)) or
                       norm_text(la[c].split("!!")[-1]) == norm_text(lb[c].split("!!")[-1])]
        if len(common_cols) < len(shared):
            print(f"  {table}: dropped {len(shared) - len(common_cols)} columns whose meaning changed between {t0} and {t1}")
        idx = a.index.intersection(b.index)
        if len(common_cols) == 0:
            raise ValueError(f"No common variables between {table} {t0} and {t1}.")
        parts.append((b.loc[idx, common_cols] - a.loc[idx, common_cols]).add_suffix("_diff"))
        label_map.update({f"{c}_diff": f"{table}: {lab.get(c) or c}" for c in common_cols})
        print(f"  {table:7s} {len(common_cols):4d} shared variables, {len(idx)} tracts")
    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.notna().mean() >= MIN_NONMISSING]          # drop mostly-missing columns
    X = X.loc[:, X.nunique(dropna=True) > 1]                   # drop constant columns
    X = X.loc[:, ~X.T.duplicated()]                            # drop exact duplicate columns
    print(f"  -> {X.shape[1]} usable features (paper: ~150)")

    # ------------------------------------------------------------------ responses
    cost12 = housing_cost(PIVOT_YEAR)
    cost16 = real_cost(housing_cost(END_YEAR), END_YEAR, PIVOT_YEAR)
    y_cost_cont = (cost16 - cost12).rename("y_cost_cont")                  # >0 : costs rose in real terms
    y_income_cont = (income_H(PIVOT_YEAR) - income_H(END_YEAR)).rename("y_income_cont")  # >0 : more affluent
    obs_cost = (real_cost(housing_cost(t1), t1, t0) - housing_cost(t0)).rename("obs_cost")
    obs_inc = (income_H(t0) - income_H(t1)).rename("obs_inc")             # observation-window change

    df = X.join(pd.concat([obs_cost, obs_inc, y_cost_cont, y_income_cont], axis=1), how="inner")
    df = df.dropna(subset=["obs_cost", "obs_inc", "y_cost_cont", "y_income_cont"])
    print(f"Tracts with complete features + responses: {len(df)}")

    # ------------------------------------------------------------------ geometry
    shp = next((RAW / "tiger_tracts").glob("*.shp"), None)
    if shp is None:
        raise FileNotFoundError("Tract shapefile missing - run 01_validate_data.py first.")
    gdf = gpd.read_file(shp)
    gcol = next(c for c in ("GEOID", "GEOID10") if c in gdf.columns)
    gdf["GEOID"] = gdf[gcol].astype(str).str.zfill(11)
    gdf = gdf[gdf["GEOID"].str.startswith(STATE_FIPS)].drop_duplicates("GEOID").reset_index(drop=True)
    lost = len(df) - len(df.index.intersection(gdf["GEOID"]))
    df = df.loc[df.index.intersection(gdf["GEOID"])]
    print(f"Tracts in shapefile: {len(gdf)} | modelling tracts after geometry join: {len(df)} (dropped {lost})")
    if len(df) < 7000:
        print("WARNING: far fewer tracts than the paper's 8,056 - tract boundaries may differ between "
              "vintages or a download was not statewide.")

    # ------------------------------------------------------------------ 90 / 5 / 5 split
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(df))
    n_hold = int(round(HOLDOUT_FRAC * len(df)))
    split = np.full(len(df), "train", dtype=object)
    split[perm[:n_hold]] = "val"
    split[perm[n_hold:2 * n_hold]] = "test"
    df["split"] = split
    train = (df["split"] == "train").to_numpy()

    # ------------------------------------------------------------------ spatial features
    A = queen_adjacency(gdf)
    pos = pd.Series(np.arange(len(gdf)), index=gdf["GEOID"]).loc[df.index].to_numpy()
    for name in ("cost", "inc"):
        y = np.full(len(gdf), np.nan)
        y[pos] = df[f"obs_{name}"].to_numpy()
        valid = ~np.isnan(y)
        # lag_i = sum_{j~i} y_j / #neighbours (neighbours without data are ignored)
        num, cnt = A @ np.where(valid, y, 0.0), A @ valid.astype(float)
        lag = np.divide(num, cnt, out=np.zeros_like(num), where=cnt > 0)
        # Local Moran's I_i = Z_i / (sum_j Z_j^2 / n) * sum_{j~i} Z_j ; mean & variance from the training sample
        ytr = df.loc[train, f"obs_{name}"].to_numpy()
        mu = ytr.mean()
        m2 = ((ytr - mu) ** 2).mean()
        z = np.where(valid, y - mu, 0.0)
        moran = z / m2 * (A @ z) if m2 > 0 else np.zeros_like(z)
        df[f"lag_{name}"], df[f"moran_{name}"] = lag[pos], moran[pos]
    label_map.update({"lag_cost": "Spatial lag - housing cost change (2010-11)",
                      "lag_inc": "Spatial lag - income distribution shift (2010-11)",
                      "moran_cost": "Local Moran's I - housing cost",
                      "moran_inc": "Local Moran's I - income distribution"})

    # ------------------------------------------------------------------ labels + report
    df["y_cost"] = (df["y_cost_cont"] > 0).astype(int)
    df["y_income"] = (df["y_income_cont"] > 0).astype(int)
    df = df.drop(columns=["obs_cost", "obs_inc"])
    df.to_csv(PROC / "dataset.csv")
    pd.Series(label_map, name="label").rename_axis("column").to_csv(PROC / "feature_labels.csv")

    rho = df[["y_cost_cont", "y_income_cont"]].corr().iloc[0, 1]
    print("\n==== DATASET SUMMARY ====")
    print(f"Tracts: {len(df)}  | features: {df.shape[1] - 5}  (incl. 4 spatial)")
    print("Split:", df["split"].value_counts().to_dict(), "(paper: 7262 / 397 / 397)")
    print(f"Positive rate  cost: {df.y_cost.mean():.3f}   income: {df.y_income.mean():.3f}  (paper: roughly balanced)")
    print(f"Correlation between the two responses: {rho:.3f}  (paper: 0.06)")
    print(f"Saved {PROC / 'dataset.csv'}")


if __name__ == "__main__":
    main()
