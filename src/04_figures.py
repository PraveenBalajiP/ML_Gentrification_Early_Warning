"""Step 4 (optional): analogues of the paper's Figure 1 (income-distribution shift) and Figure 2 (tract map)."""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from common import income_bins, income_H


def fig1():
    b1, b2 = income_bins(PIVOT_YEAR).dropna(), income_bins(END_YEAR).dropna()
    idx = b1.index.intersection(b2.index)
    big = (b1.loc[idx].sum(axis=1) > 0) & (b2.loc[idx].sum(axis=1) > 0)
    idx = idx[big.to_numpy()]
    shift = (income_H(PIVOT_YEAR) - income_H(END_YEAR)).loc[idx]
    tract = shift.idxmax()
    p1, p2 = (b.loc[tract] / b.loc[tract].sum() for b in (b1, b2))
    x = np.arange(len(p1)); w = 0.4
    plt.figure(figsize=(9, 4.5))
    plt.bar(x - w / 2, p1.values, w, label=f"Income dist. {PIVOT_YEAR}", color="blue")
    plt.bar(x + w / 2, p2.values, w, label=f"Income dist. {END_YEAR}", color="salmon")
    plt.xticks(x, p1.index, rotation=60); plt.ylabel("share of households")
    plt.title(f"Tract {tract}: income distribution shift (largest shift toward affluence)")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT / "figure1_income_shift.png", dpi=200); plt.close()
    print("Saved figure1_income_shift.png")


def fig2():
    shp = next((RAW / "tiger_tracts").glob("*.shp"))
    gdf = gpd.read_file(shp)
    gcol = next(c for c in ("GEOID", "GEOID10") if c in gdf.columns)
    gdf["GEOID"] = gdf[gcol].astype(str).str.zfill(11)
    d = pd.read_csv(PROC / "dataset.csv", index_col="GEOID", usecols=["GEOID", "y_cost"])
    d.index = d.index.astype(str).str.zfill(11)
    gdf = gdf.merge(d, left_on="GEOID", right_index=True, how="left")
    ax = gdf.plot(column="y_cost", cmap="coolwarm", linewidth=0, figsize=(7, 8),
                  missing_kwds={"color": "lightgrey"}, legend=False)
    ax.set_axis_off(); ax.set_title("California tracts: red = real housing cost rose 2012-2016")
    plt.tight_layout(); plt.savefig(OUT / "figure2_tract_map.png", dpi=200); plt.close()
    print("Saved figure2_tract_map.png")


if __name__ == "__main__":
    fig1()
    fig2()
