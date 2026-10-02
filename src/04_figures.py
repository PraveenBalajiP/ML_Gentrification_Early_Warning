"""Step 4: generate the paper's analogue figures and confusion-matrix visualization."""
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import *
from common import income_bins, income_H


def confusion_matrix_figure():
    """Plot the 2x2 matrices recorded by step 3 as annotated heatmaps."""
    text = (OUT / "confusion_matrices.txt").read_text()
    matrices = []
    for block in text.strip().split("\n\n"):
        lines = block.splitlines()
        if len(lines) < 3:
            continue
        values = [int(value) for value in re.findall(r"\d+", " ".join(lines[1:]))]
        if len(values) != 4:
            raise ValueError(f"Expected four values in confusion-matrix block: {lines[0]}")
        response, setting, _ = lines[0].split(" | ", 2)
        matrices.append((response.replace("Response: ", ""), setting,
                         np.array(values).reshape(2, 2)))

    if not matrices:
        raise ValueError(f"No confusion matrices found in {OUT / 'confusion_matrices.txt'}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    for ax, (response, setting, matrix) in zip(axes.flat, matrices):
        image = ax.imshow(matrix, cmap="Blues")
        for row in range(2):
            for col in range(2):
                value = matrix[row, col]
                ax.text(col, row, str(value), ha="center", va="center",
                        color="white" if value > matrix.max() * 0.55 else "black",
                        fontsize=13, fontweight="bold")
        ax.set_xticks((0, 1), labels=("Predicted 0", "Predicted 1"))
        ax.set_yticks((0, 1), labels=("True 0", "True 1"))
        ax.set_title(f"{response}\n{setting}", fontsize=10)
        fig.colorbar(image, ax=ax, shrink=0.8, label="Count")

    fig.suptitle("Random Forest confusion matrices", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrices.png", dpi=200)
    plt.close(fig)
    print("Saved confusion_matrices.png")


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
    confusion_matrix_figure()
    fig1()
    fig2()
