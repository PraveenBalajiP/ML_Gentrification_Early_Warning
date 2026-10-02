"""Step 3: train and evaluate the paper's four classifiers on both responses.

Models  : Random Forest (Gini), XGBoost (logistic loss), L1-penalised logistic regression (LASSO),
          majority-vote ensemble of the three.
Tuning  : two-stage grid search on the validation set (coarse grid, then a grid refined around the winner).
Reports : accuracy, precision, recall, no-information rate on the held-out test set
          (a) with grid-search hyper-parameters, (b) with the paper's published hyper-parameters (Table 2).
Usage   : python src/03_train_evaluate.py          (standard grids, a few minutes)
          python src/03_train_evaluate.py --fast   (tiny grids, ~1 minute, for a quick check)
          python src/03_train_evaluate.py --full   (the paper's full grids incl. 25-500 trees; slowest)
"""
import sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from config import *

warnings.filterwarnings("ignore")
FAST, FULL = "--fast" in sys.argv, "--full" in sys.argv
MODE = "FAST" if FAST else "FULL" if FULL else "STANDARD"
NON_FEATURES = {"y_cost", "y_income", "y_cost_cont", "y_income_cont", "split"}
TITLES = {"y_cost": "Response: change in monthly cost of housing",
          "y_income": "Response: change in income distribution"}
MODELS = ("RandomForest", "XGBoost", "L1Logit")

STAGE1_FULL = {    # the ranges described in the paper (n in {25,50,100,150,250,500}, ...)
    "RandomForest": {"n_estimators": [25, 50, 100, 150, 250, 500], "max_features": [5, 10, 18, 30]},
    "XGBoost": {"learning_rate": [0.05, 0.1, 0.25, 0.5], "max_depth": [2, 4, 6], "reg_lambda": [1, 5, 25, 100]},
    "L1Logit": {"C": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]},
}
STAGE1 = {         # standard: same idea, fewer combinations
    "RandomForest": {"n_estimators": [50, 100, 150, 250], "max_features": [10, 18, 30]},
    "XGBoost": {"learning_rate": [0.1, 0.25, 0.5], "max_depth": [2, 4, 6], "reg_lambda": [5, 25, 100]},
    "L1Logit": {"C": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0]},
}
STAGE1_FAST = {
    "RandomForest": {"n_estimators": [50, 125, 250], "max_features": [10, 18]},
    "XGBoost": {"learning_rate": [0.1, 0.25], "max_depth": [4], "reg_lambda": [5, 25]},
    "L1Logit": {"C": [0.005, 0.05]},
}


def refine(name, best):
    """Stage 2: a narrower grid centred on the stage-1 winner."""
    if name == "RandomForest":
        n, m = best["n_estimators"], best["max_features"]
        return {"n_estimators": sorted({max(10, int(n * f)) for f in (0.8, 1, 1.25)}),
                "max_features": sorted({max(2, int(round(m * f))) for f in (0.75, 1, 1.5)})}
    if name == "XGBoost":
        lr, d, lam = best["learning_rate"], best["max_depth"], best["reg_lambda"]
        return {"learning_rate": sorted({lr * 0.5, lr, lr * 1.5}),
                "max_depth": sorted({max(1, d - 1), d, d + 1}),
                "reg_lambda": sorted({lam / 2, lam, lam * 2})}
    return {"C": sorted({best["C"] * f for f in (0.5, 0.75, 1, 1.5, 2)})}


def make(name, params, n_feat):
    p = dict(params)
    if name == "RandomForest":
        p["max_features"] = min(int(p["max_features"]), n_feat)
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(random_state=SEED, n_jobs=-1, **p))
    if name == "XGBoost":
        return make_pipeline(SimpleImputer(strategy="median"),
                             XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1, **p))
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000,
                                            random_state=SEED, **p))


def two_stage_search(name, Xtr, ytr, Xva, yva, n_feat):
    def run(grid):
        best = (-1.0, None)
        for p in ParameterGrid(grid):
            acc = accuracy_score(yva, make(name, p, n_feat).fit(Xtr, ytr).predict(Xva))
            if acc > best[0]:
                best = (acc, p)
        return best
    grids = STAGE1_FAST if FAST else STAGE1_FULL if FULL else STAGE1
    print(f"   searching {name} ...", flush=True)
    a1, p1 = run(grids[name])
    a2, p2 = run(refine(name, p1))
    acc, p = (a2, p2) if a2 >= a1 else (a1, p1)
    print(f"   {name:12s} stage1 val-acc={a1:.3f} -> stage2 val-acc={a2:.3f}  best={p}")
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()}


def score_rows(preds, yte, target, setting):
    nir = max(yte.mean(), 1 - yte.mean())
    rows = []
    for m, p in preds.items():
        rows.append({"Response": TITLES[target].replace("Response: ", ""), "Hyperparameters": setting, "Model": m,
                     "Accuracy": accuracy_score(yte, p),
                     "Precision": precision_score(yte, p, zero_division=0),
                     "Recall": recall_score(yte, p, zero_division=0),
                     "F1": f1_score(yte, p, zero_division=0),
                     "NoInfoRate": nir, "PredictedPositiveShare": float(np.mean(p))})
    return rows


def fit_predict(params_by_model, Xtr, ytr, Xte, n_feat):
    models = {m: make(m, params_by_model[m], n_feat).fit(Xtr, ytr) for m in MODELS}
    preds = {m: models[m].predict(Xte) for m in MODELS}
    preds["Ensemble"] = (sum(preds[m] for m in MODELS) >= 2).astype(int)      # majority vote
    return models, preds


def short_label(col, labels):
    """'S2502: Estimate!!Occupied housing units!!...!!Black or African American alone' -> compact name."""
    lab = labels.get(col, col)
    if col.startswith(("lag_", "moran_")):
        return lab
    table, _, rest = lab.partition(": ")
    parts = [p for p in rest.replace("Estimate!!", "").split("!!") if p]
    if len(parts) < 2:
        return f"{table}: {rest or col}"[:80]
    return f"{table}: {parts[-1][:38]} ({parts[-2].title()[:28]})"[:85] + " [chg]"


def main():
    df = pd.read_csv(PROC / "dataset.csv", index_col="GEOID")
    feat = [c for c in df.columns if c not in NON_FEATURES]
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))
    print(f"train={len(tr)}  val={len(va)}  test={len(te)}  features={len(feat)}  mode={MODE}")
    labels = pd.read_csv(PROC / "feature_labels.csv", index_col="column")["label"].to_dict()
    all_rows, best_params, cms, xgb_cost = [], {}, [], None

    for target in ("y_cost", "y_income"):
        print(f"\n=== {TITLES[target]} ===")
        Xtr, ytr, Xva, yva, Xte, yte = tr[feat], tr[target], va[feat], va[target], te[feat], te[target]
        print(" grid search (selection by validation accuracy):")
        tuned = {m: two_stage_search(m, Xtr, ytr, Xva, yva, len(feat)) for m in MODELS}
        best_params[target] = tuned
        for setting, params in (("grid search", tuned), ("paper Table 2", PAPER_PARAMS)):
            models, preds = fit_predict(params, Xtr, ytr, Xte, len(feat))
            rows = score_rows(preds, yte, target, setting)
            all_rows += rows
            print(f"\n [{setting}]")
            print(pd.DataFrame(rows)[["Model", "Accuracy", "Precision", "Recall", "NoInfoRate",
                                      "PredictedPositiveShare"]].round(3).to_string(index=False))
            cm = confusion_matrix(yte, preds["RandomForest"], labels=[0, 1])
            cms.append(f"{TITLES[target]} | {setting} | Random Forest confusion matrix "
                       f"(rows=true 0/1, cols=pred 0/1)\n{cm}\n")
            if target == "y_cost" and setting == "grid search":
                xgb_cost = models["XGBoost"]

    res = pd.DataFrame(all_rows).round(3)
    res.to_csv(OUT / "results.csv", index=False)
    (OUT / "best_params.json").write_text(json.dumps(best_params, indent=2))
    (OUT / "confusion_matrices.txt").write_text("\n".join(cms))

    imp = pd.Series(xgb_cost[-1].feature_importances_, index=feat).sort_values(ascending=False)
    top = imp.head(10).rename_axis("column").reset_index(name="importance")
    top["label"] = top["column"].map(lambda c: short_label(c, labels))
    top.to_csv(OUT / "xgb_top_features.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["label"][::-1], top["importance"][::-1], color="maroon")
    ax.set_xlabel("XGBoost feature importance (gain)")
    ax.set_title("Most important predictors - housing-cost response")
    plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=200); plt.close()

    print("\nTop XGBoost predictors (cost response):")
    print(top[["label", "importance"]].round(4).to_string(index=False))
    print(f"\nSaved results to {OUT}")


if __name__ == "__main__":
    main()
