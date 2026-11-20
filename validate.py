# Fit every model on train (2015-2019), score on validation (2020-2021), pick hyperparameters by validation Brier.
# Writes results/val_grid.csv (every setting), results/val_metrics.csv (chosen settings) and results/frozen.json.
# The test split is never touched here.
# run: .venv/bin/python validate.py
# Author(s): David
# Last Revised: 2026-11-19

import json
from pathlib import Path
import pandas as pd
from common import BUILDERS, LABEL, LEARNED, load_deals, scores

C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10]
ESS_GRID = [1, 5, 10, 20, 50]

# hyperparameter grid for each model in common.BUILDERS
GRIDS = {
    "Base rate": [{}],
    "Market (raw)": [{}],
    "Market (calibrated)": [{"C": c} for c in C_GRID],
    "LR features only": [{"C": c} for c in C_GRID],
    "LR market + features": [{"C": c} for c in C_GRID],
    "Bayes net (expert)": [{"ess": e} for e in ESS_GRID],
    "Bayes net (TAN)": [{"ess": e} for e in ESS_GRID],
    "LR + fees (leakage check)": [{"C": c} for c in C_GRID],
}

df = load_deals()
train = df[df["split"] == "train"]
val = df[df["split"] == "val"]
# Expect train 582 deals / 69 failures, val 179 deals / 30 failures
print(f"train {len(train)} deals / {(1 - train[LABEL]).sum()} failures, val {len(val)} deals / {(1 - val[LABEL]).sum()} failures")

rows = []
for name, grid in GRIDS.items():
    for params in grid:
        p = BUILDERS[name](params).fit(train).predict(val)
        for subgroup, mask in [("all", val.index == val.index), ("friendly", val["friendly"])]:
            rows.append({"model": name, "params": json.dumps(params), "subgroup": subgroup,
                         **scores(val.loc[mask, LABEL].to_numpy(), p[mask])})
grid = pd.DataFrame(rows)

# pick each model's setting by Brier on all validation deals
best = (grid[grid["subgroup"] == "all"].sort_values("brier").drop_duplicates("model").set_index("model"))
chosen = grid[grid.apply(lambda r: r["params"] == best.loc[r["model"], "params"], axis=1)]

Path("results").mkdir(exist_ok=True)
grid.round(4).to_csv("results/val_grid.csv", index=False)
chosen.round(4).to_csv("results/val_metrics.csv", index=False)

primary = best.loc[LEARNED, "brier"].idxmin()
frozen = {"params": {m: json.loads(best.loc[m, "params"]) for m in GRIDS}, "primary": primary}
Path("results/frozen.json").write_text(json.dumps(frozen, indent=2))

pd.set_option("display.width", 200)
print(chosen.pivot(index="model", columns="subgroup", values=["brier", "auc"]).loc[list(GRIDS)].round(4))
print("chosen settings:", frozen["params"])
print("primary model for the test run:", primary)
