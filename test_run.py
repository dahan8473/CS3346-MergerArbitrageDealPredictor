# Single test run: refit the frozen settings on train + validation (2015-2021) and score test (2022-2024) once.
# Run only after results/frozen.json is committed and tagged "frozen". Nothing gets tuned after this.
# Writes results/test_metrics.csv, test_vs_baselines.csv, test_by_year.csv, completion_by_year.csv,
# lr_coefficients.csv, bn_expert_cpds.csv and test_run.json.
# Per-deal predictions go to data/predictions_test.csv (licensed data, gitignored).
# run: .venv/bin/python test_run.py
# Author(s): David
# Last Revised: 2026-11-19

import itertools
import json
import subprocess
from datetime import datetime
from pathlib import Path
import pandas as pd
from common import (BUILDERS, GROUP, LABEL, bootstrap_brier, bootstrap_brier_diff, load_deals, scores)

frozen = json.loads(Path("results/frozen.json").read_text())
version = subprocess.run(["git", "describe", "--tags", "--always", "--dirty"], capture_output=True, text=True).stdout.strip()
if not version.startswith("frozen"):
    print(f"warning: running from {version}, not the frozen tag")

df = load_deals()
dev = df[df["split"].isin(["train", "val"])]
test = df[df["split"] == "test"].copy()
# Expect dev 761 deals / 99 failures, test 339 deals / 66 failures
print(f"dev {len(dev)} deals / {(1 - dev[LABEL]).sum()} failures, test {len(test)} deals / {(1 - test[LABEL]).sum()} failures")

models, preds = {}, {}
for name, params in frozen["params"].items():
    models[name] = BUILDERS[name](params).fit(dev)
    preds[name] = models[name].predict(test)
    test[f"p_{name}"] = preds[name]

subgroups = {"all": test.index == test.index, "friendly": test["friendly"].to_numpy()}
primary = frozen["primary"]

# scores per model and subgroup, with a 95% interval on Brier
rows = []
for name, p in preds.items():
    for sub, mask in subgroups.items():
        y, g = test.loc[mask, LABEL].to_numpy(), test.loc[mask, GROUP].to_numpy()
        _, lo, hi = bootstrap_brier(y, p[mask], g)
        rows.append({"model": name, "subgroup": sub, **scores(y, p[mask]), "brier_lo": lo, "brier_hi": hi})
metrics = pd.DataFrame(rows)

# paired differences against the two baselines
rows = []
for name, p in preds.items():
    for ref in ["Market (calibrated)", "Base rate"]:
        if name == ref:
            continue
        for sub, mask in subgroups.items():
            y, g = test.loc[mask, LABEL].to_numpy(), test.loc[mask, GROUP].to_numpy()
            diff, lo, hi = bootstrap_brier_diff(y, p[mask], preds[ref][mask], g)
            rows.append({"model": name, "vs": ref, "subgroup": sub, "brier_diff": diff, "lo": lo, "hi": hi,
                         "clear_win": hi < 0})
diffs = pd.DataFrame(rows)

# regime shift: test Brier by year for the main comparison, and completion rate by year across all splits
rows = []
for year, grp in test.groupby("year"):
    for name in ["Base rate", "Market (calibrated)", primary]:
        rows.append({"year": year, "model": name, **scores(grp[LABEL].to_numpy(), grp[f"p_{name}"].to_numpy())})
by_year = pd.DataFrame(rows)
completion_by_year = df.groupby(["year", "split"]).agg(deals=(LABEL, "size"), completion_rate=(LABEL, "mean"),
                                                       hostile_share=("hostile", "mean")).reset_index()

# what the models learned, fitted on 2015-2021
coefs = models["LR market + features"].coefficients().rename("coef").reset_index().rename(columns={"index": "input"})
cpd_rows = []
for node in ["completion", "market"]:
    cpd = models["Bayes net (expert)"].model.get_cpds(node)
    names = cpd.state_names
    parents = cpd.variables[1:]
    for combo in itertools.product(*[names[v] for v in parents]):
        for state in names[node]:
            prob = cpd.get_value(**{node: state, **dict(zip(parents, combo))})
            cpd_rows.append({"node": node, "state": state, **dict(zip(parents, combo)), "prob": prob})
cpds = pd.DataFrame(cpd_rows)

Path("results").mkdir(exist_ok=True)
metrics.round(4).to_csv("results/test_metrics.csv", index=False)
diffs.round(4).to_csv("results/test_vs_baselines.csv", index=False)
by_year.round(4).to_csv("results/test_by_year.csv", index=False)
completion_by_year.round(4).to_csv("results/completion_by_year.csv", index=False)
coefs.round(4).to_csv("results/lr_coefficients.csv", index=False)
cpds.round(4).to_csv("results/bn_expert_cpds.csv", index=False)
test[["SDC Deal No", "year", LABEL, "friendly"] + [f"p_{m}" for m in preds]].to_csv("data/predictions_test.csv", index=False)
Path("results/test_run.json").write_text(json.dumps({
    "version": version, "run_at": datetime.now().isoformat(timespec="seconds"), "primary": primary,
    "dev_deals": len(dev), "test_deals": len(test), "test_failures": int((1 - test[LABEL]).sum()),
    "test_friendly_failures": int((1 - test.loc[test["friendly"], LABEL]).sum()),
}, indent=2))

pd.set_option("display.width", 200)
print(metrics.pivot(index="model", columns="subgroup", values=["brier", "auc"]).loc[list(preds)].round(4))
print(diffs[diffs["vs"] == "Market (calibrated)"].round(4).to_string(index=False))
