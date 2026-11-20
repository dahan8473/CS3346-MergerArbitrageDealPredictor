# Figures for the report, from the test run outputs. Run after test_run.py and bidding_war.py --test.
# Writes figures/*.png. Plots show aggregates only.
# run: .venv/bin/python figures.py
# Author(s): David
# Last Revised: 2026-11-19

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bidding_war import p_original_wins

INK, MUTED, LIGHT, ACCENT = "#2E3440", "#7B8394", "#D8DEE9", "#BF616A"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
Path("figures").mkdir(exist_ok=True)

frozen = json.loads(Path("results/frozen.json").read_text())
primary = frozen["primary"]
metrics = pd.read_csv("results/test_metrics.csv")
order = ["Base rate", "Market (raw)", "Market (calibrated)", "LR features only", "LR market + features",
         "Bayes net (expert)", "Bayes net (TAN)", "LR + fees (leakage check)"]

# 1. Test Brier with 95% intervals, all deals and friendly deals
fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), sharey=True)
for ax, sub, title in zip(axes, ["all", "friendly"], ["All cash deals", "Friendly at announcement"]):
    m = metrics[metrics["subgroup"] == sub].set_index("model").loc[order]
    colors = [ACCENT if n == primary else (LIGHT if "leakage" in n else MUTED) for n in m.index]
    y = np.arange(len(m))
    ax.barh(y, m["brier"], color=colors, height=0.6)
    ax.errorbar(m["brier"], y, xerr=[m["brier"] - m["brier_lo"], m["brier_hi"] - m["brier"]],
                fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
    ax.set_yticks(y, m.index)
    ax.invert_yaxis()
    ax.set_title(f"{title} (n={int(m['n'].iloc[0])}, {int(m['failures'].iloc[0])} failed)", fontsize=9)
    ax.set_xlabel("Brier score (lower is better)")
fig.savefig("figures/brier_test.png")

# 2. Reliability: predicted vs actual completion rate, equal-count bins
pred = pd.read_csv("data/predictions_test.csv")
fig, ax = plt.subplots(figsize=(3.4, 3.2))
ax.plot([0, 1], [0, 1], color=LIGHT, lw=1)
for name, color, marker in [("Market (raw)", LIGHT, "s"), ("Market (calibrated)", MUTED, "o"), (primary, ACCENT, "D")]:
    bins = pd.qcut(pred[f"p_{name}"].rank(method="first"), 6, labels=False)
    g = pred.groupby(bins).agg(p=(f"p_{name}", "mean"), y=("completion", "mean"))
    ax.plot(g["p"], g["y"], marker=marker, color=color if color != LIGHT else MUTED,
            ls="--" if name == "Market (raw)" else "-", lw=1.2, ms=4, label=name)
ax.set_xlabel("Predicted completion probability")
ax.set_ylabel("Actual completion rate")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.set_title("Calibration on 2022-2024 test deals", fontsize=9)
fig.savefig("figures/reliability_test.png")

# 3. Logistic regression coefficients (market + features, fitted on 2015-2021)
coefs = pd.read_csv("results/lr_coefficients.csv").sort_values("coef")
fig, ax = plt.subplots(figsize=(3.4, 4.2))
ax.barh(coefs["input"], coefs["coef"], color=[ACCENT if c < 0 else MUTED for c in coefs["coef"]], height=0.6)
ax.axvline(0, color=INK, lw=0.6)
ax.set_xlabel("Change in log-odds of completion per s.d.")
ax.set_title("LR market + features", fontsize=9)
fig.savefig("figures/lr_coefficients.png")

# 4. Regime shift: completion rate by announcement year
cy = pd.read_csv("results/completion_by_year.csv")
fig, ax = plt.subplots(figsize=(3.4, 2.4))
shade = {"train": "#F3F5F8", "val": "#E5E9F0", "test": LIGHT}
for split, grp in cy.groupby("split"):
    ax.axvspan(grp["year"].min() - 0.5, grp["year"].max() + 0.5, color=shade[split], lw=0)
    ax.text(grp["year"].mean(), 0.995, split, ha="center", va="top", fontsize=7, color=MUTED)
ax.plot(cy["year"], cy["completion_rate"], color=ACCENT, marker="o", ms=3, lw=1.2)
ax.set_ylim(0.75, 1.0)
ax.set_xticks(cy["year"])
ax.tick_params(axis="x", labelrotation=45)
ax.set_ylabel("Share of deals completed")
fig.savefig("figures/completion_by_year.png")

# 5. Expert Bayes net structure
pos = {"hostile": (0.0, 1.0), "tender": (1.2, 1.0), "regulated": (2.4, 1.0), "foreign_acquirer": (3.8, 1.0),
       "completion": (1.9, 0.0), "market": (0.6, -1.0)}
from common import BayesNet
fig, ax = plt.subplots(figsize=(4.8, 2.6))
for a, b in BayesNet.EXPERT_EDGES:
    ax.annotate("", xy=pos[b], xytext=pos[a], arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1,
                                                               shrinkA=16, shrinkB=16))
for node, (x, y) in pos.items():
    ax.text(x, y, node.replace("_", " "), ha="center", va="center", fontsize=8, color="white" if node == "completion" else INK,
            bbox=dict(boxstyle="round,pad=0.4", fc=ACCENT if node == "completion" else "#ECEFF4", ec=MUTED, lw=0.6))
ax.set_xlim(-0.6, 4.6)
ax.set_ylim(-1.4, 1.4)
ax.axis("off")
fig.savefig("figures/bn_expert.png")

# 6. Bidding war: P(original acquirer wins the bidding) against the initial premium, at the frozen parameters
bw = json.loads(Path("results/bidding_war_test.json").read_text())["params"]
premiums = np.linspace(0.0, 1.0, 101)
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for backed, color, label in [(True, ACCENT, "board backs original bidder"), (False, MUTED, "original bid is hostile")]:
    k = 1.0 if backed else bw["k"]
    ax.step(premiums, [p_original_wins(1 + pr, backed, bw["h"], bw["theta"], bw["b"])[0] * k for pr in premiums],
            where="post", color=color, lw=1.4, label=label)
ax.set_xlabel("Initial premium over unaffected price")
ax.set_ylabel("P(original acquirer prevails)")
ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=7)
fig.savefig("figures/bidding_war.png")

print("wrote", sorted(str(p) for p in Path("figures").glob("*.png")))
