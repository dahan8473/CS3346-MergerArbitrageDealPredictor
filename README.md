# SE3346-MergerArbitrageDealPredictor

CS3346A Group 32. When a company agrees to be bought for cash, its stock trades a little below the offer. That gap (the merger arbitrage spread) implies a market probability that the deal closes. We ask whether deal features predict completion better than the spread does, using all-cash acquisitions of US public companies announced 2015-2024.

## Data

Licensed LSEG SDC Platinum and Bloomberg exports pulled through the Ivey Business Library. Course use only, no commercial use, and never committed: `data/` is gitignored. Get the files from the team drive and place them at:

```
data/raw/SDC Platinum/2015.xlsx ... 2024.xlsx
data/raw/Bloomberg/Bloomberg Export.xlsx
data/sic1987_all.csv        public, github.com/danielruss/codingsystems
```

`download_log.csv` records how each file was pulled.

## Setup

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip
```

`pywinpty` only builds on Windows. On macOS or Linux, drop that line.

## Run

| Step | Command | Output |
|---|---|---|
| 1. Clean, label, split, features | `python build_table.py` | `data/deals.csv` |
| 2. Tune on validation (2020-2021) | `python validate.py` | `results/val_grid.csv`, `val_metrics.csv`, `frozen.json` |
| 3. Bidding war, validation | `python bidding_war.py` | `results/bidding_war_val.json` |
| 4. Single test run (2022-2024) | `python test_run.py` | `results/test_*.csv`, `test_run.json` |
| 5. Bidding war, test | `python bidding_war.py --test` | `results/bidding_war_test.json` |
| 6. Figures | `python figures.py` | `figures/*.png` |

Steps 4 and 5 were run once, from the git tag `frozen`. Nothing was tuned after that.

## Models

| Model | What it is |
|---|---|
| Base rate | Training completion rate for every deal |
| Market (raw) | Implied probability `(P_day1 - P_4wk) / (offer - P_4wk)`, clipped to [0.01, 0.99] |
| Market (calibrated) | Logistic regression on the implied probability's log-odds and flags. The bar to beat |
| LR features only / market + features | L2 logistic regression on deal features, with and without the market inputs |
| Bayes net (expert) | Hand-drawn structure (`figures/bn_expert.png`), BDeu prior, variable elimination |
| Bayes net (TAN) | Tree-augmented naive Bayes, feature tree learned from training data |
| LR + fees | Leakage check only: termination fee fields added back |
| Bidding war | Expectimax search over a bidding contest, for deals that drew a rival bid (`bidding_war.py`) |

Design choices and the reasons for them are in `DECISIONS.md`.
