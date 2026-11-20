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

Python 3.11 or newer. Run everything from the repo folder.

Windows:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS / Linux (delete the `pywinpty` line from `requirements.txt` first, it only builds on Windows):
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

With the virtual environment active, run these in order:

```
python build_table.py
python validate.py
python bidding_war.py
python test_run.py
python bidding_war.py --test
python figures.py
```

| Step | What it does | Output |
|---|---|---|
| `build_table.py` | Clean, label, split, build features | `data/deals.csv` |
| `validate.py` | Fit on 2015-2019, tune on 2020-2021 | `results/val_grid.csv`, `val_metrics.csv`, `frozen.json` |
| `bidding_war.py` | Bidding war, validation | `results/bidding_war_val.json` |
| `test_run.py` | Refit on 2015-2021, score 2022-2024 once | `results/test_*.csv`, `test_run.json` |
| `bidding_war.py --test` | Bidding war, test | `results/bidding_war_test.json` |
| `figures.py` | Report figures | `figures/*.png` |

The whole run takes under a minute.

## Check it worked

The scripts print these numbers. If yours match, your setup reproduces ours.

| Script | Should print |
|---|---|
| `build_table.py` | `10 files, 1100 rows`, `606 deals matched to Bloomberg`, `22 attitudes corrected` |
| `validate.py` | `primary model for the test run: Bayes net (expert)` |
| `test_run.py` | `dev 761 deals / 99 failures, test 339 deals / 66 failures`, and Bayes net (expert) Brier `0.0778` on all deals |
| `bidding_war.py --test` | `contested deals: fit 65, test 23` and `"expectimax": 0.2006...` under brier |

`build_table.py` also stops with an error if the data looks wrong (duplicate deals, missing values, leaked columns).

After a full run, `git status` should only show `results/test_run.json` as changed (it records the run time). Every other result and figure comes out byte-identical.

## Don't change the models

The test years (2022-2024) were scored once, from the git tag `frozen`. Rerunning the scripts as they are is fine: they give the same numbers. Changing a model, feature or setting now means choosing it after seeing test results. Any such change has to be reported as a separate follow-up experiment, not as the main result.

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

## Results

Test split, 2022-2024, scored once from the `frozen` tag. Models refit on 2015-2021.

| Model | Brier, all (339 deals, 66 failed) | Brier, friendly (298 deals, 26 failed) | AUC, all |
|---|---|---|---|
| Base rate | 0.161 | 0.082 | 0.50 |
| Market (raw) | 0.159 | 0.134 | 0.73 |
| Market (calibrated) | 0.139 | 0.082 | 0.78 |
| LR features only | 0.077 | 0.079 | 0.88 |
| LR market + features | 0.077 | 0.079 | 0.88 |
| **Bayes net (expert)**, primary | **0.078** | **0.078** | 0.83 |
| Bayes net (TAN) | 0.074 | 0.080 | 0.85 |
| LR + fees (leakage check) | 0.069 | 0.071 | 0.93 |

- **All cash deals: every learned model clearly beats the calibrated market.** The primary model cuts Brier by 0.062 (95% interval 0.038 to 0.086, resampling targets), and wins in each test year.
- **Most of the gain comes from the bid's attitude at announcement.** The spread misreads unsolicited bids: the target's price reflects the chance of *some* takeover, not of this bidder closing.
- **Friendly deals: nothing clearly beats the calibrated market or the base rate.** All differences are within about 0.01, and every 95% interval crosses zero. On the true merger-arbitrage question, the market and the base rate are hard to beat with these features.
- **The raw spread is worse than the base rate on friendly deals.** Spreads also pay time value and a risk premium. Calibrating on training data fixes that.
- **Leakage check:** adding termination fee fields produces a "clear win" on friendly deals (-0.011, interval -0.016 to -0.005). That comes from how SDC records fees, which is why they're excluded.
- **Regime shift:** completion fell from about 88% of deals (2015-2019) to about 80% (2022-2024). See `figures/completion_by_year.png`.
- **Bidding war, 23 contested test deals:** the expectimax model scores Brier 0.201, against 0.252 for the contested base rate and 0.165 for completion rate by attitude. The tree reproduces the real effects (a high first premium deters rivals, board backing matters) but doesn't beat the one-feature rule.
