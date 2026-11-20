# Decisions

Newest first. Each entry says what was decided and why, for the report's methods section and reproducibility appendix.

## 2026-11-19: Freeze protocol
Hyperparameters were picked on validation only (`validate.py`, `bidding_war.py`). `results/frozen.json` and all code were committed and tagged `frozen` before the test split was scored. `test_run.py` refits the frozen settings on 2015-2021 and scores 2022-2024 once. The test script was dry-run beforehand with validation standing in for test, so that a bug couldn't force a change after the test labels were seen.

## 2026-11-19: Bidding war model, one revision
The first version (rival-strength uncertainty plus board backing) beat the contested base rate on validation (Brier 0.238 vs 0.249). It lost to a one-feature rule, contested completion rate by attitude (0.204), and two parameters sat on their grid edges. Added one chance node for board defences against a hostile winner (poison pill, just-say-no) and widened the grids. Validation Brier became 0.229. Stopped there: with 48 training contests, more structure would be fitting noise. Reported as is.

## 2026-11-19: Termination fees excluded from the main models
A termination fee only exists once a merger agreement is signed, and SDC fills it in later. Evidence:
- 0 of 97 failed unsolicited bids have a target fee recorded; 9 of 16 completed ones do.
- Among friendly deals, coverage is 72% for completed vs 34% for failed, even though fees are set at signing.

Fee fields stay in `deals.csv`. They're used only in the "LR + fees (leakage check)" model, which shows how much they would inflate results.

## 2026-11-19: Attitude at announcement from Bloomberg
SDC's Deal Attitude is the final attitude: a bid that starts unsolicited and is later agreed shows as Friendly. Bloomberg's Nature of Bid keeps the history ("Unsolicited to Friendly"). `build_table.py` matches 606 of 1,100 deals to Bloomberg by target CUSIP, a 30-day window and acquirer name, and takes the first part. That corrects 22 attitudes. The other 494 deals keep SDC's value, so some leak may remain there (limitation).

## 2026-11-19: Labels: SDC status kept
The SDC/Bloomberg conflicts were matching errors, not label errors: Bloomberg rows for a different bidder on the same target (Twitter, Geeknet). Matching on acquirer name leaves one disagreement: Blueknight Energy Partners / Ergon, 2019. SDC's label is kept.

## 2026-11-19: Scope
All cash deals are the main result, with attitude at announcement as a feature. Deals friendly at announcement are reported as a subgroup: the harder, true merger-arbitrage question, with few failures (27 train, 13 validation, 19 test). No 2010-2014 data pull.

## 2026-11-13: Splits and cleaning (Daniel)
- Train 2015-2019, validation 2020-2021, test 2022-2024, by announcement year. This differs from the proposal's 2015-2018 / 2019-2020 / 2021-2024, to give training more years.
- 100% cash, outcome known (Completed or Withdrawn), target price at least $1 four weeks before: 1,100 deals.
- Implied probability is only valid where the offer exceeds the 4-week price by more than 2% of the offer.
- Prediction point is the day after announcement, the price SDC provides. That replaces the proposal's t+5.
