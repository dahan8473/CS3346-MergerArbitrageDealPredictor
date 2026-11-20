# Bidding war as a search tree (course topic: adversarial search, expectimax with chance nodes)
#
# When a rival bidder shows up, does the original acquirer still close?
# Prices are in units of the target's unaffected price (4 weeks before), so the original offer is 1 + premium.
#   - The original acquirer A values the target at offer * (1 + h): it has some room to raise.
#   - The rival R values it at 1 + s, and A doesn't know s, so the root is a chance node over rival types.
#     Strong rivals are more likely when theta is high.
#   - The board backs the bidder it signed with: a friendly A makes R top by an extra margin b,
#     a hostile A has to top a rival (often a white knight) by b.
#   - The bidder who is behind tops by one step or walks. Each bidder maximises its own surplus (value minus price),
#     so every decision node is a max node for the player moving.
#   - A hostile A that wins the bidding still faces the board's defences (poison pill, just-say-no):
#     a chance node lets it through with probability k. A friendly A has the board's agreement already.
#   - The winner's deal then passes a regulator chance node (closing rate of uncontested training deals).
# h, theta, b and k are fitted by grid search on contested training deals, then frozen.
#
# Contested deals are only known after announcement (Competing Offer Flag), so this is a separate, conditional model:
# "given a contest, who closes?". It never feeds the main models. The flag is read from the raw SDC files here.
#
# run: .venv/bin/python bidding_war.py          fit on train, report validation
#      .venv/bin/python bidding_war.py --test   fit on train + validation, report test (after the freeze only)
# Author(s): David
# Last Revised: 2026-11-19

import itertools
import json
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
from common import LABEL, SECTOR_FLAGS, brier, load_deals, log_loss

STEP = 0.05                                   # minimum raise, 5% of the unaffected price
RIVAL_TYPES = [0.2, 0.4, 0.6, 0.8, 1.0]       # rival's value premium over the unaffected price
H_GRID = np.round(np.arange(0.0, 0.51, 0.05), 2)
THETA_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
B_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6]
K_GRID = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def rival_weights(theta):
    """Chance node probabilities over RIVAL_TYPES: weight theta^k, so theta > 1 favours strong rivals."""
    w = np.array([theta ** k for k in range(len(RIVAL_TYPES))])
    return w / w.sum()


def play(offer, value_a, value_r, a_backed, b):
    """Perfect-information bidding once nature has picked the rival's value.
    Returns (winner, price). Minimax-style search over (standing price, leader) with memoisation."""
    # the board-disfavoured bidder needs an extra margin b to take the lead
    margin = {"A": 0.0 if a_backed else b, "R": b if a_backed else 0.0}
    value = {"A": value_a, "R": value_r}

    @lru_cache(maxsize=None)
    def node(price, leader):
        mover = "R" if leader == "A" else "A"
        bid = round(price + STEP + margin[mover], 6)
        walk = (leader, price)                       # leader wins at the standing price
        if bid > value[mover]:                       # topping would cost more than the target is worth to the mover
            return walk
        top = node(bid, mover)                       # what happens after the mover tops
        surplus_if_top = value[mover] - top[1] if top[0] == mover else 0.0
        return top if surplus_if_top > 0 else walk   # max node: the mover keeps the better of top / walk

    return node(round(offer, 6), "A")


def p_original_wins(offer, a_backed, h, theta, b):
    """Expectimax value at the root: average the game's outcome over the rival-type chance node.
    Returns (P(A wins), expected final price if A wins)."""
    weights = rival_weights(theta)
    win, price_if_win = 0.0, 0.0
    for w, s in zip(weights, RIVAL_TYPES):
        winner, price = play(offer, offer * (1 + h), 1 + s, a_backed, b)
        if winner == "A":
            win += w
            price_if_win += w * price
    return win, (price_if_win / win if win > 0 else offer)


def predict(deals, h, theta, b, k, close_rate):
    """P(original deal completes) = P(A wins the bidding) x P(board defences let a hostile A through)
    x P(regulator lets the winner close)."""
    out = []
    for offer, backed, regulated in zip(1 + deals["premium"], deals["hostile"].eq(0), deals["regulated"]):
        win, _ = p_original_wins(offer, backed, h, theta, b)
        out.append(win * (1.0 if backed else k) * close_rate[regulated])
    return np.clip(np.array(out), 0.01, 0.99)


def contested_deals():
    """Deals from deals.csv that drew a competing offer. Flag and final price change come from the raw SDC files."""
    raw = pd.concat([pd.read_excel(f, header=2) for f in sorted(Path("data/raw/SDC Platinum").glob("*.xlsx"))])
    raw = raw.dropna(subset=["SDC Deal No"])[["SDC Deal No", "Competing Offer Flag",
                                              "Percentage Change Final Offer Price to Initial Offer Price"]]
    df = load_deals().merge(raw, on="SDC Deal No")
    df["regulated"] = df[SECTOR_FLAGS].max(axis=1).astype(int)
    df["contested"] = df["Competing Offer Flag"].astype(str).str.contains("True")
    df["final_raise"] = pd.to_numeric(df["Percentage Change Final Offer Price to Initial Offer Price"], errors="coerce") / 100
    return df


def explain(offer=1.30, a_backed=True, h=0.2, theta=1.0, b=0.1):
    """Text trace of the tree for one stylised deal, for the demo."""
    lines = [f"Original offer {offer:.2f}x unaffected price, A values the target at {offer * (1 + h):.2f}x, "
             f"board backs {'A' if a_backed else 'the rival'} (margin {b:.2f})"]
    for w, s in zip(rival_weights(theta), RIVAL_TYPES):
        winner, price = play(offer, offer * (1 + h), 1 + s, a_backed, b)
        lines.append(f"  chance p={w:.2f}: rival values it at {1 + s:.2f}x -> {winner} wins at {price:.2f}x")
    win, price = p_original_wins(offer, a_backed, h, theta, b)
    lines.append(f"  expectimax: P(A wins) = {win:.2f}, expected price if A wins = {price:.2f}x")
    return "\n".join(lines)


if __name__ == "__main__":
    test_mode = "--test" in sys.argv
    df = contested_deals()
    fit_splits, eval_split = (["train", "val"], "test") if test_mode else (["train"], "val")
    fit_all = df[df["split"].isin(fit_splits)]
    fit = fit_all[fit_all["contested"]]
    ev = df[(df["split"] == eval_split) & df["contested"]]
    # Expect 48 contested training deals, 17 validation, 23 test
    print(f"contested deals: fit {len(fit)}, {eval_split} {len(ev)}")

    # regulator chance node: closing rate of friendly uncontested deals, by regulated sector
    calm = fit_all[~fit_all["contested"] & fit_all["hostile"].eq(0)]
    close_rate = calm.groupby("regulated")[LABEL].mean().to_dict()

    # grid search for h, theta, b, k by log-likelihood on the contested fitting deals
    best = min(itertools.product(H_GRID, THETA_GRID, B_GRID, K_GRID),
               key=lambda hp: log_loss(fit[LABEL], predict(fit, *hp, close_rate)))
    h, theta, b, k = best
    p = predict(ev, h, theta, b, k, close_rate)

    # comparators: contested base rate, and contested completion rate by attitude (the obvious one-feature rule)
    base = np.full(len(ev), fit[LABEL].mean())
    by_attitude = ev["hostile"].map(fit.groupby("hostile")[LABEL].mean()).to_numpy()
    y = ev[LABEL].to_numpy()
    metrics = {
        "mode": "test" if test_mode else "validation", "fit_deals": len(fit), "eval_deals": len(ev),
        "eval_completion_rate": float(y.mean()),
        "params": {"h": float(h), "theta": float(theta), "b": float(b), "k": float(k),
                   "close_rate": {str(k): round(v, 4) for k, v in close_rate.items()}},
        "brier": {"expectimax": brier(y, p), "base rate": brier(y, base), "attitude rule": brier(y, by_attitude)},
        "log_loss": {"expectimax": log_loss(y, p), "base rate": log_loss(y, base), "attitude rule": log_loss(y, by_attitude)},
    }

    # secondary check: predicted vs actual raise for deals the original acquirer won
    won = ev[ev[LABEL].eq(1) & ev["final_raise"].notna()]
    if len(won):
        pred_raise = [p_original_wins(1 + pr, bk, h, theta, b)[1] / (1 + pr) - 1
                      for pr, bk in zip(won["premium"], won["hostile"].eq(0))]
        naive = fit.loc[fit[LABEL].eq(1), "final_raise"].median()
        metrics["raise_mae"] = {"expectimax": float(np.mean(np.abs(np.array(pred_raise) - won["final_raise"]))),
                                "training median": float(np.mean(np.abs(naive - won["final_raise"]))), "n": len(won)}

    Path("results").mkdir(exist_ok=True)
    out = Path(f"results/bidding_war_{'test' if test_mode else 'val'}.json")
    out.write_text(json.dumps(metrics, indent=2))
    Path("results/bidding_war_example.txt").write_text(explain(h=h, theta=theta, b=b) + "\n\n" +
                                                       explain(a_backed=False, h=h, theta=theta, b=b) + "\n")
    print(json.dumps(metrics, indent=2))
    print(explain(h=h, theta=theta, b=b))
