# Shared pieces for validate.py and test_run.py: loading deals.csv, the models, and the metrics
# Author(s): David
# Last Revised: 2026-11-19

import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, module="pgmpy")
from pgmpy.estimators import TreeSearch
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.parameter_estimator import DiscreteBayesianEstimator

SEED = 3346
LABEL = "completion"
# bidders for the same target are not independent deals, so the bootstrap resamples targets, not rows
GROUP = "Target 6-digit CUSIP"

SECTOR_FLAGS = ["banks", "insurance", "utilities", "telecom_media", "pharma_health", "defense", "airlines"]
FEATURES = [
    "hostile", "tender", "go_shop", "financial_buyer", "foreign_acquirer", "toehold",
    "premium", "premium_missing", "log_deal_value", "same_industry",
] + SECTOR_FLAGS
# Termination fees are left out of the main models. A fee only exists once an agreement is signed, and SDC fills it in
# later: 0 of 97 failed unsolicited bids have one, 9 of 16 completed ones do, and among friendly deals coverage is
# 72% for completed vs 34% for failed. Kept only for the leakage check model.
FEE_FEATURES = ["target_fee_missing", "target_fee_pct", "acquiror_fee_missing", "acquiror_fee_pct"]


def load_deals():
    df = pd.read_csv("data/deals.csv", parse_dates=["Date Announced"])
    missing = set(FEATURES + FEE_FEATURES + [LABEL, "split", "mip", "mip_valid", "mip_prob", GROUP]) - set(df.columns)
    assert not missing, f"deals.csv is missing {missing}, rerun build_table.py"
    df["year"] = df["Date Announced"].dt.year
    # friendly at announcement (Bloomberg-corrected attitude), the hard subgroup
    df["friendly"] = df["hostile"].eq(0)
    return df


def market_inputs(df):
    """The market signal as model inputs: log-odds of the implied probability plus three flags.
    Deals without a valid implied probability get log-odds 0 and mip_valid = 0."""
    valid = df["mip_valid"].astype(bool)
    p = df["mip_prob"].clip(0.01, 0.99)
    X = pd.DataFrame(index=df.index)
    X["mip_logit"] = np.log(p / (1 - p)).where(valid, 0.0)
    X["mip_valid"] = valid.astype(int)
    # trading above the offer means the market expects a topping bid
    X["above_offer"] = (df["mip"] > 1).astype(int)
    # trading below the price 4 weeks before means the market doubts the deal
    X["below_prior"] = (df["mip"] < 0).astype(int)
    return X


# Models. Each one has fit(train_df) and predict(df) -> P(completion)

class BaseRate:
    """Every deal gets the training completion rate."""
    def __init__(self):
        self.params = {}

    def fit(self, df):
        self.rate = df[LABEL].mean()
        return self

    def predict(self, df):
        return np.full(len(df), self.rate)


class RawMarket:
    """Implied probability straight from the spread, clipped to [0.01, 0.99].
    Falls back to the training completion rate where it can't be computed."""
    def __init__(self):
        self.params = {}

    def fit(self, df):
        self.rate = df[LABEL].mean()
        return self

    def predict(self, df):
        p = df["mip_prob"].clip(0.01, 0.99)
        return p.where(df["mip_valid"].astype(bool), self.rate).to_numpy()


class LogReg:
    """L2 logistic regression on the market inputs, the deal features, or both.
    Market-only is the calibrated market baseline: it learns how to read the spread."""
    def __init__(self, use_market, use_features, C=1.0, use_fees=False):
        self.use_market, self.use_features, self.use_fees = use_market, use_features, use_fees
        self.params = {"C": C}

    def inputs(self, df):
        parts = []
        if self.use_market:
            parts.append(market_inputs(df))
        if self.use_features:
            parts.append(df[FEATURES].astype(float))
        if self.use_fees:
            parts.append(df[FEE_FEATURES].astype(float))
        return pd.concat(parts, axis=1)

    def fit(self, df):
        X = self.inputs(df)
        self.columns = list(X.columns)
        self.model = make_pipeline(StandardScaler(), LogisticRegression(C=self.params["C"], max_iter=5000))
        self.model.fit(X, df[LABEL])
        return self

    def predict(self, df):
        return self.model.predict_proba(self.inputs(df))[:, 1]

    def coefficients(self):
        """Standardised coefficients: change in log-odds of completion per standard deviation."""
        return pd.Series(self.model[-1].coef_[0], index=self.columns)


class BayesNet:
    """Discrete Bayes net over binned deal features, BDeu prior, exact inference by variable elimination.
    structure="expert": hand-drawn edges (see EXPERT_EDGES).
    structure="tan": tree-augmented naive Bayes, the feature tree learned from training data."""

    # completion depends on the bid attitude, the deal form and regulatory exposure;
    # the market price is read as evidence about completion, and also reacts to the attitude directly
    EXPERT_EDGES = [
        ("hostile", "completion"), ("tender", "completion"),
        ("regulated", "completion"), ("foreign_acquirer", "completion"),
        ("completion", "market"), ("hostile", "market"),
    ]
    TAN_NODES = [
        "hostile", "tender", "financial_buyer", "foreign_acquirer", "regulated",
        "go_shop", "same_industry", "premium_bin", "size_bin", "market",
    ]

    def __init__(self, structure, ess=10):
        self.structure = structure
        self.params = {"ess": ess}

    def discretize(self, df):
        """Bin the inputs. Bin edges come from the training data (set in fit)."""
        D = pd.DataFrame(index=df.index)
        for col in ["hostile", "tender", "financial_buyer", "foreign_acquirer", "go_shop", "same_industry"]:
            D[col] = df[col].astype(int).astype(str)
        D["regulated"] = df[SECTOR_FLAGS].max(axis=1).astype(int).astype(str)
        # market: no valid implied probability, or low / mid / high tercile of the training values
        market = pd.Series(np.digitize(df["mip_prob"].fillna(0), self.mip_edges), index=df.index)
        D["market"] = market.map({0: "low", 1: "mid", 2: "high"}).where(df["mip_valid"].astype(bool), "none")
        D["premium_bin"] = pd.Series(np.digitize(df["premium"], self.premium_edges), index=df.index).astype(str)
        D["size_bin"] = pd.Series(np.digitize(df["log_deal_value"], self.size_edges), index=df.index).astype(str)
        if LABEL in df:
            D[LABEL] = df[LABEL].astype(int).astype(str)
        return D

    def fit(self, df):
        valid_mip = df.loc[df["mip_valid"].astype(bool), "mip_prob"]
        self.mip_edges = valid_mip.quantile([1 / 3, 2 / 3]).to_numpy()
        self.premium_edges = df["premium"].quantile([1 / 3, 2 / 3]).to_numpy()
        self.size_edges = df["log_deal_value"].quantile([1 / 3, 2 / 3]).to_numpy()
        D = self.discretize(df)

        if self.structure == "expert":
            edges = self.EXPERT_EDGES
        else:
            dag = TreeSearch(D[self.TAN_NODES + [LABEL]], n_jobs=1).estimate(
                estimator_type="tan", class_node=LABEL, show_progress=False)
            edges = list(dag.edges())
        self.model = DiscreteBayesianNetwork(edges)
        nodes = list(self.model.nodes())

        # list every state up front so a bin unseen in training doesn't break inference later
        all_states = {"market": ["high", "low", "mid", "none"], "premium_bin": ["0", "1", "2"], "size_bin": ["0", "1", "2"]}
        states = {c: all_states.get(c, ["0", "1"]) for c in nodes}
        self.model.fit(D[nodes], estimator=DiscreteBayesianEstimator(
            state_names=states, prior_type="BDeu", equivalent_sample_size=self.params["ess"]))
        self.evidence_nodes = [c for c in nodes if c != LABEL]
        self.infer = VariableElimination(self.model)
        return self

    def predict(self, df):
        D = self.discretize(df)[self.evidence_nodes]
        cache = {}
        out = np.empty(len(D))
        for i, row in enumerate(D.itertuples(index=False)):
            if row not in cache:
                q = self.infer.query([LABEL], evidence=dict(zip(self.evidence_nodes, row)), show_progress=False)
                cache[row] = q.values[q.state_names[LABEL].index("1")]
            out[i] = cache[row]
        return out


# model name -> builder taking a hyperparameter dict; validate.py tunes these, test_run.py refits them
BUILDERS = {
    "Base rate": lambda p: BaseRate(),
    "Market (raw)": lambda p: RawMarket(),
    "Market (calibrated)": lambda p: LogReg(True, False, **p),
    "LR features only": lambda p: LogReg(False, True, **p),
    "LR market + features": lambda p: LogReg(True, True, **p),
    "Bayes net (expert)": lambda p: BayesNet("expert", **p),
    "Bayes net (TAN)": lambda p: BayesNet("tan", **p),
    # leakage check, not a candidate: the same LR with termination fee fields added
    "LR + fees (leakage check)": lambda p: LogReg(True, True, use_fees=True, **p),
}
LEARNED = ["LR features only", "LR market + features", "Bayes net (expert)", "Bayes net (TAN)"]


# Metrics

def brier(y, p):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(y, p):
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(y, p, bins=10):
    """Expected calibration error: average gap between predicted and actual completion rate over 10 equal-width bins."""
    y, p = np.asarray(y), np.asarray(p)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def scores(y, p):
    return {"n": len(y), "failures": int((1 - np.asarray(y)).sum()), "brier": brier(y, p),
            "log_loss": log_loss(y, p), "auc": auc(y, p), "ece": ece(y, p)}


def cluster_bootstrap(values, groups, reps=2000, seed=SEED):
    """Mean of per-deal values with a 95% interval from resampling targets (all of a target's deals together)."""
    values, groups = np.asarray(values, dtype=float), np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    group_sum, group_n = np.bincount(inv, weights=values), np.bincount(inv)
    draws = np.random.default_rng(seed).integers(0, len(uniq), size=(reps, len(uniq)))
    boot = group_sum[draws].sum(axis=1) / group_n[draws].sum(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def bootstrap_brier(y, p, groups):
    """Brier score with a 95% interval."""
    y, p = np.asarray(y), np.asarray(p)
    return cluster_bootstrap((p - y) ** 2, groups)


def bootstrap_brier_diff(y, p_model, p_ref, groups):
    """Brier(model) - Brier(reference) with a 95% interval. Negative means the model is better.
    Same resampled targets for both, so the interval reflects the paired difference."""
    y, p_model, p_ref = map(np.asarray, (y, p_model, p_ref))
    return cluster_bootstrap((p_model - y) ** 2 - (p_ref - y) ** 2, groups)
