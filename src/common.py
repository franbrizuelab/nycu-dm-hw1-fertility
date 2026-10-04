"""Shared constants, feature definitions, folds and metrics.

Every experiment imports from here so that all methods share the same
data split, the same folds and the same metrics (spec requirement).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RES = ROOT / "results"
FIG = RES / "figures"
TAB = RES / "tables"
for d in (FIG, TAB):
    d.mkdir(parents=True, exist_ok=True)

HORIZON = 10                       # years ahead
START_YEARS = list(range(1955, 2011, 5))
TRAIN_MAX_T = 1995                 # training rows: t <= 1995 (targets <= 2005)
GAP_YEARS = [2000]                 # dropped: target 2010 is unknown to a planner standing in 2005
TEST_YEARS = [2005, 2010]          # final, once-only evaluation (targets 2015, 2020)
VAL_YEARS = [1980, 1985, 1990, 1995]  # rolling-origin CV: validate at v, train on t <= v - 10
THRESHOLD = 0.2                    # children per woman, see Section 6 of the paper
SEED = 0


# ----------------------------------------------------------------------------- data
def load_panel():
    return pd.read_csv(PROC / "panel.csv")


def train_panel():
    return load_panel().query("split == 'train'").reset_index(drop=True)


# ----------------------------------------------------------------------------- features
REGIONS = ["Advanced Economies", "East Asia and the Pacific", "Europe and Central Asia",
           "Latin America and the Caribbean", "Middle East and North Africa", "South Asia",
           "Sub-Saharan Africa"]
REGION_COLS = [f"reg_{r.split()[0].lower()}" for r in REGIONS[1:]]  # Advanced Economies = reference

RAW_COLS = ["tfr", "edu_f1524", "edu_f2564", "edu_m1524", "edu_m2564", "q5", "gdppc", "urban"] + REGION_COLS
STAGE_KNOT = 5.5  # TFR above which a country is still pre-transition (Section 4, Fig. 1d); CV-checked


def add_features(df, knot=None):
    """Return a copy of df with every candidate feature as a column."""
    X = df.copy()
    for r, col in zip(REGIONS[1:], REGION_COLS):
        X[col] = (X["region"] == r).astype(float)
    # Momentum (lag) features: recent change in fertility.
    X["d5"] = X["tfr"] - X["tfr_l5"]
    # Curvature: decline is slow at both ends of the transition and fast in the middle.
    X["tfr2"] = X["tfr"] ** 2
    # Stage-gated education: schooling of the cohort entering childbearing ages,
    # switched on only while fertility is still high (pre/early transition).
    X["stage"] = np.maximum(0.0, X["tfr"] - (STAGE_KNOT if knot is None else knot))
    X["edu_x_stage"] = X["edu_f1524"] * X["stage"]
    # Gender recoding of schooling (removes the female/male collinearity).
    X["edu_avg1524"] = (X["edu_f1524"] + X["edu_m1524"]) / 2
    X["edu_gap1524"] = X["edu_f1524"] - X["edu_m1524"]
    # Log transforms for right-skewed quantities.
    X["log_q5"] = np.log(X["q5"])
    X["log_gdppc"] = np.log(X["gdppc"])
    return X


# Feature groups of the final model (Set C); used for leave-one-group-out ablation.
GROUPS = {
    "level": ["tfr"],
    "momentum": ["d5"],
    "curvature": ["tfr2"],
    "education_stage": ["edu_f1524", "edu_x_stage", "stage"],
    "development_logs": ["log_q5", "log_gdppc", "urban"],
    "region": REGION_COLS,
}
SET_C = [c for g in GROUPS.values() for c in g]


# ----------------------------------------------------------------------------- folds
def make_folds(df):
    """Rolling-origin folds with a 10-year embargo.

    For validation start year v, training rows are those with t <= v - 10, so every
    training target (year t + 10) is already observed at the forecast date v.
    """
    folds = []
    for v in VAL_YEARS:
        tr = np.flatnonzero(df["t"].values <= v - HORIZON)
        va = np.flatnonzero(df["t"].values == v)
        assert (df["t"].values[tr] + HORIZON).max() <= v
        folds.append((v, tr, va))
    return folds


def save_folds(df, path=RES / "folds.json"):
    out = {str(v): {"train": df.loc[tr, ["iso3", "t"]].values.tolist(),
                    "val": df.loc[va, ["iso3", "t"]].values.tolist()}
           for v, tr, va in make_folds(df)}
    path.write_text(json.dumps(out))


def load_folds(df, path=RES / "folds.json"):
    """Re-create fold indices for df from the saved (iso3, t) keys, so all methods share them."""
    saved = json.loads(path.read_text())
    key = {(r.iso3, int(r.t)): i for i, r in enumerate(df.itertuples())}
    return [(int(v), np.array([key[(a, int(b))] for a, b in f["train"]]),
             np.array([key[(a, int(b))] for a, b in f["val"]])) for v, f in saved.items()]


# ----------------------------------------------------------------------------- metrics
def metrics(y, p):
    e = np.asarray(y) - np.asarray(p)
    ss = ((np.asarray(y) - np.mean(y)) ** 2).sum()
    return {"MAE": np.abs(e).mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "R2": 1 - (e ** 2).sum() / ss, "Hit": (np.abs(e) < THRESHOLD).mean()}


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")
