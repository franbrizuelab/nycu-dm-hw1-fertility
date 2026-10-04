"""Practical use of the frozen forecast (analysis only; the model is not changed).

1. Benchmark against the UN's own projections made at the same time:
   WPP 2008 (for the 2005 origin, targets 2015) and WPP 2010 (2010 origin, targets 2020).
2. The planner's decision: which UN scenario (low / medium / high) to plan for.
3. 80% forecast ranges from cross-validation residuals, with test coverage.
4. Errors translated into births per year.
"""
import tarfile

import json

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import RAW, RES, SEED, TAB, THRESHOLD, add_features, load_panel

rng = np.random.default_rng(SEED)
OFFSET = 0.4  # UN high/low variants: medium +/- ~0.4 children at a 5-10 year horizon (WPP 2010 files: 0.40-0.45)


def read_tar_table(tar, member):
    with tarfile.open(RAW / tar) as tf:
        return pd.read_csv(tf.extractfile(member), sep="\t")


def un_point(df, year):
    """UN values are 5-year period averages; the single year `year` lies between the two
    periods centred 2.5 years either side of it, so take their mean."""
    return (df[f"{year - 5}-{year}"] + df[f"{year}-{year + 5}"]) / 2


def cluster_boot(df, d, n=2000):
    g = pd.DataFrame({"iso3": df.iso3.values, "d": np.asarray(d)}).groupby("iso3")["d"].agg(["sum", "count"])
    keys = np.arange(len(g))
    vals = [g.iloc[p]["sum"].sum() / g.iloc[p]["count"].sum() for p in (rng.choice(keys, len(keys)) for _ in range(n))]
    return float(np.mean(d)), *np.percentile(vals, [2.5, 97.5])


def vintage_level(df, year):
    """TFR at `year` as the UN saw it then: its last two 5-year period estimates ending at `year`,
    extrapolated half a period forward (period centres are 2.5 years before/after the boundaries)."""
    a, b = df[f"{year - 5}-{year}"], df[f"{year - 10}-{year - 5}"]
    return pd.Series((a + 0.5 * (a - b)).values, index=df.country_code)


def realtime_predictions(te, est08, est10):
    """Apply the frozen model to the fertility inputs available at the forecast date.

    The model is refit exactly as in 06 (same rows, features, alpha); only TFR(t) and TFR(t-5)
    of the test rows are replaced by the estimates of the WPP revision closest to the forecast date
    (WPP 2008 for t=2005, WPP 2010 for t=2010). Schooling keeps the Barro-Lee v3 values (no vintage).
    """
    cfg = json.loads((RES / "final_config.json").read_text())
    panel = add_features(load_panel(), cfg["knot"])
    tr = panel[panel.split == "train"]
    model = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                          Ridge(cfg["alpha"])).fit(tr[cfg["features"]], tr.y)
    rows = panel[panel.split == "test"].set_index(["iso3", "t"]).loc[list(zip(te.iso3, te.t))].reset_index()
    for t, est in ((2005, est08), (2010, est10)):
        m = rows.t == t
        rows.loc[m, "tfr"] = rows.loc[m, "iso3"].map(te.drop_duplicates("iso3").set_index("iso3").locid).map(
            vintage_level(est, t)).values
        rows.loc[m, "tfr_l5"] = rows.loc[m, "iso3"].map(te.drop_duplicates("iso3").set_index("iso3").locid).map(
            vintage_level(est, t - 5)).values
    rows = add_features(rows.drop(columns=["d5", "tfr2", "stage", "edu_x_stage"], errors="ignore"), cfg["knot"])
    ok = rows[cfg["features"]].notna().all(axis=1).values
    out = np.full(len(rows), np.nan)
    out[ok] = model.predict(rows.loc[ok, cfg["features"]])
    return out, rows.tfr.values


def scenario(x, med):
    """Which UN path a value is closest to: low (-1), medium (0) or high (+1)."""
    return np.where(x < med - OFFSET / 2, -1, np.where(x > med + OFFSET / 2, 1, 0))


def main():
    te = pd.read_csv(TAB / "test_predictions.csv")
    wpp = pd.read_csv(RAW / "WPP2024_Demographic_Indicators_Medium.csv.gz",
                      usecols=["ISO3_code", "LocID", "Time", "TFR", "Births"], low_memory=False)
    loc = wpp.dropna(subset=["ISO3_code"]).drop_duplicates("ISO3_code").set_index("ISO3_code")["LocID"]
    te["locid"] = te.iso3.map(loc)
    log = []

    # ------------------------------------------------------------ 1. UN projections made at the forecast date
    w08 = read_tar_table("wpp2008_1.0-1.tar.gz", "wpp2008/data/tfr.txt")
    w10 = read_tar_table("wpp2010_1.2-0.tar.gz", "wpp2010/data/tfrprojMed.txt")
    w10h = read_tar_table("wpp2010_1.2-0.tar.gz", "wpp2010/data/tfrprojHigh.txt")
    un08 = pd.Series(un_point(w08, 2015).values, index=w08.country_code)
    un10 = pd.Series(un_point(w10, 2020).values, index=w10.country_code)
    off10 = (un_point(w10h, 2020) - un_point(w10, 2020)).round(2).value_counts()
    log.append(f"WPP 2010 high-minus-medium offset at 2020: {off10.head(3).to_dict()}")
    te["un"] = np.where(te.t == 2005, te.locid.map(un08), te.locid.map(un10))
    # Sudan was split in 2011, so the old 'Sudan' series is not comparable with today's.
    te.loc[te.iso3 == "SDN", "un"] = np.nan
    est08 = w08  # WPP 2008 table holds estimates up to 2005-2010 and medium projections after
    est10 = read_tar_table("wpp2010_1.2-0.tar.gz", "wpp2010/data/tfr.txt")
    te["pred_rt"], te["tfr_vintage"] = realtime_predictions(te, est08, est10)
    rev = (te.tfr_vintage - te.tfr).abs()
    log.append(f"start-level revision |vintage TFR(t) - WPP 2024 TFR(t)|: mean {rev.mean():.3f}; "
               f"TFR>=4: {rev[te.tfr >= 4].mean():.3f}; TFR<2.5: {rev[te.tfr < 2.5].mean():.3f}")
    pd.DataFrame([{"rev_all": rev.mean(), "rev_high": rev[te.tfr >= 4].mean(), "rev_low": rev[te.tfr < 2.5].mean()}]
                 ).to_csv(TAB / "vintage_revision.csv", index=False)
    cmp_ = te.dropna(subset=["un", "pred_rt"]).copy()
    cmp_["avg"] = (cmp_.pred_rt + cmp_.un) / 2
    log.append(f"UN comparison rows: {len(cmp_)} of {len(te)} (missing: "
               f"{sorted(te.loc[te.un.isna(), 'country'] + ' ' + te.loc[te.un.isna(), 't'].astype(str))})")
    rows = []
    for lab, sub in (("2005 origin (vs WPP 2008)", cmp_[cmp_.t == 2005]), ("2010 origin (vs WPP 2010)", cmp_[cmp_.t == 2010]),
                     ("both", cmp_)):
        e = {k: sub.y - sub[c] for k, c in (("ours", "pred"), ("rt", "pred_rt"), ("un", "un"), ("avg", "avg"),
                                            ("gbm", "pred_gbm"))}
        g = cluster_boot(sub, e["rt"].abs() - e["un"].abs())
        ga = cluster_boot(sub, e["avg"].abs() - e["un"].abs())
        rows.append({"rows": lab, "n": len(sub), "MAE_ours": e["ours"].abs().mean(), "MAE_rt": e["rt"].abs().mean(),
                     "MAE_un": e["un"].abs().mean(),
                     "MAE_avg": e["avg"].abs().mean(), "MAE_gbm": e["gbm"].abs().mean(),
                     "bias_ours": e["ours"].mean(), "bias_rt": e["rt"].mean(), "bias_un": e["un"].mean(),
                     "gap_rt_un": g[0], "gap_lo": g[1], "gap_hi": g[2],
                     "gap_avg_un": ga[0], "gapavg_lo": ga[1], "gapavg_hi": ga[2],
                     "hit_ours": (e["ours"].abs() < THRESHOLD).mean(), "hit_rt": (e["rt"].abs() < THRESHOLD).mean(),
                     "hit_un": (e["un"].abs() < THRESHOLD).mean(),
                     "hit_avg": (e["avg"].abs() < THRESHOLD).mean()})
    un_tab = pd.DataFrame(rows)
    un_tab.to_csv(TAB / "un_comparison.csv", index=False)
    log.append("ours (revised inputs), ours real-time (vintage inputs), UN (bias = actual - forecast; "
               "gap = MAE(real-time ours) - MAE(UN), country-cluster bootstrap):\n"
               + un_tab.round(3).to_string(index=False))
    cmp_["stage"] = pd.cut(cmp_.tfr, [0, 2.5, 4, 99], right=False, labels=["<2.5", "2.5-4", ">=4"])
    st = cmp_.groupby("stage", observed=True).apply(
        lambda s: pd.Series({"n": len(s), "MAE_ours": (s.y - s.pred).abs().mean(), "MAE_rt": (s.y - s.pred_rt).abs().mean(),
                             "MAE_un": (s.y - s.un).abs().mean(),
                             "MAE_avg": (s.y - s.avg).abs().mean()}), include_groups=False)
    st.to_csv(TAB / "un_comparison_by_stage.csv")
    log.append("by stage:\n" + st.round(3).to_string())

    # ------------------------------------------------------------ 2. the planner's decision
    truth = scenario(cmp_.y, cmp_.un)
    pol = {"always UN medium": np.zeros(len(cmp_), int), "our forecast": scenario(cmp_.pred_rt, cmp_.un),
           "average of ours and UN": scenario(cmp_.avg, cmp_.un),
           "our forecast, revised inputs (hindsight)": scenario(cmp_.pred, cmp_.un)}
    dec = []
    for name, choice in pol.items():
        right = choice == truth
        dec.append({"policy": name, "accuracy": right.mean(),
                    "acc_2005": right[cmp_.t.values == 2005].mean(), "acc_2010": right[cmp_.t.values == 2010].mean(),
                    "acc_low": right[(cmp_.tfr < 2.5).values].mean(), "acc_high": right[(cmp_.tfr >= 2.5).values].mean(),
                    "wrong_by_two": (np.abs(choice - truth) == 2).mean()})
    dec = pd.DataFrame(dec)
    dec.to_csv(TAB / "scenario_decision.csv", index=False)
    dist = pd.Series(truth).map({-1: "low", 0: "medium", 1: "high"}).value_counts().to_dict()
    log.append(f"best UN scenario in hindsight: {dist}")
    log.append("scenario choice accuracy:\n" + dec.round(3).to_string(index=False))
    conf = pd.crosstab(pd.Series(truth, name="best"), pd.Series(pol["our forecast"], name="ours"))
    log.append(f"confusion (rows: best in hindsight, cols: chosen from our forecast):\n{conf}")
    # McNemar-style exact test of ours vs always-medium on discordant pairs
    a_right = pol["our forecast"] == truth
    b_right = pol["always UN medium"] == truth
    n01, n10 = int((~a_right & b_right).sum()), int((a_right & ~b_right).sum())
    from scipy.stats import binomtest
    p_mc = binomtest(n10, n01 + n10, 0.5).pvalue if n01 + n10 else np.nan
    log.append(f"ours right & medium wrong: {n10}; medium right & ours wrong: {n01}; exact McNemar p={p_mc:.4f}")
    pd.DataFrame([{"n10": n10, "n01": n01, "p": p_mc}]).to_csv(TAB / "scenario_mcnemar.csv", index=False)

    # ------------------------------------------------------------ 3. 80% forecast ranges from CV residuals
    oof = pd.read_csv(TAB / "cv_oof_predictions.csv")
    panel = pd.read_csv(TAB.parent.parent / "data" / "processed" / "panel.csv")[["iso3", "t", "tfr"]]
    oof = oof.merge(panel, on=["iso3", "t"])
    oof["res"] = oof.y - oof["Ours: Ridge + engineered (Set C)"]
    bins, labs = [0, 2.5, 4, 99], ["<2.5", "2.5-4", ">=4"]
    oof["stage"] = pd.cut(oof.tfr, bins, right=False, labels=labs)
    te["stage"] = pd.cut(te.tfr, bins, right=False, labels=labs)
    q = oof.groupby("stage", observed=True)["res"].quantile([0.1, 0.9]).unstack()
    te["lo"] = te.pred + te.stage.map(q[0.1]).astype(float)
    te["hi"] = te.pred + te.stage.map(q[0.9]).astype(float)
    te["covered"] = (te.y >= te.lo) & (te.y <= te.hi)
    pi = te.groupby("stage", observed=True).agg(n=("y", "size"), coverage=("covered", "mean")).join(
        q.rename(columns={0.1: "q10", 0.9: "q90"}))
    pi["width"] = pi.q90 - pi.q10
    pi.loc["all"] = [len(te), te.covered.mean(), np.nan, np.nan, (te.hi - te.lo).mean()]
    pi.to_csv(TAB / "prediction_intervals.csv")
    log.append("80% ranges from CV residual quantiles (by TFR stage at t), test coverage:\n" + pi.round(3).to_string())

    # ------------------------------------------------------------ 4. errors in births per year
    wb = wpp.dropna(subset=["ISO3_code"]).set_index(["ISO3_code", "Time"])
    te["births_k"] = [wb.Births.get((i, y), np.nan) for i, y in zip(te.iso3, te.target_year)]
    te["births_per_child"] = te.births_k / te.y  # thousand births per 1.0 of TFR, at the target year
    te["err_births_k"] = (te.y - te.pred) * te.births_per_child
    te["rel_err"] = (te.y - te.pred).abs() / te.y
    births = te.groupby("stage", observed=True).agg(n=("y", "size"), median_rel_err=("rel_err", "median"),
                                                    mean_rel_err=("rel_err", "mean"))
    births.loc["all"] = [len(te), te.rel_err.median(), te.rel_err.mean()]
    births.to_csv(TAB / "births_error.csv")
    log.append("relative error (= relative error in annual births, holding the number of women fixed):\n"
               + births.round(3).to_string())
    focus = te[te.iso3.isin(["TWN", "KOR", "JPN", "NGA", "IND", "USA"])][
        ["country", "target_year", "y", "pred", "lo", "hi", "un", "births_k", "err_births_k"]]
    focus.to_csv(TAB / "births_focus.csv", index=False)
    log.append("focus countries (births in thousands):\n" + focus.round(2).to_string(index=False))
    te.to_csv(TAB / "test_predictions_practical.csv", index=False)

    (TAB / "practical_log.txt").write_text("\n\n".join(log))
    print("\n\n".join(log))


if __name__ == "__main__":
    main()
