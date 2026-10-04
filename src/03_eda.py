"""Section 4: exploratory feature analysis. TRAINING ROWS ONLY (start years <= 1995).

Writes figures/fig_eda.pdf and the tables used in Sections 3-4 of the paper.
"""
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import PROC, SEED, TAB, add_features, load_panel, savefig, train_panel
from plotstyle import BLUE, BLUE_RAMP, COL_W, FULL_W, GRAY, INK, MUTED, ORANGE, plt

rng = np.random.default_rng(SEED)


def resid(y, X):
    """Residual of y after OLS on X (with intercept)."""
    X = np.column_stack([np.ones(len(y)), X])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def partial_r(df, a, b, controls):
    ra = resid(df[a].values, df[controls].values)
    rb = resid(df[b].values, df[controls].values)
    return np.corrcoef(ra, rb)[0, 1]


def cluster_boot(df, stat, n=1000):
    """95% CI of stat(df) by resampling countries (all rows of a country together)."""
    groups = {k: v.index.values for k, v in df.groupby("iso3")}
    keys = np.array(list(groups))
    vals = []
    for _ in range(n):
        pick = rng.choice(keys, len(keys))
        vals.append(stat(df.loc[np.concatenate([groups[k] for k in pick])]))
    return np.nanpercentile(vals, [2.5, 97.5])


def main():
    full = add_features(load_panel())
    df = add_features(train_panel())
    df["dy"] = df["y"] - df["tfr"]  # future 10-year change in TFR
    log = []

    # ---------------------------------------------------------------- Section 3 summary table
    cols = {"y": "TFR at t+10 (target)", "tfr": "TFR at t", "d5": "TFR change, t-5 to t",
            "edu_f1524": "Female schooling 15-24 (yr)", "edu_f2564": "Female schooling 25-64 (yr)",
            "edu_m1524": "Male schooling 15-24 (yr)", "q5": "Under-5 mortality (per 1,000)",
            "gdppc": "GDP per capita (2017 US$ PPP)", "urban": "Urban share (%)"}
    summ = pd.DataFrame({
        "Variable": list(cols.values()),
        "Mean": [df[c].mean() for c in cols], "SD": [df[c].std() for c in cols],
        "Min": [df[c].min() for c in cols], "Max": [df[c].max() for c in cols],
        "Missing %": [100 * df[c].isna().mean() for c in cols]})
    summ.to_csv(TAB / "summary_stats_train.csv", index=False)
    counts = full.groupby("split").agg(rows=("y", "size"), countries=("iso3", "nunique"),
                                       t_min=("t", "min"), t_max=("t", "max"))
    counts.to_csv(TAB / "split_counts.csv")
    log.append(f"split counts:\n{counts}")

    # ---------------------------------------------------------------- association table (H1, Set B ranking)
    cand = ["tfr", "d5", "tfr2", "edu_f1524", "edu_f2534", "edu_f2564", "edu_m1524", "edu_m2564",
            "edu_avg1524", "edu_gap1524", "q5", "log_q5", "gdppc", "log_gdppc", "urban", "edu_x_stage"]
    cc = df.dropna(subset=cand)  # GDP complete cases for comparability
    mi = mutual_info_regression(cc[cand].values, cc["y"].values, random_state=SEED)
    assoc = pd.DataFrame({
        "feature": cand,
        "pearson_y": [cc[c].corr(cc["y"]) for c in cand],
        "spearman_y": [cc[c].corr(cc["y"], method="spearman") for c in cand],
        "MI_y": mi,
        "pearson_dy": [cc[c].corr(cc["dy"]) for c in cand],
        "partial_y_given_tfr": [np.nan if c in ("tfr", "tfr2") else partial_r(cc, c, "y", ["tfr", "tfr2"])
                                for c in cand],
    }).sort_values("MI_y", ascending=False)
    assoc.to_csv(TAB / "eda_association.csv", index=False)
    log.append(f"association (n={len(cc)}):\n{assoc.round(3).to_string(index=False)}")

    # ---------------------------------------------------------------- redundancy (H3)
    raw = ["tfr", "edu_f1524", "edu_f2564", "edu_m1524", "edu_m2564", "log_q5", "log_gdppc", "urban"]
    Z = (cc[raw] - cc[raw].mean()) / cc[raw].std()
    Zc = np.column_stack([np.ones(len(Z)), Z.values])
    vif = pd.Series([variance_inflation_factor(Zc, i + 1) for i in range(len(raw))], index=raw)
    vif.to_csv(TAB / "eda_vif_raw.csv")
    cm = cc[raw].corr()
    cm.to_csv(TAB / "eda_corr_raw.csv")
    log.append(f"VIF raw:\n{vif.round(1).to_string()}")
    log.append(f"r(edu_f1524, edu_m1524) = {cc.edu_f1524.corr(cc.edu_m1524):.3f}; "
               f"r(edu_f1524, edu_f2564) = {cc.edu_f1524.corr(cc.edu_f2564):.3f}; "
               f"r(tfr, log_q5) = {cc.tfr.corr(cc.log_q5):.3f}")
    log.append(f"H3 partial r(gap, y | avg schooling, tfr, tfr2) = "
               f"{partial_r(cc, 'edu_gap1524', 'y', ['edu_avg1524', 'tfr', 'tfr2']):.3f}")

    # ---------------------------------------------------------------- confounders (H4)
    for ctrl in (["log_gdppc"], ["log_gdppc", "log_q5"], ["log_gdppc", "log_q5", "urban"],
                 ["tfr", "tfr2"], ["tfr", "tfr2", "d5", "log_gdppc", "log_q5"]):
        log.append(f"H4 partial r(edu_f1524, y | {ctrl}) = {partial_r(cc, 'edu_f1524', 'y', ctrl):.3f}")
    log.append(f"H4 raw r(edu_f1524, y) = {cc.edu_f1524.corr(cc.y):.3f}")

    # ---------------------------------------------------------------- cohort lag curve (H2), Barro-Lee v2.2
    bands = pd.read_csv(PROC / "edu_5yr_bands.csv").rename(columns={"year": "t"})
    order = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64"]
    lb = df.merge(bands, on=["iso3", "t"], how="inner").dropna(subset=order)
    lag = pd.DataFrame({
        "band": order,
        "spearman_y": [lb[b].corr(lb["y"], method="spearman") for b in order],
        "partial_y_given_tfr": [partial_r(lb, b, "y", ["tfr", "tfr2"]) for b in order]})
    lag.to_csv(TAB / "eda_cohort_lag.csv", index=False)
    log.append(f"cohort lag curve (n={len(lb)}):\n{lag.round(3).to_string(index=False)}")

    # ---------------------------------------------------------------- stage slices (H-onset)
    bins = [0, 2.5, 4, 5.5, 99]
    labels = ["<2.5", "2.5-4", "4-5.5", ">=5.5"]
    df["stage_bin"] = pd.cut(df["tfr"], bins, right=False, labels=labels)
    rows = []
    for lab in labels:
        s = df[df.stage_bin == lab].reset_index(drop=True)
        stat = lambda d: partial_r(d, "edu_f1524", "dy", ["tfr"]) if len(d) > 10 else np.nan  # noqa: E731
        lo, hi = cluster_boot(s, stat)
        rows.append({"stage": lab, "n": len(s), "mean_dy": s.dy.mean(), "sd_dy": s.dy.std(),
                     "partial_r": stat(s), "ci_lo": lo, "ci_hi": hi})
    stage = pd.DataFrame(rows)
    stage.to_csv(TAB / "eda_stage_slices.csv", index=False)
    log.append(f"stage slices: partial r(edu_f1524, future change | tfr), country-cluster bootstrap CI:\n"
               f"{stage.round(3).to_string(index=False)}")

    # ---------------------------------------------------------------- stability over time (H5)
    df["decade"] = (df["t"] // 10) * 10
    stab = []
    for dec, s in df.groupby("decade"):
        X = np.column_stack([np.ones(len(s)), s["tfr"], s["edu_f1524"]])
        beta, res, *_ = np.linalg.lstsq(X, s["dy"].values, rcond=None)
        e = s["dy"].values - X @ beta
        cov = np.linalg.inv(X.T @ X) * (e @ e) / (len(s) - 3)
        stab.append({"decade": int(dec), "n": len(s), "slope_edu": beta[2], "se": np.sqrt(cov[2, 2]),
                     "intercept_mean_dy": s.dy.mean()})
    stab = pd.DataFrame(stab)
    stab.to_csv(TAB / "eda_stability.csv", index=False)
    log.append(f"H5 slope of future change on edu_f1524 | tfr, by start decade:\n{stab.round(3).to_string(index=False)}")

    # ---------------------------------------------------------------- Figure: 4 panels
    fig, axs = plt.subplots(2, 2, figsize=(FULL_W, 3.35), gridspec_kw={"wspace": 0.28, "hspace": 0.62})
    ax = axs.ravel()
    # (a) transition curve
    decs = sorted(df.decade.unique())
    for i, dec in enumerate(decs):
        s = df[df.decade == dec]
        ax[0].scatter(s.edu_f1524, s.y, s=4, color=BLUE_RAMP[min(i, 5)], alpha=0.6, lw=0,
                      label=f"t in {dec}s")
    for iso, name, k, off in [("KOR", "Korea", 0, (3, 3)), ("IRN", "Iran", -1, (-4, -9)),
                              ("NER", "Niger", -1, (4, -2))]:
        s = df[df.iso3 == iso].sort_values("t")
        ax[0].plot(s.edu_f1524, s.y, color=ORANGE, lw=1.1)
        ax[0].annotate(name, (s.edu_f1524.iloc[k], s.y.iloc[k]), fontsize=6.5, color=INK,
                       xytext=off, textcoords="offset points")
    ax[0].set(xlabel="Female schooling at t, ages 15-24 (years)", ylabel="TFR at t+10 (children/woman)",
              title="(a) Transition curve")
    ax[0].legend(loc="upper right", markerscale=2.5, handletextpad=0.1, labelspacing=0.2)
    # (b) future change vs current level
    sc = ax[1].scatter(df.tfr, df.dy, c=df.edu_f1524, cmap="Blues", s=4, lw=0, vmin=-2, vmax=13)
    ax[1].axhline(0, color=GRAY, lw=0.6)
    ax[1].set(xlabel="TFR at t (children/woman)", ylabel="10-yr change (children/woman)",
              title="(b) Future change vs. current level")
    cax = ax[1].inset_axes([0.06, 0.1, 0.03, 0.3])
    cb = fig.colorbar(sc, cax=cax)
    cb.ax.set_title("school.\n(yr)", fontsize=5.5, loc="left")
    cb.ax.tick_params(labelsize=5.5)
    # (c) cohort lag curve
    x = np.arange(len(order))
    ax[2].plot(x, lag.spearman_y.abs(), "o-", color=BLUE, ms=3.5, label=r"$|\rho|$ with TFR(t+10)")
    ax[2].plot(x, lag.partial_y_given_tfr.abs(), "s-", color=ORANGE, ms=3.5,
               label="|partial r| given TFR(t)")
    ax[2].set_xticks(x, order, rotation=40)
    ax[2].set_ylim(0, 1)
    ax[2].set(xlabel="Age band of women at t (Barro-Lee v2.2)", ylabel="Association (absolute)",
              title="(c) Which cohort matters")
    ax[2].legend(loc="center right")
    # (d) stage slices
    xs = np.arange(len(stage))
    ax[3].errorbar(xs, stage.partial_r, yerr=[stage.partial_r - stage.ci_lo, stage.ci_hi - stage.partial_r],
                   fmt="o", color=BLUE, ms=4, capsize=2, lw=1)
    for xi, (r, n) in enumerate(zip(stage.partial_r, stage.n)):
        ax[3].annotate(f"n={n}", (xi, 0.02), ha="center", fontsize=6, color=MUTED)
    ax[3].axhline(0, color=GRAY, lw=0.6)
    ax[3].set_xticks(xs, ["<2.5", "2.5-4", "4-5.5", "$\\geq$5.5"])
    ax[3].set_ylim(-0.65, 0.1)
    ax[3].set(xlabel="TFR at t (transition stage)", ylabel="partial r",
              title="(d) Schooling vs. future change | TFR(t)")
    savefig(fig, "fig_eda")

    (TAB / "eda_log.txt").write_text("\n\n".join(log))
    print("\n\n".join(log))


if __name__ == "__main__":
    main()
