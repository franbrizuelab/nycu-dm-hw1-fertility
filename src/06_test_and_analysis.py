"""Sections 7-8: one-time test evaluation, significance, coefficients, error analysis.

The design is read from results/final_config.json (written by 05 from CV only).
A lock file records the hash of that config the first time the test set is scored;
if the config changes afterwards the script refuses to run, so the test set can
never feed back into design decisions. Re-running with the same config is allowed
(it reproduces identical numbers).
"""
import datetime
import hashlib
import json

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from common import (RAW_COLS, RES, SEED, TAB, THRESHOLD, add_features, load_panel, metrics, savefig)
from plotstyle import AQUA, BLUE, COL_W, FULL_W, GRAY, INK, MUTED, ORANGE, plt

rng = np.random.default_rng(SEED)
SHORT = {"Central African Republic": "C. African Rep."}
CFG_PATH, LOCK = RES / "final_config.json", RES / "TEST_LOCK.json"


REPORT_ONLY = ("significance_cv", "best_baseline_cv")  # CV statistics, not design choices


def design_hash(cfg):
    """Hash of the design choices only (features, knot, hyperparameters), rounded so that
    floating-point noise from parallel tree fitting cannot change it."""
    def rnd(x):
        if isinstance(x, float):
            return round(x, 8)
        if isinstance(x, list):
            return [rnd(v) for v in x]
        return x
    design = {k: rnd(v) for k, v in cfg.items() if k not in REPORT_ONLY}
    return hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()


def check_lock():
    h = design_hash(json.loads(CFG_PATH.read_text()))
    if LOCK.exists():
        lock = json.loads(LOCK.read_text())
        if lock["design_sha256"] != h:
            raise SystemExit("the frozen design changed after the test set was scored; refusing to re-score.")
        print(f"test set already scored at {lock['first_scored']} with this exact design; reproducing.")
    else:
        LOCK.write_text(json.dumps({"design_sha256": h, "first_scored": datetime.datetime.now().isoformat()}))


def lin(model):
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), model)


def parse(s):
    return {k: float(v) for k, v in (kv.split("=") for kv in s.split(","))}


def cluster_boot(df, err_a, err_b=None, n=2000):
    """MAE (or MAE gap a-b) with a 95% CI from resampling countries."""
    d = err_a.abs() if err_b is None else err_a.abs() - err_b.abs()
    g = pd.DataFrame({"iso3": df.iso3, "d": d}).groupby("iso3")["d"].agg(["sum", "count"])
    keys = np.arange(len(g))
    vals = [g.iloc[pick]["sum"].sum() / g.iloc[pick]["count"].sum()
            for pick in (rng.choice(keys, len(keys)) for _ in range(n))]
    return d.mean(), *np.percentile(vals, [2.5, 97.5])


def main():
    check_lock()
    cfg = json.loads(CFG_PATH.read_text())
    knot, FINAL = cfg["knot"], cfg["features"]
    panel = add_features(load_panel(), knot)
    tr = panel[panel.split == "train"].reset_index(drop=True)
    te = panel[panel.split == "test"].reset_index(drop=True)
    log = [f"train rows {len(tr)} (t {tr.t.min()}-{tr.t.max()}), test rows {len(te)} (t {sorted(te.t.unique())})"]
    no_edu = [c for c in FINAL if c not in ("edu_f1524", "edu_x_stage", "stage")]

    # ------------------------------------------------------------ refit every method on all training rows
    preds = {"Training mean": np.full(len(te), tr.y.mean()),
             "Persistence": te.tfr.values,
             "Damped trend": (te.tfr + cfg["damped_phi"] * 2 * te.d5).values}
    m = lin(LinearRegression()).fit(tr[["edu_f1524"]], tr.y)
    preds["Single-feature OLS (schooling)"] = m.predict(te[["edu_f1524"]])
    p = parse(cfg["rf_param"].replace("leaf", "min_samples_leaf").replace("mf", "max_features"))
    rf_raw = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
        n_estimators=400, min_samples_leaf=int(p["min_samples_leaf"]), max_features=p["max_features"],
        random_state=SEED, n_jobs=-1)).fit(tr[RAW_COLS], tr.y)
    preds["Random forest (raw)"] = rf_raw.predict(te[RAW_COLS])
    p = parse(cfg["gbm_param"])
    gb = HistGradientBoostingRegressor(learning_rate=p["lr"], max_depth=int(p["depth"]), max_iter=300,
                                       random_state=SEED).fit(tr[RAW_COLS], tr.y)
    preds["Gradient boosting (raw)"] = gb.predict(te[RAW_COLS])
    preds["Ridge (raw, Set A)"] = lin(Ridge(cfg["ridge_raw_alpha"])).fit(tr[RAW_COLS], tr.y).predict(te[RAW_COLS])
    ours = lin(Ridge(cfg["alpha"])).fit(tr[FINAL], tr.y)
    preds["Ours: Ridge + engineered (Set C)"] = ours.predict(te[FINAL])
    rf_c = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
        n_estimators=400, min_samples_leaf=1, max_features=1.0, random_state=SEED, n_jobs=-1)).fit(tr[FINAL], tr.y)
    preds["Random forest (Set C)"] = rf_c.predict(te[FINAL])
    # ablation-only models
    k = len(FINAL)
    rank = tr[cfg["setB_pool"]].corrwith(tr.y, method="spearman").abs().sort_values(ascending=False)
    setb_cols = rank.index[:k].tolist()
    abl_preds = {"Set A: raw columns": preds["Ridge (raw, Set A)"],
                 f"Set B: top-{k} raw by |Spearman|": lin(Ridge(cfg["setB_alpha"])).fit(tr[setb_cols], tr.y)
                 .predict(te[setb_cols]),
                 "Set C: ours": preds["Ours: Ridge + engineered (Set C)"],
                 "C minus education_stage": lin(Ridge(cfg["no_edu_alpha"])).fit(tr[no_edu], tr.y).predict(te[no_edu])}
    log.append(f"Set B columns on full training data: {setb_cols}")

    # Diagnostic (post-freeze, does not change the final model): does the gain of the education group
    # come from education itself or from the TFR hinge max(0, TFR - knot) that it contains?
    # CV with the same folds and alpha grid, plus the test score.
    from common import load_folds
    base = tr.copy()
    folds = load_folds(base)
    hinge_only = [c for c in FINAL if c not in ("edu_f1524", "edu_x_stage")]
    diag = []
    # H2/H3 diagnostics: adult instead of young-cohort schooling; adding the gender gap.
    base["edu_f2564_x_stage"] = base["edu_f2564"] * base["stage"]
    tr["edu_f2564_x_stage"] = tr["edu_f2564"] * tr["stage"]
    te["edu_f2564_x_stage"] = te["edu_f2564"] * te["stage"]
    adult = ["tfr", "d5", "edu_f2564", "edu_f2564_x_stage", "stage"]
    for name, cols in (("C minus education, keep TFR hinge", hinge_only), ("Set C: ours", FINAL),
                       ("C with adult (25-64) schooling", adult), ("C plus gender gap", FINAL + ["edu_gap1524"])):
        best = None
        for a in np.logspace(-3, 3, 13):
            maes = []
            for v, itr, iva in folds:
                mm = lin(Ridge(a)).fit(base.loc[itr, cols], base.y[itr])
                maes.append(np.abs(base.y[iva] - mm.predict(base.loc[iva, cols])).mean())
            if best is None or np.mean(maes) < best[1]:
                best = (a, np.mean(maes), np.std(maes, ddof=1), maes)
        mm = lin(Ridge(best[0])).fit(tr[cols], tr.y)
        abl_preds[name] = mm.predict(te[cols])
        diag.append({"set": name, "alpha": best[0], "CV_MAE": best[1], "CV_MAE_sd": best[2],
                     "fold_MAE": np.round(best[3], 3).tolist(), "Test_MAE": metrics(te.y, abl_preds[name])["MAE"]})
    pd.DataFrame(diag).to_csv(TAB / "diag_hinge_vs_education.csv", index=False)
    log.append("diagnostic hinge vs education:\n" + pd.DataFrame(diag).round(4).to_string(index=False))

    # ------------------------------------------------------------ main table: CV + test
    cv = pd.read_csv(TAB / "cv_main.csv").set_index("method")
    rows = []
    for name, p in preds.items():
        mt = metrics(te.y, p)
        rows.append({"method": name, "CV_MAE": cv.loc[name, "MAE_mean"], "CV_MAE_sd": cv.loc[name, "MAE_sd"],
                     "CV_RMSE": cv.loc[name, "RMSE_mean"], "CV_RMSE_sd": cv.loc[name, "RMSE_sd"],
                     "Test_MAE": mt["MAE"], "Test_RMSE": mt["RMSE"], "Test_R2": mt["R2"], "Test_Hit": mt["Hit"]})
    res = pd.DataFrame(rows)
    res.to_csv(TAB / "test_main.csv", index=False)
    log.append("main results:\n" + res.round(3).to_string(index=False))

    e_ours = te.y - preds["Ours: Ridge + engineered (Set C)"]
    sig = []
    for name in ["Gradient boosting (raw)", "Random forest (raw)", "Random forest (Set C)", "Damped trend",
                 "Ridge (raw, Set A)", "Persistence"]:
        g, lo, hi = cluster_boot(te, e_ours, te.y - preds[name])
        sig.append({"vs": name, "gap": g, "ci_lo": lo, "ci_hi": hi})
    sig = pd.DataFrame(sig)
    sig.to_csv(TAB / "test_significance.csv", index=False)
    log.append("test MAE gap ours - baseline (country-cluster bootstrap 95% CI):\n" + sig.round(3).to_string(index=False))
    mae_ci = cluster_boot(te, e_ours)
    log.append(f"ours test MAE {mae_ci[0]:.3f} [{mae_ci[1]:.3f}, {mae_ci[2]:.3f}]; threshold {THRESHOLD}")
    log.append(f"min prediction ours {preds['Ours: Ridge + engineered (Set C)'].min():.3f}")

    # ------------------------------------------------------------ ablation on test + slices
    cva = pd.read_csv(TAB / "cv_ablation.csv").set_index("method")
    ab = []
    dg = pd.DataFrame(diag).set_index("set")
    for name, p in abl_preds.items():
        src = cva if name in cva.index else dg
        ab.append({"set": name, "CV_MAE": src.loc[name, "MAE_mean" if src is cva else "CV_MAE"],
                   "CV_MAE_sd": src.loc[name, "MAE_sd" if src is cva else "CV_MAE_sd"],
                   "Test_MAE": metrics(te.y, p)["MAE"]})
    ab = pd.DataFrame(ab)
    ab.to_csv(TAB / "test_ablation.csv", index=False)
    log.append("ablation (test):\n" + ab.round(3).to_string(index=False))
    e_noedu = te.y - abl_preds["C minus education_stage"]
    te["stage_bin"] = pd.cut(te.tfr, [0, 2.5, knot, 5.5, 99], right=False,
                             labels=["<2.5", f"2.5-{knot:g}", f"{knot:g}-5.5", ">=5.5"])
    sl = []
    for key in ("stage_bin", "region"):
        for lev, s in te.groupby(key, observed=True):
            i = s.index
            sl.append({"by": key, "slice": lev, "n": len(s),
                       "Set A": np.abs(te.y[i] - abl_preds["Set A: raw columns"][i]).mean(),
                       "C w/o edu": e_noedu[i].abs().mean(), "Set C": e_ours[i].abs().mean(),
                       "GBM (raw)": np.abs(te.y[i] - preds["Gradient boosting (raw)"][i]).mean(),
                       "bias C": e_ours[i].mean()})
    sl = pd.DataFrame(sl)
    sl.to_csv(TAB / "test_slices.csv", index=False)
    log.append("test slices (MAE; bias = mean actual-pred):\n" + sl.round(3).to_string(index=False))
    act = te.tfr >= knot
    for lab, msk in (("all", np.ones(len(te), bool)), (f"TFR>={knot}", act), (f"TFR<{knot}", ~act)):
        g = cluster_boot(te[msk], e_ours[msk], e_noedu[msk])
        log.append(f"education gap on test, {lab} (n={msk.sum()}): {g[0]:.3f} [{g[1]:.3f}, {g[2]:.3f}]")
    e_hinge = te.y - abl_preds["C minus education, keep TFR hinge"]
    for lab, msk in (("all", np.ones(len(te), bool)), (f"TFR>={knot}", act), (f"TFR<{knot}", ~act)):
        g = cluster_boot(te[msk], e_ours[msk], e_hinge[msk])
        log.append(f"education gap given the hinge, test, {lab}: {g[0]:.3f} [{g[1]:.3f}, {g[2]:.3f}]")
    for t in sorted(te.t.unique()):
        s = te.t == t
        log.append(f"t={t}: MAE ours {e_ours[s].abs().mean():.3f}, bias {e_ours[s].mean():+.3f}; "
                   f"persistence MAE {np.abs(te.y[s] - te.tfr[s]).mean():.3f}")

    # ------------------------------------------------------------ coefficients (Section 8.1)
    Z = (tr[FINAL] - tr[FINAL].mean()) / tr[FINAL].std()
    ols = sm.OLS(tr.y, sm.add_constant(Z)).fit(cov_type="cluster", cov_kwds={"groups": tr.iso3})
    Zc = sm.add_constant(Z).values
    vif = [variance_inflation_factor(Zc, i + 1) for i in range(len(FINAL))]
    # Ridge (selected alpha) with a country-cluster bootstrap CI, for comparison.
    groups = {g: idx.values for g, idx in tr.groupby("iso3").groups.items()}
    keys = np.array(list(groups))
    boot = []
    for _ in range(1000):
        idx = np.concatenate([groups[g] for g in rng.choice(keys, len(keys))])
        r = Ridge(cfg["alpha"]).fit(Z.iloc[idx], tr.y.iloc[idx])
        boot.append(r.coef_)
    boot = np.array(boot)
    ridge_full = Ridge(cfg["alpha"]).fit(Z, tr.y)
    coef = pd.DataFrame({
        "feature": FINAL, "beta_ols": ols.params[FINAL].values,
        "ci_lo": ols.conf_int().loc[FINAL, 0].values, "ci_hi": ols.conf_int().loc[FINAL, 1].values,
        "p": ols.pvalues[FINAL].values, "vif": vif, "beta_ridge": ridge_full.coef_,
        "ridge_boot_lo": np.percentile(boot, 2.5, axis=0), "ridge_boot_hi": np.percentile(boot, 97.5, axis=0),
        "sd": tr[FINAL].std().values})
    coef.to_csv(TAB / "coefficients.csv", index=False)
    log.append(f"OLS, standardized features, country-clustered SE (n={len(tr)}, {tr.iso3.nunique()} clusters), "
               f"R2={ols.rsquared:.3f}:\n" + coef.round(4).to_string(index=False))
    # Practical effect: change in predicted TFR(t+10) per extra year of young women's schooling,
    # holding TFR(t) and its trend fixed, at different starting TFR levels.
    raw = sm.OLS(tr.y, sm.add_constant(tr[FINAL])).fit(cov_type="cluster", cov_kwds={"groups": tr.iso3})
    eff = []
    for tfr0 in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0):
        grad = np.zeros(len(raw.params))
        names = list(raw.params.index)
        grad[names.index("edu_f1524")] = 1
        grad[names.index("edu_x_stage")] = max(0.0, tfr0 - knot)
        est = grad @ raw.params.values
        se = np.sqrt(grad @ raw.cov_params().values @ grad)
        eff.append({"tfr_t": tfr0, "effect_per_year": est, "ci_lo": est - 1.96 * se, "ci_hi": est + 1.96 * se})
    eff = pd.DataFrame(eff)
    eff.to_csv(TAB / "education_effect_by_stage.csv", index=False)
    log.append("effect of +1 year female schooling (15-24) on TFR(t+10), children/woman, clustered 95% CI:\n"
               + eff.round(3).to_string(index=False))
    log.append("raw-unit OLS coefficients:\n" + raw.params.round(4).to_string())

    # ------------------------------------------------------------ error analysis (Section 8.2)
    te["pred"] = preds["Ours: Ridge + engineered (Set C)"]
    te["err"] = te.y - te.pred  # actual - predicted
    te["pred_noedu"] = abl_preds["C minus education_stage"]
    cols = ["country", "t", "target_year", "tfr", "d5", "edu_f1524", "y", "pred", "err", "pred_noedu"]
    worst = te.reindex(te.err.abs().sort_values(ascending=False).index)[cols].head(8)
    worst.to_csv(TAB / "test_worst_cases.csv", index=False)
    log.append("largest test errors:\n" + worst.round(2).to_string(index=False))
    focus = te[te.iso3.isin(["TWN", "KOR", "JPN", "HKG", "SGP", "CHN"])][cols]
    focus.to_csv(TAB / "test_east_asia.csv", index=False)
    log.append("East Asia focus:\n" + focus.round(2).to_string(index=False))
    log.append(f"residual vs TFR(t): corr {np.corrcoef(te.err, te.tfr)[0, 1]:.3f}; "
               f"share of over-prediction among TFR(t)<2: {(te.err[te.tfr < 2] < 0).mean():.2f}")

    # ------------------------------------------------------------ robustness: leave-one-region-out
    lor = []
    for reg, s in te.groupby("region"):
        m = lin(Ridge(cfg["alpha"])).fit(tr.loc[tr.region != reg, FINAL], tr.y[tr.region != reg])
        lor.append({"region": reg, "n": len(s), "MAE_in": e_ours[s.index].abs().mean(),
                    "MAE_held_out": np.abs(s.y - m.predict(s[FINAL])).mean()})
    lor = pd.DataFrame(lor)
    lor.to_csv(TAB / "test_leave_region_out.csv", index=False)
    log.append("leave-one-region-out (train without region, test on it):\n" + lor.round(3).to_string(index=False))

    # ------------------------------------------------------------ figure: test predictions and errors
    fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.1), gridspec_kw={"wspace": 0.3, "width_ratios": [1, 1.15]})
    lim = (0.5, 8)
    ax[0].plot(lim, lim, color=GRAY, lw=0.7)
    ax[0].fill_between(lim, [lim[0] - THRESHOLD, lim[1] - THRESHOLD], [lim[0] + THRESHOLD, lim[1] + THRESHOLD],
                       color=GRAY, alpha=0.15, lw=0, label=f"$\\pm${THRESHOLD} threshold")
    for t, c in zip(sorted(te.t.unique()), (BLUE, ORANGE)):
        s = te[te.t == t]
        ax[0].scatter(s.pred, s.y, s=7, color=c, lw=0, alpha=0.8, label=f"t={t} (target {t + 10})")
    for _, r in worst.head(3).iterrows():
        right = r.pred > 5  # keep labels clear of the legend in the upper-left corner
        ax[0].annotate(f"{SHORT.get(r.country, r.country)} {r.target_year}", (r.pred, r.y), fontsize=5.8,
                       xytext=(4, -9) if right else (3, -3), ha="left", textcoords="offset points", color=INK)
    tw = te[te.iso3 == "TWN"]
    ax[0].scatter(tw.pred, tw.y, s=22, facecolor="none", edgecolor=INK, lw=0.8, label="Taiwan")
    ax[0].set(xlim=lim, ylim=lim, xlabel="Predicted TFR at t+10 (children/woman)",
              ylabel="Actual TFR at t+10 (children/woman)", title="(a) Test set: predicted vs. actual")
    ax[0].legend(loc="upper left", markerscale=1.4)
    s2 = sl[sl.by == "stage_bin"].reset_index(drop=True)
    x = np.arange(len(s2))
    w = 0.26
    for j, (col, c, lab) in enumerate((("C w/o edu", GRAY, "Set C without education"),
                                       ("Set C", BLUE, "Set C (ours)"), ("GBM (raw)", AQUA, "Gradient boosting (raw)"))):
        ax[1].bar(x + (j - 1) * w, s2[col], w * 0.92, color=c, label=lab)
    ax[1].axhline(THRESHOLD, color=ORANGE, lw=0.9, ls="--", label=f"threshold {THRESHOLD}")
    ax[1].set_xticks(x, [f"{a}\n(n={n})" for a, n in zip(s2.slice.astype(str).str.replace(">=", "$\\geq$"), s2.n)])
    ax[1].set(xlabel="TFR at t (transition stage)", ylabel="Test MAE (children/woman)",
              title="(b) Test error by transition stage")
    ax[1].legend(loc="upper left")
    savefig(fig, "fig_test")

    (TAB / "test_log.txt").write_text("\n\n".join(log))
    print("\n\n".join(log))


if __name__ == "__main__":
    main()
