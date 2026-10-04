"""Sections 5-7 (CV part): model choice, tuning, baselines, ablation, slices.

Uses only the training portion (start years <= 1995) and the saved folds.
Writes results/tables/cv_*.csv and results/final_config.json (read by 06).
"""
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (GROUPS, RAW_COLS, RES, SEED, SET_C, TAB, THRESHOLD, add_features, load_folds,
                    metrics, train_panel)

ALPHAS = np.logspace(-3, 3, 13)
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------------------- model zoo
def linear(model):
    # Median imputation (+ missing indicator) only matters for GDP; fit inside each fold.
    return lambda: make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), model)


def ridge(a):
    return linear(Ridge(alpha=a))


def rf(leaf, mf):
    return lambda: make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
        n_estimators=400, min_samples_leaf=leaf, max_features=mf, random_state=SEED, n_jobs=-1))


def gbm(lr, depth):
    return lambda: HistGradientBoostingRegressor(learning_rate=lr, max_depth=depth, max_iter=300,
                                                 random_state=SEED)


# ----------------------------------------------------------------------------- CV engine
def cv_predict(df, folds, cols, factory):
    """Out-of-fold predictions for the 4 validation years (NaN elsewhere)."""
    oof = pd.Series(np.nan, index=df.index)
    for v, tr, va in folds:
        m = factory()
        m.fit(df.loc[tr, cols], df.loc[tr, "y"])
        oof.loc[va] = m.predict(df.loc[va, cols])
    return oof


def fold_scores(df, folds, pred):
    return pd.DataFrame([{"val_t": v, **metrics(df.loc[va, "y"], pred.loc[va])} for v, tr, va in folds])


def tune(df, folds, cols, grid):
    """grid: {label: factory}. Pick the label with the lowest mean fold MAE."""
    rows, preds = [], {}
    for label, factory in grid.items():
        p = cv_predict(df, folds, cols, factory)
        sc = fold_scores(df, folds, p)
        rows.append({"param": label, "MAE_mean": sc.MAE.mean(), "MAE_sd": sc.MAE.std(ddof=1)})
        preds[label] = p
    curve = pd.DataFrame(rows)
    best = curve.loc[curve.MAE_mean.idxmin(), "param"]
    return best, preds[best], curve


def summarize(name, df, folds, pred, param=""):
    sc = fold_scores(df, folds, pred)
    out = {"method": name, "param": param}
    for k in ("MAE", "RMSE", "R2", "Hit"):
        out[f"{k}_mean"], out[f"{k}_sd"] = sc[k].mean(), sc[k].std(ddof=1)
    out.update({f"MAE_{v}": m for v, m in zip(sc.val_t, sc.MAE)})
    return out


def ridge_grid():
    return {f"{a:.3g}": ridge(a) for a in ALPHAS}


def top_k_corr_factory(k, pool):
    """Set B: inside each training fold, keep the k pool features with the largest |Spearman r| with y."""
    class TopK:
        def fit(self, X, y):
            r = X[pool].corrwith(pd.Series(y, index=X.index), method="spearman").abs()
            self.cols = r.sort_values(ascending=False).index[:k].tolist()
            self.inner = self.factory()
            self.inner.fit(X[self.cols], y)
            return self

        def predict(self, X):
            return self.inner.predict(X[self.cols])
    return TopK


def cluster_boot_gap(df, idx, p_a, p_b, n=2000):
    """95% CI of MAE(a) - MAE(b) on rows idx, resampling countries."""
    sub = df.loc[idx]
    ea, eb = (sub.y - p_a.loc[idx]).abs(), (sub.y - p_b.loc[idx]).abs()
    g = pd.DataFrame({"iso3": sub.iso3, "d": ea - eb}).groupby("iso3")["d"].agg(["sum", "count"])
    keys = np.arange(len(g))
    vals = [g.iloc[pick]["sum"].sum() / g.iloc[pick]["count"].sum()
            for pick in (rng.choice(keys, len(keys)) for _ in range(n))]
    return (ea - eb).mean(), *np.percentile(vals, [2.5, 97.5])


# ----------------------------------------------------------------------------- main
def main():
    base = train_panel()
    folds = load_folds(base)
    val_idx = np.concatenate([va for _, _, va in folds])
    log = []

    # --- stage knot: domain prior 5.5 from Fig. 1d, compared with neighbours by CV
    knot_rows = []
    for knot in (4.0, 4.5, 5.0, 5.5, 6.0):
        dk = add_features(base, knot)
        best, p, curve = tune(dk, folds, SET_C, ridge_grid())
        knot_rows.append({"knot": knot, "alpha": best, "MAE_mean": curve.MAE_mean.min()})
    knots = pd.DataFrame(knot_rows)
    knots.to_csv(TAB / "cv_knot.csv", index=False)
    knot = float(knots.loc[knots.MAE_mean.idxmin(), "knot"])
    log.append(f"stage knot CV:\n{knots.round(4).to_string(index=False)}\n-> knot = {knot}")
    df = add_features(base, knot)

    # --- backward group elimination on CV (decided here, before the test set is touched):
    # repeatedly drop the group whose removal lowers mean CV MAE the most; stop when none does.
    groups = dict(GROUPS)
    elim = []
    best_c, p_c, curve_c = tune(df, folds, SET_C, ridge_grid())
    cur = summarize("all groups", df, folds, p_c, best_c)["MAE_mean"]
    elim.append({"step": 0, "removed": "-", "MAE_mean": cur})
    drop = []
    while len(groups) > 1:
        trial = {}
        for g in groups:
            keep = [c for h, cols in groups.items() if h != g for c in cols]
            b, p, _ = tune(df, folds, keep, ridge_grid())
            trial[g] = summarize(g, df, folds, p, b)["MAE_mean"]
        g_best = min(trial, key=trial.get)
        log.append(f"elimination step {len(drop) + 1}: " + ", ".join(f"-{g}: {m:.4f}" for g, m in trial.items()))
        if trial[g_best] >= cur:
            break
        cur = trial[g_best]
        drop.append(g_best)
        groups.pop(g_best)
        elim.append({"step": len(drop), "removed": g_best, "MAE_mean": cur})
    pd.DataFrame(elim).to_csv(TAB / "cv_group_elimination.csv", index=False)
    final_groups = groups
    FINAL = [c for cols in final_groups.values() for c in cols]
    log.append(f"groups dropped by CV: {drop}; final features ({len(FINAL)}): {FINAL}")

    # Re-check the stage knot on the final feature set (it is a hyperparameter of the education group).
    if "education_stage" in final_groups:
        kr = []
        for k in (4.0, 4.5, 5.0, 5.5, 6.0):
            _, _, cv = tune(add_features(base, k), folds, FINAL, ridge_grid())
            kr.append({"knot": k, "MAE_mean": cv.MAE_mean.min()})
        kr = pd.DataFrame(kr)
        kr.to_csv(TAB / "cv_knot_final.csv", index=False)
        knot = float(kr.loc[kr.MAE_mean.idxmin(), "knot"])
        df = add_features(base, knot)
        log.append(f"knot on final set:\n{kr.round(4).to_string(index=False)}\n-> knot = {knot}")

    # --- linear variant choice on the final feature set
    best_r, p_ours, curve_r = tune(df, folds, FINAL, ridge_grid())
    curve_r.to_csv(TAB / "cv_tuning_ridge.csv", index=False)
    variants = [summarize("OLS", df, folds, cv_predict(df, folds, FINAL, linear(LinearRegression())))]
    variants.append(summarize("Ridge", df, folds, p_ours, best_r))
    b, p, _ = tune(df, folds, FINAL, {f"{a:.3g}": linear(Lasso(alpha=a, max_iter=50000))
                                      for a in np.logspace(-4, 0, 9)})
    variants.append(summarize("Lasso", df, folds, p, b))
    b, p, _ = tune(df, folds, FINAL, {f"{a:.3g},{l}": linear(ElasticNet(alpha=a, l1_ratio=l, max_iter=50000))
                                      for a in np.logspace(-4, 0, 9) for l in (0.2, 0.5, 0.8)})
    variants.append(summarize("Elastic Net", df, folds, p, b))
    pd.DataFrame(variants).to_csv(TAB / "cv_variants.csv", index=False)
    log.append("variants:\n" + pd.DataFrame(variants)[["method", "param", "MAE_mean", "MAE_sd"]].round(4)
               .to_string(index=False))

    # --- baselines
    main_rows, preds = [], {}
    p = pd.Series(np.nan, index=df.index)
    for v, tr, va in folds:
        p.loc[va] = df.loc[tr, "y"].mean()
    preds["Training mean"] = p
    preds["Persistence"] = df["tfr"].where(df.index.isin(val_idx))
    # Damped trend: continue phi x the last 5-year change for two more 5-year steps; phi tuned by CV.
    phis = np.round(np.arange(0, 1.01, 0.1), 1)
    dt = {phi: fold_scores(df, folds, df["tfr"] + phi * 2 * df["d5"]).MAE.mean() for phi in phis}
    phi = min(dt, key=dt.get)
    preds["Damped trend"] = (df["tfr"] + phi * 2 * df["d5"]).where(df.index.isin(val_idx))
    preds["Single-feature OLS (schooling)"] = cv_predict(df, folds, ["edu_f1524"], linear(LinearRegression()))
    params = {"Damped trend": f"phi={phi}"}
    b, preds["Random forest (raw)"], _ = tune(df, folds, RAW_COLS, {f"leaf={l},mf={m}": rf(l, m)
                                             for l in (1, 5, 10) for m in (0.33, 1.0)})
    params["Random forest (raw)"] = b
    b, preds["Gradient boosting (raw)"], _ = tune(df, folds, RAW_COLS, {f"lr={lr},depth={d}": gbm(lr, d)
                                                 for lr in (0.03, 0.1) for d in (2, 3, 5)})
    params["Gradient boosting (raw)"] = b
    b, preds["Ridge (raw, Set A)"], _ = tune(df, folds, RAW_COLS, ridge_grid())
    params["Ridge (raw, Set A)"] = b
    preds["Ours: Ridge + engineered (Set C)"] = p_ours
    params["Ours: Ridge + engineered (Set C)"] = best_r
    b, preds["Random forest (Set C)"], _ = tune(df, folds, FINAL, {f"leaf={l},mf={m}": rf(l, m)
                                               for l in (1, 5, 10) for m in (0.33, 1.0)})
    params["Random forest (Set C)"] = b
    for name, p in preds.items():
        main_rows.append(summarize(name, df, folds, p, params.get(name, "")))
    main = pd.DataFrame(main_rows)
    main.to_csv(TAB / "cv_main.csv", index=False)
    log.append("main CV:\n" + main[["method", "param", "MAE_mean", "MAE_sd", "RMSE_mean", "R2_mean", "Hit_mean"]]
               .round(3).to_string(index=False))

    # significance vs. the strongest baseline (lowest CV MAE among non-ours rows)
    others = main[~main.method.str.startswith("Ours")]
    best_base = others.loc[others.MAE_mean.idxmin(), "method"]
    ours_f = fold_scores(df, folds, p_ours).MAE.values
    base_f = fold_scores(df, folds, preds[best_base]).MAE.values
    t, pv = stats.ttest_rel(ours_f, base_f)
    gap, lo, hi = cluster_boot_gap(df, val_idx, p_ours, preds[best_base])
    sig = {"best_baseline": best_base, "paired_t": t, "paired_p": pv, "pooled_gap": gap, "ci_lo": lo, "ci_hi": hi}
    log.append(f"Ours vs {best_base}: paired t over 4 folds t={t:.2f}, p={pv:.3f}; "
               f"pooled CV MAE gap {gap:.3f} [{lo:.3f}, {hi:.3f}] (country-cluster bootstrap)")

    # --- ablation: Set A / Set B / Set C with identical Ridge tuning and folds
    # Set B ranks the raw columns (Set A) by |Spearman r| with y inside each training fold; k = |Set C|
    # so B and C have the same number of inputs. "B+" also lets the ranking pick the Section 4
    # engineered candidates, a stricter test of whether ranking alone would have found our features.
    pool_raw = list(RAW_COLS)
    pool_ext = sorted(set(RAW_COLS) | set(SET_C) | {"edu_avg1524", "edu_gap1524", "edu_f2534", "log_q5"})

    def set_b(k, pool):
        grid = {}
        for a in ALPHAS:
            cls = top_k_corr_factory(k, pool)
            grid[f"{a:.3g}"] = (lambda c=cls, a=a: type("M", (c,), {"factory": staticmethod(ridge(a))})())
        b, p, _ = tune(df, folds, pool, grid)
        return b, p

    k_rows, setb_pred = [], None
    for k in sorted({3, 5, 8, 11, len(FINAL)}):
        b, p = set_b(k, pool_raw)
        k_rows.append(summarize(f"Set B: top-{k} raw by |Spearman|", df, folds, p, b))
        if k == len(FINAL):
            setb, setb_pred = k_rows[-1], p
    b, p_bext = set_b(len(FINAL), pool_ext)
    setb_ext = summarize(f"Set B+: top-{len(FINAL)} of raw+engineered", df, folds, p_bext, b)
    pd.DataFrame(k_rows + [setb_ext]).to_csv(TAB / "cv_setB_k.csv", index=False)

    # Leave-one-group-out relative to the FINAL set, plus add-back rows for the groups CV dropped.
    ab_rows = [summarize("Set A: raw columns", df, folds, preds["Ridge (raw, Set A)"], params["Ridge (raw, Set A)"]),
               setb, setb_ext, summarize("Set C: ours", df, folds, p_ours, best_r)]
    for g, cols in final_groups.items():
        b, p, _ = tune(df, folds, [c for c in FINAL if c not in cols], ridge_grid())
        ab_rows.append(summarize(f"C minus {g}", df, folds, p, b))
    for g in drop:
        b, p, _ = tune(df, folds, FINAL + GROUPS[g], ridge_grid())
        ab_rows.append(summarize(f"C plus {g}", df, folds, p, b))
    abl = pd.DataFrame(ab_rows)
    for other, pred in (("A", preds["Ridge (raw, Set A)"]), ("B", setb_pred), ("B+", p_bext)):
        fo = fold_scores(df, folds, pred).MAE.values
        t_, p_ = stats.ttest_rel(ours_f, fo)
        g_, l_, h_ = cluster_boot_gap(df, val_idx, p_ours, pred)
        log.append(f"Set C vs Set {other}: fold gaps {np.round(ours_f - fo, 3)}, paired t p={p_:.3f}; "
                   f"pooled gap {g_:.3f} [{l_:.3f}, {h_:.3f}]")
    abl.to_csv(TAB / "cv_ablation.csv", index=False)
    log.append("Set B by k:\n" + pd.DataFrame(k_rows)[["method", "MAE_mean", "MAE_sd"]].round(4).to_string(index=False))
    log.append("ablation:\n" + abl[["method", "param", "MAE_mean", "MAE_sd", "MAE_1980", "MAE_1985", "MAE_1990",
                                    "MAE_1995"]].round(3).to_string(index=False))

    # --- slices: transition stage and region (pooled out-of-fold rows)
    no_edu_cols = [c for c in FINAL if c not in GROUPS["education_stage"]]
    b, p_noedu, _ = tune(df, folds, no_edu_cols, ridge_grid())
    sl = df.loc[val_idx].copy()
    sl["stage"] = pd.cut(sl.tfr, [0, 2.5, 4, 5.5, 99], right=False, labels=["<2.5", "2.5-4", "4-5.5", ">=5.5"])
    cols = {"Set A": preds["Ridge (raw, Set A)"], "Set C w/o education": p_noedu, "Set C": p_ours,
            "Damped trend": preds["Damped trend"], "RF (raw)": preds["Random forest (raw)"]}
    for name, p in cols.items():
        sl[f"ae_{name}"] = (sl.y - p.loc[val_idx]).abs()
    slices = []
    for key in ("stage", "region"):
        g = sl.groupby(key, observed=True)
        for lev, s in g:
            row = {"slice_by": key, "slice": lev, "n": len(s)}
            row.update({name: s[f"ae_{name}"].mean() for name in cols})
            slices.append(row)
    slices = pd.DataFrame(slices)
    slices.to_csv(TAB / "cv_slices.csv", index=False)
    log.append("slices (pooled CV MAE):\n" + slices.round(3).to_string(index=False))
    pre = sl.tfr >= knot  # rows where the stage gate is active
    gaps = {"all": cluster_boot_gap(df, sl.index, p_ours, p_noedu),
            f"TFR>={knot}": cluster_boot_gap(df, sl.index[pre], p_ours, p_noedu),
            f"TFR<{knot}": cluster_boot_gap(df, sl.index[~pre], p_ours, p_noedu)}
    pd.DataFrame([{"rows": k, "n": int(len(sl) if k == "all" else (pre if ">=" in k else ~pre).sum()),
                   "gap": g[0], "ci_lo": g[1], "ci_hi": g[2]} for k, g in gaps.items()]
                 ).to_csv(TAB / "cv_education_gap.csv", index=False)
    log.append("education effect, MAE(C) - MAE(C w/o education), country-cluster bootstrap 95% CI:\n" +
               "\n".join(f"  {k}: {g[0]:.3f} [{g[1]:.3f}, {g[2]:.3f}]" for k, g in gaps.items()))

    oof = pd.DataFrame({"iso3": df.iso3, "t": df.t, "y": df.y, **{k: v for k, v in preds.items()},
                        "Set C w/o education": p_noedu}).loc[val_idx]
    oof.to_csv(TAB / "cv_oof_predictions.csv", index=False)

    cfg = {"knot": knot, "features": FINAL, "alpha": float(best_r), "dropped_groups": drop,
           "damped_phi": float(phi), "rf_param": params["Random forest (raw)"],
           "gbm_param": params["Gradient boosting (raw)"], "ridge_raw_alpha": float(params["Ridge (raw, Set A)"]),
           "setB_alpha": float(setb["param"]), "setB_pool": pool_raw, "no_edu_alpha": float(b),
           "best_baseline_cv": best_base, "significance_cv": sig, "threshold": THRESHOLD}
    (RES / "final_config.json").write_text(json.dumps(cfg, indent=2, default=float))
    (TAB / "cv_log.txt").write_text("\n\n".join(log))
    print("\n\n".join(log))


if __name__ == "__main__":
    main()
