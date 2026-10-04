"""Turn results/tables/*.csv into LaTeX tables and number macros for the paper.

Every number quoted in paper/main.tex comes from paper/generated/*.tex, so the
paper cannot drift from the experiments.
"""
import json
import re

import pandas as pd

from common import RES, ROOT, TAB, THRESHOLD

GEN = ROOT / "paper" / "generated"
GEN.mkdir(parents=True, exist_ok=True)
macros = {}
values = {}  # unformatted numbers, for macros derived from other macros


def mac(name, value, fmt="{:.3f}"):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    values[name] = value
    txt = fmt.format(value) if not isinstance(value, str) else value
    # Typeset minus signs properly in both text and math mode.
    macros[name] = txt.replace("-", "\\ensuremath{-}", 1) if txt.startswith("-") else txt


def f3(x):
    return f"{x:.3f}"


def bold_min(vals):
    m = min(vals)
    return [f"\\textbf{{{f3(v)}}}" if abs(v - m) < 5e-4 else f3(v) for v in vals]


def bold_max(vals):
    m = max(vals)
    return [f"\\textbf{{{f3(v)}}}" if abs(v - m) < 5e-4 else f3(v) for v in vals]


def main():
    cfg = json.loads((RES / "final_config.json").read_text())

    # ---------------------------------------------------------------- summary statistics (Section 3)
    s = pd.read_csv(TAB / "summary_stats_train.csv")
    short = ["TFR$_{t+10}$ (target)", "TFR$_t$", "$\\Delta_5$TFR$_t$", "Fem.\\ school.\\ 15--24 (yr)",
             "Fem.\\ school.\\ 25--64 (yr)", "Male school.\\ 15--24 (yr)", "Under-5 mort.\\ (\\textperthousand)",
             "GDP p.c.\\ (k\\$ PPP)", "Urban share (\\%)"]
    lines = []
    for lab, (_, r) in zip(short, s.iterrows()):
        k = 1000 if "GDP" in r.Variable else 1
        lines.append(f"{lab} & {r.Mean / k:.2f} & {r.SD / k:.2f} & {r.Min / k:.2f} & {r.Max / k:.2f} & {r['Missing %']:.0f} \\\\")
    (GEN / "tab_summary.tex").write_text("\n".join(lines))
    sc = pd.read_csv(TAB / "split_counts.csv").set_index("split")
    for sp in ("train", "test", "gap"):
        mac(f"rows{sp.capitalize()}", int(sc.loc[sp, "rows"]), "{:d}")
    mac("nCountries", int(sc.loc["train", "countries"]), "{:d}")
    mac("gdpMissing", s.loc[s.Variable.str.startswith("GDP"), "Missing %"].iloc[0], "{:.0f}")

    # ---------------------------------------------------------------- EDA numbers (Section 4)
    a = pd.read_csv(TAB / "eda_association.csv").set_index("feature")
    mac("rEduY", a.loc["edu_f1524", "pearson_y"], "{:.2f}")
    mac("rhoEduY", a.loc["edu_f1524", "spearman_y"], "{:.2f}")
    mac("rTfrY", a.loc["tfr", "pearson_y"], "{:.2f}")
    mac("rDfiveDy", a.loc["d5", "pearson_dy"], "{:.2f}")
    mac("partialEduTfr", a.loc["edu_f1524", "partial_y_given_tfr"], "{:.2f}")
    mac("partialQfiveTfr", a.loc["log_q5", "partial_y_given_tfr"], "{:.2f}")
    mac("miEdu", a.loc["edu_f1524", "MI_y"], "{:.2f}")
    mac("miTfr", a.loc["tfr", "MI_y"], "{:.2f}")
    log = (TAB / "eda_log.txt").read_text()
    grab = lambda pat: float(re.search(pat, log).group(1))  # noqa: E731
    mac("rFemMale", grab(r"r\(edu_f1524, edu_m1524\) = ([-\d.]+)"), "{:.2f}")
    mac("rTfrQfive", grab(r"r\(tfr, log_q5\) = ([-\d.]+)"), "{:.2f}")
    mac("partialGap", grab(r"partial r\(gap, y \| avg schooling, tfr, tfr2\) = ([-\d.]+)"), "{:.2f}")
    mac("partialEduGdpQ", grab(r"H4 partial r\(edu_f1524, y \| \['log_gdppc', 'log_q5'\]\) = ([-\d.]+)"), "{:.2f}")
    mac("partialEduAll", grab(r"H4 partial r\(edu_f1524, y \| \['tfr', 'tfr2', 'd5', 'log_gdppc', 'log_q5'\]\) = ([-\d.]+)"), "{:.2f}")
    v = pd.read_csv(TAB / "eda_vif_raw.csv", index_col=0).iloc[:, 0]
    mac("vifEduMax", v[[c for c in v.index if c.startswith("edu")]].max(), "{:.0f}")
    lag = pd.read_csv(TAB / "eda_cohort_lag.csv").set_index("band")
    mac("lagYoung", lag.loc["20-24", "partial_y_given_tfr"], "{:.2f}")
    mac("lagOld", lag.loc["55-59", "partial_y_given_tfr"], "{:.2f}")
    st = pd.read_csv(TAB / "eda_stage_slices.csv").set_index("stage")
    mac("stagePreR", st.loc[">=5.5", "partial_r"], "{:.2f}")
    mac("stagePreLo", st.loc[">=5.5", "ci_lo"], "{:.2f}")
    mac("stagePreHi", st.loc[">=5.5", "ci_hi"], "{:.2f}")
    mac("stageMidR", st.loc["4-5.5", "partial_r"], "{:.2f}")
    mac("stageLowR", st.loc["<2.5", "partial_r"], "{:.2f}")
    mac("stagePreN", int(st.loc[">=5.5", "n"]), "{:d}")
    stab = pd.read_csv(TAB / "eda_stability.csv").set_index("decade")
    mac("slopeSixties", stab.loc[1960, "slope_edu"], "{:.3f}")
    mac("slopeNineties", stab.loc[1990, "slope_edu"], "{:.3f}")

    # ---------------------------------------------------------------- method / tuning (Section 5)
    mac("knot", cfg["knot"], "{:.0f}")
    mac("alphaSel", cfg["alpha"], "{:g}")
    mac("nFeatures", len(cfg["features"]), "{:d}")
    mac("dampedPhi", cfg["damped_phi"], "{:.1f}")
    tune = pd.read_csv(TAB / "cv_tuning_ridge.csv")
    cells = " & ".join(f"{float(p):g}" for p in tune.param[::2])
    vals = " & ".join(f"{m:.3f}" for m in tune.MAE_mean[::2])
    (GEN / "tab_tuning.tex").write_text(f"$\\alpha$ & {cells} \\\\\nCV MAE & {vals} \\\\")
    el = pd.read_csv(TAB / "cv_group_elimination.csv")
    mac("cvAllGroups", el.MAE_mean.iloc[0])
    kn = pd.read_csv(TAB / "cv_knot_final.csv").set_index("knot")
    mac("cvKnotFiveFive", kn.loc[5.5, "MAE_mean"])
    var = pd.read_csv(TAB / "cv_variants.csv").set_index("method")
    mac("cvOLS", var.loc["OLS", "MAE_mean"])
    mac("cvLasso", var.loc["Lasso", "MAE_mean"])

    # ---------------------------------------------------------------- main results (Section 7.1)
    r = pd.read_csv(TAB / "test_main.csv")
    order = ["Training mean", "Persistence", "Damped trend", "Single-feature OLS (schooling)",
             "Random forest (raw)", "Gradient boosting (raw)", "Ridge (raw, Set A)", "Random forest (Set C)",
             "Ours: Ridge + engineered (Set C)"]
    label = {"Training mean": "Training mean", "Persistence": "Persistence",
             "Damped trend": "Damped trend",
             "Single-feature OLS (schooling)": "OLS, schooling only",
             "Random forest (raw)": "Random forest (raw)", "Gradient boosting (raw)": "Grad.\\ boosting (raw)",
             "Ridge (raw, Set A)": "Ridge (raw)", "Random forest (Set C)": "Rand.\\ forest (our feat.)",
             "Ours: Ridge + engineered (Set C)": "\\textit{Ours}: Ridge + feat."}
    r = r.set_index("method").loc[order].reset_index()
    cols = {"CV_MAE": bold_min(r.CV_MAE), "CV_RMSE": bold_min(r.CV_RMSE), "Test_MAE": bold_min(r.Test_MAE),
            "Test_RMSE": bold_min(r.Test_RMSE), "Test_R2": bold_max(r.Test_R2), "Test_Hit": bold_max(r.Test_Hit)}
    lines = []
    for i, row in r.iterrows():
        sep = "\\midrule\n" if row.method == "Ours: Ridge + engineered (Set C)" else ""
        lines.append(f"{sep}{label[row.method]} & {cols['CV_MAE'][i]}\\,{{\\tiny$\\pm${row.CV_MAE_sd:.2f}}} & "
                     f"{cols['Test_MAE'][i]} & {cols['Test_RMSE'][i]} & "
                     f"{cols['Test_R2'][i].replace('0.', '.')} & {cols['Test_Hit'][i].replace('0.', '.')} \\\\")
    (GEN / "tab_main.tex").write_text("\n".join(lines))
    R = r.set_index("method")
    for key, nm in (("Ours", "Ours: Ridge + engineered (Set C)"), ("Gbm", "Gradient boosting (raw)"),
                    ("Rf", "Random forest (raw)"), ("RfC", "Random forest (Set C)"), ("Damped", "Damped trend"),
                    ("Persist", "Persistence"), ("RidgeRaw", "Ridge (raw, Set A)"), ("Single", "Single-feature OLS (schooling)")):
        mac(f"cvMae{key}", R.loc[nm, "CV_MAE"])
        mac(f"cvSd{key}", R.loc[nm, "CV_MAE_sd"])
        mac(f"teMae{key}", R.loc[nm, "Test_MAE"])
        mac(f"teRmse{key}", R.loc[nm, "Test_RMSE"])
        mac(f"teHit{key}", 100 * R.loc[nm, "Test_Hit"], "{:.0f}")
        mac(f"teRsq{key}", R.loc[nm, "Test_R2"], "{:.2f}")
    sig = pd.read_csv(TAB / "test_significance.csv").set_index("vs")
    for key, nm in (("Gbm", "Gradient boosting (raw)"), ("Rf", "Random forest (raw)"), ("RfC", "Random forest (Set C)"),
                    ("Damped", "Damped trend"), ("RidgeRaw", "Ridge (raw, Set A)")):
        mac(f"gap{key}", sig.loc[nm, "gap"])
        mac(f"gap{key}Lo", sig.loc[nm, "ci_lo"])
        mac(f"gap{key}Hi", sig.loc[nm, "ci_hi"])
    s = cfg["significance_cv"]
    mac("cvPairedP", s["paired_p"], "{:.2f}")
    mac("cvPairedT", s["paired_t"], "{:.2f}")
    mac("cvGapBest", s["pooled_gap"])
    mac("cvGapBestLo", s["ci_lo"])
    mac("cvGapBestHi", s["ci_hi"])
    tl = (TAB / "test_log.txt").read_text()
    m = re.search(r"ours test MAE ([\d.]+) \[([\d.]+), ([\d.]+)\]", tl)
    mac("teMaeLo", float(m.group(2)))
    mac("teMaeHi", float(m.group(3)))
    mac("threshold", THRESHOLD, "{:.1f}")
    m = re.findall(r"t=(\d+): MAE ours ([\d.]+), bias ([+-][\d.]+); persistence MAE ([\d.]+)", tl)
    for (t, mae, bias, pm), key in zip(m, ("A", "B")):
        mac(f"teMaeYear{key}", float(mae))
        mac(f"teBiasYear{key}", float(bias), "{:+.2f}")
    mac("minPred", float(re.search(r"min prediction ours ([\d.]+)", tl).group(1)), "{:.2f}")

    # ---------------------------------------------------------------- ablation (Section 7.2)
    ab = pd.read_csv(TAB / "test_ablation.csv").set_index("set")
    cva = pd.read_csv(TAB / "cv_ablation.csv").set_index("method")
    rows = [("A: raw columns", "Set A: raw columns"),
            (f"B: top-{len(cfg['features'])} by $|\\rho|$", f"Set B: top-{len(cfg['features'])} raw by |Spearman|"),
            ("C: ours", "Set C: ours")]
    fold_cols = ["MAE_1980", "MAE_1985", "MAE_1990", "MAE_1995"]
    z = lambda x: f"{x:.3f}".replace("0.", ".", 1)  # noqa: E731  drop leading zero to save width

    def line(lab, mean, sd, folds, test):
        return (f"{lab} & {z(mean)}\\,{{\\tiny$\\pm${sd:.2f}}} & " + " & ".join(z(x) for x in folds)
                + f" & {test} \\\\")
    lines = []
    for lab, nm in rows:
        lines.append(line(lab, cva.loc[nm, "MAE_mean"], cva.loc[nm, "MAE_sd"], cva.loc[nm, fold_cols],
                          z(ab.loc[nm, "Test_MAE"])))
    lines.append("\\midrule")
    dg = pd.read_csv(TAB / "diag_hinge_vs_education.csv").set_index("set")
    extra = [("C $-$ level", "C minus level"), ("C $-$ momentum", "C minus momentum"),
             ("C $-$ education", "C minus education_stage"),
             ("C $+$ TFR$^2$", "C plus curvature"), ("C $+$ region", "C plus region"),
             ("C $+$ develop.", "C plus development_logs")]
    for lab, nm in extra:
        lines.append(line(lab, cva.loc[nm, "MAE_mean"], cva.loc[nm, "MAE_sd"], cva.loc[nm, fold_cols],
                          z(ab.loc[nm, "Test_MAE"]) if nm in ab.index else "--"))
    d = dg.loc["C minus education, keep TFR hinge"]
    lines.append(line("C $-$ school., keep $h$", d.CV_MAE, d.CV_MAE_sd, json.loads(d.fold_MAE), z(d.Test_MAE)))
    (GEN / "tab_ablation.tex").write_text("\n".join(lines))
    mac("abA", cva.loc["Set A: raw columns", "MAE_mean"])
    mac("abB", cva.loc[rows[1][1], "MAE_mean"])
    mac("abBplus", cva.loc[f"Set B+: top-{len(cfg['features'])} of raw+engineered", "MAE_mean"])
    mac("abC", cva.loc["Set C: ours", "MAE_mean"])
    mac("abNoEdu", cva.loc["C minus education_stage", "MAE_mean"])
    mac("abNoMom", cva.loc["C minus momentum", "MAE_mean"])
    mac("teA", ab.loc["Set A: raw columns", "Test_MAE"])
    mac("teB", ab.loc[rows[1][1], "Test_MAE"])
    mac("teNoEdu", ab.loc["C minus education_stage", "Test_MAE"])
    mac("teHinge", d.Test_MAE)
    mac("cvHinge", d.CV_MAE)
    mac("cvAdult", dg.loc["C with adult (25-64) schooling", "CV_MAE"])
    mac("cvGap", dg.loc["C plus gender gap", "CV_MAE"])
    cvlog = (TAB / "cv_log.txt").read_text()
    for key, other in (("A", "A"), ("B", "B")):
        m = re.search(rf"Set C vs Set {other}: fold gaps \[([^\]]+)\], paired t p=([\d.]+); pooled gap ([-\d.]+) \[([-\d.]+), ([-\d.]+)\]", cvlog)
        mac(f"pCvs{key}", float(m.group(2)), "{:.2f}")
        mac(f"gapCvs{key}", float(m.group(3)))
        mac(f"gapCvs{key}Lo", float(m.group(4)))
        mac(f"gapCvs{key}Hi", float(m.group(5)))
    eg = pd.read_csv(TAB / "cv_education_gap.csv").set_index("rows")
    mac("cvEduGapAll", eg.loc["all", "gap"])
    mac("cvEduGapAllLo", eg.loc["all", "ci_lo"])
    mac("cvEduGapAllHi", eg.loc["all", "ci_hi"])
    k = f"TFR>={cfg['knot']}"
    mac("cvEduGapHigh", eg.loc[k, "gap"])
    mac("cvEduGapHighLo", eg.loc[k, "ci_lo"])
    mac("cvEduGapHighHi", eg.loc[k, "ci_hi"])
    mac("cvEduGapHighN", int(eg.loc[k, "n"]), "{:d}")
    for lab, key in (("all", "All"), (f"TFR>={cfg['knot']}", "High"), (f"TFR<{cfg['knot']}", "Low")):
        m = re.search(rf"education gap on test, {re.escape(lab)} \(n=(\d+)\): ([-\d.]+) \[([-\d.]+), ([-\d.]+)\]", tl)
        mac(f"teEduGap{key}", float(m.group(2)))
        mac(f"teEduGap{key}Lo", float(m.group(3)))
        mac(f"teEduGap{key}Hi", float(m.group(4)))
        mac(f"teEduGap{key}N", int(m.group(1)), "{:d}")
        m = re.search(rf"education gap given the hinge, test, {re.escape(lab)}: ([-\d.]+) \[([-\d.]+), ([-\d.]+)\]", tl)
        mac(f"teHingeGap{key}", float(m.group(1)))
        mac(f"teHingeGap{key}Lo", float(m.group(2)))
        mac(f"teHingeGap{key}Hi", float(m.group(3)))
    sl = pd.read_csv(TAB / "test_slices.csv")
    st = sl[sl.by == "stage_bin"].set_index("slice")
    for key, lev in (("Low", "<2.5"), ("Mid", f"2.5-{cfg['knot']:g}"), ("Onset", f"{cfg['knot']:g}-5.5"), ("Pre", ">=5.5")):
        mac(f"slC{key}", st.loc[lev, "Set C"])
        mac(f"slNoEdu{key}", st.loc[lev, "C w/o edu"])
        mac(f"slGbm{key}", st.loc[lev, "GBM (raw)"])
        mac(f"slN{key}", int(st.loc[lev, "n"]), "{:d}")
    rg = sl[sl.by == "region"].set_index("slice")
    mac("slSsaC", rg.loc["Sub-Saharan Africa", "Set C"])
    mac("slSsaBias", rg.loc["Sub-Saharan Africa", "bias C"], "{:+.2f}")
    mac("slMenaC", rg.loc["Middle East and North Africa", "Set C"])
    mac("slMenaBias", rg.loc["Middle East and North Africa", "bias C"], "{:+.2f}")
    mac("slAdvC", rg.loc["Advanced Economies", "Set C"])
    lor = pd.read_csv(TAB / "test_leave_region_out.csv").set_index("region")
    mac("lorMaxDiff", (lor.MAE_held_out - lor.MAE_in).abs().max())
    mac("lorSsaIn", lor.loc["Sub-Saharan Africa", "MAE_in"])
    mac("lorSsaOut", lor.loc["Sub-Saharan Africa", "MAE_held_out"])

    # ---------------------------------------------------------------- coefficients (Section 8.1)
    c = pd.read_csv(TAB / "coefficients.csv")
    names = {"tfr": "TFR$_t$", "d5": "$\\Delta_5$TFR$_t$", "edu_f1524": "Fem.\\ school.\\ 15--24",
             "edu_x_stage": "School.\\ $\\times$ stage", "stage": f"Stage $(\\mathrm{{TFR}}_t-{cfg['knot']:g})_+$"}
    hyp = {"tfr": "--", "d5": "H6", "edu_f1524": "H1, H4", "edu_x_stage": "H5", "stage": "H5"}
    lines = []
    for _, row in c.iterrows():
        p = "$<$0.001" if row.p < 0.001 else f"{row.p:.3f}"
        lines.append(f"{names[row.feature]} & {row.beta_ols:+.3f} & [{row.ci_lo:+.3f}, {row.ci_hi:+.3f}] & {p} & "
                     f"{row.vif:.1f} \\\\")
    (GEN / "tab_coef.tex").write_text("\n".join(lines))
    C = c.set_index("feature")
    mac("bTfr", C.loc["tfr", "beta_ols"])
    mac("bDfive", C.loc["d5", "beta_ols"])
    mac("bEdu", C.loc["edu_f1524", "beta_ols"])
    mac("bEduX", C.loc["edu_x_stage", "beta_ols"])
    mac("vifMax", C.vif.max(), "{:.0f}")
    mac("ridgeCiDiff", max((C.ridge_boot_lo - C.ci_lo).abs().max(), (C.ridge_boot_hi - C.ci_hi).abs().max()), "{:.2f}")
    eff = pd.read_csv(TAB / "education_effect_by_stage.csv").set_index("tfr_t")
    for key, t in (("Two", 2.0), ("Five", 5.0), ("Seven", 7.0)):
        mac(f"eff{key}", eff.loc[t, "effect_per_year"], "{:.3f}")
        mac(f"eff{key}Lo", eff.loc[t, "ci_lo"], "{:.3f}")
        mac(f"eff{key}Hi", eff.loc[t, "ci_hi"], "{:.3f}")
    m = re.search(r"R2=([\d.]+)", tl)
    mac("olsRsq", float(m.group(1)), "{:.2f}")

    # ---------------------------------------------------------------- error analysis (Section 8.2)
    w = pd.read_csv(TAB / "test_worst_cases.csv")
    lines = []
    for i, row in w.head(5).iterrows():
        nm = {"Central African Republic": "C.\\ African Rep."}.get(row.country, row.country)
        lines.append(f"{nm} & {row.tfr:.2f} & {row.d5:+.2f} & {row.edu_f1524:.1f} & "
                     f"{row.y:.2f} & {row.pred:.2f} & {row.err:+.2f} \\\\")
    (GEN / "tab_errors.tex").write_text("\n".join(lines))
    ea = pd.read_csv(TAB / "test_east_asia.csv")
    for iso, key in (("Republic of Korea", "Kor"), ("Taiwan", "Twn"), ("China", "Chn")):
        rr = ea[(ea.country == iso) & (ea.target_year == 2020)].iloc[0]
        mac(f"ea{key}Y", rr.y, "{:.2f}")
        mac(f"ea{key}P", rr.pred, "{:.2f}")
        mac(f"ea{key}E", rr.err, "{:+.2f}")
    rr = ea[(ea.country == "Taiwan") & (ea.target_year == 2015)].iloc[0]
    mac("eaTwnFifteenY", rr.y, "{:.2f}")

    # ---------------------------------------------------------------- derived numbers quoted in the text
    from common import load_panel, RAW
    panel = load_panel()
    mac("yMin", panel.y.min(), "{:.2f}")
    mac("yMax", panel.y.max(), "{:.2f}")
    wpp = pd.read_csv(RAW / "WPP2024_Demographic_Indicators_Medium.csv.gz", usecols=["ISO3_code", "Time", "TFR"], low_memory=False)
    tw = wpp[wpp.ISO3_code == "TWN"].set_index("Time").TFR
    mac("twnSixty", tw.loc[1960], "{:.1f}")
    mac("twnTwentyThree", tw.loc[2023], "{:.2f}")
    mac("eduGainLo", abs(values["teEduGapAll"]))
    mac("gapGbmAbs", abs(values["gapGbm"]))
    mac("gapDampedAbs", abs(values["gapDamped"]))
    mac("cvEduGainAbs", abs(values["cvEduGapAll"]))
    mac("eduGainHi", abs(values["teHingeGapAll"]))
    raw = re.search(r"raw-unit OLS coefficients:\n(.*?)\n\n", tl + "\n\n", re.S).group(1)
    rawc = {l.split()[0]: float(l.split()[1]) for l in raw.strip().splitlines()}
    mac("bDfiveRaw", rawc["d5"], "{:.2f}")
    mac("effThreeLow", 3 * abs(eff.loc[2.0, "effect_per_year"]), "{:.2f}")
    mac("effThreeHigh", 3 * abs(eff.loc[7.0, "effect_per_year"]), "{:.2f}")
    mac("vifTfr", C.loc["tfr", "vif"], "{:.0f}")
    mac("vifStage", C.loc["stage", "vif"], "{:.0f}")
    mac("abNoLevel", cva.loc["C minus level", "MAE_mean"])
    mac("rDfiveY", abs(a.loc["d5", "pearson_y"]), "{:.2f}")
    mac("thrExcess", R.loc["Ours: Ridge + engineered (Set C)", "Test_MAE"] - THRESHOLD)
    fl = json.loads((RES / "folds.json").read_text())
    sizes = [len(f["train"]) for f in fl.values()]
    mac("foldMinRows", min(sizes), "{:d}")
    mac("foldMaxRows", max(sizes), "{:d}")
    mac("eaTwnFifteenP", rr.pred, "{:.2f}")

    # Proper minus signs in generated table cells ("-0.26" -> "\ensuremath{-}0.26").
    for f in GEN.glob("tab_*.tex"):
        f.write_text(re.sub(r"(?<![\w$\-{])-(?=\d)", r"\\ensuremath{-}", f.read_text()))
    (GEN / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    print(f"wrote {len(macros)} macros and tables to {GEN}")


if __name__ == "__main__":
    main()
