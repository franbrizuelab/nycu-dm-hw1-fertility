# Post-Build Assessment, 2026-10-03

Companion to `01_feasibility_assessment.md`. This document checks the finished build against `Specs.txt` and against what the pre-build assessment predicted. It also lists what the team still has to do.

## 1. Verdict

**The project holds up and the report satisfies the spec. The scientific result is honest but modest. That is a strength if it's presented confidently, and a weakness if the team tries to oversell it.**

- The core pipeline is leak-free and reproducible (`./run_all.sh`, about 3 minutes). Every number in the paper is generated from results files. The test set was scored once, under a design-hash lock.
- **Headline result:** a 5-feature linear model (TFR level, 5-year momentum, young women's schooling, a stage hinge (TFR−4)+ and schooling × hinge) reaches **test MAE 0.224** children per woman (95% CI [0.200, 0.250]). It **significantly beats** gradient boosting (gap −0.049 [−0.073, −0.025]), random forests and the damped-trend rule on the test set. In CV the gap to the best baseline is **not** significant (p = 0.53).
- **The education answer is mostly null.** In CV the education group lowers MAE by 0.017 (not significant). On test it lowers MAE by 0.004 (with the hinge's own effect included) or 0.013 (schooling given the hinge, CI [−0.027, 0.000]). In-sample, the coefficient is significant and stage-dependent: −0.041 children per extra year of schooling at TFR ≤ 4, rising to −0.179 at TFR 7.
- **Threshold:** the 0.2 decision threshold (derived from the UN's own ±0.4 variant spacing) is **missed narrowly** (0.224). The spec allows this, as long as the paper says it plainly, and it does.

## 2. Spec compliance, section by section

| Spec item (points) | Status | Where / evidence |
|---|---|---|
| Final predictor linear; other families only as baselines | ✅ | Ridge (α = 0.001); RF and GBM appear only as baselines |
| Continuous target with unit | ✅ | TFR at t+10, children per woman, range 0.81–8.86 |
| Real, documented data | ✅ | WPP 2024, Barro-Lee, PWT 10.01, WUP 2025; `data/raw/MANIFEST.tsv` with SHA-256 |
| Same split and metrics for all methods | ✅ | `results/folds.json` reused by every method; `common.metrics` |
| ACL format, ≤ 6 pages excl. refs, `[preprint]` | ✅ | Content ends on p.6 with ~10 lines spare; references start p.6/7 |
| Title, team ID, names, IDs, **GitHub URL on p.1** | ⚠️ **placeholders** | `paper/main.tex` lines with `[TEAM ID]`, `[Member n]`, `TODO-OWNER/TODO-REPO`. **Without the repo link the report scores 0.** |
| Figures/tables numbered, captioned, referenced, axes with units | ✅ | 3 figures, 6 tables, all referenced |
| Abstract (7): why, who, contribution, ≤ 150 words, main number | ✅ | 143 words; user = school/pension planners; MAE 0.224 |
| §1: user + decision, precise RQ, 2–3 contributions verified later | ✅ | Decision = which UN variant to plan for; C1→§7.1, C2→§7.2, C3→§7.2/§8.1 |
| §2 (6): ≥ 3 real sources with task/data/features/model/error, our difference | ✅ | Alkema'11, Vollset'20, Adhikari'24 (+Kebede'19, Lutz & KC'11, Bongaarts'03); Table 1; all DOIs checked on Crossref |
| §3 (12): target, unit, range, row, inputs and prediction time, source/period/size, summary-stats table, justified preprocessing | ✅ | Task card + Table 2 + six justified preprocessing steps |
| §4 (10): target-vs-feature figure, Pearson/Spearman/MI, redundancy, keep/transform/drop decisions, H1…, training only | ✅ | Fig. 1 (4 panels), H1–H7, each with a decision; association table in repo |
| §5 (13): pipeline figure; ≥ 1 engineered group with rationale tied to §4; variant justified by data; CV tuning on training only; range and selected value reported | ✅ | Fig. 2 (TikZ); groups tied to H2/H5/H6; Ridge justified by built-in collinearity (VIF 16/12); 13-value α grid and curve |
| §6 (10): held-out test scored once; time-aware CV; fold count justified; mean ± sd; identical folds; ≥ 2 metrics justified; decision threshold; trivial/simple/strong baselines | ✅ | Embargoed rolling origin (4 folds); lock file; MAE/RMSE/R²/Hit@0.2; threshold 0.2 from WPP methodology; 8 baselines |
| §7.1 (8): all baselines, bold best, threshold comparison, why win/lose vs strong baseline, named significance test | ✅ | Table 3; country-cluster bootstrap + paired t over folds |
| §7.2 (10): Sets A/B/C, same model/split/tuning, B selected inside folds with k justified, slices, cost discussion | ✅ | Table 4 (fold-level), B with k = |C| = 5 inside folds, B+ variant, stage slices (Fig. 3b), cost paragraph |
| §8.1 (8): standardized β, 95% CI, p, named test, VIF for every feature; hypotheses checked; reversals; practical vs statistical | ✅ | Table 5 (OLS with country-clustered SE, Wald t); Ridge cluster-bootstrap CIs agree within 0.01; H1–H6 revisited; effects in children per woman |
| §8.2 (8): ≥ 3 worst cases with numbers and context; causes; shared pattern; first fix | ✅ | Table 6 (5 cases); bias +0.15 at the 2005 origin; East Asia opposite failure; Taiwan; first fix named |
| §9 (5): answer RQ with numbers, which features mattered, recommendation, limitations, next experiment | ✅ | Includes data-vintage, CV-optimism, linearity, threshold and causal caveats |
| Contributions + AI usage (3) | ⚠️ **team must write** | Bullets are placeholders; the AI statement is drafted and needs the team's part |
| References real and verifiable | ✅ | 14 entries; each checked via Crossref or the publisher; one wrong first name (Adhikari) was caught and fixed |
| Repo: code, datasets, short README | ✅ locally / ⚠️ not pushed | `README.md`, `requirements.txt`, `run_all.sh`, `data/raw` (27 MB, under GitHub's limits) |

## 3. Contribution claims vs. evidence

| Claim | Supported? | Caveat to state when presenting |
|---|---|---|
| C1: the linear model beats tree ensembles and damped trend on 2015/2020 | **Yes on test** (CIs exclude 0) | **Not significant in CV.** Say "consistent edge", as the paper does. |
| C2: level + momentum carry most skill; C beats A and B | **Yes vs A** (every fold, pooled CI excludes 0) | vs B: 3 of 4 folds, CI includes 0. Momentum is the feature ranking can't find. |
| C3: education stage-dependent in sample, small out of sample | **Yes** | This is the main finding. Don't rephrase it as "education doesn't matter". It matters as an association, but it adds little forecast skill once momentum is known. |

## 4. What the pre-build assessment predicted vs. what happened

| Prediction (01_feasibility) | Outcome |
|---|---|
| Education adds ~nothing on average beyond TFR + trend | **Confirmed** (test gain 0.004–0.013) |
| Education helps in pre-transition countries (~11% in the pilot) | **Confirmed in CV only** (TFR ≥ 4: −0.032, CI includes 0). **Not on test**: only 61 test rows are above the knot, and the gain vanished there. |
| "Trees can't extrapolate below training lows" is mostly false | Right to drop it. The real reason trees lose is coarse piecewise-constant fits in the mid-transition band (MAE 0.50 vs 0.32 at TFR 4–5.5). |
| Threshold 0.2 might not be met | Missed narrowly (0.224) |
| Non-stationarity (H5 in the brief) | Confirmed: the education slope shrinks across decades, and the bias at the 2005 origin is +0.15 (stalls and rebounds) |
| RF vs Ridge would be a real contest | Yes: a tie in CV, Ridge ahead on test |

## 5. Integrity notes (all disclosed in the paper or here)

1. **Test-set peek before modelling:** during feasibility, baseline-only statistics (persistence/damped-trend MAE, and which test countries fall below TFR 1) were computed on test rows. No model was fitted on test data. This is disclosed in §6.
2. **Lock migration:** the first lock hashed the entire config file. A full rerun changed reported CV statistics at the 1e-15 level (parallel random forest), so the lock now hashes design fields only. The design fields were verified identical before migrating (see `results/TEST_LOCK.json`, "note").
3. **Post-freeze diagnostics:** "hinge kept, schooling removed", "adult instead of young schooling" and "plus gender gap" were run after the test scoring, as analysis only. They did not change the final model, and the paper labels them as diagnostics.
4. **CV optimism:** knot choice and backward group elimination reused the same 4 folds, so the CV MAE of the chosen design (0.328) is optimistic. This is stated in Limitations; the test set is the unbiased check.
5. **Data vintage:** WPP 2024 back-estimates use hindsight. This is stated in Limitations.

## 6. Weak points a grader (or the LLM TA judge) may probe

- **"Your education feature doesn't help, so why keep it?"** Answer: it was chosen by CV before the test set was seen, and it still helps given the hinge (−0.013). The honest null result *is* the contribution (C3).
- **Four CV folds.** Defended by the 5-year data spacing. Paired t-tests over 4 folds have little power, which is why the country-cluster bootstrap exists.
- **Creativity.** The education–fertility topic is classic. The distinctive parts are the planner/UN-variant decision framing, the embargoed protocol, cohort logic (Fig. 1c) and the "significant ≠ predictive" lesson. Lead with those on the slides.
- **Region dummies were dropped** even though there is regional bias at the 2005 origin. Leave-one-region-out shows the model transfers (Δ ≤ 0.025), and the bias is temporal rather than regional.

## 7. What the team must do (checklist)

1. **Create the GitHub repo and push.** `git init` is done locally; nothing is committed yet. Put the URL in `\repourl` in `paper/main.tex` (page 1). **This is mandatory: without it the report scores 0.**
2. Fill in `\teamid`, member names and student IDs (title block), the **Contributions** bullets, and the team's part of the **AI usage** statement. About 10 lines of space remain on page 6; check that the contributions don't push content onto page 7 (references may spill, content may not).
3. Rebuild with `cd paper && latexmk -pdf main.tex`, or upload `paper/` (including `generated/` and `../results/figures/`) to Overleaf. On Overleaf, copy the figures into the project and fix the two `\includegraphics` paths.
4. Read the paper end to end. You must be able to defend every claim in the Q&A.
5. Build the 8-minute slides (the human side). Suggested figure reuse: Fig. 1a/1c (the key insight), the protocol diagram (embargo), Table 3, Fig. 3b, Table 6 plus Taiwan.
6. Submit `HW1_<TeamID>.zip` containing `HW1_<TeamID>_report.pdf` and `HW1_<TeamID>_presentation.pdf/.pptx` (exact names, or −5 points), by **Mon 2026-10-19 23:59**.
7. Optional: submit to the LLM TA Judge for the +6% bonus.
