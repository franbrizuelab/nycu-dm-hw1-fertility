# Feasibility Assessment (pre-build), 2026-10-03
Project: forecasting a country's fertility 10 years ahead from women's schooling (NYCU Data Mining HW1)

## Context
You asked for an honest check, before any code, of whether the fertility brief (`fertility_project_brief.txt`) can carry an HW1 that scores well under `Specs.txt`. You also want a second assessment when the work is done. The slides and the presentation are yours; I build everything up to and including the paper draft.

What I did in plan mode (read-only): downloaded the real data sources to memory only, checked coverage, and ran a **pilot on training-period years only** (start years ≤ 1995), plus a few citation checks. Part A below gets saved as `docs/01_feasibility_assessment.md`. Part B is how I'll execute it.

---

# PART A: FEASIBILITY ASSESSMENT (pre-build)

## A1. Verdict
**Yes, it's a good base. It is not safe as written.** The data exist, are free, include Taiwan, and fit every hard constraint in the spec. The temporal-embargo protocol in the brief is correct, and it's the brief's best part. **But the headline hypothesis, "women's education predicts fertility 10 years ahead," fails in the pilot when tested the way the brief tests it.** Once current TFR and its recent trend are in the model, adding education leaves CV error the same or makes it worse. Education only helps in one place: countries whose fertility has **not started falling yet**. There it cuts error by about 11%.

So the project is feasible and can score well, **if we reframe the research question around that finding** (A5). Under the original framing we would likely claim contributions that §7 contradicts. That's a listed deduction ("contributions that later results do not support"), and it also weakens Results (18 pts) and Ablation (10 pts).

## A2. Data: verified now, not assumed
| Source | Verified fact | Consequence |
|---|---|---|
| Barro-Lee v3 (`BL_v3_F/M.csv`, GitHub barrolee/BarroLeeDataSet) | 146 countries, 1950–2015 in 5-year steps, **10-year age bands only** (15-24, 25-34, …), by sex. Taiwan is included (code TWN). | Brief §4.2 assumes 5-year bands. v3 doesn't have them. |
| Barro-Lee v2.2 (`BL2013_F_v2.2.csv`) | **5-year bands** (15-19, 20-24, 25-29, …), 1950–**2010**, 146 countries. | Enough for the cohort lag-curve figure (H2), since every start year we use is ≤ 2010. The main features come from v3, with v2.2 as a robustness check. |
| UN WPP 2024 Demographic Indicators CSV (16 MB gz, direct URL works) | TFR and Q5 (under-5 mortality) yearly, 1950–2023 estimates. Taiwan = "China, Taiwan Province of China". | Targets up to 2020 are observed estimates. Start years t ≤ 2010 are therefore valid, and t = 2015 is not (its target would be a projection). |
| Join | 144 of 146 Barro-Lee countries match WPP by ISO3. The other 2 are code aliases (ROM→ROU, SER→SRB), which is trivial to fix. | Panel: **1,440 training rows (t = 1950–1995), 288 test rows (t = 2005, 2010)**. |
| PWT (GDP), UN WUP (urbanization) | Not yet downloaded. Both are documented as covering Taiwan. | Optional features. The pilot suggests they add little (see A3). I'll verify them in step 1. |

Not checked yet: World Bank/UNESCO, which don't matter because they exclude Taiwan, as the brief already notes.

## A3. Pilot evidence (training years only, rolling origin with a 10-year embargo)
Folds: validate at t ∈ {1980, 1985, 1990, 1995}, train on start years ≤ t−10. Ridge with standardized features and α chosen internally. Female schooling = Barro-Lee v3, women 15-24. This is a crude version: no region, no tuning beyond RidgeCV, no cohort features. Validation MAE in children per woman:

| Feature set | 1980 | 1985 | 1990 | 1995 | mean |
|---|---|---|---|---|---|
| Persistence TFR(t+10)=TFR(t) | .645 | .752 | .722 | .524 | .661 |
| Damped trend (½ of last decade's change) | .499 | .564 | .462 | .300 | .456 |
| Ridge [TFR, ΔTFR₁₀] | .425 | .433 | .340 | .274 | .368 |
| Ridge [TFR, ΔTFR, TFR²] | .371 | .394 | .325 | .266 | **.339** |
| … + TFR×edu | .380 | .380 | .333 | .291 | .346 |
| … + hinge max(0,10−edu) and its TFR interaction | .382 | .375 | .329 | .283 | .342 |
| … + linear edu | .438 | .403 | .327 | .297 | .366 |
| … + log child mortality | — | — | — | — | .400 (worse) |
| Random forest (raw: TFR, ΔTFR, 3 education measures, Q5, gender gap) | .346 | .375 | .384 | .318 | .356 |

Slices (pooled validation rows, education interaction vs. no education):
| Stage at t | n | no edu | with edu |
|---|---|---|---|
| Pre-transition, TFR ≥ 5.5 | 158 | .511 | **.456** |
| Mid-transition, 3–5.5 | 185 | .306 | .365 (worse) |
| Late, < 3 | 237 | .251 | .256 |

Also: education correlates with the *future 10-year change* (r ≈ 0.29–0.47), but once TFR(t) is controlled the residual correlation drops to −0.15 to +0.02.

**Reading:** current fertility already absorbs most of what education says. Fertility's own momentum (level, trend, curvature) is the forecast workhorse. Education carries information about **when the decline begins**, and momentum can't supply that, because a pre-transition country has a flat trend. That's a clean, defensible, demographically meaningful result. It also sits close to Kebede, Goujon & Lutz (2019), who link African fertility stalls to disrupted female education.

Disclosure for integrity: while checking coverage I also computed descriptive **baseline-only** statistics on the test rows (persistence MAE 0.38, damped trend 0.29, and which test countries fall below TFR 1.0). No model was fitted or evaluated on test data, and no design choice below uses those numbers. The threshold comes from UN methodology, not from them. The paper's §6 will state this.

## A4. Requirement-by-requirement scorecard (spec section → how this topic supports it → risk)
| Spec section (pts) | Support from this topic | Risk | Must fix now |
|---|---|---|---|
| Hard rules: linear final model, continuous target with unit, documented real data, same splits/metrics | TFR in children per woman, range about 0.7–8. Ridge as the final model. UN/Barro-Lee data are citable. | Low | Folds are saved to a file and reused by every method. |
| Abstract/Intro (7) | Clear user (planners: school capacity, pensions) and a clear decision (which UN scenario to plan for). Taiwan school mergers as the hook. | Low | The research question must match what the data can answer (A5). |
| Related work (6) | Strong verifiable literature with reported errors (UN BayesTFR, IHME, Wittgenstein/Lutz). | Low | Verify every reference (A8). |
| §3 Task & data (12) | Task card is clean. Prediction-time rule is natural (features at t, target at t+10). | Medium | State the projection cutoff, the WPP **data-vintage** caveat (2024 estimates of 2005 use hindsight a 2005 planner lacked), and the ISO fixes. |
| §4 EDA (10) | Rich: transition curve, cohort lag curve, gender-gap redundancy, partial correlations, stability over decades. | Low | Add the **stage-slice figure** and the hypothesis H-onset. Training rows only. |
| §5 Method (13) | Engineered groups have clear domain motivation: momentum (lags), curvature (TFR²), stage-gated education, cohort schooling. Ridge is justified by collinearity (education, Q5, TFR all r > 0.8). | Low | Pipeline figure. Every feature tied to an H. |
| §6 Setup (10) | Embargoed rolling origin, a strong argument. 4 folds is few, but they're principled. | Medium | Fold count is justified by the number of 5-year periods. Report mean ± sd. Use 5-year lags so t=1955 is usable and fold 1 has more data. |
| §7 Results (18) | The ladder of baselines is natural and strong. Damped trend is a demographer's rule, which makes a credible "simple" baseline. | **High under the original framing** | Reframe (A5). Linear vs RF was close in the pilot (.339 vs .356), so it's a real contest. |
| §7.2 Ablation (10) | Set A (raw) vs B (top-k correlated) vs C (engineered). Momentum features should give C a clear win over A. Education-group ablation plus slice metrics is the core finding. | Medium | Set A must exclude engineered lags. Top-k is selected inside each fold. |
| §8.1 Coefficients (8) | Interpretable effects in children per woman. VIF is high, which fits the brief's H4 "is it really education?" | Medium | Rows overlap in time and repeat per country, so naive t-tests are invalid. Use **country-clustered** SEs / cluster bootstrap and name it. |
| §8.2 Error analysis (8) | Real failure stories exist: African stalls, Central Asian rebounds (Kazakhstan), Egypt's late-2000s rise, East Asian ultra-low fertility. | Low | Pick cases from the test residuals after the single test evaluation. |
| §9 Conclusion (5) | Actionable recommendation: use education to time the decline in pre-transition countries, and use momentum elsewhere. | Low | — |
| Creativity (judged) | The education–fertility link is textbook. The distinctive parts are the forecasting framing with an embargo, the onset-timing finding, cohort logic, and Taiwan/East Asia for an NYCU audience. | Medium | Keep the onset angle front and center. Don't sell "education lowers fertility" as new. |

## A5. Recommended reframing (the key decision)
- **Research question:** "How much does knowing women's schooling today add to a 10-year fertility forecast, beyond fertility's own trajectory, and for which countries?"
- **Contributions, each one checkable in §7–8:**
  1. A leak-free, embargoed benchmark for forecasting TFR 10 years ahead across about 144 countries, with damped-trend and tree baselines.
  2. Evidence that momentum features (level, trend, curvature) carry most of the skill a linear model has.
  3. Evidence that education's added value is **stage-dependent**: it improves forecasts in pre-transition countries (the onset of decline) and not elsewhere. This is quantified by slice metrics and the ablation.
- **Feature consequence:** a stage-gated education feature, e.g. `edu × 1[TFR(t) ≥ 5]` or `edu × max(0, TFR(t) − 4)`. It's motivated by H-onset in §4 and chosen in CV, never on test.
- If the full model with region, cohort features, etc. shows education helping more broadly, so much the better. The framing covers either outcome honestly.

## A6. Problems in the brief to correct
1. **"Tree models can't extrapolate below training lows" is mostly false here.** The lowest training target is 0.86 (Macao 2005). Hong Kong and Singapore are already near 1.0. Taiwan's test targets are 1.18 (2015) and 0.99 (2020). **Taiwan's 0.87 is a 2023 value, outside the test window** (the last target year is 2020). The Taiwan case stays as an error-analysis or narrative item, not as an "unseen regime".
2. **The decision threshold needs the verified UN rule.** UN WPP 2024 methodology: the high/low scenarios are ±0.25 births in years 1–5, **±0.4 in years 6–10**, and ±0.5 after that. So at our 10-year horizon the scenarios are 0.4 apart. **Threshold: MAE ≤ 0.2 children per woman.** Below the midpoint between medium and high/low, the forecast picks the right UN scenario for a planner. Also report **scenario hit-rate**, the share of countries with |error| < 0.2, as a user-aligned metric. 0.4 is the "inside the UN band" reference line.
3. **5-year age bands** come from v2.2 only (to 2010). Main cohort feature: v3 women 15-24 at t, who are 25-34 at t+10, i.e. peak childbearing ages. Note that schooling at 15-19 is truncated because many are still in school.
4. **Non-stationarity (H5).** Persistence error shrinks across folds (0.75 → 0.52) as the world moves through the transition. A pooled intercept learned in the 1960s–80s may over-predict decline later. Test for it, and if needed add a calendar-time term or say it's a limitation.
5. **Significance.** With 4 folds a paired t-test is weak but still required. Primary: a **country-cluster bootstrap** CI of the test-MAE gap. §8.1 coefficients: OLS with country-clustered SEs (Ridge's α will be small) plus a bootstrap for the Ridge β.
6. **Leakage via data vintage.** Not fixable, but it must be stated. Optional benchmark: archived older WPP projections, if the old revision files can be obtained.

## A7. Risk register
| Risk | Likelihood | Mitigation |
|---|---|---|
| Education adds nothing even in the slice once region etc. are added | Medium | The null is itself a contribution under A5. Report it honestly together with the slice analysis. |
| RF beats Ridge on test | Medium | Allowed by the spec. Explain it with variance and interpretability, and use a cluster-bootstrap CI. |
| Threshold 0.2 not met | Medium–high | Allowed. Report the hit-rate and the distance to the threshold, and say who can still use the forecast. |
| 6-page limit | High (rich topic) | Strict figure budget: about 5 figures and 5 tables. Put the slice table in the ablation section. |
| Fabricated or wrong citations | Low | Each one checked by DOI or publisher page before it enters `refs.bib`. |
| Team ID, names, GitHub URL missing → report scores 0 | — | **You provide them.** Until then I use placeholders and leave a checklist. |

## A8. Citations verified so far (others get checked during the build)
- Kebede, Goujon & Lutz (2019), *PNAS* 116(8):2891–2896: female education disruptions and African fertility stalls. ✔
- Alkema, Raftery, Gerland et al. (2011), *Demography* 48:815–839: probabilistic TFR projections (UN BayesTFR). ✔
- UN DESA (2024), *WPP 2024 Methodology Report*: variant definitions quoted in A6. ✔
- Kebede et al. (2024), *PNAS*, "Forecasting Africa's fertility decline by female education groups", doi 10.1073/pnas.2320247121. ✔ (found; details still to read)
- To verify: Barro & Lee (2013) *J. Dev. Econ.* 104; Lutz & KC (2011) *Science* 333; Vollset et al. (2020) *Lancet* (IHME, which uses education to forecast fertility); Myrskylä, Kohler & Billari (2009) *Nature* 460; Feenstra et al. (2015) *AER* 105 (PWT, only if GDP is used).
