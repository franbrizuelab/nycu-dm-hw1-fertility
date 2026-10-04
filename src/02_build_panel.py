"""Build the country x start-year panel.

One row = one country c at start year t (t = 1955, 1960, ..., 2010).
Target  = total fertility rate (TFR) of country c in year t+10 (UN WPP 2024 estimate).
Features are all measured at year t or earlier, so they are known when the forecast is made.

Outputs
  data/processed/panel.csv           main modelling panel
  data/processed/edu_5yr_bands.csv   female schooling by 5-year age band (Barro-Lee v2.2, EDA only)
"""
import numpy as np
import pandas as pd

from common import PROC, RAW, START_YEARS, TEST_YEARS, GAP_YEARS, TRAIN_MAX_T

# Barro-Lee uses a few legacy codes that differ from the UN's ISO3 codes.
ISO_ALIAS = {"ROM": "ROU", "SER": "SRB"}
LAST_OBSERVED_YEAR = 2023  # WPP 2024: values after 2023 are projections


def barro_lee(path):
    bl = pd.read_csv(RAW / path)
    bl["iso3"] = bl["WBcode"].replace(ISO_ALIAS)
    return bl


def pop_weighted_schooling(bl, ages):
    """Average years of schooling over the given age bands, weighted by band population."""
    sub = bl[(bl["agefrom"].astype(str) + "-" + bl["ageto"].astype(str)).isin(ages)].copy()
    sub["w"] = sub["pop"] * sub["yr_sch"]
    g = sub.groupby(["iso3", "year"])[["w", "pop"]].sum()
    return (g["w"] / g["pop"]).rename(None)


def load_wpp():
    wpp = pd.read_csv(RAW / "WPP2024_Demographic_Indicators_Medium.csv.gz", low_memory=False)
    wpp = wpp[wpp["ISO3_code"].notna() & (wpp["Time"] <= LAST_OBSERVED_YEAR)]
    return wpp.set_index(["ISO3_code", "Time"])[["TFR", "Q5"]]


def load_pwt():
    pwt = pd.read_stata(RAW / "pwt1001.dta")
    pwt["gdppc"] = pwt["rgdpe"] / pwt["pop"]  # expenditure-side real GDP per person, 2017 US$ PPP
    return pwt.set_index(["countrycode", "year"])["gdppc"]


def load_urban():
    w = pd.read_excel(RAW / "WUP2025-F02-Degree-of-Urbanization_percPop_by_category.xlsx",
                      sheet_name="Rural")
    w = w[w["ISO3_Code"].notna()]
    w.columns = [int(float(c)) if str(c).replace(".0", "").isdigit() else c for c in w.columns]
    years = [c for c in w.columns if isinstance(c, int)]
    long = w.melt(id_vars="ISO3_Code", value_vars=years, var_name="year", value_name="rural")
    long["urban"] = 100 - long["rural"]  # share living in towns or cities (Degree of Urbanisation)
    return long.set_index(["ISO3_Code", "year"])["urban"]


def main():
    blf, blm = barro_lee("BL_v3_F.csv"), barro_lee("BL_v3_M.csv")
    meta = blf.drop_duplicates("iso3").set_index("iso3")[["country", "region_code"]]
    edu = pd.DataFrame({
        "edu_f1524": pop_weighted_schooling(blf, {"15-24"}),
        "edu_f2534": pop_weighted_schooling(blf, {"25-34"}),
        "edu_f2564": pop_weighted_schooling(blf, {"25-34", "35-44", "45-54", "55-64"}),
        "edu_m1524": pop_weighted_schooling(blm, {"15-24"}),
        "edu_m2564": pop_weighted_schooling(blm, {"25-34", "35-44", "45-54", "55-64"}),
    })
    wpp, gdp, urban = load_wpp(), load_pwt(), load_urban()

    rows = []
    for iso in meta.index:
        for t in START_YEARS:
            def tfr(year):
                return wpp["TFR"].get((iso, year), np.nan)
            rows.append({
                "iso3": iso, "country": meta.at[iso, "country"], "region": meta.at[iso, "region_code"],
                "t": t, "target_year": t + 10, "y": tfr(t + 10),
                "tfr": tfr(t), "tfr_l5": tfr(t - 5), "tfr_l10": tfr(t - 10),
                "q5": wpp["Q5"].get((iso, t), np.nan),
                **{k: edu[k].get((iso, t), np.nan) for k in edu.columns},
                "gdppc": gdp.get((iso, t), np.nan),
                "urban": urban.get((iso, t), np.nan),
            })
    panel = pd.DataFrame(rows)
    panel["split"] = np.select(
        [panel["t"] <= TRAIN_MAX_T, panel["t"].isin(GAP_YEARS), panel["t"].isin(TEST_YEARS)],
        ["train", "gap", "test"], "unused")

    # Core columns must be present; GDP may be missing (imputed inside each fold, see common.py).
    core = ["y", "tfr", "tfr_l5", "q5", "edu_f1524", "edu_f2564", "edu_m1524", "edu_m2564", "urban"]
    dropped = panel[panel[core].isna().any(axis=1)]
    panel = panel.dropna(subset=core).reset_index(drop=True)

    # Leakage and coverage checks.
    assert panel["target_year"].max() <= 2020 <= LAST_OBSERVED_YEAR, "target must be an observed estimate"
    assert panel.loc[panel.split == "train", "target_year"].max() <= min(TEST_YEARS), \
        "a training target is later than the first test forecast date"
    assert {"TWN", "KOR", "JPN"} <= set(panel["iso3"]), "Taiwan/Korea/Japan missing"

    PROC.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROC / "panel.csv", index=False)
    print(f"panel: {len(panel)} rows, {panel.iso3.nunique()} countries; "
          f"dropped {len(dropped)} rows with missing core values "
          f"({sorted(dropped.iso3.unique())[:10]}...)")
    print(panel.groupby("split").size().to_dict())
    print("GDP missing share by split:", panel.groupby("split")["gdppc"].apply(lambda s: s.isna().mean()).round(3).to_dict())

    # 5-year age bands (v2.2) for the cohort lag-curve analysis in Section 4.
    v22 = barro_lee("BL2013_F_v2.2.csv")
    v22 = v22[v22["ageto"] != 999]
    v22["band"] = v22["agefrom"].astype(str) + "-" + v22["ageto"].astype(str)
    bands = v22.pivot_table(index=["iso3", "year"], columns="band", values="yr_sch").reset_index()
    bands.to_csv(PROC / "edu_5yr_bands.csv", index=False)


if __name__ == "__main__":
    main()
