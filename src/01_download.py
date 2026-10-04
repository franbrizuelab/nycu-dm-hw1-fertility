"""Download every raw data file used in the project and record SHA-256 checksums.

All sources are public and need no login. Files land in data/raw/; a manifest
(data/raw/MANIFEST.tsv) records URL, size and checksum so results are traceable.
"""
import hashlib
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

BL = "https://raw.githubusercontent.com/barrolee/BarroLeeDataSet/master/BLData/"
SOURCES = {
    # Barro-Lee v3 (2021 update): attainment by sex and 10-year age group, 1950-2015
    "BL_v3_F.csv": BL + "BL_v3_F.csv",
    "BL_v3_M.csv": BL + "BL_v3_M.csv",
    # Barro-Lee v2.2 (2013): attainment by sex and 5-year age group, 1950-2010 (EDA lag curve only)
    "BL2013_F_v2.2.csv": BL + "BL2013_F_v2.2.csv",
    # UN World Population Prospects 2024: TFR, under-5 mortality, etc. (estimates 1950-2023)
    "WPP2024_Demographic_Indicators_Medium.csv.gz":
        "https://population.un.org/wpp/assets/Excel%20Files/1_Indicator%20(Standard)/CSV_FILES/"
        "WPP2024_Demographic_Indicators_Medium.csv.gz",
    # Penn World Table 10.01: real GDP and population
    "pwt1001.dta": "https://dataverse.nl/api/access/datafile/354098",
    # UN World Urbanization Prospects 2025: population share by Degree of Urbanisation
    "WUP2025-F02-Degree-of-Urbanization_percPop_by_category.xlsx":
        "https://population.un.org/wup/assets/Download/Countries%20and%20Aggregates/"
        "WUP2025-F02-Degree-of-Urbanization_percPop_by_category.xlsx",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, url in SOURCES.items():
        dest = RAW / name
        if not dest.exists():
            print(f"downloading {name}")
            # Default urllib user agent: a browser-like one triggers dataverse.nl's bot check.
            with urllib.request.urlopen(url) as r:
                data = r.read()
            if data[:15].lower().startswith(b"<!doctype html"):
                raise RuntimeError(f"{url} returned an HTML page instead of data")
            dest.write_bytes(data)
        rows.append(f"{name}\t{dest.stat().st_size}\t{sha256(dest)}\t{url}")
        print(f"ok {name} ({dest.stat().st_size:,} bytes)")
    (RAW / "MANIFEST.tsv").write_text("file\tbytes\tsha256\turl\n" + "\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
