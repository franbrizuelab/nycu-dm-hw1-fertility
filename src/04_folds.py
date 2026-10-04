"""Write the rolling-origin CV folds (with 10-year embargo) to results/folds.json.

Every method in 05_experiments.py reloads these exact (country, start year) keys.
"""
from common import load_folds, make_folds, save_folds, train_panel


def main():
    df = train_panel()
    save_folds(df)
    for v, tr, va in load_folds(df):
        print(f"validate t={v}: train t in [{df.t[tr].min()}, {df.t[tr].max()}] "
              f"({len(tr)} rows), validate {len(va)} rows; latest training target year {df.t[tr].max() + 10}")


if __name__ == "__main__":
    main()
