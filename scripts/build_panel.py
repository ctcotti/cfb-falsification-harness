"""Build the coach panel and report the treatment set.

This is the number that sets the power of the whole project: how many clean
year-one coach observations exist in the collective/revenue-share era.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb.client import CFBDClient  # noqa: E402
from cfb.panel import build_panel, treatment_set  # noqa: E402

LONG_WINDOW = list(range(2015, 2026))
THESIS_WINDOW = list(range(2022, 2026))
OUT = Path(__file__).resolve().parents[1] / "data" / "panel"


def main() -> None:
    pd.set_option("display.width", 140)
    client = CFBDClient()
    panel = build_panel(client, LONG_WINDOW)

    OUT.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT / "coach_panel.parquet", index=False)

    print(f"panel rows: {len(panel)}  "
          f"team-seasons: {panel.groupby(['school','season']).ngroups}")

    thesis = panel[panel["season"].isin(THESIS_WINDOW)]
    print("\n--- attrition, thesis window (2022-25) ---")
    steps = [
        ("FBS team-seasons", thesis.groupby(["school", "season"]).ngroups),
        ("coach rows", len(thesis)),
        ("first season at school", int(thesis["first_season_at_school"].sum())),
        ("  ...coach of record for season",
         int((thesis["first_season_at_school"]
              & thesis["is_primary_coach"]).sum())),
        ("  ...not interim",
         int((thesis["first_season_at_school"]
              & thesis["is_primary_coach"]
              & ~thesis["is_interim"]).sum())),
        ("  ...not an FBS debut season",
         int((thesis["first_season_at_school"]
              & thesis["is_primary_coach"]
              & ~thesis["is_interim"]
              & ~thesis["is_fbs_debut"]).sum())),
        ("  ...hired before July (CLEAN)", int(thesis["clean_year_one"].sum())),
    ]
    for label, n in steps:
        print(f"  {label:<34} {n:>5}")

    print("\n--- clean year-one by season ---")
    by_season = (panel[panel["clean_year_one"]]
                 .groupby("season").size().reindex(LONG_WINDOW, fill_value=0))
    for season, n in by_season.items():
        mark = "  <- thesis window" if season in THESIS_WINDOW else ""
        print(f"  {season}  {n:>3}{mark}")
    print(f"\n  thesis-window total: {by_season.loc[THESIS_WINDOW].sum()}")
    print(f"  long-window total:   {by_season.sum()}")

    print("\n--- data quality ---")
    miss = panel["hire_date"].isna().sum()
    print(f"  rows missing tenure hire_date: {miss} "
          f"({100*miss/len(panel):.1f}%)")
    print(f"  attribution_complete=False:    "
          f"{int((~panel['attribution_complete']).sum())}")

    ts = treatment_set(thesis)
    print("\n--- hire-month distribution, clean year-one (thesis window) ---")
    print(ts["hire_month"].value_counts().sort_index().to_string())

    print("\nsample of the treatment set:")
    cols = ["season", "school", "coach_name", "hire_date", "tenure_year"]
    print(ts.sort_values(["season", "school"])[cols].head(12).to_string(index=False))
    print(f"\nwrote {OUT / 'coach_panel.parquet'}")


if __name__ == "__main__":
    main()
