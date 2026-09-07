"""Is q_p a quarterback trait, or is it noise?

Same logic that killed qb_run_share as a coach axis, applied to the QB side. If
a quarterback's situational contrast in season y tells you nothing about his
contrast in y+1, then q_p is not a property of the player and Phi has no QB
side -- which collapses option A to option C, the nested null.

Two statistics, because they answer different questions:

  year-over-year corr  = sigma2_trait / (trait + year + sampling)
  split-half reliability = (trait + year) / (trait + year + sampling)

A low YoY correlation with decent split-half reliability means the measurement
is fine but the thing measured is not stable. Both low means it is all
sampling noise. The benchmark is overall EPA per dropback -- a metric nobody
doubts is a real QB attribute -- measured on the same players and seasons, so
"is 0.2 low?" has an answer instead of a shrug.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
MIN_OPD = 25
MIN_ED = 40


def yoy(df: pd.DataFrame, col: str, label: str) -> None:
    d = df[(df.opd_n >= MIN_OPD) & (df.ed_n >= MIN_ED)].copy()
    # Season-standardise so league drift is not counted as signal.
    d[col + "_z"] = d.groupby("season")[col].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))
    nxt = d[["passer", "season", col + "_z", "school"]].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(columns={col + "_z": "next", "school": "next_school"})
    j = d.merge(nxt, on=["passer", "season"], how="inner")
    if len(j) < 10:
        print(f"  {label:<26} too few pairs ({len(j)})")
        return
    r = float(np.corrcoef(j[col + "_z"], j["next"])[0, 1])
    moved = j[j.school != j.next_school]
    rm = (float(np.corrcoef(moved[col + "_z"], moved["next"])[0, 1])
          if len(moved) >= 10 else float("nan"))
    # Fisher CI on r
    z = np.arctanh(r); se = 1 / np.sqrt(len(j) - 3)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    print(f"  {label:<26} n={len(j):>4}  r={r:+.3f} [{lo:+.2f},{hi:+.2f}]"
          f"   transfers only: n={len(moved):>3} r={rm:+.3f}")


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "panel" / "qb_profiles_raw.parquet")
    df["epa_overall"] = (
        (df.ed_epa * df.ed_n + df.opd_epa * df.opd_n) / (df.ed_n + df.opd_n))

    n = ((df.opd_n >= MIN_OPD) & (df.ed_n >= MIN_ED)).sum()
    print(f"QB-seasons meeting opd_n>={MIN_OPD}, ed_n>={MIN_ED}: {n} "
          f"({n / df.season.nunique():.0f}/season)\n")

    print("year-over-year correlation, same passer, consecutive seasons:")
    yoy(df, "contrast_raw", "situational contrast")
    yoy(df, "epa_overall", "overall EPA/dropback")
    yoy(df, "ed_epa", "  early-down EPA only")
    yoy(df, "opd_epa", "  passing-down EPA only")


if __name__ == "__main__":
    main()
