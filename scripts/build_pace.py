"""Build the team-season pace panel and report what kind of axis it is.

Diagnostics only -- no outcome variable is touched here. Whether pace predicts
anything is a question for the registered test; this script answers the prior
question of whether the axis is admissible at all.

    python scripts/build_pace.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfb.pace import build_pace  # noqa: E402

CACHE = ROOT / "data" / "cache"
OUT = ROOT / "data" / "panel"
SEASONS = list(range(2015, 2026))


def main() -> None:
    pace = build_pace(str(CACHE), SEASONS)
    OUT.mkdir(parents=True, exist_ok=True)
    pace.to_parquet(OUT / "pace.parquet", index=False)

    print(f"team-seasons: {len(pace)}   seasons "
          f"{int(pace.season.min())}-{int(pace.season.max())}")
    print(f"qualifying neutral drives: {int(pace.drives.sum()):,}   "
          f"snaps: {int(pace.plays.sum()):,}")
    print(f"two-way (offense + defense) R2 on drive seconds/play: "
          f"{pace.attrs['two_way_r2']:.4f}")
    print(f"\npace_raw  mean {pace.pace_raw.mean():.2f}s  "
          f"sd {pace.pace_raw.std():.2f}")
    print(f"pace_adj  mean {pace.pace_adj.mean():.2f}s  "
          f"sd {pace.pace_adj.std():.2f}")
    print(f"corr(raw, opponent-adjusted) = "
          f"{np.corrcoef(pace.pace_raw, pace.pace_adj)[0, 1]:.4f}")

    print("\nleague mean by season (the reason everything is z-scored within it)")
    by = pace.groupby("season")["pace_raw"].agg(["mean", "std", "size"])
    print("  " + "  ".join(f"{int(s)}:{r['mean']:.1f}" for s, r in by.iterrows()))

    # Construct validity: is this a new axis, or a restatement of Phase 1's?
    tend = pd.read_parquet(OUT / "tendency.parquet")
    tend["edpr_z"] = tend.groupby("season")["early_down_pass_rate"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))
    j = pace.merge(tend[["school", "season", "edpr_z"]],
                   on=["school", "season"], how="inner")
    print(f"\nCONSTRUCT CHECK  (n={len(j)} team-seasons)")
    for col in ("pace_raw_z", "pace_adj_z"):
        r = float(np.corrcoef(j[col], j["edpr_z"])[0, 1])
        print(f"  corr({col}, early_down_pass_rate_z) = {r:+.3f}"
              f"   shared variance {r*r*100:.1f}%")
    print("  A pass-heavy offense stops the clock more often, so some overlap is"
          "\n  mechanical. The question is whether it is small enough that this"
          "\n  is a different construct and not Phase 1's axis in new units.")

    print("\nfastest and slowest, opponent-adjusted, most recent season")
    last = pace[pace.season == pace.season.max()].nsmallest(5, "pace_adj")
    slow = pace[pace.season == pace.season.max()].nlargest(5, "pace_adj")
    for lab, d in (("fastest", last), ("slowest", slow)):
        print(f"  {lab}: " + ", ".join(
            f"{r.school} {r.pace_adj:.1f}s" for r in d.itertuples()))


if __name__ == "__main__":
    main()
