"""Stage 3 gate, game level -- fixes two defects in the team-season version.

1. Test A's first stage must include the OPPONENT. A team-season mean market
   margin conflates team strength with schedule strength, which is why the
   collapsed version returned R2=0.57 against the brief's expected >0.9. The
   market prices a matchup, so the main-effects model has to see both sides.

2. The era interaction is degenerate inside 2022-25 (era == 1 for every row,
   so phi_x_era IS phi_z). It is only estimable on the long window.

Test B is still reported at team-season level as the primary estimate -- Phi is
constant within a team-season, so game rows are not independent observations --
but game level is used here for the week-decay test, which needs within-season
variation. Standard errors are clustered on team-season throughout.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
PANEL = ROOT / "data" / "panel"
EARLY_WEEKS = 4


def _rows(endpoint: str):
    for meta in glob.glob(str(CACHE / endpoint / "*.meta.json")):
        data = meta.replace(".meta.json", ".json")
        if os.path.exists(data):
            with open(data, encoding="utf-8") as fh:
                p = json.load(fh)
            if isinstance(p, list):
                yield from p


def build() -> pd.DataFrame:
    tg = pd.read_parquet(PANEL / "team_games.parquet")
    phi = pd.read_parquet(PANEL / "phi.parquet")

    sp = pd.DataFrame([{"school": r.get("team"), "season": r.get("year"),
                        "sp": r.get("rating")} for r in _rows("ratings_sp")])
    sp = sp.dropna(subset=["school", "season"]).drop_duplicates(["school", "season"])
    sp["season"] += 1  # prior-season rating, known ex ante
    sp = sp.rename(columns={"sp": "sp_prior"})

    rec = pd.DataFrame([{"school": r.get("team"), "season": r.get("year"),
                         "recruit": r.get("points")} for r in _rows("recruiting_teams")])
    rec = rec.dropna(subset=["school", "season"]).drop_duplicates(["school", "season"])

    feat = (phi[["school", "season", "phi", "live", "measurable"]]
            .merge(sp, on=["school", "season"], how="left")
            .merge(rec, on=["school", "season"], how="left"))
    feat["phi_z"] = (feat["phi"] - feat["phi"].mean()) / feat["phi"].std(ddof=0)

    own = feat.rename(columns={c: c + "_own" for c in
                               ["phi", "phi_z", "sp_prior", "recruit",
                                "live", "measurable"]})
    opp = feat.rename(columns={"school": "opponent"}).rename(
        columns={c: c + "_opp" for c in
                 ["phi", "phi_z", "sp_prior", "recruit", "live", "measurable"]})

    df = (tg.merge(own, on=["school", "season"], how="inner")
            .merge(opp, on=["opponent", "season"], how="inner"))
    df = df[df["measurable_own"] & df["measurable_opp"]].copy()
    df["phi_diff"] = df["phi_z_own"] - df["phi_z_opp"]
    df["ts"] = df["school"] + "_" + df["season"].astype(str)
    df["early"] = (df["week"] <= EARLY_WEEKS).astype(float)
    df["era"] = (df["season"] >= 2022).astype(float)
    return df


def fit(y, X, label, names, groups):
    X = sm.add_constant(X, has_constant="add")
    ok = X.notna().all(axis=1) & y.notna()
    m = sm.OLS(y[ok], X[ok]).fit(cov_type="cluster",
                                 cov_kwds={"groups": groups[ok]})
    print(f"\n  {label}   n={int(m.nobs)}  clusters={groups[ok].nunique()}  "
          f"R2={m.rsquared:.3f}")
    for nm in names:
        b, se, p = m.params[nm], m.bse[nm], m.pvalues[nm]
        print(f"    {nm:<16} {b:+7.3f}  se {se:.3f}  "
              f"95% CI [{b-1.96*se:+.2f},{b+1.96*se:+.2f}]  p={p:.3f}")
    return m


def main() -> None:
    df = build()
    for label, d in [("THESIS 2022-25", df[df.season.between(2022, 2025)]),
                     ("LONG 2015-25", df)]:
        print(f"\n{'='*68}\n{label}   team-games={len(d)}  "
              f"live sides={int(d.live_own.sum())}\n{'='*68}")

        print("\n--- TEST A: is Phi already in the market price?")
        base = d[["sp_prior_own", "sp_prior_opp", "is_home",
                  "recruit_own", "recruit_opp"]].astype(float)
        ok = base.notna().all(axis=1)
        mA = sm.OLS(d.loc[ok, "market_margin"],
                    sm.add_constant(base[ok])).fit()
        print(f"    first stage R2 = {mA.rsquared:.3f}   (brief expects >0.9)")
        rA = pd.Series(np.nan, index=d.index)
        rA[ok] = mA.resid
        fit(rA, d[["phi_diff"]].astype(float), "resid_market ~ Phi_diff",
            ["phi_diff"], d["ts"])

        print("\n--- TEST B: does Phi predict what the market missed?")
        fit(d["resid"], d[["phi_diff"]].astype(float),
            "(margin - market) ~ Phi_diff", ["phi_diff"], d["ts"])

        X = d[["phi_diff", "early"]].astype(float)
        X["phi_x_early"] = X["phi_diff"] * X["early"]
        fit(d["resid"], X, "with week-decay interaction",
            ["phi_diff", "phi_x_early"], d["ts"])

        if d["era"].nunique() > 1:
            X = d[["phi_diff", "era"]].astype(float)
            X["phi_x_era"] = X["phi_diff"] * X["era"]
            fit(d["resid"], X, "with era interaction",
                ["phi_diff", "phi_x_era"], d["ts"])
        else:
            print("\n  era interaction: not estimable (single era)")


if __name__ == "__main__":
    main()
