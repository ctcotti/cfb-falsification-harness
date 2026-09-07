"""Stage 3 decision gate.

Two regressions, deliberately kept apart. The brief conflated them and attached
the wrong conclusion to the wrong one.

  TEST A -- is fit already in the price?
      close ~ main effects;  resid_close ~ Phi
      Want gamma_A ~ 0. That CONFIRMS A4 (openers are main-effects models) and
      is the precondition for an edge, not a kill.

  TEST B -- does fit predict what the price missed?
      (margin - market) ~ Phi
      Want gamma_B != 0. This is the tradeable claim.

  PURE-INTERACTION TEST -- belongs in B, not A.
      (margin - market) ~ Phi + coach quality + QB quality
      The hypothesis forbids main effects: the last two must be ~0. If either
      predicts, the market is leaving simple money on the table, which is
      implausible, and fit is not the mechanism behind whatever was found.

Estimated at team-season level. Phi is constant within a team-season, so the
game-level N is a costume: 12 games sharing one Phi are one observation plus
noise. Collapsing to the cluster mean is the honest estimator and is exactly
what the power calculation assumed.
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
THESIS = (2022, 2025)


def _rows(endpoint: str):
    for meta in glob.glob(str(CACHE / endpoint / "*.meta.json")):
        data = meta.replace(".meta.json", ".json")
        if os.path.exists(data):
            with open(data, encoding="utf-8") as fh:
                payload = json.load(fh)
            if isinstance(payload, list):
                yield from payload


def controls() -> pd.DataFrame:
    sp = pd.DataFrame([{"school": r.get("team"), "season": r.get("year"),
                        "sp_rating": r.get("rating")} for r in _rows("ratings_sp")])
    sp = sp.dropna(subset=["school", "season"]).drop_duplicates(["school", "season"])
    rec = pd.DataFrame([{"school": r.get("team"), "season": r.get("year"),
                         "recruit": r.get("points")} for r in _rows("recruiting_teams")])
    rec = rec.dropna(subset=["school", "season"]).drop_duplicates(["school", "season"])
    ret = pd.DataFrame([{"school": r.get("team"), "season": r.get("season"),
                         "ret_ppa": r.get("percentPPA")} for r in _rows("player_returning")])
    ret = ret.dropna(subset=["school", "season"]).drop_duplicates(["school", "season"])
    return sp, rec, ret


def build() -> pd.DataFrame:
    tg = pd.read_parquet(PANEL / "team_games.parquet")
    phi = pd.read_parquet(PANEL / "phi.parquet")
    sp, rec, ret = controls()

    ts = tg.groupby(["school", "season"]).agg(
        games=("margin", "size"),
        mean_market=("market_margin", "mean"),
        mean_resid=("resid", "mean"),
    ).reset_index()

    df = ts.merge(
        phi[["school", "season", "phi", "distance", "live", "transfer",
             "new_coach", "tenure_year", "era_collective", "measurable",
             "coach_id", "passer"]],
        on=["school", "season"], how="inner")
    df = df.merge(sp, on=["school", "season"], how="left")
    df = df.merge(rec, on=["school", "season"], how="left")
    df = df.merge(ret, on=["school", "season"], how="left")

    prior = sp.copy(); prior["season"] += 1
    df = df.merge(prior.rename(columns={"sp_rating": "sp_prior"}),
                  on=["school", "season"], how="left")

    df = df[df["measurable"] & df["games"].ge(8)].copy()
    # Standardise Phi so gamma reads as points per SD.
    df["phi_z"] = (df["phi"] - df["phi"].mean()) / df["phi"].std(ddof=0)
    return df


def ols(y, X, label, names):
    X = sm.add_constant(X, has_constant="add")
    m = sm.OLS(y, X, missing="drop").fit(cov_type="HC1")
    print(f"\n  {label}   n={int(m.nobs)}  R2={m.rsquared:.3f}")
    for nm in names:
        if nm not in m.params:
            continue
        b, se, p = m.params[nm], m.bse[nm], m.pvalues[nm]
        lo, hi = b - 1.96 * se, b + 1.96 * se
        print(f"    {nm:<18} {b:+7.3f}  se {se:.3f}  "
              f"95% CI [{lo:+.2f},{hi:+.2f}]  p={p:.3f}")
    return m


def main() -> None:
    pd.set_option("display.width", 150)
    df = build()
    win = df[df.season.between(*THESIS)]
    print(f"team-seasons (measurable): {len(df)}   thesis window: {len(win)}")
    print(f"  live (nonzero distance): {int(win.live.sum())}")

    for label, d in [("THESIS WINDOW 2022-25", win), ("LONG WINDOW 2015-25", df)]:
        print(f"\n{'='*66}\n{label}   n={len(d)}\n{'='*66}")

        print("\n--- TEST A: is fit already in the price? (want gamma_A ~ 0)")
        base = d[["sp_prior", "recruit", "ret_ppa", "new_coach"]].astype(float)
        mA = sm.OLS(d["mean_market"], sm.add_constant(base), missing="drop").fit()
        print(f"    first stage R2 = {mA.rsquared:.3f}  (brief expects > 0.9)")
        keep = base.dropna().index
        resid = pd.Series(mA.resid, index=keep)
        ols(resid, d.loc[keep, ["phi_z"]].astype(float),
            "resid_close ~ Phi", ["phi_z"])

        print("\n--- TEST B: does fit predict what the price missed? "
              "(want gamma_B != 0)")
        ols(d["mean_resid"], d[["phi_z"]].astype(float),
            "(margin - market) ~ Phi", ["phi_z"])

        X = d[["phi_z", "new_coach", "transfer"]].astype(float)
        X["era"] = d["era_collective"].astype(float)
        X["phi_x_era"] = X["phi_z"] * X["era"]
        ols(d["mean_resid"], X, "with era interaction",
            ["phi_z", "phi_x_era", "era"])

        print("\n--- PURE-INTERACTION TEST (main effects must be ~0)")
        Z = pd.DataFrame({
            "phi_z": d["phi_z"].astype(float),
            "coach_quality": d["sp_prior"].astype(float),
            "recruit": d["recruit"].astype(float),
        })
        ols(d["mean_resid"], Z, "(margin - market) ~ Phi + quality",
            ["phi_z", "coach_quality", "recruit"])


if __name__ == "__main__":
    main()
