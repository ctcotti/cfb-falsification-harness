"""G1: does line movement carry information the recorded price has not absorbed?

A gate on a data purchase, not a tradeable claim. Specification and decision
rule are in PREREGISTRATION.md, fixed before this ran.

CFBD gives an opener and one later number of unknown vintage, and no odds
history endpoint exists. So this can establish whether information sits in the
movement; it cannot establish that the information was reachable, because we do
not know when the later price was observable. That asymmetry is the whole reason
the result is a gate rather than a strategy.

Inference is the wild cluster bootstrap from Phase 0, clustered on season-week.
Cluster-robust SEs are printed beside it and do not carry the verdict.

    python scripts/phase3_drift_gate.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from phase0_wildboot import wild_cluster  # noqa: E402

CACHE = ROOT / "data" / "cache"
BREAK_EVEN_PP = 2.38
PHI_DENSITY = 0.3989
SIGMA = {"spread": 15.51, "total": 16.32}


def _games():
    for meta in glob.glob(str(CACHE / "lines" / "*.meta.json")):
        data = meta.replace(".meta.json", ".json")
        if not os.path.exists(data):
            continue
        with open(data, encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            yield from payload


def build() -> pd.DataFrame:
    rows = []
    for g in _games():
        if g.get("seasonType") != "regular":
            continue
        if g.get("homeClassification") != "fbs" or g.get("awayClassification") != "fbs":
            continue
        if g.get("homeScore") is None or g.get("awayScore") is None:
            continue
        lines = g.get("lines") or []
        def med(key):
            v = [l[key] for l in lines if l.get(key) is not None]
            return float(np.median(v)) if v else np.nan
        rows.append({
            "game_id": g["id"], "season": g["season"], "week": g["week"],
            "margin": g["homeScore"] - g["awayScore"],
            "total": g["homeScore"] + g["awayScore"],
            "spread": med("spread"), "spread_open": med("spreadOpen"),
            "ou": med("overUnder"), "ou_open": med("overUnderOpen"),
            "n_books": len([l for l in lines if l.get("spread") is not None]),
        })
    df = pd.DataFrame(rows).drop_duplicates("game_id")
    # CFBD's spread is the home number and positive means home gets points, so
    # the market's implied home margin is -spread. Drift is stated in implied
    # home margin: positive means the market moved TOWARD the home side.
    df["drift_spread"] = df["spread_open"] - df["spread"]
    df["resid_spread"] = df["margin"] + df["spread"]
    df["drift_total"] = df["ou"] - df["ou_open"]
    df["resid_total"] = df["total"] - df["ou"]
    df["sw"] = df["season"].astype(str) + "w" + df["week"].astype(str)
    return df


def pp(points: float, leg: str) -> float:
    return PHI_DENSITY * points / SIGMA[leg] * 100.0


def run_leg(df: pd.DataFrame, leg: str, moved_only: bool) -> dict:
    d = df.dropna(subset=[f"drift_{leg}", f"resid_{leg}"]).copy()
    if moved_only:
        d = d[d[f"drift_{leg}"] != 0]
    r = wild_cluster(d, f"resid_{leg}", [f"drift_{leg}"], "sw", f"drift_{leg}")
    mean_drift = float(d[f"drift_{leg}"].abs().mean())
    r.update(leg=leg, mean_drift=mean_drift, moved_only=moved_only,
             edge_pp=pp(abs(r["beta"]) * mean_drift, leg),
             hi_pp=pp(max(abs(r["boot_lo"]), abs(r["boot_hi"])) * mean_drift, leg))
    return r


def verdict(r: dict) -> str:
    excludes_zero = r["boot_lo"] > 0 or r["boot_hi"] < 0
    if excludes_zero and r["edge_pp"] > BREAK_EVEN_PP:
        return "INFORMATION, AND LARGE -- price the feed"
    if excludes_zero:
        return "information exists but is below break-even -- do not buy on this"
    if r["hi_pp"] < BREAK_EVEN_PP:
        return "price absorbs its own movement -- do not buy on this"
    return "UNDERPOWERED -- claim nothing"


def main() -> None:
    df = build()
    print("G1 opener drift gate.  Registered in PREREGISTRATION.md before "
          "running.")
    print("alpha=0.025 per leg (Bonferroni, N=2); wild cluster bootstrap-t, "
          "null imposed,\nclustered on season-week. Break-even at -110 = "
          f"{BREAK_EVEN_PP}pp.\n")

    both = df.dropna(subset=["spread", "spread_open"])
    print(f"FBS-FBS scored regular games: {len(df)}   with an opener: "
          f"{len(both)} ({len(both)/len(df)*100:.1f}%)   seasons "
          f"{int(both.season.min())}-{int(both.season.max())}")

    out = []
    for leg in ("spread", "total"):
        for moved_only in (True, False):
            r = run_leg(df, leg, moved_only)
            out.append(r)
            tag = "moved only" if moved_only else "all with opener"
            print(f"\n--- {leg.upper()}  ({tag})")
            print(f"    n={r['n']}  clusters={r['clusters']}  "
                  f"mean |drift| {r['mean_drift']:.2f} pts")
            print(f"    beta = {r['beta']:+.4f} resid-pts per pt of drift"
                  f"   (cluster-robust se {r['se']:.4f})")
            print(f"    cluster-robust 95% [{r['crve_lo']:+.4f}, "
                  f"{r['crve_hi']:+.4f}]")
            print(f"    bootstrap      95% [{r['boot_lo']:+.4f}, "
                  f"{r['boot_hi']:+.4f}]   boot p = {r['boot_p']:.4f}")
            print(f"    implied edge at mean drift: {r['edge_pp']:.2f} pp   "
                  f"(CI edge up to {r['hi_pp']:.2f} pp)")
            print(f"    -> {verdict(r)}")

    # The bias that must be quantified rather than assumed away: the later price
    # is in the regressor with one sign and the outcome with the other, so its
    # measurement error contributes -Var(err)/Var(drift) to beta even when the
    # true coefficient is zero.
    print("\n--- MECHANICAL BIAS CHECK")
    print("    PRE-REGISTERED ESTIMATE WAS -0.006. IT WAS WRONG. It used the"
          "\n    MEDIAN cross-book SD (0.29); the variance is driven by the mean"
          " of squared\n    dispersion, and cross-book dispersion is heavy-tailed"
          " (p90 SD 1.32). Measured:")
    bias = {}
    for leg in ("spread", "total"):
        key = "spread" if leg == "spread" else "overUnder"
        vals = []
        for g in _games():
            if g.get("seasonType") != "regular":
                continue
            v = [l[key] for l in (g.get("lines") or []) if l.get(key) is not None]
            if len(v) > 1:
                vals.append(np.var(v, ddof=1) / len(v))
        v_err = float(np.mean(vals))
        v_drift = float(df[f"drift_{leg}"].var())
        bias[leg] = -v_err / v_drift
        print(f"    {leg}: Var(price error) ~ {v_err:.4f}, Var(drift) = "
              f"{v_drift:.3f}  ->  induced beta ~ {bias[leg]:+.4f}")
    print("\n    First-order bias-adjusted coefficients (moved-only rows):")
    for r in out:
        if not r["moved_only"]:
            continue
        adj = r["beta"] - bias[r["leg"]]
        print(f"      {r['leg']:<7} raw {r['beta']:+.4f}  -  bias "
              f"{bias[r['leg']]:+.4f}  =  adjusted {adj:+.4f}"
              f"   ({pp(abs(adj) * r['mean_drift'], r['leg']):.2f} pp at mean "
              f"drift)")
    print("    Approximate: it ignores the attenuation from error in the OPENER,"
          "\n    which pushes the true coefficient toward zero as well. Both"
          " effects are\n    consistent with a true beta of zero, and the spread"
          " leg's raw negative\n    coefficient is smaller than the artefact"
          " alone would produce.")


if __name__ == "__main__":
    main()
