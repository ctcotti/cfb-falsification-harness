"""Falsification test: do systems travel with the coach?

The premise of the whole project is that a coach's system lives in his head and
arrives intact at a new school. If a coach's own play-call tendency shifts when
he changes jobs, then d_c is a property of the roster he inherited, not of him,
and Phi measures personnel rather than scheme.

The benchmark is NOT zero. Some across-school movement is ordinary year-to-year
wobble. So compare two one-year differences on the same footing:

    delta_stay = |axis(y+1) - axis(y)|   same coach, same school
    delta_move = |axis(first at B) - axis(last at A)|   same coach, new school

Both are one-season differences, so they are directly comparable. Writing
axis = mu + alpha_coach + beta_stint + eps_year,

    Var(delta_stay) = 2 * sigma2_eps
    Var(delta_move) = 2 * sigma2_beta + 2 * sigma2_eps

so the ratio Var(delta_move)/Var(delta_stay) = 1 + sigma2_beta/sigma2_eps.

PRE-REGISTERED DECISION RULE (fixed before looking at the number):
    ratio < 2.0  -> systems travel; the school effect is smaller than a
                    coach's ordinary year-to-year variation.
    ratio >= 2.0 -> a real school effect exists; d_c is contaminated by roster
                    and must be measured at the PRIOR school only.

Axes are z-scored within season, because league-wide pass rates trend over the
decade and an undetrended difference would count that drift as instability.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
AXES = ["early_down_pass_rate", "qb_run_share"]
RATIO_THRESHOLD = 2.0


def load() -> pd.DataFrame:
    tend = pd.read_parquet(ROOT / "data" / "panel" / "tendency.parquet")
    panel = pd.read_parquet(ROOT / "data" / "panel" / "coach_panel.parquet")
    coach = panel[panel["is_primary_coach"]][
        ["school", "season", "coach_id", "coach_name", "tenure_year"]
    ]
    df = tend.merge(coach, on=["school", "season"], how="inner")
    # Detrend: league-wide play-calling drifts across the decade.
    for ax in AXES:
        df[ax + "_z"] = df.groupby("season")[ax].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0)
        )
    return df.sort_values(["coach_id", "season"])


def deltas(df: pd.DataFrame, axis: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    col = axis + "_z"
    stay, move = [], []
    for cid, g in df.groupby("coach_id"):
        g = g.sort_values("season")
        rows = g.to_dict("records")
        for a, b in zip(rows, rows[1:]):
            gap = b["season"] - a["season"]
            if gap != 1:
                continue  # a gap year is neither a clean stay nor a clean move
            rec = {
                "coach_id": cid,
                "coach_name": a["coach_name"],
                "from_school": a["school"],
                "to_school": b["school"],
                "season_from": a["season"],
                "season_to": b["season"],
                "delta": b[col] - a[col],
            }
            (stay if a["school"] == b["school"] else move).append(rec)
    return pd.DataFrame(stay), pd.DataFrame(move)


def main() -> None:
    df = load()
    print(f"team-seasons with a primary coach: {len(df)}  "
          f"coaches: {df['coach_id'].nunique()}")

    for axis in AXES:
        stay, move = deltas(df, axis)
        v_stay = float(np.var(stay["delta"], ddof=1))
        v_move = float(np.var(move["delta"], ddof=1))
        ratio = v_move / v_stay
        # sigma2_beta / sigma2_eps
        school_to_noise = ratio - 1.0
        # Correlation across the move, for intuition.
        r = np.corrcoef(
            df.set_index(["coach_id", "season"]).loc[
                list(zip(move["coach_id"], move["season_from"]))
            ][axis + "_z"],
            df.set_index(["coach_id", "season"]).loc[
                list(zip(move["coach_id"], move["season_to"]))
            ][axis + "_z"],
        )[0, 1]

        # Bootstrap the ratio -- n_move is small and the ratio is a ratio of
        # variances, so a point estimate alone would overstate precision.
        rng = np.random.default_rng(0)
        boots = []
        for _ in range(4000):
            bs = rng.choice(stay["delta"].to_numpy(), len(stay), replace=True)
            bm = rng.choice(move["delta"].to_numpy(), len(move), replace=True)
            boots.append(np.var(bm, ddof=1) / np.var(bs, ddof=1))
        lo, hi = np.percentile(boots, [2.5, 97.5])

        print(f"\n=== {axis} ===")
        print(f"  stays (same coach, same school, consecutive): {len(stay)}")
        print(f"  moves (same coach, new school, consecutive):  {len(move)}")
        print(f"  Var(delta_stay) = {v_stay:.4f}   [noise floor]")
        print(f"  Var(delta_move) = {v_move:.4f}")
        print(f"  ratio = {ratio:.2f}   95% CI [{lo:.2f}, {hi:.2f}]")
        print(f"  implied sigma2_school / sigma2_year = {school_to_noise:.2f}")
        print(f"  corr(axis before move, axis after move) = {r:.3f}")
        verdict = ("TRAVELS" if ratio < RATIO_THRESHOLD
                   else "SCHOOL EFFECT -- measure d_c at prior school only")
        print(f"  verdict (rule: ratio < {RATIO_THRESHOLD}): {verdict}")

    print("\nlargest moves by |delta| (early_down_pass_rate):")
    _, mv = deltas(df, "early_down_pass_rate")
    mv["abs"] = mv["delta"].abs()
    cols = ["coach_name", "from_school", "to_school", "season_to", "delta"]
    print(mv.nlargest(8, "abs")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
