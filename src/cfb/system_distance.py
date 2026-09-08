"""System distance: Phi = -(d_current - d_prior)^2.

The surviving form of the thesis. There is no QB attribute vector -- the
within-QB situational contrast had zero split-half reliability, so q_p is not
measurable at CFBD play counts. What remains is adaptation cost: a quarterback
is now in a system some distance from the one he last played in, and the
market, pricing additive power ratings, has no term for that distance.

The hypothesis still has no main effect. Distance is symmetric, so neither the
coach being good nor the QB being good predicts it.

Both sides use the SAME estimator, which matters more than either being exactly
right. Comparing a coach's multi-season mean against a single school-season
value put a 0.57 z-unit noise floor under the distance and attenuated gamma by
roughly 30%. Here both sides are shrunk multi-season estimates of a coach's
play-calling, evaluated as of the same date, so a returning QB under a
continuing coach gets a distance of exactly zero rather than a noisy near-zero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Seasons of history used for a coach's system estimate, and the per-season
# decay. Recent seasons weigh more; a coach four years ago is still evidence.
WINDOW = 4
DECAY = 0.75
# Shrinkage toward the league mean, in units of effective seasons. A coach with
# one season of tape is a weak estimate and should be pulled toward average.
SHRINK_K = 1.5

ERA_BREAK = 2022  # collective era begins


def coach_system_estimates(tendency: pd.DataFrame,
                           panel: pd.DataFrame,
                           axis: str = "early_down_pass_rate") -> pd.DataFrame:
    """Shrunk, decayed estimate of each coach's system as of each season.

    Returned keyed on (coach_id, as_of_season), using only seasons strictly
    before as_of_season -- so it is always usable ex ante.

    `axis` selects the column to summarise. It defaults to the Phase 1 axis so
    every existing caller is unchanged; Phase 2 passes a pace column through the
    same estimator, which matters because the decay, the shrinkage constant and
    the as-of rule are what make the two comparable.
    """
    t = tendency.copy()
    t["dz"] = t.groupby("season")[axis].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))
    coach = panel[panel["is_primary_coach"]][["school", "season", "coach_id"]]
    hist = t.merge(coach, on=["school", "season"])[["coach_id", "season", "dz"]]

    seasons = sorted(t["season"].unique())
    rows = []
    for cid, g in hist.groupby("coach_id"):
        by_season = dict(zip(g["season"], g["dz"]))
        for as_of in seasons:
            num = den = 0.0
            for lag in range(1, WINDOW + 1):
                val = by_season.get(as_of - lag)
                if val is None:
                    continue
                w = DECAY ** (lag - 1)
                num += w * val
                den += w
            if den == 0:
                continue
            rows.append({
                "coach_id": cid,
                "as_of_season": as_of,
                "d_raw": num / den,
                "eff_seasons": den,
                # Shrink toward 0 (the season mean, by construction of dz).
                "d": (num / den) * (den / (den + SHRINK_K)),
            })
    return pd.DataFrame(rows)


def system_reliability(tendency: pd.DataFrame, panel: pd.DataFrame,
                       seed: int = 11) -> float:
    """Split-sample reliability of a coach's system estimate.

    Randomly halve each coach's seasons, estimate independently on each half,
    correlate across coaches, Spearman-Brown correct. This is the attenuation
    factor on gamma: measured directly rather than inferred from residuals.
    """
    t = tendency.copy()
    t["dz"] = t.groupby("season")["early_down_pass_rate"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))
    coach = panel[panel["is_primary_coach"]][["school", "season", "coach_id"]]
    hist = t.merge(coach, on=["school", "season"])[["coach_id", "season", "dz"]]

    rng = np.random.default_rng(seed)
    a, b = [], []
    for _, g in hist.groupby("coach_id"):
        if len(g) < 4:
            continue
        idx = rng.permutation(len(g))
        h1, h2 = idx[: len(idx) // 2], idx[len(idx) // 2:]
        a.append(g["dz"].to_numpy()[h1].mean())
        b.append(g["dz"].to_numpy()[h2].mean())
    r = float(np.corrcoef(a, b)[0, 1])
    return 2 * r / (1 + r), len(a)


def build_phi(tendency: pd.DataFrame, panel: pd.DataFrame,
              qb_profiles: pd.DataFrame, min_prior_dropbacks: int = 100
              ) -> pd.DataFrame:
    """One row per team-season with Phi and the flags Stage 3 needs."""
    est = coach_system_estimates(tendency, panel)
    coach = panel[panel["is_primary_coach"]][
        ["school", "season", "coach_id", "coach_name", "tenure_year"]]

    q = qb_profiles.sort_values("dropbacks", ascending=False)
    qb1 = q.drop_duplicates(["school", "season"])[
        ["school", "season", "passer", "dropbacks"]]
    # Where the QB actually played last season, and how much.
    vol = (q.groupby(["passer", "season"])
             .agg(prior_db=("dropbacks", "sum")).reset_index())
    where = (q.drop_duplicates(["passer", "season"])[
        ["passer", "season", "school"]]
        .rename(columns={"school": "prev_school", "season": "prev_season"}))

    df = coach.merge(qb1, on=["school", "season"])
    df["prev_season"] = df["season"] - 1
    df = df.merge(vol.rename(columns={"season": "prev_season"}),
                  on=["passer", "prev_season"], how="left")
    df = df.merge(where, on=["passer", "prev_season"], how="left")

    # Current system: this coach, as of this season.
    df = df.merge(est.rename(columns={"as_of_season": "season", "d": "d_current"})[
        ["coach_id", "season", "d_current", "eff_seasons"]],
        on=["coach_id", "season"], how="left")

    # Prior system: the coach of the team the QB played for last season,
    # evaluated as of the same date so both sides are the same estimator.
    prev_coach = coach.rename(columns={
        "school": "prev_school", "season": "prev_season",
        "coach_id": "prev_coach_id"})[["prev_school", "prev_season",
                                       "prev_coach_id"]]
    df = df.merge(prev_coach, on=["prev_school", "prev_season"], how="left")
    df = df.merge(
        est.rename(columns={"coach_id": "prev_coach_id",
                            "as_of_season": "season", "d": "d_prior"})[
            ["prev_coach_id", "season", "d_prior"]],
        on=["prev_coach_id", "season"], how="left")

    df["transfer"] = df["prev_school"].notna() & df["prev_school"].ne(df["school"])
    df["new_coach"] = df["tenure_year"].eq(1)
    df["qb_has_tape"] = df["prior_db"].fillna(0).ge(min_prior_dropbacks)
    df["measurable"] = (df["d_current"].notna() & df["d_prior"].notna()
                        & df["qb_has_tape"])
    df["distance"] = df["d_current"] - df["d_prior"]
    df["phi"] = -(df["distance"] ** 2)
    df["live"] = df["measurable"] & (df["transfer"] | df["new_coach"])
    df["era_collective"] = df["season"].ge(ERA_BREAK).astype(int)
    return df
