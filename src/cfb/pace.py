"""Neutral-script, opponent-adjusted pace: the axis for Phase 2.

WHY SECONDS PER PLAY AND NOT THE OTHER TWO

Under the Phase 1 identity T = D(pi_ij + pi_ji), the quantity whose mispricing
moves a total is D, drives per game. But D is not a coach attribute. Three
measures are linked by one identity -- game seconds = D x plays_per_drive x
seconds_per_play -- and they are not equally admissible:

  drives per game     jointly produced. 70% of the variance in a team's
                      drive count is game-level, shared with the opponent by
                      construction, and its season mean carries a persistent
                      conference schedule component that would read as a school
                      effect in a travel test.
  plays per drive     an OUTCOME. More first downs means longer drives. Phase 1's
                      rule -- classify a coach by choices, never by output --
                      excludes it, and for the same reason: output is co-produced
                      by the roster.
  seconds per play    the choice. How long the offense takes between snaps is
                      decided on the sideline. It is the channel through which a
                      coach actually moves D.

So the axis is seconds per play, and D is left as the thing it acts on.

The honest caveat, measured rather than waved at: seconds per play is not a pure
choice. An incompletion stops the clock, so a pass-heavy offense mechanically
posts a lower figure. That makes the axis partly a restatement of Phase 1's
early-down pass rate, which the Phase 2 brief forbids. build_pace() reports the
correlation; it is not tuned to make it small.

WHY /drives AND NOT /plays

The `clock` on a play is frozen at the drive's start value for 35-41% of drives
in 2015-19, 5-8% by 2024-25. Play-to-play deltas are therefore unavailable over
most of the long window, and the defect shrinks monotonically across seasons --
it would enter a variance decomposition as a year effect. /drives carries
`elapsed` directly and cleanly.
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# Neutral script, matched to tendency.py so the two axes are filtered alike.
NEUTRAL_MARGIN = 14
NEUTRAL_PERIODS = (1, 2, 3)

# A drive starting inside the last two minutes of a half is a two-minute drill:
# tempo is dictated by the clock, not by the coach.
HALF_END_PERIODS = (2, 4)
TWO_MINUTE_S = 120

# One- and two-play drives are dominated by a single explosive play or a
# turnover; seconds-per-play there measures the play, not the tempo.
MIN_DRIVE_PLAYS = 3
# Guard rails for source noise -- validated as ~1 in 1,558 on a sample week.
MIN_SEC_PER_PLAY, MAX_SEC_PER_PLAY = 4.0, 90.0
# Too thin to characterise a team-season.
MIN_TEAM_DRIVES = 60


def _sec(t: dict | None) -> int | None:
    if not t or t.get("minutes") is None:
        return None
    return int(t["minutes"]) * 60 + int(t["seconds"])


def iter_cached_drives(cache_dir: str, season: int):
    for meta_path in glob.glob(os.path.join(cache_dir, "drives", "*.meta.json")):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta["params"].get("year") != season:
            continue
        data_path = meta_path.replace(".meta.json", ".json")
        if not os.path.exists(data_path):
            continue
        with open(data_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            yield from payload


def _neutral(d: dict) -> bool:
    if d.get("startPeriod") not in NEUTRAL_PERIODS:
        return False
    # A drive that crosses a period boundary carries a built-in stoppage and an
    # ambiguous elapsed; 7.5% of drives, cheap to drop.
    if d.get("endPeriod") != d.get("startPeriod"):
        return False
    off, deff = d.get("startOffenseScore"), d.get("startDefenseScore")
    if off is None or deff is None or abs(off - deff) > NEUTRAL_MARGIN:
        return False
    start = _sec(d.get("startTime"))
    if start is None:
        return False
    if d["startPeriod"] in HALF_END_PERIODS and start < TWO_MINUTE_S:
        return False
    return True


def drive_frame(cache_dir: str, seasons: list[int]) -> pd.DataFrame:
    """Qualifying drives, one row each, with the two team-season keys."""
    rows = []
    for season in seasons:
        for d in iter_cached_drives(cache_dir, season):
            if not _neutral(d):
                continue
            plays = d.get("plays")
            elapsed = _sec(d.get("elapsed"))
            if not plays or plays < MIN_DRIVE_PLAYS or elapsed is None:
                continue
            spp = elapsed / plays
            if not (MIN_SEC_PER_PLAY <= spp <= MAX_SEC_PER_PLAY):
                continue
            rows.append({
                "season": season,
                "game_id": d.get("gameId"),
                "offense": d.get("offense"),
                "defense": d.get("defense"),
                "plays": plays,
                "elapsed": elapsed,
                "spp": spp,
            })
    return pd.DataFrame(rows)


def _two_way_adjust(df: pd.DataFrame, value: str = "spp",
                    n_iter: int = 40, tol: float = 1e-8
                    ) -> tuple[pd.Series, pd.Series, float]:
    """Additive offense/defense team-season effects by alternating projection.

    y_d = mu + alpha_offense + delta_defense + e. Solved by sweeping means
    rather than building a 2,800-column design; the two-way problem is
    well conditioned here because every team plays a varied schedule.
    Weighted by plays, so a twelve-snap drive counts for more than a three.
    """
    w = df["plays"].to_numpy(float)
    y = df[value].to_numpy(float)
    off = pd.factorize(df["off_key"])[0]
    dfn = pd.factorize(df["def_key"])[0]
    n_off, n_def = off.max() + 1, dfn.max() + 1
    mu = float(np.average(y, weights=w))
    a = np.zeros(n_off)
    d = np.zeros(n_def)
    wsum_off = np.bincount(off, weights=w, minlength=n_off)
    wsum_def = np.bincount(dfn, weights=w, minlength=n_def)
    for _ in range(n_iter):
        r = y - mu - d[dfn]
        a_new = np.bincount(off, weights=w * r, minlength=n_off) / wsum_off
        a_new -= np.average(a_new, weights=wsum_off)
        r = y - mu - a_new[off]
        d_new = np.bincount(dfn, weights=w * r, minlength=n_def) / wsum_def
        d_new -= np.average(d_new, weights=wsum_def)
        if max(np.abs(a_new - a).max(), np.abs(d_new - d).max()) < tol:
            a, d = a_new, d_new
            break
        a, d = a_new, d_new
    resid = y - mu - a[off] - d[dfn]
    r2 = 1.0 - float(np.average(resid**2, weights=w)) / float(
        np.average((y - mu) ** 2, weights=w))
    off_lab = pd.factorize(df["off_key"])[1]
    def_lab = pd.factorize(df["def_key"])[1]
    return (pd.Series(a + mu, index=off_lab, name="pace_adj"),
            pd.Series(d, index=def_lab, name="pace_def_allowed"),
            r2)


def build_pace(cache_dir: str, seasons: list[int]) -> pd.DataFrame:
    """One row per (school, season): raw and opponent-adjusted seconds/play."""
    dr = drive_frame(cache_dir, seasons)
    dr["off_key"] = dr["offense"] + "|" + dr["season"].astype(str)
    dr["def_key"] = dr["defense"] + "|" + dr["season"].astype(str)

    raw = dr.groupby(["offense", "season"]).agg(
        drives=("spp", "size"),
        plays=("plays", "sum"),
        elapsed=("elapsed", "sum"),
    ).reset_index().rename(columns={"offense": "school"})
    raw["pace_raw"] = raw["elapsed"] / raw["plays"]
    raw = raw[raw["drives"] >= MIN_TEAM_DRIVES].copy()

    adj, allowed, r2 = _two_way_adjust(dr)
    adj = adj.rename_axis("off_key").reset_index()
    adj[["school", "season"]] = adj["off_key"].str.rsplit("|", n=1, expand=True)
    adj["season"] = adj["season"].astype(int)
    out = raw.merge(adj[["school", "season", "pace_adj"]],
                    on=["school", "season"], how="left")
    out.attrs["two_way_r2"] = r2

    # Detrend: league pace drifts hard across the decade, and an undetrended
    # difference would count that drift as instability -- same reason test 01
    # z-scores within season.
    for col in ("pace_raw", "pace_adj"):
        out[col + "_z"] = out.groupby("season")[col].transform(
            lambda s: (s - s.mean()) / s.std(ddof=0))
    return out
