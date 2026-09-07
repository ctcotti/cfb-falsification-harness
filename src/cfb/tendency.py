"""Coach play-call tendency: the d_c side of Phi.

Everything here is a *choice the coach makes*, never an outcome. Classifying a
coach by offensive output would be circular, because output is co-produced by
the quarterback -- which is the whole thing Phi is trying to measure.

All rates are computed on neutral game states. Unconditioned play-call rates
measure game script, not scheme: a team trailing by 21 throws on every down
regardless of what its coach believes. Filtering to neutral states costs sample
but is the difference between measuring a system and measuring a scoreboard.
"""
from __future__ import annotations

import json
import glob
import os
from collections import defaultdict

import pandas as pd

from .playtext import DROPBACK_TYPES, RUSH_TYPES, attribute

# Neutral script: competitive, before the endgame, out of both red zones.
NEUTRAL_MARGIN = 14
NEUTRAL_PERIODS = (1, 2, 3)
FIELD_MIN, FIELD_MAX = 20, 80

# Early downs only. Third down is a situation, not an identity.
EARLY_DOWNS = (1, 2)
SECOND_DOWN_MAX_DISTANCE = 7

KNEEL_MARKERS = ("kneel", "takes a knee")


def _neutral(p: dict) -> bool:
    if p.get("period") not in NEUTRAL_PERIODS:
        return False
    off, deff = p.get("offenseScore"), p.get("defenseScore")
    if off is None or deff is None or abs(off - deff) > NEUTRAL_MARGIN:
        return False
    ytg = p.get("yardsToGoal")
    if ytg is None or not (FIELD_MIN <= ytg <= FIELD_MAX):
        return False
    return True


def _early_down(p: dict) -> bool:
    down, dist = p.get("down"), p.get("distance")
    if down not in EARLY_DOWNS or dist is None:
        return False
    return down == 1 or dist <= SECOND_DOWN_MAX_DISTANCE


def _is_kneel(text: str | None) -> bool:
    t = (text or "").lower()
    return any(m in t for m in KNEEL_MARKERS)


def iter_cached_plays(cache_dir: str, season: int):
    """Stream plays for one season off disk. 1.1GB total, so never load all."""
    for meta_path in glob.glob(os.path.join(cache_dir, "plays", "*.meta.json")):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta["params"].get("year") != season:
            continue
        data_path = meta_path.replace(".meta.json", ".json")
        if not os.path.exists(data_path):
            continue
        with open(data_path, encoding="utf-8") as fh:
            for play in json.load(fh):
                yield play


def season_tendency(cache_dir: str, season: int) -> pd.DataFrame:
    """One row per (team, season) of play-call rates on neutral early downs."""
    # Pass 1: identify each team's primary passer, so QB runs can be separated
    # from running-back runs. Uses all dropbacks, not just neutral ones.
    passers: dict[str, defaultdict] = defaultdict(lambda: defaultdict(int))
    for p in iter_cached_plays(cache_dir, season):
        if p.get("playType") in DROPBACK_TYPES:
            a = attribute(p.get("playType"), p.get("playText"))
            if a.name:
                passers[p["offense"]][a.name] += 1
    primary = {
        team: max(names.items(), key=lambda kv: kv[1])[0]
        for team, names in passers.items() if names
    }

    acc: dict[str, dict] = defaultdict(lambda: {
        "ed_plays": 0, "ed_pass": 0,
        "rush_all": 0, "rush_qb": 0,
        "dropbacks": 0,
    })

    for p in iter_cached_plays(cache_dir, season):
        team = p.get("offense")
        ptype = p.get("playType")
        if not team or _is_kneel(p.get("playText")):
            continue
        is_pass = ptype in DROPBACK_TYPES
        is_rush = ptype in RUSH_TYPES
        if not (is_pass or is_rush):
            continue
        a = acc[team]
        if is_pass:
            a["dropbacks"] += 1
        if not _neutral(p):
            continue

        if _early_down(p):
            a["ed_plays"] += 1
            a["ed_pass"] += int(is_pass)

        if is_rush:
            a["rush_all"] += 1
            who = attribute(ptype, p.get("playText"))
            if who.name and who.name == primary.get(team):
                a["rush_qb"] += 1

    rows = []
    for team, a in acc.items():
        if a["ed_plays"] < 100:  # too thin to characterise a system
            continue
        rows.append({
            "school": team,
            "season": season,
            "primary_passer": primary.get(team),
            "early_down_pass_rate": a["ed_pass"] / a["ed_plays"],
            "qb_run_share": (a["rush_qb"] / a["rush_all"]
                             if a["rush_all"] else None),
            "ed_plays": a["ed_plays"],
            "rush_all": a["rush_all"],
            "dropbacks": a["dropbacks"],
        })
    return pd.DataFrame(rows)


def build_tendency(cache_dir: str, seasons: list[int]) -> pd.DataFrame:
    return pd.concat(
        [season_tendency(cache_dir, s) for s in seasons], ignore_index=True
    )
