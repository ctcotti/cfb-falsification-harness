"""Quarterback situational profile: the q_p side of Phi.

The axis has to be the same one d_c lives on -- pass orientation of the system.
The QB-side analogue is not "is he good at passing" (that is a main effect, and
the market prices it) but "does he hold up when the defense knows a pass is
coming", which is what a pass-heavy system demands of him.

    q_p = mean(EPA | dropback, obvious passing down)
        - mean(EPA | dropback, neutral early down)

Two properties make this the right estimator rather than a convenient one:

1. Both bins are restricted to *dropbacks by this quarterback*. Including
   handoffs in the early-down bin would measure the running back.

2. It is a within-QB contrast, so a QB's overall passing quality -- a level
   shift affecting both bins equally -- differences out. What survives is his
   situational profile alone. That is exactly the structure the thesis needs:
   no main effect, all interaction.

Obvious passing downs are the identifying trick. On 3rd-and-8 every coach in
the country throws, so the play call is forced by the situation rather than
chosen by the coach. That is what makes the bin a measurement of the player
instead of another measurement of his previous offensive coordinator.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .playtext import DROPBACK_TYPES, attribute
from .tendency import _early_down, _is_kneel, iter_cached_plays

# Obvious passing down: the defense knows, the coach has no real choice.
OPD_DOWNS = (3, 4)
OPD_MIN_DISTANCE = 7

# q_p uses a looser script filter than d_c, deliberately. The neutral-script
# filter exists to stop d_c measuring the scoreboard instead of the scheme --
# a team down 21 throws every down regardless of what its coach believes. That
# rationale does not carry over here: on an obvious passing down the call is
# already forced, and EPA is itself down/distance/field-position adjusted. The
# only thing genuinely worth excluding is true garbage time, where defensive
# intent changes. Using the strict filter cost roughly half the denominator on
# the bin that was already the binding constraint.
GARBAGE_PERIOD = 4
GARBAGE_MARGIN = 21


def _script_ok(p: dict) -> bool:
    period = p.get("period") or 0
    off, deff = p.get("offenseScore"), p.get("defenseScore")
    if off is None or deff is None:
        return False
    return not (period >= GARBAGE_PERIOD and abs(off - deff) > GARBAGE_MARGIN)


def _obvious_passing_down(p: dict) -> bool:
    down, dist = p.get("down"), p.get("distance")
    return down in OPD_DOWNS and dist is not None and dist >= OPD_MIN_DISTANCE


def season_qb_profiles(cache_dir: str, season: int) -> pd.DataFrame:
    """One row per (passer, team, season) with both situational bins."""
    acc: dict[tuple[str, str], dict] = defaultdict(lambda: {
        "ed_n": 0, "ed_sum": 0.0,
        "opd_n": 0, "opd_sum": 0.0,
        "dropbacks": 0,
    })

    for p in iter_cached_plays(cache_dir, season):
        ptype = p.get("playType")
        if ptype not in DROPBACK_TYPES or _is_kneel(p.get("playText")):
            continue
        who = attribute(ptype, p.get("playText"))
        if not who.name:
            continue
        team = p.get("offense")
        if not team:
            continue
        a = acc[(who.name, team)]
        a["dropbacks"] += 1

        ppa = p.get("ppa")
        if ppa is None or not _script_ok(p):
            continue
        if _obvious_passing_down(p):
            a["opd_n"] += 1
            a["opd_sum"] += ppa
        elif _early_down(p):
            a["ed_n"] += 1
            a["ed_sum"] += ppa

    rows = []
    for (name, team), a in acc.items():
        if a["ed_n"] == 0 or a["opd_n"] == 0:
            continue
        ed = a["ed_sum"] / a["ed_n"]
        opd = a["opd_sum"] / a["opd_n"]
        rows.append({
            "passer": name,
            "school": team,
            "season": season,
            "dropbacks": a["dropbacks"],
            "ed_n": a["ed_n"], "ed_epa": ed,
            "opd_n": a["opd_n"], "opd_epa": opd,
            "contrast_raw": opd - ed,
        })
    return pd.DataFrame(rows)


def build_qb_profiles(cache_dir: str, seasons: list[int]) -> pd.DataFrame:
    return pd.concat(
        [season_qb_profiles(cache_dir, s) for s in seasons], ignore_index=True
    )


def shrink(df: pd.DataFrame, value: str = "contrast_raw",
           n_col: str = "opd_n") -> pd.DataFrame:
    """Empirical-Bayes shrinkage toward the season mean.

    Measurement error in the contrast is largest exactly where the contrast is
    most extreme, because an extreme value usually means a thin denominator.
    Ranking on an unshrunk contrast would therefore select noise -- the same
    winner's-curse mechanism that Layer 3 corrects for at the bet level, applied
    here at the measurement level.

    weight = n / (n + n0),  n0 = sigma2_within / sigma2_between
    """
    out = df.copy()
    parts = []
    for season, g in out.groupby("season"):
        g = g.copy()
        # Per-play variance of EPA, pooled, gives the within-QB noise scale.
        total_var = float(np.var(g[value], ddof=1))
        # Sampling variance of the contrast for a QB with n_opd plays: the OPD
        # bin dominates because it is the thinner of the two.
        per_play = float(np.var(g["opd_epa"], ddof=1) * g[n_col].median())
        sigma2_within = per_play / g[n_col]
        sigma2_between = max(total_var - float(sigma2_within.mean()), 1e-6)
        n0 = per_play / sigma2_between
        w = g[n_col] / (g[n_col] + n0)
        mu = float(np.average(g[value], weights=g[n_col]))
        g["shrink_weight"] = w
        g["contrast"] = mu + w * (g[value] - mu)
        g["n0"] = n0
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    # Standardise within season so the axis is comparable to d_c.
    out["q_p"] = out.groupby("season")["contrast"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))
    return out
