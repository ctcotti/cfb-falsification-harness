"""Game panel: outcomes joined to market prices, at team-game level.

CFBD's `spread` is the home-team number (positive means the home side is
getting points), so the market's implied home margin is -spread. That
convention is verified against realised margins rather than assumed.

Health warning carried through to every downstream result: CFBD lines have no
timestamp. `spread` is whatever CFBD last recorded, not a verified close. It is
adequate for a Stage 3 gate, which asks whether a signal is in the price at
all; it is NOT adequate for CLV, which needs a timestamped close from a second
source.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"
OUT = ROOT / "data" / "panel"


def _rows(endpoint: str):
    for meta in glob.glob(str(CACHE / endpoint / "*.meta.json")):
        data = meta.replace(".meta.json", ".json")
        if not os.path.exists(data):
            continue
        with open(data, encoding="utf-8") as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            yield from payload


def load_lines() -> pd.DataFrame:
    recs = []
    for g in _rows("lines"):
        spreads = [ln["spread"] for ln in (g.get("lines") or [])
                   if ln.get("spread") is not None]
        opens = [ln["spreadOpen"] for ln in (g.get("lines") or [])
                 if ln.get("spreadOpen") is not None]
        if not spreads:
            continue
        recs.append({
            "game_id": g["id"], "season": g["season"], "week": g["week"],
            "season_type": g.get("seasonType"),
            "home": g["homeTeam"], "away": g["awayTeam"],
            "home_points": g.get("homeScore"), "away_points": g.get("awayScore"),
            "spread": float(np.median(spreads)),
            "spread_open": float(np.median(opens)) if opens else np.nan,
            "n_books": len(spreads),
        })
    return pd.DataFrame(recs).drop_duplicates("game_id")


def to_team_game(lines: pd.DataFrame) -> pd.DataFrame:
    """Two rows per game, one per team, from that team's perspective."""
    lines = lines.dropna(subset=["home_points", "away_points"])
    lines = lines[lines["season_type"] == "regular"]
    home = pd.DataFrame({
        "game_id": lines.game_id, "season": lines.season, "week": lines.week,
        "school": lines.home, "opponent": lines.away, "is_home": 1,
        "margin": lines.home_points - lines.away_points,
        # market's implied margin for this team
        "market_margin": -lines.spread,
        "market_margin_open": -lines.spread_open,
        "n_books": lines.n_books,
    })
    away = pd.DataFrame({
        "game_id": lines.game_id, "season": lines.season, "week": lines.week,
        "school": lines.away, "opponent": lines.home, "is_home": 0,
        "margin": lines.away_points - lines.home_points,
        "market_margin": lines.spread,
        "market_margin_open": lines.spread_open,
        "n_books": lines.n_books,
    })
    tg = pd.concat([home, away], ignore_index=True)
    tg["resid"] = tg["margin"] - tg["market_margin"]
    return tg


def main() -> None:
    lines = load_lines()
    tg = to_team_game(lines)
    OUT.mkdir(parents=True, exist_ok=True)
    tg.to_parquet(OUT / "team_games.parquet", index=False)

    print(f"games with a line: {len(lines)}   team-games: {len(tg)}")
    print(f"seasons: {int(tg.season.min())}-{int(tg.season.max())}")
    print(f"median books per game: {lines.n_books.median():.0f}")
    print(f"opening spread available: {lines.spread_open.notna().mean()*100:.1f}%")

    # Verify the sign convention rather than trusting it.
    r = np.corrcoef(tg.margin, tg.market_margin)[0, 1]
    print(f"\ncorr(margin, market_margin) = {r:+.3f}   (must be strongly +)")
    print(f"mean resid = {tg.resid.mean():+.3f} pts   "
          f"(market should be ~unbiased)")
    print(f"SD  resid = {tg.resid.std():.2f} pts       "
          f"(the sigma_game in every power calc)")
    hfa = tg[tg.is_home == 1].market_margin.mean() - tg[tg.is_home == 0].market_margin.mean()
    print(f"implied home-field edge = {hfa/2:+.2f} pts  (sanity: ~2-3)")


if __name__ == "__main__":
    main()
