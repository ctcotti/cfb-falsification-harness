"""Coach panel: one row per (school, season, coach), correctly dated.

This is the treatment-assignment table. Everything downstream -- theta_c, the
new-coach indicator, the as-of filtering in the walk-forward harness -- reads
from here, so the dating discipline lives in one place.

Two traps this module exists to avoid:

1. /coaches returns a `hireDate` that is the coach's FIRST head-coaching hire
   ever, not the hire for the tenure in question. It is 98.5% populated and
   entirely plausible-looking, and it is wrong for every coach on a second-plus
   job -- i.e. most of the treatment set. Verified: Moorhead/Akron 2022 reads
   2017-11-28 (his Mississippi State hire) there vs 2021-12-04 on
   /coaches/tenures. Tenure-level dates come from /coaches/tenures only.

2. `isInterim` does not catch every midseason hire. Kevin Whitley, Georgia
   Southern 2021, hireDate 2021-09-26, isInterim=False. Hire month is checked
   independently.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from .client import CFBDClient

# A coach hired after this month is arriving mid-season or too late to install
# a system; year one is not a clean natural experiment for him.
LATE_HIRE_MONTH = 7


def _parse_date(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.Timestamp(value).tz_convert("UTC")
    except (TypeError, ValueError):
        try:
            return pd.Timestamp(value).tz_localize("UTC")
        except (TypeError, ValueError):
            return None


def load_tenures(client: CFBDClient, schools: list[str]) -> pd.DataFrame:
    """Tenure-level records: the only trustworthy source of hire dates."""
    rows = []
    for school in schools:
        resp = client.get("coaches/tenures", {"team": school})
        for t in resp.data:
            coach = t.get("coach") or {}
            team = t.get("team") or {}
            rows.append({
                "school": team.get("school") or school,
                "coach_id": coach.get("id"),
                "coach_name": f"{coach.get('firstName','')} "
                              f"{coach.get('lastName','')}".strip(),
                "hire_date": _parse_date(t.get("hireDate")),
                "effective_start": _parse_date(t.get("effectiveStart")),
                "effective_end": _parse_date(t.get("effectiveEnd")),
                "start_year": t.get("startYear"),
                "end_year": t.get("endYear"),
                "is_interim": bool(t.get("isInterim")),
                "attribution_complete": bool(t.get("attributionComplete")),
            })
    return pd.DataFrame(rows)


def load_coach_seasons(client: CFBDClient, min_year: int,
                       max_year: int) -> pd.DataFrame:
    """(coach, school, season) rows. Reliable for *who*, not for *when*."""
    resp = client.get("coaches", {"minYear": min_year, "maxYear": max_year})
    rows = []
    for c in resp.data:
        for s in c.get("seasons") or []:
            rows.append({
                "coach_id": c.get("id"),
                "coach_name": f"{c.get('firstName','')} "
                              f"{c.get('lastName','')}".strip(),
                "school": s.get("school"),
                "season": s.get("year"),
                "games": s.get("games"),
                "sp_offense": s.get("spOffense"),
                "sp_defense": s.get("spDefense"),
            })
    return pd.DataFrame(rows)


def fbs_schools(client: CFBDClient, years: list[int]) -> dict[int, set[str]]:
    out = {}
    for year in years:
        rows = client.get("teams/fbs", {"year": year}, season=year).data
        out[year] = {t["school"] for t in rows if t.get("school")}
    return out


def build_panel(client: CFBDClient, years: list[int]) -> pd.DataFrame:
    """One row per (school, season, coach), FBS only, with treatment flags."""
    fbs = fbs_schools(client, years)
    all_schools = sorted(set().union(*fbs.values()))

    seasons = load_coach_seasons(client, min(years) - 5, max(years))
    seasons = seasons[seasons["season"].isin(years)]
    seasons = seasons[[
        s in fbs.get(y, set())
        for s, y in zip(seasons["school"], seasons["season"])
    ]].copy()

    tenures = load_tenures(client, all_schools)

    # Attach the tenure covering this (coach, school, season).
    merged = seasons.merge(
        tenures.drop(columns=["coach_name"]),
        on=["coach_id", "school"], how="left", suffixes=("", "_ten"),
    )
    covers = (
        merged["start_year"].le(merged["season"])
        & (merged["end_year"].isna() | merged["end_year"].ge(merged["season"]))
    )
    # Keep the covering tenure; fall back to the row itself if none matched so
    # the coach is never silently dropped from the panel.
    merged = merged[covers | merged["start_year"].isna()].copy()
    merged = merged.sort_values(["school", "season", "coach_id", "start_year"])
    merged = merged.drop_duplicates(["school", "season", "coach_id"], keep="last")

    # Treatment flags.
    merged["games"] = merged["games"].fillna(0).astype(int)
    merged["n_coaches_in_cell"] = merged.groupby(
        ["school", "season"])["coach_id"].transform("size")

    # "Sole coach of record" is the wrong criterion twice over. CFBD emits
    # phantom 0-game rows for outgoing coaches whose endYear runs a season long
    # (Holgorsen appears in Houston 2024 with 0 games, which would exclude
    # Fritz, who coached all 12). And a late interim taking a bowl game should
    # not disqualify the man who coached the other twelve. The unit is the
    # coach of record -- the one who actually worked the season.
    cell = merged.groupby(["school", "season"])["games"]
    merged["n_active_coaches"] = merged["games"].gt(0).groupby(
        [merged["school"], merged["season"]]).transform("sum")
    merged["cell_games"] = cell.transform("sum")
    merged["games_share"] = (
        merged["games"] / merged["cell_games"].replace(0, pd.NA))
    merged["is_primary_coach"] = (
        merged["games"].eq(cell.transform("max")) & merged["games"].gt(0))
    merged["hire_month"] = merged["hire_date"].dt.month
    merged["hire_year"] = merged["hire_date"].dt.year

    # tenure_year must come from hire_date, NOT start_year. CFBD's startYear is
    # FBS-scoped: it reports a coach's first *FBS* season at the school, not his
    # first season there. Cignetti/James Madison reads startYear=2022 against a
    # 2018-12-14 hire (JMU played FCS through 2021); Bohannon/Kennesaw State
    # reads startYear=2024 against a 2013-03-24 hire. Using startYear labels
    # every FCS-to-FBS reclassification coach as year-one.
    hire_season = merged["hire_year"].where(
        merged["hire_month"].lt(LATE_HIRE_MONTH), merged["hire_year"] + 1)
    merged["hire_season"] = hire_season
    merged["tenure_year"] = (merged["season"] - hire_season + 1).astype("Float64")
    # Fall back to startYear only where the tenure carries no hire date.
    fallback = merged["season"] - merged["start_year"] + 1
    merged["tenure_year"] = merged["tenure_year"].fillna(fallback)
    merged["first_season_at_school"] = merged["tenure_year"].eq(1)

    # A school's first FBS season is its own confound -- new schedule, roster
    # built for a lower division -- so it is never a clean coaching experiment.
    debut = {s: min(y for y in years if s in fbs.get(y, set()))
             for s in all_schools}
    first_year = min(years)
    merged["fbs_debut_season"] = merged["school"].map(debut)
    merged["is_fbs_debut"] = (
        merged["season"].eq(merged["fbs_debut_season"])
        & merged["fbs_debut_season"].gt(first_year)  # left-censored, not a debut
    )

    # A clean year-one observation: new to the school, sole coach of record,
    # not an interim, and hired early enough to install a system.
    early = (
        merged["hire_date"].notna()
        & (
            merged["hire_year"].lt(merged["season"])
            | merged["hire_month"].lt(LATE_HIRE_MONTH)
        )
    )
    merged["clean_year_one"] = (
        merged["first_season_at_school"]
        & merged["is_primary_coach"]
        & ~merged["is_interim"]
        & ~merged["is_fbs_debut"]
        & early
    )
    merged["built_at"] = datetime.now(timezone.utc).isoformat()
    return merged.reset_index(drop=True)


def treatment_set(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[panel["clean_year_one"]].copy()
