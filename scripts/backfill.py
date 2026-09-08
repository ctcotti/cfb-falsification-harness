"""Backfill CFBD data into the local cache.

Two windows, per the split-fitting design:
  LONG   (2015-2025) -- nuisance parameters: efficiency, pace, home field, the
                        empirical margin distribution. Structurally stable, so
                        a long history is both safe and necessary.
  THESIS (2022-2025) -- collective/revenue-share era only. gamma, W, theta_c.

Deliberately prices itself before spending: `--dry-run` reports the number of
live calls the plan needs, grouped by endpoint. Nothing here fetches
/plays/stats -- QB attribution is parsed from play text and validated against
/plays/stats on a sample, which is two orders of magnitude cheaper.

Usage:
    python scripts/backfill.py --dry-run
    python scripts/backfill.py
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cfb.client import BudgetExceeded, CFBDClient, CFBDError  # noqa: E402

LONG_WINDOW = list(range(2015, 2026))
THESIS_WINDOW = list(range(2022, 2026))
PORTAL_WINDOW = list(range(2018, 2026))  # portal data does not predate this

Job = tuple[str, dict, int | None]


def season_scoped_jobs() -> list[Job]:
    """Everything that can be planned without first knowing the team list."""
    jobs: list[Job] = [("coaches", {"minYear": 2010, "maxYear": 2025}, None)]
    for year in LONG_WINDOW:
        jobs += [
            ("teams/fbs", {"year": year}, year),
            ("calendar", {"year": year}, year),
            ("games", {"year": year, "seasonType": "regular",
                       "classification": "fbs"}, year),
            ("games", {"year": year, "seasonType": "postseason",
                       "classification": "fbs"}, year),
            ("lines", {"year": year, "seasonType": "regular"}, year),
            ("recruiting/teams", {"year": year}, year),
            ("player/returning", {"year": year}, year),
            ("ratings/sp", {"year": year}, year),
        ]
    for year in PORTAL_WINDOW:
        jobs.append(("player/portal", {"year": year}, year))
    return jobs


def play_jobs(client: CFBDClient, years: list[int]) -> list[Job]:
    """Week-scoped play pulls, sized from /calendar rather than guessed."""
    jobs: list[Job] = []
    for year in years:
        try:
            cal = client.get("calendar", {"year": year}, season=year).data
        except (CFBDError, BudgetExceeded):
            weeks = list(range(0, 17))
        else:
            weeks = sorted({
                w["week"] for w in cal
                if w.get("seasonType") == "regular" and w.get("week") is not None
            }) or list(range(0, 17))
        for wk in weeks:
            jobs.append(("plays", {"year": year, "week": wk,
                                   "seasonType": "regular",
                                   "classification": "fbs"}, year))
    return jobs


def drive_jobs(client: CFBDClient, years: list[int]) -> list[Job]:
    """Week-scoped drive pulls, sized from /calendar exactly as plays are.

    Not redundant with /plays. The `clock` on a play is frozen at the drive's
    start value for 35-41% of drives in 2015-19, falling to 5-8% by 2024-25, so
    seconds-per-play is not recoverable from play-to-play deltas over most of
    the long window -- and because the defect shrinks monotonically across the
    panel it would enter any variance decomposition as a year effect. /drives
    carries `elapsed` directly and it is clean: on 2016 week 5, a season 37%
    frozen at play level, elapsed matches startTime - endTime on 100% of
    within-period drives, with 1 negative and 8 implausible values in 1,558.
    """
    jobs: list[Job] = []
    for year in years:
        try:
            cal = client.get("calendar", {"year": year}, season=year).data
        except (CFBDError, BudgetExceeded):
            weeks = list(range(0, 17))
        else:
            weeks = sorted({
                w["week"] for w in cal
                if w.get("seasonType") == "regular" and w.get("week") is not None
            }) or list(range(0, 17))
        for wk in weeks:
            jobs.append(("drives", {"year": year, "week": wk,
                                    "seasonType": "regular",
                                    "classification": "fbs"}, year))
    return jobs


def tenure_jobs(client: CFBDClient, years: list[int]) -> list[Job]:
    """One /coaches/tenures call per distinct FBS team.

    Tenures are the only source of a tenure-level hireDate. The hireDate on
    /coaches is the coach's first head-coaching hire ever, not the hire for the
    tenure in question, and is wrong for every coach on a second-plus job.
    """
    teams: set[str] = set()
    for year in years:
        try:
            rows = client.get("teams/fbs", {"year": year}, season=year).data
        except (CFBDError, BudgetExceeded):
            continue
        teams.update(t["school"] for t in rows if t.get("school"))
    return [("coaches/tenures", {"team": t}, None) for t in sorted(teams)]


def report(client: CFBDClient, jobs: list[Job], label: str) -> int:
    misses = Counter()
    for endpoint, params, season in jobs:
        if not client.is_cached(endpoint, params, season=season):
            misses[endpoint] += 1
    total = sum(misses.values())
    print(f"\n{label}: {len(jobs)} jobs, {total} live calls needed")
    for endpoint, n in misses.most_common():
        print(f"    {n:>5}  {endpoint}")
    if not misses:
        print("       0  (fully cached)")
    return total


def run(client: CFBDClient, jobs: list[Job], label: str) -> None:
    total = len(jobs)
    for i, (endpoint, params, season) in enumerate(jobs, 1):
        try:
            resp = client.get(endpoint, params, season=season)
        except BudgetExceeded as exc:
            print(f"\nSTOPPED: {exc}")
            raise SystemExit(2)
        except CFBDError as exc:
            print(f"  [{i}/{total}] FAIL {endpoint} {params}: {exc}")
            continue
        if not resp.from_cache:
            print(f"  [{i}/{total}] {endpoint} {params} -> {len(resp)} rows")
    print(f"{label}: done ({client.stats['hits']} cached, "
          f"{client.stats['misses']} fetched)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="price the plan without spending any calls")
    args = ap.parse_args()

    client = CFBDClient()
    used = client.calls_this_month()
    print(f"local ledger: {used} calls this month "
          f"(guard {client.budget_guard})")

    base = season_scoped_jobs()
    if args.dry_run:
        n = report(client, base, "phase 1 (season-scoped)")
        # Week-scoped pulls can only be priced once /calendar is cached.
        try:
            n += report(client, drive_jobs(client, LONG_WINDOW), "drives")
        except (CFBDError, BudgetExceeded):
            print(f"\ndrives: needs /calendar cached to plan; expect "
                  f"~{len(LONG_WINDOW) * 15} calls")
        print(f"\nphase 2 (tenures/plays) needs phase 1 cached to plan; "
              f"expect ~145 tenure calls + ~{len(LONG_WINDOW) * 15} play calls")
        print(f"\ntotal estimate: {n} calls")
        return

    run(client, base, "phase 1")
    tenures = tenure_jobs(client, LONG_WINDOW)
    run(client, tenures, "phase 2a (coach tenures)")
    plays = play_jobs(client, LONG_WINDOW)
    run(client, plays, "phase 2b (plays)")
    drives = drive_jobs(client, LONG_WINDOW)
    run(client, drives, "phase 2c (drives)")

    print(f"\ntotal live calls this month: {client.calls_this_month()}")


if __name__ == "__main__":
    main()
