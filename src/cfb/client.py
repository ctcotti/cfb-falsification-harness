"""CFBD API client: on-disk response cache + call budgeting.

Budget is a real constraint, not a nicety. The $5/mo tier allows 30,000 calls
per month and the play-level attribution backfill (/plays/stats, hard-capped at
2000 rows per request) needs roughly 800 calls per season. A runaway loop burns
the month, so every call passes a budget guard and every response is cached
content-addressed on (endpoint, sorted params). Re-running any backfill is free.

Completed seasons are immutable and cached permanently. The in-progress season
carries a TTL so late-arriving corrections are picked up.

Deliberately stdlib-only so the harness bootstraps before any pip install.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

BASE_URL = "https://api.collegefootballdata.com"
REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data" / "cache"
LEDGER = CACHE_DIR / "_ledger.jsonl"

# Leave headroom under the 30k tier ceiling; raise rather than silently spending
# the last of the month's budget on a buggy loop.
MONTHLY_CEILING = 30_000
BUDGET_GUARD = 25_000

CURRENT_SEASON_TTL = timedelta(hours=12)
MIN_INTERVAL_S = 0.12  # polite pacing between live calls


class BudgetExceeded(RuntimeError):
    """Raised before a call that would push us past BUDGET_GUARD."""


class CFBDError(RuntimeError):
    pass


def current_season(now: datetime | None = None) -> int:
    """CFB season label. The season year rolls over in August."""
    now = now or datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1


def _load_key() -> str:
    key = os.environ.get("CFBD_API_KEY")
    if not key:
        env = REPO_ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("CFBD_API_KEY="):
                    key = line.split("=", 1)[1].strip()
                    break
    if not key:
        raise CFBDError("CFBD_API_KEY not found in environment or .env")
    return key


def _canonical(endpoint: str, params: dict[str, Any]) -> tuple[str, str]:
    """Return (query string, cache key). Params sorted so the key is stable."""
    clean = {k: v for k, v in sorted(params.items()) if v is not None}
    qs = urllib.parse.urlencode(clean, doseq=True)
    raw = endpoint.strip("/") + "?" + qs
    return qs, hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _slug(endpoint: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", endpoint.strip("/").lower()).strip("_")


@dataclass
class Response:
    data: Any
    from_cache: bool
    fetched_at: datetime
    endpoint: str
    params: dict[str, Any]

    def __len__(self) -> int:
        return len(self.data) if isinstance(self.data, list) else 1


class CFBDClient:
    def __init__(
        self,
        cache_dir: Path = CACHE_DIR,
        budget_guard: int = BUDGET_GUARD,
        offline: bool = False,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget_guard = budget_guard
        self.offline = offline
        self._key = None if offline else _load_key()
        self._last_call = 0.0
        self.stats = {"hits": 0, "misses": 0, "bytes": 0}

    # ---------------------------------------------------------------- budget
    def calls_this_month(self) -> int:
        """Count live calls logged in the current UTC calendar month."""
        if not LEDGER.exists():
            return 0
        cutoff = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        n = 0
        with LEDGER.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if datetime.fromisoformat(rec["ts"]) >= cutoff:
                    n += 1
        return n

    def _log(self, endpoint: str, key: str, status: int, nbytes: int) -> None:
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "endpoint": endpoint,
            "key": key,
            "status": status,
            "bytes": nbytes,
        }
        with LEDGER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    # ----------------------------------------------------------------- cache
    def _paths(self, endpoint: str, key: str) -> tuple[Path, Path]:
        d = self.cache_dir / _slug(endpoint)
        d.mkdir(parents=True, exist_ok=True)
        return d / (key + ".json"), d / (key + ".meta.json")

    def _cache_valid(self, meta_path: Path, season: int | None) -> bool:
        if not meta_path.exists():
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        # Completed seasons never change.
        if season is not None and season < current_season():
            return True
        fetched = datetime.fromisoformat(meta["fetched_at"])
        return datetime.now(timezone.utc) - fetched < CURRENT_SEASON_TTL

    def is_cached(
        self, endpoint: str, params: dict[str, Any] | None = None, *,
        season: int | None = None,
    ) -> bool:
        """True if get() would be served from cache. Costs nothing."""
        params = dict(params or {})
        if season is not None:
            params.setdefault("year", season)
        _, key = _canonical(endpoint, params)
        return self._cache_valid(self._paths(endpoint, key)[1], season)

    # ------------------------------------------------------------------ core
    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        season: int | None = None,
        max_retries: int = 4,
        force: bool = False,
    ) -> Response:
        """GET an endpoint, serving from cache when possible.

        `season` tells the cache whether the underlying data is final. Pass it
        whenever the request is scoped to a single season. `force` bypasses the
        cache for endpoints whose value is the live reading (e.g. info/usage).
        """
        params = dict(params or {})
        if season is not None:
            params.setdefault("year", season)
        qs, key = _canonical(endpoint, params)
        data_path, meta_path = self._paths(endpoint, key)

        if not force and self._cache_valid(meta_path, season):
            self.stats["hits"] += 1
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return Response(
                data=json.loads(data_path.read_text(encoding="utf-8")),
                from_cache=True,
                fetched_at=datetime.fromisoformat(meta["fetched_at"]),
                endpoint=endpoint,
                params=params,
            )

        if self.offline:
            raise CFBDError("offline and no cache for " + endpoint + "?" + qs)

        used = self.calls_this_month()
        if used >= self.budget_guard:
            raise BudgetExceeded(
                str(used) + " calls used this month, guard is "
                + str(self.budget_guard) + " (tier ceiling "
                + str(MONTHLY_CEILING) + "). Refusing " + endpoint + "."
            )

        url = BASE_URL + "/" + endpoint.strip("/") + (("?" + qs) if qs else "")
        body = self._fetch(url, endpoint, key, max_retries)
        now = datetime.now(timezone.utc)

        data_path.write_text(json.dumps(body), encoding="utf-8")
        meta_path.write_text(
            json.dumps(
                {
                    "endpoint": endpoint,
                    "params": params,
                    "fetched_at": now.isoformat(),
                    "season": season,
                    "rows": len(body) if isinstance(body, list) else None,
                }
            ),
            encoding="utf-8",
        )
        self.stats["misses"] += 1
        return Response(body, False, now, endpoint, params)

    def _fetch(self, url: str, endpoint: str, key: str, max_retries: int) -> Any:
        delay = 1.0
        for attempt in range(max_retries):
            gap = time.monotonic() - self._last_call
            if gap < MIN_INTERVAL_S:
                time.sleep(MIN_INTERVAL_S - gap)
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": "Bearer " + str(self._key),
                    "Accept": "application/json",
                    "User-Agent": "cfb-coaching-fit/0.1",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=180) as resp:
                    raw = resp.read()
                    self._last_call = time.monotonic()
                    self._log(endpoint, key, resp.status, len(raw))
                    self.stats["bytes"] += len(raw)
                    return json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                self._last_call = time.monotonic()
                self._log(endpoint, key, exc.code, 0)
                if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                detail = exc.read().decode("utf-8", "replace")[:300]
                raise CFBDError(
                    "HTTP " + str(exc.code) + " on " + url + ": " + detail
                ) from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise CFBDError("network failure on " + url + ": " + str(exc)) from exc
        raise CFBDError("exhausted retries on " + url)

    # ------------------------------------------------------------- utilities
    def usage(self) -> dict[str, Any]:
        """Server-side usage for the live rolling window."""
        return self.get("info/usage", force=True).data

    def paged_by_week(
        self, endpoint: str, season: int, weeks: Iterable[int], **params: Any
    ) -> list[dict]:
        """Concatenate a week-scoped endpoint across weeks for one season."""
        out: list[dict] = []
        for wk in weeks:
            resp = self.get(endpoint, dict(params, week=wk), season=season)
            if isinstance(resp.data, list):
                out.extend(resp.data)
        return out


if __name__ == "__main__":
    c = CFBDClient()
    print("calls this month (local ledger):", c.calls_this_month())
    print(json.dumps(c.usage().get("totals", {}), indent=2))
