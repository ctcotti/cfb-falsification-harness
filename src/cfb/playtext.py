"""Attribute scrimmage plays to the quarterback from CFBD play text.

Why parse text instead of using /plays/stats, which carries a structured
athleteId: /plays/stats is hard-capped at 2000 rows per request, so full
coverage costs ~800 calls per season (~3,200 for the thesis window). Play text
comes free with /plays, which we need anyway. The tradeoff is parse risk, so
this module is paired with a validation script that scores it against
/plays/stats on a sample of games and reports the disagreement rate.

Names are returned verbatim. Resolution to athlete IDs is a separate concern --
this module only answers "who was the quarterback on this play".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Plays where a quarterback is the responsible party. Interceptions count:
# the throw is his. Kneels are excluded downstream, not here.
DROPBACK_TYPES = {
    "Pass Reception", "Pass Incompletion", "Passing Touchdown",
    "Pass Interception Return", "Interception", "Sack",
    "Pass", "Pass Completion", "Pass Interception",
}
RUSH_TYPES = {"Rush", "Rushing Touchdown"}
SCRIMMAGE_TYPES = DROPBACK_TYPES | RUSH_TYPES

# A leading "X fumbles, X recovers, " clause displaces the actor, so it is
# stripped before matching rather than handled in every pattern.
_FUMBLE_PREFIX = re.compile(
    r"^.{2,40}?\s+fumbl(?:e|es|ed)[^,]*,\s*(?:.{2,40}?\s+recover(?:s|ed)[^,]*,\s*)?",
    re.I,
)

_NAME = r"([A-Z][A-Za-z.'\-]*(?:\s+[A-Za-z.'\-]+){0,3}?)"

# Ordered: first match wins. Standard formats first, box-score TD formats after.
_PASSER_PATTERNS = [
    re.compile(_NAME + r"\s+pass\s+(?:complete|incomplete|intercepted)", re.I),
    re.compile(_NAME + r"\s+pass\s*,\s*to\b", re.I),          # "D. Vick pass,to B. Cope"
    re.compile(_NAME + r"\s+sacked\s+(?:by|for)\b", re.I),
    # "Reid 15 Yd pass from Vick", also "... 46 Yd TD pass from ..."
    re.compile(r"\d+\s+Yd\s+(?:TD\s+)?pass\s+from\s+" + _NAME, re.I),
    re.compile(_NAME + r"\s+pass\s+intercepted", re.I),
]
_RUSHER_PATTERNS = [
    re.compile(_NAME + r"\s+run\s+for\b", re.I),
    re.compile(_NAME + r"\s+(?:rush|rushed)\s+for\b", re.I),
    # "Irons 17 Yd Run (Kick)", also "Farrow 26 Yd TD Run (Kick)"
    re.compile(_NAME + r"\s+\d+\s+Yd\s+(?:TD\s+)?Run\b", re.I),
    re.compile(_NAME + r"\s+kneel", re.I),
]

_TRAILING_NOISE = re.compile(r"\s*\((?:.*)\)\s*$")


@dataclass(frozen=True)
class Attribution:
    name: str | None
    role: str | None  # "passer" | "rusher"
    pattern: int | None


def _clean(name: str) -> str:
    name = _TRAILING_NOISE.sub("", name).strip(" ,.")
    return re.sub(r"\s+", " ", name)


def attribute(play_type: str | None, play_text: str | None) -> Attribution:
    """Identify the offensive actor on a scrimmage play."""
    if not play_text or play_type not in SCRIMMAGE_TYPES:
        return Attribution(None, None, None)
    text = _FUMBLE_PREFIX.sub("", play_text.strip(), count=1).strip()

    patterns, role = (
        (_PASSER_PATTERNS, "passer") if play_type in DROPBACK_TYPES
        else (_RUSHER_PATTERNS, "rusher")
    )
    for i, rx in enumerate(patterns):
        m = rx.search(text) if i else rx.match(text)
        if m:
            name = _clean(m.group(1))
            if name:
                return Attribution(name, role, i)
    return Attribution(None, role, None)


def is_dropback(play_type: str | None) -> bool:
    """Sacks are dropbacks. This is the denominator for QB rate stats."""
    return play_type in DROPBACK_TYPES
