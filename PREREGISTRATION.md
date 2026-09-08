# Phase 2 pre-registration

Written 2026-09-07, before any Phase 2 specification touched a betting line.

Phase 1 returned a null on coach–quarterback fit and bounded any effect below
0.95pp of cover probability against a 2.38pp break-even. Serial hypothesis
testing after a null converts a pre-registered study into a specification
search, so this document fixes what may be claimed before anything is run.

## Scope

**N = 1 primary hypothesis.** α = 0.05, two-sided, 80% power. This is not the
N = 2 the Phase 2 brief anticipated. The second hypothesis — defensive
continuity mispricing — was dropped before registration because it is not
testable on this source (below). Filling the slot to preserve N = 2 would have
cost a Bonferroni split against the one hypothesis that remained, buying
nothing.

Anything not registered here is exploratory, is labelled as such wherever it
appears, and cannot support a claim.

## H2 — defensive continuity. Dropped, not tested.

The specification requires coordinator quality θᴰ_c. CFBD exposes head coaches
only. Verified by direct probe 2026-09-07: `/coordinators`,
`/coaches/coordinators`, `/staff`, `/coaching/staff` and `/teams/staff` all
return 404, and no OpenAPI document is served, so endpoint discovery is by
probe. The `/coaches` record carries no position, role or title field, and
neither does `/coaches/tenures`. The `spOffense`/`spDefense` figures on a coach
season are team efficiency under that head coach, not a coordinator attribute.

Not testable as specified. Reviving it needs a new data layer, which under the
brief's own §7 requires a written cost and coverage assessment first.

## H1 — pace under-adjustment in totals. Registered, then NOT RUN.

**Claim.** The market's implied weight on predictable, coach-attributable pace
is below its predictive weight, so totals under-adjust.

**Why totals rather than sides.** Under T = D(π_ij + π_ji) and
M = D(π_ij − π_ji), a pace error enters the total at π_ij + π_ji and the margin
at π_ij − π_ji. The ratio is total/|spread|: median 6.1 empirically, p25 3.3,
p75 13.1, unbounded at a pick'em. In cover probability, 7.53 pp per drive on
totals against 1.29 pp per drive on sides.

**Axis.** Neutral-script, opponent-adjusted seconds per play, from `/drives`.
Chosen over the two alternatives on the Phase 1 rule that a coach is classified
by choices and never by output: plays per drive is an efficiency outcome, and
drives per game is 70% game-level variance with a persistent conference
schedule component that would read as a school effect. Seconds per play is the
sideline decision. Sourced from `/drives` because the `clock` on `/plays` is
frozen at drive start for 35–41% of 2015–19 drives, a defect that shrinks
monotonically across the panel and would enter a variance decomposition as a
year effect.

**Predictor.** u = the shrunk, decayed, as-of coach pace estimate — the same
estimator as Phase 1's d_c, walk-forward, prior seasons only — residualised
against the team's own lagged pace, which the market plainly already holds.

**Decision rule, fixed in advance.** Run only if the minimum detectable effect
at 80% power is below the 2.38pp break-even. Estimate by the two-stage weight
comparison, SEs by wild cluster bootstrap rather than CRVE.

### Result of the power calculation: DO NOT RUN

Measured by `scripts/phase2_power.py`. No outcome variable is read anywhere in
that script — not a total, not a spread, not a residual — so this decision was
fixable without seeing the answer.

The axis is sound and the signal is real:

| quantity | value |
|---|---|
| split-half reliability of the pace axis | **0.927** (Phase 1's axis: 0.886) |
| correlation with Phase 1's early-down pass rate | −0.433 (18.8% shared) |
| coach estimate variance surviving orthogonalisation | 28.1% |
| does u still predict realised pace? | **β = +0.664, se 0.154, p < 0.0001** |
| incremental R² over lagged own pace | +0.029 |

So a coach's pace tendency does travel and does carry information the team's own
history does not. What kills the hypothesis is the size of what that buys, with
every link measured rather than assumed:

```
1 SD of u -> 0.165 SD of realised pace -> 0.0669 drives/game
          -> 0.206 points of total -> 0.50 pp of cover probability
                                       against 2.38 pp needed
```

The weak link is pace → drives: **−0.406 drives per game per SD of pace**, not
the −1.51 an SD-for-SD conversion would have given. Checked three ways —
cross-section −0.406, team fixed effects −0.399, first differences −0.352. Game
clock is fixed, so a slower snap is absorbed mostly by plays per drive rather
than by the drive count. The identity T = D(π_ij + π_ji) is right; the
assumption that a coach's pace preference moves D much is not.

Two independent grounds for not running it:

1. **The ceiling is below break-even.** 0.50pp at one SD, *assuming the market
   prices none of a signal built from public coaching history*. At game level
   with both teams' signals added, 34 of 2,472 games (1.38%) clear 2.38pp under
   that assumption; if the market prices half, **zero** do.
2. **The test cannot distinguish the ceiling from zero.** On the 2,472 games
   where both teams have a usable predictor, the MDE is 2.569 points per unit of
   u against a mechanical ceiling of 0.828 — a ratio of **3.10**. No outcome
   would change a decision.

Registered and not run, recorded here rather than dropped, because a decision
not to run is exactly what a file drawer would swallow.

## Layer 4 allocator

Proceeds regardless, per the brief. Validated against a synthetic edge source
with known ground truth, so correctness does not depend on a real edge existing.

## Reproduction

```bash
python scripts/backfill.py --dry-run    # prices /drives at 168 calls
python scripts/build_pace.py            # the axis, and its construct check
python scripts/phase2_power.py          # the power calculation and the verdict
```
