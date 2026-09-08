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

## G1 — opener drift. An instrumental gate, not a tradeable claim.

Registered 2026-09-08, before the first regression was run.

**Status.** This is not a third primary hypothesis and cannot support a claim
about an edge. It is a gate on a *data purchase*: whether to pay for a
timestamped, limit-carrying odds feed. Registering it anyway, with the rule
fixed first, because the alternative — running it informally and deciding
afterwards what it meant — is the specification search §3 exists to prevent.

**Why it is not tradeable.** CFBD records `spreadOpen`/`overUnderOpen` and one
later number of unknown vintage. No odds-history endpoint exists — `/lines/
history`, `/odds`, `/odds/history`, `/lines/movement`, `/betting/lines` and
`/lines/providers` all 404, verified 2026-09-08. So a result here says whether
information exists in line movement; it cannot say whether that information was
reachable, because we do not know when the later price was observable.

**Specification.** Per game, with CFBD's home-team sign convention:

```
spread:  drift = spreadOpen - spread          (drift in implied HOME margin)
         resid = home_margin + spread
total:   drift = overUnder - overUnderOpen
         resid = actual_total - overUnder
```

Regress `resid ~ drift`. β > 0 means the number moved the right way and stopped
short — follow the move. β < 0 means it overshot. β = 0 means the recorded price
has fully absorbed its own movement, and there is no visible structure to buy a
feed for.

**Sample.** FBS–FBS regular-season games with both an opener and a later number:
3,723 spread, 3,728 total, seasons **2021–2025 only** — openers do not exist
before 2021. Median across books on both legs.

**Inference.** Wild cluster bootstrap-t with the null imposed, clustered on
season-week (~75 clusters), reusing `phase0_wildboot.py`. Cluster-robust SEs are
reported alongside but do not carry the verdict; Phase 0 measured the CRVE
over-rejecting at 9.5% against a nominal 5% on 35 clusters, and 75 is not
comfortably clear of that.

**Multiplicity.** Two legs, so α = 0.025 each (Bonferroni at N = 2).

**MDE at 80% power, α = 0.025.** Computed from drift dispersion and σ alone; no
residual was regressed on anything to obtain it.

| leg | n moved | drift SD | MDE (resid pts per drift pt) | at mean drift |
|---|---:|---:|---:|---:|
| spread | 3,225 | 2.20 | 0.382 | 1.57 pp |
| total | 3,400 | 2.23 | 0.387 | 1.70 pp |

Both below the 2.38pp break-even, so the gate can answer its question.

**Known mechanical bias, quantified in advance.** The later price appears in the
regressor with one sign and in the outcome with the other, so measurement error
in it induces a spurious negative β of roughly −Var(err)/Var(drift). With a
cross-book SD near 0.29 over 2–4 books, Var(err) ≈ 0.028 against Var(drift) ≈
4.84, giving a bias near **−0.006** — an order of magnitude under the MDE, but
reported rather than assumed away, and it means a *small negative* β is the one
result that must not be believed.

**Decision rule, fixed in advance.**

- Bootstrap CI excludes zero **and** the implied edge at mean |drift| exceeds
  2.38pp → information exists and is large; price the feed.
- CI excludes zero but the implied edge is below 2.38pp → information exists and
  is too small to trade at these stakes; do not buy on this basis.
- CI includes zero and excludes effects above 2.38pp → the recorded price
  absorbs its own movement; do not buy on this basis.
- CI includes zero and does not exclude 2.38pp → underpowered, report as such,
  claim nothing.

### Result: the price absorbs its own movement. Do not buy the feed on this basis.

Run 2026-09-08 by `scripts/phase3_drift_gate.py`. Third decision-rule branch on
both legs: CI includes zero *and* excludes any effect above break-even.

| leg | n | G | β (resid pts per drift pt) | bootstrap 95% | boot p | edge at mean drift | CI edge |
|---|---:|---:|---:|---|---:|---:|---:|
| spread, moved | 3225 | 77 | −0.058 | [−0.288, +0.164] | 0.600 | 0.24 pp | 1.19 pp |
| spread, all | 3723 | 77 | −0.057 | [−0.281, +0.165] | 0.613 | 0.20 pp | 1.00 pp |
| total, moved | 3400 | 77 | +0.072 | [−0.174, +0.315] | 0.560 | 0.32 pp | 1.38 pp |
| total, all | 3728 | 77 | +0.064 | [−0.183, +0.307] | 0.599 | 0.25 pp | 1.23 pp |

The two legs disagree in sign and neither is distinguishable from zero. This is a
bounded null, not an underpowered one: even the CI-edge effect reaches 1.38pp
against the 2.38pp needed at −110. Whatever the market learned between the
opener and the recorded price, the recorded price already contains it.

**The pre-registered bias estimate was wrong, by a factor of thirteen.** It said
the artefact would be about −0.006. Measured, it is **−0.078 for the spread leg**
and −0.032 for the total. The error was mine and it was elementary: I estimated
Var(error) from the *median* cross-book SD of 0.29, when the variance is governed
by the mean of squared dispersion and cross-book dispersion is heavy-tailed —
p90 SD is 1.32, and single games run to spreads of 8.5 against 12.5.

This matters more than a footnote, because for the spread leg **the artefact
alone is larger than the observed coefficient**. Adjusting to first order:

| leg | raw β | artefact | adjusted β | edge at mean drift |
|---|---:|---:|---:|---:|
| spread | −0.058 | −0.078 | **+0.020** | 0.08 pp |
| total | +0.072 | −0.032 | **+0.104** | 0.46 pp |

So the one apparently-negative result is fully explained by measurement error,
and the adjustment moves both legs to small positive coefficients that remain far
inside their intervals. The adjustment is approximate — it ignores the
attenuation from error in the opener, which pushes the true coefficient toward
zero as well. Every version of this is a null.

**What this does and does not license.** It says the recorded price has absorbed
its own movement, on 3,700 games across 2021–25, well enough to bound any
follow-the-move effect below break-even. It does *not* say line movement is
uninformative in general: the vintage of the later price is unknown, so a feed
with real timestamps could still reveal intraday structure this cannot see. It
removes the cheapest reason to buy one, not every reason.

## Layer 4 allocator

Proceeds regardless, per the brief. Validated against a synthetic edge source
with known ground truth, so correctness does not depend on a real edge existing.

## Reproduction

```bash
python scripts/backfill.py --dry-run    # prices /drives at 168 calls
python scripts/build_pace.py            # the axis, and its construct check
python scripts/phase2_power.py          # the power calculation and the verdict
```
