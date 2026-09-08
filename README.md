# Coaching-Fit Betting Model — falsification harness

A quantitative test of whether **coach–quarterback fit** is an unpriced edge in
FBS college football betting, and the null it returned.

The hypothesis had an attractive structure: fit is an *interaction*, betting
lines are built from additive power ratings, and an additive model has no term
for an interaction. If mismatch costs real points, that cost sits outside the
price by construction.

It does not survive contact with the data.

## Result

**Test B — does system distance predict what the market missed?**

| Window | γ̂ (pts / SD of Φ) | 95% CI | p |
|---|---|---|---|
| Thesis 2022–25 | +0.009 | [−0.33, +0.35] | 0.958 |
| Long 2015–25 | +0.057 | [−0.24, +0.36] | 0.711 |

5,022 team-games, 770 team-season clusters, SEs clustered on team-season. The
interval **excludes any effect larger than ~0.95pp of cover probability**,
against a 2.38pp break-even at −110. Nothing in the tail either: |distance| >
p90 gives +0.137 [−0.55, +0.83].

The tail specifications are thin — the top decile has 35 clusters — and the
cluster-robust variance estimator is asymptotic in *clusters*, not observations.
Simulated on that exact cluster structure it rejects a true null **9.5%** of the
time at a nominal 5%. Re-run under a wild cluster bootstrap-t with the null
imposed (Rademacher, 9,999 replications, interval by test inversion), which
rejects at **5.9%**:

| Specification | n | G | γ̂ | cluster-robust | bootstrap | bootstrap upper, pp |
|---|---|---|---|---|---|---|
| Test B, thesis | 2350 | 330 | +0.009 | [−0.33, +0.35] | [−0.39, +0.37] | 0.95 |
| Test B, long | 5022 | 770 | +0.057 | [−0.24, +0.36] | [−0.28, +0.37] | 0.94 |
| \|distance\| > p75 | 581 | 83 | +0.270 | [−0.25, +0.79] | [−0.43, +0.79] | 2.02 |
| \|distance\| > p90 | 230 | 35 | +0.137 | [−0.55, +0.83] | [−0.91, +0.76] | **1.95** |

Correction widens every interval — 1.21× at the top decile — but asymmetrically,
and to the left. The top-decile upper bound *falls*, from 2.12pp to 1.95pp, so
the exclusion claim strengthens rather than survives.

**Test A — is it already in the price?** With a thin first stage (R²=0.656) Φ
looked significant (+0.365, p=0.010); adding returning production and a
coaching-change indicator (R²=0.673) drops it to +0.192, p=0.122. So both sides
are null. This is not "the market got there first" — the interaction does not
exist at measurable magnitude.

Writeup: <https://claude.ai/code/artifact/2c64214d-545f-4c94-b2fb-b57da4887253>

## What held and what died

Three falsification tests, each with its decision rule fixed in advance, all run
before any model was built.

| # | Test | Verdict |
|---|---|---|
| 01 | Does a coach's system travel between schools? | **Held.** Variance ratio 1.26 [0.83, 1.72] against a pre-registered 2.0 threshold. Decomposition: coach 55.8%, school 9.1%, year 35.1%. |
| 02 | Is designed QB run rate a coach attribute? | **Killed.** Correlation across a job change 0.005; coach variance 0.5%. It measures the quarterback, not the system. |
| 03 | Is the QB situational profile measurable? | **Killed.** Split-half reliability **−0.008**. Same QB, same season, plays split at random, zero agreement. |

Test 02 is the methodological one worth reading: the variance-ratio rule alone
would have passed QB run share (1.62 < 2.0). The ratio compares the school
effect to year noise and is silent on whether a coach effect exists at all. Only
the near-zero correlation exposed σ²_coach ≈ 0. Both statistics are needed.

Test 03 killed the QB side of Φ entirely, collapsing the design to system
distance (adaptation cost). The bind is general: **what makes a quarterback
measure fit-relevant — no main effect — is exactly what makes it unstable.**

## Source defects found

Each is plausible-looking, populated, and wrong. None throws.

- **`/coaches.hireDate`** is the coach's *first head-coaching hire ever*, not the
  hire for the tenure in question. Moorhead/Akron 2022 reads `2017-11-28` (his
  Mississippi State hire) vs `2021-12-04` on `/coaches/tenures`. Wrong for every
  coach on a second-plus job.
- **`/coaches/tenures.startYear`** is FBS-scoped. Cignetti's James Madison
  tenure reads `startYear=2022` against a 2018 hire, because JMU played FCS
  through 2021. Labels every reclassification coach as year-one.
- **Phantom 0-game coach rows.** An outgoing coach's `endYear` runs a season
  long, so Holgorsen appears in Houston 2024 with 0 games — turning a "sole
  coach of record" filter into the exclusion of Fritz, who coached all twelve.

Also: `/passing/plays` and `/rushing/plays` (air yards, `passDepth`, structured
passer IDs) hold data for **2025 only**, advertised with no coverage caveat. And
no scramble play type exists in the 49-type vocabulary — a scramble and a
designed QB run are the same record. QB attribution is therefore parsed from
play text at **99.963%** coverage over 259,446 scrimmage plays.

## Phase 2 — a second null, and the allocator

Full detail and decision rules in [PREREGISTRATION.md](PREREGISTRATION.md),
written before any Phase 2 specification touched a betting line.

**Defensive continuity: dropped before registration.** It needs coordinator
quality. CFBD has head coaches only — `/coordinators`, `/coaches/coordinators`,
`/staff`, `/coaching/staff` and `/teams/staff` all 404, no OpenAPI document is
served, and neither `/coaches` nor `/coaches/tenures` carries a position field.

**Pace under-adjustment in totals: registered, powered, not run.** The axis is
good — neutral-script, opponent-adjusted seconds per play from `/drives`, with
split-half reliability **0.927** against Phase 1's 0.886 — and the signal is
real. A coach's as-of pace estimate keeps 28.1% of its variance after being
residualised against the team's own lagged pace, and what survives still
predicts realised pace at β = +0.664, se 0.154, p < 0.0001.

It dies on magnitude, every link measured rather than assumed:

```
1 SD of the orthogonalised coach signal
  -> 0.165 SD of realised pace
  -> 0.0669 drives per game
  -> 0.206 points of total
  -> 0.50 pp of cover probability      against 2.38 pp needed at -110
```

The weak link is pace → drives: **−0.406 drives per game per SD of pace**, not
the −1.51 an SD-for-SD conversion assumes. Cross-section −0.406, team fixed
effects −0.399, first differences −0.352. Game clock is fixed, so a slower snap
is absorbed mostly by plays per drive rather than by the drive count. The
identity T = D(π_ij + π_ji) holds; the assumption that a coach's pace preference
moves D much does not.

Two independent grounds for not running it. The ceiling is below break-even —
0.50pp at one SD *assuming the market prices none* of a signal built from public
coaching history; at game level 34 of 2,472 games clear 2.38pp under that
assumption and **zero** do if the market prices half. And on that same sample
the MDE is **3.10× the mechanical ceiling**, so no outcome would change a
decision. `scripts/phase2_power.py` reads no total, spread or residual anywhere,
so the decision was fixable without seeing the answer.

**The allocator was built anyway**, per the brief — a rigorous null plus a
working, validated optimiser is a stronger pair than a marginal positive.

    max_f E[log(1 + f'R)]   s.t.  f >= 0,  f_i <= l_i,  sum f_i <= 1

Concave under log utility, so there is a unique optimum and the solver can be
*checked* rather than trusted. Correlation is carried by the scenarios, not by a
covariance matrix: team strengths are drawn once per scenario and shared across
that team's games, so two bets on the same team are dependent in the sample
without anyone writing down a ρ. Twelve validation checks against ground truth
the optimiser cannot fake — closed-form Kelly, brute-force grid search, a KKT
certificate that is *shown to reject* a perturbed allocation, and out-of-sample
growth against flat and edge-proportional staking.

Two results from that suite worth stating on their own:

- At λ = 0.25 the allocator keeps 43% of the growth rate while lifting
  1st-percentile wealth from 0.68 to 0.92. That is the fractional-Kelly trade,
  measured.
- When edge estimates are noisier than the spread of true edges, betting on them
  **destroys** wealth: log-growth −0.0029 raw. James–Stein shrinkage cuts the
  damage to −0.0001 but does not reverse the sign. With no measurable edge the
  right allocation is zero, and the allocator finds it — check 6 puts 0.4% of
  bankroll at risk on a fair book.

## Layout

```
src/cfb/
  client.py           CFBD client: content-addressed disk cache, call ledger,
                      hard budget guard (25k against the tier's 30k ceiling)
  panel.py            Coach panel — the treatment-assignment table. Both hireDate
                      traps and both midseason detectors live here.
  playtext.py         QB attribution from play text
  tendency.py         d_c — coach play-call rates on neutral early downs
  qb.py               q_p — the situational contrast (retained; it failed test 03)
  system_distance.py  Phi, with matched estimators on both sides
scripts/
  backfill.py            prices the plan before spending (--dry-run)
  build_panel.py         coach panel + attrition table
  test_systems_travel.py test 01
  test_qp_stability.py   test 03
  build_games.py         outcomes joined to market prices
  stage3_gate.py         the gate, team-season level
  stage3_gate_game.py    the gate, game level (carries the verdict)
  phase0_wildboot.py     wild cluster bootstrap-t on the tails; --validate
                         runs the size check that justifies it
  phase0_contrast_arith.py  why the q_p contrast was arithmetically doomed
  build_pace.py          the Phase 2 axis, and its construct check
  phase2_power.py        the power calculation that refused the hypothesis
  test_allocator.py      12 checks against known ground truth
```

Phase 2 additions to `src/cfb/`:

```
  pace.py             neutral-script, opponent-adjusted seconds per play
  scenarios.py        joint posterior predictive; correlation via shared
                      team parameters, not a covariance matrix
  allocator.py        the convex program, James-Stein, fractional Kelly
```

## Running it

Needs a CFBD API key in `.env` (see `.env.example`). The $5/month tier is ample;
the full backfill is 405 calls.

```bash
python scripts/backfill.py --dry-run   # prices the plan before spending
python scripts/backfill.py             # ~405 calls, ~1.1GB cached
python scripts/build_panel.py
python scripts/test_systems_travel.py
python scripts/test_qp_stability.py
python scripts/build_games.py
python scripts/stage3_gate_game.py     # the gate
python scripts/phase0_wildboot.py      # few-cluster correction on the tails
python scripts/phase0_contrast_arith.py

python scripts/build_pace.py           # Phase 2 axis (needs /drives, 168 calls)
python scripts/phase2_power.py         # the verdict: do not run
python scripts/test_allocator.py       # allocator validation, no data needed
```

Every response is cached content-addressed on (endpoint, sorted params), so
re-runs are free and offline. Completed seasons cache permanently; the live
season carries a 12h TTL.

## Discipline

- **Walk-forward only.** Coaching hires, portal moves and recruiting classes are
  filtered by announcement date, not season label. No pooled cross-validation
  across seasons.
- **Both sides of Φ use the same estimator.** Comparing a coach's multi-season
  mean against a single school-season value put a 0.57 z-unit noise floor under
  the distance and attenuated γ by ~30%. Matched estimators drop the structural
  zeros to 0.0004 and lift split-sample reliability to 0.886.
- **Effect sizes fixed before the gate ran.** MDE in probability points is
  invariant to σ_game, which is why the reported bounds survived discovering
  that σ_game is 15.51 rather than the assumed 13.

## Limitations

This tested **adaptation cost**, not coach–QB fit — the strong version was
unbuildable, not disproven. Systems are represented by one axis (k=1). CFBD
lines carry no timestamp, so these are late prices of unknown vintage: adequate
to ask whether a signal is in the price, not adequate for closing-line value.
An effect below ~0.95pp of cover probability remains entirely consistent with
these data.
