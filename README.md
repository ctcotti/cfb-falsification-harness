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
interval **excludes any effect larger than ~0.88pp of cover probability**,
against a 2.38pp break-even at −110. Nothing in the tail either: |distance| >
p90 gives +0.149 [−0.53, +0.83].

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
An effect below ~0.9pp of cover probability remains entirely consistent with
these data.
