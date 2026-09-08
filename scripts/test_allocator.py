"""Validate the Layer 4 allocator against ground truth it cannot fake.

The point of this file is that the allocator's correctness does NOT depend on a
real edge existing. Phase 2 found no tradeable signal; the allocator still has
to be right, and "right" has to mean something checkable. Eight checks, each
with an answer known independently of the optimiser:

  1  single bet          against the closed-form Kelly fraction
  2  two independent     against a brute-force grid search
  3  KKT certificate     sufficient for optimality, since the problem is concave
  4  correlation         must reduce total stake at fixed marginals
  5  caps                must bind exactly, and reallocate the remainder
  6  no edge             a book of -110 bets with no edge must get zero
  7  growth race         must beat flat and edge-proportional staking OUT OF SAMPLE
  8  James-Stein         must beat raw estimates when edges are noisy

    python scripts/test_allocator.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfb.allocator import (  # noqa: E402
    DEFAULT_FRACTION, ODDS_M110, james_stein, kkt_residual, log_growth,
    optimal_fractions,
)
from cfb.scenarios import (  # noqa: E402
    break_even, game_scenarios, implied_edge, synthetic_returns,
)

PASS, FAIL = "PASS", "**FAIL**"
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    _results.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name}\n         {detail}")


def kelly_closed_form(p: float, b: float = ODDS_M110) -> float:
    """f* = (p(1+b) - 1) / b, the textbook single-bet answer."""
    return max(0.0, (p * (1 + b) - 1) / b)


def test_single_bet() -> None:
    print("\n1. SINGLE BET vs the closed-form Kelly fraction")
    for p in (0.55, 0.60, 0.65):
        R = synthetic_returns(np.array([p]), n_scenarios=400_000, seed=1)
        f = optimal_fractions(R)[0]
        want = kelly_closed_form(p)
        # Monte Carlo error in p is ~1/sqrt(N); allow for it rather than
        # pretending the sample is the distribution.
        ok = abs(f - want) < 0.01
        check(f"p={p:.2f}", ok,
              f"solver {f:.4f}  closed form {want:.4f}  diff {f-want:+.4f}")


def test_two_independent() -> None:
    print("\n2. TWO INDEPENDENT BETS vs brute-force grid search")
    p = np.array([0.58, 0.54])
    R = synthetic_returns(p, rho=0.0, n_scenarios=200_000, seed=2)
    f = optimal_fractions(R)
    grid = np.linspace(0, 0.35, 176)
    best, bf = -np.inf, None
    for a in grid:
        for b in grid:
            if a + b >= 1.0:
                continue
            g = log_growth(np.array([a, b]), R)
            if g > best:
                best, bf = g, (a, b)
    ok = abs(f[0] - bf[0]) < 0.006 and abs(f[1] - bf[1]) < 0.006
    check("2-bet optimum", ok,
          f"solver ({f[0]:.4f}, {f[1]:.4f})  grid ({bf[0]:.4f}, {bf[1]:.4f})  "
          f"growth {log_growth(f, R):.6f} vs {best:.6f}")


def test_kkt() -> None:
    print("\n3. KKT CERTIFICATE on a 20-bet correlated book")
    rng = np.random.default_rng(3)
    p = 0.5 + rng.uniform(0.0, 0.06, 20)
    R = synthetic_returns(p, rho=0.3, n_scenarios=40_000, seed=3)
    caps = np.full(20, 0.05)
    f = optimal_fractions(R, caps=caps)
    r = kkt_residual(f, R, caps=caps)
    check("KKT residual ~ 0", r < 1e-4,
          f"max violation {r:.2e}  (sufficient for optimality: the objective "
          f"is concave)  sum(f)={f.sum():.4f}")
    # A certificate that never fails certifies nothing. Perturb off the optimum
    # and it must reject, and the perturbed point must have lower growth.
    bad = f.copy()
    bad[0] = min(caps[0], bad[0] + 0.01)
    bad[1] = max(0.0, bad[1] - 0.01)
    r_bad = kkt_residual(bad, R, caps=caps)
    check("certificate REJECTS a perturbed allocation",
          r_bad > 100 * max(r, 1e-9) and log_growth(bad, R) < log_growth(f, R),
          f"perturbed violation {r_bad:.2e} vs optimal {r:.2e}; "
          f"growth {log_growth(bad, R):.7f} < {log_growth(f, R):.7f}")


def test_correlation_reduces_stake() -> None:
    print("\n4. CORRELATION must reduce total stake at fixed marginals")
    p = np.full(10, 0.57)
    totals = []
    for rho in (0.0, 0.3, 0.6, 0.9):
        R = synthetic_returns(p, rho=rho, n_scenarios=60_000, seed=4)
        totals.append(optimal_fractions(R).sum())
    ok = all(totals[i] > totals[i + 1] - 1e-9 for i in range(len(totals) - 1))
    check("monotone decreasing in rho", ok,
          "  ".join(f"rho={r}: {t:.4f}"
                    for r, t in zip((0.0, 0.3, 0.6, 0.9), totals)))


def test_caps() -> None:
    print("\n5. CAPS must bind exactly and the remainder reallocate")
    p = np.array([0.62, 0.56, 0.53])
    R = synthetic_returns(p, n_scenarios=120_000, seed=5)
    free = optimal_fractions(R)
    capped = optimal_fractions(R, caps=np.array([0.02, 1.0, 1.0]))
    ok = (abs(capped[0] - 0.02) < 1e-6 and capped[0] < free[0]
          and capped[1] >= free[1] - 1e-6)
    check("cap binds, others take up slack", ok,
          f"uncapped {np.round(free, 4)}  capped {np.round(capped, 4)}")


def test_no_edge() -> None:
    print("\n6. NO EDGE must produce no bet")
    p = np.full(8, break_even())
    R = synthetic_returns(p, rho=0.2, n_scenarios=200_000, seed=6)
    f = optimal_fractions(R)
    check("fair book gets ~zero stake", f.sum() < 0.01,
          f"break-even p={break_even():.4f}  sum(f)={f.sum():.5f}  "
          f"max f_i={f.max():.5f}")


def test_growth_race() -> None:
    print("\n7. OUT-OF-SAMPLE GROWTH vs naive staking rules")
    rng = np.random.default_rng(7)
    n_games, n_teams = 60, 24
    home = rng.integers(0, n_teams, n_games)
    away = (home + 1 + rng.integers(0, n_teams - 1, n_games)) % n_teams
    line = rng.normal(0, 7, n_games).round()
    edge = rng.normal(1.2, 0.8, n_games)          # the synthetic signal, in points

    fit = game_scenarios(home, away, line, edge, n_teams=n_teams,
                         n_scenarios=30_000, seed=70)
    hold = game_scenarios(home, away, line, edge, n_teams=n_teams,
                          n_scenarios=30_000, seed=71)

    f_kelly = optimal_fractions(fit, caps=0.05)
    ev = fit.mean(axis=0)
    f_flat = np.where(ev > 0, f_kelly.sum() / max(1, (ev > 0).sum()), 0.0)
    f_flat = np.minimum(f_flat, 0.05)
    prop = np.clip(ev, 0, None)
    f_prop = prop / prop.sum() * f_kelly.sum() if prop.sum() > 0 else prop
    f_prop = np.minimum(f_prop, 0.05)

    g = {k: log_growth(v, hold) for k, v in
         (("log-optimal", f_kelly), ("flat", f_flat),
          ("edge-proportional", f_prop))}
    ok = g["log-optimal"] >= max(g["flat"], g["edge-proportional"]) - 1e-9
    check("log-optimal wins out of sample", ok,
          "  ".join(f"{k}: {v:.5f}" for k, v in g.items())
          + f"   (staked {f_kelly.sum():.3f} over {int((f_kelly > 1e-4).sum())}"
            f" of {n_games} games)")

    frac = optimal_fractions(fit, caps=0.05, fraction=DEFAULT_FRACTION)
    wealth = 1.0 + hold @ frac
    full = 1.0 + hold @ f_kelly
    check(f"fractional lambda={DEFAULT_FRACTION} trades growth for drawdown",
          np.percentile(wealth, 1) > np.percentile(full, 1),
          f"growth {log_growth(frac, hold):.5f} vs {g['log-optimal']:.5f}; "
          f"1st-pct wealth {np.percentile(wealth, 1):.4f} vs "
          f"{np.percentile(full, 1):.4f}")


def test_james_stein() -> None:
    print("\n8. JAMES-STEIN shrinkage vs raw estimates, edges measured with noise")
    rng = np.random.default_rng(8)
    n_bets, n_trials = 25, 40
    raw_g, js_g = [], []
    for t in range(n_trials):
        true_p = 0.5 + rng.normal(0.0, 0.012, n_bets)
        se = 0.02
        obs_p = true_p + rng.normal(0.0, se, n_bets)
        truth = synthetic_returns(true_p, rho=0.25, n_scenarios=8_000,
                                  seed=1000 + t)
        # What the allocator believes, raw and shrunk.
        belief_raw = synthetic_returns(np.clip(obs_p, .01, .99), rho=0.25,
                                       n_scenarios=8_000, seed=2000 + t)
        js_p = np.clip(james_stein(obs_p, se), .01, .99)
        belief_js = synthetic_returns(js_p, rho=0.25, n_scenarios=8_000,
                                      seed=2000 + t)
        raw_g.append(log_growth(optimal_fractions(belief_raw, caps=0.05), truth))
        js_g.append(log_growth(optimal_fractions(belief_js, caps=0.05), truth))
    raw_m, js_m = float(np.mean(raw_g)), float(np.mean(js_g))
    check("shrunk beats raw on true-model growth", js_m > raw_m,
          f"raw {raw_m:.6f}   James-Stein {js_m:.6f}   "
          f"improvement {js_m - raw_m:+.6f} over {n_trials} trials")


def main() -> None:
    print("Layer 4 allocator -- validation against known ground truth")
    print("=" * 66)
    test_single_bet()
    test_two_independent()
    test_kkt()
    test_correlation_reduces_stake()
    test_caps()
    test_no_edge()
    test_growth_race()
    test_james_stein()
    n_ok = sum(ok for _, ok, _ in _results)
    print("\n" + "=" * 66)
    print(f"{n_ok}/{len(_results)} checks passed")
    if n_ok != len(_results):
        print("FAILURES:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  {name}: {detail}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
