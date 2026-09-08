"""Layer 4: the bankroll allocator.

    max_f  E[log(1 + f'R)]   s.t.  f >= 0,  f_i <= l_i,  sum_i f_i <= 1

Log utility makes this a genuine convex program with a unique optimum, so there
is a right answer and the solver can be checked against it rather than trusted.
Three properties are worth stating because they are what make the numerical
approach safe:

1. The objective is concave in f (log of an affine function, averaged), so a
   local optimum is global and KKT conditions are sufficient, not just
   necessary. test_allocator.py checks them.

2. log is its own barrier. Wealth 1 + f'R hits zero when every bet in a fully
   staked book loses, and the objective goes to -infinity there, so the
   sum(f) <= 1 constraint is never active at an interior optimum. The bankroll
   cannot be wiped out by construction, not by a side condition.

3. Correlation is handled by the SCENARIOS, not by a covariance matrix. Sampling
   from a joint posterior predictive in which games share team parameters means
   two bets on the same team are correlated in the sample, and the optimiser
   sees it without anyone having to specify rho. This is the whole reason for
   sample-average approximation over a closed-form Kelly.

Two corrections sit in front of the optimiser, both from Phase 1's Layer 3:

  James-Stein shrinkage. Estimated edges are noisy and the optimiser is a
  maximiser, so it loads onto whichever edge is most overstated -- the winner's
  curse, applied to bet selection. Shrinking toward the grand mean before
  optimising is the standard correction and it is not optional at these sample
  sizes.

  Fractional scaling. Full Kelly is growth-optimal only if the estimated
  distribution is the true one. It never is. lambda in [0.25, 0.5] gives up a
  little growth for a large reduction in drawdown, and is what anybody actually
  sizing real money uses.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize

# American -110 both sides: win 100/110 of the stake, lose all of it.
ODDS_M110 = 100.0 / 110.0
DEFAULT_FRACTION = 0.25


def american_to_decimal_profit(odds: float) -> float:
    """Profit per unit staked on a winning bet, from American odds."""
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def james_stein(edges: np.ndarray, se: np.ndarray | float) -> np.ndarray:
    """Shrink estimated edges toward their grand mean.

    The positive-part James-Stein estimator. With k >= 4 estimates of comparable
    precision it dominates the raw estimates under squared-error loss -- and the
    loss that matters here is worse than squared error, because the allocator
    actively seeks out the largest estimate.
    """
    e = np.asarray(edges, dtype=float)
    k = e.size
    if k < 4:
        return e.copy()
    s2 = float(np.mean(np.asarray(se, dtype=float) ** 2))
    mu = float(e.mean())
    ss = float(((e - mu) ** 2).sum())
    if ss <= 0:
        return np.full_like(e, mu)
    shrink = max(0.0, 1.0 - (k - 3) * s2 / ss)
    return mu + shrink * (e - mu)


def _neg_log_growth(f, R, w):
    wealth = 1.0 + R @ f
    if np.any(wealth <= 1e-12):
        return 1e9, np.zeros_like(f)
    g = -float(w @ np.log(wealth))
    grad = -(R * (w / wealth)[:, None]).sum(axis=0)
    return g, grad


def optimal_fractions(returns: np.ndarray,
                      caps: np.ndarray | float = 1.0,
                      total_cap: float = 1.0,
                      weights: np.ndarray | None = None,
                      fraction: float = 1.0) -> np.ndarray:
    """Solve the log-optimal allocation over sampled scenarios.

    returns : (n_scenarios, n_bets) per-unit return of each bet in each scenario
    caps    : per-bet ceiling l_i, scalar or vector
    fraction: lambda, applied AFTER solving. Scaling the solution is not the
              same as solving the scaled problem, and the former is what
              fractional Kelly means.
    """
    R = np.asarray(returns, dtype=float)
    if R.ndim != 2:
        raise ValueError("returns must be (n_scenarios, n_bets)")
    n_s, n_b = R.shape
    w = (np.full(n_s, 1.0 / n_s) if weights is None
         else np.asarray(weights, float) / np.sum(weights))
    cap = np.broadcast_to(np.asarray(caps, dtype=float), (n_b,)).astype(float)

    # Stay strictly inside the barrier: at sum(f) == total_cap a scenario that
    # loses every bet gives wealth 0 and an infinite objective.
    eps = 1e-6
    res = optimize.minimize(
        _neg_log_growth, x0=np.full(n_b, min(0.01, total_cap / (2 * n_b))),
        args=(R, w), jac=True, method="SLSQP",
        bounds=[(0.0, float(c)) for c in cap],
        constraints=[{"type": "ineq",
                      "fun": lambda f: total_cap - eps - f.sum(),
                      "jac": lambda f: -np.ones_like(f)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    f = np.clip(res.x, 0.0, cap)
    return fraction * f


def log_growth(f: np.ndarray, returns: np.ndarray) -> float:
    """Expected log growth per period at allocation f, on given scenarios."""
    wealth = 1.0 + np.asarray(returns, float) @ np.asarray(f, float)
    if np.any(wealth <= 0):
        return -np.inf
    return float(np.mean(np.log(wealth)))


def kkt_residual(f: np.ndarray, returns: np.ndarray,
                 caps: np.ndarray | float = 1.0,
                 total_cap: float = 1.0, tol: float = 1e-6) -> float:
    """Max violation of the KKT conditions. Zero means provably optimal.

    For a concave objective with linear constraints these are sufficient, so
    this is a certificate rather than a heuristic check.
    """
    R = np.asarray(returns, float)
    f = np.asarray(f, float)
    n_b = R.shape[1]
    cap = np.broadcast_to(np.asarray(caps, float), (n_b,)).astype(float)
    wealth = 1.0 + R @ f
    grad = (R / wealth[:, None]).mean(axis=0)          # d/df of E log wealth
    # Multiplier on the budget constraint. Complementary slackness: if the
    # budget is not binding it is exactly zero, and every interior coordinate
    # must then have zero gradient. If it binds, it is the common interior
    # gradient. Inferring it from the data in the slack case would let a wrong
    # allocation certify itself.
    interior = (f > tol) & (f < cap - tol)
    if f.sum() < total_cap - tol:
        nu = 0.0
    else:
        nu = float(np.mean(grad[interior])) if interior.any() else 0.0
    viol = 0.0
    for i in range(n_b):
        if f[i] <= tol:                     # at lower bound: gradient must not push up
            viol = max(viol, grad[i] - nu)
        elif f[i] >= cap[i] - tol:          # at upper bound: gradient must not push down
            viol = max(viol, nu - grad[i])
        else:                               # interior: gradient must equal nu
            viol = max(viol, abs(grad[i] - nu))
    return float(viol)
