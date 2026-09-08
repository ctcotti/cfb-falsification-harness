"""Joint posterior predictive sampling for the allocator.

The allocator maximises sample-average log wealth, so everything it knows about
dependence between bets comes from these scenarios. Two samplers:

`synthetic_returns` -- bets with SPECIFIED true win probabilities and an
equicorrelated Gaussian copula. Nothing about football in it. It exists so the
optimiser can be checked against a ground truth that is known by construction,
which is the only way to demonstrate correctness without depending on a real
edge existing.

`game_scenarios` -- the realistic structure. Team strengths are drawn once per
scenario and shared by every game that team plays, so two bets involving the
same team are correlated in the sample without anyone specifying a covariance.
That is the point of sample-average approximation here: a coach's twelve games
are not twelve independent draws, and a closed-form Kelly would need the
correlation matrix written down by hand.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

PUSH = 0.0


def payoff(win: np.ndarray, profit: float | np.ndarray,
           push: np.ndarray | None = None) -> np.ndarray:
    """Per-unit return: +profit on a win, -1 on a loss, 0 on a push."""
    r = np.where(win, profit, -1.0)
    if push is not None:
        r = np.where(push, PUSH, r)
    return r


def synthetic_returns(p_win: np.ndarray, profit: float = 100 / 110,
                      rho: float = 0.0, n_scenarios: int = 20_000,
                      seed: int = 0) -> np.ndarray:
    """(n_scenarios, n_bets) returns for bets of KNOWN win probability.

    Dependence is an equicorrelated Gaussian copula: a single common factor
    plus idiosyncratic noise, thresholded at the marginal probability. The
    marginals are exactly `p_win` whatever rho is, which is what makes this
    usable as ground truth -- changing the correlation does not smuggle in a
    change to the edges.
    """
    p = np.asarray(p_win, dtype=float)
    rng = np.random.default_rng(seed)
    n_b = p.size
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    common = rng.standard_normal((n_scenarios, 1))
    idio = rng.standard_normal((n_scenarios, n_b))
    z = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio
    win = z < stats.norm.ppf(p)[None, :]
    return payoff(win, profit)


def game_scenarios(home: np.ndarray, away: np.ndarray, line: np.ndarray,
                   true_edge_pts: np.ndarray | float = 0.0,
                   n_teams: int | None = None,
                   sigma_team: float = 8.0, sigma_game: float = 15.51,
                   profit: float = 100 / 110, n_scenarios: int = 20_000,
                   seed: int = 0) -> np.ndarray:
    """Returns for bets on the HOME side of each game, correlated by team.

    home, away     : integer team indices, one entry per game
    line           : the market's implied margin for the home side; the bet wins
                     when the realised margin exceeds it
    true_edge_pts  : how many points the market is wrong by, per game. This is
                     the synthetic signal -- set it to zero and the bets are
                     exactly fair minus the vig.

    Team strengths are redrawn per scenario and shared across that team's games.
    """
    home = np.asarray(home, int)
    away = np.asarray(away, int)
    line = np.asarray(line, float)
    edge = np.broadcast_to(np.asarray(true_edge_pts, float), home.shape)
    n_t = int(n_teams if n_teams is not None else max(home.max(), away.max()) + 1)
    rng = np.random.default_rng(seed)

    theta = rng.normal(0.0, sigma_team, size=(n_scenarios, n_t))
    mu = theta[:, home] - theta[:, away] + edge[None, :] + line[None, :]
    margin = mu + rng.normal(0.0, sigma_game, size=mu.shape)
    # Half-point lines never push; integer lines do. Model the push explicitly
    # rather than pretending it away -- it is 3.9% of totals and it matters to a
    # log-wealth calculation, which is sensitive near zero.
    margin = np.round(margin)
    win = margin > line[None, :]
    push = margin == line[None, :]
    return payoff(win, profit, push)


def implied_edge(p_win: np.ndarray, profit: float = 100 / 110) -> np.ndarray:
    """Expected per-unit return of each bet. Positive means +EV."""
    p = np.asarray(p_win, float)
    return p * profit - (1.0 - p)


def break_even(profit: float = 100 / 110) -> float:
    """Win probability at which a bet is exactly fair."""
    return 1.0 / (1.0 + profit)
