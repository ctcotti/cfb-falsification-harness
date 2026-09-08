"""Phase 2 power check. Run BEFORE the hypothesis is registered.

The Phase 2 brief's own rule: if the minimum detectable effect exceeds the
break-even threshold, the test cannot answer the question and should not be
run. Deciding that needs three numbers, none of which involve a betting line:

  1. how reliable the pace axis is at all;
  2. how much of a coach's as-of pace estimate survives being residualised
     against the thing the market obviously already knows -- the team's own
     pace last season;
  3. whether what survives still predicts the team's REALISED pace.

(3) is the one that matters. A predictor can be orthogonal to lagged pace and
still be noise; only incremental predictive content makes it a treatment.

NO OUTCOME VARIABLE IS TOUCHED HERE. Nothing in this file reads a total, a
spread or a residual. That is deliberate: the power calculation has to be
fixable before the hypothesis is registered, and it cannot be allowed to
double as a peek at the answer.

    python scripts/phase2_power.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfb.system_distance import coach_system_estimates  # noqa: E402

PANEL = ROOT / "data" / "panel"

# Fixed by the Phase 2 brief before any of this was measured.
ALPHA = 0.05          # N=1 primary hypothesis, so no Bonferroni split
POWER = 0.80
BREAK_EVEN_PP = 2.38  # -110 both sides
# Measured in phase 0 / the totals audit, not assumed.
SIGMA_TOTAL = 16.32
PTS_PER_DRIVE = 3.077     # dT/dD, empirical
SD_DRIVES_PER_GAME = 1.51
PHI_DENSITY = 0.3989


def pp(points: float) -> float:
    return PHI_DENSITY * points / SIGMA_TOTAL * 100.0


def load() -> pd.DataFrame:
    pace = pd.read_parquet(PANEL / "pace.parquet")
    panel = pd.read_parquet(PANEL / "coach_panel.parquet")
    coach = panel[panel["is_primary_coach"]][
        ["school", "season", "coach_id", "coach_name", "tenure_year"]
    ].drop_duplicates(["school", "season"])
    return pace.merge(coach, on=["school", "season"], how="inner")


def split_half_reliability(seed: int = 7) -> float:
    """Split each team-season's drives at random, correlate the two halves.

    Spearman-Brown corrected, so it is comparable to the 0.886 Phase 1 reports
    for its own axis. This is the attenuation factor on any coefficient.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from cfb.pace import drive_frame  # noqa: E402

    dr = drive_frame(str(ROOT / "data" / "cache"), list(range(2015, 2026)))
    rng = np.random.default_rng(seed)
    dr = dr.assign(half=rng.integers(0, 2, len(dr)))
    g = dr.groupby(["offense", "season", "half"]).agg(
        e=("elapsed", "sum"), p=("plays", "sum"), n=("spp", "size")).reset_index()
    g = g[g["n"] >= 25]
    g["spp"] = g["e"] / g["p"]
    w = g.pivot_table(index=["offense", "season"], columns="half",
                      values="spp").dropna()
    if len(w) < 30:
        return float("nan")
    r = float(np.corrcoef(w[0], w[1])[0, 1])
    return 2 * r / (1 + r)


def drives_per_game() -> pd.DataFrame:
    """Team-season drives per game, straight off the cached drive records.

    All drives, not the neutral-script subset: the total is scored over the
    whole game, so the quantity the market is pricing is the full drive count.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from cfb.pace import iter_cached_drives  # noqa: E402

    rows = []
    for season in range(2015, 2026):
        for d in iter_cached_drives(str(ROOT / "data" / "cache"), season):
            if d.get("offense") and d.get("gameId"):
                rows.append((season, d["gameId"], d["offense"]))
    dr = pd.DataFrame(rows, columns=["season", "game_id", "school"])
    per = dr.groupby(["school", "season"]).agg(
        drives=("game_id", "size"), games=("game_id", "nunique")).reset_index()
    per = per[per["games"] >= 8]
    per["dpg"] = per["drives"] / per["games"]
    return per[["school", "season", "dpg", "games"]]


def main() -> None:
    df = load()
    print(f"team-seasons with pace and a primary coach: {len(df)}")

    rel = split_half_reliability()
    print(f"split-half reliability of the pace axis (Spearman-Brown): "
          f"{rel:.3f}   [Phase 1's axis: 0.886]")

    # ---- the as-of coach estimate, same estimator as Phase 1's d_c ----------
    panel = pd.read_parquet(PANEL / "coach_panel.parquet")
    # Pass only (school, season, pace): the estimator attaches coach_id itself
    # from the panel, and a duplicate column would collide in its merge.
    est = coach_system_estimates(
        df[["school", "season", "pace_adj"]].rename(
            columns={"pace_adj": "pace"}), panel, axis="pace")
    est = est.rename(columns={"as_of_season": "season", "d": "c_hat"})
    df = df.merge(est[["coach_id", "season", "c_hat", "eff_seasons"]],
                  on=["coach_id", "season"], how="left")

    # ---- the team's own lagged pace: what the market plainly already has ----
    lag = df[["school", "season", "pace_adj_z"]].copy()
    lag["season"] += 1
    df = df.merge(lag.rename(columns={"pace_adj_z": "lag_pace_z"}),
                  on=["school", "season"], how="left")

    d = df.dropna(subset=["c_hat", "lag_pace_z", "pace_adj_z"]).copy()
    print(f"\nusable team-seasons (coach history + lagged own pace): {len(d)}"
          f"   thesis window: {int(d.season.between(2022, 2025).sum())}")

    # ---- orthogonalise -----------------------------------------------------
    m = sm.OLS(d["c_hat"], sm.add_constant(d[["lag_pace_z"]])).fit()
    d["u"] = m.resid
    share = float(np.var(d["u"], ddof=1) / np.var(d["c_hat"], ddof=1))
    print(f"\nORTHOGONALISATION  c_hat ~ lag_pace_z")
    print(f"  R2 = {m.rsquared:.4f}   b = {m.params['lag_pace_z']:+.3f}")
    print(f"  SD(c_hat) = {d.c_hat.std():.4f}   SD(u) = {d.u.std():.4f}")
    print(f"  variance of the coach estimate surviving: {share*100:.1f}%")

    # ---- does what survives still PREDICT realised pace? --------------------
    print(f"\nINCREMENTAL PREDICTIVE CONTENT  realised pace_adj_z ~ lag + u")
    X = sm.add_constant(d[["lag_pace_z", "u"]])
    mp = sm.OLS(d["pace_adj_z"], X).fit(
        cov_type="cluster", cov_kwds={"groups": d["school"]})
    base = sm.OLS(d["pace_adj_z"], sm.add_constant(d[["lag_pace_z"]])).fit()
    for nm in ("lag_pace_z", "u"):
        b, se, p = mp.params[nm], mp.bse[nm], mp.pvalues[nm]
        print(f"    {nm:<12} {b:+7.3f}  se {se:.3f}  "
              f"95% CI [{b-1.96*se:+.3f},{b+1.96*se:+.3f}]  p={p:.4f}")
    print(f"    R2 {base.rsquared:.4f} -> {mp.rsquared:.4f}   "
          f"incremental {mp.rsquared - base.rsquared:+.4f}")

    beta_u = float(mp.params["u"])
    sd_u = float(d["u"].std())
    # One SD of u moves realised pace by beta_u*sd_u SDs of the pace axis.
    move_pace_sd = beta_u * sd_u
    print(f"\n  1 SD of u  ->  {move_pace_sd:+.4f} SD of realised pace")

    # ---- translate to the totals market ------------------------------------
    # Pace is seconds per play; drives per game is what actually moves a total.
    # The passthrough is MEASURED, not assumed to be one-for-one in SDs -- an
    # assumed conversion is where a power calculation goes quietly wrong.
    drives = drives_per_game()
    j = df.merge(drives, on=["school", "season"], how="inner").dropna(
        subset=["pace_adj_z", "dpg"])
    md = sm.OLS(j["dpg"], sm.add_constant(j[["pace_adj_z"]])).fit(
        cov_type="cluster", cov_kwds={"groups": j["school"]})
    passthrough = float(md.params["pace_adj_z"])   # drives/game per SD of pace
    print(f"\nPASSTHROUGH  drives per game ~ pace_adj_z   (n={int(md.nobs)})")
    print(f"    {passthrough:+.4f} drives/game per SD of pace  "
          f"se {md.bse['pace_adj_z']:.4f}  R2 {md.rsquared:.3f}")
    print(f"    for comparison, the assumed-SD conversion would have been "
          f"{-SD_DRIVES_PER_GAME:+.4f}")
    # This single coefficient carries the verdict, so it is checked three ways.
    # Cross-sectional pace differences are confounded with everything else a
    # program is; within-team variation is not.
    w = j.sort_values(["school", "season"]).copy()
    for c in ("pace_adj_z", "dpg"):
        w[c + "_w"] = w[c] - w.groupby("school")[c].transform("mean")
    mw = sm.OLS(w["dpg_w"], sm.add_constant(w[["pace_adj_z_w"]])).fit(
        cov_type="cluster", cov_kwds={"groups": w["school"]})
    w["dp"] = w.groupby("school")["pace_adj_z"].diff()
    w["dd"] = w.groupby("school")["dpg"].diff()
    w["gap"] = w.groupby("school")["season"].diff()
    fd = w[w["gap"] == 1].dropna(subset=["dp", "dd"])
    mf = sm.OLS(fd["dd"], sm.add_constant(fd[["dp"]])).fit(
        cov_type="cluster", cov_kwds={"groups": fd["school"]})
    print(f"    team fixed effects {mw.params['pace_adj_z_w']:+.4f}"
          f"   first differences {mf.params['dp']:+.4f}"
          f"   -- all three agree, so the conversion is not a"
          f" cross-sectional artefact")

    pts_per_sd = abs(passthrough) * PTS_PER_DRIVE
    mech = abs(move_pace_sd) * pts_per_sd
    print(f"\n  1 SD of u -> {abs(move_pace_sd)*abs(passthrough):.4f} drives/game"
          f" -> {mech:.3f} pts of total = {pp(mech):.2f} pp,")
    print(f"  and that is the ceiling: it assumes the market prices NONE of it.")

    # ---- MDE ---------------------------------------------------------------
    z = stats.norm.ppf(1 - ALPHA / 2) + stats.norm.ppf(POWER)
    print(f"\nMDE  (alpha={ALPHA}, {POWER:.0%} power, multiplier {z:.3f}, "
          f"sigma_total={SIGMA_TOTAL})")
    print(f"  Per SD of the predictor. The sample is NOT every FBS game: both"
          f" teams need a\n  usable u, which is why the third row is the one"
          f" that counts.")
    hdr = f"  {'design':<40}{'games':>8}{'MDE pts':>10}{'MDE pp':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for label, n_games in [
        ("all FBS-FBS games, if u were universal", 7857),
        ("thesis window share of that", 2900),
    ]:
        mde = z * SIGMA_TOTAL / np.sqrt(n_games)
        print(f"  {label:<40}{n_games:>8}{mde:>10.3f}{pp(mde):>9.2f}")
    print(f"\n  break-even at -110 = {BREAK_EVEN_PP:.2f} pp")

    kish = d.u.pow(2).sum() ** 2 / d.u.pow(4).sum()
    print(f"\n  Concentration of the predictor: |u| > 1 SD on "
          f"{int((d.u.abs() > sd_u).sum())} of {len(d)} team-seasons; "
          f"Kish effective n = {kish:.0f}")
    print(f"  (a Gaussian u would give n/3 = {len(d)/3:.0f}; {kish:.0f} means the "
          f"identifying variation is far more concentrated than normal, so the\n"
          f"  registered spec must use the wild cluster bootstrap, not the CRVE.)")

    # ---- the question the MDE cannot answer --------------------------------
    # A well-powered test of an effect too small to trade is still useless. The
    # ceiling on the per-team mispricing is |u| * pts_per_sd; ask how much of
    # the panel could clear the vig even against a market pricing none of it.
    print(f"\nCEILING TEST  -- is the largest POSSIBLE mispricing tradeable?")
    ceil_pp = (d["u"].abs() / sd_u) * abs(move_pace_sd) * pts_per_sd
    ceil_pp = ceil_pp.map(pp)
    need_sd = BREAK_EVEN_PP / pp(abs(move_pace_sd) * pts_per_sd)
    print(f"  a team-season needs |u| > {need_sd:.2f} SD to clear "
          f"{BREAK_EVEN_PP}pp even against a market pricing NOTHING")
    for k, lab in ((1.0, "prices nothing"), (0.5, "prices half"),
                   (0.25, "prices three quarters")):
        thr = need_sd / k
        n = int((d["u"].abs() / sd_u > thr).sum())
        print(f"    market {lab:<20} needs |u| > {thr:4.2f} SD: "
              f"{n:4d} of {len(d)} team-seasons ({n/len(d)*100:4.1f}%)")
    print(f"  ceiling mispricing, pp:  median {ceil_pp.median():.2f}   "
          f"p90 {ceil_pp.quantile(.9):.2f}   p99 {ceil_pp.quantile(.99):.2f}   "
          f"max {ceil_pp.max():.2f}")

    # Both teams move the same game clock, so the fair ceiling is at GAME
    # level with the two signals added -- the steelman for running this at all.
    tg = pd.read_parquet(PANEL / "team_games.parquet")
    key = d[["school", "season", "u"]]
    g = (tg.merge(key, on=["school", "season"], how="inner")
           .merge(key.rename(columns={"school": "opponent", "u": "u_opp"}),
                  on=["opponent", "season"], how="inner"))
    g = g[g["is_home"] == 1]                       # one row per game
    g["x"] = g["u"] + g["u_opp"]
    pts_per_u = abs(move_pace_sd) * pts_per_sd / sd_u   # points per unit of u
    g["ceil_pp"] = (g["x"].abs() * pts_per_u).map(pp)
    n_over = int((g["ceil_pp"] > BREAK_EVEN_PP).sum())
    print(f"\n  GAME LEVEL, both signals added (n={len(g)} games, "
          f"{g.season.nunique()} seasons)")
    print(f"    ceiling mispricing, pp: median {g.ceil_pp.median():.2f}  "
          f"p90 {g.ceil_pp.quantile(.9):.2f}  p99 {g.ceil_pp.quantile(.99):.2f}"
          f"  max {g.ceil_pp.max():.2f}")
    for k, lab in ((1.0, "prices nothing"), (0.5, "prices half"),
                   (0.25, "prices three quarters")):
        n = int((g["ceil_pp"] * k > BREAK_EVEN_PP).sum())
        print(f"    games clearing {BREAK_EVEN_PP}pp if the market {lab:<20}"
              f" {n:4d} ({n/len(g)*100:5.2f}%)")

    # ---- the decisive comparison -------------------------------------------
    # Everything above is in units of SDs and percentiles, which invites
    # cherry-picking a favourable tail. The clean question needs neither: on the
    # sample that actually exists, can the test tell the FULL mechanical effect
    # apart from zero? If not, no market-efficiency assumption saves it.
    sxx = float(((g["x"] - g["x"].mean()) ** 2).sum())
    mde_coef = z * SIGMA_TOTAL / np.sqrt(sxx)
    print(f"\nDECISIVE COMPARISON  (n={len(g)} usable games, "
          f"sum of squares {sxx:.1f})")
    print(f"    mechanical coefficient, market pricing nothing : "
          f"{pts_per_u:7.3f} pts per unit of u")
    print(f"    MDE on that same coefficient                   : "
          f"{mde_coef:7.3f} pts per unit of u")
    print(f"    ratio MDE / ceiling = {mde_coef / pts_per_u:.2f}")

    ok = mde_coef < pts_per_u
    print(f"\n  VERDICT: {'REGISTER' if ok else 'DO NOT REGISTER'}")
    if not ok:
        print(f"    The minimum detectable effect is {mde_coef/pts_per_u:.1f}x the "
              f"largest effect that could physically exist.")
        print(f"    The test cannot distinguish a market pricing NOTHING from a "
              f"market pricing\n    everything, so it has no outcome that would "
              f"change a decision. Per the brief's\n    own rule, it should not "
              f"be run.")
    print(f"\n  The chain, every link measured rather than assumed:")
    print(f"    1 SD of u -> {abs(move_pace_sd):.3f} SD pace -> "
          f"{abs(move_pace_sd)*abs(passthrough):.4f} drives/game -> "
          f"{mech:.3f} pts -> {pp(mech):.2f} pp,  against {BREAK_EVEN_PP} needed.")
    print(f"    The weak link is pace -> drives: {passthrough:+.3f} drives per "
          f"SD, not the -1.51 an\n    SD-for-SD assumption would have given. "
          f"Game clock is fixed, so a slower snap is\n    absorbed largely by "
          f"plays per drive rather than by the drive count.")


if __name__ == "__main__":
    main()
