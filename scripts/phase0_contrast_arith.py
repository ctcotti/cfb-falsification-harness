"""Phase 0b: was the contrast's failure forced, or is the bind general?

The writeup claims something stronger than the evidence supports: that what
makes a quarterback measure fit-relevant is what makes it unstable. Test 03
showed the situational contrast has no signal. It did not show that no contrast
could have had any, and the difference matters -- one is a result, the other is
a theorem nobody proved.

The classical difference-score reliability (Cronbach & Furby 1970) is

    rho_D = (rho_11 s1^2 + rho_22 s2^2 - 2 rho_12 s1 s2)
            / (s1^2 + s2^2 - 2 rho_12 s1 s2)

so a difference of two moderately reliable, positively correlated components is
arithmetically doomed regardless of what the components measure. If the observed
component reliabilities and their observed intercorrelation already predict the
observed contrast reliability, the failure is a property of the DIFFERENCE
OPERATOR, not of fit-relevance -- and the remedies left untried (hierarchical
shrinkage of the contrast, multi-season pooling, latent-trait treatment with
known measurement error) were never ruled out.

rho_12 is measured here, not assumed. Everything else is read off Test 03.

    python scripts/phase0_contrast_arith.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from test_qp_stability import MIN_ED, MIN_OPD  # noqa: E402


def yoy(d: pd.DataFrame, col: str) -> tuple[float, int]:
    """Year-over-year correlation, season-standardised, same passer."""
    z = col + "_z"
    d = d.copy()
    d[z] = d.groupby("season")[col].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))
    nxt = d[["passer", "season", z]].copy()
    nxt["season"] -= 1
    nxt = nxt.rename(columns={z: "next"})
    j = d.merge(nxt, on=["passer", "season"], how="inner")
    return float(np.corrcoef(j[z], j["next"])[0, 1]), len(j)


def rho_diff(r11: float, r22: float, r12: float,
             s1: float, s2: float) -> float:
    num = r11 * s1**2 + r22 * s2**2 - 2 * r12 * s1 * s2
    den = s1**2 + s2**2 - 2 * r12 * s1 * s2
    return num / den


def main() -> None:
    df = pd.read_parquet(ROOT / "data" / "panel" / "qb_profiles_raw.parquet")
    d = df[(df.opd_n >= MIN_OPD) & (df.ed_n >= MIN_ED)].copy()
    print(f"QB-seasons meeting opd_n>={MIN_OPD}, ed_n>={MIN_ED}: {len(d)}\n")

    # Season-demean so league drift is not counted, but keep the raw scale --
    # the two bins have genuinely different variances and the general form of
    # the formula needs them.
    for c in ("ed_epa", "opd_epa", "contrast_raw"):
        d[c + "_c"] = d[c] - d.groupby("season")[c].transform("mean")

    s1 = float(d["opd_epa_c"].std(ddof=1))   # X1: passing-down bin
    s2 = float(d["ed_epa_c"].std(ddof=1))    # X2: early-down bin
    r12 = float(np.corrcoef(d["opd_epa_c"], d["ed_epa_c"])[0, 1])

    r_opd, n_opd = yoy(d, "opd_epa")
    r_ed, n_ed = yoy(d, "ed_epa")
    r_obs, n_obs = yoy(d, "contrast_raw")

    print("measured inputs")
    print(f"    SD passing-down EPA          s1   = {s1:.4f}")
    print(f"    SD early-down EPA            s2   = {s2:.4f}")
    print(f"    intercorrelation, same season r12 = {r12:+.4f}   "
          f"(n={len(d)})")
    print(f"    reliability, passing-down    r11  = {r_opd:+.4f}   "
          f"(YoY, n={n_opd})")
    print(f"    reliability, early-down      r22  = {r_ed:+.4f}   "
          f"(YoY, n={n_ed})\n")

    pred = rho_diff(r_opd, r_ed, r12, s1, s2)
    pred_eq = (r_opd + r_ed - 2 * r12) / (2 - 2 * r12)
    print("prediction vs observation")
    print(f"    predicted rho_D, general form (unequal variances) = {pred:+.4f}")
    print(f"    predicted rho_D, equal-variance form              = {pred_eq:+.4f}")
    print(f"    OBSERVED  rho_D (YoY of the contrast)             = {r_obs:+.4f}"
          f"   (n={n_obs})\n")

    # The intercorrelation at which the contrast has exactly zero reliability.
    crossover = (r_opd + r_ed) / 2
    print(f"    zero-reliability crossover: r12 = {crossover:.4f} under equal "
          f"variances")
    print(f"    observed r12 = {r12:.4f}, i.e. "
          f"{'above' if r12 > crossover else 'below'} it\n")

    print("what intercorrelation would a usable contrast have needed?")
    for target in (0.20, 0.30, 0.40):
        # solve rho_D = t for r12 under equal variances
        need = (r_opd + r_ed - 2 * target) / (2 - 2 * target)
        print(f"    rho_D = {target:.2f}  requires r12 <= {need:+.3f}")
    print("\n    The components are positively correlated, as any two measures"
          " of the same\n    player's passing must be. At these r11 and r22 no"
          " ATTAINABLE intercorrelation\n    yields a usable single-season raw"
          " difference: the requirement is negative.\n")

    # ------------------------------------------------------------------ caveat
    print("two things this does NOT show")
    print("    1. r11 and r22 here are year-over-year correlations, which are"
          " reliability\n       TIMES trait persistence, so they are lower"
          " bounds on reliability. Reading\n       them as reliabilities biases"
          " the prediction down, which is the direction of\n       the gap"
          f" above ({pred:+.3f} predicted vs {r_obs:+.3f} observed)."
          " Disattenuating r12 by them\n       gives"
          f" {r12 / np.sqrt(r_opd * r_ed):.2f} > 1, which is impossible and"
          " confirms they are not\n       reliabilities. The identity is"
          " therefore checked in YoY units against the\n       YoY contrast"
          " (+0.068), NOT against the split-half headline (-0.008): Test 03\n"
          "       never computed component split-halves, so that comparison"
          " cannot be made.")
    print("    2. The identity governs UNSHRUNK SINGLE-SEASON DIFFERENCES."
          " It says nothing\n       about estimators that do not form a raw"
          " difference. Pooling seasons raises\n       component reliability by"
          " Spearman-Brown, and the contrast rides along:")
    for k in (2, 3, 4):
        sb = lambda r: k * r / (1 + (k - 1) * r)  # noqa: E731
        print(f"         {k} seasons pooled -> r11={sb(r_opd):.3f} "
              f"r22={sb(r_ed):.3f}  =>  rho_D ~ "
              f"{(sb(r_opd) + sb(r_ed) - 2 * r12) / (2 - 2 * r12):+.3f}")
    print("       That is an illustration, not a result: it holds r12 fixed"
          " when pooling\n       would also raise it. It is enough to show the"
          " door was never closed.\n")
    print("Corrected claim: the null was forced by the difference operator on"
          " components\nof this reliability, not by fit-relevance as such."
          " Hierarchical shrinkage of\nthe contrast, multi-season pooling and"
          " latent-trait treatment with known\nmeasurement error remain untried."
          " None of this rescues the thesis -- Test B is\na hard null on its own"
          " terms -- it corrects what the writeup calls impossible.")


if __name__ == "__main__":
    main()
