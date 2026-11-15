"""A1 - CPR tomography identifiability: what does the half-step map determine?

The manuscript's closing claim is that, if the collapse holds, the width map
returns the transparency tau(nu) of the weak link as an inverse problem. This
script quantifies that sentence instead of leaving it as prose. Precedent is
acknowledged: Tjotta et al. (arXiv:2511.14313) extract CPR harmonics from
I_c-vs-power fits; the variant tested here inverts the *fractional-step width
map* w_net(alpha) measured with the floor-free estimator.

Questions answered, with numbers:

  1. Sensitivity. How does the net half-step width curve respond to the two
     leading harmonic ratios r2 = c2/c1, r3 = c3/c1? (Jacobian by central
     differences of full RSJ sweeps.)
  2. Identifiability. Given a stated per-point measurement precision (2% and 5%
     of the deterministic net width, the manuscript's tiers), what are
     sigma(r2), sigma(r3) and their correlation? Is the half-step map a
     c2-meter, a (c2,c3)-meter, or degenerate?
  3. Truncation bias. The truth is the full 5-harmonic Beenakker CPR; the fit
     model is 3 harmonics (c1, c2, c3). Does the truncated model bias the
     recovered r2?
  4. Material meaning. Convert sigma(r2) to sigma(tau) through the stored
     Beenakker table (tau = 0.5, 0.8, 0.95), quadratic-local derivative at 0.8.

Design: 7 deterministic RSJ alpha-sweeps (truth, model centre, 4 perturbations,
sinusoidal control), alpha = 0.2..1.2 step 0.1, idc window [2.2375, 3.0375]
step 0.005 (centre of the deterministic step +-0.4, >=12 sigma_k margins).
Monte-Carlo inversion is done on the linearized model (2000 draws per noise
tier) -- valid for small noise and stated. Deterministic; no RNG in the physics,
seeded RNG only for the synthetic noise draws.
"""
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junction.cpr import normalize_cpr    # noqa: E402
from junction.rsj import OMEGA, run_rsj   # noqa: E402

OUT = ROOT / "out"
IDC = np.arange(2.2375, 3.0375 + 1e-9, 0.005)
ALPHAS = np.round(np.arange(0.2, 1.2 + 1e-9, 0.1), 4)
SIGMA_K = 0.025
V_T = 0.5 * OMEGA
DELTA = 0.03                    # central-difference step in r2, r3
W_NET_DET = 0.1342              # T1 deterministic net width (noise scale anchor)
NOISE_TIERS = {"2pct": 0.02 * W_NET_DET, "5pct": 0.05 * W_NET_DET}
N_MC = 2000
MC_SEED = 20260805


def soft_width(v, idc=IDC, v_target=V_T, sk=SIGMA_K):
    di = np.gradient(np.asarray(idc, float))
    return float(np.sum(np.exp(-0.5 * ((v - v_target) / sk) ** 2) * di))


def sweep(c):
    """Net-free soft width curve w(alpha) for CPR c (control subtracted later)."""
    out = []
    for a in ALPHAS:
        v = run_rsj(IDC, np.full_like(IDC, a * OMEGA), np.asarray(c, float))
        out.append(soft_width(v))
    return np.array(out)


def model_cpr(r2, r3):
    c, _ = normalize_cpr([1.0, r2, r3])
    return np.asarray(c, float)


def main():
    mat = json.loads((ROOT / "material.json").read_text())
    hb = mat["cpr"]["harmonics_by_tau"]
    c_truth, _ = normalize_cpr(hb["0.8"])
    c_truth = np.asarray(c_truth, float)
    r2_0 = hb["0.8"][1] / hb["0.8"][0]
    r3_0 = hb["0.8"][2] / hb["0.8"][0]

    print("=" * 78)
    print("A1 - CPR inversion identifiability from the half-step width map")
    print(f"  truth: full 5-harmonic tau=0.8 (r2={r2_0:+.5f}, r3={r3_0:+.5f})")
    print(f"  fit model: 3 harmonics [1, r2, r3]; alpha grid {ALPHAS[0]}..{ALPHAS[-1]}"
          f" ({len(ALPHAS)} pts); {len(IDC)}-pt idc window")
    print("=" * 78)

    t0 = time.time()
    w_ctl = sweep([1.0])
    print(f"  control sweep done [{time.time()-t0:.0f}s]")
    y_truth = sweep(c_truth) - w_ctl
    print(f"  truth sweep done   [{time.time()-t0:.0f}s]")
    y_c = sweep(model_cpr(r2_0, r3_0)) - w_ctl
    print(f"  centre sweep done  [{time.time()-t0:.0f}s]")
    y_r2p = sweep(model_cpr(r2_0 + DELTA, r3_0)) - w_ctl
    y_r2m = sweep(model_cpr(r2_0 - DELTA, r3_0)) - w_ctl
    y_r3p = sweep(model_cpr(r2_0, r3_0 + DELTA)) - w_ctl
    y_r3m = sweep(model_cpr(r2_0, r3_0 - DELTA)) - w_ctl
    print(f"  4 perturbation sweeps done [{time.time()-t0:.0f}s]")

    J = np.column_stack([(y_r2p - y_r2m) / (2 * DELTA),
                         (y_r3p - y_r3m) / (2 * DELTA)])
    # sensitivity summary
    s2, s3 = np.linalg.norm(J[:, 0]), np.linalg.norm(J[:, 1])
    print(f"\n  |dw/dr2| (L2 over alpha grid) = {s2:.4f}")
    print(f"  |dw/dr3| (L2 over alpha grid) = {s3:.4f}   (ratio {s2/s3:.1f}x)")

    JtJ = J.T @ J
    cond = float(np.linalg.cond(JtJ))
    JtJi = np.linalg.inv(JtJ)
    rho = float(JtJi[0, 1] / np.sqrt(JtJi[0, 0] * JtJi[1, 1]))
    print(f"  cond(J^T J) = {cond:.1f}, correlation rho(r2,r3) = {rho:+.3f}")

    # truncation bias: project the truth-vs-truncated-model mismatch
    dbias = JtJi @ J.T @ (y_truth - y_c)
    print(f"  truncation bias: d_r2 = {dbias[0]:+.5f}, d_r3 = {dbias[1]:+.5f}"
          f"   (truth has c4/c1={hb['0.8'][3]/hb['0.8'][0]:+.4f}, "
          f"c5/c1={hb['0.8'][4]/hb['0.8'][0]:+.4f})")

    # local d(r2)/d(tau) at 0.8 from the quadratic through the stored table
    taus = np.array([0.5, 0.8, 0.95])
    r2s = np.array([hb[k][1] / hb[k][0] for k in ("0.5", "0.8", "0.95")])
    p = np.polyfit(taus, r2s, 2)
    dr2_dtau = float(2 * p[0] * 0.8 + p[1])
    print(f"  d(r2)/d(tau) at tau=0.8 = {dr2_dtau:+.4f}")

    rng = np.random.default_rng(MC_SEED)
    tiers = {}
    for name, sig in NOISE_TIERS.items():
        cov = JtJi * sig**2
        s_r2, s_r3 = np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1])
        # linearized Monte Carlo (validates the Gaussian propagation)
        draws = rng.standard_normal((N_MC, len(ALPHAS))) * sig
        fits = draws @ J @ JtJi.T
        mc_r2, mc_r3 = fits[:, 0].std(ddof=1), fits[:, 1].std(ddof=1)
        s_tau = s_r2 / abs(dr2_dtau)
        tiers[name] = {"sigma_y": sig, "sigma_r2": float(s_r2),
                       "sigma_r3": float(s_r3), "mc_sigma_r2": float(mc_r2),
                       "mc_sigma_r3": float(mc_r3), "sigma_tau": float(s_tau),
                       "rel_r2_pct": float(100 * s_r2 / abs(r2_0)),
                       "rel_r3_pct": float(100 * s_r3 / abs(r3_0))}
        print(f"\n  [{name}] sigma_y = {sig:.5f}")
        print(f"    sigma(r2) = {s_r2:.5f} ({100*s_r2/abs(r2_0):.1f}% of r2)   "
              f"[MC check {mc_r2:.5f}]")
        print(f"    sigma(r3) = {s_r3:.5f} ({100*s_r3/abs(r3_0):.0f}% of r3)   "
              f"[MC check {mc_r3:.5f}]")
        print(f"    => sigma(tau) = {s_tau:.4f} at tau = 0.8")

    print("\nVERDICT: the half-step map is a c2-meter -- and through Beenakker, a "
          f"transparency meter with sigma(tau) = {tiers['2pct']['sigma_tau']:.3f} "
          f"at 2% width precision. r3 is weakly constrained at best "
          f"({tiers['2pct']['rel_r3_pct']:.0f}% at 2%, "
          f"{tiers['5pct']['rel_r3_pct']:.0f}% at 5%): resolving it needs the "
          "Omega/3 step. NOTE the truncation bias on r2 exceeds the 2%-tier "
          "statistical error -- fit the one-parameter Beenakker family (tau "
          "directly) rather than free harmonics to remove it by construction.")

    result = {
        "description": "A1 identifiability of CPR harmonics from the net "
                       "half-step width map w(alpha), floor-free estimator, "
                       "linearized inversion about the tau=0.8 truth.",
        "truth": {"r2": r2_0, "r3": r3_0,
                  "c4_over_c1": hb["0.8"][3] / hb["0.8"][0],
                  "c5_over_c1": hb["0.8"][4] / hb["0.8"][0]},
        "alpha_grid": ALPHAS.tolist(),
        "curves": {"y_truth": y_truth.tolist(), "y_model_centre": y_c.tolist(),
                   "w_control": w_ctl.tolist()},
        "jacobian": J.tolist(),
        "sensitivity_L2": {"r2": float(s2), "r3": float(s3)},
        "correlation_r2_r3": rho,
        "cond_JtJ": cond,
        "truncation_bias": {"d_r2": float(dbias[0]), "d_r3": float(dbias[1])},
        "dr2_dtau_at_0p8": dr2_dtau,
        "noise_tiers": tiers,
        "mc": {"n": N_MC, "seed": MC_SEED,
               "note": "linearized MC validates Gaussian propagation; nonlinear "
                       "refits would need ~10^2 forward sweeps per draw and are "
                       "unnecessary at these noise levels"},
    }
    (OUT / "cpr_inversion.json").write_text(json.dumps(result, indent=2))
    print(f"  json: {OUT/'cpr_inversion.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
