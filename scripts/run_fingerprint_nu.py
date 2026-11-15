"""M3 Deliverable 3: AT-3 fractional fingerprint vs filling nu (paper Fig 3b).

Reads material.json (M1). Per filling nu:
  Ic(nu) from nu_table  ->  Omega(nu) = 5.0 * 30/Ic   (f_c = 2e*Ic*Rn/h prop. Ic)
                            Gamma(nu) = 0.140 * 30/Ic (E_J prop. Ic)
  CPR shape: Beenakker harmonics at tau = 0.8 (M1 brief Sec. 4.4: amplitude
  prop. Delta(nu) is absorbed into Ic(nu); shape fixed at tau_nominal).
Grids scale with Omega so that alpha = i_ac/Omega in [0, 2.4] and
i_dc/Omega in [0.2, 1.6] match the validated AT-3 stage-B geometry exactly.
Averaging scales as n_avg = 300 * Omega/5 so the v-resolution Omega/n_avg
stays 0.0167 and the plateau tolerance 0.025 (= 1.5x v_res) is unchanged.
Deterministic RK4 (spp = 60) — no RNG.

Usage: python run_fingerprint_nu.py --nu -2.4 [--smoke]
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junction.analysis import width_curve
from junction.cpr import normalize_cpr
from junction.rsj import grid_run

IC_CONTRACT_NA = 30.0
OMEGA_CONTRACT = 5.0
GAMMA_CONTRACT = 0.140


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid, short averaging: CLI/plumbing test only")
    args = ap.parse_args()

    mat = json.loads((ROOT / "material.json").read_text())
    row = next((r for r in mat["nu_table"] if abs(r["nu"] - args.nu) < 1e-9), None)
    if row is None:
        raise SystemExit(f"nu={args.nu} not in material.json nu_table")
    if row["Ic_nA"] <= 0.0:
        raise SystemExit(f"nu={args.nu}: Ic=0 (outside SC dome), no fingerprint")

    ic = row["Ic_nA"]
    omega = OMEGA_CONTRACT * IC_CONTRACT_NA / ic
    gamma = GAMMA_CONTRACT * IC_CONTRACT_NA / ic

    c_raw = np.array(mat["cpr"]["harmonics_by_tau"]["0.8"], float)
    c, renorm = normalize_cpr(c_raw)
    # M1 says max-normalized already; renormalize defensively and log
    if abs(renorm - 1.0) > 1e-6:
        print(f"note: M1 harmonics renormalized by {renorm:.8f}")

    scale = omega / OMEGA_CONTRACT
    n_transient = int(np.ceil(150 * scale))
    n_avg = int(np.ceil(300 * scale))
    spp = 60
    tol = 0.025  # = 1.5 * v_res, v_res = omega/n_avg = 0.0167 by construction

    if args.smoke:
        iac = omega * np.arange(0.0, 2.4 + 1e-9, 0.6)
        idc = omega * np.arange(0.2, 1.6 + 1e-9, 0.1)
        n_transient, n_avg = 30, 60
        tol = 1.5 * omega / n_avg
    else:
        iac = omega * np.arange(0.0, 2.4 + 1e-9, 0.1)     # 25 pts, alpha step 0.1
        idc = omega * np.arange(0.2, 1.6 + 1e-9, 0.005)   # 281 pts, step 0.005*Omega

    t0 = time.time()
    v = grid_run(idc, iac, c, omega=omega,
                 n_transient=n_transient, n_avg=n_avg, spp=spp)
    dt = time.time() - t0

    w_half = width_curve(v, idc, 0.5 * omega, tol)
    w_int1 = width_curve(v, idc, 1.0 * omega, tol)
    w_third = width_curve(v, idc, omega / 3.0, tol)
    k_best = int(np.argmax(w_half))

    # widths are in units of Ic(nu); also report i_dc-grid-normalized alpha
    result = {
        "nu": args.nu, "Ic_nA": ic, "Omega": omega, "Gamma": gamma,
        "Tc_K": row["Tc_K"], "Delta_meV": row["Delta_meV"],
        "cpr_harmonics": c.tolist(), "cpr_renorm_factor": renorm,
        "tau_shape": 0.8,
        "numerics": {"spp": spp, "n_transient": n_transient, "n_avg": n_avg,
                     "tol": tol, "v_resolution": omega / n_avg,
                     "rng": "deterministic - no RNG", "smoke": bool(args.smoke)},
        "i_ac": iac.tolist(), "alpha": (iac / omega).tolist(),
        "w_half": w_half.tolist(), "w_int1": w_int1.tolist(),
        "w_third": w_third.tolist(),
        "w_half_max": float(w_half[k_best]),
        "i_ac_opt": float(iac[k_best]),
        "alpha_opt": float(iac[k_best] / omega),
        "w_half_max_nA": float(w_half[k_best] * ic),
        "runtime_s": dt,
    }

    outdir = ROOT / "out" / "fingerprint"
    outdir.mkdir(parents=True, exist_ok=True)
    slug = f"nu_{args.nu:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
    tag = "_smoke" if args.smoke else ""
    np.savez(outdir / f"{slug}{tag}.npz", i_ac=iac, i_dc=idc, v=v,
             w_half=w_half, w_int1=w_int1, w_third=w_third,
             c=c, omega=omega, gamma=gamma, spp=spp,
             n_transient=n_transient, n_avg=n_avg, tol=tol)
    (outdir / f"{slug}{tag}.json").write_text(json.dumps(result, indent=1))
    print(f"nu={args.nu}: Ic={ic} nA, Omega={omega:.3f}, Gamma={gamma:.3f} | "
          f"w_half_max={result['w_half_max']:.4f} Ic ({result['w_half_max_nA']:.2f} nA) "
          f"at alpha={result['alpha_opt']:.2f} | t={dt:.0f}s")
    print(f"saved: {outdir / (slug + tag)}.npz/.json")


if __name__ == "__main__":
    main()
