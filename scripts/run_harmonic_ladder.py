"""H1 - the fractional-step harmonic ladder: is there a real m = 3 step?

Motivation, sharpened by A1
---------------------------
A1 (out/cpr_inversion.json) measured that the half-step width map pins r2 = c2/c1
to ~1% but leaves r3 = c3/c1 22-54% uncertain: resolving the third harmonic
needs the Omega/3 step itself. Separately, `run_fingerprint_nu.py` has computed
`w_third` since the beginning and nothing ever used it — for good reason: read
off the stored fingerprints, w_third "peaks" at alpha = 0 for nu = -2.7, -2.8,
i.e. with the drive off. That is the hard-estimator detection floor T1
identified, not a step. The m = 3 data was quietly unusable.

This script re-asks the question with the floor-free machinery: soft
(Gaussian-kernel) width at v = Omega/2 AND v = Omega/3, minus a sinusoidal-CPR
control at identical drive — the control carries no harmonic content, so
whatever it shows at Omega/3 is definitionally artifact.

Why it matters: a second fractional order turns the fingerprint into a RATIO,
R23(nu) = W_net(Omega/3)/W_net(Omega/2), fixed by the CPR's harmonic content and
insensitive to overall drive calibration. It would complete the tomography (pin
r3); its absence is equally reportable as the observable's boundary.

Method: deterministic only (RK4, no RNG) — this asks what the CPR produces;
thermal survival would follow only if a step exists. Per (nu, m): the i_dc
window is centred where the ZERO-DRIVE staircase crosses v = Omega/m (so the
window does not presuppose the step), half-width +-0.5 absolute, step 0.005;
alpha = 0.0..1.2 step 0.1 with i_ac = alpha*Omega; n_transient/n_avg scale as
Omega/5 (repo convention, v-resolution held at 0.0167).

Chunking: one JSON block per (nu, m) under out/ladder_blocks/; a run computes
only missing blocks (restrict with --nu / --m), and the summary
out/harmonic_ladder.json is written only when all 10 blocks exist.
"""
import argparse
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
BLOCKS = OUT / "ladder_blocks"
IDC_STEP = 0.005
IDC_HALF_WINDOW = 0.5
SIGMA_K = 0.025
ALPHAS = np.round(np.arange(0.0, 1.2 + 1e-9, 0.10), 4)
NUS = [-2.4, -2.5, -2.6, -2.7, -2.8]
ORDERS = (2, 3)


def nu_slug(nu):
    return f"nu_{nu:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")


def soft_width(v, idc, v_target, sk=SIGMA_K):
    di = np.gradient(np.asarray(idc, float))
    return float(np.sum(np.exp(-0.5 * ((np.asarray(v, float) - v_target) / sk) ** 2) * di))


def scaled_numerics(omega):
    scale = omega / 5.0
    return int(np.ceil(150 * scale)), int(np.ceil(300 * scale))


def window_for(omega, m, c, n_tr, n_av):
    """i_dc window centred where the zero-drive staircase crosses v = Omega/m."""
    v_t = omega / m
    coarse = np.arange(0.2 * omega, 1.6 * omega, 0.02 * omega / 5.0)
    v = run_rsj(coarse, np.zeros_like(coarse), c, omega=omega,
                n_transient=n_tr, n_avg=n_av)
    centre = float(np.interp(v_t, v, coarse))
    idc = np.arange(centre - IDC_HALF_WINDOW, centre + IDC_HALF_WINDOW + 1e-9,
                    IDC_STEP)
    return idc, v_t


def compute_block(nu, m, c, c_sin):
    fp = json.loads((OUT / "fingerprint" / f"{nu_slug(nu)}.json").read_text())
    omega = float(fp["Omega"])
    n_tr, n_av = scaled_numerics(omega)
    t0 = time.time()
    idc, v_t = window_for(omega, m, c, n_tr, n_av)
    w_sig, w_ctl = [], []
    for a in ALPHAS:
        iac = np.full_like(idc, a * omega)
        v_s = run_rsj(idc, iac, c, omega=omega, n_transient=n_tr, n_avg=n_av)
        v_c = run_rsj(idc, iac, c_sin, omega=omega, n_transient=n_tr, n_avg=n_av)
        w_sig.append(soft_width(v_s, idc, v_t))
        w_ctl.append(soft_width(v_c, idc, v_t))
    w_sig, w_ctl = np.array(w_sig), np.array(w_ctl)
    net = w_sig - w_ctl
    k = int(np.argmax(net))
    return {
        "nu": nu, "m": m, "Omega": omega, "v_target": v_t,
        "n_transient": n_tr, "n_avg": n_av,
        "idc_window": [float(idc[0]), float(idc[-1])], "idc_step": IDC_STEP,
        "alpha": ALPHAS.tolist(),
        "w_soft_signal": w_sig.tolist(), "w_soft_control": w_ctl.tolist(),
        "w_net": net.tolist(),
        "w_net_max": float(net[k]), "alpha_opt": float(ALPHAS[k]),
        "net_at_alpha0": float(net[0]),
        "control_at_alpha_opt": float(w_ctl[k]),
        "runtime_s": round(time.time() - t0, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=None,
                    help="compute only this filling's missing blocks")
    ap.add_argument("--m", type=int, default=None, choices=ORDERS,
                    help="compute only this fractional order")
    args = ap.parse_args()
    BLOCKS.mkdir(parents=True, exist_ok=True)

    mat = json.loads((ROOT / "material.json").read_text())
    c, _ = normalize_cpr(mat["cpr"]["harmonics_by_tau"]["0.8"])
    c = np.asarray(c, float)
    c_sin = np.array([1.0])

    todo_nu = NUS if args.nu is None else [args.nu]
    todo_m = ORDERS if args.m is None else (args.m,)

    print("=" * 78)
    print("H1 - fractional-step harmonic ladder (deterministic, floor-free)")
    print(f"  c2/c1 = {c[1]/c[0]:+.5f}, c3/c1 = {c[2]/c[0]:+.5f}; "
          f"alpha 0..1.2 step 0.1")
    print("=" * 78)

    for nu in todo_nu:
        for m in todo_m:
            bp = BLOCKS / f"{nu_slug(nu)}_m{m}.json"
            if bp.exists():
                print(f"  [resume] nu={nu} m={m}")
                continue
            block = compute_block(nu, m, c, c_sin)
            bp.write_text(json.dumps(block, indent=1))
            print(f"  nu={nu:<5} m={m}: net_max={block['w_net_max']:+.4f} "
                  f"@alpha={block['alpha_opt']:.2f} "
                  f"(ctl {block['control_at_alpha_opt']:.4f}, "
                  f"alpha0 net {block['net_at_alpha0']:+.4f}) "
                  f"[{block['runtime_s']:.0f}s]")

    # ---- assemble only when complete --------------------------------------
    missing = [f"{nu}/m{m}" for nu in NUS for m in ORDERS
               if not (BLOCKS / f"{nu_slug(nu)}_m{m}.json").exists()]
    if missing:
        print("-" * 78)
        print(f"PARTIAL: missing {missing}; rerun with --nu/--m. No summary written.")
        return

    per_nu = {}
    for nu in NUS:
        row = {"nu": nu, "orders": {}}
        for m in ORDERS:
            row["orders"][f"m{m}"] = json.loads(
                (BLOCKS / f"{nu_slug(nu)}_m{m}.json").read_text())
        m2, m3 = row["orders"]["m2"], row["orders"]["m3"]
        row["Omega"] = m2["Omega"]
        row["ratio_m3_over_m2"] = (m3["w_net_max"] / m2["w_net_max"]
                                   if m2["w_net_max"] > 0 else None)
        per_nu[str(nu)] = row
        print(f"nu={nu:<5} Om={row['Omega']:7.3f}  "
              f"m2 net={m2['w_net_max']:.4f}@a={m2['alpha_opt']:.2f}  "
              f"m3 net={m3['w_net_max']:+.4f}@a={m3['alpha_opt']:.2f}  "
              f"R23={row['ratio_m3_over_m2']:+.4f}")

    m2n = np.array([per_nu[str(n)]["orders"]["m2"]["w_net_max"] for n in NUS])
    m3n = np.array([per_nu[str(n)]["orders"]["m3"]["w_net_max"] for n in NUS])
    m3a = np.array([per_nu[str(n)]["orders"]["m3"]["alpha_opt"] for n in NUS])
    resolved = bool(np.all(m3n > 0.02 * m2n) and np.all(m3a > 0.05))
    print("-" * 78)
    print(f"VERDICT: m = 3 step {'PRESENT' if resolved else 'NOT RESOLVED'} "
          f"across the dome on this grid.")
    print(f"  m3/m2 net ratios: {np.round(m3n/m2n, 4).tolist()}")

    (OUT / "harmonic_ladder.json").write_text(json.dumps({
        "description": "H1: deterministic, control-subtracted soft widths at "
                       "v = Omega/2 and Omega/3 across the dome; the Omega/3 "
                       "step is the observable A1 says is needed to pin c3/c1.",
        "cpr_normalized": c.tolist(),
        "c2_over_c1": float(c[1] / c[0]), "c3_over_c1": float(c[2] / c[0]),
        "numerics": {"integrator": "RK4 deterministic (no RNG)",
                     "idc_step": IDC_STEP, "idc_half_window": IDC_HALF_WINDOW,
                     "sigma_k": SIGMA_K, "alpha_grid": ALPHAS.tolist(),
                     "control_cpr": [1.0],
                     "window_rule": "centred on zero-drive crossing of Omega/m"},
        "per_nu": per_nu,
        "verdict": {"m3_resolved": resolved,
                    "m3_net_widths": m3n.tolist(),
                    "m3_fraction_of_m2": (m3n / m2n).tolist(),
                    "m3_alpha_opt": m3a.tolist(),
                    "criterion": "net > 2% of m2 net at every filling AND "
                                 "optimum at finite drive (alpha > 0.05)"},
    }, indent=2))
    print(f"  json: {OUT/'harmonic_ladder.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
