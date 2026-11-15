"""F1 - the capacitive mimic: can beta_c alone fake a half-integer step?

Why this exists
---------------
The contract states beta_c ~ 0.04 and declares the junction overdamped, hence
RSJ-valid (M3_BRIEF Sec. 1). That declaration has never been *measured*: it is
classic RCSJ physics that a finite capacitance generates subharmonic Shapiro
steps even for a PURELY SINUSOIDAL CPR (no c2 at all) — which is precisely the
class of confound our manuscript's discriminants exist to exclude. Zhu et al.
(2512.01586) catalogued the vortex mimic; the capacitive mimic is the other
canonical one, and it lives inside our own contract parameters.

Two questions, both answered with the floor-free estimator of T1:

  Q1 (self-defence): at the contract beta_c = 0.04, how much half-step width
     does a pure-sine CPR fake? If it is not << our 0.134 net signal, the
     manuscript has a problem. This is a backtest of our own RSJ approximation.
  Q2 (mimic catalogue): sweeping beta_c upward, at what damping does the fake
     half-step become comparable to the real one? This is the first quantified
     entry of the "false fractional signatures" guide.

Method
------
Full RCSJ integration, RK4 on the 2D system
    beta_c * dv/dtau + v = i_dc + i_ac cos(Omega tau) - sum_m c_m sin(m phi),
    dphi/dtau = v,
with the validated production geometry at nu = -2.4: Omega = 5, i_ac = 3.5
(alpha = 0.70), i_dc window [2.10, 3.10] step 0.005 (the T1 window), spp = 60,
150 T transient / 300 T averaging. Independent initial conditions per bias point
(phi = 0, v = i_dc), same convention as run_rsj — in the underdamped regime the
staircase is protocol-dependent (hysteretic), and this protocol is stated, not
hidden. Deterministic; no RNG.

Cross-integrator backtest: at beta_c -> 0 the RCSJ integrator must reproduce the
validated first-order RSJ result for BOTH CPRs; run at beta_c = 0.005 and
compared against junction.rsj.run_rsj on the identical grid.

Outputs out/rcsj_confound.json.
"""
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junction.cpr import normalize_cpr, cpr as cpr_eval   # noqa: E402
from junction.rsj import OMEGA, run_rsj                   # noqa: E402

OUT = ROOT / "out"
IDC = np.arange(2.10, 3.10 + 1e-9, 0.005)
I_AC = 3.5
SPP = 60
N_TR, N_AV = 150, 300
V_T = 0.5 * OMEGA
SIGMA_K = 0.025
# Explicit RK4 needs dt < ~2.8*beta_c for the stiff damping term, so spp scales
# as 1/beta_c below 0.03. The extra points 0.16 and 0.35 bracket the plasma
# resonance omega_p = 1/sqrt(beta_c) = Omega/2, i.e. beta_c = (2/Omega)^2 = 0.16.
BETAS = [0.005, 0.02, 0.04, 0.1, 0.16, 0.25, 0.35, 0.5, 1.0, 2.0]


def spp_for(beta_c):
    need = int(np.ceil(2.0 * np.pi / (OMEGA * 2.0 * beta_c)))   # dt < 2*beta_c
    return max(SPP, need)


def soft_width(v, idc, v_target=V_T, sk=SIGMA_K):
    di = np.gradient(np.asarray(idc, float))
    return float(np.sum(np.exp(-0.5 * ((v - v_target) / sk) ** 2) * di))


def run_rcsj(i_dc, i_ac, c, beta_c, omega=OMEGA,
             n_transient=N_TR, n_avg=N_AV, spp=None):
    """Vectorized fixed-step RK4 on (phi, v); returns time-averaged dphi/dtau."""
    if spp is None:
        spp = spp_for(beta_c)
    i_dc = np.asarray(i_dc, float)
    T = 2.0 * np.pi / omega
    dt = T / spp
    phi = np.zeros_like(i_dc)
    v = i_dc.copy()                       # start near the running branch

    def acc(tau, ph, vv):
        return (i_dc + i_ac * np.cos(omega * tau) - cpr_eval(ph, c) - vv) / beta_c

    n1, n2 = int(n_transient * spp), int(n_avg * spp)
    tau = 0.0
    phi_a, tau_a = phi.copy(), tau
    for step in range(n1 + n2):
        if step == n1:
            phi_a, tau_a = phi.copy(), tau
        k1p, k1v = v, acc(tau, phi, v)
        k2p, k2v = v + 0.5 * dt * k1v, acc(tau + 0.5 * dt, phi + 0.5 * dt * k1p, v + 0.5 * dt * k1v)
        k3p, k3v = v + 0.5 * dt * k2v, acc(tau + 0.5 * dt, phi + 0.5 * dt * k2p, v + 0.5 * dt * k2v)
        k4p, k4v = v + dt * k3v, acc(tau + dt, phi + dt * k3p, v + dt * k3v)
        phi = phi + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        v = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        tau += dt
    return (phi - phi_a) / (tau - tau_a)


def main():
    mat = json.loads((ROOT / "material.json").read_text())
    c_skew, _ = normalize_cpr(mat["cpr"]["harmonics_by_tau"]["0.8"])
    c_skew = np.asarray(c_skew, float)
    c_sin = np.array([1.0])

    print("=" * 78)
    print("F1 - capacitive (beta_c) mimic of the half-integer step, pure-sine CPR")
    print(f"  window {IDC[0]}..{IDC[-1]} step 0.005, i_ac={I_AC}, Omega={OMEGA}, "
          f"spp={SPP} (dt resolves the plasma period down to beta_c~0.005)")
    print("=" * 78)

    # ---- cross-integrator backtest at near-zero beta_c --------------------
    t0 = time.time()
    v_ref_sin = run_rsj(IDC, np.full_like(IDC, I_AC), c_sin)
    v_ref_skew = run_rsj(IDC, np.full_like(IDC, I_AC), c_skew)
    w_ref_sin, w_ref_skew = soft_width(v_ref_sin, IDC), soft_width(v_ref_skew, IDC)
    v_r_sin = run_rcsj(IDC, I_AC, c_sin, 0.005)
    v_r_skew = run_rcsj(IDC, I_AC, c_skew, 0.005)
    dev_sin = float(np.max(np.abs(v_r_sin - v_ref_sin)))
    dev_skew = float(np.max(np.abs(v_r_skew - v_ref_skew)))
    print(f"[backtest beta_c->0] max|v_RCSJ - v_RSJ|: sine {dev_sin:.4f}, "
          f"skewed {dev_skew:.4f}  (v-resolution 0.0167)   [{time.time()-t0:.0f}s]")

    # ---- the sweep --------------------------------------------------------
    rows = []
    for b in BETAS:
        t0 = time.time()
        v_sin = run_rcsj(IDC, I_AC, c_sin, b)
        w_sin = soft_width(v_sin, IDC)
        fake = w_sin - w_ref_sin              # excess over the RSJ transit floor
        row = {"beta_c": b, "w_soft_sine": w_sin, "fake_halfstep_net": fake}
        if abs(b - 0.04) < 1e-12:             # contract point: also the real CPR
            v_sk = run_rcsj(IDC, I_AC, c_skew, b)
            row["w_soft_skewed_at_contract"] = soft_width(v_sk, IDC)
            row["max_dev_skewed_vs_rsj"] = float(np.max(np.abs(v_sk - v_ref_skew)))
        rows.append(row)
        extra = (f"  | skewed: {row['w_soft_skewed_at_contract']:.4f} "
                 f"(RSJ {w_ref_skew:.4f})" if "w_soft_skewed_at_contract" in row else "")
        print(f"  beta_c={b:<6} w_sine={w_sin:.4f}  fake_net={fake:+.4f}"
              f"{extra}   [{time.time()-t0:.0f}s]")

    net_signal = 0.1342                       # T1 deterministic net (control-subtracted)
    fake_at_contract = next(r["fake_halfstep_net"] for r in rows
                            if abs(r["beta_c"] - 0.04) < 1e-12)
    print("-" * 78)
    print(f"VERDICT Q1: at contract beta_c = 0.04 the pure-sine fake half-step is "
          f"{fake_at_contract:+.4f} vs the real net signal 0.1342 "
          f"({abs(fake_at_contract)/net_signal*100:.1f}%).")

    result = {
        "description": "F1: RCSJ capacitive mimic of the half-integer step with a "
                       "purely sinusoidal CPR, floor-free estimator, production "
                       "geometry (nu=-2.4, alpha=0.70).",
        "protocol_note": "Independent initial conditions (phi=0, v=i_dc) per bias "
                         "point; in the underdamped regime the staircase is "
                         "protocol-dependent and this protocol is stated.",
        "cross_integrator_backtest": {
            "beta_c": 0.005, "max_dev_sine": dev_sin, "max_dev_skewed": dev_skew,
            "v_resolution": OMEGA / N_AV},
        "rsj_reference": {"w_soft_sine_floor": w_ref_sin,
                          "w_soft_skewed": w_ref_skew},
        "net_signal_reference": net_signal,
        "sweep": rows,
        "fake_at_contract_beta": fake_at_contract,
        "fake_over_signal_at_contract": abs(fake_at_contract) / net_signal,
    }
    (OUT / "rcsj_confound.json").write_text(json.dumps(result, indent=2))
    print(f"  json: {OUT/'rcsj_confound.json'}")
    print("=" * 78)


if __name__ == "__main__":
    main()
