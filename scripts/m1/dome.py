"""Superconducting dome parametrization and material.json assembly.

Faithful port of the m1_reference SC-dome block:
  Tc(nu) = Tc_max * [1 - ((nu - nu_opt)/nu_hw)^2]_+
  Delta(Tc, T) = 1.764 kB Tc tanh(1.74 sqrt(Tc/T - 1))
and the nu_table / material.json builder consumed by M3 and M4.
"""

import numpy as np

from m1.bm import KB_MEV_K, RATIO, WAB
from m1.cpr import beenakker_harmonics

TC_MAX, NU_OPT, NU_HW = 2.0, -2.4, 0.5


def Tc_of_nu(nu):
    """Phenomenological dome: critical temperature (K) at filling nu."""
    z = 1.0 - ((nu - NU_OPT) / NU_HW) ** 2
    return TC_MAX * max(0.0, z)


def Delta_of(Tc, T=0.1):
    """Gap (meV) from Tc via the interpolation formula; 0 when Tc <= T."""
    if Tc <= T:
        return 0.0
    return 1.764 * KB_MEV_K * Tc * np.tanh(1.74 * np.sqrt(Tc / T - 1.0))


def nu_grid():
    """Filling grid used for the nu_table (matches material.json rows)."""
    return np.round(np.arange(-2.9, -1.899, 0.1), 2)


def build_nu_table():
    """Row-for-row nu_table: nu, Tc_K, Delta_meV, Ic_nA (Ic scaled to 30 nA at optimum)."""
    D0 = Delta_of(TC_MAX)
    table = []
    for nu in nu_grid():
        Tc = Tc_of_nu(nu)
        D = Delta_of(Tc)
        table.append({"nu": float(nu), "Tc_K": round(Tc, 4),
                      "Delta_meV": round(D, 4),
                      "Ic_nA": round(30.0 * D / D0, 2)})
    return table


def build_material(theta_magic, hbv, Wm_N3, Wm_N4, Wq, Wp,
                   taus=(0.5, 0.8, 0.95)):
    """Assemble the full material.json dict from computed band/dome/CPR inputs.

    Schema mirrors the provided material.json key-for-key. Flat-band scalars
    are passed in (computed with m1.bm); dome and CPR are regenerated here.
    """
    D0 = Delta_of(TC_MAX)
    cpr = {str(t): beenakker_harmonics(t).tolist() for t in taus}
    material = {
        "contract_version": "0.2",
        "bm_params": {"theta_design_deg": theta_magic,
                      "hbar_vF_meV_nm": hbv,
                      "vF_1e6_m_s": round(hbv / 658.0, 3),
                      "wAB_meV": WAB, "wAA_over_wAB": RATIO,
                      "basis_shells_N": 4},
        "flat_band": {"theta_magic_deg": theta_magic,
                      "W_meV": round(float(Wm_N4), 3),
                      "W_convergence_N3_N4": [round(float(Wm_N3), 3),
                                              round(float(Wm_N4), 3)],
                      "W_pm_0p05deg_meV": [round(float(Wq), 3),
                                           round(float(Wp), 3)]},
        "sc_dome": {"Tc_max_K": TC_MAX, "nu_opt": NU_OPT, "nu_halfwidth": NU_HW,
                    "Delta_form": "1.764*kB*Tc*tanh(1.74*sqrt(Tc/T-1))"},
        "Delta0_meV": round(D0, 4), "Tc_K": TC_MAX,
        "Ic_nA": 30.0, "Rn_kOhm": 2.0,
        "cpr": {"model": "Beenakker short junction, T=0.1 K, normalized max=1",
                "tau_nominal": 0.8, "harmonics_by_tau": cpr},
        "nu_table": build_nu_table(),
        "caveats": ["pairing mechanism not modeled: Delta,Tc,Ic phenomenological",
                    "Rn(nu) held fixed at 2 kOhm (experimental input needed)",
                    "lattice relaxation beyond wAA/wAB ratio not included"],
    }
    return material
