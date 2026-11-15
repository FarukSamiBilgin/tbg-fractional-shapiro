"""Power chain and standing-wave TCMT for the M4 integration module.

Closes the drive-power loop between the M2 photonic cavity and the M3 junction
(M4_BRIEF_DRAFT.md Sec. 1, 3). All physics here is analytic / cross-module: it
*consumes* the locked constants in ``integration/__init__.py`` and the M3
export ``junction.json`` and performs no EM or junction-dynamics recomputation
(M4_BRIEF_DRAFT.md Sec. 5 "out of scope").

Conventions (immutable, sourced):

* Standing-wave two-port TCMT, mirror-symmetric side-coupled defect cavity
  (M2_BRIEF.md Sec. 2, lines 41-46; restated M4_BRIEF_DRAFT.md line 19)::

      t(delta) = (j*delta + gamma_i) / (j*delta + gamma_i + gamma_w)
      r(delta) = -gamma_w         / (j*delta + gamma_i + gamma_w)

  with delta = omega - omega0 and gamma_x = omega0 / (2*Q_x). On resonance
  (delta = 0)::

      T = (gamma_i/(gamma_i+gamma_w))**2
      R = (gamma_w/(gamma_i+gamma_w))**2
      A = 2*gamma_i*gamma_w/(gamma_i+gamma_w)**2   ( = 2*Q_i*Q_w/(Q_i+Q_w)**2 )

  Undercoupled target A = 1.0% => Q_w = 200*Q_i (M2_BRIEF.md line 46). The
  *exact* absorption at Q_w = 200*Q_i is 400/201**2 = 0.99007% (undercoupled
  approximation 2*Q_i/Q_w = 1.000%).

* Power budget (M2_BRIEF.md line 52-53; M4_BRIEF_DRAFT.md line 21)::

      P_in --A--> P_J = A*P_in --> V_ac = sqrt(2*P_J*Rn) --> alpha = e*V_ac/(hbar*omega)

  With the locked reference P_in = 9 nW, A = 1%: P_J = 90 pW, V_ac = 0.600 mV,
  alpha = 1.000 (the photon voltage hbar*omega/e = 0.59967 mV at f0 = 145 GHz
  makes the exact alpha = 1.0005, within the +-0.002 acceptance band).

* Drive mapping to the junction (M3_BRIEF.md Sec. 2, line 34): alpha = i_ac/Omega,
  i.e. i_ac = alpha*Omega.

Everything is vectorised with numpy: every function accepts scalars or arrays
that broadcast together.
"""
import json

import numpy as np

from integration import (
    E_CHARGE,
    H_PLANCK,
    HBAR,
    F0_HZ,
    V_PHOTON,
    RN_OHM,
    A_TARGET,
    QW_OVER_QI,
    P_IN_REF_W,
    P_J_REF_W,
)

#: angular drive frequency omega0 = 2*pi*f0 (rad/s), used for gamma <-> Q.
OMEGA0_RAD = 2.0 * np.pi * F0_HZ

__all__ = [
    "tcmt_amplitudes",
    "on_resonance_TRA",
    "tcmt_power",
    "absorption_from_Q",
    "gamma_from_Q",
    "v_ac_from_pj",
    "alpha_from_vac",
    "alpha_from_pin",
    "pin_for_alpha",
    "pj_from_pin",
    "i_ac_from_alpha",
    "operating_points",
]


# --------------------------------------------------------------------------- #
# 1. Standing-wave temporal coupled-mode theory (M2_BRIEF.md Sec. 2)
# --------------------------------------------------------------------------- #
def tcmt_amplitudes(delta, gamma_i, gamma_w):
    """Complex transmission ``t`` and reflection ``r`` amplitudes.

    Standing-wave two-port convention (M2_BRIEF.md lines 41-42;
    M4_BRIEF_DRAFT.md line 19)::

        t = (1j*delta + gamma_i) / (1j*delta + gamma_i + gamma_w)
        r = -gamma_w             / (1j*delta + gamma_i + gamma_w)

    Parameters
    ----------
    delta : float or array
        Detuning omega - omega0 (same units as the rates).
    gamma_i, gamma_w : float or array
        Intrinsic (radiation + material + junction) and waveguide-coupling
        energy-decay rates. Broadcasts against ``delta``.

    Returns
    -------
    (t, r) : complex ndarray
        Field amplitudes. ``|t|**2`` is transmittance, ``|r|**2`` reflectance,
        and ``1 - |t|**2 - |r|**2`` the absorbed fraction.
    """
    delta = np.asarray(delta, dtype=float)
    gamma_i = np.asarray(gamma_i, dtype=float)
    gamma_w = np.asarray(gamma_w, dtype=float)
    denom = 1j * delta + gamma_i + gamma_w
    t = (1j * delta + gamma_i) / denom
    r = -gamma_w / denom
    return t, r


def tcmt_power(delta, gamma_i, gamma_w):
    """Off-resonance power fractions ``(T, R, A)`` from the TCMT amplitudes.

    ``T = |t|**2``, ``R = |r|**2`` and ``A = 1 - T - R`` (energy conservation;
    M2_BRIEF.md line 43). Analytically
    ``A(delta) = 2*gamma_i*gamma_w / (delta**2 + (gamma_i+gamma_w)**2)`` and
    ``T + R = 1 - A <= 1`` with equality iff ``gamma_i = 0`` (loss-free cavity),
    the narrowband-mirror limit checked in AT-M4-2.
    """
    t, r = tcmt_amplitudes(delta, gamma_i, gamma_w)
    T = np.abs(t) ** 2
    R = np.abs(r) ** 2
    A = 1.0 - T - R
    return T, R, A


def on_resonance_TRA(gamma_i, gamma_w):
    """On-resonance transmittance/reflectance/absorptance (delta = 0).

    Closed form (M2_BRIEF.md line 43; M4_BRIEF_DRAFT.md line 19)::

        T = (gamma_i/(gamma_i+gamma_w))**2
        R = (gamma_w/(gamma_i+gamma_w))**2
        A = 2*gamma_i*gamma_w/(gamma_i+gamma_w)**2

    ``T + R + A = 1`` identically (energy conservation, AT-M4-2).
    """
    gamma_i = np.asarray(gamma_i, dtype=float)
    gamma_w = np.asarray(gamma_w, dtype=float)
    tot = gamma_i + gamma_w
    T = (gamma_i / tot) ** 2
    R = (gamma_w / tot) ** 2
    A = 2.0 * gamma_i * gamma_w / tot ** 2
    return T, R, A


def absorption_from_Q(Q_i, Q_w):
    """On-resonance absorbed fraction in Q form.

    ``A = 2*Q_i*Q_w/(Q_i+Q_w)**2`` (M2_BRIEF.md line 44), the exact partner of
    :func:`on_resonance_TRA` under ``gamma_x = omega0/(2*Q_x)``. The
    undercoupled approximation is ``A ~ 2*Q_i/Q_w``; at the locked target
    ``Q_w = 200*Q_i`` (QW_OVER_QI) the exact value is ``400/201**2 = 0.99007%``
    versus the ``1.000%`` approximation (AT-M4-2).
    """
    Q_i = np.asarray(Q_i, dtype=float)
    Q_w = np.asarray(Q_w, dtype=float)
    return 2.0 * Q_i * Q_w / (Q_i + Q_w) ** 2


def gamma_from_Q(Q, omega0=OMEGA0_RAD):
    """Energy-decay rate ``gamma = omega0/(2*Q)`` (M2_BRIEF.md line 38)."""
    return omega0 / (2.0 * np.asarray(Q, dtype=float))


# --------------------------------------------------------------------------- #
# 2. Power chain  (M2_BRIEF.md line 52-53; M4_BRIEF_DRAFT.md line 21)
# --------------------------------------------------------------------------- #
def pj_from_pin(P_in, A=A_TARGET):
    """Absorbed junction power ``P_J = A*P_in`` (A = absorbed fraction)."""
    return np.asarray(P_in, dtype=float) * A


def v_ac_from_pj(P_J, Rn=RN_OHM):
    """AC voltage across the junction from absorbed power.

    ``V_ac = sqrt(2*P_J*Rn)`` (M2_BRIEF.md line 53). With P_J = 90 pW and
    Rn = 2 kOhm this is exactly 0.600 mV.
    """
    return np.sqrt(2.0 * np.asarray(P_J, dtype=float) * Rn)


def alpha_from_vac(V_ac):
    """Reduced drive strength ``alpha = e*V_ac/(hbar*omega)`` (M3_BRIEF.md line 34).

    Equivalent to ``V_ac / V_PHOTON`` where ``V_PHOTON = hbar*omega/e =
    h*f0/e = 0.59967 mV`` at f0 = 145 GHz (M4_BRIEF_DRAFT.md line 22).
    """
    return E_CHARGE * np.asarray(V_ac, dtype=float) / (HBAR * OMEGA0_RAD)


def alpha_from_pin(P_in, A=A_TARGET, Rn=RN_OHM):
    """Reduced drive strength from cavity input power.

    Full chain ``P_in -> P_J = A*P_in -> V_ac = sqrt(2*P_J*Rn) -> alpha``
    (M4_BRIEF_DRAFT.md line 21). At the locked reference P_in = 9 nW, A = 1%,
    Rn = 2 kOhm this returns 1.0005 (== 1.000 within the AT-M4-1 +-0.002 band).
    """
    return alpha_from_vac(v_ac_from_pj(pj_from_pin(P_in, A), Rn))


def pin_for_alpha(alpha, A=A_TARGET, Rn=RN_OHM):
    """Cavity input power required for a target drive strength (inverse chain).

    Inverts :func:`alpha_from_pin`::

        V_ac = alpha * V_PHOTON,  P_J = V_ac**2/(2*Rn),  P_in = P_J / A

    so ``P_in = alpha**2 * (hbar*omega/e)**2 / (2*A*Rn)``. Exact inverse of the
    physical chain; ``pin_for_alpha(1.0) = 8.990 nW`` (the locked 9 nW budget to
    the 0.05% photon-voltage rounding). Vectorised in ``alpha``.
    """
    alpha = np.asarray(alpha, dtype=float)
    V_ac = alpha * (HBAR * OMEGA0_RAD / E_CHARGE)
    P_J = V_ac ** 2 / (2.0 * Rn)
    return P_J / A


def i_ac_from_alpha(alpha, Omega):
    """Junction AC bias current from drive strength: ``i_ac = alpha*Omega``.

    High-frequency mapping alpha = eV_ac/hbar*omega ~ i_ac/Omega
    (M3_BRIEF.md line 34). Broadcasts ``alpha`` against ``Omega``.
    """
    return np.asarray(alpha, dtype=float) * np.asarray(Omega, dtype=float)


# --------------------------------------------------------------------------- #
# 3. Operating-point table (pure consumption of junction.json fingerprint)
# --------------------------------------------------------------------------- #
def operating_points(path_to_junction_json):
    """Build the per-filling operating-point table (M4_BRIEF_DRAFT.md Sec. 4.2).

    Reads only the ``fingerprint`` block exported by M3 (nu, alpha_opt,
    i_ac_opt, w_half, w_half_nA, Omega, Gamma) and *adds* the drive-power
    columns via the locked power chain -- no junction recomputation
    (M4_BRIEF_DRAFT.md line 65, "consume junction.json as-is").

    Power columns (M4_BRIEF_DRAFT.md line 44, ``P_in_opt = 9 nW * alpha_opt**2``)::

        P_in_opt = P_IN_REF_W * alpha_opt**2
        P_J_opt  = A_TARGET   * P_in_opt          (absorbed 1%)
        V_ac_opt = sqrt(2*P_J_opt*Rn)             (via v_ac_from_pj)

    Anchoring P_in_opt to the locked 9 nW reference makes the ratio
    ``P_in_opt/alpha_opt**2`` exactly P_IN_REF_W for every filling (AT-M4-3
    ratio identity) and the spot value at nu = -2.4 exactly 9*0.7**2 = 4.41 nW.

    Parameters
    ----------
    path_to_junction_json : str or path
        Location of the M3 export ``junction.json``.

    Returns
    -------
    list of dict
        One dict per filling (in file order) with keys:
        ``nu, alpha_opt, i_ac_opt, w_half, w_half_nA, Omega, Gamma,
        P_in_opt_nW, P_J_opt_pW, V_ac_opt_mV``.
    """
    with open(path_to_junction_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    fp = data["fingerprint"]

    nu = np.asarray(fp["nu"], dtype=float)
    alpha_opt = np.asarray(fp["alpha_opt"], dtype=float)
    i_ac_opt = np.asarray(fp["i_ac_opt"], dtype=float)
    w_half = np.asarray(fp["w_half"], dtype=float)
    w_half_nA = np.asarray(fp["w_half_nA"], dtype=float)
    Omega = np.asarray(fp["Omega"], dtype=float)
    Gamma = np.asarray(fp["Gamma"], dtype=float)

    # Locked power chain, vectorised over the filling axis.
    P_in_opt = P_IN_REF_W * alpha_opt ** 2          # W
    P_J_opt = pj_from_pin(P_in_opt, A_TARGET)       # W  (= A_TARGET * P_in_opt)
    V_ac_opt = v_ac_from_pj(P_J_opt, RN_OHM)        # V

    rows = []
    for k in range(nu.size):
        rows.append(
            {
                "nu": float(nu[k]),
                "alpha_opt": float(alpha_opt[k]),
                "i_ac_opt": float(i_ac_opt[k]),
                "w_half": float(w_half[k]),
                "w_half_nA": float(w_half_nA[k]),
                "Omega": float(Omega[k]),
                "Gamma": float(Gamma[k]),
                "P_in_opt_nW": float(P_in_opt[k] / 1e-9),
                "P_J_opt_pW": float(P_J_opt[k] / 1e-12),
                "V_ac_opt_mV": float(V_ac_opt[k] / 1e-3),
            }
        )
    return rows
