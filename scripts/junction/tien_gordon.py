"""Tien-Gordon above-gap module (M3_BRIEF.md Sec. 4.4 / deliverable 4).

Voltage-driven photon-assisted quasiparticle transport across the driven
TBG Josephson junction, plus the Owen-Scalapino linearized pair-breaking
kinetics that couple the absorbed cavity power to the gap.

Gap value
---------
`material.json` from M1 does not exist yet, so Delta0 is taken from the locked
contract (M3_BRIEF.md Sec. 1): the 145 GHz drive equals 2*Delta0/h at Tc = 2 K,
hence Delta0 = h * 72.5 GHz ~= 0.2998 meV and hbar*omega_drive = 2*Delta0.
These are **contract values, to be superseded by M1 material.json**.

Unit convention
----------------
Energies are in electron-volts (eV); voltages in volts (V); resistances in ohm;
currents in ampere. Because 1 eV = e * (1 V), the bias energy in eV is
numerically equal to the bias voltage in V, and a photon energy hbar*omega in
eV adds directly to a voltage in V (hbar*omega / e in volts == hbar*omega in eV
numerically). All internal integrals therefore drop the electron charge:
    I(V) = (1/Rn) * integral dx rho(x) rho(x+u) [f(x) - f(x+u)],   u = V.

Pair-breaking reference units
-----------------------------
eta_abs, tau_rec, P_abs and n0 are FREE PARAMETERS (M4 supplies P_abs from the
photonics module). Reference defaults: eta_abs = 1, n0 = 1, tau_rec = 1. In
these units P_abs carries units of energy (eV) so that eta_abs*P_abs/Delta has
units of n0/tau_rec; the steady-state quasiparticle density is
    n_qp* = sqrt(eta_abs * P_abs * n0 * tau_rec / Delta).
"""
import numpy as np
from scipy.special import jv

# ---------------------------------------------------------------- constants
H_EV_S = 4.135667696e-15      # Planck constant [eV*s]
HBAR_EV_S = H_EV_S / (2.0 * np.pi)
KB_EV_K = 8.617333262e-5      # Boltzmann constant [eV/K]
E_CHARGE = 1.602176634e-19    # elementary charge [C]

# Contract gap (to be superseded by M1 material.json)
DELTA0 = H_EV_S * 72.5e9      # eV  (~0.2998 meV)
HW_DRIVE = H_EV_S * 145e9     # eV  == 2 * DELTA0  (145 GHz drive photon energy)

# Dynes broadening: fraction of Delta used to regularize the BCS square-root
# edge. 3e-3 keeps sub-gap leakage ~1e-3 of the above-gap current while the
# gap-edge peak stays resolvable on the default energy grid.
DYNES_DEFAULT = 3.0e-3


# ---------------------------------------------------------------- BCS blocks
def bcs_dos(E, Delta, dynes=DYNES_DEFAULT):
    """Normalized BCS quasiparticle DOS N(E)/N(0) = |E|/sqrt(E^2 - Delta^2).

    The square-root edge is regularized with the Dynes prescription
    E -> E + i*gamma (gamma = dynes * Delta), which yields a finite, smooth DOS
    that tends to 1 for |E| >> Delta and to gamma/sqrt(gamma^2+Delta^2) in the
    gap. Vectorized over E.
    """
    E = np.asarray(E, float)
    g = dynes * Delta
    z = E + 1j * g
    return np.abs(np.real(z / np.sqrt(z * z - Delta * Delta)))


def fermi(E, kT):
    """Fermi-Dirac occupation f(E) = 1/(exp(E/kT)+1); kT<=0 gives the step."""
    E = np.asarray(E, float)
    if kT <= 0.0:
        return np.where(E < 0.0, 1.0, np.where(E > 0.0, 0.0, 0.5))
    return 1.0 / (np.exp(np.clip(E / kT, -500.0, 500.0)) + 1.0)


def iqp0_bcs(V, Delta, T, Rn, dynes=DYNES_DEFAULT,
             d_target=5.0e-4, n_grid_max=200000, margin_kT=30.0,
             margin_delta=4.0):
    """BCS (symmetric SIS) quasiparticle I-V, semiconductor model.

        I(V) = (1/Rn) * integral dE rho(E) rho(E+eV) [f(E) - f(E+eV)]

    with the Dynes-regularized DOS `bcs_dos` and Fermi factors at temperature T.
    Sharp current onset at eV = 2*Delta for T -> 0; ohmic asymptote I -> V/Rn
    for eV >> 2*Delta. Odd in V. Vectorized over V.

    Numerics: uniform energy grid with target spacing `d_target*Delta`, capped
    at `n_grid_max` points, spanning the Fermi-difference window plus
    max(margin_kT*kT, margin_delta*Delta) on each relevant side.
    """
    scalar = np.ndim(V) == 0
    V = np.atleast_1d(np.asarray(V, float))
    sign = np.sign(V)
    u = np.abs(V)                       # bias energy (eV) == |V| numerically
    kT = KB_EV_K * T
    margin = max(margin_kT * kT, margin_delta * Delta)
    umax = float(np.max(u)) if u.size else 0.0
    span = umax + 2.0 * margin
    n = int(min(n_grid_max, max(2000, span / (d_target * Delta))))
    # window for a positive bias u >= 0: contributing region is [-u, 0] widened
    # by the thermal/gap margin on both ends.
    x = np.linspace(-(umax + margin), margin, n)
    rho_x = bcs_dos(x, Delta, dynes)
    f_x = fermi(x, kT)
    xu = x[:, None] + u[None, :]        # (n_grid, n_V)
    integ = rho_x[:, None] * bcs_dos(xu, Delta, dynes) * (f_x[:, None] - fermi(xu, kT))
    I = np.trapezoid(integ, x=x, axis=0) / Rn
    I = I * sign
    return float(I[0]) if scalar else I


# ---------------------------------------------------------------- Tien-Gordon
def default_nmax(alpha):
    """Contract truncation for the photon-sideband sum: ceil(3*alpha) + 10."""
    return int(np.ceil(3.0 * abs(alpha))) + 10


def bessel_weights(alpha, nmax=None):
    """Sideband weights w_n = J_n(alpha)^2 for n in [-nmax, nmax].

    Returns (n, w). sum_n w_n -> 1 (closure of the Bessel functions); the
    truncation error at the contract `default_nmax` is far below 1e-10 for the
    alpha values used here. `nmax` is exposed so callers can enlarge it.
    """
    if nmax is None:
        nmax = default_nmax(alpha)
    n = np.arange(-nmax, nmax + 1)
    w = jv(n, alpha) ** 2
    return n, w


def tien_gordon_iv(V0, alpha, Delta, T, Rn, hw=HW_DRIVE, nmax=None,
                   weight_floor=1e-12, **iqp_kw):
    """Photon-assisted quasiparticle I-V (Tien-Gordon).

        I_dc(V0) = sum_n J_n(alpha)^2 * iqp0_bcs(V0 + n*hbar*omega/e)

    Sidebands |n| <= nmax (default `default_nmax(alpha)` = ceil(3*alpha)+10,
    which is exposed and overridable). `hw` is the photon energy in eV; the
    voltage shift n*hbar*omega/e in volts equals n*hw numerically. alpha = 0
    recovers iqp0_bcs exactly. Sidebands with J_n(alpha)^2 < weight_floor are
    skipped (their contribution is below machine precision for the currents
    here); pass weight_floor=0 to force every |n| <= nmax to be evaluated.
    Vectorized over V0.
    """
    if nmax is None:
        nmax = default_nmax(alpha)
    scalar = np.ndim(V0) == 0
    V0 = np.atleast_1d(np.asarray(V0, float))
    n = np.arange(-nmax, nmax + 1)
    w = jv(n, alpha) ** 2
    I = np.zeros_like(V0)
    for nn, wn in zip(n, w):
        if wn < weight_floor:
            continue
        I += wn * iqp0_bcs(V0 + nn * hw, Delta, T, Rn, **iqp_kw)
    return float(I[0]) if scalar else I


# ---------------------------------------------- Owen-Scalapino pair-breaking
def delta_of_nqp(n_qp, n0, Delta0):
    """Linearized gap suppression Delta(n_qp) = Delta0*(1 - n_qp/(4*n0))."""
    return Delta0 * (1.0 - n_qp / (4.0 * n0))


def steady_state_delta(eta_abs, P_abs, tau_rec, n0, Delta0,
                       tol=1e-12, max_iter=500, relax=0.5, return_nqp=False):
    """Self-consistent steady-state gap under absorbed power P_abs.

    Solves dn_qp/dtau = eta_abs*P_abs/Delta - n_qp^2/(n0*tau_rec) = 0 together
    with Delta(n_qp) = Delta0*(1 - n_qp/(4*n0)) by fixed-point iteration of
        n_qp* = sqrt(eta_abs * P_abs * n0 * tau_rec / Delta),
        Delta  = Delta0 * (1 - n_qp*/(4*n0)),
    under-relaxed by `relax` for stability. Vectorized over P_abs.

    P_abs = 0 -> n_qp = 0 -> Delta = Delta0. Delta decreases monotonically with
    P_abs. Returns Delta (or (Delta, n_qp) if return_nqp). The linearized model
    is only physical while the suppression stays modest (n_qp < 4*n0); Delta is
    clipped at 0 and a note left for the caller if that bound is hit.
    """
    P_abs = np.asarray(P_abs, float)
    scalar = P_abs.ndim == 0
    P = np.atleast_1d(P_abs)
    Delta = np.full_like(P, float(Delta0))
    n_qp = np.zeros_like(P)
    K = eta_abs * P * n0 * tau_rec
    for _ in range(max_iter):
        Delta_safe = np.maximum(Delta, 1e-30 * Delta0)
        n_new = np.sqrt(np.maximum(K / Delta_safe, 0.0))
        d_new = np.clip(delta_of_nqp(n_new, n0, Delta0), 0.0, Delta0)
        if np.max(np.abs(d_new - Delta)) < tol * Delta0:
            Delta, n_qp = d_new, n_new
            break
        n_qp = n_new
        Delta = (1.0 - relax) * Delta + relax * d_new
    if scalar:
        Delta, n_qp = float(Delta[0]), float(n_qp[0])
    if return_nqp:
        return Delta, n_qp
    return Delta


def evolve_nqp(eta_abs, P_abs, tau_rec, n0, Delta0, n_qp0=0.0,
               dtau=1e-2, n_steps=20000):
    """Fixed-step RK4 transient of the pair-breaking ODE.

        dn_qp/dtau = eta_abs*P_abs/Delta(n_qp) - n_qp^2/(n0*tau_rec),
        Delta(n_qp) = Delta0*(1 - n_qp/(4*n0)).

    Deterministic (no RNG). Returns (tau, n_qp, Delta) time series of length
    n_steps+1. The long-time limit reproduces `steady_state_delta`.
    """
    def rhs(n):
        d = max(delta_of_nqp(n, n0, Delta0), 1e-30 * Delta0)
        return eta_abs * P_abs / d - n * n / (n0 * tau_rec)

    tau = np.empty(n_steps + 1)
    nq = np.empty(n_steps + 1)
    tau[0], nq[0] = 0.0, float(n_qp0)
    n = float(n_qp0)
    t = 0.0
    for i in range(1, n_steps + 1):
        k1 = rhs(n)
        k2 = rhs(n + 0.5 * dtau * k1)
        k3 = rhs(n + 0.5 * dtau * k2)
        k4 = rhs(n + dtau * k3)
        n += dtau / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        n = max(n, 0.0)
        t += dtau
        tau[i], nq[i] = t, n
    Delta = delta_of_nqp(nq, n0, Delta0)
    return tau, nq, Delta
