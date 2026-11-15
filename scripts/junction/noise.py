"""Langevin RSJ integrator (M3_BRIEF.md Sec. 2 and Sec. 4, deliverable 2).

    dphi/dtau = i_dc + i_ac cos(Omega tau) - sum_m c_m sin(m phi) + eta(tau)
    <eta(tau) eta(tau')> = 2 Gamma delta(tau - tau'),   Gamma = k_B T / E_J

Stochastic Heun with additive noise (Ito and Stratonovich coincide here).
Fixed step; brief Sec. 5 requires spp >= 100 for noisy runs — enforced.
"""
import numpy as np

from .cpr import cpr
from .rsj import OMEGA


def run_rsj_langevin(i_dc, i_ac, c, Gamma, omega=OMEGA,
                     n_transient=150, n_avg=300, spp=100,
                     seed=None, rng=None, phi0=None):
    """Vectorized stochastic-Heun Langevin run over flat bias arrays.

    Returns time-averaged reduced voltage v = <dphi/dtau> per instance.
    Pass either `seed` (logged by callers) or an existing `rng`.
    """
    if spp < 100:
        raise ValueError("brief Sec. 5: fixed step with spp >= 100 for noisy runs")
    i_dc = np.asarray(i_dc, float)
    i_ac = np.asarray(i_ac, float)
    if rng is None:
        rng = np.random.default_rng(seed)
    T = 2.0 * np.pi / omega
    dt = T / spp
    sig = np.sqrt(2.0 * Gamma * dt)
    phi = np.zeros_like(i_dc) if phi0 is None else phi0.astype(float).copy()

    def drift(tau, ph):
        return i_dc + i_ac * np.cos(omega * tau) - cpr(ph, c)

    n1 = int(n_transient * spp)
    n2 = int(n_avg * spp)
    tau = 0.0
    phi_a, tau_a = phi.copy(), tau
    for step in range(n1 + n2):
        if step == n1:
            phi_a, tau_a = phi.copy(), tau
        xi = sig * rng.standard_normal(phi.shape)
        f1 = drift(tau, phi)
        pred = phi + dt * f1 + xi
        f2 = drift(tau + dt, pred)
        phi = phi + 0.5 * dt * (f1 + f2) + xi
        tau += dt

    return (phi - phi_a) / (tau - tau_a)


def ensemble_staircase(idc_grid, i_ac, c, Gamma, n_real=20, seed=0, **kw):
    """n_real independent noisy staircases in one vectorized run.

    Returns v with shape (n_real, n_idc); ensemble mean/std are up to the caller.
    """
    idc_grid = np.asarray(idc_grid, float)
    idc = np.tile(idc_grid, n_real)
    iac = np.full_like(idc, float(i_ac))
    v = run_rsj_langevin(idc, iac, c, Gamma, seed=seed, **kw)
    return v.reshape(n_real, len(idc_grid))
