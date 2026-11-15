"""Deterministic overdamped RSJ integrator (M3_BRIEF.md Sec. 2).

Dimensionless model (tau = omega_c t, i = I/Ic, v = <dphi/dtau> = V/(Ic Rn)):
    dphi/dtau = i_dc + i_ac cos(Omega tau) - sum_m c_m sin(m phi)

Reference numerics: vectorized fixed-step RK4, spp = 60 steps/period,
150 T transient, 300 T averaging. Shapiro steps at v = (n/m) Omega.
"""
import numpy as np

from .cpr import cpr

OMEGA = 5.0


def run_rsj(i_dc, i_ac, c, omega=OMEGA,
            n_transient=150, n_avg=300, spp=60, phi0=None):
    """Vectorized fixed-step RK4 over flat arrays i_dc, i_ac (equal length).

    Returns time-averaged reduced voltage v = <dphi/dtau>.
    """
    i_dc = np.asarray(i_dc, float)
    i_ac = np.asarray(i_ac, float)
    T = 2.0 * np.pi / omega
    dt = T / spp
    phi = np.zeros_like(i_dc) if phi0 is None else phi0.astype(float).copy()

    def rhs(tau, ph):
        return i_dc + i_ac * np.cos(omega * tau) - cpr(ph, c)

    tau = 0.0
    for _ in range(int(n_transient * spp)):
        k1 = rhs(tau, phi)
        k2 = rhs(tau + 0.5 * dt, phi + 0.5 * dt * k1)
        k3 = rhs(tau + 0.5 * dt, phi + 0.5 * dt * k2)
        k4 = rhs(tau + dt, phi + dt * k3)
        phi += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        tau += dt

    phi_a, tau_a = phi.copy(), tau
    for _ in range(int(n_avg * spp)):
        k1 = rhs(tau, phi)
        k2 = rhs(tau + 0.5 * dt, phi + 0.5 * dt * k1)
        k3 = rhs(tau + 0.5 * dt, phi + 0.5 * dt * k2)
        k4 = rhs(tau + dt, phi + dt * k3)
        phi += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        tau += dt

    return (phi - phi_a) / (tau - tau_a)


def grid_run(idc_grid, iac_grid, c, **kw):
    """Run all (i_ac, i_dc) combinations at once; return v with shape (n_iac, n_idc)."""
    IAC, IDC = np.meshgrid(iac_grid, idc_grid, indexing="ij")
    v = run_rsj(IDC.ravel(), IAC.ravel(), c, **kw)
    return v.reshape(IAC.shape)
