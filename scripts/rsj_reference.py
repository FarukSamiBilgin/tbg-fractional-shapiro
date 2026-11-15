"""
M3 reference implementation — driven RSJ dynamics of a TBG Josephson junction.
Physics contract v0.1:
  Ic = 30 nA, Rn = 2 kOhm, C ~ 0.1 fF  ->  beta_c ~ 0.04 (overdamped, RSJ valid)
  f_c = 2e*Ic*Rn/h = 29.0 GHz ; drive f = 145 GHz  ->  Omega = omega/omega_c = 5.0
Dimensionless RSJ:
  dphi/dtau = i_dc + i_ac*cos(Omega*tau) - sum_m c_m sin(m*phi)
  tau = omega_c t,  i = I/Ic,  v = <dphi/dtau> = V/(Ic*Rn),  Shapiro steps at v = (n/m)*Omega
High-frequency mapping: alpha = e*V_ac/(hbar*omega) ~ i_ac/Omega ; step-0 width = 2|J0(alpha)|.
"""

import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy.special import jv
    def J(n, x):
        return jv(n, np.asarray(x, float))
except Exception:  # fallback: integral representation
    def J(n, x):
        th = np.linspace(0.0, np.pi, 2001)
        x = np.atleast_1d(np.asarray(x, float))
        out = np.trapz(np.cos(n * th[None, :] - x[:, None] * np.sin(th)[None, :]),
                       th, axis=1) / np.pi
        return out

OMEGA = 5.0

# ---------------------------------------------------------------- CPR helpers
def cpr(phi, c):
    f = np.zeros_like(phi)
    for m, cm in enumerate(c, start=1):
        if cm != 0.0:
            f += cm * np.sin(m * phi)
    return f

def normalize_cpr(c_raw):
    """Rescale harmonic coefficients so that max_phi I(phi) = 1 (current in units of true Ic)."""
    c_raw = np.asarray(c_raw, float)
    phi = np.linspace(0.0, 2.0 * np.pi, 40001)
    fmax = cpr(phi, c_raw).max()
    return c_raw / fmax, fmax

C_SIN, _ = normalize_cpr([1.0])            # sinusoidal benchmark CPR
C_SKEW, SKEW_MAX = normalize_cpr([1.0, -0.25])  # forward-skewed CPR (stand-in for high-transparency TBG link)

# ---------------------------------------------------------------- integrator
def run_rsj(i_dc, i_ac, c, omega=OMEGA,
            n_transient=150, n_avg=300, spp=60, phi0=None):
    """Vectorized fixed-step RK4 over flat arrays i_dc, i_ac (equal length).
    Returns time-averaged reduced voltage v = <dphi/dtau>."""
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

def step_width(v_row, idc_grid, v_target, tol):
    """Width of a Shapiro plateau: measure of i_dc points locked to v_target."""
    mask = np.abs(v_row - v_target) < tol
    if not mask.any():
        return 0.0
    di = idc_grid[1] - idc_grid[0]
    return mask.sum() * di

# ---------------------------------------------------------------- stages
results = {"contract": {
    "Omega": OMEGA, "Ic_nA": 30.0, "Rn_kOhm": 2.0, "f_c_GHz": 29.0,
    "f_drive_GHz": 145.0, "V_step_uV": 300.0, "IcRn_uV": 60.0,
    "beta_c": 0.04, "Gamma_noise_100mK": 0.140, "EJ_over_kB_K": 0.715,
    "cpr_sin": list(C_SIN), "cpr_skew": list(C_SKEW),
    "skew_renorm_factor": SKEW_MAX,
}}

t0 = time.time()

# ---- Stage A: benchmark staircases, sinusoidal CPR --------------------------
idcA = np.arange(0.0, 13.0 + 1e-9, 0.05)
iacA_vals = [0.0, 4.0, 9.2]           # 9.2 = Omega * 1.841 (first max of J1)
vA = grid_run(idcA, np.array(iacA_vals), C_SIN)
results["stageA"] = {"i_dc": idcA.tolist(), "i_ac_values": iacA_vals,
                     "v": [row.tolist() for row in vA]}
print(f"[A] staircases done  t={time.time()-t0:.1f}s")

# ---- Stage C: step-0 width vs i_ac -> Bessel J0 check -----------------------
idcC = np.arange(0.0, 1.4 + 1e-9, 0.02)
iacC = np.arange(0.0, 14.0 + 1e-9, 0.35)
vC = grid_run(idcC, iacC, C_SIN)
tolC = 0.025
w0 = np.array([2.0 * step_width(vC[k], idcC, 0.0, tolC) for k in range(len(iacC))])
w0_theory = 2.0 * np.abs(J(0, iacC / OMEGA))
# numeric location of the first J0 zero: minimum of w0 near predicted 12.0
k_lo = np.searchsorted(iacC, 9.0)
k_zero = k_lo + int(np.argmin(w0[k_lo:]))
results["stageC"] = {"i_ac": iacC.tolist(), "w0_numeric": w0.tolist(),
                     "w0_theory": w0_theory.tolist(),
                     "iac_J0zero_numeric": float(iacC[k_zero]),
                     "iac_J0zero_theory": 2.4048 * OMEGA}
print(f"[C] J0 check done: numeric zero at i_ac={iacC[k_zero]:.2f} "
      f"(theory 12.02)  t={time.time()-t0:.1f}s")

# ---- Stage B: fractional (half-integer) step map, skew vs sin ---------------
idcB = np.arange(1.0, 8.0 + 1e-9, 0.025)
iacB = np.arange(0.0, 12.0 + 1e-9, 0.5)
tolB = 0.025
half_v = 0.5 * OMEGA
vB_skew = grid_run(idcB, iacB, C_SKEW)
vB_sin = grid_run(idcB, iacB, C_SIN)
w_half_skew = np.array([step_width(vB_skew[k], idcB, half_v, tolB)
                        for k in range(len(iacB))])
w_half_sin = np.array([step_width(vB_sin[k], idcB, half_v, tolB)
                       for k in range(len(iacB))])
w_int1_skew = np.array([step_width(vB_skew[k], idcB, OMEGA, tolB)
                        for k in range(len(iacB))])
k_best = int(np.argmax(w_half_skew))
results["stageB"] = {"i_ac": iacB.tolist(),
                     "w_half_skew": w_half_skew.tolist(),
                     "w_half_sin": w_half_sin.tolist(),
                     "w_int1_skew": w_int1_skew.tolist(),
                     "iac_best_half": float(iacB[k_best]),
                     "w_half_max": float(w_half_skew[k_best])}
print(f"[B] fractional map done: max half-step width {w_half_skew[k_best]:.3f} "
      f"at i_ac={iacB[k_best]:.1f}; sin control max {w_half_sin.max():.3f}  "
      f"t={time.time()-t0:.1f}s")

# staircase at the optimal drive for the figure
v_zoom_skew = run_rsj(idcB, np.full_like(idcB, iacB[k_best]), C_SKEW)
v_zoom_sin = run_rsj(idcB, np.full_like(idcB, iacB[k_best]), C_SIN)

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

# Fig 1 — benchmark staircases
fig, ax = plt.subplots(figsize=(6.4, 4.6))
for row, ia in zip(vA, iacA_vals):
    ax.plot(idcA, row / OMEGA, lw=1.4, label=f"$i_{{ac}}={ia:g}$")
for n in (1, 2):
    ax.axhline(n, ls="--", c="gray", lw=0.8)
ax.set_xlabel(r"DC bias  $i_{dc}=I_{dc}/I_c$")
ax.set_ylabel(r"$\langle v\rangle/\Omega = 2e\langle V\rangle/\hbar\omega$")
ax.set_title(r"M3 benchmark: integer Shapiro steps (sinusoidal CPR, $\Omega=5$)")
ax.legend()
fig.tight_layout()
fig.savefig("/home/claude/fig_m3_benchmark_shapiro.png", dpi=160)

# Fig 2 — fractional steps (the experimental fingerprint)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.4))
a1.plot(idcB, v_zoom_sin / OMEGA, lw=1.2, c="C0",
        label="sinusoidal CPR")
a1.plot(idcB, v_zoom_skew / OMEGA, lw=1.2, c="C3",
        label=r"skewed CPR ($c_2/c_1=-0.25$)")
for y in (0.5, 1.0, 1.5):
    a1.axhline(y, ls="--", c="gray", lw=0.8)
a1.set_xlim(idcB[0], idcB[-1]); a1.set_ylim(0, 1.7)
a1.set_xlabel(r"$i_{dc}$"); a1.set_ylabel(r"$\langle v\rangle/\Omega$")
a1.set_title(f"staircase at $i_{{ac}}={iacB[k_best]:g}$")
a1.legend(fontsize=9)
a2.plot(iacB, w_half_skew, "o-", c="C3", ms=4, label="skewed CPR")
a2.plot(iacB, w_half_sin, "s-", c="C0", ms=4, label="sinusoidal (control)")
a2.set_xlabel(r"drive amplitude $i_{ac}$")
a2.set_ylabel(r"half-step width $\Delta i_{1/2}$")
a2.set_title(r"fractional step at $v=\Omega/2$")
a2.legend(fontsize=9)
fig.suptitle("Fractional Shapiro steps from CPR skewness — fingerprint prediction")
fig.tight_layout()
fig.savefig("/home/claude/fig_m3_fractional_steps.png", dpi=160)

# Fig 3 — Bessel validation
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.plot(iacC, w0_theory, "-", c="k", lw=1.5,
        label=r"$2|J_0(i_{ac}/\Omega)|$ (Tien–Gordon limit)")
ax.plot(iacC, w0, "o", c="C1", ms=5, label="RSJ numerics")
ax.axvline(2.4048 * OMEGA, ls=":", c="gray")
ax.text(2.4048 * OMEGA + 0.15, 1.6, r"$J_0$ zero: $i_{ac}=12.0$", fontsize=9)
ax.set_xlabel(r"drive amplitude $i_{ac}$")
ax.set_ylabel(r"step-0 width $\Delta i_0$ (units of $I_c$)")
ax.set_title(r"Acceptance test: critical-current suppression follows $J_0$")
ax.legend()
fig.tight_layout()
fig.savefig("/home/claude/fig_m3_bessel_check.png", dpi=160)

with open("/home/claude/junction_reference.json", "w") as fh:
    json.dump(results, fh, indent=1)

print(f"ALL DONE t={time.time()-t0:.1f}s")
print("key numbers ->",
      f"J0zero: {results['stageC']['iac_J0zero_numeric']:.2f}",
      f"| half-step max: {results['stageB']['w_half_max']:.3f} at i_ac={results['stageB']['iac_best_half']:.1f}",
      f"| sin half-step floor: {max(results['stageB']['w_half_sin']):.3f}")
