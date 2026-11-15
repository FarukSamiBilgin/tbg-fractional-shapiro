"""
M1 reference — Bistritzer–MacDonald continuum model of magic-angle TBG.

Locks for the paper:
  * measured magic angle for the parameter set (bandwidth minimum),
  * flat-band width W and its sensitivity to twist disorder (+-0.05 deg),
  * Beenakker short-junction CPR harmonics (feeds M3 fingerprint),
  * superconducting dome parametrization Delta(nu, T),
  * writes material.json  — the file module M3 is waiting for.

Single valley, small-angle BM Hamiltonian:
  H = [[ hbar vF sigma.(k+g)        ,  T(g'-g)                  ],
       [ T^dagger                   ,  hbar vF sigma.(k+g+q0)   ]]
  T_j = wAA*s0 + wAB*(cos(2pi j/3) sx + sin(2pi j/3) sy),  j = 0,1,2
Momentum lattice g = m*b1 + n*b2, b1 = q1-q0, b2 = q2-q0.
"""

import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KB_MEV_K = 0.08617          # meV / K
A_G = 0.246                 # graphene lattice constant, nm
KD = 4.0 * np.pi / (3.0 * A_G)   # Dirac momentum, 1/nm

S0 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)

WAB = 110.0                 # meV (contract)
RATIO = 0.8                 # wAA/wAB corrugation (contract)
WAA = RATIO * WAB

def geometry(theta_deg):
    th = np.deg2rad(theta_deg)
    ktheta = 2.0 * KD * np.sin(th / 2.0)
    q0 = ktheta * np.array([0.0, -1.0])
    q1 = ktheta * np.array([np.sqrt(3) / 2.0, 0.5])
    q2 = ktheta * np.array([-np.sqrt(3) / 2.0, 0.5])
    b1, b2 = q1 - q0, q2 - q0
    return ktheta, (q0, q1, q2), b1, b2

def basis(N):
    idx = np.arange(-N, N + 1)
    mm, nn = np.meshgrid(idx, idx, indexing="ij")
    return np.stack([mm.ravel(), nn.ravel()], axis=1)      # (Ng, 2) integers

T_MATS = [WAA * S0 + WAB * (np.cos(2 * np.pi * j / 3) * SX
                            + np.sin(2 * np.pi * j / 3) * SY)
          for j in range(3)]
# layer-1 site g couples to layer-2 site g' with g'-g = {0, b1, b2} -> T0,T1,T2
HOPS = [(0, 0), (1, 0), (0, 1)]

def hamiltonian(k, mn, hbv, ktheta, qs, b1, b2):
    Ng = len(mn)
    gs = mn[:, 0][:, None] * b1[None, :] + mn[:, 1][:, None] * b2[None, :]
    dim = 4 * Ng                    # 2 layers x 2 sublattices x Ng
    H = np.zeros((dim, dim), dtype=complex)
    key = {(int(m), int(n)): i for i, (m, n) in enumerate(mn)}
    o = 2 * Ng
    for i in range(Ng):
        p1 = k + gs[i]                       # layer 1 momentum
        p2 = k + gs[i] + qs[0]               # layer 2 momentum
        H[2*i:2*i+2, 2*i:2*i+2] = hbv * (p1[0] * SX + p1[1] * SY)
        H[o+2*i:o+2*i+2, o+2*i:o+2*i+2] = hbv * (p2[0] * SX + p2[1] * SY)
        for j, (dm, dn) in enumerate(HOPS):
            tgt = key.get((int(mn[i, 0] + dm), int(mn[i, 1] + dn)))
            if tgt is not None:
                blk = T_MATS[j]
                H[o+2*tgt:o+2*tgt+2, 2*i:2*i+2] = blk           # layer2 <- layer1
                H[2*i:2*i+2, o+2*tgt:o+2*tgt+2] = blk.conj().T  # h.c.
    return H

def flat_band_width(theta_deg, hbv, N=3, grid=7):
    ktheta, qs, b1, b2 = geometry(theta_deg)
    mn = basis(N)
    Ng = len(mn)
    lo, hi = [], []
    for i in range(grid):
        for j in range(grid):
            k = (i / grid) * b1 + (j / grid) * b2
            ev = np.linalg.eigvalsh(hamiltonian(k, mn, hbv, ktheta, qs, b1, b2))
            lo.append(ev[2 * Ng - 1])        # lower flat band
            hi.append(ev[2 * Ng])            # upper flat band
    lo, hi = np.array(lo), np.array(hi)
    return hi.max() - lo.min(), (lo.min(), lo.max(), hi.min(), hi.max())

def band_path(theta_deg, hbv, N=4, npts=30, nbands=10):
    ktheta, qs, b1, b2 = geometry(theta_deg)
    mn = basis(N)
    Ng = len(mn)
    K = np.array([0.0, 0.0])
    Kp = -qs[0]                                    # (0, +ktheta)
    G = np.array([np.sqrt(3) * ktheta / 2.0, ktheta / 2.0])
    M = 0.5 * (K + Kp)
    segs = [(K, G), (G, M), (M, Kp)]
    ks = []
    for A, B in segs:
        for t in np.linspace(0, 1, npts, endpoint=False):
            ks.append(A + t * (B - A))
    ks.append(Kp)
    ks = np.array(ks)
    x = np.concatenate([[0.0],
                        np.cumsum(np.linalg.norm(np.diff(ks, axis=0), axis=1))])
    half = nbands // 2
    bands = np.empty((len(ks), 2 * half))
    for i, k in enumerate(ks):
        ev = np.linalg.eigvalsh(hamiltonian(k, mn, hbv, ktheta, qs, b1, b2))
        bands[i] = ev[2 * Ng - half: 2 * Ng + half]
    ticks = [0.0]
    for A, B in segs:
        ticks.append(ticks[-1] + np.linalg.norm(B - A))
    return x, bands, ticks, [r"$K_m$", r"$\Gamma_m$", r"$M_m$", r"$K'_m$"]

# ------------------------------------------------ magic-angle scan ----------
t0 = time.time()
thetas = np.arange(0.90, 1.201, 0.025)
hbv_candidates = [658.0, 620.0, 601.0]           # meV*nm  (vF = 1.00/0.94/0.91e6)
scan = {}
print("bandwidth W(theta) [meV], N=3, 7x7 BZ grid")
for hbv in hbv_candidates:
    W = np.array([flat_band_width(th, hbv)[0] for th in thetas])
    scan[hbv] = W
    th_m = thetas[np.argmin(W)]
    print(f"  hbar*vF={hbv:5.0f}: magic angle {th_m:.3f} deg, "
          f"W_min={W.min():.2f} meV   (t={time.time()-t0:.0f}s)")

# pick the candidate whose bandwidth minimum lands closest to 1.05 deg
best_hbv = min(hbv_candidates,
               key=lambda h: abs(thetas[np.argmin(scan[h])] - 1.05))
W_best = scan[best_hbv]
theta_magic = float(thetas[np.argmin(W_best)])

# refined numbers at the magic angle, convergence and disorder sensitivity
Wm_N3, _ = flat_band_width(theta_magic, best_hbv, N=3)
Wm_N4, _ = flat_band_width(theta_magic, best_hbv, N=4)
Wp, _ = flat_band_width(theta_magic + 0.05, best_hbv, N=3)
Wq, _ = flat_band_width(theta_magic - 0.05, best_hbv, N=3)
print(f"\nLOCKED: hbar*vF={best_hbv:.0f} meV*nm (vF={best_hbv/658:.2f}e6 m/s), "
      f"theta_magic={theta_magic:.3f} deg")
print(f"W(magic): N=3 {Wm_N3:.2f} / N=4 {Wm_N4:.2f} meV (convergence)")
print(f"twist disorder +-0.05deg -> W = {Wq:.2f} / {Wm_N3:.2f} / {Wp:.2f} meV")

x, bands, ticks, labels = band_path(theta_magic, best_hbv)

# ------------------------------------------------ CPR harmonics -------------
def beenakker_harmonics(tau, Delta_meV=0.304, T_K=0.1, nH=5):
    phi = np.linspace(0, 2 * np.pi, 8192, endpoint=False)
    EA = Delta_meV * np.sqrt(1.0 - tau * np.sin(phi / 2.0) ** 2)
    I = tau * np.sin(phi) / np.sqrt(1.0 - tau * np.sin(phi / 2.0) ** 2)
    I *= np.tanh(EA / (2.0 * KB_MEV_K * T_K))
    c = [2.0 * np.mean(I * np.sin((m + 1) * phi)) for m in range(nH)]
    c = np.array(c)
    # normalize so that max_phi sum c_m sin(m phi) = 1  (M3 convention)
    f = sum(cm * np.sin((m + 1) * phi) for m, cm in enumerate(c))
    return c / f.max()

taus = [0.5, 0.8, 0.95]
cpr = {str(t): beenakker_harmonics(t).tolist() for t in taus}
for t in taus:
    c = cpr[str(t)]
    print(f"tau={t}: c2/c1={c[1]/c[0]:+.3f}  c3/c1={c[2]/c[0]:+.3f}")

# ------------------------------------------------ SC dome + nu table --------
TC_MAX, NU_OPT, NU_HW = 2.0, -2.4, 0.5
def Tc_of_nu(nu):
    z = 1.0 - ((nu - NU_OPT) / NU_HW) ** 2
    return TC_MAX * max(0.0, z)
def Delta_of(Tc, T=0.1):
    if Tc <= T:
        return 0.0
    return 1.764 * KB_MEV_K * Tc * np.tanh(1.74 * np.sqrt(Tc / T - 1.0))

nus = np.round(np.arange(-2.9, -1.899, 0.1), 2)
table = []
D0 = Delta_of(TC_MAX)
for nu in nus:
    Tc = Tc_of_nu(nu)
    D = Delta_of(Tc)
    table.append({"nu": float(nu), "Tc_K": round(Tc, 4),
                  "Delta_meV": round(D, 4),
                  "Ic_nA": round(30.0 * D / D0, 2)})

# ------------------------------------------------ material.json -------------
material = {
    "contract_version": "0.2",
    "bm_params": {"theta_design_deg": theta_magic,
                  "hbar_vF_meV_nm": best_hbv,
                  "vF_1e6_m_s": round(best_hbv / 658.0, 3),
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
    "nu_table": table,
    "caveats": ["pairing mechanism not modeled: Delta,Tc,Ic phenomenological",
                "Rn(nu) held fixed at 2 kOhm (experimental input needed)",
                "lattice relaxation beyond wAA/wAB ratio not included"],
}
with open("/home/claude/material.json", "w") as fh:
    json.dump(material, fh, indent=1)

# ------------------------------------------------ figure --------------------
plt.rcParams.update({"font.size": 10.5, "axes.grid": True, "grid.alpha": 0.3})
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(12.6, 4.2))

nb = bands.shape[1]
for b in range(nb):
    col, lw = ("C3", 1.8) if b in (nb//2 - 1, nb//2) else ("0.55", 1.0)
    a1.plot(x, bands[:, b], c=col, lw=lw)
for t in ticks:
    a1.axvline(t, c="k", lw=0.5, alpha=0.4)
a1.set_xticks(ticks); a1.set_xticklabels(labels)
a1.set_xlim(x[0], x[-1]); a1.set_ylim(-120, 120)
a1.set_ylabel("E  (meV)")
a1.set_title(f"BM bands, $\\theta={theta_magic:.3f}^\\circ$"
             f"  (flat W={Wm_N4:.1f} meV)")

for hbv in hbv_candidates:
    lw, al = (2.2, 1.0) if hbv == best_hbv else (1.1, 0.6)
    a2.plot(thetas, scan[hbv], "o-", ms=3, lw=lw, alpha=al,
            label=f"$\\hbar v_F$={hbv:.0f}")
a2.axvspan(theta_magic - 0.05, theta_magic + 0.05, color="C3", alpha=0.12)
a2.set_xlabel(r"twist angle $\theta$ (deg)")
a2.set_ylabel("flat-band width W (meV)")
a2.set_title(r"magic-angle dip & $\pm0.05^\circ$ disorder window")
a2.legend(fontsize=8)

phi = np.linspace(0, 2 * np.pi, 600)
for t in taus:
    c = cpr[str(t)]
    I = sum(cm * np.sin((m + 1) * phi) for m, cm in enumerate(c))
    a3.plot(phi / np.pi, I, lw=1.6,
            label=f"$\\tau$={t}, $c_2/c_1$={c[1]/c[0]:+.2f}")
a3.plot(phi / np.pi, np.sin(phi), "k--", lw=1.0, label="sinusoidal")
a3.set_xlabel(r"$\varphi/\pi$"); a3.set_ylabel(r"$I(\varphi)/I_c$")
a3.set_title("Beenakker CPR (feeds M3 fingerprint)")
a3.legend(fontsize=8)

fig.suptitle("M1: TBG material module — flat bands, disorder sensitivity, CPR")
fig.tight_layout()
fig.savefig("/home/claude/fig_m1_material.png", dpi=160)
print(f"done t={time.time()-t0:.0f}s")
