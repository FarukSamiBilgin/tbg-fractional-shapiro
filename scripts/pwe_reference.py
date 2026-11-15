"""
M2 reference — 2D plane-wave expansion (PWE) bands of a triangular air-hole
lattice in silicon, used to LOCK the photonic-crystal lattice constant so that
the TBG-JJ drive frequency f0 = 145 GHz sits at midgap.

Two dielectric models bracket the real 3D slab:
  eps = 11.68  : bulk Si (2D ideal limit)
  eps =  8.12  : effective slab index n_eff = 2.85 for a ~0.5a-thick Si slab
Design polarization: TE (H_z; E in-plane) -- the gap-bearing polarization for
hole-type lattices and the TE-like slab modes.

Outputs: band diagram + gap-vs-(r/a) figure, JSON with locked numbers.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.special import j1

C_MM_GHZ = 299792.458 / 1000.0     # mm * GHz  (c = 299792.458 mm*GHz)
F0_GHZ = 145.0
LAMBDA0_UM = C_MM_GHZ / F0_GHZ * 1000.0   # ~2067.5 um

# ---------------- reciprocal lattice (a = 1) --------------------------------
b1 = 2.0 * np.pi * np.array([1.0, -1.0 / np.sqrt(3.0)])
b2 = 2.0 * np.pi * np.array([0.0,  2.0 / np.sqrt(3.0)])

def g_vectors(N):
    idx = np.arange(-N, N + 1)
    M, Nn = np.meshgrid(idx, idx, indexing="ij")
    return M.ravel()[:, None] * b1[None, :] + Nn.ravel()[:, None] * b2[None, :]

def fourier_coeffs(dG_norm, r_over_a, val_hole, val_bg):
    """Fourier coefficients of a function equal to val_hole inside circular
    holes of radius r and val_bg outside, on a triangular lattice (a=1)."""
    f = (2.0 * np.pi / np.sqrt(3.0)) * r_over_a ** 2      # filling fraction
    out = np.empty_like(dG_norm)
    zero = dG_norm < 1e-9
    out[zero] = val_bg + f * (val_hole - val_bg)
    x = dG_norm[~zero] * r_over_a
    out[~zero] = (val_hole - val_bg) * 2.0 * f * j1(x) / x
    return out

def k_path(npts=31):
    G = np.array([0.0, 0.0])
    M = 0.5 * b1
    K = (2.0 * b1 + b2) / 3.0
    segs = [(G, M), (M, K), (K, G)]
    ks, ticks, labels, dist = [], [0.0], [r"$\Gamma$", "M", "K", r"$\Gamma$"], 0.0
    for A, B in segs:
        for t in np.linspace(0, 1, npts, endpoint=False):
            ks.append(A + t * (B - A))
        dist += np.linalg.norm(B - A)
        ticks.append(dist)
    ks.append(G)
    ks = np.array(ks)
    x = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(ks, axis=0), axis=1))])
    return ks, x, ticks, labels

def bands(eps_bg, r_over_a, N=5, nbands=6, npts=31):
    Gs = g_vectors(N)
    dG = Gs[:, None, :] - Gs[None, :, :]
    dGn = np.linalg.norm(dG, axis=2)
    eta = fourier_coeffs(dGn, r_over_a, 1.0, 1.0 / eps_bg)       # 1/eps
    epsm = fourier_coeffs(dGn, r_over_a, 1.0, eps_bg)            # eps
    ks, x, ticks, labels = k_path(npts)
    te = np.empty((len(ks), nbands))
    tm = np.empty((len(ks), nbands))
    for i, k in enumerate(ks):
        kG = k[None, :] + Gs                                     # (Ng,2)
        dots = kG @ kG.T
        # TE (H_z): Hermitian standard problem
        w2 = eigh(eta * dots, eigvals_only=True)[:nbands]
        te[i] = np.sqrt(np.clip(w2, 0, None)) / (2.0 * np.pi)    # nu = a/lambda
        # TM (E_z): generalized problem diag(|k+G|^2) e = w2 * eps e
        A = np.diag(np.einsum("ij,ij->i", kG, kG))
        w2 = eigh(A, epsm, eigvals_only=True)[:nbands]
        tm[i] = np.sqrt(np.clip(w2, 0, None)) / (2.0 * np.pi)
    return te, tm, x, ticks, labels

def te_gap(eps_bg, r_over_a, N=5):
    te, _, _, _, _ = bands(eps_bg, r_over_a, N=N, nbands=3)
    lo, hi = te[:, 0].max(), te[:, 1].min()
    return (lo, hi) if hi > lo else (np.nan, np.nan)

# ---------------- design scan ------------------------------------------------
cases = {"bulk eps=11.68": 11.68, "slab n_eff=2.85 (eps=8.12)": 8.12}
r_scan = [0.25, 0.30, 0.35, 0.40]
scan = {}
print("TE gap edges (a/lambda) vs r/a")
for name, eps in cases.items():
    scan[name] = {}
    for r in r_scan:
        lo, hi = te_gap(eps, r)
        scan[name][r] = (lo, hi)
        if np.isnan(lo):
            print(f"  {name:28s} r/a={r:.2f}:  no gap")
        else:
            mid = 0.5 * (lo + hi)
            print(f"  {name:28s} r/a={r:.2f}:  {lo:.4f}-{hi:.4f}"
                  f"  mid={mid:.4f}  gap/mid={100*(hi-lo)/mid:4.1f}%"
                  f"  -> a={mid*LAMBDA0_UM:6.1f} um")

# convergence check for the design point
DESIGN_R = 0.30
for N in (4, 5, 6):
    lo, hi = te_gap(8.12, DESIGN_R, N=N)
    print(f"convergence eps=8.12 r/a=0.30 N={N}: mid a/lambda = {0.5*(lo+hi):.4f}")

lo_s, hi_s = scan["slab n_eff=2.85 (eps=8.12)"][DESIGN_R]
lo_b, hi_b = scan["bulk eps=11.68"][DESIGN_R]
mid_s = 0.5 * (lo_s + hi_s)
a_design = mid_s * LAMBDA0_UM
print(f"\nLOCKED: r/a=0.30, slab model midgap a/lambda={mid_s:.4f}"
      f" -> a = {a_design:.0f} um ; f0 at midgap = 145 GHz"
      f" ; gap span = {lo_s*C_MM_GHZ/ (a_design/1000):.0f}-"
      f"{hi_s*C_MM_GHZ/(a_design/1000):.0f} GHz")

# ---------------- figures ----------------------------------------------------
te, tm, x, ticks, labels = bands(8.12, DESIGN_R, nbands=5)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.0, 4.6))

for b in range(te.shape[1]):
    a1.plot(x, te[:, b], c="C3", lw=1.4,
            label="TE" if b == 0 else None)
    a1.plot(x, tm[:, b], c="C0", lw=1.0, ls="--",
            label="TM" if b == 0 else None)
a1.axhspan(lo_s, hi_s, color="C3", alpha=0.15)
a1.axhline(mid_s, ls=":", c="k", lw=1.0)
a1.text(0.02, mid_s + 0.004, "145 GHz (midgap)", fontsize=9)
a1.set_xticks(ticks); a1.set_xticklabels(labels)
a1.set_xlim(x[0], x[-1]); a1.set_ylim(0, 0.65)
a1.set_ylabel(r"frequency  $a/\lambda$")
a1.set_title(f"slab model $\\epsilon_{{eff}}=8.12$, r/a=0.30, a={a_design:.0f} µm")
a1.legend(loc="upper left", fontsize=9)

for name, mstyle, col in ((list(cases)[0], "s", "C0"), (list(cases)[1], "o", "C3")):
    los = [scan[name][r][0] for r in r_scan]
    his = [scan[name][r][1] for r in r_scan]
    a2.fill_between(r_scan, los, his, alpha=0.18, color=col)
    a2.plot(r_scan, los, mstyle + "-", c=col, ms=5, label=name)
    a2.plot(r_scan, his, mstyle + "-", c=col, ms=5)
a2.set_xlabel("hole radius  r/a")
a2.set_ylabel(r"TE gap edges  $a/\lambda$")
a2.set_title("TE gap map (design corridor)")
a2.legend(fontsize=8, loc="upper left")

fig.suptitle("M2: photonic-crystal band design for the 145 GHz TBG-JJ cavity")
fig.tight_layout()
fig.savefig("/home/claude/fig_m2_bands.png", dpi=160)

out = {
    "f0_GHz": F0_GHZ, "lambda0_um": LAMBDA0_UM,
    "design": {"r_over_a": DESIGN_R, "slab_eps_eff": 8.12,
               "midgap_a_over_lambda": mid_s, "a_um": a_design,
               "slab_thickness": "0.5*a",
               "gap_edges_a_over_lambda": [lo_s, hi_s],
               "bulk_gap_edges": [lo_b, hi_b]},
    "scan": {n: {str(r): list(v) for r, v in d.items()} for n, d in scan.items()},
}
with open("/home/claude/photonics_targets.json", "w") as fh:
    json.dump(out, fh, indent=1)
print("done")
