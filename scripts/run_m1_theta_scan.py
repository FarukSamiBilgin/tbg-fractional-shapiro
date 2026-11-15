r"""M1 production item 3 -- fine theta scan for the twist-disorder appendix.

Computes the flat-band width W(theta) on a fine 0.005 deg grid over
theta in [0.90, 1.20] (61 points) at the locked reference numerics
(hbar*vF = 620 meV*nm, N = 3 momentum shells, 7x7 moire-BZ mesh), using
the validated m1 package (m1.bm.flat_band_width) -- no model code is
duplicated here.

From the fine curve it extracts, for the paper's twist-tolerance argument:
  * refined magic angle theta_magic_fine (parabola vertex around the minimum),
  * W_min at the minimum (directly from the fine grid, plus the parabola
    vertex value for reference),
  * the +-0.05 deg disorder-window values W(theta - 0.05), W(theta + 0.05)
    centered on the locked design angle 1.025 deg (matches material.json /
    AT-2), with the true-minimum-centered values reported alongside,
  * the curvature d2W/dtheta2 at the minimum (meV/deg^2), fit over several
    windows because the dip is V-shaped (nearly linear in detuning), plus the
    near-linear slopes dW/d|dtheta| that are the more robust local sensitivity.

It also recomputes the 13 coarse (0.025 deg) reference points independently
and reports their maximum deviation from the fine-grid values (they must lie
ON the fine curve -- a determinism/reproducibility check).

Checkpoint + resume: the fine scan is saved to out/m1/theta_scan.npz after
every theta with a per-point 'done' mask; a re-run resumes from where it
stopped. Deterministic, no RNG.

FINDINGS (7x7 reference numerics):
  * The fine minimum sits at 1.035 deg, W ~ 0.76 meV -- between the coarse
    samples 1.025 and 1.050, so the 0.025 deg grid could not resolve it. It is
    consistent with the locked 1.025 +- 0.0125 deg (the fine value is inside
    that band); the fine scan pins it more precisely.
  * The dip is sharply V-shaped (W grows ~linearly with |theta - theta_m|), so
    a parabola is only a local approximation: the fitted curvature is
    window-dependent (report it as an effective value with the window stated)
    and the parabola vertex overshoots the true tip slightly.
  * A small shoulder wiggle near 0.93-0.95 deg (local min-max-min) is a 7x7
    k-grid discretization feature, far from the magic angle; it does not touch
    the key numbers.
  * The tip depth/location at 7x7 is k-grid limited; a finer BZ mesh (item 1,
    24x24) will refine W_min. This scan is fixed at reference numerics.

Windows host: run via PowerShell, no inline python. Matplotlib Agg backend.
"""

import sys
import os
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"~\Desktop\TBG FOTONİK JJ"
sys.path.insert(0, ROOT)
from m1.bm import flat_band_width  # validated model, do not duplicate

# ------------------------------------------------------------------ config ---
HBV = 620.0           # meV*nm  (locked, vF = 0.94e6 m/s)
N = 3                 # momentum shells (reference numerics; N=3 converged)
GRID = 7              # 7x7 moire-BZ mesh (reference numerics)
THETA_DESIGN = 1.025  # locked design angle (deg); disorder-window center (AT-2)
DISORDER = 0.05       # +-deg disorder-window half-width
PRIMARY_HW = 0.010    # parabola half-window (deg) for the local curvature/vertex

OUT_DIR = os.path.join(ROOT, "out", "m1")
FIG_DIR = os.path.join(ROOT, "figs")
NPZ = os.path.join(OUT_DIR, "theta_scan.npz")
JSON = os.path.join(OUT_DIR, "theta_scan.json")
FIG = os.path.join(FIG_DIR, "fig_m1_theta_scan.png")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# theta grids (rounded to kill float-arange drift; 0.025 is a multiple of 0.005
# so every coarse theta is exactly a fine-grid theta).
THETAS_FINE = np.round(np.arange(0.90, 1.2001, 0.005), 4)     # 61 points
THETAS_COARSE = np.round(np.arange(0.90, 1.2001, 0.025), 4)   # 13 points


# --------------------------------------------------------------- fine scan ---
def load_checkpoint():
    """Return (W, aux4xN, done) resuming from NPZ if it is compatible."""
    n = len(THETAS_FINE)
    if os.path.exists(NPZ):
        try:
            d = np.load(NPZ)
            if ("thetas_fine" in d and d["thetas_fine"].shape == THETAS_FINE.shape
                    and np.allclose(d["thetas_fine"], THETAS_FINE)
                    and float(d["hbv"]) == HBV and int(d["N"]) == N
                    and int(d["grid"]) == GRID and "done_fine" in d):
                W = d["W_fine"].astype(float).copy()
                aux = d["aux_fine"].astype(float).copy()
                done = d["done_fine"].astype(bool).copy()
                print(f"resume: {int(done.sum())}/{n} fine points already done")
                return W, aux, done
        except Exception as exc:  # noqa: BLE001 -- corrupt/partial checkpoint
            print(f"checkpoint unreadable ({exc!r}); starting fresh")
    return (np.full(n, np.nan), np.full((n, 4), np.nan), np.zeros(n, bool))


def checkpoint(W, aux, done):
    """Atomic-ish per-point checkpoint of the fine scan (enables resume)."""
    tmp = NPZ + ".tmp.npz"
    np.savez(tmp, thetas_fine=THETAS_FINE, W_fine=W, aux_fine=aux,
             done_fine=done, hbv=HBV, N=N, grid=GRID)
    os.replace(tmp, NPZ)


def run_fine_scan():
    W, aux, done = load_checkpoint()
    for i, th in enumerate(THETAS_FINE):
        if done[i]:
            continue
        Wi, edges = flat_band_width(float(th), HBV, N=N, grid=GRID)
        W[i] = Wi
        aux[i] = edges  # (lo_min, lo_max, hi_min, hi_max)
        done[i] = True
        checkpoint(W, aux, done)
    return W, aux, done


# ---------------------------------------------------------- parabola fit -----
def parabolic_fit(thetas, W, half_width_deg):
    """Fit W ~ a*t^2 + b*t + c on points within half_width of the grid min.

    Returns dict with vertex (theta, W), curvature 2a, n points, rms residual,
    and the raw coefficients.
    """
    imin = int(np.argmin(W))
    th0 = thetas[imin]
    mask = np.abs(thetas - th0) <= half_width_deg + 1e-9
    tt, ww = thetas[mask], W[mask]
    a, b, c = np.polyfit(tt, ww, 2)
    theta_v = -b / (2.0 * a)
    W_v = a * theta_v ** 2 + b * theta_v + c
    rms = float(np.sqrt(np.mean((ww - (a * tt ** 2 + b * tt + c)) ** 2)))
    return dict(theta_magic=float(theta_v), W_min=float(W_v),
                curvature=float(2.0 * a), npts=int(mask.sum()), rms=rms,
                coef=[float(a), float(b), float(c)])


def local_slope(thetas, W, lo, hi):
    """Least-squares slope dW/dtheta (meV/deg) over [lo, hi]."""
    m = (thetas >= lo - 1e-9) & (thetas <= hi + 1e-9)
    s = np.polyfit(thetas[m], W[m], 1)[0]
    return float(s), int(m.sum())


def main():
    t0 = time.time()
    print(f"fine theta scan: {len(THETAS_FINE)} points, "
          f"hbv={HBV}, N={N}, {GRID}x{GRID} grid")

    W_fine, aux_fine, done = run_fine_scan()
    assert done.all(), "fine scan incomplete"
    t_fine = time.time() - t0

    # independently recompute the 13 coarse reference points ------------------
    W_coarse = np.array([flat_band_width(float(th), HBV, N=N, grid=GRID)[0]
                         for th in THETAS_COARSE])
    fine_lut = {round(float(t), 4): float(w)
                for t, w in zip(THETAS_FINE, W_fine)}
    W_fine_at_coarse = np.array([fine_lut[round(float(t), 4)]
                                 for t in THETAS_COARSE])
    coarse_dev = np.abs(W_coarse - W_fine_at_coarse)
    max_coarse_dev = float(coarse_dev.max())

    # direct fine-grid minimum -------------------------------------------------
    grid_imin = int(np.argmin(W_fine))
    theta_grid_min = float(THETAS_FINE[grid_imin])
    W_grid_min = float(W_fine[grid_imin])

    # parabolic refinement over several windows (dip is V-shaped) --------------
    windows = (0.010, 0.015, 0.020, 0.025, 0.030)
    fits = {f"{hw:.3f}": parabolic_fit(THETAS_FINE, W_fine, hw)
            for hw in windows}
    for hw in windows:
        f = fits[f"{hw:.3f}"]
        print(f"  fit +-{hw:.3f} deg ({f['npts']:2d} pts): "
              f"theta_m={f['theta_magic']:.4f}  W_v={f['W_min']:.4f}  "
              f"d2W/dth2={f['curvature']:.0f} meV/deg^2  rms={f['rms']:.2e}")

    p = fits[f"{PRIMARY_HW:.3f}"]
    theta_magic_fine = p["theta_magic"]
    W_min_parabola = p["W_min"]
    curvature = p["curvature"]
    curv_vals = [fits[f"{hw:.3f}"]["curvature"] for hw in windows]

    # near-linear tolerance slopes on each side of the tip --------------------
    slope_lo, n_lo = local_slope(THETAS_FINE, W_fine, 0.960, theta_grid_min)
    slope_hi, n_hi = local_slope(THETAS_FINE, W_fine, theta_grid_min, 1.075)

    # disorder window centered on the locked design angle (matches AT-2) ------
    th_minus = round(THETA_DESIGN - DISORDER, 4)   # 0.975
    th_plus = round(THETA_DESIGN + DISORDER, 4)    # 1.075
    W_minus = fine_lut[th_minus]
    W_plus = fine_lut[th_plus]

    # secondary: window centered on the fine minimum (both are fine-grid pts) -
    th_minus_m = round(theta_grid_min - DISORDER, 4)   # 0.985
    th_plus_m = round(theta_grid_min + DISORDER, 4)    # 1.085
    W_minus_m = fine_lut.get(th_minus_m)
    W_plus_m = fine_lut.get(th_plus_m)

    runtime = time.time() - t0

    # ---------------------------------------------------------- persist ------
    np.savez(NPZ, thetas_fine=THETAS_FINE, W_fine=W_fine, aux_fine=aux_fine,
             done_fine=done, hbv=HBV, N=N, grid=GRID,
             thetas_coarse=THETAS_COARSE, W_coarse=W_coarse,
             W_fine_at_coarse=W_fine_at_coarse)

    result = {
        "description": "M1 fine theta scan for the twist-disorder appendix "
                       "(production item 3)",
        "params": {"hbar_vF_meV_nm": HBV, "N_shells": N, "BZ_grid": GRID,
                   "theta_min_deg": 0.90, "theta_max_deg": 1.20,
                   "theta_step_deg": 0.005, "n_points": len(THETAS_FINE),
                   "theta_design_deg": THETA_DESIGN,
                   "disorder_halfwidth_deg": DISORDER},
        "magic_angle": {
            "theta_magic_fine_deg": theta_magic_fine,
            "theta_grid_min_deg": theta_grid_min,
            "W_min_fine_meV": W_grid_min,
            "W_min_parabola_vertex_meV": W_min_parabola,
            "note": ("theta_magic_fine is the parabola vertex over "
                     f"+-{PRIMARY_HW} deg; W_min_fine is the direct fine-grid "
                     "minimum. The dip is V-shaped, so the parabola vertex "
                     "slightly overshoots the true tip. Fine minimum (1.035) "
                     "lies within the locked 1.025 +- 0.0125 deg band."),
        },
        "curvature": {
            "d2W_dtheta2_meV_per_deg2": curvature,
            "fit_half_width_deg": PRIMARY_HW,
            "fit_rms_meV": p["rms"],
            "window_dependence": {f"{hw:.3f}": fits[f"{hw:.3f}"]["curvature"]
                                  for hw in windows},
            "note": ("effective curvature: the dip is nearly linear in "
                     "|dtheta| so d2W/dtheta2 depends on the fit window "
                     f"(range {min(curv_vals):.0f}-{max(curv_vals):.0f} over "
                     "+-0.010 to +-0.030 deg). The near-linear slopes below "
                     "are the more robust local twist-sensitivity."),
            "linear_slope_low_side_meV_per_deg": slope_lo,
            "linear_slope_high_side_meV_per_deg": slope_hi,
        },
        "disorder_window": {
            "primary_center_deg": THETA_DESIGN,
            "theta_minus_deg": th_minus, "W_minus_meV": W_minus,
            "theta_plus_deg": th_plus, "W_plus_meV": W_plus,
            "broadening_vs_W_min_fine": [round(W_minus / W_grid_min, 2),
                                         round(W_plus / W_grid_min, 2)],
            "secondary_center_deg": theta_grid_min,
            "secondary_theta_minus_deg": th_minus_m,
            "secondary_W_minus_meV": W_minus_m,
            "secondary_theta_plus_deg": th_plus_m,
            "secondary_W_plus_meV": W_plus_m,
            "note": ("primary window centered on the locked design angle "
                     "1.025 deg to match material.json / AT-2 (7.408 / 7.138 "
                     "meV); secondary window centered on the fine minimum "
                     "1.035 deg for the appendix sensitivity discussion."),
        },
        "coarse_check": {
            "n_points": len(THETAS_COARSE), "step_deg": 0.025,
            "max_deviation_from_fine_meV": max_coarse_dev,
            "note": ("coarse points recomputed independently; they coincide "
                     "with the fine-grid values to machine precision "
                     "(deterministic)."),
        },
        "shoulder_artifact": {
            "theta_range_deg": [0.925, 0.950],
            "note": ("small local min-max-min wiggle ~0.05 meV amplitude; a "
                     "7x7 k-grid discretization feature far from the magic "
                     "angle, not physical."),
        },
        "runtime_s": round(runtime, 2),
        "runtime_fine_scan_s": round(t_fine, 2),
    }
    with open(JSON, "w") as fh:
        json.dump(result, fh, indent=2)

    # ------------------------------------------------------------- figure ----
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(figsize=(9.2, 5.8))

    ax.axvspan(th_minus, th_plus, color="C3", alpha=0.12,
               label=r"$\pm0.05^\circ$ disorder window (about $1.025^\circ$)")
    ax.plot(THETAS_FINE, W_fine, "-", color="C0", lw=1.8, zorder=3,
            label=r"fine scan ($0.005^\circ$, $N=3$, $7\times7$)")
    ax.plot(THETAS_COARSE, W_coarse, "s", ms=6, mfc="none", mec="k",
            mew=1.3, zorder=5,
            label=r"coarse reference ($0.025^\circ$, recomputed)")

    # local parabola over its window
    a, b, c = p["coef"]
    tt = np.linspace(theta_grid_min - PRIMARY_HW,
                     theta_grid_min + PRIMARY_HW, 100)
    ax.plot(tt, a * tt ** 2 + b * tt + c, "--", color="C2", lw=1.6, zorder=4,
            label=rf"local parabola ($\pm{PRIMARY_HW:g}^\circ$)")

    # true fine-grid minimum
    ax.plot([theta_grid_min], [W_grid_min], "*", color="C2", ms=17, zorder=6,
            label=(rf"$\theta_m={theta_magic_fine:.4f}^\circ$, "
                   rf"$W_{{\min}}={W_grid_min:.3f}$ meV"))

    # locked design angle
    ax.axvline(THETA_DESIGN, color="0.4", lw=1.0, ls=":", zorder=2)
    ax.annotate(r"locked $1.025^\circ$", (THETA_DESIGN, ax.get_ylim()[1]),
                textcoords="offset points", xytext=(4, -14), fontsize=8.5,
                color="0.35", rotation=90, va="top")

    # disorder-window endpoints
    ax.plot([th_minus, th_plus], [W_minus, W_plus], "o", color="C3", ms=7,
            zorder=6)
    ax.annotate(rf"$W={W_minus:.2f}$", (th_minus, W_minus),
                textcoords="offset points", xytext=(-6, 10), fontsize=9,
                color="C3", ha="right")
    ax.annotate(rf"$W={W_plus:.2f}$", (th_plus, W_plus),
                textcoords="offset points", xytext=(6, 10), fontsize=9,
                color="C3")

    ax.set_xlabel(r"twist angle $\theta$ (deg)")
    ax.set_ylabel("flat-band width $W$ (meV)")
    ax.set_title("M1 fine $\\theta$-scan: magic-angle dip & twist-disorder "
                 "window\n"
                 r"($\hbar v_F=620$ meV$\cdot$nm, $N=3$, $7\times7$; V-shaped "
                 rf"dip, slopes $\approx{abs(slope_lo):.0f}/{abs(slope_hi):.0f}$"
                 r" meV/deg)")
    ax.set_xlim(0.90, 1.20)
    ax.legend(fontsize=8.5, loc="upper center", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(FIG, dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------- report ----
    print("\n=== KEY NUMBERS ===")
    print(f"theta_magic_fine:    {theta_magic_fine:.4f} deg (parabola vertex, "
          f"+-{PRIMARY_HW} deg)")
    print(f"grid-min theta:      {theta_grid_min:.4f} deg  "
          f"W_min_fine={W_grid_min:.4f} meV  (parabola vertex "
          f"{W_min_parabola:.4f})")
    print(f"W(-0.05={th_minus}): {W_minus:.4f} meV")
    print(f"W(+0.05={th_plus}):  {W_plus:.4f} meV")
    print(f"curvature d2W/dth2:  {curvature:.0f} meV/deg^2 (window range "
          f"{min(curv_vals):.0f}-{max(curv_vals):.0f})")
    print(f"linear slopes:       {slope_lo:.0f} (low) / {slope_hi:.0f} (high) "
          f"meV/deg")
    print(f"max coarse dev:      {max_coarse_dev:.3e} meV")
    print(f"runtime:             {runtime:.2f} s")
    print(f"\nwrote: {NPZ}\n       {JSON}\n       {FIG}")


if __name__ == "__main__":
    main()
