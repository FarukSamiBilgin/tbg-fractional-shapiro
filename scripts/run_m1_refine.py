"""M1 production refinement (M1_BRIEF Sec. 4, items 1-2).

Owned by this task. Imports the validated m1 package (does NOT duplicate its
Hamiltonian assembly -- the critical "build kinetic blocks once, add only the
interlayer h.c." logic lives in m1.bm.hamiltonian and is reused as-is).

Deliverables
------------
1. Flat-band width W at theta_design = 1.025 deg, hbar*vF = 620 meV*nm on a
   24x24 moire-BZ grid for basis cutoffs N in {3, 4, 5}; report W and the
   successive cutoff drifts |W_{N+1} - W_N| (verify the brief's "N=3 already
   converged to 0.01 meV" claim at this finer grid).
2. Remote-band gap at theta_design, N=4, 24x24 grid: per k collect the 6
   eigenvalues 2Ng-3 .. 2Ng+2 (2nd/1st remote below, two flat, 1st/2nd remote
   above) and report
       gap_above = min_k E_remote_above - max_k E_flat_upper
       gap_below = min_k E_flat_lower  - max_k E_remote_below
   with their ratio to the ħω = 0.6 meV drive photon and the argmin/argmax
   BZ locations (direct vs indirect character).
3. Save all arrays (per-N .npz) + a JSON summary. Checkpoint per k-row so a
   timeout resumes from the last completed row.

Deterministic, no RNG. Report only computed numbers.

Windows: run via PowerShell, no inline `python -c`. Long enough to checkpoint.
"""

import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m1 import bm  # validated band module (prerequisite)

# ---- locked contract parameters (M1_BRIEF Sec. 1) --------------------------
THETA_DESIGN = 1.025      # deg (measured magic angle)
HBV = 620.0               # meV*nm  (hbar*vF, vF = 0.94e6 m/s)
GRID = 24                 # 24x24 moire-BZ mesh
DRIVE_PHOTON_MEV = 0.6    # ħω drive photon for the remote-band argument

OUT = ROOT / "out" / "m1"
OUT.mkdir(parents=True, exist_ok=True)


def window_indices(Ng):
    """Eigenvalue indices 2Ng-3 .. 2Ng+2 (ascending eigvalsh ordering).

    Layout of the 6-band window (columns):
      0: 2Ng-3  second remote below
      1: 2Ng-2  first  remote below
      2: 2Ng-1  lower flat band
      3: 2Ng    upper flat band
      4: 2Ng+1  first  remote above
      5: 2Ng+2  second remote above
    """
    base = 2 * Ng
    return np.array([base - 3, base - 2, base - 1, base, base + 1, base + 2])


def compute_windows(N, grid=GRID, theta=THETA_DESIGN, hbv=HBV):
    """Per-k 6-band window over the grid x grid moire BZ, checkpointed by row.

    k = (i/grid) * b1 + (j/grid) * b2  for i, j in [0, grid)  -- identical mesh
    convention to m1.bm.flat_band_width so W is directly comparable.

    Returns windows of shape (grid, grid, 6): windows[i, j] = the 6 eigenvalues
    at row i, column j.
    """
    ktheta, qs, b1, b2 = bm.geometry(theta)
    mn = bm.basis(N)
    Ng = len(mn)
    dim = 4 * Ng
    widx = window_indices(Ng)

    ck = OUT / f"refine_N{N}_ck.npz"
    windows = np.full((grid, grid, 6), np.nan)
    rows_done = 0
    if ck.exists():
        data = np.load(ck)
        if (int(data["N"]) == N and int(data["grid"]) == grid
                and float(data["theta"]) == theta and float(data["hbv"]) == hbv):
            windows = data["windows"]
            rows_done = int(data["rows_done"])
            print(f"  [N={N}] resume from row {rows_done}/{grid}", flush=True)

    t0 = time.time()
    for i in range(rows_done, grid):
        for j in range(grid):
            k = (i / grid) * b1 + (j / grid) * b2
            ev = np.linalg.eigvalsh(
                bm.hamiltonian(k, mn, hbv, ktheta, qs, b1, b2))
            windows[i, j] = ev[widx]
        rows_done = i + 1
        np.savez(ck, windows=windows, rows_done=rows_done, N=N, grid=grid,
                 theta=theta, hbv=hbv, dim=dim, Ng=Ng)
        print(f"  [N={N}] row {rows_done}/{grid} done "
              f"(t={time.time() - t0:.1f}s)", flush=True)
    return windows, Ng, dim


def flat_width_from_windows(windows):
    """W = max_k E_flat_upper - min_k E_flat_lower (cols 3 and 2)."""
    e_flat_lower = windows[:, :, 2]
    e_flat_upper = windows[:, :, 3]
    return float(e_flat_upper.max() - e_flat_lower.min())


def gap_report(windows, grid=GRID, theta=THETA_DESIGN, hbv=HBV):
    """Remote-band gaps above/below the flat manifold + indirect character."""
    e_remote_below = windows[:, :, 1]   # 2Ng-2, first remote below
    e_flat_lower = windows[:, :, 2]     # 2Ng-1
    e_flat_upper = windows[:, :, 3]     # 2Ng
    e_remote_above = windows[:, :, 4]   # 2Ng+1, first remote above

    def loc(arr, kind):
        idx = np.argmin(arr) if kind == "min" else np.argmax(arr)
        i, j = np.unravel_index(idx, arr.shape)
        ktheta, qs, b1, b2 = bm.geometry(theta)
        kvec = (i / grid) * b1 + (j / grid) * b2
        return {"i": int(i), "j": int(j),
                "k_frac": [round(i / grid, 6), round(j / grid, 6)],
                "k_vec_invnm": [round(float(kvec[0]), 6),
                                round(float(kvec[1]), 6)],
                "E_meV": round(float(arr[i, j]), 6)}

    ra_min = loc(e_remote_above, "min")
    fu_max = loc(e_flat_upper, "max")
    fl_min = loc(e_flat_lower, "min")
    rb_max = loc(e_remote_below, "max")

    gap_above = ra_min["E_meV"] - fu_max["E_meV"]
    gap_below = fl_min["E_meV"] - rb_max["E_meV"]

    above_indirect = (ra_min["i"], ra_min["j"]) != (fu_max["i"], fu_max["j"])
    below_indirect = (fl_min["i"], fl_min["j"]) != (rb_max["i"], rb_max["j"])

    return {
        "gap_above_meV": round(float(gap_above), 6),
        "gap_below_meV": round(float(gap_below), 6),
        "gap_above_over_drive": round(float(gap_above / DRIVE_PHOTON_MEV), 4),
        "gap_below_over_drive": round(float(gap_below / DRIVE_PHOTON_MEV), 4),
        "drive_photon_meV": DRIVE_PHOTON_MEV,
        "above": {"remote_min": ra_min, "flat_upper_max": fu_max,
                  "indirect": bool(above_indirect)},
        "below": {"flat_lower_min": fl_min, "remote_below_max": rb_max,
                  "indirect": bool(below_indirect)},
        "flat_manifold": {
            "E_flat_lower_min_meV": round(float(e_flat_lower.min()), 6),
            "E_flat_upper_max_meV": round(float(e_flat_upper.max()), 6),
            "E_remote_above_min_meV": round(float(e_remote_above.min()), 6),
            "E_remote_below_max_meV": round(float(e_remote_below.max()), 6),
        },
    }


def main():
    t_start = time.time()
    print(f"M1 refine: theta={THETA_DESIGN} deg, hbar*vF={HBV} meV*nm, "
          f"{GRID}x{GRID} BZ grid", flush=True)

    # ---- item 1: W convergence in cutoff N at the 24x24 grid ---------------
    W = {}
    windows_by_N = {}
    dims = {}
    for N in (3, 4, 5):
        print(f"[cutoff N={N}] computing 6-band windows over "
              f"{GRID}x{GRID} grid ...", flush=True)
        win, Ng, dim = compute_windows(N)
        windows_by_N[N] = win
        dims[N] = dim
        W[N] = flat_width_from_windows(win)
        print(f"[cutoff N={N}] W_{GRID}x{GRID} = {W[N]:.6f} meV "
              f"(Ng={Ng}, dim={dim})", flush=True)
        # persist per-N arrays
        np.savez(OUT / f"refine_N{N}.npz",
                 windows=win, N=N, grid=GRID, theta=THETA_DESIGN, hbv=HBV,
                 Ng=Ng, dim=dim, W_meV=W[N])

    drift_N4_N5 = abs(W[5] - W[4])
    drift_N3_N4 = abs(W[4] - W[3])

    # ---- item 2: remote-band gap at N=4 ------------------------------------
    gaps = gap_report(windows_by_N[4])

    runtime = time.time() - t_start

    summary = {
        "task": "M1 production refinement (brief Sec.4 items 1-2)",
        "params": {"theta_design_deg": THETA_DESIGN, "hbar_vF_meV_nm": HBV,
                   "wAB_meV": bm.WAB, "wAA_over_wAB": bm.RATIO,
                   "grid": f"{GRID}x{GRID}", "n_k_points": GRID * GRID},
        "W_convergence": {
            "W_24x24_N3_meV": round(W[3], 6),
            "W_24x24_N4_meV": round(W[4], 6),
            "W_24x24_N5_meV": round(W[5], 6),
            "drift_N3_N4_meV": round(float(drift_N3_N4), 6),
            "drift_N4_N5_meV": round(float(drift_N4_N5), 6),
            "dims": {"N3": dims[3], "N4": dims[4], "N5": dims[5]},
            "note": ("brief claim: N=3 already converged to 0.01 meV; "
                     "verified at 24x24 -- both cutoff drifts are the "
                     "measured values above"),
        },
        "remote_band_gap": gaps,
        "runtime_s": round(runtime, 1),
    }

    with open(OUT / "refine.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n==== SUMMARY ====", flush=True)
    print(f"W_24x24_N4 = {W[4]:.6f} meV", flush=True)
    print(f"W_24x24_N5 = {W[5]:.6f} meV", flush=True)
    print(f"drift N4->N5 = {drift_N4_N5:.6f} meV "
          f"(N3->N4 = {drift_N3_N4:.6f} meV)", flush=True)
    print(f"gap_above = {gaps['gap_above_meV']:.4f} meV "
          f"(ratio/0.6 = {gaps['gap_above_over_drive']:.2f}, "
          f"indirect={gaps['above']['indirect']})", flush=True)
    print(f"gap_below = {gaps['gap_below_meV']:.4f} meV "
          f"(ratio/0.6 = {gaps['gap_below_over_drive']:.2f}, "
          f"indirect={gaps['below']['indirect']})", flush=True)
    print(f"runtime = {runtime:.1f}s", flush=True)
    print(f"wrote {OUT / 'refine.json'}", flush=True)


if __name__ == "__main__":
    main()
