"""M3 Deliverable 1 - full deterministic Shapiro map (M3_BRIEF.md Sec. 4.1).

Produces the dV/dI colormap over the (i_dc, i_ac) bias plane for the skewed
CPR (with the sinusoidal CPR as a control), a spp/averaging convergence check,
and the paper figure. Fully deterministic: the RSJ integrator is RK4 with no
RNG, so every number here is reproducible bit-for-bit (seed status: none).

Reference numerics (brief Sec. 3): vectorized fixed-step RK4, spp = 60,
150 T transient, 300 T averaging, plateau tolerance |v - v_step| < 0.025.
v-resolution of the 300 T window = Omega / n_avg = 5 / 300 = 0.0167, and the
reference tolerance 0.025 = 1.5 x that resolution.

Checkpoint / resume: each i_ac row-block is written to
out/shapiro_blocks/<cpr>_block_<start>.npz. Re-running the script skips any
block already on disk, so a timed-out run resumes where it stopped. Delete the
block directory to force a clean recompute.

Owned outputs: scripts/run_shapiro_map.py, out/shapiro_map.npz,
out/shapiro_blocks/*, out/shapiro_convergence.json, figs/fig_m3_shapiro_map.png.
"""
import json
import pathlib
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junction.analysis import dvdi, step_width
from junction.cpr import C_SIN, C_SKEW
from junction.rsj import OMEGA, grid_run

# ----------------------------------------------------------------- paths / dirs
OUT = ROOT / "out"
BLOCKS = OUT / "shapiro_blocks"
FIGS = ROOT / "figs"
for d in (OUT, BLOCKS, FIGS):
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- fixed numerics
SPP = 60
N_TRANSIENT = 150
N_AVG = 300
TOL = 0.025                       # = 1.5 * (Omega / N_AVG) = 1.5 * 0.01667
V_RES_MAIN = OMEGA / N_AVG        # 0.01667
BLOCK_ROWS = 20
HALF_V = 0.5 * OMEGA             # v = Omega/2 = 2.5, the fractional locus
SEED_STATUS = "deterministic - no RNG"

# ----------------------------------------------------------------- bias grids
# step 0.05 (281 pts) x step 0.07 (201 pts) -> 201 x 281 >= 200 x 280
I_DC = np.linspace(0.0, 14.0, 281)
I_AC = np.linspace(0.0, 14.0, 201)

CPRS = {"skew": C_SKEW, "sin": C_SIN}


# ================================================================= main map
def compute_map(name, c):
    """Row-blocked, checkpointed grid_run over the full (i_ac, i_dc) plane."""
    n_iac = len(I_AC)
    v_full = np.empty((n_iac, len(I_DC)), float)
    starts = list(range(0, n_iac, BLOCK_ROWS))
    for s in starts:
        e = min(s + BLOCK_ROWS, n_iac)
        iac_blk = I_AC[s:e]
        f = BLOCKS / f"{name}_block_{s:03d}.npz"
        if f.exists():
            d = np.load(f)
            if d["iac"].shape == iac_blk.shape and np.allclose(d["iac"], iac_blk):
                v_full[s:e] = d["v"]
                print(f"  [{name}] block rows {s:3d}-{e-1:3d}  (cached)")
                continue
            print(f"  [{name}] block rows {s:3d}-{e-1:3d}  (stale -> recompute)")
        t0 = time.time()
        v_blk = grid_run(I_DC, iac_blk, c,
                         n_transient=N_TRANSIENT, n_avg=N_AVG, spp=SPP)
        np.savez(f, iac=iac_blk, idc=I_DC, v=v_blk)
        v_full[s:e] = v_blk
        print(f"  [{name}] block rows {s:3d}-{e-1:3d}  ({time.time()-t0:5.1f}s)")
    return v_full


# ================================================================= convergence
def convergence_check():
    """Decimated-grid spp / averaging convergence for the skewed CPR.

    Base variant matches the map numerics (spp=60, 150/300 T). Two comparison
    variants: doubled spp, and 1000 T averaging with a 500 T transient. Each is
    checkpointed to its own block file so this section also resumes.
    """
    idc_d = I_DC[::4]                          # 71 pts (idx 0,4,...,280)
    iac_d = I_AC[::4]                           # 51 pts (idx 0,4,...,200)
    variants = {
        "v1_spp60_300T":  dict(spp=60,  n_transient=150, n_avg=300),
        "v2_spp120_300T": dict(spp=120, n_transient=150, n_avg=300),
        "v3_spp60_1000T": dict(spp=60,  n_transient=500, n_avg=1000),
    }
    vmaps = {}
    for key, kw in variants.items():
        f = BLOCKS / f"conv_{key}.npz"
        if f.exists():
            d = np.load(f)
            if (d["iac"].shape == iac_d.shape and np.allclose(d["iac"], iac_d)
                    and d["idc"].shape == idc_d.shape and np.allclose(d["idc"], idc_d)):
                vmaps[key] = d["v"]
                print(f"  [conv] {key}  (cached)")
                continue
        t0 = time.time()
        v = grid_run(idc_d, iac_d, C_SKEW, **kw)
        np.savez(f, iac=iac_d, idc=idc_d, v=v)
        vmaps[key] = v
        print(f"  [conv] {key}  ({time.time()-t0:5.1f}s)")

    def pair(a, b):
        d = np.abs(vmaps[a] - vmaps[b])
        return {"max_abs_dv": float(d.max()),
                "p99_abs_dv": float(np.percentile(d, 99))}

    v_res_1000 = OMEGA / 1000.0                # 0.005
    tol_1000 = 2.5 * v_res_1000                # 0.0125
    result = {
        "description": "Deterministic spp/averaging convergence, skewed CPR, "
                       "decimated grid (every 4th point in both axes).",
        "seed_status": SEED_STATUS,
        "grid": {"n_iac": len(iac_d), "n_idc": len(idc_d),
                 "i_ac_min": float(iac_d[0]), "i_ac_max": float(iac_d[-1]),
                 "i_dc_min": float(idc_d[0]), "i_dc_max": float(idc_d[-1])},
        "variants": {
            "v1_spp60_300T":  {"spp": 60,  "n_transient": 150, "n_avg": 300,
                               "v_resolution": OMEGA / 300, "tol": 0.025},
            "v2_spp120_300T": {"spp": 120, "n_transient": 150, "n_avg": 300,
                               "v_resolution": OMEGA / 300, "tol": 0.025},
            "v3_spp60_1000T": {"spp": 60,  "n_transient": 500, "n_avg": 1000,
                               "v_resolution": v_res_1000, "tol": tol_1000},
        },
        "pairs": {
            "spp_60_vs_120":  pair("v1_spp60_300T", "v2_spp120_300T"),
            "avg_300T_vs_1000T": pair("v1_spp60_300T", "v3_spp60_1000T"),
        },
        "tolerance_1000T": tol_1000,
        "tolerance_1000T_derivation":
            "v_res = Omega / n_avg = 5 / 1000 = 0.005; tol = 2.5 * v_res = 0.0125",
    }
    return result


# ================================================================= figure
def make_figure(v_skew, dvdi_skew):
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    # clip dV/dI for contrast: plateaus ~0 (dark), transitions bright
    vmax = float(np.percentile(dvdi_skew, 99))
    pcm = ax.pcolormesh(I_DC, I_AC, dvdi_skew, shading="auto",
                        cmap="magma", vmin=0.0, vmax=vmax)
    cb = fig.colorbar(pcm, ax=ax, pad=0.02)
    cb.set_label(r"$dV/dI$  [$R_n$]  (reduced $dv/di_{dc}$)")

    # overlay v-contours: integer loci v = n*Omega (solid), half loci (dashed)
    int_levels = [5.0, 10.0]                       # n = 1, 2
    half_levels = [2.5, 7.5, 12.5]                 # n/2 = 1/2, 3/2, 5/2
    ci = ax.contour(I_DC, I_AC, v_skew, levels=int_levels,
                    colors="cyan", linewidths=1.3, linestyles="solid")
    ch = ax.contour(I_DC, I_AC, v_skew, levels=half_levels,
                    colors="white", linewidths=1.0, linestyles="dashed")
    ax.clabel(ci, fmt=lambda x: f"v={x:g} (n={x/OMEGA:g})", fontsize=7)
    ax.clabel(ch, fmt=lambda x: f"v={x:g} (n/2={x/OMEGA:g})", fontsize=7)

    # legend proxies
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="cyan", lw=1.3, ls="solid",
               label=r"integer loci $v=n\,\Omega$"),
        Line2D([0], [0], color="white", lw=1.0, ls="dashed",
               label=r"fractional loci $v=(n/2)\,\Omega$"),
    ], loc="upper left", fontsize=8, framealpha=0.85)

    ax.set_xlabel(r"DC bias  $i_{dc}=I_{dc}/I_c$")
    ax.set_ylabel(r"drive amplitude  $i_{ac}=I_{ac}/I_c$")
    ax.set_title(r"M3 Shapiro map: $dV/dI$, skewed CPR ($c_2/c_1=-0.25$), "
                 r"$\Omega=5$" "\n"
                 r"solid = integer steps, dashed = fractional ($v=\Omega/2$) fingerprint")
    fig.tight_layout()
    out = FIGS / "fig_m3_shapiro_map.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


# ================================================================= driver
def main():
    t_start = time.time()
    print("=" * 70)
    print("M3 Deliverable 1 - deterministic Shapiro map")
    print(f"grid: {len(I_AC)} x {len(I_DC)} (i_ac x i_dc), "
          f"spp={SPP}, {N_TRANSIENT}T transient, {N_AVG}T avg, tol={TOL}")
    print("=" * 70)

    # ---- main maps
    print("[1/4] main maps (skew, sin)")
    v_skew = compute_map("skew", C_SKEW)
    v_sin = compute_map("sin", C_SIN)
    dvdi_skew = dvdi(v_skew, I_DC)
    dvdi_sin = dvdi(v_sin, I_DC)

    # ---- save map npz
    map_path = OUT / "shapiro_map.npz"
    np.savez(
        map_path,
        i_ac=I_AC, i_dc=I_DC,
        v_skew=v_skew, dvdi_skew=dvdi_skew,
        v_sin=v_sin, dvdi_sin=dvdi_sin,
        spp=SPP, n_transient=N_TRANSIENT, n_avg=N_AVG, tol=TOL,
        omega=OMEGA, cpr_skew=C_SKEW, cpr_sin=C_SIN,
        v_resolution=V_RES_MAIN, seed_status=SEED_STATUS,
    )
    print(f"[2/4] saved {map_path}")

    # ---- convergence
    print("[3/4] convergence check (decimated grid)")
    conv = convergence_check()
    conv_path = OUT / "shapiro_convergence.json"
    conv_path.write_text(json.dumps(conv, indent=2))
    print(f"      saved {conv_path}")

    # ---- figure
    print("[4/4] figure")
    fig_path = make_figure(v_skew, dvdi_skew)
    print(f"      saved {fig_path}")

    # ============================================================= sanity + key numbers
    # nearest grid row to i_ac = 4.0 (grid step 0.07 -> 4.0 not exact)
    k4 = int(np.argmin(np.abs(I_AC - 4.0)))
    iac4 = float(I_AC[k4])
    w_half_skew_4 = step_width(v_skew[k4], I_DC, HALF_V, TOL)
    w_half_sin_4 = step_width(v_sin[k4], I_DC, HALF_V, TOL)

    # fraction of skew map locked to ANY integer step v = n*Omega
    n_near = np.round(v_skew / OMEGA)
    locked = np.abs(v_skew - n_near * OMEGA) < TOL
    frac_locked = float(locked.mean())

    # convergence headline numbers
    sp = conv["pairs"]["spp_60_vs_120"]
    av = conv["pairs"]["avg_300T_vs_1000T"]

    runtime = time.time() - t_start

    print("=" * 70)
    print("INPUT FILES:")
    print(f"  {ROOT / 'junction' / 'cpr.py'}   (C_SKEW, C_SIN)")
    print(f"  {ROOT / 'junction' / 'rsj.py'}   (grid_run, OMEGA)")
    print(f"  {ROOT / 'junction' / 'analysis.py'}   (dvdi, step_width)")
    print("SEED STATUS:")
    print(f"  {SEED_STATUS}  (RK4 integrator, no stochastic term)")
    print("OUTPUT PATHS:")
    print(f"  {map_path}")
    print(f"  {BLOCKS}\\*.npz")
    print(f"  {conv_path}")
    print(f"  {fig_path}")
    print("-" * 70)
    print("KEY NUMBERS")
    print(f"  half-step width @ i_ac~4 (row {k4}, i_ac={iac4:.3f}):")
    print(f"     skew = {w_half_skew_4:.4f}   (AT-3 target 0.250 +- 0.03; grid quantum 0.05)")
    print(f"     sin  = {w_half_sin_4:.4f}   (control, must be <= 0.05)")
    print(f"  convergence spp 60 vs 120:      max|dv|={sp['max_abs_dv']:.5f}  "
          f"p99|dv|={sp['p99_abs_dv']:.5f}")
    print(f"  convergence 300T vs 1000T:      max|dv|={av['max_abs_dv']:.5f}  "
          f"p99|dv|={av['p99_abs_dv']:.5f}")
    print(f"  tolerance re-derived (1000T variant): {conv['tolerance_1000T']:.4f}  "
          f"(2.5 x v_res 0.005)")
    print(f"  fraction of skew map locked to any integer step: {frac_locked:.4f}")
    print(f"  runtime: {runtime:.1f} s")
    print("=" * 70)
    # machine-readable summary line for the caller
    print("SUMMARY_JSON " + json.dumps({
        "w_half_skew_iac4": w_half_skew_4,
        "w_half_sin_iac4": w_half_sin_4,
        "iac4_row": iac4,
        "conv_spp_max_dv": sp["max_abs_dv"], "conv_spp_p99_dv": sp["p99_abs_dv"],
        "conv_avg_max_dv": av["max_abs_dv"], "conv_avg_p99_dv": av["p99_abs_dv"],
        "tol_1000T": conv["tolerance_1000T"],
        "frac_locked_integer": frac_locked,
        "runtime_s": runtime,
    }))


if __name__ == "__main__":
    main()
