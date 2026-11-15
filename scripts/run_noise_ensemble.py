"""M3 Deliverable 2 - Langevin noise ensemble (brief Sec. 4.2).

Thermal rounding of the fractional (half-integer) Shapiro step at v = Omega/2.
Stochastic-Heun Langevin ensemble (junction.noise.ensemble_staircase) on the
deterministic stage-B bias grid, at the AT-3 optimum i_ac = 4.0.

Runs (each RNG seed fixed and logged in output metadata):
  a) skewed CPR, Gamma = 0.140  (operating point, 100 mK)  seed 20260722
  b) skewed CPR, Gamma = 0.014  (clean comparison, 10 mK-eq) seed 20260723
  c) sinusoidal CPR, Gamma = 0.140 (control floor)          seed 20260724
  d) deterministic (Gamma = 0) skewed staircase (junction.rsj.run_rsj/grid_run)
     on the same grid, for overlay / contrast.

THE PAPER NUMBER: survival fraction = W_half(Gamma=0.14, ensemble mean) / 0.250
(0.250 = deterministic half-step width, AT-3, i_ac=4.0).

Numerics
--------
Noisy: stochastic Heun, fixed step, spp = 100 (brief Sec. 5), n_transient = 150,
n_avg = 300, n_real = 20. Deterministic overlay: RK4 spp = 60, 150/300 T
(the validated stage-B reference numerics).

Tolerance: plateau tol = 0.025. The averaging window n_avg = 300 is unchanged
from the deterministic reference, so tau_avg = 300 * (2*pi/Omega) = 120*pi and
the voltage resolution 2*pi/tau_avg = Omega/n_avg = 5/300 = 0.01667 is identical
to the reference. tol = 0.025 = 1.5 * v_resolution is therefore carried over
unchanged (spp affects the integration step, not the averaging resolution).

Checkpoint / resume
-------------------
Each run is checkpointed to a per-run .npz block under out/ as soon as it
finishes (out/_block_{a,b,c,det}.npz). Re-invoking the script loads any existing
block instead of recomputing, so a timed-out run resumes where it left off.
Delete the block files (or run with --fresh) to force a full recompute.
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
from junction.cpr import C_SIN, C_SKEW, SKEW_RENORM
from junction.noise import ensemble_staircase
from junction.rsj import OMEGA, run_rsj

# --------------------------------------------------------------------------
# Fixed contract / numerics
# --------------------------------------------------------------------------
OUT = ROOT / "out"
FIGS = ROOT / "figs"
OUT.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

FRESH = "--fresh" in sys.argv[1:]

IDC = np.arange(1.0, 8.0 + 1e-9, 0.025)     # 281 pts, stage-B grid
I_AC = 4.0                                   # AT-3 optimum
SPP = 100                                    # brief Sec. 5: noisy spp >= 100
N_TRANSIENT = 150
N_AVG = 300
N_REAL = 20
V_TARGET = 0.5 * OMEGA                        # 2.5, the half-step
TOL = 0.025
V_RES = OMEGA / N_AVG                          # 2*pi/tau_avg = 0.016667
W_DET_REF = 0.250                              # AT-3 deterministic half-step width

SEED_A = 20260722   # skew, Gamma 0.14
SEED_B = 20260723   # skew, Gamma 0.014
SEED_C = 20260724   # sin,  Gamma 0.14

NOISE_KW = dict(spp=SPP, n_transient=N_TRANSIENT, n_avg=N_AVG, n_real=N_REAL)

RUNS = [
    # tag, label, CPR coeffs, Gamma, seed
    ("a", "skew Gamma=0.140", C_SKEW, 0.140, SEED_A),
    ("b", "skew Gamma=0.014", C_SKEW, 0.014, SEED_B),
    ("c", "sin  Gamma=0.140", C_SIN, 0.140, SEED_C),
]


def _block_path(tag):
    return OUT / f"_block_{tag}.npz"


def run_noisy(tag, c, Gamma, seed):
    """Compute (or load cached) per-realization v array, shape (N_REAL, n_idc)."""
    bp = _block_path(tag)
    if bp.exists() and not FRESH:
        d = np.load(bp)
        # guard: cached grid / config must match the current request
        if (d["v"].shape == (N_REAL, len(IDC))
                and int(d["seed"]) == seed
                and float(d["Gamma"]) == Gamma):
            print(f"  [resume] loaded cached block {bp.name}")
            return d["v"]
        print(f"  [stale]  {bp.name} config mismatch -> recompute")
    t0 = time.time()
    v = ensemble_staircase(IDC, I_AC, c, Gamma, seed=seed, **NOISE_KW)
    dt = time.time() - t0
    np.savez(bp, v=v, seed=seed, Gamma=Gamma, i_ac=I_AC, idc=IDC,
             spp=SPP, n_transient=N_TRANSIENT, n_avg=N_AVG, n_real=N_REAL)
    print(f"  computed {tag} ({N_REAL} real x {len(IDC)} pts) in {dt:.1f}s "
          f"-> {bp.name}")
    return v


def run_det():
    """Deterministic (Gamma=0) skewed staircase on the same grid (RK4, spp=60)."""
    bp = _block_path("det")
    if bp.exists() and not FRESH:
        d = np.load(bp)
        if d["v"].shape == IDC.shape:
            print(f"  [resume] loaded cached block {bp.name}")
            return d["v"]
    t0 = time.time()
    # reference stage-B deterministic numerics: RK4, spp=60, 150/300 T
    v = run_rsj(IDC, np.full_like(IDC, I_AC), C_SKEW,
                n_transient=150, n_avg=300, spp=60)
    dt = time.time() - t0
    np.savez(bp, v=v, i_ac=I_AC, idc=IDC, spp=60, n_transient=150, n_avg=300)
    print(f"  computed deterministic ({len(IDC)} pts, RK4 spp=60) in {dt:.1f}s")
    return v


def half_step_center(v_det):
    """i_dc window of the deterministic half-step plateau (|v-2.5|<tol)."""
    mask = np.abs(v_det - V_TARGET) < TOL
    if not mask.any():
        # fall back to nearest crossing
        k = int(np.argmin(np.abs(v_det - V_TARGET)))
        return IDC[k], k
    idc_in = IDC[mask]
    center = 0.5 * (idc_in.min() + idc_in.max())
    kc = int(np.argmin(np.abs(IDC - center)))
    return center, kc


def main():
    print("=" * 72)
    print("M3 Deliverable 2 - Langevin noise ensemble")
    print(f"grid: i_dc in [1.0, 8.0] step 0.025 ({len(IDC)} pts), i_ac={I_AC}")
    print(f"noisy numerics: stochastic Heun spp={SPP}, {N_TRANSIENT}T transient, "
          f"{N_AVG}T avg, n_real={N_REAL}")
    print(f"v_target={V_TARGET} (Omega/2), tol={TOL}, v_res={V_RES:.5f} "
          f"(tol/v_res={TOL/V_RES:.2f}), CPR renorm={SKEW_RENORM:.4f}")
    print("=" * 72)

    # ---- compute ensembles --------------------------------------------
    ens = {}
    for tag, label, c, Gamma, seed in RUNS:
        print(f"[run {tag}] {label}  seed={seed}")
        ens[tag] = run_noisy(tag, c, Gamma, seed)
    print("[run det] deterministic skew Gamma=0")
    v_det = run_det()

    # ---- analysis ------------------------------------------------------
    summary = {
        "description": "M3 deliverable 2: Langevin thermal rounding of the "
                       "v=Omega/2 fractional Shapiro step at i_ac=4.0",
        "grid": {"i_dc_start": 1.0, "i_dc_stop": 8.0, "i_dc_step": 0.025,
                 "n_idc": len(IDC), "i_ac": I_AC},
        "numerics": {"integrator": "stochastic-Heun (additive noise)",
                     "spp": SPP, "n_transient": N_TRANSIENT, "n_avg": N_AVG,
                     "n_real": N_REAL, "v_target": V_TARGET, "tol": TOL,
                     "v_resolution": V_RES, "tol_over_v_res": TOL / V_RES,
                     "det_integrator": "RK4 spp=60 150/300 T",
                     "cpr_skew_renorm": SKEW_RENORM},
        "seeds": {"a_skew_G0.140": SEED_A, "b_skew_G0.014": SEED_B,
                  "c_sin_G0.140": SEED_C, "det_skew_G0": None},
        "w_det_reference": W_DET_REF,
        "runs": {},
    }

    # deterministic width on THIS grid (self-consistency vs 0.250 reference)
    w_det_grid = step_width(v_det, IDC, V_TARGET, TOL)
    summary["w_det_on_grid"] = w_det_grid

    for tag, label, c, Gamma, seed in RUNS:
        v = ens[tag]
        v_mean = v.mean(axis=0)
        v_std = v.std(axis=0)
        # (i) width of the ensemble-MEAN staircase
        w_mean = step_width(v_mean, IDC, V_TARGET, TOL)
        # (ii) per-realization widths -> mean +- std
        w_per = np.array([step_width(v[r], IDC, V_TARGET, TOL)
                          for r in range(N_REAL)])
        summary["runs"][tag] = {
            "label": label, "Gamma": Gamma, "seed": seed,
            "w_half_ensemble_mean": float(w_mean),
            "w_half_per_real_mean": float(w_per.mean()),
            "w_half_per_real_std": float(w_per.std()),
            "w_half_per_real_min": float(w_per.min()),
            "w_half_per_real_max": float(w_per.max()),
        }

    # survival fractions (paper number uses ensemble-mean width / 0.250)
    wa = summary["runs"]["a"]["w_half_ensemble_mean"]
    wb = summary["runs"]["b"]["w_half_ensemble_mean"]
    wc = summary["runs"]["c"]["w_half_ensemble_mean"]
    summary["survival_fraction"] = {
        "PAPER_NUMBER_a_G0.140": wa / W_DET_REF,
        "b_G0.014": wb / W_DET_REF,
        "c_sin_control_floor_width": wc,
        "c_sin_control_floor_fraction": wc / W_DET_REF,
    }

    # dV/dI rounding measure at v = 2.5.  In reduced units v=V/(Ic Rn),
    # i=I/Ic, the differential resistance dv/di_dc is already in units of Rn
    # (it asymptotes to 1 far from any step).  The ideal deterministic step is
    # dissipationless: dv/di = 0 on the plateau.  Thermal noise lifts the
    # flattest point of the rounded step to a finite residual resistance, which
    # (being already Rn-normalised) is itself the "rounding ratio" R_res/Rn.
    center, kc = half_step_center(v_det)
    dvdi_det = dvdi(v_det, IDC)
    dvdi_a = dvdi(ens["a"].mean(axis=0), IDC)
    dvdi_b = dvdi(ens["b"].mean(axis=0), IDC)
    pmask = np.abs(v_det - V_TARGET) < TOL          # deterministic plateau window
    if not pmask.any():
        pmask = np.zeros_like(v_det, bool)
        pmask[kc] = True

    def _at(d):
        return abs(float(d[kc]))

    def _pmin(d):
        return float(np.abs(d[pmask]).min())

    d_det_c, d_a_c, d_b_c = _at(dvdi_det), _at(dvdi_a), _at(dvdi_b)
    d_det_min, d_a_min, d_b_min = _pmin(dvdi_det), _pmin(dvdi_a), _pmin(dvdi_b)
    # finite "rounding ratio": residual resistance of the Gamma=0.14 (operating)
    # rounded step relative to the Gamma=0.014 (clean) step, at the plateau min.
    ratio_a_over_b = (d_a_min / d_b_min) if d_b_min > 1e-6 else None
    summary["dvdi_rounding"] = {
        "units": "reduced dV/dI = dv/di_dc, already normalised to Rn "
                 "(-> 1 far from steps; 0 = ideal dissipationless lock)",
        "step_center_i_dc": float(center),
        "dvdi_det_at_center": d_det_c,
        "dvdi_noisy_a_at_center": d_a_c,
        "dvdi_noisy_b_at_center": d_b_c,
        "dvdi_det_plateau_min": d_det_min,
        "dvdi_noisy_a_plateau_min": d_a_min,
        "dvdi_noisy_b_plateau_min": d_b_min,
        "rounding_ratio_a_residual_over_Rn": d_a_min,
        "rounding_ratio_b_residual_over_Rn": d_b_min,
        "rounding_ratio_a_over_b": ratio_a_over_b,
        "note": "Deterministic step is dissipationless (dV/dI ~ 5e-13 ~ 0). "
                "Thermal rounding lifts the flattest point of the step to a "
                "residual differential resistance of {:.2f} Rn at Gamma=0.14 vs "
                "{:.3f} Rn at Gamma=0.014. The noisy-vs-deterministic ratio is "
                "formally divergent (denominator ~0); the physical rounding "
                "measure is the residual dV/dI itself (in units of Rn), and the "
                "operating/clean ratio is {}.".format(
                    d_a_min, d_b_min,
                    ("%.2f" % ratio_a_over_b) if ratio_a_over_b else "n/a"),
    }

    # ---- save NPZ ------------------------------------------------------
    npz_path = OUT / "noise_ensemble.npz"
    np.savez(
        npz_path,
        i_dc=IDC, i_ac=I_AC, v_target=V_TARGET, tol=TOL, v_resolution=V_RES,
        spp=SPP, n_transient=N_TRANSIENT, n_avg=N_AVG, n_real=N_REAL,
        seed_a=SEED_A, seed_b=SEED_B, seed_c=SEED_C,
        Gamma_a=0.140, Gamma_b=0.014, Gamma_c=0.140,
        c_skew=C_SKEW, c_sin=C_SIN, cpr_skew_renorm=SKEW_RENORM,
        v_real_a=ens["a"], v_real_b=ens["b"], v_real_c=ens["c"],
        v_mean_a=ens["a"].mean(axis=0), v_mean_b=ens["b"].mean(axis=0),
        v_mean_c=ens["c"].mean(axis=0),
        v_std_a=ens["a"].std(axis=0), v_std_b=ens["b"].std(axis=0),
        v_std_c=ens["c"].std(axis=0),
        v_det=v_det,
    )

    # ---- save JSON -----------------------------------------------------
    json_path = OUT / "noise_summary.json"
    # allow_nan=False -> hard error if any inf/nan slips in (keeps JSON strict/valid)
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    # ---- figure --------------------------------------------------------
    fig_path = FIGS / "fig_m3_noise_staircase.png"
    make_figure(ens, v_det, center, fig_path)

    # ---- console report ------------------------------------------------
    print("-" * 72)
    print("RESULTS")
    print(f"  deterministic W_half on grid          = {w_det_grid:.4f} "
          f"(AT-3 reference 0.250)")
    for tag, _, _, Gamma, seed in RUNS:
        r = summary["runs"][tag]
        print(f"  [{tag}] {r['label']:>18s}: "
              f"W_half(ens mean)={r['w_half_ensemble_mean']:.4f}  "
              f"per-real={r['w_half_per_real_mean']:.4f}+-"
              f"{r['w_half_per_real_std']:.4f}")
    sf = summary["survival_fraction"]
    print(f"  SURVIVAL FRACTION (paper) a G=0.140    = {sf['PAPER_NUMBER_a_G0.140']:.3f}"
          f"  ({wa:.4f}/0.250)")
    print(f"  survival fraction         b G=0.014    = {sf['b_G0.014']:.3f}"
          f"  ({wb:.4f}/0.250)")
    print(f"  sin control floor width   c G=0.140    = {wc:.4f} "
          f"(fraction {sf['c_sin_control_floor_fraction']:.3f})")
    dv = summary["dvdi_rounding"]
    rr = dv["rounding_ratio_a_over_b"]
    print(f"  dV/dI @ v=2.5 (i_dc={dv['step_center_i_dc']:.3f}) [units of Rn]:")
    print(f"    at center : det={dv['dvdi_det_at_center']:.2e} "
          f"a(G0.140)={dv['dvdi_noisy_a_at_center']:.4f} "
          f"b(G0.014)={dv['dvdi_noisy_b_at_center']:.4f}")
    print(f"    plateau-min: det={dv['dvdi_det_plateau_min']:.2e} "
          f"a(G0.140)={dv['dvdi_noisy_a_plateau_min']:.4f} "
          f"b(G0.014)={dv['dvdi_noisy_b_plateau_min']:.4f}")
    print(f"    residual dV/dI (rounding measure): a={dv['dvdi_noisy_a_plateau_min']:.3f} Rn, "
          f"b={dv['dvdi_noisy_b_plateau_min']:.3f} Rn, a/b ratio="
          f"{('%.2f' % rr) if rr else 'n/a'}")
    print("-" * 72)
    print("SEEDS: a(skew,G0.140)=%d  b(skew,G0.014)=%d  c(sin,G0.140)=%d  det=None"
          % (SEED_A, SEED_B, SEED_C))
    print("PATHS:")
    print(f"  npz : {npz_path}")
    print(f"  json: {json_path}")
    print(f"  fig : {fig_path}")
    print("=" * 72)


def make_figure(ens, v_det, center, fig_path):
    va, vb, vc = ens["a"], ens["b"], ens["c"]
    ma, sa = va.mean(0), va.std(0)
    mb, sb = vb.mean(0), vb.std(0)
    mc = vc.mean(0)
    guides = [(0.5, "v/Ω=0.5"), (1.0, "v/Ω=1"), (1.5, "v/Ω=1.5")]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 5.2))

    # ---- Panel 1: full staircase ----
    for gy, gl in guides:
        ax0.axhline(gy * OMEGA, color="0.6", lw=0.8, ls="--", zorder=0)
        ax0.text(IDC[-1], gy * OMEGA + 0.06, gl, color="0.4",
                 fontsize=8, ha="right", va="bottom")
    ax0.plot(IDC, v_det, color="k", lw=1.6, label="deterministic (Γ=0)", zorder=5)
    ax0.fill_between(IDC, ma - sa, ma + sa, color="tab:red", alpha=0.25, lw=0)
    ax0.plot(IDC, ma, color="tab:red", lw=1.3,
             label="skew Γ=0.140 (100 mK)  mean±σ")
    ax0.fill_between(IDC, mb - sb, mb + sb, color="tab:blue", alpha=0.25, lw=0)
    ax0.plot(IDC, mb, color="tab:blue", lw=1.3,
             label="skew Γ=0.014 (10 mK-eq)  mean±σ")
    ax0.plot(IDC, mc, color="tab:green", lw=1.0, ls=":",
             label="sin Γ=0.140 (control)")
    ax0.set_xlabel(r"$i_{dc}\;/\;I_c$")
    ax0.set_ylabel(r"$v=\langle d\phi/d\tau\rangle = V/(I_c R_n)$")
    ax0.set_title("Noise-rounded Shapiro staircase  ($i_{ac}=4.0$, $\\Omega=5$)")
    ax0.legend(fontsize=8, loc="upper left")
    ax0.set_xlim(IDC[0], IDC[-1])

    # ---- Panel 2: zoom on the half-step (v=2.5) ----
    win = 0.55
    lo, hi = center - win, center + win
    zm = (IDC >= lo) & (IDC <= hi)
    ax1.axhline(V_TARGET, color="0.6", lw=0.9, ls="--")
    ax1.axhline(V_TARGET + TOL, color="0.8", lw=0.7, ls=":")
    ax1.axhline(V_TARGET - TOL, color="0.8", lw=0.7, ls=":")
    ax1.text(lo, V_TARGET + TOL + 0.003, f"±tol={TOL}", color="0.4",
             fontsize=8, va="bottom")
    ax1.plot(IDC[zm], v_det[zm], color="k", lw=1.8, marker="o", ms=3,
             label="deterministic ($W_{1/2}$=0.250)")
    ax1.fill_between(IDC[zm], (ma - sa)[zm], (ma + sa)[zm],
                     color="tab:red", alpha=0.25, lw=0)
    ax1.plot(IDC[zm], ma[zm], color="tab:red", lw=1.4, marker="s", ms=3,
             label="skew Γ=0.140")
    ax1.fill_between(IDC[zm], (mb - sb)[zm], (mb + sb)[zm],
                     color="tab:blue", alpha=0.25, lw=0)
    ax1.plot(IDC[zm], mb[zm], color="tab:blue", lw=1.4, marker="^", ms=3,
             label="skew Γ=0.014")
    ax1.plot(IDC[zm], mc[zm], color="tab:green", lw=1.1, ls=":",
             label="sin Γ=0.140")
    ax1.set_xlabel(r"$i_{dc}\;/\;I_c$")
    ax1.set_ylabel(r"$v$")
    ax1.set_title(r"Half-step zoom ($v=\Omega/2=2.5$): thermal rounding")
    ax1.set_ylim(V_TARGET - 0.28, V_TARGET + 0.28)
    ax1.set_xlim(lo, hi)
    ax1.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
