"""T1 - thermometry calibration W_soft(Gamma) for the half-integer Shapiro step.

Purpose
-------
`run_survival_curve.py` answered a yes/no question ("is the half-step visible at
the production Gamma?") and its estimator is fit for that purpose: a hard
plateau-count width on a coarse i_dc grid (step 0.025), which quantises the
answer into multiples of 0.025/W_det = 12.5 %. That is unusable as a *measuring
instrument*.

This script re-runs the same physics as an inverse-problem calibration: given a
measured half-step width, what is Gamma = k_B T / E_J, hence T_e? It changes
three things and nothing else:

  1. i_dc window narrowed to [2.10, 3.10] (the deterministic half-step sits at
     2.540..2.730 for the skewed CPR and the v = 2.5 transit at 2.600..2.680 for
     the sinusoidal control, so the window carries >= 0.4 of margin on both
     sides) and refined to step 0.005 -> 201 pts. Same total cost as the old
     281-pt full-range grid, 5x the resolution where it matters.
  2. n_real 20 -> N_REAL_SIGNAL (default 160), so the standard error on the mean
     width is ~0.6 % of W_det instead of ~4 %.
  3. A SOFT (kernel) width estimator alongside the existing hard one:

         W_soft = Integral K((v(i_dc) - v_target)/sigma_k) d i_dc,   K = Gaussian

     which is continuous in the grid, has no threshold, and is monotone in Gamma
     over the usable range. The hard estimator is still computed so every number
     here can be compared to out/survival_curve.json.

  4. A SINUSOIDAL-CPR control at every Gamma. W_soft has an irreducible floor
     even with no step at all: on the ohmic branch v passes through v_target
     with finite slope s, contributing sigma_k*sqrt(2*pi)/s ~ 0.06. That floor is
     the soft-estimator analogue of the AT-1 tolerance artifact, and the control
     measures it at the same Gamma with the same seeds structure. The
     thermometer observable is the NET width

         W_net(Gamma) = W_soft_skewed(Gamma) - W_soft_sinusoidal(Gamma)

     which goes to zero when the step dies, so the calibration has a physical
     zero rather than an empirical floor.

Everything else is held at the validated values: stochastic-Heun, spp = 100
(brief Sec. 5), 150 T transient, 300 T averaging, Omega = 5, i_ac = 3.5
(= i_ac_opt at nu = -2.4), measured Beenakker tau = 0.8 CPR.

Other fillings: `--nu` derives Omega(nu) and i_ac_opt(nu) from the deterministic
fingerprint artifact out/fingerprint/<slug>.json rather than re-deriving them,
scales the averaging as n_avg = 300*Omega/5 (repo convention, holds the
v-resolution at 0.0167), and centres the same 201-point window on that filling's
deterministic half-step. The window half-width and the i_dc step stay ABSOLUTE:
the kernel is sigma_k wide in v and the ohmic branch has dv/di_dc = 1, so
scaling the step with Omega -- as the deterministic fingerprint grids do -- would
leave the kernel sampled by fewer than two points at Omega = 13.9. Outputs for
nu != -2.4 are written with a filling suffix; the reference filling keeps the
unsuffixed names.

Gamma grid: 14 points log-ish over [0.005, 0.10]. The old 11-point grid is a
subset in spirit but not in value; the range is deliberately truncated at 0.10
because out/survival_curve.json shows the estimator saturating at its floor
beyond that -- past Gamma ~ 0.1 the step carries no temperature information and
a thermometer must not pretend otherwise.

Seeds: seed = 20260804000 + round(10000*Gamma) (+ 500000 for the control
branch). Distinct from run_survival_curve.py's 20260725xxx family, so the two
result sets are independent samples and neither reuses the other's noise.

Checkpointing: one .npz per (branch, Gamma) under out/thermo_blocks/. Re-running
loads finished blocks; --fresh forces recompute. Jobs are distributed over
--workers processes (default: cpu_count - 1, capped at 4).

Outputs
-------
  out/thermometry_calibration.json   summary + per-Gamma stats + bootstrap CIs
  out/thermometry_calibration.npz    per-realization widths and staircases

The inverse map (W_net -> Gamma -> T_e per filling) is built from these by
scripts/make_thermometer.py.
"""
import os

# one BLAS thread per worker process: these are element-wise ops on ~30k-element
# arrays, so threading inside numpy only causes oversubscription across workers.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junction.analysis import step_width          # noqa: E402
from junction.cpr import normalize_cpr            # noqa: E402
from junction.noise import ensemble_staircase     # noqa: E402
from junction.rsj import OMEGA, run_rsj           # noqa: E402

OUT = ROOT / "out"
BLOCKS = OUT / "thermo_blocks"

# ---------------------------------------------------------------- grid/contract
# Defaults are the nu = -2.4 (Omega = 5) production configuration. `--nu` derives
# the corresponding numbers for any other filling; see build_config().
IDC_START, IDC_STOP, IDC_STEP = 2.10, 3.10, 0.005
IDC = np.arange(IDC_START, IDC_STOP + 1e-9, IDC_STEP)      # 201 pts
I_AC = 3.5                    # i_ac_opt at nu = -2.4 (out/fingerprint/nu_m2_4.json)
SPP = 100                     # brief Sec. 5: noisy runs need spp >= 100
N_TRANSIENT = 150
N_AVG = 300
V_TARGET = 0.5 * OMEGA        # 2.5
TOL = 0.025                   # hard-estimator plateau tolerance (unchanged)
SIGMA_K = 0.025               # soft-kernel width, set equal to TOL by choice
V_RES = OMEGA / N_AVG         # 0.016667

NU_REF = -2.4                 # the filling these module-level defaults describe

# Half-width of the i_dc window about the deterministic half-step centre, in
# ABSOLUTE reduced-current units. It must not scale with Omega: the soft kernel
# is sigma_k wide in v, the ohmic branch has dv/di_dc = 1, so the kernel is
# sigma_k = 0.025 wide in i_dc at every Omega. IDC_STEP is fixed for the same
# reason -- scaling it with Omega (as the deterministic fingerprint grids do)
# would leave the kernel sampled by <2 points at Omega = 13.9.
IDC_HALF_WINDOW = 0.5

# nu = -2.4 was computed before build_config() existed and its blocks are on
# disk under the window below. It is centred on the deterministic step
# (2.550..2.725) with >= 0.37 margin on both sides, i.e. it satisfies the same
# requirement build_config() enforces; it is pinned here only so the production
# artifact is reproducible byte-for-byte rather than recomputed for a 0.04 shift.
LEGACY_WINDOWS = {NU_REF: (IDC_START, IDC_STOP)}

GAMMAS = [0.005, 0.007, 0.010, 0.014, 0.018, 0.021, 0.025,
          0.028, 0.035, 0.042, 0.055, 0.070, 0.085, 0.100]

N_REAL_SIGNAL = 160
N_REAL_CONTROL = 80
N_BOOT = 2000
BOOT_SEED = 20260804777


def nu_slug(nu):
    return f"nu_{nu:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")


def gamma_seed(gamma, branch, nu=NU_REF):
    """Deterministic seed; branch in {'skewed', 'sin'}.

    The nu term is 0 at the reference filling, so the nu = -2.4 seeds are exactly
    those of the original production run and its blocks stay valid.
    """
    base = 20260804000 + int(round(10000.0 * gamma))
    base += (500000 if branch == "sin" else 0)
    base += 1000000 * int(round(10 * abs(nu - NU_REF)))
    return base


def output_tag(cfg):
    """Filename suffix for a run's artifacts: empty at the reference filling so
    the production names stay stable, "_nu_mX_Y" elsewhere.

    A function rather than an inline expression because it was previously a
    local named `tag` that a later loop variable of the same name clobbered,
    producing files called `thermometry_calibration[   6.7s].json`.
    """
    return "" if cfg["is_reference"] else "_" + nu_slug(cfg["nu"])


def build_config(nu):
    """Derive the full run configuration for a filling from stored artifacts.

    Omega(nu) and i_ac_opt(nu) come from the deterministic fingerprint run
    (out/fingerprint/<slug>.json, produced by scripts/run_fingerprint_nu.py), so
    this script never re-derives the dome physics -- it reuses the numbers the
    fingerprint already locked. Averaging scales as n_avg = 300*Omega/5 (the
    repo-wide convention) so the v-resolution Omega/n_avg stays 0.0167 and the
    fixed tolerance 0.025 keeps its meaning of 1.5 v-resolutions.
    """
    fp_path = ROOT / "out" / "fingerprint" / f"{nu_slug(nu)}.json"
    if not fp_path.exists():
        raise SystemExit(
            f"{fp_path} not found -- run scripts/run_fingerprint_nu.py --nu {nu} "
            "first (this script reuses its Omega and i_ac_opt, it does not "
            "re-derive them)")
    fp = json.loads(fp_path.read_text())
    omega = float(fp["Omega"])
    # the fingerprint stores i_ac_opt as a grid product, so it carries ULP noise
    # (3.5000000000000004 at nu = -2.4); snap it so cached blocks written under
    # the literal value still validate.
    i_ac = round(float(fp["i_ac_opt"]), 12)
    scale = omega / OMEGA

    if nu in LEGACY_WINDOWS:
        lo, hi = LEGACY_WINDOWS[nu]
    else:
        # centre on the deterministic half-step measured by the fingerprint run
        npz = np.load(ROOT / "out" / "fingerprint" / f"{nu_slug(nu)}.npz")
        iac_grid, idc_grid, v = npz["i_ac"], npz["i_dc"], npz["v"]
        k = int(np.argmin(np.abs(iac_grid - i_ac)))
        vv = v[k] if v.shape[0] == len(iac_grid) else v[:, k]
        on = np.abs(vv - 0.5 * omega) < TOL
        if not on.any():
            raise SystemExit(f"nu={nu}: no deterministic half-step plateau found")
        centre = 0.5 * (idc_grid[on].min() + idc_grid[on].max())
        lo, hi = centre - IDC_HALF_WINDOW, centre + IDC_HALF_WINDOW

    idc = np.arange(lo, hi + 1e-9, IDC_STEP)
    return {
        "nu": nu,
        "omega": omega,
        "i_ac": i_ac,
        "alpha": i_ac / omega,
        "idc": idc,
        "idc_lo": float(lo), "idc_hi": float(hi), "idc_step": IDC_STEP,
        "v_target": 0.5 * omega,
        "n_transient": int(np.ceil(N_TRANSIENT * scale)),
        "n_avg": int(np.ceil(N_AVG * scale)),
        "spp": SPP,
        "tol": TOL,
        "sigma_k": SIGMA_K,
        "v_res": omega / int(np.ceil(N_AVG * scale)),
        "is_reference": bool(abs(nu - NU_REF) < 1e-9),
    }


def soft_width(v, idc, v_target=V_TARGET, sigma_k=SIGMA_K):
    """Kernel-integral width: Integral exp(-((v-v_t)/sigma_k)^2/2) d i_dc.

    Continuous in the grid (no threshold), so it does not quantise into
    multiples of the i_dc step the way a plateau count does. Reduces to
    sqrt(2*pi)*sigma_k*(plateau length)/... only in the flat limit; its absolute
    normalisation is irrelevant because the calibration is built from the ratio
    to the deterministic reference and the control-subtracted net.
    """
    v = np.asarray(v, float)
    di = np.gradient(np.asarray(idc, float))
    return float(np.sum(np.exp(-0.5 * ((v - v_target) / sigma_k) ** 2) * di))


def load_cprs():
    mat = json.loads((ROOT / "material.json").read_text())
    c_raw = mat["cpr"]["harmonics_by_tau"]["0.8"]
    c, renorm = normalize_cpr(c_raw)
    return np.asarray(c_raw, float), np.asarray(c, float), float(renorm)


def _block_path(branch, seed):
    return BLOCKS / f"_{branch}_{seed}.npz"


def _job(spec):
    """Worker entry point. spec = (branch, gamma, c_list, n_real, fresh, cfg)."""
    branch, gamma, c_list, n_real, fresh, cfg = spec
    c = np.asarray(c_list, float)
    idc = np.asarray(cfg["idc"], float)
    seed = gamma_seed(gamma, branch, cfg["nu"])
    bp = _block_path(branch, seed)
    if bp.exists() and not fresh:
        d = np.load(bp)
        ok = (d["v"].shape == (n_real, len(idc))
              and int(d["seed"]) == seed
              and float(d["Gamma"]) == gamma
              and np.isclose(float(d["i_ac"]), cfg["i_ac"], rtol=0, atol=1e-12)
              and int(d["spp"]) == cfg["spp"]
              and np.allclose(d["idc"], idc))
        if ok:
            return branch, gamma, str(bp), True, 0.0
    t0 = time.time()
    v = ensemble_staircase(idc, cfg["i_ac"], c, gamma, n_real=n_real, seed=seed,
                           omega=cfg["omega"], spp=cfg["spp"],
                           n_transient=cfg["n_transient"], n_avg=cfg["n_avg"])
    dt = time.time() - t0
    np.savez(bp, v=v, seed=seed, Gamma=gamma, i_ac=cfg["i_ac"], idc=idc,
             spp=cfg["spp"], n_transient=cfg["n_transient"], n_avg=cfg["n_avg"],
             n_real=n_real, branch=branch, nu=cfg["nu"], omega=cfg["omega"])
    return branch, gamma, str(bp), False, dt


def bootstrap_ci(per_real, n_boot=N_BOOT, seed=BOOT_SEED, level=0.95):
    """Percentile bootstrap CI on the mean of a per-realization statistic."""
    x = np.asarray(per_real, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nu", type=float, default=NU_REF,
                    help="moire filling; Omega and i_ac_opt are read from "
                         "out/fingerprint/<slug>.json")
    ap.add_argument("--gammas", type=float, nargs="+", default=None,
                    help="run only these Gamma values (default: the full grid). "
                         "Blocks are per-(branch, Gamma), so a run restricted to "
                         "a subset composes with earlier subsets; the summary is "
                         "only written when every Gamma in the grid has a block.")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--workers", type=int, default=min(4, max(1, os.cpu_count() - 1)))
    ap.add_argument("--n-real", type=int, default=N_REAL_SIGNAL)
    ap.add_argument("--n-real-control", type=int, default=N_REAL_CONTROL)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    BLOCKS.mkdir(exist_ok=True)

    cfg = build_config(args.nu)
    idc = cfg["idc"]
    i_ac = cfg["i_ac"]
    omega = cfg["omega"]
    v_target = cfg["v_target"]
    sk = cfg["sigma_k"]
    tag = output_tag(cfg)

    c_raw, c, renorm = load_cprs()
    c_sin = np.array([1.0])

    print("=" * 78)
    print("T1 - thermometry calibration W_soft(Gamma), half-integer Shapiro step")
    print(f"CPR (signal) : measured Beenakker tau=0.8, renorm={renorm:.10f}")
    print(f"               c = {list(np.round(c, 6))}")
    print("CPR (control): sinusoidal c = [1]  (soft-estimator floor at same Gamma)")
    print(f"nu   : {args.nu}   Omega={omega:.4f}   i_ac={i_ac:.4f} "
          f"(alpha={cfg['alpha']:.3f})"
          + ("   [reference filling]" if cfg["is_reference"] else "   [extension]"))
    print(f"grid : i_dc [{cfg['idc_lo']:.4f}, {cfg['idc_hi']:.4f}] step {IDC_STEP} "
          f"-> {len(idc)} pts")
    print(f"noisy: stochastic-Heun spp={cfg['spp']}, {cfg['n_transient']}T transient, "
          f"{cfg['n_avg']}T avg (v_res={cfg['v_res']:.5f})")
    print(f"       n_real signal={args.n_real}, control={args.n_real_control}")
    print(f"est. : hard (tol={TOL}) + soft (sigma_k={sk}), v_target={v_target}")
    print(f"Gamma grid ({len(GAMMAS)}): {GAMMAS}")
    print(f"workers: {args.workers}")
    print("=" * 78)

    # ---------------- deterministic references (Gamma = 0) ----------------
    t0 = time.time()
    det_kw = dict(omega=omega, n_transient=cfg["n_transient"], n_avg=cfg["n_avg"])
    v_det = run_rsj(idc, np.full_like(idc, i_ac), c, **det_kw)
    v_det_sin = run_rsj(idc, np.full_like(idc, i_ac), c_sin, **det_kw)
    w_det_hard = step_width(v_det, idc, v_target, TOL)
    w_det_soft = soft_width(v_det, idc, v_target, sk)
    w_det_sin_hard = step_width(v_det_sin, idc, v_target, TOL)
    w_det_sin_soft = soft_width(v_det_sin, idc, v_target, sk)
    w_det_net = w_det_soft - w_det_sin_soft
    print(f"[det] skewed : hard={w_det_hard:.4f} soft={w_det_soft:.4f}")
    print(f"[det] sinus. : hard={w_det_sin_hard:.4f} soft={w_det_sin_soft:.4f}  "
          f"(soft floor from the ohmic transit)")
    print(f"[det] NET soft = {w_det_net:.4f}   ({time.time()-t0:.1f}s)")
    print("-" * 78)

    # ---------------- noisy ensembles, parallel over (branch, Gamma) -------
    job_cfg = dict(cfg, idc=idc.tolist())
    todo = GAMMAS if args.gammas is None else [
        g for g in GAMMAS if any(abs(g - x) < 1e-9 for x in args.gammas)]
    if args.gammas is not None:
        missing = [x for x in args.gammas
                   if not any(abs(g - x) < 1e-9 for g in GAMMAS)]
        if missing:
            raise SystemExit(f"--gammas values not on the grid: {missing}")
        print(f"restricted to {len(todo)}/{len(GAMMAS)} Gamma values: {todo}")
    jobs = ([("skewed", g, c.tolist(), args.n_real, args.fresh, job_cfg)
             for g in todo]
            + [("sin", g, c_sin.tolist(), args.n_real_control, args.fresh, job_cfg)
               for g in todo])
    # longest jobs first so the tail does not serialise
    jobs.sort(key=lambda s: (-s[3], -s[1]))

    results = {}
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for branch, gamma, path, cached, dt in ex.map(_job, jobs):
            results[(branch, gamma)] = path
            status = "[resume]" if cached else f"[{dt:6.1f}s]"
            print(f"  {status} {branch:<7} Gamma={gamma:<6} -> {pathlib.Path(path).name}"
                  f"   (elapsed {time.time()-t_start:.0f}s)")
    print("-" * 78)

    # ---------------- per-Gamma statistics ---------------------------------
    have = [g for g in GAMMAS
            if _block_path("skewed", gamma_seed(g, "skewed", args.nu)).exists()
            and _block_path("sin", gamma_seed(g, "sin", args.nu)).exists()]
    if len(have) < len(GAMMAS):
        print("-" * 78)
        print(f"PARTIAL: {len(have)}/{len(GAMMAS)} Gamma values have both branches "
              f"on disk. Missing: {[g for g in GAMMAS if g not in have]}")
        print("No summary written -- re-run with the remaining --gammas to finish.")
        return
    per_gamma = []
    store = {}
    for gamma in GAMMAS:
        # read from disk, not from `results`: with --gammas this invocation only
        # ran a subset, but the summary covers every Gamma that has both blocks.
        vs = np.load(_block_path("skewed", gamma_seed(gamma, "skewed", args.nu)))["v"]
        vc = np.load(_block_path("sin", gamma_seed(gamma, "sin", args.nu)))["v"]
        store[gamma] = (vs, vc)

        soft_s = np.array([soft_width(vs[r], idc, v_target, sk)
                           for r in range(vs.shape[0])])
        soft_c = np.array([soft_width(vc[r], idc, v_target, sk)
                           for r in range(vc.shape[0])])
        hard_s = np.array([step_width(vs[r], idc, v_target, TOL)
                           for r in range(vs.shape[0])])

        # ensemble-mean staircase estimators (old convention, for comparison)
        w_ens_hard = step_width(vs.mean(axis=0), idc, v_target, TOL)
        w_ens_soft = soft_width(vs.mean(axis=0), idc, v_target, sk)

        m_s, m_c = float(soft_s.mean()), float(soft_c.mean())
        sem_s = float(soft_s.std(ddof=1) / np.sqrt(len(soft_s)))
        sem_c = float(soft_c.std(ddof=1) / np.sqrt(len(soft_c)))
        net = m_s - m_c
        sem_net = float(np.hypot(sem_s, sem_c))
        lo_s, hi_s = bootstrap_ci(soft_s)
        lo_c, hi_c = bootstrap_ci(soft_c)

        entry = {
            "Gamma": gamma,
            "EJ_over_kBT": 1.0 / gamma,
            "seed_skewed": gamma_seed(gamma, "skewed", args.nu),
            "seed_sin": gamma_seed(gamma, "sin", args.nu),
            "n_real_skewed": int(vs.shape[0]),
            "n_real_sin": int(vc.shape[0]),
            "soft_skewed_mean": m_s, "soft_skewed_sem": sem_s,
            "soft_skewed_ci95": [lo_s, hi_s],
            "soft_sin_mean": m_c, "soft_sin_sem": sem_c,
            "soft_sin_ci95": [lo_c, hi_c],
            "soft_net_mean": net, "soft_net_sem": sem_net,
            "soft_net_ci95": [lo_s - hi_c, hi_s - lo_c],
            "soft_net_over_det": net / w_det_net,
            "hard_per_real_mean": float(hard_s.mean()),
            "hard_per_real_std": float(hard_s.std(ddof=1)),
            "hard_ens_mean_staircase": float(w_ens_hard),
            "soft_ens_mean_staircase": float(w_ens_soft),
            "survival_hard_ens_old_convention": float(w_ens_hard / w_det_hard),
        }
        per_gamma.append(entry)
        print(f"Gamma={gamma:<6} soft_s={m_s:.4f}+-{sem_s:.4f}  "
              f"soft_c={m_c:.4f}+-{sem_c:.4f}  NET={net:.4f}+-{sem_net:.4f}  "
              f"net/det={net/w_det_net:.3f}   hard_ens={w_ens_hard:.4f}")

    # ---------------- monotonicity audit -----------------------------------
    nets = np.array([e["soft_net_mean"] for e in per_gamma])
    sems = np.array([e["soft_net_sem"] for e in per_gamma])
    diffs = np.diff(nets)
    # a rise is only a real violation if it exceeds the combined 1-sigma error
    comb = np.hypot(sems[:-1], sems[1:])
    viol = [{"gamma_lo": GAMMAS[i], "gamma_hi": GAMMAS[i + 1],
             "delta": float(diffs[i]), "combined_sem": float(comb[i]),
             "n_sigma": float(diffs[i] / comb[i])}
            for i in range(len(diffs)) if diffs[i] > comb[i]]
    print("-" * 78)
    if viol:
        print(f"MONOTONICITY: {len(viol)} rise(s) beyond 1 combined SEM:")
        for v_ in viol:
            print(f"  Gamma {v_['gamma_lo']} -> {v_['gamma_hi']}: "
                  f"+{v_['delta']:.4f} ({v_['n_sigma']:.1f} sigma)")
    else:
        print("MONOTONICITY: net soft width is monotone decreasing within 1 SEM "
              "over the whole grid (invertible calibration).")

    summary = {
        "description": "T1 thermometry calibration: net soft half-step width vs "
                       "Gamma at the v0.3 production operating point (nu=-2.4, "
                       "Omega=5, i_ac=3.5, measured Beenakker tau=0.8 CPR), with "
                       "a sinusoidal-CPR control subtracting the soft-estimator "
                       "transit floor at every Gamma.",
        "purpose": "Inverse calibration W_net -> Gamma -> T_e = Gamma * E_J/k_B. "
                   "Supersedes out/survival_curve.json as a MEASUREMENT map; the "
                   "old hard-estimator survival numbers are reproduced here under "
                   "'survival_hard_ens_old_convention' for continuity.",
        "operating_point": {
            "nu": args.nu, "Omega": omega, "i_ac": i_ac, "alpha": cfg["alpha"],
            "is_reference_filling": cfg["is_reference"],
            "omega_and_iac_source": f"out/fingerprint/{nu_slug(args.nu)}.json",
            "cpr_model": "measured Beenakker short junction, tau=0.8, "
                         "normalized max=1 (material.json harmonics_by_tau['0.8'])",
            "cpr_raw": c_raw.tolist(), "cpr_normalized": c.tolist(),
            "cpr_renorm_factor": renorm,
            "control_cpr": [1.0]},
        "grid": {"i_dc_start": cfg["idc_lo"], "i_dc_stop": cfg["idc_hi"],
                 "i_dc_step": IDC_STEP, "n_idc": len(idc), "i_ac": i_ac,
                 "window_rationale": "deterministic half-step at 2.540..2.730 "
                                     "(skewed) and v=2.5 transit at 2.600..2.680 "
                                     "(sinusoidal); >=0.4 margin both sides"},
        "numerics": {"noisy_integrator": "stochastic-Heun (additive noise)",
                     "spp": SPP, "n_transient": N_TRANSIENT, "n_avg": N_AVG,
                     "n_real_signal": args.n_real,
                     "n_real_control": args.n_real_control,
                     "v_target": v_target, "tol_hard": TOL,
                     "sigma_k_soft": sk, "v_resolution": cfg["v_res"],
                     "n_transient": cfg["n_transient"], "n_avg": cfg["n_avg"],
                     "det_integrator": "RK4 spp=60 150/300 T (validated fingerprint)",
                     "n_bootstrap": N_BOOT, "bootstrap_seed": BOOT_SEED},
        "seed_formula": "20260804000 + round(10000*Gamma) (+500000 for control, +1000000*round(10*|nu-(-2.4)|) for filling)",
        "deterministic_reference": {
            "w_hard_skewed": float(w_det_hard),
            "w_soft_skewed": float(w_det_soft),
            "w_hard_sinusoidal": float(w_det_sin_hard),
            "w_soft_sinusoidal": float(w_det_sin_soft),
            "w_soft_net": float(w_det_net),
            "note": "w_hard_skewed on this refined grid should reproduce the "
                    "0.200 of out/fingerprint/nu_m2_4.json to within one i_dc "
                    "step; w_soft_sinusoidal is the ohmic-transit floor "
                    "sigma_k*sqrt(2pi)/slope, not a step."},
        "gamma_grid": GAMMAS,
        "per_gamma": per_gamma,
        "monotonicity_violations": viol,
        "usable_range_note": "Gamma grid deliberately capped at 0.10: "
                             "out/survival_curve.json shows the hard estimator "
                             "saturating at its floor beyond that, so the step "
                             "carries no temperature information there.",
    }
    (OUT / f"thermometry_calibration{tag}.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False))

    np.savez(
        OUT / f"thermometry_calibration{tag}.npz",
        gamma_grid=np.array(GAMMAS, float),
        idc=idc, i_ac=i_ac, omega=omega, v_target=v_target, nu=args.nu,
        tol_hard=TOL, sigma_k=SIGMA_K, spp=SPP,
        n_transient=N_TRANSIENT, n_avg=N_AVG,
        cpr_raw=c_raw, cpr_normalized=c, cpr_control=c_sin,
        v_det_skewed=v_det, v_det_sin=v_det_sin,
        w_det_hard=w_det_hard, w_det_soft=w_det_soft,
        w_det_sin_soft=w_det_sin_soft, w_det_net=w_det_net,
        soft_net_mean=np.array([e["soft_net_mean"] for e in per_gamma]),
        soft_net_sem=np.array([e["soft_net_sem"] for e in per_gamma]),
        soft_skewed_mean=np.array([e["soft_skewed_mean"] for e in per_gamma]),
        soft_sin_mean=np.array([e["soft_sin_mean"] for e in per_gamma]),
        v_skewed=np.stack([store[g][0] for g in GAMMAS]),
        v_sin=np.stack([store[g][1] for g in GAMMAS]),
    )

    print("-" * 78)
    print("PATHS:")
    print(f"  json: {OUT/f'thermometry_calibration{tag}.json'}")
    print(f"  npz : {OUT/f'thermometry_calibration{tag}.npz'}")
    print(f"total wall time: {time.time()-t_start:.0f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
