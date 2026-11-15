"""M3 TASK A3 - two-branch noisy nu-scan (paper Fig. 3b overlay data).

For each filling nu in {-2.4, -2.5, -2.6, -2.7, -2.8} we run a Langevin
(stochastic-Heun) ensemble of the v = Omega/2 half-integer Shapiro step at the
deterministic AT-3 optimum drive i_ac = i_ac_opt(nu), on the SAME bias grid /
Omega-scaled averaging as the deterministic fingerprint (out/fingerprint/
nu_m2_*.json), and measure how much of the deterministic half-step width
w_half_max(nu) survives thermal rounding.

Two thermal branches per nu (separating intrinsic Omega(nu) suppression from
thermal suppression), from the contract change v0.2 -> v0.3 (IcRn = 60 uV
unchanged, so ALL reduced-unit deterministic results stand; only Gamma
relabels and absolute currents/powers scale x10):

  v0.2 (narrow junction, archive branch): Gamma_v02(nu) = json "Gamma"
        (0.140 at nu=-2.4  ...  0.389 at nu=-2.8)
  v0.3 (wide junction x10, production):  Gamma_v03(nu) = Gamma_v02(nu) / 10
        (0.014 at nu=-2.4  ...  0.039 at nu=-2.8)

Measured-CPR ensemble: cpr_harmonics as stored in each nu json (already
max-normalized; identical Beenakker tau=0.8 shape across nu).

Numerics (v-resolution held fixed, matching the fingerprint runs exactly):
  omega        = Omega(nu)                             (from json)
  i_ac         = i_ac_opt(nu)                           (from json)
  i_dc grid    = omega * arange(0.2, 1.6+1e-9, 0.005)   (281 pts, fingerprint grid)
  spp          = 100                                    (brief Sec. 5, noisy)
  n_transient  = ceil(150 * Omega/5)
  n_avg        = ceil(300 * Omega/5)
  tol          = 0.025  (= 1.5 * v_res, v_res = Omega/n_avg = 0.0167 by constr.)
  n_real       = 20 independent realizations
  seed         = 20260725100 + round(10*|nu|*10) + (0 v0.2 / 1 v0.3)  [logged]

Deliverables per (nu, branch):
  W_half (ensemble-mean staircase width, reduced-current units) at v = Omega/2
  survival_fraction = W_half(ens-mean staircase) / w_half_max(nu)   [PRIMARY,
      project convention: cf. noise_summary.json, DECISION A1 "50% survival
      of the deterministic 0.200", memory note "0.014 -> 70% survival"]
  plus per-realization width mean/std/min/max and a per-realization survival
  count fraction (realizations retaining >= 50% of w_half_max) as diagnostics.

Headline: the separation statement at nu = -2.8 (Omega = 13.9 edge excursion):
  the v0.2 branch survival (expected ~floor) vs the v0.3 branch (expected
  substantial), quantified.

Checkpoint / resume: each (nu, branch) ensemble is cached to a per-run .npz
block in the session scratchpad as soon as it finishes; re-invoking loads the
cached block (grid/seed/Gamma guarded) instead of recomputing, so a timed-out
run resumes. Pass --fresh to force a full recompute.

Deterministic under the logged seeds. numpy only (no MEEP / no GPU).
Owned files: scripts/run_nu_survival.py, out/nu_survival.json, out/nu_survival.npz.
"""
import json
import math
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junction.analysis import step_width
from junction.noise import ensemble_staircase

# --------------------------------------------------------------------------
# Fixed contract / numerics
# --------------------------------------------------------------------------
OUT = ROOT / "out"
FP = OUT / "fingerprint"
OUT.mkdir(exist_ok=True)

# checkpoint blocks live in the session scratchpad (not an owned out/ file)
CKPT = pathlib.Path(
    r"C:\Users\PROFDR~1.SAM\AppData\Local\Temp\claude"
    r"\C--Users-prof-dr-f-sami-Desktop-TBG-FOTON-K-JJ--claude-worktrees-m3-junction-kickoff-5b7ffa"
    r"\25b168c7-1c23-4306-a67b-2b331560356e\scratchpad\nusurv_ckpt"
)
CKPT.mkdir(parents=True, exist_ok=True)

FRESH = "--fresh" in sys.argv[1:]

NU_LIST = [-2.4, -2.5, -2.6, -2.7, -2.8]
SPP = 100                 # brief Sec. 5: noisy spp >= 100
N_REAL = 20
TOL = 0.025
DI_STEP = 0.005           # i_dc/Omega step (fingerprint grid)
SEED_BASE = 20260725100
OMEGA_CONTRACT = 5.0
SURV_THRESH = 0.5         # per-realization "survives" if width >= 50% of w_det


def seed_for(nu, branch):
    """seed = 20260725100 + round(10*|nu|*10) + (0 v0.2 / 1 v0.3)."""
    return SEED_BASE + int(round(10 * abs(nu) * 10)) + (0 if branch == "v02" else 1)


def load_fingerprint():
    """Map nu -> fingerprint dict, keyed off the json's own 'nu' field."""
    table = {}
    for p in sorted(FP.glob("nu_m2_*.json")):
        d = json.loads(p.read_text())
        table[round(float(d["nu"]), 3)] = (d, p.name)
    return table


def run_branch(nu, branch, fp, verbose=True):
    """Compute (or load cached) per-realization v array, shape (N_REAL, n_idc)."""
    omega = float(fp["Omega"])
    i_ac = float(fp["i_ac_opt"])
    c = np.array(fp["cpr_harmonics"], float)
    gamma_v02 = float(fp["Gamma"])
    gamma = gamma_v02 if branch == "v02" else gamma_v02 / 10.0
    seed = seed_for(nu, branch)

    scale = omega / OMEGA_CONTRACT
    n_transient = int(math.ceil(150 * scale))
    n_avg = int(math.ceil(300 * scale))
    idc = omega * np.arange(0.2, 1.6 + 1e-9, DI_STEP)   # 281 pts, fingerprint grid

    slug = f"nu_{nu:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
    bp = CKPT / f"_nusurv_{slug}_{branch}.npz"
    if bp.exists() and not FRESH:
        b = np.load(bp)
        if (b["v"].shape == (N_REAL, len(idc))
                and int(b["seed"]) == seed
                and abs(float(b["Gamma"]) - gamma) < 1e-12
                and abs(float(b["omega"]) - omega) < 1e-12):
            if verbose:
                print(f"  [resume] {bp.name} (seed={seed}, Gamma={gamma:.5g})")
            return b["v"], dict(omega=omega, i_ac=i_ac, gamma=gamma, seed=seed,
                                n_transient=n_transient, n_avg=n_avg, idc=idc,
                                c=c, gamma_v02=gamma_v02)
        if verbose:
            print(f"  [stale]  {bp.name} config mismatch -> recompute")

    t0 = time.time()
    v = ensemble_staircase(idc, i_ac, c, gamma, n_real=N_REAL, seed=seed,
                           omega=omega, spp=SPP,
                           n_transient=n_transient, n_avg=n_avg)
    dt = time.time() - t0
    np.savez(bp, v=v, seed=seed, Gamma=gamma, omega=omega, i_ac=i_ac, idc=idc,
             spp=SPP, n_transient=n_transient, n_avg=n_avg, n_real=N_REAL)
    if verbose:
        print(f"  computed {branch} Gamma={gamma:.5g} seed={seed} "
              f"({N_REAL}x{len(idc)}, nt={n_transient}, na={n_avg}) in {dt:.1f}s")
    return v, dict(omega=omega, i_ac=i_ac, gamma=gamma, seed=seed,
                   n_transient=n_transient, n_avg=n_avg, idc=idc, c=c,
                   gamma_v02=gamma_v02)


def analyze(v, meta, w_det):
    """Widths + survival metrics for one (nu, branch) ensemble."""
    omega, idc = meta["omega"], meta["idc"]
    v_target = 0.5 * omega
    v_mean = v.mean(axis=0)
    w_ens_mean = step_width(v_mean, idc, v_target, TOL)           # width of mean staircase
    w_per = np.array([step_width(v[r], idc, v_target, TOL)
                      for r in range(v.shape[0])])
    surv_prim = w_ens_mean / w_det if w_det > 0 else 0.0          # PRIMARY convention
    surv_per = float(w_per.mean()) / w_det if w_det > 0 else 0.0
    surv_count = float(np.mean(w_per >= SURV_THRESH * w_det))     # frac >= 50% of w_det
    return {
        "v_target": v_target,
        "w_half_ens_mean": float(w_ens_mean),
        "w_half_perreal_mean": float(w_per.mean()),
        "w_half_perreal_std": float(w_per.std()),
        "w_half_perreal_min": float(w_per.min()),
        "w_half_perreal_max": float(w_per.max()),
        "w_half_perreal": w_per.tolist(),
        "survival_fraction": float(surv_prim),
        "survival_fraction_perreal_mean": float(surv_per),
        "survival_count_frac_ge50pct": surv_count,
    }, v_mean, v.std(axis=0), w_per


def main():
    print("=" * 74)
    print("M3 TASK A3 - two-branch noisy nu-scan (Fig. 3b overlay)")
    print(f"nu = {NU_LIST}  |  spp={SPP}, n_real={N_REAL}, tol={TOL}")
    print(f"grid = Omega*arange(0.2,1.6,{DI_STEP}) (281 pts); nt/na = ceil(150/300*Omega/5)")
    print(f"seed = {SEED_BASE} + round(10*|nu|*10) + (0 v0.2 / 1 v0.3)")
    print("=" * 74)

    fptable = load_fingerprint()

    summary = {
        "description": "M3 TASK A3 two-branch noisy nu-scan: Langevin thermal "
                       "survival of the v=Omega/2 half-integer Shapiro step at "
                       "each nu's i_ac_opt, v0.2 (narrow) vs v0.3 (wide x10) "
                       "thermal branch. Fig 3b noisy overlay data.",
        "contract_note": "v0.2->v0.3: IcRn=60uV unchanged -> reduced-unit "
                         "deterministic results (Omega, i_ac_opt, w_half_max) "
                         "identical; Gamma relabels 0.14->0.014 (x1/10 in E_J), "
                         "absolute currents/powers x10. Gamma_v03 = Gamma_v02/10.",
        "numerics": {"integrator": "stochastic-Heun (additive noise)",
                     "spp": SPP, "n_real": N_REAL, "tol": TOL,
                     "idc_grid": f"Omega*arange(0.2,1.6+1e-9,{DI_STEP}) [281 pts]",
                     "n_transient_rule": "ceil(150*Omega/5)",
                     "n_avg_rule": "ceil(300*Omega/5)",
                     "v_resolution_rule": "Omega/n_avg (held ~0.0167)",
                     "cpr": "measured Beenakker tau=0.8, json cpr_harmonics"},
        "seed_rule": f"{SEED_BASE} + round(10*|nu|*10) + (0 v0.2 / 1 v0.3)",
        "survival_definitions": {
            "survival_fraction": "W_half(ensemble-mean staircase) / w_half_max(nu) "
                                 "[PRIMARY, project convention]",
            "survival_fraction_perreal_mean": "mean_r W_half(real r) / w_half_max(nu)",
            "survival_count_frac_ge50pct": "fraction of realizations with "
                                           "W_half >= 0.5 * w_half_max(nu)",
        },
        "per_nu": {},
    }

    # for the NPZ dump
    npz = {}

    for nu in NU_LIST:
        fp, fname = fptable[round(nu, 3)]
        omega = float(fp["Omega"])
        i_ac = float(fp["i_ac_opt"])
        w_det = float(fp["w_half_max"])
        ic_na = float(fp["Ic_nA"])
        gamma_v02 = float(fp["Gamma"])
        print(f"[nu={nu:+.1f}]  ({fname})  Omega={omega:.4f}  i_ac_opt={i_ac:.4f}  "
              f"w_half_max={w_det:.5f}  Gamma_v02={gamma_v02:.5g}  Gamma_v03={gamma_v02/10:.5g}")

        entry = {
            "nu": nu, "fingerprint_file": fname,
            "Ic_nA_v02": ic_na, "Ic_nA_v03": ic_na * 10.0,
            "Omega": omega, "i_ac_opt": i_ac,
            "w_half_max_det": w_det,
            "w_half_max_det_nA_v02": w_det * ic_na,
            "w_half_max_det_nA_v03": w_det * ic_na * 10.0,
            "Gamma_v02": gamma_v02, "Gamma_v03": gamma_v02 / 10.0,
            "branches": {},
        }

        for branch in ("v02", "v03"):
            v, meta = run_branch(nu, branch, fp)
            res, v_mean, v_std, w_per = analyze(v, meta, w_det)
            res["Gamma"] = meta["gamma"]
            res["seed"] = meta["seed"]
            res["n_transient"] = meta["n_transient"]
            res["n_avg"] = meta["n_avg"]
            res["v_resolution"] = omega / meta["n_avg"]
            entry["branches"][branch] = res
            slug = f"nu_{nu:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")
            npz[f"v_real_{slug}_{branch}"] = v
            npz[f"v_mean_{slug}_{branch}"] = v_mean
            npz[f"v_std_{slug}_{branch}"] = v_std
            npz[f"idc_{slug}"] = meta["idc"]
            print(f"    {branch}: W_half(ens-mean)={res['w_half_ens_mean']:.5f}  "
                  f"survival={res['survival_fraction']:.3f}  "
                  f"per-real={res['w_half_perreal_mean']:.5f}+-{res['w_half_perreal_std']:.5f}  "
                  f"surv>=50%={res['survival_count_frac_ge50pct']:.2f}")

        # per-nu separation ratio (v0.3 vs v0.2 survival)
        s02 = entry["branches"]["v02"]["survival_fraction"]
        s03 = entry["branches"]["v03"]["survival_fraction"]
        entry["separation_v03_over_v02"] = (s03 / s02) if s02 > 1e-9 else None
        entry["separation_v03_minus_v02"] = s03 - s02
        summary["per_nu"][f"{nu:+.1f}"] = entry

    # ---- headline: the separation statement at nu = -2.8 ----------------
    e28 = summary["per_nu"]["-2.8"]
    b28_02 = e28["branches"]["v02"]
    b28_03 = e28["branches"]["v03"]
    summary["headline_nu_-2.8"] = {
        "Omega": e28["Omega"],
        "note": "Omega=13.9 edge excursion (partly reflects the Rn(nu)-fixed "
                "caveat); largest intrinsic + thermal suppression of the scan.",
        "w_half_max_det": e28["w_half_max_det"],
        "Gamma_v02": e28["Gamma_v02"], "Gamma_v03": e28["Gamma_v03"],
        "v02_survival_fraction": b28_02["survival_fraction"],
        "v03_survival_fraction": b28_03["survival_fraction"],
        "v02_W_half": b28_02["w_half_ens_mean"],
        "v03_W_half": b28_03["w_half_ens_mean"],
        "v02_survival_count_frac_ge50pct": b28_02["survival_count_frac_ge50pct"],
        "v03_survival_count_frac_ge50pct": b28_03["survival_count_frac_ge50pct"],
        "separation_v03_over_v02": e28["separation_v03_over_v02"],
        "statement": (
            f"At nu=-2.8 (Omega={e28['Omega']:.2f}, Gamma_v02={e28['Gamma_v02']:.3f} "
            f"vs Gamma_v03={e28['Gamma_v03']:.4f}), the half-integer step "
            f"survival is {b28_02['survival_fraction']:.2f} on the v0.2 "
            f"(narrow-junction) branch vs {b28_03['survival_fraction']:.2f} on "
            f"the v0.3 (wide-junction x10) branch."
        ),
    }

    # ---- save JSON + NPZ ------------------------------------------------
    json_path = OUT / "nu_survival.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    npz_path = OUT / "nu_survival.npz"
    meta_arr = dict(
        nu_list=np.array(NU_LIST, float), spp=SPP, n_real=N_REAL, tol=TOL,
        seed_base=SEED_BASE,
        seeds=np.array([[seed_for(nu, "v02"), seed_for(nu, "v03")]
                        for nu in NU_LIST]),
        Gamma_v02=np.array([float(fptable[round(nu, 3)][0]["Gamma"]) for nu in NU_LIST]),
        Gamma_v03=np.array([float(fptable[round(nu, 3)][0]["Gamma"]) / 10 for nu in NU_LIST]),
        Omega=np.array([float(fptable[round(nu, 3)][0]["Omega"]) for nu in NU_LIST]),
        i_ac_opt=np.array([float(fptable[round(nu, 3)][0]["i_ac_opt"]) for nu in NU_LIST]),
        w_half_max_det=np.array([float(fptable[round(nu, 3)][0]["w_half_max"]) for nu in NU_LIST]),
        surv_v02=np.array([summary["per_nu"][f"{nu:+.1f}"]["branches"]["v02"]["survival_fraction"] for nu in NU_LIST]),
        surv_v03=np.array([summary["per_nu"][f"{nu:+.1f}"]["branches"]["v03"]["survival_fraction"] for nu in NU_LIST]),
    )
    np.savez(npz_path, **npz, **meta_arr)

    # ---- console report -------------------------------------------------
    print("-" * 74)
    print("RESULTS  (survival = W_half[ens-mean] / w_half_max_det)")
    print(f"  {'nu':>5} {'Omega':>7} | {'G_v02':>6} {'surv02':>7} {'W02':>7} | "
          f"{'G_v03':>6} {'surv03':>7} {'W03':>7} | {'v03/v02':>8}")
    for nu in NU_LIST:
        e = summary["per_nu"][f"{nu:+.1f}"]
        b02, b03 = e["branches"]["v02"], e["branches"]["v03"]
        sep = e["separation_v03_over_v02"]
        print(f"  {nu:>5.1f} {e['Omega']:>7.3f} | "
              f"{e['Gamma_v02']:>6.3f} {b02['survival_fraction']:>7.3f} {b02['w_half_ens_mean']:>7.4f} | "
              f"{e['Gamma_v03']:>6.4f} {b03['survival_fraction']:>7.3f} {b03['w_half_ens_mean']:>7.4f} | "
              f"{(('%.2f' % sep) if sep else 'n/a'):>8}")
    print("-" * 74)
    print("HEADLINE: " + summary["headline_nu_-2.8"]["statement"])
    print("-" * 74)
    print("SEEDS:")
    for nu in NU_LIST:
        print(f"  nu={nu:+.1f}: v0.2 seed={seed_for(nu,'v02')}  v0.3 seed={seed_for(nu,'v03')}")
    print("PATHS:")
    print(f"  json: {json_path}")
    print(f"  npz : {npz_path}")
    print("=" * 74)


if __name__ == "__main__":
    main()
