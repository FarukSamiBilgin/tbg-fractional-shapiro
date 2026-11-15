"""A1 - survival curve W_half(Gamma) at the v0.3 production operating point.

Contract change v0.2 -> v0.3 (DECISION_contract_v0p3.md, approved 2026-07-25):
IcRn = 60 uV is UNCHANGED, so the entire dimensionless problem is identical;
only Gamma = k_B T / E_J relabels (E_J x10 -> Gamma(100 mK) 0.140 -> 0.014).
This script maps how the paper's primary fingerprint (the half-integer Shapiro
step at v = Omega/2) survives thermal noise as a function of Gamma, and locates
the visibility threshold Gamma* at 50 % survival.

Operating point (production, nu = -2.4)
----------------------------------------
  CPR    : MEASURED Beenakker short-junction, tau = 0.8, normalised max = 1
           (material.json cpr.harmonics_by_tau["0.8"] -> junction.cpr.normalize_cpr).
           This is exactly the CPR behind out/fingerprint/nu_m2_4.json.
  Omega  : 5.0
  i_ac   : 3.5   (= i_ac_opt at nu = -2.4, out/fingerprint/nu_m2_4.json, alpha=0.7)
  i_dc   : arange(1.0, 8.0, 0.025)  (281 pts, stage-B grid)

Numerics
--------
Noisy  : stochastic Heun (junction.noise.ensemble_staircase), spp = 100
         (brief Sec. 5: fixed step, spp >= 100 for noisy runs), 150 T transient,
         300 T averaging, n_real = 20 per Gamma. One ensemble_staircase call per
         Gamma (each is 20 x 281 instances integrated in a single vectorised run).
Det.   : junction.rsj.run_rsj on the same grid (validated fingerprint numerics:
         RK4, spp = 60, 150/300 T) -> W_det ~ 0.200 (matches nu_m2_4.json).
Plateau: v_target = Omega/2 = 2.5, tol = 0.025 = 1.5 * v_resolution
         (v_resolution = Omega/n_avg = 5/300 = 0.01667, spp-independent), via
         junction.analysis.step_width.

Gamma grid (log-ish), 11 points:
  0.007, 0.014, 0.021, 0.028, 0.042, 0.07, 0.10, 0.14, 0.21, 0.28, 0.39
  0.014 = v0.3 production (100 mK, Ic=300 nA)
  0.042 = Ic=100 nA equivalent (0.014 * 300/100) -> "does 100 nA suffice?"
  0.14  = v0.2 narrow-junction archive (100 mK, Ic=30 nA)

Seeds (logged): seed = 20260725000 + round(1000*Gamma) per Gamma. Deterministic
under these seeds. Every seed is printed and stored in the output JSON/NPZ.

Headline outputs
----------------
  survival(Gamma) = W_half_ensemble_mean(Gamma) / W_det
  Gamma*  : survival crosses 0.50, located by linear interpolation in log(Gamma)
  E_J/k_BT threshold = 1 / Gamma*   (visibility criterion E_J/k_BT > 1/Gamma*)
  survival at Gamma = 0.042 -> explicit "does Ic = 100 nA suffice" answer.

Checkpoint / resume
-------------------
Each Gamma's per-realization v array is checkpointed to
out/survival_blocks/_g{seed}.npz as soon as it finishes. Re-invoking loads any
matching block instead of recomputing, so a timed-out run resumes. --fresh
forces a full recompute.
"""
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junction.analysis import step_width
from junction.cpr import normalize_cpr
from junction.noise import ensemble_staircase
from junction.rsj import OMEGA, run_rsj

# --------------------------------------------------------------------------
# Fixed contract / numerics
# --------------------------------------------------------------------------
OUT = ROOT / "out"
BLOCKS = OUT / "survival_blocks"
OUT.mkdir(exist_ok=True)
BLOCKS.mkdir(exist_ok=True)

FRESH = "--fresh" in sys.argv[1:]

IDC = np.arange(1.0, 8.0 + 1e-9, 0.025)      # 281 pts, stage-B grid
I_AC = 3.5                                    # i_ac_opt at nu = -2.4 (nu_m2_4.json)
SPP = 100                                     # brief Sec. 5: noisy spp >= 100
N_TRANSIENT = 150
N_AVG = 300
N_REAL = 20
V_TARGET = 0.5 * OMEGA                         # 2.5, the half-integer step
TOL = 0.025
V_RES = OMEGA / N_AVG                           # 2*pi/tau_avg = 0.016667

GAMMAS = [0.007, 0.014, 0.021, 0.028, 0.042, 0.07, 0.10, 0.14, 0.21, 0.28, 0.39]
GAMMA_PROD = 0.014       # v0.3 production (100 mK, Ic = 300 nA)
GAMMA_IC100 = 0.042      # Ic = 100 nA equivalent
GAMMA_ARCHIVE = 0.14     # v0.2 narrow-junction archive

NOISE_KW = dict(spp=SPP, n_transient=N_TRANSIENT, n_avg=N_AVG, n_real=N_REAL)


def gamma_seed(gamma):
    """Deterministic per-Gamma seed = 20260725000 + round(1000*Gamma)."""
    return 20260725000 + int(round(1000.0 * gamma))


def load_measured_cpr():
    """Measured Beenakker CPR, tau = 0.8, normalised via junction.cpr.normalize_cpr."""
    mat = json.loads((ROOT / "material.json").read_text())
    c_raw = mat["cpr"]["harmonics_by_tau"]["0.8"]
    c, renorm = normalize_cpr(c_raw)
    return np.asarray(c_raw, float), c, float(renorm)


def _block_path(seed):
    return BLOCKS / f"_g{seed}.npz"


def run_gamma(gamma, c, seed):
    """Compute (or load cached) per-realization v array, shape (N_REAL, n_idc)."""
    bp = _block_path(seed)
    if bp.exists() and not FRESH:
        d = np.load(bp)
        if (d["v"].shape == (N_REAL, len(IDC))
                and int(d["seed"]) == seed
                and float(d["Gamma"]) == gamma
                and float(d["i_ac"]) == I_AC
                and int(d["spp"]) == SPP):
            print(f"  [resume] loaded cached block {bp.name} "
                  f"(Gamma={gamma}, seed={seed})")
            return d["v"], True
        print(f"  [stale]  {bp.name} config mismatch -> recompute")
    t0 = time.time()
    v = ensemble_staircase(IDC, I_AC, c, gamma, seed=seed, **NOISE_KW)
    dt = time.time() - t0
    np.savez(bp, v=v, seed=seed, Gamma=gamma, i_ac=I_AC, idc=IDC,
             spp=SPP, n_transient=N_TRANSIENT, n_avg=N_AVG, n_real=N_REAL)
    print(f"  computed Gamma={gamma} seed={seed} "
          f"({N_REAL} real x {len(IDC)} pts) in {dt:.1f}s -> {bp.name}")
    return v, False


def log_interp_crossing(gammas, survival, level=0.5):
    """First Gamma (scanning low->high) where survival drops through `level`,
    by linear interpolation in log(Gamma). Returns (gamma_star, bracket) or
    (None, reason)."""
    g = np.asarray(gammas, float)
    s = np.asarray(survival, float)
    order = np.argsort(g)
    g, s = g[order], s[order]
    if s[0] < level:
        return None, "survival already below level at smallest Gamma"
    if s[-1] >= level:
        return None, "survival still above level at largest Gamma"
    for i in range(len(g) - 1):
        if s[i] >= level > s[i + 1]:
            lg0, lg1 = np.log(g[i]), np.log(g[i + 1])
            s0, s1 = s[i], s[i + 1]
            # linear in log(Gamma): find lg where s(lg) = level
            frac = (s0 - level) / (s0 - s1)
            lg_star = lg0 + frac * (lg1 - lg0)
            return float(np.exp(lg_star)), {
                "gamma_lo": float(g[i]), "gamma_hi": float(g[i + 1]),
                "survival_lo": float(s0), "survival_hi": float(s1)}
    return None, "no monotone crossing found"


def main():
    c_raw, c, renorm = load_measured_cpr()

    print("=" * 74)
    print("A1 - survival curve W_half(Gamma)  [v0.3 production operating point]")
    print(f"CPR: measured Beenakker tau=0.8 (normalised, renorm={renorm:.10f})")
    print(f"  c = {list(np.round(c, 6))}")
    print(f"grid: i_dc in [1.0, 8.0] step 0.025 ({len(IDC)} pts), i_ac={I_AC}, Omega={OMEGA}")
    print(f"noisy: stochastic-Heun spp={SPP}, {N_TRANSIENT}T transient, {N_AVG}T avg, "
          f"n_real={N_REAL}")
    print(f"v_target={V_TARGET} (Omega/2), tol={TOL}, v_res={V_RES:.5f} "
          f"(tol/v_res={TOL/V_RES:.2f})")
    print(f"Gamma grid ({len(GAMMAS)}): {GAMMAS}")
    print("seeds: " + ", ".join(f"G{g}->{gamma_seed(g)}" for g in GAMMAS))
    print("=" * 74)

    # ---- deterministic reference width on this grid (validated numerics) ----
    v_det = run_rsj(IDC, np.full_like(IDC, I_AC), c)   # RK4 spp=60, 150/300 T
    w_det = step_width(v_det, IDC, V_TARGET, TOL)
    print(f"[det] W_half = {w_det:.4f} (fingerprint nu_m2_4 reference 0.200)")
    print("-" * 74)

    # ---- ensembles per Gamma ------------------------------------------------
    per_gamma = []
    v_real_all = {}
    for gamma in GAMMAS:
        seed = gamma_seed(gamma)
        print(f"[Gamma={gamma:<6}] seed={seed}")
        v, cached = run_gamma(gamma, c, seed)
        v_real_all[gamma] = v
        v_mean = v.mean(axis=0)
        v_std = v.std(axis=0)
        w_mean = step_width(v_mean, IDC, V_TARGET, TOL)      # width of ensemble-mean staircase
        w_per = np.array([step_width(v[r], IDC, V_TARGET, TOL) for r in range(N_REAL)])
        surv_ens = w_mean / w_det                             # PRIMARY survival metric
        surv_per_mean = float(w_per.mean() / w_det)
        entry = {
            "Gamma": gamma,
            "seed": seed,
            "EJ_over_kBT": 1.0 / gamma,
            "cached": bool(cached),
            "w_half_ensemble_mean": float(w_mean),
            "w_half_per_real_mean": float(w_per.mean()),
            "w_half_per_real_std": float(w_per.std()),
            "w_half_per_real_min": float(w_per.min()),
            "w_half_per_real_max": float(w_per.max()),
            "survival_ensemble_mean": float(surv_ens),
            "survival_per_real_mean": surv_per_mean,
            "survival_per_real_std": float(w_per.std() / w_det),
            # fraction of realizations that individually retain >=50% of W_det
            "frac_real_above_50pct": float(np.mean(w_per >= 0.5 * w_det)),
        }
        per_gamma.append(entry)
        print(f"    W_half(ens mean)={w_mean:.4f}  survival(ens)={surv_ens:.3f}  "
              f"per-real={w_per.mean():.4f}+-{w_per.std():.4f}  "
              f"frac_real>=50%={entry['frac_real_above_50pct']:.2f}")

    # ---- headline: Gamma* at 50% survival (log-interpolated) ----------------
    gammas_arr = [e["Gamma"] for e in per_gamma]
    surv_arr = [e["survival_ensemble_mean"] for e in per_gamma]
    gamma_star, bracket = log_interp_crossing(gammas_arr, surv_arr, level=0.5)

    def surv_at(g):
        for e in per_gamma:
            if e["Gamma"] == g:
                return e["survival_ensemble_mean"]
        return None

    surv_prod = surv_at(GAMMA_PROD)
    surv_ic100 = surv_at(GAMMA_IC100)
    surv_archive = surv_at(GAMMA_ARCHIVE)
    ic100_suffices = (surv_ic100 is not None and surv_ic100 >= 0.5)

    headline = {
        "gamma_star_50pct_survival": gamma_star,
        "gamma_star_bracket": bracket if isinstance(bracket, dict) else None,
        "gamma_star_note": None if gamma_star is not None else bracket,
        "EJ_over_kBT_threshold": (1.0 / gamma_star) if gamma_star else None,
        "visibility_criterion": (
            f"E_J/k_BT > {1.0/gamma_star:.1f}  (i.e. Gamma < {gamma_star:.4f}) "
            "for the half-integer Shapiro step to retain >50% of its "
            "deterministic width" if gamma_star else
            "no 50% crossing within the Gamma grid"),
        "production_v0p3": {
            "Gamma": GAMMA_PROD, "survival": surv_prod,
            "EJ_over_kBT": 1.0 / GAMMA_PROD,
            "T_op_mK": 100, "Ic_nA": 300},
        "Ic100nA_equivalent": {
            "Gamma": GAMMA_IC100, "survival": surv_ic100,
            "EJ_over_kBT": 1.0 / GAMMA_IC100,
            "Ic_nA": 100,
            "suffices": bool(ic100_suffices),
            "answer": (
                f"Ic = 100 nA (Gamma = {GAMMA_IC100}) gives survival = "
                f"{surv_ic100:.3f}. "
                + ("This is >= 50%, so 100 nA SUFFICES to keep the "
                   "half-integer step visible."
                   if ic100_suffices else
                   "This is < 50%, so 100 nA does NOT suffice; the full "
                   "300 nA widening is needed.")
                if surv_ic100 is not None else "not computed")},
        "v0p2_archive": {
            "Gamma": GAMMA_ARCHIVE, "survival": surv_archive,
            "EJ_over_kBT": 1.0 / GAMMA_ARCHIVE,
            "T_op_mK": 100, "Ic_nA": 30},
    }

    summary = {
        "description": "A1 v0.3 production survival curve W_half(Gamma) of the "
                       "half-integer Shapiro step (v=Omega/2) at the measured "
                       "Beenakker CPR (tau=0.8), Omega=5, i_ac=i_ac_opt(nu=-2.4)=3.5.",
        "contract": "v0.3 (IcRn=60uV unchanged; reduced-unit results invariant; "
                    "Gamma relabels via E_J x10, currents/powers x10).",
        "operating_point": {
            "nu": -2.4, "Omega": OMEGA, "i_ac": I_AC,
            "cpr_model": "measured Beenakker short junction, tau=0.8, "
                         "normalized max=1 (material.json harmonics_by_tau['0.8'])",
            "cpr_raw": list(c_raw), "cpr_normalized": list(c),
            "cpr_renorm_factor": renorm},
        "grid": {"i_dc_start": 1.0, "i_dc_stop": 8.0, "i_dc_step": 0.025,
                 "n_idc": len(IDC), "i_ac": I_AC},
        "numerics": {"noisy_integrator": "stochastic-Heun (additive noise)",
                     "spp": SPP, "n_transient": N_TRANSIENT, "n_avg": N_AVG,
                     "n_real": N_REAL, "v_target": V_TARGET, "tol": TOL,
                     "v_resolution": V_RES, "tol_over_v_res": TOL / V_RES,
                     "det_integrator": "RK4 spp=60 150/300 T (validated fingerprint)"},
        "seed_formula": "20260725000 + round(1000*Gamma)",
        "seeds": {str(g): gamma_seed(g) for g in GAMMAS},
        "w_det_reference": float(w_det),
        "w_det_fingerprint_nu_m2_4": 0.200,
        "gamma_grid": GAMMAS,
        "per_gamma": per_gamma,
        "headline": headline,
    }

    # ---- save NPZ -----------------------------------------------------------
    npz_path = OUT / "survival_curve.npz"
    np.savez(
        npz_path,
        gamma_grid=np.array(GAMMAS, float),
        seeds=np.array([gamma_seed(g) for g in GAMMAS]),
        i_dc=IDC, i_ac=I_AC, omega=OMEGA, v_target=V_TARGET, tol=TOL,
        v_resolution=V_RES, spp=SPP, n_transient=N_TRANSIENT, n_avg=N_AVG,
        n_real=N_REAL, cpr_raw=c_raw, cpr_normalized=c, cpr_renorm=renorm,
        w_det=w_det, v_det=v_det,
        w_half_ensemble_mean=np.array([e["w_half_ensemble_mean"] for e in per_gamma]),
        w_half_per_real_mean=np.array([e["w_half_per_real_mean"] for e in per_gamma]),
        w_half_per_real_std=np.array([e["w_half_per_real_std"] for e in per_gamma]),
        survival_ensemble_mean=np.array([e["survival_ensemble_mean"] for e in per_gamma]),
        survival_per_real_mean=np.array([e["survival_per_real_mean"] for e in per_gamma]),
        frac_real_above_50pct=np.array([e["frac_real_above_50pct"] for e in per_gamma]),
        gamma_star=(gamma_star if gamma_star is not None else np.nan),
        # full per-realization staircases + ensemble stats, one stacked array per family
        v_real=np.stack([v_real_all[g] for g in GAMMAS]),          # (n_gamma, n_real, n_idc)
        v_mean=np.stack([v_real_all[g].mean(axis=0) for g in GAMMAS]),
        v_std=np.stack([v_real_all[g].std(axis=0) for g in GAMMAS]),
    )

    # ---- save JSON ----------------------------------------------------------
    json_path = OUT / "survival_curve.json"
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False))

    # ---- console report -----------------------------------------------------
    print("-" * 74)
    print("SURVIVAL CURVE (survival = W_half_ensemble_mean / W_det, W_det=%.4f)" % w_det)
    print(f"{'Gamma':>7} {'seed':>12} {'EJ/kBT':>7} {'W_half':>7} "
          f"{'surv':>6} {'per-real(mean+-std)':>22} {'f>=50%':>7}")
    for e in per_gamma:
        tag = ""
        if e["Gamma"] == GAMMA_PROD:
            tag = " <- v0.3 production"
        elif e["Gamma"] == GAMMA_IC100:
            tag = " <- Ic=100nA equiv"
        elif e["Gamma"] == GAMMA_ARCHIVE:
            tag = " <- v0.2 archive"
        print(f"{e['Gamma']:>7} {e['seed']:>12} {e['EJ_over_kBT']:>7.1f} "
              f"{e['w_half_ensemble_mean']:>7.4f} {e['survival_ensemble_mean']:>6.3f} "
              f"{e['w_half_per_real_mean']:>10.4f}+-{e['w_half_per_real_std']:<9.4f} "
              f"{e['frac_real_above_50pct']:>7.2f}{tag}")
    print("-" * 74)
    h = headline
    if gamma_star is not None:
        print(f"HEADLINE  Gamma* (50% survival) = {gamma_star:.4f}  "
              f"[bracket {bracket['gamma_lo']}..{bracket['gamma_hi']}, "
              f"surv {bracket['survival_lo']:.3f}->{bracket['survival_hi']:.3f}]")
        print(f"          visibility criterion: E_J/k_BT > {1.0/gamma_star:.1f}")
    else:
        print(f"HEADLINE  Gamma* not bracketed: {bracket}")
    print(f"          v0.3 production Gamma=0.014: survival = {surv_prod:.3f} "
          f"(E_J/k_BT = {1/GAMMA_PROD:.1f})")
    print(f"          Ic=100nA equiv Gamma=0.042 : survival = {surv_ic100:.3f} -> "
          f"{'SUFFICES' if ic100_suffices else 'does NOT suffice'}")
    print(f"          v0.2 archive   Gamma=0.14  : survival = {surv_archive:.3f}")
    print("-" * 74)
    print("PATHS:")
    print(f"  npz : {npz_path}")
    print(f"  json: {json_path}")
    print("=" * 74)


if __name__ == "__main__":
    main()
