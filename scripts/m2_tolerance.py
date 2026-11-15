"""M2 tolerance study (brief production item 3): fabrication sensitivity of the
3D photonic-crystal-slab TE-like gap and of the 145 GHz operating line.

Perturbations studied:  +-2% on r/a,  +-5% on t/a  (need MPB),  +-2% on a
(ANALYTIC - see key insight).  Reports, at the nominal design a = 638.013 um:
the shifted gap edges (GHz), midgap shift (dGHz), the margin of 145 GHz to the
nearest gap edge, the linear worst-case combination, and the midgap
sensitivities d(f_mid)/d(r/a) and d(f_mid)/d(t/a) in GHz per %.

KEY INSIGHT - the a-perturbation is EXACT and FREE (no simulation)
-----------------------------------------------------------------
Maxwell's equations are scale-invariant: the dimensionless band frequency
nu = a/lambda is a function of the DIMENSIONLESS geometry (r/a, t/a, eps) ONLY,
never of the absolute lattice constant a.  Holding r/a and t/a fixed while
changing a leaves every nu unchanged, so the physical frequency f = nu*c/a
scales as 1/a:

        a -> a*(1+da)   =>   f -> f/(1+da)   (every band, exactly).

Hence a +2% error in a shifts EVERY frequency (both gap edges, the midgap, and
any fixed-nu cavity line) by 1/1.02 - 1 = -1.961%, and a -2% error by
1/0.98 - 1 = +2.041%.  The gap moves rigidly in fractional terms; a fixed-nu
cavity line stays at the same fractional position inside it.  Only r/a and t/a
changes alter the dimensionless geometry and therefore require new MPB runs.

Method for the r/a, t/a rows  (reuses scripts/m2_at0_slab3d.py machinery)
-------------------------------------------------------------------------
Triangular air-hole lattice, Si slab eps=11.68, supercell z = 4a, run_zeven()
(TE-like, even about the mid-slab z=0 plane).  From out/m2/at0_slab3d.json the
unperturbed fundamental gap is bracketed by two BZ high-symmetry k-points:
    lower edge = band-1 (MPB band index 0, dielectric band) MAXIMUM at K,
                 reduced k = (-1/3, 1/3)
    upper edge = band-2 (MPB band index 1, air band)        MINIMUM at M,
                 reduced k = (0, 1/2)
For the small r/a (+-2%) and t/a (+-5%) perturbations here the band extrema
stay pinned at K and M (both are symmetry points of the triangular lattice), so
we evaluate ONLY those two k-points at res 20 - a few seconds per perturbation.
Both edge bands sit far below the air light line (f ~ 0.27-0.35 vs f_LL ~
0.577 at M, 0.667 at K), so no light-line reclassification occurs under these
perturbations and the single-k evaluation of each edge is valid.

Sanity: the unperturbed (r/a=0.30, t/a=0.5) two-k evaluation must reproduce the
at0_slab3d.json res-20 gap edges to < 0.2% (identical resolution -> expected to
match to numerical precision).

Checkpoint / resume
-------------------
State lives in out/m2/tolerance_ck.json.  Each invocation runs the NEXT
un-simulated config (one run_zeven, single serial process) and exits, so the
orchestrator just re-runs the same command until it prints DONE.  Once all
configs are simulated it assembles out/m2/tolerance.json.

Usage: python m2_tolerance.py <project_root>
Deterministic - no RNG.  SERIAL, one MPB process at a time (CPU discipline).
"""
import json
import math
import os
import platform
import sys
import time

import meep as mp
from meep import mpb

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."

# ---- locked nominal geometry (brief / at0) ---------------------------------
R0 = 0.30                 # nominal r/a
T0 = 0.50                 # nominal t/a
EPS_CORE = 11.68
SUPERCELL_Z = 4.0         # supercell height in z (units of a)
RES = 20                  # in-plane resolution (px/a); z gets RES*SUPERCELL_Z px
NUM_BANDS = 8             # match at0_slab3d.py exactly for reproducibility

# ---- tolerance box ----------------------------------------------------------
DR_FRAC = 0.02            # +-2% on r/a
DT_FRAC = 0.05            # +-5% on t/a
DA_FRAC = 0.02            # +-2% on a (analytic)

# ---- edge k-points (from out/m2/at0_slab3d.json) ----------------------------
# lower gap edge: band index 0 (band 1) max at K = (-1/3, 1/3)
# upper gap edge: band index 1 (band 2) min at M = (0, 1/2)
K_LOWER = (-1.0 / 3.0, 1.0 / 3.0)   # K
K_UPPER = (0.0, 0.5)                # M
LOWER_BAND_IDX = 0                  # dielectric band -> lower edge
UPPER_BAND_IDX = 1                  # air band        -> upper edge

# ---- configs to simulate (label -> (r/a, t/a)) ------------------------------
CONFIGS = [
    ("baseline", R0, T0),               # sanity vs at0_slab3d.json
    ("r_lo", R0 * (1 - DR_FRAC), T0),   # r/a = 0.294
    ("r_hi", R0 * (1 + DR_FRAC), T0),   # r/a = 0.306
    ("t_lo", R0, T0 * (1 - DT_FRAC)),   # t/a = 0.475
    ("t_hi", R0, T0 * (1 + DT_FRAC)),   # t/a = 0.525
]

CK = f"{ROOT}/out/m2/tolerance_ck.json"
OUT = f"{ROOT}/out/m2/tolerance.json"
LOG = f"{ROOT}/out/m2/tol_run.log"

os.makedirs(f"{ROOT}/out/m2", exist_ok=True)


def logline(msg):
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


# ---- geometry / solver (mirrors m2_at0_slab3d.py make_solver) ---------------
def make_solver(r_over_a, t_over_a, resolution):
    lattice = mp.Lattice(
        size=mp.Vector3(1, 1, SUPERCELL_Z),
        basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
        basis2=mp.Vector3(math.sqrt(3) / 2, -0.5),
    )
    geometry = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, t_over_a),
                 material=mp.Medium(epsilon=EPS_CORE)),
        mp.Cylinder(radius=r_over_a, height=mp.inf, material=mp.air),
    ]
    return mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=geometry,
        default_material=mp.air,
        resolution=resolution,
        num_bands=NUM_BANDS,
    )


def run_edges(r_over_a, t_over_a):
    """run_zeven() at the two edge k-points [K, M]; return (lo_edge, hi_edge, dt).
    lo_edge = band 0 at K (all_freqs[0][0]); hi_edge = band 1 at M (all_freqs[1][1]).
    """
    ms = make_solver(r_over_a, t_over_a, RES)
    ms.k_points = [
        mp.Vector3(K_LOWER[0], K_LOWER[1], 0),   # K -> lower edge
        mp.Vector3(K_UPPER[0], K_UPPER[1], 0),   # M -> upper edge
    ]
    t0 = time.time()
    ms.run_zeven()
    dt = time.time() - t0
    freqs = [[float(v) for v in row] for row in ms.all_freqs]
    lo_edge = freqs[0][LOWER_BAND_IDX]   # band 0 at K
    hi_edge = freqs[1][UPPER_BAND_IDX]   # band 1 at M
    return lo_edge, hi_edge, dt, freqs


# ---- checkpoint I/O ---------------------------------------------------------
def load_ck():
    if os.path.exists(CK):
        return json.load(open(CK))
    return {
        "task": "M2 tolerance study (brief item 3): r/a, t/a, a fabrication sensitivity",
        "rng": "deterministic - no RNG",
        "meep_version": mp.__version__,
        "config": {
            "r0": R0, "t0": T0, "eps_core": EPS_CORE,
            "supercell_z_a": SUPERCELL_Z, "res_inplane": RES,
            "res_z_px": int(RES * SUPERCELL_Z), "num_bands": NUM_BANDS,
            "solver": "run_zeven (TE-like, even about z=0)",
            "dr_frac": DR_FRAC, "dt_frac": DT_FRAC, "da_frac": DA_FRAC,
            "edge_kpoints": {
                "lower_edge_K_reduced": list(K_LOWER),
                "lower_band_index": LOWER_BAND_IDX,
                "upper_edge_M_reduced": list(K_UPPER),
                "upper_band_index": UPPER_BAND_IDX,
            },
        },
        "runs": {},          # label -> {r_over_a,t_over_a,lo_edge,hi_edge,all_freqs}
        "timings": [],
    }


def save_ck(ck):
    with open(CK, "w") as fh:
        json.dump(ck, fh, indent=1)


# ---- main state machine -----------------------------------------------------
def main():
    ck = load_ck()
    for label, r, t in CONFIGS:
        if label in ck["runs"]:
            continue
        logline(f"config '{label}': r/a={r:.6f}, t/a={t:.6f}  (res {RES}, 2 k-points)")
        lo, hi, dt, freqs = run_edges(r, t)
        ck["runs"][label] = {
            "r_over_a": r, "t_over_a": t,
            "lo_edge_nu": lo, "hi_edge_nu": hi,
            "midgap_nu": 0.5 * (lo + hi), "gap_width_nu": hi - lo,
            "all_freqs_2k": freqs,
        }
        ck["timings"].append({"config": label, "seconds": round(dt, 1), "res": RES})
        save_ck(ck)
        logline(f"  lo(K,b0)={lo:.6f}  hi(M,b1)={hi:.6f}  "
                f"midgap={0.5*(lo+hi):.6f}  gap={hi-lo:.6f}  ({dt:.1f}s)")
        remaining = [c[0] for c in CONFIGS if c[0] not in ck["runs"]]
        if remaining:
            logline(f"MORE: {len(remaining)} configs remain {remaining}; re-invoke.")
        else:
            logline("all configs simulated; re-invoke to finalize.")
        return
    finalize(ck)


def finalize(ck):
    neff = json.load(open(f"{ROOT}/out/m2/at0_neff.json"))
    at0 = json.load(open(f"{ROOT}/out/m2/at0_slab3d.json"))
    C = neff["constants"]["c_um_GHz"]
    F0 = neff["constants"]["f0_GHz"]                       # 145.0
    LAMBDA0 = neff["constants"]["lambda0_um"]
    A_DESIGN = at0["lattice_constant"]["a_final_um_post3d"]  # 638.0131800239526 um

    def to_ghz(nu, a=A_DESIGN):
        return nu * C / a

    runs = ck["runs"]
    base = runs["baseline"]
    lo0_nu, hi0_nu = base["lo_edge_nu"], base["hi_edge_nu"]
    mid0_nu = 0.5 * (lo0_nu + hi0_nu)
    lo0, hi0, mid0 = to_ghz(lo0_nu), to_ghz(hi0_nu), to_ghz(mid0_nu)

    # ---- SANITY: reproduce at0_slab3d.json res-20 edges ----
    at0_lo_nu = at0["gap_3d"]["gap_edges_a_over_lambda"][0]
    at0_hi_nu = at0["gap_3d"]["gap_edges_a_over_lambda"][1]
    at0_mid_nu = at0["gap_3d"]["midgap_nu3d_a_over_lambda"]
    sanity = {
        "at0_res20_lo_nu": at0_lo_nu, "my_res20_lo_nu": lo0_nu,
        "lo_rel_err_pct": 100.0 * (lo0_nu - at0_lo_nu) / at0_lo_nu,
        "at0_res20_hi_nu": at0_hi_nu, "my_res20_hi_nu": hi0_nu,
        "hi_rel_err_pct": 100.0 * (hi0_nu - at0_hi_nu) / at0_hi_nu,
        "at0_res20_mid_nu": at0_mid_nu, "my_res20_mid_nu": mid0_nu,
        "mid_rel_err_pct": 100.0 * (mid0_nu - at0_mid_nu) / at0_mid_nu,
        "note": "same res (20), same 2 edge k-points -> expect match to num. precision",
    }
    sanity["max_abs_rel_err_pct"] = max(abs(sanity["lo_rel_err_pct"]),
                                        abs(sanity["hi_rel_err_pct"]),
                                        abs(sanity["mid_rel_err_pct"]))
    sanity["within_0p2pct"] = bool(sanity["max_abs_rel_err_pct"] < 0.2)

    # ---- per-perturbation rows (r/a, t/a from MPB; edges in GHz at A_DESIGN) --
    def row_from_run(label):
        r = runs[label]
        lo_g, hi_g = to_ghz(r["lo_edge_nu"]), to_ghz(r["hi_edge_nu"])
        mid_g = 0.5 * (lo_g + hi_g)
        return {
            "r_over_a": r["r_over_a"], "t_over_a": r["t_over_a"],
            "lo_edge_nu": r["lo_edge_nu"], "hi_edge_nu": r["hi_edge_nu"],
            "lo_edge_GHz": lo_g, "hi_edge_GHz": hi_g,
            "midgap_GHz": mid_g, "gap_width_GHz": hi_g - lo_g,
            "midgap_shift_GHz": mid_g - F0,
            "lo_edge_shift_GHz": lo_g - lo0, "hi_edge_shift_GHz": hi_g - hi0,
            "margin_to_lo_GHz": F0 - lo_g,      # 145 above lower edge (want > 0)
            "margin_to_hi_GHz": hi_g - F0,      # 145 below upper edge (want > 0)
            "margin_min_GHz": min(F0 - lo_g, hi_g - F0),
            "f0_in_gap": bool(lo_g <= F0 <= hi_g),
        }

    # ---- a-perturbation rows (analytic; nu unchanged, edges scale as 1/a) -----
    def row_from_a(da_frac):
        a = A_DESIGN * (1 + da_frac)
        lo_g, hi_g = to_ghz(lo0_nu, a), to_ghz(hi0_nu, a)
        mid_g = 0.5 * (lo_g + hi_g)
        return {
            "a_um": a, "da_frac": da_frac,
            "lo_edge_GHz": lo_g, "hi_edge_GHz": hi_g,
            "midgap_GHz": mid_g, "gap_width_GHz": hi_g - lo_g,
            "midgap_shift_GHz": mid_g - F0,
            "lo_edge_shift_GHz": lo_g - lo0, "hi_edge_shift_GHz": hi_g - hi0,
            "margin_to_lo_GHz": F0 - lo_g, "margin_to_hi_GHz": hi_g - F0,
            "margin_min_GHz": min(F0 - lo_g, hi_g - F0),
            "f0_in_gap": bool(lo_g <= F0 <= hi_g),
            "note": "nu fixed by scale invariance; f = nu*c/a (exact)",
        }

    rows = {
        "baseline": row_from_run("baseline"),
        "r_lo (-2%)": row_from_run("r_lo"),
        "r_hi (+2%)": row_from_run("r_hi"),
        "t_lo (-5%)": row_from_run("t_lo"),
        "t_hi (+5%)": row_from_run("t_hi"),
        "a_lo (-2%)": row_from_a(-DA_FRAC),
        "a_hi (+2%)": row_from_a(+DA_FRAC),
    }

    # ---- sensitivities (central differences, GHz per % of parameter) ----------
    # r/a spans -2%..+2% => 4 percentage points; t/a spans -5%..+5% => 10 pts.
    r_lo_r, r_hi_r = rows["r_lo (-2%)"], rows["r_hi (+2%)"]
    t_lo_r, t_hi_r = rows["t_lo (-5%)"], rows["t_hi (+5%)"]

    dmid_dr = (r_hi_r["midgap_GHz"] - r_lo_r["midgap_GHz"]) / (2 * 100 * DR_FRAC)
    dmid_dt = (t_hi_r["midgap_GHz"] - t_lo_r["midgap_GHz"]) / (2 * 100 * DT_FRAC)
    dlo_dr = (r_hi_r["lo_edge_GHz"] - r_lo_r["lo_edge_GHz"]) / (2 * 100 * DR_FRAC)
    dhi_dr = (r_hi_r["hi_edge_GHz"] - r_lo_r["hi_edge_GHz"]) / (2 * 100 * DR_FRAC)
    dlo_dt = (t_hi_r["lo_edge_GHz"] - t_lo_r["lo_edge_GHz"]) / (2 * 100 * DT_FRAC)
    dhi_dt = (t_hi_r["hi_edge_GHz"] - t_lo_r["hi_edge_GHz"]) / (2 * 100 * DT_FRAC)
    # a: f = nu*c/a  =>  df/d(a%) = -f/100 per 1% change in a (exact, per edge).
    dlo_da = -lo0 / 100.0
    dhi_da = -hi0 / 100.0
    dmid_da = -mid0 / 100.0

    sensitivities = {
        "d_fmid_d_rOa_GHz_per_pct": dmid_dr,
        "d_fmid_d_tOa_GHz_per_pct": dmid_dt,
        "d_fmid_d_a_GHz_per_pct": dmid_da,
        "d_flo_d_rOa_GHz_per_pct": dlo_dr, "d_fhi_d_rOa_GHz_per_pct": dhi_dr,
        "d_flo_d_tOa_GHz_per_pct": dlo_dt, "d_fhi_d_tOa_GHz_per_pct": dhi_dt,
        "d_flo_d_a_GHz_per_pct": dlo_da, "d_fhi_d_a_GHz_per_pct": dhi_da,
        "note": ("central differences on GHz edges/midgap at a=%.3f um; "
                 "GHz per 1%% change in the named parameter. a-derivatives "
                 "analytic (scale invariance)." % A_DESIGN),
    }

    # ---- linear worst-case combination over the full tolerance box -----------
    # Move each edge the WORST direction for keeping 145 in-gap:
    #   lower margin (145 - f_lo): worst when f_lo is highest -> add |slopes|*range
    #   upper margin (f_hi - 145): worst when f_hi is lowest  -> subtract |slopes|*range
    r_rng, t_rng, a_rng = 100 * DR_FRAC, 100 * DT_FRAC, 100 * DA_FRAC  # in %
    lo_up = abs(dlo_dr) * r_rng + abs(dlo_dt) * t_rng + abs(dlo_da) * a_rng
    hi_dn = abs(dhi_dr) * r_rng + abs(dhi_dt) * t_rng + abs(dhi_da) * a_rng
    f_lo_worst = lo0 + lo_up          # highest plausible lower edge
    f_hi_worst = hi0 - hi_dn          # lowest plausible upper edge
    margin_lo_worst = F0 - f_lo_worst
    margin_hi_worst = F0 * 0 + (f_hi_worst - F0)
    # worst-case midgap excursion magnitude (each param at its extreme, same sign)
    mid_excursion = (abs(dmid_dr) * r_rng + abs(dmid_dt) * t_rng
                     + abs(dmid_da) * a_rng)

    worst_case = {
        "method": "linear superposition of per-parameter slopes over |dr|<=2%, |dt|<=5%, |da|<=2%",
        "baseline_gap_GHz": [lo0, hi0],
        "baseline_gap_width_GHz": hi0 - lo0,
        "f_lo_worst_high_GHz": f_lo_worst,
        "f_hi_worst_low_GHz": f_hi_worst,
        "worst_case_lower_margin_GHz": margin_lo_worst,
        "worst_case_upper_margin_GHz": margin_hi_worst,
        "worst_case_min_margin_GHz": min(margin_lo_worst, margin_hi_worst),
        "max_midgap_excursion_GHz": mid_excursion,
        "worst_case_gap_shrink_to_GHz": f_hi_worst - f_lo_worst,
        "f0_stays_in_gap_worst_case": bool(margin_lo_worst > 0 and margin_hi_worst > 0),
        "note": ("145 GHz is the fixed physical operating line; edges move under "
                 "fabrication error. Worst case moves BOTH edges toward 145 GHz "
                 "simultaneously (an over-conservative bound - the a-error cannot "
                 "actually push both edges inward at once)."),
    }

    result = {
        "task": ck["task"],
        "rng": "deterministic - no RNG",
        "versions": {"python": platform.python_version(),
                     "meep_mpb": mp.__version__},
        "key_insight_a_analytic": (
            "nu=a/lambda depends only on r/a,t/a,eps (Maxwell scale invariance); "
            "f=nu*c/a scales as 1/a, so +2% in a => -1.961% in every frequency, "
            "-2% in a => +2.041%. No simulation needed for a."),
        "config": ck["config"],
        "constants": {"f0_GHz": F0, "lambda0_um": LAMBDA0, "c_um_GHz": C,
                      "a_design_um": A_DESIGN},
        "sanity_vs_at0_slab3d": sanity,
        "baseline_GHz": {"lo": lo0, "hi": hi0, "mid": mid0,
                         "gap_width": hi0 - lo0,
                         "margin_each_side": min(F0 - lo0, hi0 - F0)},
        "table": rows,
        "sensitivities": sensitivities,
        "worst_case": worst_case,
        "timings_seconds": ck["timings"],
    }

    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=1)

    logline("FINALIZED:")
    logline(f"  SANITY vs at0 (res20): max rel err {sanity['max_abs_rel_err_pct']:.4f}% "
            f"(<0.2%: {sanity['within_0p2pct']})")
    logline(f"  baseline gap = [{lo0:.3f}, {hi0:.3f}] GHz, mid {mid0:.3f}, "
            f"width {hi0-lo0:.3f} GHz")
    for k in ["r_lo (-2%)", "r_hi (+2%)", "t_lo (-5%)", "t_hi (+5%)",
              "a_lo (-2%)", "a_hi (+2%)"]:
        rw = rows[k]
        logline(f"  {k:>11}: gap [{rw['lo_edge_GHz']:.3f}, {rw['hi_edge_GHz']:.3f}] "
                f"mid {rw['midgap_GHz']:.3f} (d {rw['midgap_shift_GHz']:+.3f}) "
                f"min-margin {rw['margin_min_GHz']:.3f} GHz  in-gap={rw['f0_in_gap']}")
    logline(f"  sensitivity d(f_mid)/d(r/a) = {dmid_dr:+.4f} GHz/%")
    logline(f"  sensitivity d(f_mid)/d(t/a) = {dmid_dt:+.4f} GHz/%")
    logline(f"  sensitivity d(f_mid)/d(a)   = {dmid_da:+.4f} GHz/%")
    logline(f"  WORST-CASE: lower margin {margin_lo_worst:.3f} GHz, "
            f"upper margin {margin_hi_worst:.3f} GHz, "
            f"145 in-gap = {worst_case['f0_stays_in_gap_worst_case']}")
    logline(f"  saved {OUT}")
    logline("DONE")


if __name__ == "__main__":
    main()
