"""M2 device-specific CAVITY-LINE fabrication tolerance at the device a.

Open item, PROJECT_CLOSING_REPORT Sec. 5.3 (first bullet): the +-2%/+-5%/+-2%
(r/a, t/a, a) tolerance box in out/m2/tolerance.json was computed on the BULK
gap at the AT-0 midgap-centred a = 638.013 um.  The DEVICE uses the cavity-line
a = 586.735 um (photonics.json:geometry.a_um_precise), where the binding margin
is the +7.5 GHz distance from the 145 GHz cavity line DOWN to the lower gap
edge, not the ~+15 GHz bulk-gap margins.  Nobody had re-simulated how the L3
cavity LINE itself moves under the box.  This script does exactly that.

What is measured here (and what is not)
---------------------------------------
MEASURED (new 3D FDTD runs): the isolated-L3 cavity resonance nu0 under
  * r/a -> 0.30*(1 +- 0.02)      (ALL holes, incl. the rdef=1.15 defect ring)
  * t/a -> 0.50*(1 +- 0.05)
  * one worst-case (r, t) sign stack chosen from the measured single-
    perturbation signs, in the direction that shrinks the LOWER margin most.
REUSED (no new bulk bands): the gap-edge motion under the same perturbations,
  taken from out/m2/tolerance.json:table.<row>.lo_edge_nu / hi_edge_nu (MPB
  run_zeven, res 20, K/M edge k-points).  Those are DIMENSIONLESS (nu = a/lambda)
  and therefore transfer to the device a by pure Maxwell scale invariance:
  f = nu * c / a_device.  No bulk re-run is needed or performed.
ANALYTIC (no run): the a-error.  nu depends only on (r/a, t/a, eps), so
  a -> a(1+da) sends every frequency f -> f/(1+da).  The cavity line and BOTH
  gap edges move together: +-2% in a moves the 145 GHz line by -+2.9 GHz but
  changes the margin only by the same 1/(1+da) factor (~+-0.15 GHz on 7.5 GHz).
  The a-error is a cavity/qubit DETUNING axis, not an in-gap axis.

Cavity method (established, unchanged -- scripts/m2_3d_l3.py)
-------------------------------------------------------------
The exact AT-2 winner single-point setup used for the stored cavity line:
isolated 3D L3 in the 16a x 12*sqrt3/2 a x 4.5a supercell, PML 1a on all six
faces, s = 0.20, rdef = 1.15, res 18 WITH subpixel averaging (--eps-avg),
--lossless, Hz dipole at the cavity centre, mirror phases (+1,-1,+1),
off-plane harminv monitor at (0.30, 0.20, 0.0), fcen 0.29 / df 0.06, chunked
trailing-window harminv.  This is scripts/m2_3d_l3.py driven as a module -- no
new solver, no new geometry generator, no new analysis.

The 16a cell's Q (8576) is a KNOWN lateral-leakage artifact of the small cell
(decisions.md 2026-07-24 "AT-2 chain v2 complete"; physical ceiling 2.7-3.0e4).
That does NOT matter here: this study reports FREQUENCY shifts, and the mode
frequency is converged in the 16a cell -- the same cell defined the stored
nu0 = 0.2837860 that fixes the device a in the first place, so every shift is
measured against its own baseline in an identical cell.  Q is not used.

Ring-down length: 1200 periods (chunks of 300).  In the stored 12000-period
deep run the frequency was already nu = 0.2837879 after the first 1200-period
chunk vs 0.2837860 at 12000 periods -- a 1.9e-6 truncation error in nu, i.e.
0.001 GHz at the device a, against perturbation shifts of order 1 GHz.  The
baseline case here re-runs that identical 1200-period window, so it must
reproduce the stored chunk-1 value essentially exactly (determinism check).

The ONE deliberate deviation from m2_3d_l3.build_geometry
----------------------------------------------------------
The original computes the hole-centre cut-offs xmax/ymax from R_OVER_A:
    xmax = SX/2 - DPML - R_OVER_A - PML_PAD
At nominal r/a = 0.30 this gives xmax = 6.500, and the odd-row holes at
x = +-6.5 sit EXACTLY on the boundary (included, since the test is `>`).  A
+2% r perturbation would push xmax to 6.494 and silently DELETE those 8
far-corner holes -- a discrete change in hole COUNT that is not a fabrication
effect and would contaminate the r_hi row.  So here the lattice EXTENT is
frozen at the nominal r/a = 0.30 while only the hole RADII are perturbed
(which is what an etch-bias error physically does).  Everything else is a
verbatim copy.  Case "baseline" asserts the patched generator reproduces the
original object-for-object (centres + radii + slab thickness) before any
perturbation is applied; the assertion result is stored in the output JSON.

Convention: the box is MULTIPLICATIVE on r/a (r -> r*(1+dr)), matching
out/m2/tolerance.json.  On the rdef=1.15 defect ring this is 0.0069 a absolute
vs 0.0060 a on the mirror holes; an additive etch bias would differ by <0.001 a.

Usage (inside WSL m2env; driven by scripts/m2_device_tolerance.sh)
------------------------------------------------------------------
  mpirun -np 8 python m2_device_tolerance.py run <root> --case <label> \
         [--periods 1200] [--chunk 300]
  python m2_device_tolerance.py aggregate <root>

Cases: baseline r_hi r_lo t_hi t_lo stack
  ("stack" reads the partial out/m2/device_tolerance.json to pick the sign
   combination that minimises the lower margin, then runs it.)

Per-case raw output: out/m2/devtol/l3_3d_devtol_<case>_s020_rdef115_res18_lossless.json
Aggregated deliverable: out/m2/device_tolerance.json (rewritten after EVERY
completed case -> partial progress survives interruption).
Deterministic - no RNG anywhere.
"""
import argparse
import json
import math
import os
import platform
import sys
import time

# ---- device / operating constants ------------------------------------------
A_DEV_UM = 586.735322        # photonics.json:geometry.a_um_precise (cavity-line a)
A_BULK_UM = 638.0131800239526  # out/m2/at0_slab3d.json (bulk-midgap-centred a)
C_UM_GHZ = 299792.458
F_OP_GHZ = 145.0             # physical operating line (qubit side), fixed
NU0_STORED = 0.28378600407268306   # out/m2/l3_3d_deep_s020_rdef115_res18_lossless.json
NU0_STORED_1200P = 0.28378793196610247  # same file, chunks[0] (1200-period window)

R0, T0 = 0.30, 0.50
DR_FRAC, DT_FRAC, DA_FRAC = 0.02, 0.05, 0.02

# cavity knobs locked to the AT-2 winner
S_SHIFT, RDEF, RES, FCEN, DF = 0.20, 1.15, 18, 0.29, 0.06

# case -> (r/a scale, t/a scale); "stack" resolved at run time
CASES = {
    "baseline": (1.0, 1.0),
    "r_hi": (1.0 + DR_FRAC, 1.0),
    "r_lo": (1.0 - DR_FRAC, 1.0),
    "t_hi": (1.0, 1.0 + DT_FRAC),
    "t_lo": (1.0, 1.0 - DT_FRAC),
}
SINGLE_ORDER = ["r_hi", "r_lo", "t_hi", "t_lo"]

# bulk-study row keys (out/m2/tolerance.json:table) matching each case
BULK_ROW = {
    "baseline": "baseline",
    "r_hi": "r_hi (+2%)",
    "r_lo": "r_lo (-2%)",
    "t_hi": "t_hi (+5%)",
    "t_lo": "t_lo (-5%)",
}


def dev_ghz(nu):
    """Reduced frequency -> GHz at the DEVICE lattice constant."""
    return nu * C_UM_GHZ / A_DEV_UM


def case_paths(root, case):
    outdir = os.path.join(root, "out", "m2", "devtol")
    base = f"l3_3d_devtol_{case}_s{int(round(S_SHIFT * 100)):03d}_rdef{int(round(RDEF * 100)):03d}_res{RES}_lossless"
    return outdir, os.path.join(outdir, base + ".json")


# =============================================================================
# RUN MODE  (needs meep; must be launched under mpirun inside WSL)
# =============================================================================
def make_patched_build_geometry(L3, mp, r_lattice):
    """Verbatim copy of L3.build_geometry with the lattice EXTENT frozen at
    r_lattice (nominal r/a) so a radius perturbation cannot change the hole
    COUNT.  Radii and slab thickness read the (possibly perturbed) module
    constants L3.R_OVER_A / L3.T_OVER_A at call time."""
    SQRT3 = L3.SQRT3

    def build_geometry(s, lossy, fcen, rdef=1.0):
        r_over_a = L3.R_OVER_A
        t_over_a = L3.T_OVER_A
        sigma_D = (2.0 * math.pi * fcen * L3.TAN_DELTA) if lossy else 0.0
        slab_mat = mp.Medium(epsilon=L3.EPS_CORE, D_conductivity=sigma_D)
        geom = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, t_over_a),
                         center=mp.Vector3(), material=slab_mat)]
        r_ring = r_over_a * rdef
        # ---- FROZEN extents (the single deviation; see module docstring) ----
        xmax = L3.SX / 2.0 - L3.DPML - r_lattice - L3.PML_PAD
        ymax = L3.SY / 2.0 - L3.DPML - r_lattice - L3.PML_PAD
        jmax = int(math.ceil(ymax / (SQRT3 / 2.0))) + 1
        imax = int(math.ceil(xmax)) + 2
        for j in range(-jmax, jmax + 1):
            y = j * (SQRT3 / 2.0)
            if abs(y) > ymax:
                continue
            off = 0.5 * (j % 2)
            for i in range(-imax, imax + 1):
                x = i + off
                if abs(x) > xmax:
                    continue
                if j == 0 and off == 0.0 and i in (-1, 0, 1):
                    continue
                xc = x
                rr = r_ring if (L3._is_end_hole(j, off, i)
                                or L3._is_transverse_ring(j, off, i)) else r_over_a
                if j == 0 and off == 0.0 and i == 2:
                    xc = 2.0 + s
                elif j == 0 and off == 0.0 and i == -2:
                    xc = -(2.0 + s)
                geom.append(mp.Cylinder(radius=rr, height=mp.inf,
                                        center=mp.Vector3(xc, y, 0.0),
                                        material=mp.air))
        return geom
    return build_geometry


def _geom_signature(geom, mp):
    sig = []
    for g in geom:
        if isinstance(g, mp.Cylinder):
            sig.append(("C", round(g.center.x, 12), round(g.center.y, 12),
                        round(g.center.z, 12), round(g.radius, 12)))
        else:
            sig.append(("B", round(g.size.z, 12)))
    return sig


def resolve_stack(root):
    """Pick the (r, t) sign combination that shrinks the lower margin most,
    from the already-measured single-perturbation rows (linear superposition
    across the two axes, exact per axis)."""
    agg_path = os.path.join(root, "out", "m2", "device_tolerance.json")
    agg = json.load(open(agg_path))
    rows = {p["case"]: p for p in agg["perturbations"]}
    missing = [c for c in SINGLE_ORDER if c not in rows]
    if missing:
        raise SystemExit(f"stack needs all single perturbations first; missing {missing}")
    best = None
    for rc in ("r_hi", "r_lo"):
        for tc in ("t_hi", "t_lo"):
            dm = rows[rc]["margin_shift_GHz"] + rows[tc]["margin_shift_GHz"]
            if best is None or dm < best[0]:
                best = (dm, rc, tc)
    dm, rc, tc = best
    fr = CASES[rc][0]
    ft = CASES[tc][1]
    return fr, ft, rc, tc, dm


def cmd_check(args):
    """No-FDTD preflight: import the established module, patch the geometry
    generator, assert object-for-object identity at nominal parameters, and
    report the hole count / radii for every case."""
    import meep as mp
    import m2_3d_l3 as L3

    L3.SX, L3.NY_ROWS = 16.0, 12
    L3.SY = L3.NY_ROWS * L3.SQRT3 / 2.0
    orig = L3.build_geometry
    patched = make_patched_build_geometry(L3, mp, R0)
    ref = _geom_signature(orig(S_SHIFT, False, FCEN, RDEF), mp)
    new = _geom_signature(patched(S_SHIFT, False, FCEN, RDEF), mp)
    print(f"nominal identity: {ref == new}  (objects={len(ref)})")
    L3.build_geometry = patched
    rows = dict(CASES)
    # what the ORIGINAL generator would have done under the same perturbation
    for case, (fr, ft) in rows.items():
        L3.R_OVER_A, L3.T_OVER_A = R0 * fr, T0 * ft
        sp = _geom_signature(patched(S_SHIFT, False, FCEN, RDEF), mp)
        so = _geom_signature(orig(S_SHIFT, False, FCEN, RDEF), mp)
        radii = sorted({s[4] for s in sp if s[0] == "C"})
        print(f"  {case:>9}: r/a={L3.R_OVER_A:.6f} t/a={L3.T_OVER_A:.6f} "
              f"objects(frozen)={len(sp)} objects(unfrozen)={len(so)} "
              f"radii={[round(x, 5) for x in radii]} slab_z={sp[0][1]}")
    print(f"SZ (fixed) = {L3.SZ}")


def cmd_run(args):
    import meep as mp
    import m2_3d_l3 as L3

    root = args.root
    case = args.case
    outdir, ckpath = case_paths(root, case)
    os.makedirs(outdir, exist_ok=True)

    # ---- resume: a completed case is never re-run --------------------------
    if os.path.exists(ckpath):
        try:
            prev = json.load(open(ckpath))
            if prev.get("status") == "done":
                if mp.am_really_master():
                    print(f"[devtol] case '{case}' already done -> skip", flush=True)
                return
        except Exception:
            pass

    if case == "stack":
        fr, ft, rc, tc, dm_pred = resolve_stack(root)
        stack_meta = {"from": [rc, tc], "predicted_margin_shift_GHz": dm_pred}
    else:
        fr, ft = CASES[case]
        stack_meta = None

    orig_bg = L3.build_geometry
    patched = make_patched_build_geometry(L3, mp, R0)

    # ---- geometry-identity assertion at NOMINAL parameters ------------------
    ref_sig = _geom_signature(orig_bg(S_SHIFT, False, FCEN, RDEF), mp)
    new_sig = _geom_signature(patched(S_SHIFT, False, FCEN, RDEF), mp)
    geom_identical = bool(ref_sig == new_sig)
    n_obj_nominal = len(ref_sig)
    if not geom_identical:
        raise SystemExit("patched build_geometry differs from the original at "
                         "nominal parameters -- refusing to run")

    # ---- apply the perturbation --------------------------------------------
    L3.R_OVER_A = R0 * fr
    L3.T_OVER_A = T0 * ft
    L3.build_geometry = patched
    pert_sig = _geom_signature(patched(S_SHIFT, False, FCEN, RDEF), mp)
    n_obj_pert = len(pert_sig)
    if n_obj_pert != n_obj_nominal:
        raise SystemExit(f"hole count changed under perturbation "
                         f"({n_obj_nominal} -> {n_obj_pert}) -- extents not frozen")

    if mp.am_really_master():
        print(f"[devtol] case={case} r/a={L3.R_OVER_A:.6f} t/a={L3.T_OVER_A:.6f} "
              f"SZ={L3.SZ} objects={n_obj_pert} geom_identity_at_nominal="
              f"{geom_identical}", flush=True)

    t0 = time.time()
    sys.argv = [
        "m2_3d_l3.py", root,
        "--s", f"{S_SHIFT}", "--rdef", f"{RDEF}", "--res", f"{RES}",
        "--eps-avg", "--lossless",
        "--until-periods", f"{args.periods}", "--chunk-periods", f"{args.chunk}",
        "--fcen", f"{FCEN}", "--df", f"{DF}",
        "--outdir", outdir, "--label", f"devtol_{case}",
    ]
    L3.main()
    wall = time.time() - t0

    # ---- stamp the case metadata into the run's own checkpoint file ---------
    if mp.am_really_master():
        d = json.load(open(ckpath))
        d["devtol"] = {
            "case": case, "r_scale": fr, "t_scale": ft,
            "r_over_a": R0 * fr, "t_over_a": T0 * ft,
            "lattice_extent_r_frozen_at": R0,
            "geometry_identity_at_nominal": geom_identical,
            "n_geometry_objects": n_obj_pert,
            "SZ_fixed_a": L3.SZ,
            "wall_seconds": round(wall, 1),
            "mpi_np": int(os.environ.get("OMPI_COMM_WORLD_SIZE", "0")) or None,
            "stack_meta": stack_meta,
        }
        json.dump(d, open(ckpath, "w"), indent=1)
        h = d["result"]["headline_in_gap"]
        print(f"[devtol] DONE case={case} nu0={h['nu']:.7f} "
              f"f_dev={dev_ghz(h['nu']):.4f} GHz  ({wall:.0f} s)", flush=True)


# =============================================================================
# AGGREGATE MODE  (pure stdlib; safe to run on Windows or in WSL)
# =============================================================================
def _read_case(root, case):
    _, ckpath = case_paths(root, case)
    if not os.path.exists(ckpath):
        return None
    d = json.load(open(ckpath))
    # "devtol" is stamped only AFTER m2_3d_l3.main() marks the run done, so a
    # file with status=done but no devtol block is a run killed in that window.
    if d.get("status") != "done" or "devtol" not in d:
        return None
    return d


def cmd_aggregate(args):
    root = args.root
    tol = json.load(open(os.path.join(root, "out", "m2", "tolerance.json")))
    ph = json.load(open(os.path.join(root, "photonics.json")))
    tab = tol["table"]

    lo_nu = {k: tab[BULK_ROW[k]]["lo_edge_nu"] for k in BULK_ROW}
    hi_nu = {k: tab[BULK_ROW[k]]["hi_edge_nu"] for k in BULK_ROW}
    lo0_dev = dev_ghz(lo_nu["baseline"])
    hi0_dev = dev_ghz(hi_nu["baseline"])

    base = _read_case(root, "baseline")
    if base is None:
        raise SystemExit("baseline case not finished yet -- nothing to aggregate")
    nu_base = base["result"]["headline_in_gap"]["nu"]
    f_base = dev_ghz(nu_base)
    margin_base = f_base - lo0_dev

    baseline_block = {
        "nu0": nu_base,
        "f0_GHz": f_base,
        "nu0_stored_deep_12000p": NU0_STORED,
        "nu0_stored_chunk1_1200p": NU0_STORED_1200P,
        "dev_vs_stored_deep_nu": nu_base - NU0_STORED,
        "dev_vs_stored_deep_GHz": dev_ghz(nu_base) - dev_ghz(NU0_STORED),
        "dev_vs_stored_1200p_nu": nu_base - NU0_STORED_1200P,
        "dev_vs_stored_1200p_GHz": dev_ghz(nu_base) - dev_ghz(NU0_STORED_1200P),
        "lo_edge_GHz": lo0_dev,
        "hi_edge_GHz": hi0_dev,
        "margin_to_lo_GHz": margin_base,
        "margin_to_hi_GHz": hi0_dev - f_base,
        "geometry_identity_at_nominal": base["devtol"]["geometry_identity_at_nominal"],
        "n_geometry_objects": base["devtol"]["n_geometry_objects"],
        "detuning_vs_145GHz_GHz": f_base - F_OP_GHZ,
    }

    perts = []
    timings = [{"case": "baseline", "wall_seconds": base["devtol"]["wall_seconds"],
                "periods": base["params"]["total_periods"]}]

    order = SINGLE_ORDER + ["stack"]
    for case in order:
        d = _read_case(root, case)
        if d is None:
            continue
        nu = d["result"]["headline_in_gap"]["nu"]
        f = dev_ghz(nu)
        dv = d["devtol"]
        if case == "stack":
            rc, tc = dv["stack_meta"]["from"]
            d_lo = ((lo_nu[rc] - lo_nu["baseline"])
                    + (lo_nu[tc] - lo_nu["baseline"]))
            d_hi = ((hi_nu[rc] - hi_nu["baseline"])
                    + (hi_nu[tc] - hi_nu["baseline"]))
            src = (f"out/m2/tolerance.json:table['{BULK_ROW[rc]}'] + "
                   f"table['{BULK_ROW[tc]}'] (linear superposition of the two "
                   "measured single-axis edge shifts)")
        else:
            d_lo = lo_nu[case] - lo_nu["baseline"]
            d_hi = hi_nu[case] - hi_nu["baseline"]
            src = f"out/m2/tolerance.json:table['{BULK_ROW[case]}'].lo_edge_nu"
        edge_lo = lo0_dev + dev_ghz(d_lo)
        edge_hi = hi0_dev + dev_ghz(d_hi)
        margin = f - edge_lo
        perts.append({
            "case": case,
            "params": {"r_over_a": dv["r_over_a"], "t_over_a": dv["t_over_a"],
                       "r_scale": dv["r_scale"], "t_scale": dv["t_scale"],
                       "stack_from": (dv["stack_meta"]["from"]
                                      if dv.get("stack_meta") else None)},
            "nu0": nu,
            "f0_GHz": f,
            "shift_GHz": f - f_base,
            "edge_shift_GHz": dev_ghz(d_lo),
            "edge_shift_hi_GHz": dev_ghz(d_hi),
            "lo_edge_GHz": edge_lo,
            "hi_edge_GHz": edge_hi,
            "edge_src": src,
            "margin_GHz": margin,
            "margin_shift_GHz": margin - margin_base,
            "margin_to_hi_GHz": edge_hi - f,
            "in_gap": bool(edge_lo <= f <= edge_hi),
            "detuning_vs_145GHz_GHz": f - F_OP_GHZ,
            "harminv_err": d["result"]["headline_in_gap"]["err"],
            "Q_16a_cell": d["result"]["headline_in_gap"]["Q"],
        })
        timings.append({"case": case, "wall_seconds": dv["wall_seconds"],
                        "periods": d["params"]["total_periods"]})

    # ---- analytic a-axis ----------------------------------------------------
    a_err = {
        "note": ("Maxwell scale invariance: nu = a/lambda depends only on "
                 "(r/a, t/a, eps), so a -> a(1+da) sends EVERY frequency to "
                 "f/(1+da) -- cavity line and both gap edges together. No "
                 "simulation performed or needed."),
        "da_frac": [-DA_FRAC, +DA_FRAC],
        "cavity_line_GHz": {"a_minus_2pct": f_base / (1 - DA_FRAC),
                            "a_plus_2pct": f_base / (1 + DA_FRAC)},
        "cavity_line_shift_GHz": {
            "a_minus_2pct": f_base / (1 - DA_FRAC) - f_base,
            "a_plus_2pct": f_base / (1 + DA_FRAC) - f_base},
        "margin_GHz": {"a_minus_2pct": margin_base / (1 - DA_FRAC),
                       "a_plus_2pct": margin_base / (1 + DA_FRAC)},
        "margin_shift_GHz": {
            "a_minus_2pct": margin_base / (1 - DA_FRAC) - margin_base,
            "a_plus_2pct": margin_base / (1 + DA_FRAC) - margin_base},
        "interpretation": ("the a-error is a cavity/qubit DETUNING axis "
                           f"(-+{F_OP_GHZ * DA_FRAC / (1 + DA_FRAC):.2f} to "
                           f"{F_OP_GHZ * DA_FRAC / (1 - DA_FRAC):.2f} GHz on the "
                           "145 GHz line), NOT an in-gap axis: it moves line and "
                           "edges together, so the margin changes only by the "
                           "1/(1+da) factor."),
    }

    # ---- verdict ------------------------------------------------------------
    rows = {p["case"]: p for p in perts}
    stack = rows.get("stack")
    verdict = None
    worst_case_stack = None
    box_scan = None

    def _vertex_quadratic(m_neg, m_pos, half):
        """Margin-shift response along one axis through 3 points
        (-half, m_neg), (0, 0), (+half, m_pos).  Returns
        (a, b, argmin_in_box, min_value_in_box) for m(d) = a d^2 + b d."""
        a = (m_pos + m_neg) / (2.0 * half * half)
        b = (m_pos - m_neg) / (2.0 * half)
        cands = [-half, +half]
        if a > 0:                      # convex -> interior vertex is a minimum
            v = -b / (2.0 * a)
            if -half < v < half:
                cands.append(v)
        vals = [(a * d * d + b * d, d) for d in cands]
        mv, md = min(vals)
        return a, b, md, mv

    if all(c in rows for c in SINGLE_ORDER):
        ar, br, dr_min, mr_min = _vertex_quadratic(
            rows["r_lo"]["margin_shift_GHz"], rows["r_hi"]["margin_shift_GHz"],
            DR_FRAC * 100)
        at, bt, dt_min, mt_min = _vertex_quadratic(
            rows["t_lo"]["margin_shift_GHz"], rows["t_hi"]["margin_shift_GHz"],
            DT_FRAC * 100)
        box_scan = {
            "purpose": ("the margin is NOT monotonic in the box: both t signs "
                        "and r+2% INCREASE the margin, so the box minimum is "
                        "not at a corner and corner-stacking alone would miss "
                        "it. A separable quadratic through the three measured "
                        "points on each axis locates the in-box minimum."),
            "r_axis": {"quad_a_GHz_per_pct2": ar, "quad_b_GHz_per_pct": br,
                       "argmin_pct": dr_min, "min_margin_shift_GHz": mr_min,
                       "shape": "concave (minimum at an endpoint)" if ar < 0
                                else "convex (interior minimum)"},
            "t_axis": {"quad_a_GHz_per_pct2": at, "quad_b_GHz_per_pct": bt,
                       "argmin_pct": dt_min, "min_margin_shift_GHz": mt_min,
                       "shape": "concave (minimum at an endpoint)" if at < 0
                                else "convex (interior minimum)"},
            "model_box_min_margin_GHz": margin_base + mr_min + mt_min,
            "model_box_min_at": {"dr_pct": dr_min, "dt_pct": dt_min},
            "caveat": ("3 points per axis -> quadratic interpolation, not a "
                       "measured point; used only to confirm that no interior "
                       "point beats the measured worst case by a meaningful "
                       "amount."),
        }

    if stack is not None:
        # ---- honest worst case: minimum over EVERY evaluated (r,t) point,
        #      not just the corner stack (see box_scan).
        all_pts = [("baseline", margin_base)] + [(p["case"], p["margin_GHz"])
                                                 for p in perts]
        worst_case, m_worst_rt = min(all_pts, key=lambda kv: kv[1])
        model_min = (box_scan["model_box_min_margin_GHz"] if box_scan
                     else m_worst_rt)
        m_rt = min(m_worst_rt, model_min)
        m_full = m_rt / (1 + DA_FRAC)    # +2% a is the worst a-sign for margin
        singles = [p for p in perts if p["case"] != "stack"]
        binding = (min(singles, key=lambda p: p["margin_shift_GHz"])["case"]
                   if singles else "none")
        # linear-superposition cross-check delivered by the stack RUN
        pred = (rows["r_lo"]["margin_shift_GHz"] if "r_lo" in
                stack["params"]["stack_from"] else rows["r_hi"]["margin_shift_GHz"])
        pred += (rows["t_lo"]["margin_shift_GHz"] if "t_lo" in
                 stack["params"]["stack_from"] else rows["t_hi"]["margin_shift_GHz"])
        verdict = {
            "min_margin_GHz": m_full,
            "min_margin_rt_only_GHz": m_rt,
            "worst_measured_rt_case": worst_case,
            "worst_measured_rt_margin_GHz": m_worst_rt,
            "in_gap": bool(m_full > 0),
            "binding_axis": binding,
            "baseline_margin_GHz": margin_base,
            "margin_erosion_GHz": margin_base - m_full,
            "margin_erosion_pct_of_baseline": 100.0 * (margin_base - m_full) / margin_base,
            "note": ("min_margin = worst of {every measured (r,t) point, the "
                     "separable-quadratic in-box minimum} times the analytic "
                     "1/(1+da) factor at da=+2% (the worst a sign for the "
                     "margin). 'in_gap' asks whether the CAVITY LINE stays "
                     "inside the perturbed gap. NOTE the worst point is NOT "
                     "the corner stack: both t signs and r+2% raise the cavity "
                     "line faster than they raise the lower gap edge, so the "
                     "margin has an interior maximum structure and the box "
                     "minimum sits at r-2% with t near nominal."),
        }
        worst_case_stack = {
            "role": ("cross-check of linear superposition, NOT the worst point "
                     "of the box (see verdict.worst_measured_rt_case)"),
            "r_over_a": stack["params"]["r_over_a"],
            "t_over_a": stack["params"]["t_over_a"],
            "from_single_cases": stack["params"]["stack_from"],
            "cavity_line_GHz": stack["f0_GHz"],
            "lo_edge_GHz": stack["lo_edge_GHz"],
            "margin_GHz": stack["margin_GHz"],
            "margin_with_a_plus2pct_GHz": stack["margin_GHz"] / (1 + DA_FRAC),
            "predicted_margin_shift_linear_GHz": pred,
            "measured_margin_shift_GHz": stack["margin_shift_GHz"],
            "superposition_residual_GHz": stack["margin_shift_GHz"] - pred,
            "method": ("cavity line MEASURED directly at the stacked (r,t); "
                       "gap edge from linear superposition of the two measured "
                       "single-axis bulk edge shifts; a-axis applied analytically."),
        }

    result = {
        "task": ("M2 device-specific CAVITY-LINE fabrication tolerance at the "
                 "device lattice constant a = 586.735 um (closing-report Sec. 5.3 "
                 "open item)"),
        "rng": "deterministic - no RNG",
        "status": ("complete" if stack is not None else
                   f"partial ({len(perts)}/{len(order)} perturbations done)"),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "constants": {
            "a_device_um": A_DEV_UM, "a_bulk_study_um": A_BULK_UM,
            "c_um_GHz": C_UM_GHZ, "f_operating_GHz": F_OP_GHZ,
            "box": {"dr_frac": DR_FRAC, "dt_frac": DT_FRAC, "da_frac": DA_FRAC},
            "cavity": {"s": S_SHIFT, "rdef": RDEF, "res": RES,
                       "fcen": FCEN, "df": DF},
        },
        "baseline": baseline_block,
        "perturbations": perts,
        "a_error_analytic": a_err,
        "worst_case_stack": worst_case_stack,
        "box_scan": box_scan,
        "verdict": verdict,
        "method": {
            "cavity": ("scripts/m2_3d_l3.py driven as a module (no solver or "
                       "geometry change): isolated 3D L3, 16a x 12*sqrt3/2 a x "
                       "4.5a supercell, PML 1a on all six faces, s=0.20, "
                       "rdef=1.15, res 18 + subpixel averaging, --lossless, Hz "
                       "dipole at the cavity centre, mirror phases (+1,-1,+1), "
                       "harminv monitor (0.30,0.20,0.0), fcen 0.29 df 0.06, "
                       "1200-period ring-down in 4 chunks of 300 with the "
                       "established trailing-window harminv."),
            "cell_caveat": ("the 16a cell's Q ~ 8.6e3 is a known lateral-leakage "
                            "artifact (physical ceiling 2.7-3.0e4, decisions.md "
                            "2026-07-24); irrelevant here because only FREQUENCY "
                            "shifts are reported and every case uses the same "
                            "cell as the stored baseline."),
            "truncation": ("1200 periods vs the stored 12000-period deep run: "
                           f"nu 0.2837879 -> {NU0_STORED:.7f}, i.e. 1.9e-6 in nu "
                           "= 0.001 GHz at the device a, negligible against "
                           "~1 GHz perturbation shifts."),
            "geometry_deviation": ("hole-lattice EXTENT frozen at nominal "
                                   "r/a=0.30 so an r perturbation cannot change "
                                   "the hole count; radii perturbed. Asserted "
                                   "object-for-object identical to the original "
                                   "generator at nominal parameters."),
            "edges": ("REUSED, not re-run: bulk-gap edge motion from "
                      "out/m2/tolerance.json (MPB run_zeven res 20, K/M edge "
                      "k-points), transferred to the device a by scale "
                      "invariance f = nu*c/a_device."),
            "a_axis": "analytic (Maxwell scale invariance); no run.",
            "stack_selection": ("the (r,t) sign pair minimising "
                                "margin = f_cav - f_lo_edge, evaluated by linear "
                                "superposition of the four measured single-axis "
                                "margin shifts, then run directly."),
        },
        "timings": timings,
        "provenance": {
            "cavity_runs": [os.path.relpath(case_paths(root, c)[1], root).replace("\\", "/")
                            for c in (["baseline"] + order)
                            if _read_case(root, c) is not None],
            "bulk_edges_src": "out/m2/tolerance.json:table (baseline, r_lo, r_hi, t_lo, t_hi)",
            "stored_cavity_line_src": ("out/m2/l3_3d_deep_s020_rdef115_res18_lossless.json"
                                       ":result.headline_in_gap.nu"),
            "a_device_src": "photonics.json:geometry.a_um_precise",
            "cavity_line_gap_src": "photonics.json:modes.cavity_line_gap_GHz",
            "cavity_line_gap_stored_GHz": ph["modes"]["cavity_line_gap_GHz"],
            "scripts": ["scripts/m2_device_tolerance.py",
                        "scripts/m2_device_tolerance.sh",
                        "scripts/fig_device_tolerance.py"],
            "versions": {"python": platform.python_version()},
        },
    }

    outp = os.path.join(root, "out", "m2", "device_tolerance.json")
    with open(outp, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"[devtol] aggregated -> {outp}  status={result['status']}")
    print(f"  baseline nu0={nu_base:.7f}  f={f_base:.4f} GHz  "
          f"lo_edge={lo0_dev:.4f}  margin={margin_base:+.4f} GHz")
    for p in perts:
        print(f"  {p['case']:>9}: f={p['f0_GHz']:9.4f} GHz "
              f"(d{p['shift_GHz']:+7.4f})  lo_edge={p['lo_edge_GHz']:9.4f} "
              f"(d{p['edge_shift_GHz']:+7.4f})  margin={p['margin_GHz']:+8.4f} "
              f"(d{p['margin_shift_GHz']:+7.4f})  in_gap={p['in_gap']}")
    if box_scan:
        print(f"  box scan: r-axis {box_scan['r_axis']['shape']}, argmin "
              f"{box_scan['r_axis']['argmin_pct']:+.3f}% "
              f"({box_scan['r_axis']['min_margin_shift_GHz']:+.4f} GHz); "
              f"t-axis {box_scan['t_axis']['shape']}, argmin "
              f"{box_scan['t_axis']['argmin_pct']:+.3f}% "
              f"({box_scan['t_axis']['min_margin_shift_GHz']:+.4f} GHz)")
        print(f"  model in-box minimum margin = "
              f"{box_scan['model_box_min_margin_GHz']:+.4f} GHz")
    if verdict:
        print(f"  VERDICT min_margin={verdict['min_margin_GHz']:+.4f} GHz "
              f"(worst measured (r,t) point: {verdict['worst_measured_rt_case']} "
              f"at {verdict['worst_measured_rt_margin_GHz']:+.4f}) "
              f"in_gap={verdict['in_gap']} binding={verdict['binding_axis']} "
              f"erosion={verdict['margin_erosion_GHz']:.4f} GHz "
              f"({verdict['margin_erosion_pct_of_baseline']:.1f}% of baseline)")
        if worst_case_stack:
            print(f"  superposition check (stack): predicted "
                  f"{worst_case_stack['predicted_margin_shift_linear_GHz']:+.4f}, "
                  f"measured {worst_case_stack['measured_margin_shift_GHz']:+.4f}, "
                  f"residual {worst_case_stack['superposition_residual_GHz']:+.4f} GHz")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="mode", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("root")
    pr.add_argument("--case", required=True,
                    choices=list(CASES) + ["stack"])
    pr.add_argument("--periods", type=float, default=1200.0)
    pr.add_argument("--chunk", type=float, default=300.0)
    pa = sub.add_parser("aggregate")
    pa.add_argument("root")
    pc = sub.add_parser("check")
    pc.add_argument("root", nargs="?", default=".")
    args = p.parse_args()
    if args.mode == "run":
        cmd_run(args)
    elif args.mode == "check":
        cmd_check(args)
    else:
        cmd_aggregate(args)


if __name__ == "__main__":
    main()
