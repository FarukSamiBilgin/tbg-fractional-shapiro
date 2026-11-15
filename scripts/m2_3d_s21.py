"""M2 AT-3/AT-4: 3D side-coupled W1 + L3 cavity -- S21 ladder & field transfer.

Runs INSIDE the m2env WSL distro. Serial (smoke): /opt/mf/envs/mp/bin/python.
Full runs (AFTER the AT-2 chain frees the cores): mpirun -np 16
/opt/mf/envs/mpmpi/bin/python. MEEP 3D FDTD, units a = 1.

This is the 3D-slab realisation of the (validated) 2D side-coupled ladder
scripts/m2_2d_s21.py, lifted into the true Si slab whose 3D TE-like gap and
lattice constant are locked by AT-0 (out/m2/at0_slab3d.json). The 2D
exploratory run (paper/decisions.md, 2026-07-23) showed the *2D* W1 is
multimode so the clean single-mode standing-wave notch does not form there;
the brief defers the clean AT-3 notch to THIS single-mode 3D slab W1.

Geometry
--------
Triangular air-hole lattice (r/a=0.30) etched through a Si slab (eps=11.68,
t=0.5a, air cladding + PML above/below). A W1 waveguide (one full missing row)
runs along x at row j=0. The L3 cavity (three holes removed along x, end-holes
at i=+/-2 shifted OUTWARD by s -- the Akahane knob, s* from the AT-2 chain,
default 0.15a) sits N_rows to the side, in row j=N. For even N the cavity is
x-centred (x_cav=0), for odd N x_cav=0.5 (that row's lattice offset) -- a valid
L3 either way. Exactly the in-plane layout of the verified 2D script, now in a
3D slab. The crystal stops clear of the lateral PML (PML_PAD margin); the slab
Block still fills the whole cell so the guide/cavity see slab (not vacuum) at
the boundaries, but the PML region is unpatterned -> stable (see PML_PAD note).

Cell (units of a): SX x SY x SZ, PML 1a on ALL SIX faces (no periodic BCs).
  SX  = 24 (x propagation; source at -9, cavity ~x=0, far monitor +9)
  SY  = (N + JA + JB)*sqrt3/2 + 2*DPML + 2*(r+PML_PAD)  (JB=4 rows below guide;
        JA=--outer-rows rows beyond the cavity, default 8 so the cell does not
        cap Q_i (decisions.md 2026-07-24); grows with N)
  SZ  = 2*(0.25 + AIR_GAP + DPML) = 4.5  (identical to the AT-2 L3 3D cell)

Source / monitors (S21)
-----------------------
mp.EigenModeSource in the y-z plane at x=X_SRC launches the W1 fundamental
TE-like guided mode (eig_band selectable; eig_match_freq). Flux monitors:
  * transmission  at x=X_MON (far side)         -> P_tran
  * reflection    at x=X_REFL (behind source)   -> P_refl (after subtracting the
    NORMALISATION incident field via load_minus_flux_data)
A matched NORMALISATION run (straight W1, NO cavity, identical cell/monitors)
gives the incident spectrum P_inc; S21(f)=P_tran^cav/P_inc, R(f)=P_refl/P_inc.
Both a mode-resolved (single guided band, get_eigenmode_coefficients) and a
total-flux S21 are recorded -- the mode-resolved one is the clean two-port
number; total flux is the cross-check that in 3D (single-mode) they agree.
The normalisation + cavity runs are done in ONE process so the reflection DFT
data is in memory for the subtraction (the authoritative path). --normalize
forces (re)running the reference; without it a cached incident spectrum
(<base>_norm.npz) is reused for T (R then needs the in-process reference and is
flagged skipped). The incident spectrum is cell-specific (SY depends on N).

Two independent Q paths (brief Sec.5 requires the cross-check; the 2D notch
imperfection makes it mandatory here -- IMPLEMENT AND REPORT BOTH):
  (1) --mode s21     : TCMT standing-wave notch fit of S21(f)
        T(nu) = (d^2 + gi^2)/(d^2 + (gi+gw)^2),  d = 2*pi*(nu - nu0),
        gx = pi*nu0/Qx  ->  {nu0, Q_loaded, Q_i, Q_w, dip depth T_min}.
  (2) --mode harminv : harminv ring-down of the cavity. Coupled (guide open)
        -> Q_loaded; --cap fills the guide row back -> Q_i (intrinsic). Then
        1/Q_loaded = 1/Q_i + 1/Q_w. Convention-independent, the brief's PRIMARY
        method. With --lossless the capped run gives Q_rad directly (cross-check
        vs the AT-2 L3 3D chain).

AT-4 field transfer (--fields, on the s21 run)
----------------------------------------------
The broadband cavity run also DFT-samples (Ex,Ey,Ez) in a small box at the
cavity centre. For a linear system the frequency-domain response at nu0 IS the
steady-state CW response, so |E(nu0)| / sqrt(P_inc(nu0)) is the on-resonance
field per sqrt(W) of guided input -- no separate CW run needed. We report
E_center/sqrt(W), the stored energy U_FDTD = (1/2) integral eps|E|^2 over the
box (MEEP units), and the CMT consistency check U_CMT = A*P_in*Q_i/omega0
(brief AT-4), with A the on-resonance absorbed fraction 1-T-R.

Symmetry
--------
Only the Z mirror survives the side-coupled layout: the eigenmode source
propagates along +x (breaks the x-mirror) and the cavity sits on ONE side of
the guide (breaks the y-mirror). The guided/cavity fields are the TE-like slab
mode, EVEN about z=0, so mp.Mirror(Z, phase=+1) (the same Z phase verified for
the L3 in scripts/m2_3d_l3.py) halves the z work with no approximation.
--nosym disables it (correctness fallback / verification); --pz overrides.

Frequencies: nu=a/lambda; f_GHz = nu * c_um_GHz / a_final_um, a_final=638.013 um
(out/m2/at0_slab3d.json, 3D-corrected). 3D TE-like gap [0.26904, 0.34813],
midgap 0.30859. NOTE (paper/decisions.md 2026-07-23): the untuned 3D L3
fundamental sits ~0.2795 (131 GHz), ~14 GHz below the 145 GHz drive; a tuning
knob is a separate M2 step. AT-3/AT-4 here characterise the coupling at the
cavity's own nu0.

CLI: m2_3d_s21.py <project_root> [--mode s21|harminv|field]
     --nrows-sep N --s <shift> [--rdef <scale>] [--wg-shift <a>] [--wg-r1 <scale>]
     --res <px/a> --until-periods <n>
     [--normalize] [--norm-only] [--fields] [--lossless] [--cap] [--label <s>]
     [--fcen <nu> --df <nu> --nfreq <n>] [--eig-band <n>]
     [--sx <a> --outer-rows <rows> --jb <rows>] [--pz <+/-1>] [--nosym]
     [--courant <c>] [--no-eps-avg] [--chunk-periods <n>]
Deterministic - no RNG. Logs MEEP/py/numpy/scipy versions in the JSON.
"""
import argparse
import json
import math
import os
import platform
import sys
import time

import numpy as np
import meep as mp
try:
    import scipy
    from scipy.optimize import curve_fit
    _HAVE_SCIPY = True
except Exception:                              # pragma: no cover
    _HAVE_SCIPY = False

# ---- locked constants (out/m2/at0_slab3d.json, 3D-corrected) ----------------
SQRT3 = math.sqrt(3.0)
R_OVER_A = 0.30
T_OVER_A = 0.5                          # slab thickness / a
EPS_CORE = 11.68                        # HR-Si
TAN_DELTA = 2.0e-5                       # material loss tangent
Q_MAT = 1.0 / TAN_DELTA                  # material Q ceiling = 5.0e4
A_FINAL_UM = 638.0131800239526          # AT-0 3D-corrected lattice constant
C_UM_GHZ = 299792.458
GAP3D = (0.2690400358284745, 0.34813300261716335)   # 3D TE-like gap (a/lambda)
MIDGAP3D = 0.30858651922281893
_GTOL = 0.03 * (GAP3D[1] - GAP3D[0])     # FDTD edge tolerance (see m2_3d_l3.py)
GAP_INGAP = (GAP3D[0] - _GTOL, GAP3D[1] + _GTOL)

# ---- cell (units of a) ------------------------------------------------------
DPML = 1.0                              # PML thickness, all six faces
AIR_GAP = 1.0                           # air cladding each side of the slab
SZ = 2.0 * (0.5 * T_OVER_A + AIR_GAP + DPML)   # = 4.5 (matches AT-2 L3 3D)
# Holes must stay CLEAR of the lateral PML: a dielectric interface inside the
# PML makes the subpixel-averaged epsilon anisotropic there -> the MEEP PML
# instability (fields -> NaN/Inf mid-run). PML_PAD = extra slab clearance from
# the outermost hole edge to the PML inner boundary. The slab Block still fills
# the whole cell incl. the PML (isotropic, standard). SY is grown by
# 2*(R+PML_PAD) so the requested ja/jb crystal rows fit with this clearance.
PML_PAD = 0.2
SX_DEFAULT = 24.0
X_SRC = -9.0                            # eigenmode source x
X_MON = +9.0                            # transmitted-flux monitor x
X_REFL = -7.0                           # reflection monitor x (behind source)
WY = 4.0                                # source/monitor width in y
WZ = SZ - 2.0 * DPML                    # source/monitor height in z (=2.5)
JB_DEFAULT = 4                          # crystal rows below the guide (guide far side)
JA_DEFAULT = 8                          # --outer-rows default: rows beyond the
#                                         cavity (its far side); >=8 so the cell
#                                         does not cap Q_i (decisions.md 2026-07-24)

DF_DEFAULT = 0.10
FCEN_DEFAULT = 0.30


def to_ghz(nu):
    return nu * C_UM_GHZ / A_FINAL_UM


# ---- geometry ---------------------------------------------------------------
def layout(N, sx, ja, jb):
    """Return (SY, y_wg, y_cav, x_cav). Content (rows -jb..N+ja) centred in y.
    SY includes the lateral PML AND a slab margin (R+PML_PAD) each side so the
    outermost crystal row stays clear of the PML (see PML_PAD note)."""
    j_lo, j_hi = -jb, N + ja
    yc = 0.5 * (j_hi + j_lo) * (SQRT3 / 2.0)      # content centre in y
    y_wg = 0.0 - yc                               # guide row j=0 in cell coords
    y_cav = N * (SQRT3 / 2.0) - yc
    x_cav = 0.5 * (N % 2)
    SY = ((j_hi - j_lo) * (SQRT3 / 2.0) + 2.0 * DPML
          + 2.0 * (R_OVER_A + PML_PAD))
    return SY, y_wg, y_cav, x_cav


def _is_end_hole(j, i, N):
    """The 2 axial L3 end holes at row j=N, i in {-2,+2} (x_cav +/- 2)."""
    return (j == N and i in (-2, 2))


def _is_transverse_ring(j, x, x_cav, N):
    """The 8 first-ring transverse neighbours: rows j=N+/-1, at
    x - x_cav in {+/-0.5, +/-1.5} (4 per row). Same set as m2_3d_l3.py /
    m2_tune_screen.py, re-expressed for the cavity centred at (row N, x_cav)."""
    if abs(j - N) != 1:
        return False
    d = abs(x - x_cav)
    return (abs(d - 0.5) < 1e-6 or abs(d - 1.5) < 1e-6)


def build_geometry(N, sx, ja, jb, with_cavity, cap_guide, s, lossy, fcen,
                   rdef=1.0, wg_shift=0.0, wg_r1=1.0):
    """Si slab (lossy if requested) + triangular holes; W1 guide (row j=0
    removed unless capped) and, if with_cavity, the L3 at row j=N with outward
    end-hole shift s. With rdef!=1 the 10 defect-adjacent holes (8 transverse
    first-ring + 2 axial end; end holes keep their s-shift) are radius-scaled by
    rdef -- identical convention to m2_3d_l3.py so 2D/3D dnu/d(rdef) compare.
    With wg_shift!=0 the two rows immediately bounding the W1 channel (j=+/-1)
    are displaced OUTWARD by wg_shift (row j=+1 -> +(sqrt3/2 + wg_shift), j=-1
    -> -(sqrt3/2 + wg_shift)); all other rows (incl. the guide j=0 and cavity
    j=N) stay fixed, so N_rows separation semantics are unchanged.
    With wg_r1!=1 the holes of those SAME two channel-bounding rows (j=+/-1) are
    radius-scaled by wg_r1 (textbook W1 dispersion engineering: r1<1 adds
    dielectric next to the channel -> pulls the guided band down). wg_shift
    (position) and wg_r1 (radius) compose on the same rows; rdef acts on the
    disjoint cavity ring rows j=N+/-1 (N>2). N=2 REFUSED with a cavity: there
    j=N-1=+1 is BOTH a channel-bounding row and the cavity's guide-side ring, so
    wg_r1 would corrupt the ring -- handled separately.
    Holes stop clear of the lateral PML (PML_PAD); the slab fills the whole
    cell."""
    if with_cavity and N == 2 and abs(wg_r1 - 1.0) > 1e-9:
        raise SystemExit(
            "ERROR: --wg-r1 != 1 with N=2 and a cavity is refused: the W1 "
            "channel-bounding row j=+1 coincides with the cavity's guide-side "
            "transverse ring (j=N-1=1), so --wg-r1 would double-scale/corrupt "
            "the cavity ring. Decide N=2 separately.")
    SY, y_wg, y_cav, x_cav = layout(N, sx, ja, jb)
    sigma_D = (2.0 * math.pi * fcen * TAN_DELTA) if lossy else 0.0
    slab_mat = mp.Medium(epsilon=EPS_CORE, D_conductivity=sigma_D)
    geom = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, T_OVER_A),
                     center=mp.Vector3(), material=slab_mat)]
    r_ring = R_OVER_A * rdef
    r_wg1 = R_OVER_A * wg_r1
    yc = -y_wg                                    # content centre (y_wg = 0 - yc)
    # keep hole CENTRES so their edge stays PML_PAD inside the lateral PML
    xmax = sx / 2.0 - DPML - R_OVER_A - PML_PAD
    ymax = SY / 2.0 - DPML - R_OVER_A - PML_PAD
    imax = int(math.ceil(xmax)) + 2
    for j in range(-jb, N + ja + 1):              # crystal rows; PML stays slab
        y = j * (SQRT3 / 2.0) - yc
        if abs(j) == 1:                           # W1 channel-bounding rows out
            y += (1.0 if j > 0 else -1.0) * wg_shift
        if abs(y) > ymax:
            continue
        off = 0.5 * (j % 2)
        for i in range(-imax, imax + 1):
            x = i + off
            if abs(x) > xmax:
                continue
            if j == 0 and not cap_guide:           # W1 guide: whole row removed
                continue
            if with_cavity and j == N and i in (-1, 0, 1):   # L3 defect
                continue
            xc = x
            # radius: channel-bounding rows (j=+/-1) -> wg_r1; else cavity ring
            # (j=N+/-1, disjoint for N>2) -> rdef; else nominal.
            if abs(j) == 1:
                rr = r_wg1
            elif with_cavity and (_is_end_hole(j, i, N)
                                  or _is_transverse_ring(j, x, x_cav, N)):
                rr = r_ring                         # defect-adjacent ring scaled
            else:
                rr = R_OVER_A
            if with_cavity and j == N and i == 2:
                xc = x + s                          # end-hole shifted outward
            elif with_cavity and j == N and i == -2:
                xc = x - s
            geom.append(mp.Cylinder(radius=rr, height=mp.inf,
                                    center=mp.Vector3(xc, y, 0.0),
                                    material=mp.air))
    return geom


def make_symmetries(args):
    if args.nosym:
        return []
    return [mp.Mirror(mp.Z, phase=args.pz)]        # only Z survives (see docstr)


# ---- run helpers ------------------------------------------------------------
def run_until_decay(sim, mon_pt, tmax_meep, comp=mp.Ey, dt_check=50,
                    thresh=1e-4):
    dec = mp.stop_when_fields_decayed(dt_check, comp, mon_pt, thresh)

    def stopfn(s):
        if s.meep_time() > tmax_meep:
            return True
        return dec(s)
    sim.run(until_after_sources=stopfn)


def make_eig_source(fcen, df, y_wg, eig_band):
    return mp.EigenModeSource(
        mp.GaussianSource(fcen, fwidth=df),
        center=mp.Vector3(X_SRC, y_wg, 0.0),
        size=mp.Vector3(0.0, WY, WZ),
        direction=mp.X, eig_band=eig_band, eig_match_freq=True,
        eig_parity=mp.NO_PARITY)


def flux_regions(y_wg):
    fr_t = mp.FluxRegion(center=mp.Vector3(X_MON, y_wg, 0.0),
                         size=mp.Vector3(0.0, WY, WZ))
    fr_r = mp.FluxRegion(center=mp.Vector3(X_REFL, y_wg, 0.0),
                         size=mp.Vector3(0.0, WY, WZ))
    return fr_t, fr_r


# ---- TCMT standing-wave notch fit -------------------------------------------
def tcmt_T(nu, nu0, gi, gw):
    d = 2.0 * math.pi * (nu - nu0)
    return (d ** 2 + gi ** 2) / (d ** 2 + (gi + gw) ** 2)


def fit_tcmt(nu, T, nu_lo, nu_hi):
    """Fit the standing-wave notch on the band [nu_lo,nu_hi]; return dict."""
    m = np.isfinite(T) & (nu >= nu_lo) & (nu <= nu_hi)
    nu_b, T_b = nu[m], T[m]
    if nu_b.size < 6:
        return {"ok": False, "reason": "too few in-band points"}
    imin = int(np.argmin(T_b))
    nu0_0 = float(nu_b[imin])
    Tmin_0 = float(max(T_b[imin], 1e-6))
    # loaded linewidth guess: half-depth width of the (1-T) dip
    depth = 1.0 - T_b
    half = depth.max() / 2.0
    above = nu_b[depth >= half]
    fwhm = float(above.max() - above.min()) if above.size >= 2 else 0.01
    gL0 = max(math.pi * fwhm, 1e-4)                 # gi+gw ~ 2*pi*(fwhm/2)
    r0 = math.sqrt(min(Tmin_0, 0.999))              # gi/(gi+gw)
    gi0 = max(r0 * gL0, 1e-6)
    gw0 = max((1.0 - r0) * gL0, 1e-6)
    res = {"ok": False}
    if _HAVE_SCIPY:
        try:
            popt, pcov = curve_fit(
                tcmt_T, nu_b, T_b, p0=[nu0_0, gi0, gw0],
                bounds=([nu_lo, 1e-9, 1e-9], [nu_hi, 1.0, 1.0]),
                maxfev=200000)
            nu0, gi, gw = (float(popt[0]), float(popt[1]), float(popt[2]))
            perr = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else \
                np.full(3, np.nan)
            Tfit = tcmt_T(nu_b, nu0, gi, gw)
            ss_res = float(np.sum((T_b - Tfit) ** 2))
            ss_tot = float(np.sum((T_b - T_b.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            res = {"ok": True, "method": "scipy.curve_fit", "R2": r2,
                   "param_stderr": [float(x) for x in perr]}
        except Exception as e:                      # noqa: BLE001
            res = {"ok": False, "reason": f"curve_fit failed: {e!r}"}
    if not res["ok"]:
        # analytic fallback: nu0/Tmin from the minimum, gi+gw from the FWHM
        nu0, gL = nu0_0, gL0
        gi, gw = r0 * gL, (1.0 - r0) * gL
        res = {"ok": True, "method": "analytic (min+FWHM)", "R2": None,
               "param_stderr": [None, None, None]}
    Qi = math.pi * nu0 / gi if gi > 0 else float("inf")
    Qw = math.pi * nu0 / gw if gw > 0 else float("inf")
    QL = math.pi * nu0 / (gi + gw)
    Tmin_fit = (gi / (gi + gw)) ** 2
    res.update({"nu0": nu0, "f0_GHz": to_ghz(nu0), "gamma_i": gi,
                "gamma_w": gw, "Q_i": Qi, "Q_w": Qw, "Q_loaded": QL,
                "T_min_fit": Tmin_fit, "T_min_measured": Tmin_0,
                "dip_depth_fit": 1.0 - Tmin_fit,
                "A_onres": 2.0 * gi * gw / (gi + gw) ** 2,
                "overcoupled": bool(gw > gi),
                "nu0_argmin_measured": nu0_0,
                "TCMT_consistency_rel_err": (
                    abs(Tmin_fit - Tmin_0) / Tmin_0 if Tmin_0 > 0 else None)})
    return res


# ---- harminv candidate helpers (mirror m2_3d_l3.py) -------------------------
ERR_CLEAN = 1.0e-3
AMP_MAX = 1.0e9


def extract_candidates(modes):
    out = []
    for m in modes:
        nu = float(m.freq)
        Q = float(m.Q)
        amp = float(abs(m.amp))
        err = float(abs(m.err))
        clean = bool(Q > 0 and err < ERR_CLEAN and math.isfinite(amp)
                     and amp < AMP_MAX)
        out.append({"nu": nu, "GHz": to_ghz(nu), "Q": Q, "amp": amp,
                    "decay": float(m.decay), "err": err,
                    "in_gap": bool(GAP_INGAP[0] <= nu <= GAP_INGAP[1]), "clean": clean})
    out.sort(key=lambda d: -d["amp"])
    return out


def best_in_gap(cands):
    ingap = [c for c in cands if c["in_gap"]]
    if not ingap:
        return None
    pool = [c for c in ingap if c["clean"]] or ingap
    return sorted(pool, key=lambda c: (c["err"], -c["amp"]))[0]


def candidates_from_bands(bands):
    """mp.py_do_harminv output [(freq_c, amp_c, err), ...] -> candidate dicts
    (freq complex: real=nu, imag=decay; Q = nu/(-2*decay))."""
    out = []
    for freq, amp, err in bands:
        nu = float(freq.real)
        decay = float(freq.imag)
        Q = (nu / (-2.0 * decay)) if decay != 0 else float("inf")
        a = float(abs(amp))
        e = float(abs(err))
        clean = bool(Q > 0 and e < ERR_CLEAN and math.isfinite(a) and a < AMP_MAX)
        out.append({"nu": nu, "GHz": to_ghz(nu), "Q": Q, "amp": a,
                    "decay": decay, "err": e,
                    "in_gap": bool(GAP_INGAP[0] <= nu <= GAP_INGAP[1]), "clean": clean})
    out.sort(key=lambda d: -d["amp"])
    return out


def make_collector(comp, pt):
    """Step function recording `comp` at `pt` every timestep -> (fn, buf).
    buf['data'] = complex series, buf['dt'] = sample interval. get_field_point
    is MPI-collective (runs on all ranks)."""
    buf = {"data": [], "dt": 0.0, "_t": None}

    def _step(sim):
        t = sim.meep_time()
        if buf["_t"] is not None and buf["dt"] == 0.0:
            buf["dt"] = t - buf["_t"]
        buf["_t"] = t
        buf["data"].append(sim.get_field_point(comp, pt))
    return _step, buf


def analyze_window(data, data_dt, fcen, df, periods_so_far, tau_est):
    """harminv on the LAST W periods; W = min(2*tau_est, periods_so_far), full
    buffer while periods_so_far <= tau_est (or tau unknown). -> (cands, info)."""
    n_tot = len(data)
    if tau_est and periods_so_far > tau_est:
        W = min(2.0 * tau_est, periods_so_far)
    else:
        W = periods_so_far
    n_win = int(round(W / fcen / data_dt)) if (data_dt and data_dt > 0) else n_tot
    n_win = max(1, min(n_win, n_tot))
    win = list(data[-n_win:])
    bands = mp.py_do_harminv(win, data_dt, fcen - df / 2.0, fcen + df / 2.0,
                             100, 1.1, 50.0, mp.inf, 0.01, -1.0, -1.0)
    info = {"window_periods": W, "window_start_period": periods_so_far - W,
            "window_end_period": periods_so_far, "window_samples": n_win,
            "windowed": bool(W < periods_so_far - 1e-9)}   # trailing window active
    return candidates_from_bands(bands), info


# =============================================================================
# S21 mode
# =============================================================================
def run_s21(args, base, logf, result, write_ck):
    fcen, df, nfreq = args.fcen, args.df, args.nfreq
    N = args.nrows_sep
    SY, y_wg, y_cav, x_cav = layout(N, args.sx, args.outer_rows, args.jb)
    tmax_meep = args.until_periods / fcen
    syms = make_symmetries(args)
    lossy = not args.lossless
    logf(f"S21 N={N} s={args.s} res={args.res} cell={args.sx}x{SY:.3f}x{SZ} "
         f"sym={'Z+%d' % args.pz if not args.nosym else 'none'} lossy={lossy} "
         f"tmax={tmax_meep:.1f} ({args.until_periods}p) nfreq={nfreq}")

    def build_sim(with_cavity):
        geom = build_geometry(N, args.sx, args.outer_rows, args.jb, with_cavity,
                              cap_guide=False, s=args.s, lossy=lossy, fcen=fcen,
                              rdef=args.rdef, wg_shift=args.wg_shift,
                              wg_r1=args.wg_r1)
        src = make_eig_source(fcen, df, y_wg, args.eig_band)
        return mp.Simulation(cell_size=mp.Vector3(args.sx, SY, SZ),
                             geometry=geom, default_material=mp.air,
                             boundary_layers=[mp.PML(DPML)], sources=[src],
                             symmetries=syms, resolution=args.res,
                             Courant=args.courant, eps_averaging=args.eps_avg)

    fr_t, fr_r = flux_regions(y_wg)

    # ---- normalisation (straight W1) ---------------------------------------
    normpath = f"{base}_norm.npz"
    do_norm = args.normalize or not os.path.exists(normpath)
    norm_refl_data = None
    t0 = time.time()
    sim_n = build_sim(with_cavity=False)
    # add_mode_monitor (not add_flux): required for get_eigenmode_coefficients
    # when a symmetry is present; also supports get_fluxes + flux-data subtraction
    tr_n = sim_n.add_mode_monitor(fcen, df, nfreq, fr_t)
    rf_n = sim_n.add_mode_monitor(fcen, df, nfreq, fr_r)
    run_until_decay(sim_n, mp.Vector3(X_MON, y_wg, 0.0), tmax_meep)
    freqs = np.array(mp.get_flux_freqs(tr_n))
    inc_flux = np.array(mp.get_fluxes(tr_n))                # incident (all pass)
    ct_n = sim_n.get_eigenmode_coefficients(tr_n, [args.eig_band],
                                            eig_parity=mp.NO_PARITY)
    cr_n = sim_n.get_eigenmode_coefficients(rf_n, [args.eig_band],
                                            eig_parity=mp.NO_PARITY)
    pinc_band = np.abs(ct_n.alpha[0, :, 0]) ** 2            # fwd band at far mon
    pinc_refl = np.abs(cr_n.alpha[0, :, 0]) ** 2            # fwd band at refl mon
    norm_refl_data = sim_n.get_flux_data(rf_n)              # for R subtraction
    t_norm = sim_n.meep_time()
    dt_norm = time.time() - t0
    fwd0 = float(np.max(inc_flux))
    logf(f"  norm run: t_meep={t_norm:.1f} wall={dt_norm:.2f}s "
         f"max_inc_flux={fwd0:.4e} (fwd>0: {fwd0 > 0})")
    if mp.am_really_master():              # master-only file write (MPI-safe)
        np.savez(normpath, freqs=freqs, inc_flux=inc_flux, pinc_band=pinc_band,
                 pinc_refl=pinc_refl, fcen=fcen, df=df, res=args.res, N=N)
    result["normalization"] = {
        "path": normpath, "ran_inprocess": True, "t_meep": t_norm,
        "wall_seconds": round(dt_norm, 2), "max_inc_flux": fwd0,
        "forward_flux_positive": bool(fwd0 > 0),
        "s_per_meep_time": dt_norm / t_norm if t_norm > 0 else None,
        "s_per_period": (dt_norm / t_norm / fcen) if t_norm > 0 else None}
    write_ck()

    if args.norm_only:      # standalone: straight-guide reference only, then stop
        result["status"] = "done (normalization only)"
        write_ck()
        return

    # ---- cavity run --------------------------------------------------------
    t0 = time.time()
    sim_c = build_sim(with_cavity=True)
    tr_c = sim_c.add_mode_monitor(fcen, df, nfreq, fr_t)
    rf_c = sim_c.add_mode_monitor(fcen, df, nfreq, fr_r)
    if norm_refl_data is not None:
        sim_c.load_minus_flux_data(rf_c, norm_refl_data)   # subtract incident
        refl_ok = True
    else:
        refl_ok = False
    dft_cav = None
    if args.fields:
        dft_cav = sim_c.add_dft_fields(
            [mp.Ex, mp.Ey, mp.Ez], fcen, df, nfreq,
            center=mp.Vector3(x_cav, y_cav, 0.0),
            size=mp.Vector3(2.0, 2.0, T_OVER_A))
    run_until_decay(sim_c, mp.Vector3(X_MON, y_wg, 0.0), tmax_meep)
    tran_flux = np.array(mp.get_fluxes(tr_c))
    refl_flux = np.array(mp.get_fluxes(rf_c))
    ct_c = sim_c.get_eigenmode_coefficients(tr_c, [args.eig_band],
                                            eig_parity=mp.NO_PARITY)
    ptran_band = np.abs(ct_c.alpha[0, :, 0]) ** 2
    t_cav = sim_c.meep_time()
    dt_cav = time.time() - t0
    logf(f"  cavity run: t_meep={t_cav:.1f} wall={dt_cav:.2f}s "
         f"refl_subtracted={refl_ok}")

    with np.errstate(divide="ignore", invalid="ignore"):
        s21_flux = np.where(inc_flux > 0, tran_flux / inc_flux, np.nan)
        s21_band = np.where(pinc_band > 0, ptran_band / pinc_band, np.nan)
        R = np.where(inc_flux > 0, -refl_flux / inc_flux, np.nan)

    # ---- AT-4 field transfer (from the same broadband run) -----------------
    fields_block = None
    if args.fields and dft_cav is not None:
        try:
            fields_block = compute_field_transfer(
                sim_c, dft_cav, freqs, inc_flux, args.res, base, logf)
        except Exception as e:                      # noqa: BLE001
            logf(f"  field-transfer post FAILED: {e!r}")
            fields_block = {"error": repr(e)}

    # ---- TCMT fit ----------------------------------------------------------
    nu_lo, nu_hi = GAP3D
    # restrict to where the guide actually carries power (band overlap)
    pmax = float(np.nanmax(pinc_band)) if np.any(np.isfinite(pinc_band)) else 0.0
    band_mask = pinc_band > 0.15 * pmax if pmax > 0 else np.ones_like(freqs, bool)
    nu_band = freqs[band_mask]
    if nu_band.size:
        nu_lo = max(nu_lo, float(nu_band.min()))
        nu_hi = min(nu_hi, float(nu_band.max()))
    fit_band = fit_tcmt(freqs, s21_band, nu_lo, nu_hi)
    fit_flux = fit_tcmt(freqs, s21_flux, nu_lo, nu_hi)

    # off-resonance transmission >= 99.5% across +/-5 GHz of nu0 (brief AT-3)
    offres = None
    if fit_band.get("ok"):
        nu0 = fit_band["nu0"]
        dnu5 = 5.0 * A_FINAL_UM / C_UM_GHZ           # 5 GHz in a/lambda
        far = np.isfinite(s21_band) & (np.abs(freqs - nu0) >= dnu5) & band_mask
        if np.any(far):
            offres = {"nu0": nu0, "dnu_5GHz": dnu5,
                      "min_T_beyond_5GHz": float(np.nanmin(s21_band[far])),
                      "meets_99p5pct": bool(np.nanmin(s21_band[far]) >= 0.995)}

    imin = int(np.nanargmin(s21_band)) if np.any(np.isfinite(s21_band)) else -1
    result["params"].update({"N_rows": N, "cell": [args.sx, SY, SZ],
                             "y_wg": y_wg, "y_cav": y_cav, "x_cav": x_cav,
                             "eig_band": args.eig_band, "fcen": fcen, "df": df,
                             "nfreq": nfreq, "t_norm": t_norm, "t_cav": t_cav})
    result["timing"] = {
        "norm_wall_s": round(dt_norm, 2), "cav_wall_s": round(dt_cav, 2),
        "norm_t_meep": t_norm, "cav_t_meep": t_cav,
        "s_per_meep_time_norm": dt_norm / t_norm if t_norm else None,
        "s_per_meep_time_cav": dt_cav / t_cav if t_cav else None,
        "s_per_period_cav": (dt_cav / t_cav / fcen) if t_cav else None,
        "cell_volume_a3": args.sx * SY * SZ,
        "z_symmetry_factor": 1.0 if args.nosym else 2.0}
    result["spectra"] = {
        "freqs": [float(x) for x in freqs],
        "GHz": [to_ghz(float(x)) for x in freqs],
        "inc_flux": [float(x) for x in inc_flux],
        "tran_flux": [float(x) for x in tran_flux],
        "pinc_band": [float(x) for x in pinc_band],
        "ptran_band": [float(x) for x in ptran_band],
        "S21_band": [None if not np.isfinite(v) else float(v) for v in s21_band],
        "S21_flux": [None if not np.isfinite(v) else float(v) for v in s21_flux],
        "R": [None if not np.isfinite(v) else float(v) for v in R]}
    result["tcmt_fit_band"] = fit_band
    result["tcmt_fit_flux"] = fit_flux
    result["offres_check"] = offres
    result["refl_subtracted"] = refl_ok
    if fields_block is not None:
        result["field_transfer"] = fields_block
    result["status"] = "done"
    write_ck()

    if imin >= 0:
        logf(f"DONE S21 N={N}: min S21_band={np.nanmin(s21_band):.4f} at "
             f"nu={freqs[imin]:.5f} ({to_ghz(float(freqs[imin])):.3f} GHz)")
    if fit_band.get("ok"):
        logf(f"  TCMT(band): nu0={fit_band['nu0']:.5f} "
             f"({fit_band['f0_GHz']:.3f} GHz) Q_L={fit_band['Q_loaded']:.1f} "
             f"Q_i={fit_band['Q_i']:.1f} Q_w={fit_band['Q_w']:.1f} "
             f"T_min={fit_band['T_min_fit']:.4f} R2={fit_band.get('R2')}")


def compute_field_transfer(sim, dft_cav, freqs, inc_flux, res, base, logf):
    """AT-4: on-resonance |E| at the cavity centre per sqrt(W) input, stored
    energy U_FDTD, and the CMT check U = A*P_in*Q_i/omega0 (done in analyze once
    Q_i is known; here we report U_FDTD and the per-sqrt(W) transfer)."""
    def emag2(k):
        ex = np.asarray(sim.get_dft_array(dft_cav, mp.Ex, k))
        ey = np.asarray(sim.get_dft_array(dft_cav, mp.Ey, k))
        ez = np.asarray(sim.get_dft_array(dft_cav, mp.Ez, k))
        return ex, ey, ez
    E2_center = np.zeros(len(freqs))
    E2_boxint = np.zeros(len(freqs))
    for k in range(len(freqs)):
        ex, ey, ez = emag2(k)
        e2 = np.abs(ex) ** 2 + np.abs(ey) ** 2 + np.abs(ez) ** 2
        ci = tuple(s // 2 for s in e2.shape)
        E2_center[k] = float(e2[ci])
        E2_boxint[k] = float(e2.sum()) * (1.0 / res) ** 3
    with np.errstate(divide="ignore", invalid="ignore"):
        # |E(nu)| / sqrt(P_inc(nu)) : steady-state field per sqrt(W)
        transfer = np.where(inc_flux > 0, np.sqrt(E2_center) / np.sqrt(inc_flux),
                            np.nan)
    kpk = int(np.nanargmax(E2_center))
    logf(f"  field-transfer: peak |E|^2 at nu={freqs[kpk]:.5f} "
         f"({to_ghz(float(freqs[kpk])):.3f} GHz) "
         f"E/sqrtW={transfer[kpk]:.4e}")
    return {"freqs": [float(x) for x in freqs],
            "E2_center": [float(x) for x in E2_center],
            "E2_boxint": [float(x) for x in E2_boxint],
            "E_over_sqrtW": [None if not np.isfinite(v) else float(v)
                             for v in transfer],
            "peak_index": kpk, "peak_nu": float(freqs[kpk]),
            "peak_GHz": to_ghz(float(freqs[kpk])),
            "peak_E_over_sqrtW": None if not np.isfinite(transfer[kpk])
            else float(transfer[kpk]),
            "note": "steady-state per sqrt(W); U_CMT=A*P_in*Q_i/omega0 check in "
                    "analyze once Q_i (TCMT/harminv) is fixed. E2_boxint is the "
                    "|E|^2 volume integral over the 2x2x0.5 a cavity box "
                    "(MEEP units); multiply by eps and 1/2 for stored energy."}


# =============================================================================
# harminv mode (cross-check ring-down)  --  chunked/checkpointed like L3 3D
# =============================================================================
def run_harminv(args, base, logf, result, write_ck):
    fcen, df = args.fcen_h, args.df_h
    N = args.nrows_sep
    SY, y_wg, y_cav, x_cav = layout(N, args.sx, args.outer_rows, args.jb)
    syms = make_symmetries(args)
    lossy = not args.lossless
    cap = args.cap
    logf(f"harminv N={N} s={args.s} res={args.res} cap={cap} lossy={lossy} "
         f"fcen={fcen} df={df} cell={args.sx}x{SY:.3f}x{SZ}")

    geom = build_geometry(N, args.sx, args.outer_rows, args.jb, with_cavity=True,
                          cap_guide=cap, s=args.s, lossy=lossy, fcen=fcen,
                          rdef=args.rdef, wg_shift=args.wg_shift,
                          wg_r1=args.wg_r1)
    # excite the cavity off its Hz x-node; monitor off both cavity mid-planes
    src_pt = mp.Vector3(x_cav + 0.35, y_cav + 0.15, 0.0)
    mon = mp.Vector3(x_cav + 0.30, y_cav + 0.20, 0.0)
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Hz,
                         center=src_pt)]
    sim = mp.Simulation(cell_size=mp.Vector3(args.sx, SY, SZ), geometry=geom,
                        default_material=mp.air,
                        boundary_layers=[mp.PML(DPML)], sources=sources,
                        symmetries=syms, resolution=args.res,
                        Courant=args.courant, eps_averaging=args.eps_avg)
    collect, buf = make_collector(mp.Hz, mon)   # trailing-window harminv source

    total_periods = args.until_periods
    chunk_periods = args.chunk_periods if args.chunk_periods > 0 else \
        max(25.0, total_periods / 8.0)
    chunk_periods = min(chunk_periods, total_periods)
    n_chunks = max(1, int(round(total_periods / chunk_periods)))
    chunk_periods = total_periods / n_chunks
    t_chunk = chunk_periods / fcen
    Qkey = "Q_i" if cap else "Q_loaded"
    result["params"].update({"N_rows": N, "cap": cap, "fcen": fcen, "df": df,
                             "cell": [args.sx, SY, SZ], "source_pt":
                             [src_pt.x, src_pt.y, src_pt.z],
                             "monitor": [mon.x, mon.y, mon.z],
                             "total_periods": total_periods,
                             "chunk_periods": chunk_periods,
                             "n_chunks": n_chunks, "Qkey": Qkey})
    chunks = []
    result["chunks"] = chunks
    write_ck()
    driven = False
    periods_done = 0.0
    best_Q = None                              # running clean in-gap Q -> tau_est
    for ci in range(n_chunks):
        t0 = time.time()
        if not driven:
            sim.run(mp.after_sources(collect), until_after_sources=t_chunk)
            driven = True
        else:
            sim.run(collect, until=t_chunk)
        dt = time.time() - t0
        periods_done += chunk_periods
        tau_est = (best_Q / math.pi) if best_Q else None
        cands, win = analyze_window(buf["data"], buf["dt"], fcen, df,
                                    periods_done, tau_est)
        big = best_in_gap(cands)
        if big and big.get("clean"):
            best_Q = big["Q"]
        rec = {"chunk_index": ci, "cum_periods": periods_done,
               "meep_time": float(sim.round_time()), "wall_seconds": round(dt, 2),
               "n_modes": len(cands), "window": win, "candidates": cands,
               "best_in_gap": big}
        chunks.append(rec)
        result["chunks"] = chunks
        write_ck()
        tg = (f"nu0={big['nu']:.5f} ({big['GHz']:.3f}) {Qkey}={big['Q']:.1f} "
              f"amp={big['amp']:.3e}") if big else "no in-gap yet"
        logf(f"  chunk {ci+1}/{n_chunks}: +{chunk_periods:g}p "
             f"(cum {periods_done:g}p) {dt:.2f}s | "
             f"win[{win['window_start_period']:.0f},{win['window_end_period']:.0f}]p"
             f"{'*' if win['windowed'] else ''} | {len(cands)} modes | {tg}")

    clean_bests = [(c["cum_periods"], c["best_in_gap"], c["window"])
                   for c in chunks
                   if c.get("best_in_gap") and c["best_in_gap"].get("clean")]
    if clean_bests:
        headline = max(clean_bests, key=lambda t: t[0])[1]
    else:
        anyb = [c["best_in_gap"] for c in chunks if c.get("best_in_gap")]
        headline = (min(anyb, key=lambda b: b["err"]) if anyb
                    else (chunks[-1]["best_in_gap"] if chunks else None))
    q_trend = [{"cum_periods": p, "nu": b["nu"], "Q": b["Q"], "err": b["err"],
                "windowed": w.get("windowed"),
                "window_periods": w.get("window_periods")}
               for (p, b, w) in clean_bests]
    q_stat = None
    if len(q_trend) >= 2 and q_trend[-1]["Q"]:
        q_stat = bool(abs(q_trend[-1]["Q"] - q_trend[-2]["Q"])
                      / abs(q_trend[-1]["Q"]) < 0.10)
    win_Q = ([t["Q"] for t in q_trend if t["windowed"]]
             or [t["Q"] for t in q_trend])
    q_spread = None
    if win_Q:
        arr = np.array(win_Q, float)
        q_spread = {"n": int(arr.size), "Q_mean": float(arr.mean()),
                    "Q_std": float(arr.std()), "Q_min": float(arr.min()),
                    "Q_max": float(arr.max()),
                    "rel_spread": (float(arr.std() / arr.mean())
                                   if arr.mean() else None)}
    step_chunks = [c["wall_seconds"] for c in chunks[1:]] or \
        [chunks[0]["wall_seconds"]]
    result["result"] = {
        "headline_in_gap": headline, Qkey + "_headline":
        (headline["Q"] if headline else None),
        "q_trend_clean": q_trend,
        "q_stationary_last2_within_10pct": q_stat,
        "q_window_spread": q_spread,
        "Q_mat_ceiling": Q_MAT,
        "stepping_seconds_per_period": (sum(step_chunks) / len(step_chunks))
        / chunk_periods}
    result["status"] = "done"
    write_ck()
    if headline:
        logf(f"DONE harminv N={N} {Qkey}: nu0={headline['nu']:.6f} "
             f"({headline['GHz']:.3f} GHz) {Qkey}={headline['Q']:.1f} "
             f"(err={headline['err']:.2e}) stat={q_stat}")
    else:
        logf(f"DONE harminv N={N}: no in-gap candidate")


# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="M2 AT-3/AT-4 3D S21 (MEEP FDTD)")
    p.add_argument("root", help="project root (WSL path)")
    p.add_argument("--mode", choices=["s21", "harminv", "field"], default="s21",
                   help="s21 sweep+TCMT (default), or harminv ring-down x-check")
    p.add_argument("--nrows-sep", dest="nrows_sep", type=int, default=3,
                   help="cavity separation in rows (THE coupling knob, 2..5)")
    p.add_argument("--s", type=float, default=0.15,
                   help="end-hole shift (units a; s* from AT-2 chain)")
    p.add_argument("--rdef", type=float, default=1.0,
                   help="radius scale of the 10 defect-adjacent holes (8 "
                        "transverse first-ring + 2 axial end holes; end holes "
                        "keep their s-shift). Tuned cavity uses >1. Same ring "
                        "as m2_3d_l3.py / m2_tune_screen.py.")
    p.add_argument("--wg-shift", dest="wg_shift", type=float, default=0.0,
                   help="displace the two W1 channel-bounding rows (j=+/-1) "
                        "OUTWARD by this (units a) to widen the channel and "
                        "tune the guided band down. Guide (j=0) and cavity "
                        "(j=N) rows stay fixed; N_rows separation unchanged.")
    p.add_argument("--wg-r1", dest="wg_r1", type=float, default=1.0,
                   help="radius scale of the holes of the two W1 channel-"
                        "bounding rows (j=+/-1). r1<1 (e.g. 0.85, 0.70) adds "
                        "dielectric next to the channel and pulls the guided "
                        "band DOWN (textbook W1 dispersion engineering). "
                        "Composes with --wg-shift; REFUSED for N=2 with a "
                        "cavity (row j=+1 is also the cavity ring).")
    p.add_argument("--res", type=int, default=8, help="resolution px/a")
    p.add_argument("--until-periods", dest="until_periods", type=float,
                   default=100.0, help="run window / ring-down in optical periods")
    p.add_argument("--chunk-periods", dest="chunk_periods", type=float,
                   default=0.0, help="harminv checkpoint chunk (periods); 0=auto")
    p.add_argument("--normalize", action="store_true",
                   help="force (re)run the straight-W1 reference")
    p.add_argument("--norm-only", dest="norm_only", action="store_true",
                   help="run ONLY the straight-guide reference (no cavity) and "
                        "stop; standalone (no --normalize needed). Its "
                        "transmission spectrum gives the guided-band edge.")
    p.add_argument("--fields", action="store_true",
                   help="(s21) add AT-4 on-resonance field transfer")
    p.add_argument("--lossless", action="store_true",
                   help="zero D_conductivity (Q_rad / lossless mirror check)")
    p.add_argument("--cap", action="store_true",
                   help="(harminv) fill the guide row back -> Q_i")
    p.add_argument("--label", default="smoke", help="filename label")
    p.add_argument("--fcen", type=float, default=FCEN_DEFAULT,
                   help="(s21) source centre nu")
    p.add_argument("--df", type=float, default=DF_DEFAULT,
                   help="(s21) source fwidth")
    p.add_argument("--nfreq", type=int, default=200, help="(s21) flux points")
    p.add_argument("--eig-band", dest="eig_band", type=int, default=1,
                   help="guided band to launch/measure")
    p.add_argument("--fcen-h", dest="fcen_h", type=float, default=0.28,
                   help="(harminv) source centre nu")
    p.add_argument("--df-h", dest="df_h", type=float, default=0.05,
                   help="(harminv) source fwidth")
    p.add_argument("--sx", type=float, default=SX_DEFAULT, help="cell size x (a)")
    p.add_argument("--outer-rows", dest="outer_rows", type=int,
                   default=JA_DEFAULT,
                   help="mirror rows on the FAR side of the cavity (beyond it, "
                        "away from the waveguide); the Q_i lateral-confinement "
                        "knob. Default 8 so the cell does not cap Q_i "
                        "(decisions.md 2026-07-24). Does not touch the guide "
                        "side; N_rows stays the coupling knob.")
    p.add_argument("--jb", type=int, default=JB_DEFAULT,
                   help="crystal rows below the guide")
    p.add_argument("--pz", type=int, default=1, help="Mirror(Z) phase (verified +1)")
    p.add_argument("--nosym", action="store_true", help="disable Z symmetry")
    p.add_argument("--courant", type=float, default=0.5,
                   help="Courant stability factor")
    p.add_argument("--no-eps-avg", dest="eps_avg", action="store_false",
                   help="disable subpixel averaging (default ON). Subpixel is "
                        "needed for mode accuracy at res8 (noavg misidentifies "
                        "the coupled mode); it is STABLE at res8 and res18 but "
                        "UNSTABLE at res12 for this eps=11.68 slab (passivity "
                        "violation -> field growth). At res12 use --no-eps-avg.")
    p.add_argument("--outdir", default="", help="output dir (default <root>/out/m2)")
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root
    outdir = args.outdir or f"{root}/out/m2"
    os.makedirs(outdir, exist_ok=True)
    tag = f"{int(round(args.s * 100)):03d}"
    rdef_tag = (f"_rdef{int(round(args.rdef * 100)):03d}"
                if abs(args.rdef - 1.0) > 1e-9 else "")   # keep name at rdef==1
    wg_tag = (f"_wg{int(round(args.wg_shift * 100)):02d}"
              if abs(args.wg_shift) > 1e-9 else "")        # keep name at wg==0
    r1_tag = (f"_r1{int(round(args.wg_r1 * 100)):03d}"
              if abs(args.wg_r1 - 1.0) > 1e-9 else "")      # keep name at r1==1
    base = (f"{outdir}/s21_3d_{args.label}_{args.mode}_N{args.nrows_sep}"
            f"_s{tag}{rdef_tag}{wg_tag}{r1_tag}_res{args.res}"
            f"{'_cap' if (args.mode == 'harminv' and args.cap) else ''}"
            f"{'_lossless' if args.lossless else ''}"
            f"{'_nosym' if args.nosym else ''}")
    logpath = base + ".log"
    ckpath = base + ".json"

    def logf(msg):
        # master-only: under mpirun every rank would otherwise write the same
        # file concurrently and corrupt it.
        if not mp.am_really_master():
            return
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line, flush=True)
        with open(logpath, "a") as fh:
            fh.write(line + "\n")

    t_start = time.time()
    logf(f"M2 AT-3/4 3D S21 | meep {mp.__version__} py {platform.python_version()}"
         f" numpy {np.__version__} scipy {scipy.__version__ if _HAVE_SCIPY else 'NA'}"
         f" | mode={args.mode} N={args.nrows_sep} s={args.s} res={args.res}")

    result = {
        "task": "M2 AT-3/AT-4: 3D side-coupled W1+L3 S21 ladder / field transfer",
        "rng": "deterministic - no RNG",
        "meep_version": mp.__version__, "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": (scipy.__version__ if _HAVE_SCIPY else None),
        "status": "running", "mode": args.mode,
        "params": {"s": args.s, "res": args.res, "lossy": (not args.lossless),
                   "rdef": args.rdef, "outer_rows": args.outer_rows,
                   "wg_shift": args.wg_shift, "wg_r1": args.wg_r1,
                   "rdef_ring": "8 transverse (j=N+/-1) + 2 end (row N, +/-2)",
                   "courant": args.courant, "pml_pad": PML_PAD,
                   "eps_averaging": args.eps_avg,
                   "nosym": args.nosym, "mirror_phase_Z": args.pz,
                   "r_over_a": R_OVER_A, "t_over_a": T_OVER_A,
                   "eps_core": EPS_CORE, "tan_delta": TAN_DELTA,
                   "Q_mat_ceiling": Q_MAT, "a_final_um": A_FINAL_UM,
                   "gap3d_a_over_lambda": list(GAP3D),
                   "midgap3d": MIDGAP3D, "dpml": DPML, "air_gap": AIR_GAP,
                   "x_src": X_SRC, "x_mon": X_MON, "x_refl": X_REFL,
                   "wy": WY, "wz": WZ, "until_periods": args.until_periods},
    }

    def write_ck():
        if not mp.am_really_master():      # master-only: avoid MPI write races
            return
        with open(ckpath, "w") as fh:
            json.dump(result, fh, indent=1)
    write_ck()

    if args.mode == "s21" or args.mode == "field":
        if args.mode == "field":
            args.fields = True
        run_s21(args, base, logf, result, write_ck)
    elif args.mode == "harminv":
        run_harminv(args, base, logf, result, write_ck)

    result["wall_seconds_total"] = round(time.time() - t_start, 2)
    write_ck()
    logf(f"saved {ckpath} | total {result['wall_seconds_total']:.1f}s")


if __name__ == "__main__":
    main()
