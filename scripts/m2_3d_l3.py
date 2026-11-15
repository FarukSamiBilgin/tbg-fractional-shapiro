"""M2 AT-2: isolated 3D L3 photonic-crystal-slab cavity (mode, Q, V_mode).

Runs INSIDE the m2env WSL distro. Serial: /opt/mf/envs/mp/bin/python.
MPI (full runs): mpirun -np N /opt/mf/envs/mpmpi/bin/python. MEEP 3D FDTD,
units a = 1.

Geometry
--------
Triangular air-hole lattice (r/a = 0.30) etched through a Si slab
(eps = 11.68, thickness t = 0.5 a) with air cladding above/below. L3 defect =
three missing holes in a row along x (sites (-1,0),(0,0),(+1,0) removed); the
two cavity end-holes at (+/-2,0) are shifted OUTWARD by s (Akahane end-hole
shift, --s, units of a). This is the exact in-plane layout used by the verified
2D run scripts/m2_2d_l3.py, lifted into a 3D slab.

Supercell (units of a): SX x SY x SZ = 16 x 12*sqrt3/2 x 4.5, with
SZ = 0.5 (slab) + 2*1.0 (air) + 2*1.0 (PML) and PML = 1 a on ALL SIX faces
(no periodic BCs) -- an isolated cavity. Interior (non-PML) region is
14 x (12*sqrt3/2 - 2) x 2.5. NOTE: 16 a in x gives only ~4-5 mirror periods
each side of the ~5 a cavity; if the full-run Q_rad saturates below the true
out-of-plane limit, increase SX (in-plane leakage), analogous to a resolution
convergence check. Flagged for the full run, not the smoke.

Material loss (total-Q vs radiative-Q)
--------------------------------------
The Si slab carries tan_delta = 2e-5. MEEP's electric conductivity gives a
complex permittivity eps(w) = eps*(1 + i*sigma_D/w), so the loss tangent at
angular frequency w = 2*pi*f is Im/Re = sigma_D/w. To hit tan_delta AT the
resonance we set

    D_conductivity = 2*pi*nu0*tan_delta      (nu0 ~ fcen, MEEP units)

on the slab medium only (air holes/cladding stay lossless). harminv then
returns the TOTAL Q:  1/Q_tot = 1/Q_rad + 1/Q_mat, with the material ceiling
Q_mat = 1/tan_delta = 5.0e4. Pass --lossless to zero the conductivity and read
Q_rad directly; the pair of runs separates the two. (A convenience estimate
Q_rad ~ 1/(1/Q_tot - 1/Q_mat) is also reported from a lossy run, but the
--lossless run is the authority.)

Symmetries  --  phase convention VERIFIED, brief sign corrected
---------------------------------------------------------------
Source: an Hz point dipole at the cavity centre (origin, slab midplane z=0).
The L3 fundamental (Akahane Y1) has Hz field parity (x,y) = (-1,+1) in 2D
(measured in out/m2/l3_2d_s000.json: parity_x=-1.000, parity_y=+1.000); parity
carries over to the 3D slab, and TE-like slab modes are EVEN about z=0.
Target Hz field parity in 3D: (x, y, z) = (-1, +1, +1).

MEEP's mp.Mirror(dir, phase=p) enforces  F(mirror pt) = g * p * F(pt),  where g
is the INTRINSIC geometric transform of the component under that mirror. H is a
pseudovector (axial): under a reflection R with det R = -1, H_i -> det(R) R_ij
H_j. For the three coordinate mirrors the Hz (=A_z) geometric factor g is:
  * Mirror X (x->-x):  Hz -> -Hz   =>  g_x(Hz) = -1
  * Mirror Y (y->-y):  Hz -> -Hz   =>  g_y(Hz) = -1
  * Mirror Z (z->-z):  Hz -> +Hz   =>  g_z(Hz) = +1
(Both in-plane mirrors flip A_z; the z-mirror preserves it.) The realised Hz
field parity is therefore parity = g * p, so to obtain (-1, +1, +1):
  X: p = parity/g = (-1)/(-1) = +1
  Y: p = (+1)/(-1) = -1
  Z: p = (+1)/(+1) = +1
=> mp.Mirror(X, +1), mp.Mirror(Y, -1), mp.Mirror(Z, +1).

This CORRECTS the sign quoted in the kickoff brief ("Mirror(X, -1),
Mirror(Y, +1)"), which set the phase equal to the desired field parity and so
missed the pseudovector geometric factor on the two in-plane mirrors. The
correction is not just docs: the verified 2D production run used exactly
Mirror(X, +1)/Mirror(Y, -1) and measured parity (-1, +1) (l3_2d_s000.json),
while the opposite sector was shown to carry ~1e-7 of the cavity energy
(paper/decisions.md, 2026-07-23 2D-exploratory entry). The Z phase (+1) is
unaffected and agrees with the brief. Defaults below encode the corrected
phases; they are exposed as --px/--py/--pz for a cross-check, and --nosym runs
with NO symmetry and an off-centre source so the parity can be re-measured
independently (the empirical proof that the sector is right is that the chosen
symmetry yields a strong in-gap candidate at all).

Excitation / Q
--------------
Narrowband GaussianSource(fcen, fwidth=df) on Hz (fcen from --fcen, default
0.27; df=0.05), standard source cutoff. harminv samples Hz at an OFF-centre
monitor (0.30, 0.20, 0.0) -- Hz is odd in x so it vanishes on the x=0 plane;
the point sits in the L3 dielectric region and off both symmetry planes.
The ring-down is run in chunks of --chunk-periods optical periods (auto ~=
until/10 if 0); after EACH chunk the current harminv candidates + per-chunk
wall time are rewritten to the checkpoint JSON, so a killed run keeps every
completed chunk. Total ring-down window = --until-periods optical periods
(meep time = periods / fcen), on top of the source.

--resume: if a field dump from a prior --resume run exists, the fields are
reloaded and time-stepping continues (harminv RESTARTS its buffer from the
resume point -- a clean single-mode ring-down, which is fine for Q). Field
dumps are written per chunk only when --resume is set. If the dump/load API is
unavailable, the run recomputes from t=0 (documented, acceptable per brief).

--fields: add a volume DFT of (Ex,Ey,Ez) at fcen over the interior; after the
run compute V_mode = (integral eps|E|^2 dV) / max(eps|E|^2) [max-eps|E|^2
convention], report it in a^3 and um^3, measure the Hz field parity (x,y,z) as
a symmetry sanity check, and save the z=0 midplane |E|^2 (and eps) slice to
<base>_Efield.npz for the paper figure.

Units: nu = a/lambda; f_GHz = nu * c_um_GHz / a_final_um, a_final = 638.013 um
(out/m2/at0_slab3d.json, post-3D corrected). 3D TE-like gap (same file):
nu in [0.26904, 0.34813], midgap 0.30859 -> 145 GHz.

CLI: m2_3d_l3.py <project_root> --s <shift> [--rdef <scale>] --res <px/a>
     --until-periods <n> [--chunk-periods <n>] [--fcen <nu>] [--df <nu>]
     [--lossless] [--fields] [--resume] [--outdir <dir>] [--label <s>]
     [--px <+/-1> --py <+/-1> --pz <+/-1>] [--nosym] [--courant <c>] [--eps-avg]
     [--sx <a> --nrows <n>]
Deterministic - no RNG.
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

# ---- locked constants (out/m2/at0_slab3d.json, brief) -----------------------
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
# in-gap test band, widened by 3% of the gap width each side: the MPB gap edges
# are exact but a finite-res FDTD mode (esp. without subpixel averaging) can sit
# ~1-2% outside them, so a strict test spuriously rejects a real cavity mode.
_GTOL = 0.03 * (GAP3D[1] - GAP3D[0])
GAP_INGAP = (GAP3D[0] - _GTOL, GAP3D[1] + _GTOL)

# ---- supercell (units of a); SX/SY overridable via --sx/--nrows for the
#      in-plane-leakage (Q_rad lateral) convergence check ----------------------
NY_ROWS = 12
SX = 16.0
SY = NY_ROWS * SQRT3 / 2.0              # ~10.3923
DPML = 1.0                              # PML thickness, all six faces
AIR_GAP = 1.0                           # air cladding each side of the slab
SZ = 2.0 * (0.5 * T_OVER_A + AIR_GAP + DPML)   # = 4.5
# Holes must stay CLEAR of the lateral PML: a dielectric interface inside the
# PML makes the subpixel-averaged epsilon anisotropic there, which triggers the
# well-known MEEP PML instability (fields -> NaN/Inf mid-run, later at higher
# res). PML_PAD is the extra slab clearance between the outermost hole edge and
# the PML inner boundary; the slab Block still fills the whole cell (incl. the
# PML) -- isotropic and standard. Leaves an unpatterned slab margin ~1.2 a wide.
PML_PAD = 0.2

MON = mp.Vector3(0.30, 0.20, 0.0)       # harminv monitor (off both mirror planes)
DF_DEFAULT = 0.05


def to_ghz(nu):
    return nu * C_UM_GHZ / A_FINAL_UM


# ---- geometry ---------------------------------------------------------------
def _is_end_hole(j, off, i):
    """The 2 axial L3 end holes at (+/-2, 0)."""
    return (j == 0 and off == 0.0 and i in (-2, 2))


def _is_transverse_ring(j, off, i):
    """The 8 first-ring transverse neighbours: rows j=+/-1 (off=0.5),
    x = i+0.5 in {+/-0.5, +/-1.5}."""
    return (abs(j) == 1 and off == 0.5 and i in (-2, -1, 0, 1))


def build_geometry(s, lossy, fcen, rdef=1.0):
    """Si slab (lossy if requested) + triangular air holes with the L3 defect,
    outward end-hole shift s, and the 10 defect-adjacent holes' radius scaled by
    rdef (8 transverse first-ring neighbours + 2 axial end holes; end holes keep
    their s-shift). Same ring convention as scripts/m2_tune_screen.py so the 2D
    and 3D dnu/d(rdef) are comparable. Slab first so later holes carve it.
    Exactly mirror-symmetric about x=0, y=0, z=0."""
    sigma_D = (2.0 * math.pi * fcen * TAN_DELTA) if lossy else 0.0
    slab_mat = mp.Medium(epsilon=EPS_CORE, D_conductivity=sigma_D)
    geom = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, T_OVER_A),
                     center=mp.Vector3(), material=slab_mat)]
    r_ring = R_OVER_A * rdef
    # hole-CENTRE limits: outermost (regular-radius) hole edge sits PML_PAD
    # inside the PML edge (ring holes are central, so R_OVER_A sets the margin).
    xmax = SX / 2.0 - DPML - R_OVER_A - PML_PAD
    ymax = SY / 2.0 - DPML - R_OVER_A - PML_PAD
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
                continue                        # L3: three central holes gone
            xc = x
            rr = r_ring if (_is_end_hole(j, off, i)
                            or _is_transverse_ring(j, off, i)) else R_OVER_A
            if j == 0 and off == 0.0 and i == 2:
                xc = 2.0 + s                     # end-hole shifted outward
            elif j == 0 and off == 0.0 and i == -2:
                xc = -(2.0 + s)
            geom.append(mp.Cylinder(radius=rr, height=mp.inf,
                                    center=mp.Vector3(xc, y, 0.0),
                                    material=mp.air))
    return geom


def make_sim(args, fcen, df, symmetries, sources):
    return mp.Simulation(
        cell_size=mp.Vector3(SX, SY, SZ),
        geometry=build_geometry(args.s, lossy=(not args.lossless), fcen=fcen,
                                rdef=args.rdef),
        default_material=mp.air,
        boundary_layers=[mp.PML(DPML)],
        sources=sources,
        symmetries=symmetries,
        resolution=args.res,
        Courant=args.courant,
        eps_averaging=args.eps_avg,   # subpixel averaging (see --eps-avg note)
        force_complex_fields=False,
    )


# ---- harminv candidate helpers ----------------------------------------------
# A candidate is "clean" (trustworthy) only if it decays (Q>0), the harminv
# signal-processing error is small, and the amplitude is physical. harminv's
# own err_thresh (0.01) is too loose -- it let a nu=0.288 mode with |amp|=6.6e14
# through in the smoke -- so we tighten err and cap amp here.
ERR_CLEAN = 1.0e-3
AMP_MAX = 1.0e6


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
    """Most trustworthy in-gap mode: prefer clean ones, rank by harminv error
    (ascending) then amplitude (descending)."""
    ingap = [c for c in cands if c["in_gap"]]
    if not ingap:
        return None
    pool = [c for c in ingap if c["clean"]] or ingap
    return sorted(pool, key=lambda c: (c["err"], -c["amp"]))[0]


def candidates_from_bands(bands):
    """Convert mp.py_do_harminv output [(freq_c, amp_c, err), ...] into the
    same candidate dicts as extract_candidates (freq is complex: real=nu,
    imag=decay; Q = nu / (-2*decay))."""
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
    """Lightweight step function recording `comp` at `pt` every timestep.
    Returns (step_fn, buf) with buf['data'] the complex series and buf['dt']
    the sample interval. Under MPI get_field_point is collective (all ranks)."""
    buf = {"data": [], "dt": 0.0, "_t": None}

    def _step(sim):
        t = sim.meep_time()
        if buf["_t"] is not None and buf["dt"] == 0.0:
            buf["dt"] = t - buf["_t"]
        buf["_t"] = t
        buf["data"].append(sim.get_field_point(comp, pt))
    return _step, buf


def analyze_window(data, data_dt, fcen, df, periods_so_far, tau_est):
    """harminv on the LAST W periods of the recorded series.
    W = min(2*tau_est, periods_so_far); the FULL buffer is used while
    periods_so_far <= tau_est (or tau unknown) so a bootstrap estimate exists.
    Returns (candidates, window_info)."""
    n_tot = len(data)
    if tau_est and periods_so_far > tau_est:
        W = min(2.0 * tau_est, periods_so_far)
    else:
        W = periods_so_far
    if data_dt and data_dt > 0:
        n_win = int(round(W / fcen / data_dt))
    else:
        n_win = n_tot
    n_win = max(1, min(n_win, n_tot))
    win = list(data[-n_win:])
    bands = mp.py_do_harminv(win, data_dt, fcen - df / 2.0, fcen + df / 2.0,
                             100, 1.1, 50.0, mp.inf, 0.01, -1.0, -1.0)
    info = {"window_periods": W, "window_start_period": periods_so_far - W,
            "window_end_period": periods_so_far, "window_samples": n_win,
            "windowed": bool(W < periods_so_far - 1e-9)}   # trailing window active
    return candidates_from_bands(bands), info


# ---- field post-processing (V_mode, parity, midplane slice) -----------------
def _interior():
    return (mp.Vector3(), mp.Vector3(SX - 2 * DPML, SY - 2 * DPML, SZ - 2 * DPML))


def add_field_monitors(sim, fcen):
    c, sz = _interior()
    dft_E = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez], fcen, 0, 1, center=c, size=sz)
    dft_H = sim.add_dft_fields([mp.Hz], fcen, 0, 1, center=c, size=sz)
    return dft_E, dft_H


def _crop_like(a, shape):
    sl = tuple(slice(0, min(a.shape[k], shape[k])) for k in range(a.ndim))
    return a[sl]


def compute_fields(sim, dft_E, dft_H, fcen, res, base, logf):
    c, sz = _interior()
    Ex = np.asarray(sim.get_dft_array(dft_E, mp.Ex, 0))
    Ey = np.asarray(sim.get_dft_array(dft_E, mp.Ey, 0))
    Ez = np.asarray(sim.get_dft_array(dft_E, mp.Ez, 0))
    Hz = np.asarray(sim.get_dft_array(dft_H, mp.Hz, 0))
    eps = np.asarray(sim.get_array(center=c, size=sz, component=mp.Dielectric)).real

    shp = tuple(min(Ex.shape[k], Ey.shape[k], Ez.shape[k], eps.shape[k])
                for k in range(3))
    Ex, Ey, Ez = _crop_like(Ex, shp), _crop_like(Ey, shp), _crop_like(Ez, shp)
    eps = _crop_like(eps, shp)
    E2 = (np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
    u = eps * E2                       # eps|E|^2 (energy-density-like weight)

    dV = (1.0 / res) ** 3
    umax = float(u.max())
    vmode_a3 = float(u.sum() * dV / umax) if umax > 0 else float("nan")
    vmode_um3 = vmode_a3 * (A_FINAL_UM ** 3)

    # Hz parity (independent of imposed symmetry only under --nosym; a sanity
    # check that MEEP realised the intended sector otherwise).
    Hzc = _crop_like(Hz, shp)
    denom = float(np.vdot(Hzc, Hzc).real)
    if denom > 0:
        px = float(np.vdot(Hzc, Hzc[::-1, :, :]).real) / denom
        py = float(np.vdot(Hzc, Hzc[:, ::-1, :]).real) / denom
        pz = float(np.vdot(Hzc, Hzc[:, :, ::-1]).real) / denom
    else:
        px = py = pz = float("nan")

    # z=0 midplane slice of |E|^2 for the figure
    try:
        xs, ys, zs, _w = sim.get_array_metadata(center=c, size=sz)
        xs = np.asarray(xs)[:shp[0]]
        ys = np.asarray(ys)[:shp[1]]
        zs = np.asarray(zs)[:shp[2]]
        kz = int(np.argmin(np.abs(zs)))
    except Exception:
        xs = np.linspace(-(SX - 2 * DPML) / 2, (SX - 2 * DPML) / 2, shp[0])
        ys = np.linspace(-(SY - 2 * DPML) / 2, (SY - 2 * DPML) / 2, shp[1])
        kz = shp[2] // 2
    E2_mid = E2[:, :, kz]
    eps_mid = eps[:, :, kz]
    npz = f"{base}_Efield.npz"
    if mp.am_really_master():              # master-only file write (MPI-safe)
        np.savez(npz, x=xs, y=ys, z_index=kz, E2_mid=E2_mid, eps_mid=eps_mid,
                 fcen=fcen, a_um=A_FINAL_UM, res=res)
    logf(f"  fields: V_mode = {vmode_a3:.4f} a^3 = {vmode_um3:.4e} um^3 ; "
         f"parity(x,y,z)=({px:+.3f},{py:+.3f},{pz:+.3f}) ; saved {npz}")
    return {"V_mode_a3": vmode_a3, "V_mode_um3": vmode_um3,
            "V_mode_convention": "integral(eps|E|^2 dV)/max(eps|E|^2)",
            "parity_x": px, "parity_y": py, "parity_z": pz,
            "midplane_npz": npz, "grid_shape": list(shp)}


# ---- resume (best-effort field dump/load) -----------------------------------
def try_dump_fields(sim, fname, logf):
    for attr, kw in (("dump_fields", {}), ("dump", {"dump_structure": False,
                                                    "dump_fields": True})):
        fn = getattr(sim, attr, None)
        if fn is None:
            continue
        try:
            fn(fname, **kw)
            return True
        except Exception as e:                     # noqa: BLE001
            logf(f"  [resume] {attr} failed: {e!r}")
    return False


def try_load_fields(sim, fname, logf):
    sim.init_sim()
    for attr, kw in (("load_fields", {}), ("load", {"load_structure": False,
                                                    "load_fields": True})):
        fn = getattr(sim, attr, None)
        if fn is None:
            continue
        try:
            fn(fname, **kw)
            return True
        except Exception as e:                     # noqa: BLE001
            logf(f"  [resume] {attr} failed: {e!r}")
    return False


# ---- main -------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="M2 AT-2 3D L3 cavity (MEEP FDTD)")
    p.add_argument("root", help="project root (WSL path, e.g. /mnt/c/.../TBG FOTONIK JJ)")
    p.add_argument("--s", type=float, default=0.0, help="end-hole shift (units a)")
    p.add_argument("--rdef", type=float, default=1.0,
                   help="radius scale of the 10 defect-adjacent holes (8 "
                        "transverse first-ring + 2 axial end holes; end holes "
                        "keep their s-shift). >1 blue-shifts the mode toward "
                        "midgap. Same ring as scripts/m2_tune_screen.py.")
    p.add_argument("--res", type=int, default=8, help="resolution px/a")
    p.add_argument("--until-periods", dest="until_periods", type=float,
                   default=200.0, help="ring-down window in optical periods")
    p.add_argument("--chunk-periods", dest="chunk_periods", type=float,
                   default=0.0, help="checkpoint chunk size (periods); 0=auto")
    p.add_argument("--fcen", type=float, default=0.27, help="source centre nu")
    p.add_argument("--df", type=float, default=DF_DEFAULT, help="source fwidth")
    p.add_argument("--lossless", action="store_true",
                   help="zero D_conductivity -> harminv Q = Q_rad")
    p.add_argument("--fields", action="store_true",
                   help="compute V_mode, parity, midplane |E|^2 slice")
    p.add_argument("--resume", action="store_true",
                   help="reload a prior field dump and continue (best-effort)")
    p.add_argument("--outdir", default="", help="output dir (default <root>/out/m2)")
    p.add_argument("--label", default="smoke", help="filename label")
    p.add_argument("--px", type=int, default=1, help="Mirror(X) phase (verified +1)")
    p.add_argument("--py", type=int, default=-1, help="Mirror(Y) phase (verified -1)")
    p.add_argument("--pz", type=int, default=1, help="Mirror(Z) phase (verified +1)")
    p.add_argument("--nosym", action="store_true",
                   help="no symmetry + off-centre source (parity verification)")
    p.add_argument("--sx", type=float, default=SX,
                   help="in-plane cell size along x (a); bump for Q_rad lateral check")
    p.add_argument("--nrows", type=int, default=NY_ROWS,
                   help="rows along y (cell SY = nrows*sqrt3/2); bump for lateral check")
    p.add_argument("--courant", type=float, default=0.5,
                   help="Courant stability factor")
    p.add_argument("--eps-avg", dest="eps_avg", action="store_true",
                   help="enable subpixel averaging. OFF by default: for this "
                        "eps=11.68 slab it violates passivity and the field "
                        "grows exponentially at res12 (Courant/PML-independent). "
                        "It IS stable at res18 -- for the most accurate Q_rad "
                        "run res18 WITH --eps-avg (nu~0.277 in-gap, Q~7.4e3). "
                        "res12 must stay noavg (mode drifts to the gap edge).")
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root
    fcen, df = args.fcen, args.df
    global SX, SY, NY_ROWS
    SX = args.sx
    NY_ROWS = args.nrows
    SY = NY_ROWS * SQRT3 / 2.0
    outdir = args.outdir or f"{root}/out/m2"
    os.makedirs(outdir, exist_ok=True)
    tag = f"{int(round(args.s * 100)):03d}"
    rdef_tag = (f"_rdef{int(round(args.rdef * 100)):03d}"
                if abs(args.rdef - 1.0) > 1e-9 else "")   # keep name at rdef==1
    base = (f"{outdir}/l3_3d_{args.label}_s{tag}{rdef_tag}_res{args.res}"
            f"{'_lossless' if args.lossless else ''}"
            f"{'_nosym' if args.nosym else ''}")
    logpath = base + ".log"
    ckpath = base + ".json"
    dumppath = base + "_fields.dump"

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
    logf(f"M2 AT-2 3D L3 | meep {mp.__version__} py {platform.python_version()} | "
         f"s={args.s} res={args.res} fcen={fcen} df={df} "
         f"lossy={not args.lossless} fields={args.fields} nosym={args.nosym}")

    # ---- symmetries + source -------------------------------------------------
    if args.nosym:
        symmetries = []
        src_pt = mp.Vector3(0.123, 0.081, 0.037)     # off every mirror plane
    else:
        symmetries = [mp.Mirror(mp.X, phase=args.px),
                      mp.Mirror(mp.Y, phase=args.py),
                      mp.Mirror(mp.Z, phase=args.pz)]
        src_pt = mp.Vector3()                          # cavity centre / midplane
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Hz,
                         center=src_pt)]
    mon = MON

    # ---- chunk plan ----------------------------------------------------------
    total_periods = args.until_periods
    chunk_periods = args.chunk_periods
    if chunk_periods <= 0.0:
        chunk_periods = max(25.0, total_periods / 10.0)
    chunk_periods = min(chunk_periods, total_periods)
    n_chunks = max(1, int(round(total_periods / chunk_periods)))
    chunk_periods = total_periods / n_chunks
    t_chunk_meep = chunk_periods / fcen
    logf(f"plan: {total_periods:g} periods in {n_chunks} chunk(s) of "
         f"{chunk_periods:g} periods (= {t_chunk_meep:.2f} meep-time each); "
         f"cell {SX}x{SY:.3f}x{SZ} a, PML {DPML} a all sides")

    # ---- build sim -----------------------------------------------------------
    t_build0 = time.time()
    sim = make_sim(args, fcen, df, symmetries, sources)
    collect, buf = make_collector(mp.Hz, mon)   # trailing-window harminv source
    dft_E = dft_H = None
    if args.fields:
        dft_E, dft_H = add_field_monitors(sim, fcen)

    # ---- resume --------------------------------------------------------------
    driven_done = False
    resumed = False
    periods_done = 0.0
    chunks = []
    if args.resume and os.path.exists(dumppath) and os.path.exists(ckpath):
        try:
            prev = json.load(open(ckpath))
            if try_load_fields(sim, dumppath, logf):
                resumed = True
                driven_done = True
                periods_done = float(prev.get("periods_done", 0.0))
                chunks = prev.get("chunks", [])
                logf(f"[resume] loaded fields from {dumppath}; "
                     f"continuing at {periods_done:g} periods "
                     "(harminv buffer restarts)")
            else:
                logf("[resume] field load unavailable; recomputing from t=0")
        except Exception as e:                         # noqa: BLE001
            logf(f"[resume] failed ({e!r}); recomputing from t=0")
    t_build = time.time() - t_build0
    logf(f"build+monitors: {t_build:.2f} s")

    result = {
        "task": "M2 AT-2: isolated 3D L3 photonic-crystal-slab cavity",
        "rng": "deterministic - no RNG",
        "meep_version": mp.__version__,
        "python": platform.python_version(),
        "status": "running",
        "params": {
            "s": args.s, "res": args.res, "fcen": fcen, "df": df,
            "rdef": args.rdef,
            "rdef_ring": "8 transverse (j=+/-1, x=+/-0.5,+/-1.5) + 2 end (+/-2,0)",
            "courant": args.courant, "pml_pad": PML_PAD,
            "eps_averaging": args.eps_avg,
            "lossy": (not args.lossless), "fields": args.fields,
            "nosym": args.nosym, "mirror_phase_X": args.px,
            "mirror_phase_Y": args.py, "mirror_phase_Z": args.pz,
            "r_over_a": R_OVER_A, "t_over_a": T_OVER_A, "eps_core": EPS_CORE,
            "tan_delta": TAN_DELTA, "Q_mat_ceiling": Q_MAT,
            "cell": [SX, SY, SZ], "dpml": DPML, "air_gap": AIR_GAP,
            "a_final_um": A_FINAL_UM, "gap3d_a_over_lambda": list(GAP3D),
            "monitor": [mon.x, mon.y, mon.z],
            "source_pt": [src_pt.x, src_pt.y, src_pt.z],
            "total_periods": total_periods, "chunk_periods": chunk_periods,
            "n_chunks": n_chunks, "t_chunk_meep": t_chunk_meep,
            "resumed": resumed,
        },
        "build_seconds": round(t_build, 2),
        "chunks": chunks,
    }

    def write_ck():
        if not mp.am_really_master():      # master-only: avoid MPI write races
            return
        with open(ckpath, "w") as fh:
            json.dump(result, fh, indent=1)

    write_ck()

    # ---- chunked ring-down + trailing-window harminv ------------------------
    best_Q = None                              # running clean in-gap Q -> tau_est
    for ci in range(n_chunks):
        t0 = time.time()
        if not driven_done:
            sim.run(mp.after_sources(collect), until_after_sources=t_chunk_meep)
            driven_done = True
        else:
            sim.run(collect, until=t_chunk_meep)
        dt = time.time() - t0
        periods_done += chunk_periods
        tau_est = (best_Q / math.pi) if best_Q else None
        cands, win = analyze_window(buf["data"], buf["dt"], fcen, df,
                                    periods_done, tau_est)
        big = best_in_gap(cands)
        if big and big.get("clean"):
            best_Q = big["Q"]                  # inform next chunk's window
        rec = {"chunk_index": len(chunks), "periods_this_chunk": chunk_periods,
               "cum_periods": periods_done, "meep_time": float(sim.round_time()),
               "wall_seconds": round(dt, 2), "n_harminv_modes": len(cands),
               "window": win, "candidates": cands, "best_in_gap": big}
        chunks.append(rec)
        result["chunks"] = chunks
        result["periods_done"] = periods_done
        write_ck()
        if args.resume:
            if try_dump_fields(sim, dumppath, logf):
                result["dump_path"] = dumppath
        tag_g = (f"nu0={big['nu']:.5f} ({big['GHz']:.3f} GHz) Q={big['Q']:.1f} "
                 f"amp={big['amp']:.3e}") if big else "no in-gap candidate yet"
        logf(f"chunk {ci + 1}/{n_chunks}: +{chunk_periods:g}p "
             f"(cum {periods_done:g}p, t={sim.round_time():.1f}) {dt:.2f}s | "
             f"win[{win['window_start_period']:.0f},{win['window_end_period']:.0f}]p"
             f"{'*' if win['windowed'] else ''} | {len(cands)} modes | {tag_g}")

    # ---- summary -------------------------------------------------------------
    final_last = chunks[-1]["best_in_gap"] if chunks else None
    # Headline = clean candidate from the LARGEST analysed window (the converged
    # 2*tau trailing window once tau is known); fall back to min-err, then last.
    clean_bests = [(c["cum_periods"], c["best_in_gap"], c["window"])
                   for c in chunks
                   if c.get("best_in_gap") and c["best_in_gap"].get("clean")]
    if clean_bests:
        headline = max(clean_bests, key=lambda t: t[0])[1]
    else:
        anyb = [c["best_in_gap"] for c in chunks if c.get("best_in_gap")]
        headline = (min(anyb, key=lambda b: b["err"]) if anyb else final_last)
    # Q stationarity across the last TWO clean (trailing-window) analyses.
    q_trend = [{"cum_periods": p, "nu": b["nu"], "Q": b["Q"], "err": b["err"],
                "windowed": w.get("windowed"),
                "window_periods": w.get("window_periods")}
               for (p, b, w) in clean_bests]
    q_stationary = None
    if len(q_trend) >= 2 and q_trend[-1]["Q"]:
        q_stationary = bool(abs(q_trend[-1]["Q"] - q_trend[-2]["Q"])
                            / abs(q_trend[-1]["Q"]) < 0.10)
    # Q error bars from the spread of clean trailing-window analyses.
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

    result["result"] = {
        "headline_in_gap": headline,
        "last_window_best": final_last,
        "q_trend_clean": q_trend,
        "q_stationary_last2_within_10pct": q_stationary,
        "q_window_spread": q_spread,
        "n_harminv_modes_final": (chunks[-1]["n_harminv_modes"] if chunks else 0),
        "Q_mat_ceiling": Q_MAT,
    }
    if headline:
        tau_p = headline["Q"] / math.pi        # amplitude 1/e time in periods
        u_over_tau = total_periods / tau_p if tau_p else None
        result["result"]["ringdown_tau_periods"] = tau_p
        result["result"]["until_over_tau"] = u_over_tau
        result["result"]["Q_reliable"] = bool(
            q_stationary and u_over_tau and u_over_tau >= 1.0)
        if not args.lossless:
            inv = 1.0 / headline["Q"] - 1.0 / Q_MAT
            result["result"]["Q_rad_estimate_from_lossy"] = (
                (1.0 / inv) if inv > 0 else None)
            result["result"]["Q_rad_note"] = (
                "estimate = 1/(1/Q_tot - 1/Q_mat); authoritative Q_rad is the "
                "--lossless run")

    if args.fields:
        try:
            result["fields"] = compute_fields(sim, dft_E, dft_H, fcen,
                                               args.res, base, logf)
        except Exception as e:                         # noqa: BLE001
            logf(f"  fields post-processing FAILED: {e!r}")
            result["fields"] = {"error": repr(e)}

    result["status"] = "done"
    result["wall_seconds_total"] = round(time.time() - t_start, 2)
    # per-period stepping cost from pure ring-down chunks (exclude chunk 0 which
    # also plays the source) -> the number the full-run extrapolation uses.
    step_chunks = [c["wall_seconds"] for c in chunks[1:]] or [chunks[0]["wall_seconds"]]
    result["stepping_seconds_per_period"] = (
        (sum(step_chunks) / len(step_chunks)) / chunk_periods)
    write_ck()

    if headline:
        rel = result["result"].get("Q_reliable")
        logf(f"DONE: nu0={headline['nu']:.6f} ({headline['GHz']:.3f} GHz) "
             f"Q={headline['Q']:.1f} (err={headline['err']:.2e}, "
             f"reliable={rel}) in_gap={headline['in_gap']} | "
             f"total {result['wall_seconds_total']:.1f}s | saved {ckpath}")
    else:
        logf(f"DONE: NO in-gap candidate | total "
             f"{result['wall_seconds_total']:.1f}s | saved {ckpath}")


if __name__ == "__main__":
    main()
