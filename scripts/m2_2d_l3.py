"""M2 production item 1: 2D L3 photonic-crystal cavity (effective-index model).

Runs INSIDE the m2env WSL distro (conda env mp, meep). 2D TE (Ex, Ey, Hz).
Triangular lattice of air holes (r/a=0.30) in a background of the AT-0 converged
slab effective permittivity eps_eff=7.6982457690 (out/m2/at0_neff.json). L3 =
three missing holes in a row along x (removed sites (-1,0),(0,0),(+1,0)); the
two cavity end holes at (+/-2,0) are shifted OUTWARD by s (Akahane end-hole
shift). Cell 18a x 12a*sqrt(3)/2, PML 1a (inside cell), resolution 20, subpixel
averaging on. Mirror symmetries about x=0 and y=0 through the cavity centre; a
Hz point source at the centre with the parity that selects the fundamental
mode. Q via harminv on the ring-down (source cut off, then >= 3x decay).

Frequencies are reported in MEEP units nu=a/lambda and converted to GHz with
a_final_um=602.886947874021 (f_GHz = nu * c_um_GHz / a_final_um).

Subcommands (argv[2]):
  diag                 broadband no-symmetry mode-finder over the gap -> list
                       all resonances (freq, Q, amp); writes l3_2d_diag.json
  combos <f0>          center Hz source, test the 4 mirror-phase combinations
                       at narrowband f0, report which excites the mode
                       (locks the production symmetry); writes l3_2d_combos.json
  run <s> <px> <py> [T]  production run at end-hole shift s with mirror phases
                       (px,py); full harminv ring-down (T = until_after_sources,
                       auto-scaled to >=3x ringdown by default); writes
                       l3_2d_s<tag>.json with nu0, Q, parity.

Usage: python m2_2d_l3.py <project_root_wsl_path> <subcommand> [args...]
Deterministic - no RNG.
"""
import json
import math
import os
import sys

import numpy as np
import meep as mp

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
SUB = sys.argv[2] if len(sys.argv) > 2 else "diag"

# ---- locked constants -------------------------------------------------------
SQRT3 = math.sqrt(3.0)
R_OVER_A = 0.30
EPS_EFF = 7.698245769014287           # AT-0 converged slab eps_eff
A_FINAL_UM = 602.886947874021          # AT-0 converged lattice constant
C_UM_GHZ = 299792.458
GAP = (0.2576086910560098, 0.32566092221171666)   # AT-0 2D TE gap (a/lambda)
MIDGAP = 0.2916356652886985

RES = 20
DPML = 1.0
SX = 18.0
SY = 12.0 * SQRT3 / 2.0                 # = 6*sqrt(3) ~ 10.3923


def to_ghz(nu):
    return nu * C_UM_GHZ / A_FINAL_UM


def build_geometry(s):
    """Air holes on the triangular lattice, L3 defect + end-hole shift s.
    Fills the whole cell (incl. PML). Exactly mirror-symmetric about x=0,y=0."""
    holes = []
    xmax = SX / 2.0 + R_OVER_A + 1e-9
    ymax = SY / 2.0 + R_OVER_A + 1e-9
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
            # L3: remove the 3 central holes in row j=0 (off=0 => x=i integer)
            if j == 0 and off == 0.0 and i in (-1, 0, 1):
                continue
            xc = x
            # end-hole shift outward by s (holes at (+/-2, 0))
            if j == 0 and off == 0.0 and i == 2:
                xc = 2.0 + s
            elif j == 0 and off == 0.0 and i == -2:
                xc = -(2.0 + s)
            holes.append(mp.Cylinder(radius=R_OVER_A, height=mp.inf,
                                     center=mp.Vector3(xc, y),
                                     material=mp.air))
    return holes


def make_sim(s, symmetries, sources):
    return mp.Simulation(
        cell_size=mp.Vector3(SX, SY),
        geometry=build_geometry(s),
        default_material=mp.Medium(epsilon=EPS_EFF),
        boundary_layers=[mp.PML(DPML)],
        sources=sources,
        symmetries=symmetries,
        resolution=RES,
        eps_averaging=True,      # subpixel averaging ON
        force_complex_fields=False,
    )


def field_parity(sim, f0):
    """DFT the Hz field at f0 over the full cell; return overlap parities
    sym_x = <H, H(-x,y)>/<H,H>, sym_y = <H, H(x,-y)>/<H,H> (real part)."""
    d = sim.add_dft_fields([mp.Hz], f0, 0, 1,
                           center=mp.Vector3(), size=mp.Vector3(SX, SY))
    return d


def parity_from_dft(sim, dft):
    H = sim.get_dft_array(dft, mp.Hz, 0)
    H = np.asarray(H)
    # H indexed [ix, iy]; flips give the mirror images
    denom = float(np.vdot(H, H).real)
    if denom <= 0:
        return None, None, 0.0
    sx = float(np.vdot(H, H[::-1, :]).real) / denom
    sy = float(np.vdot(H, H[:, ::-1]).real) / denom
    return sx, sy, denom


# =============================================================================
if SUB == "diag":
    # broadband, no symmetry, off-centre source to excite every gap mode
    fcen = 0.5 * (GAP[0] + GAP[1])
    df = 1.15 * (GAP[1] - GAP[0])
    src_pt = mp.Vector3(0.123, 0.081)
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                         component=mp.Hz, center=src_pt)]
    sim = make_sim(0.0, [], sources)
    h = mp.Harminv(mp.Hz, src_pt, fcen, df)
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 600.0
    sim.run(mp.after_sources(h), until_after_sources=T)
    modes = []
    for m in h.modes:
        f = float(m.freq)
        modes.append({"nu": f, "GHz": to_ghz(f), "Q": float(m.Q),
                      "amp": float(abs(m.amp)), "decay": float(m.decay),
                      "in_gap": bool(GAP[0] <= f <= GAP[1])})
    modes.sort(key=lambda d: d["nu"])
    out = {"task": "M2 2D L3 diagnostic (broadband, no symmetry)",
           "rng": "deterministic - no RNG",
           "meep_version": mp.__version__,
           "params": {"r_over_a": R_OVER_A, "eps_eff": EPS_EFF, "res": RES,
                      "cell": [SX, SY], "dpml": DPML,
                      "a_final_um": A_FINAL_UM, "gap_a_over_lambda": list(GAP),
                      "source_pt": [src_pt.x, src_pt.y], "fcen": fcen, "df": df,
                      "until_after_sources": T},
           "modes": modes}
    os.makedirs(f"{ROOT}/out/m2", exist_ok=True)
    with open(f"{ROOT}/out/m2/l3_2d_diag.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print("=== L3 2D diagnostic modes (broadband, no symmetry) ===")
    for m in modes:
        tag = "  <-- IN GAP" if m["in_gap"] else ""
        print(f"  nu={m['nu']:.5f} ({m['GHz']:.3f} GHz)  Q={m['Q']:.1f}  "
              f"amp={m['amp']:.3e}{tag}")
    print(f"saved {ROOT}/out/m2/l3_2d_diag.json")

elif SUB == "combos":
    f0 = float(sys.argv[3])
    df = 0.02
    results = {}
    for px in (+1, -1):
        for py in (+1, -1):
            syms = [mp.Mirror(mp.X, phase=px), mp.Mirror(mp.Y, phase=py)]
            sources = [mp.Source(mp.GaussianSource(f0, fwidth=df),
                                 component=mp.Hz, center=mp.Vector3())]
            sim = make_sim(0.0, syms, sources)
            h = mp.Harminv(mp.Hz, mp.Vector3(0.30, 0.20), f0, df)
            sim.run(mp.after_sources(h), until_after_sources=300)
            best = None
            for m in h.modes:
                if abs(m.freq - f0) < 0.02 and abs(m.amp) > 0:
                    if best is None or abs(m.amp) > best["amp"]:
                        best = {"nu": float(m.freq), "Q": float(m.Q),
                                "amp": float(abs(m.amp))}
            results[f"{px:+d}{py:+d}"] = best
            print(f"phases (X={px:+d},Y={py:+d}): "
                  f"{best if best else 'no mode found'}")
    out = {"task": "M2 2D L3 mirror-phase combo test (center Hz source)",
           "meep_version": mp.__version__, "f0_test": f0, "df": df,
           "results": results, "rng": "deterministic - no RNG"}
    with open(f"{ROOT}/out/m2/l3_2d_combos.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"saved {ROOT}/out/m2/l3_2d_combos.json")

elif SUB == "run":
    s = float(sys.argv[3])
    px = int(sys.argv[4])
    py = int(sys.argv[5])
    fcen = float(sys.argv[6]) if len(sys.argv) > 6 else MIDGAP
    Targ = float(sys.argv[7]) if len(sys.argv) > 7 else None
    df = 0.05
    syms = [mp.Mirror(mp.X, phase=px), mp.Mirror(mp.Y, phase=py)]
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df),
                         component=mp.Hz, center=mp.Vector3())]
    sim = make_sim(s, syms, sources)
    mon = mp.Vector3(0.30, 0.20)
    h = mp.Harminv(mp.Hz, mon, fcen, df)
    # first pass: locate + rough Q
    sim.run(mp.after_sources(h), until_after_sources=400)
    cand = [m for m in h.modes if GAP[0] <= m.freq <= GAP[1] and abs(m.amp) > 0]
    cand.sort(key=lambda m: -abs(m.amp))
    if not cand:
        print("NO MODE in gap with these phases; abort")
        sys.exit(2)
    m0 = cand[0]
    Qrough = float(m0.Q)
    f0 = float(m0.freq)
    # ring-down time budget: >= 3x amplitude 1/e time tau = Q/(pi f0)
    tau = Qrough / (math.pi * f0)
    T = Targ if Targ is not None else max(400.0, 3.2 * tau)
    T = min(T, 60000.0)     # cap so a single call stays < 10 min
    # second pass: clean narrowband harminv over the requested window + parity
    sim.reset_meep()
    sources = [mp.Source(mp.GaussianSource(f0, fwidth=0.03),
                         component=mp.Hz, center=mp.Vector3())]
    sim = make_sim(s, syms, sources)
    h2 = mp.Harminv(mp.Hz, mon, f0, 0.05)
    dft = field_parity(sim, f0)
    sim.run(mp.after_sources(h2), until_after_sources=T)
    sx, sy, dnorm = parity_from_dft(sim, dft)
    best = None
    for m in h2.modes:
        if GAP[0] <= m.freq <= GAP[1] and abs(m.amp) > 0:
            if best is None or abs(m.amp) > abs(best.amp):
                best = m
    nu0 = float(best.freq)
    Q = float(best.Q)
    tag = f"{int(round(s*100)):03d}"
    out = {"task": "M2 2D L3 production run",
           "rng": "deterministic - no RNG",
           "meep_version": mp.__version__,
           "params": {"s": s, "mirror_phase_X": px, "mirror_phase_Y": py,
                      "r_over_a": R_OVER_A, "eps_eff": EPS_EFF, "res": RES,
                      "cell": [SX, SY], "dpml": DPML, "a_final_um": A_FINAL_UM,
                      "fcen_first_pass": fcen, "until_after_sources": T,
                      "ringdown_tau": tau, "T_over_tau": T / tau},
           "result": {"nu0": nu0, "f0_GHz": to_ghz(nu0), "Q_2d": Q,
                      "Q_rough_firstpass": Qrough,
                      "in_gap": bool(GAP[0] <= nu0 <= GAP[1]),
                      "parity_x": sx, "parity_y": sy,
                      "n_harminv_modes": len(h2.modes)}}
    with open(f"{ROOT}/out/m2/l3_2d_s{tag}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"s={s}: nu0={nu0:.6f} ({to_ghz(nu0):.3f} GHz) Q_2d={Q:.1f} "
          f"parity(x,y)=({sx:+.3f},{sy:+.3f}) T/tau={T/tau:.2f}")
    print(f"saved {ROOT}/out/m2/l3_2d_s{tag}.json")

else:
    print(f"unknown subcommand {SUB!r}")
    sys.exit(1)
