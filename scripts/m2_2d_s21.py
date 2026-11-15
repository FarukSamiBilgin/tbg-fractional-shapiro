"""M2 production item 2/3: 2D S21 ladder, side-coupled W1 + L3 cavity.

Runs INSIDE the m2env WSL distro (conda env mp, meep). 2D TE (Ex,Ey,Hz),
background eps_eff=7.6982457690 (AT-0), r/a=0.30, resolution 20 (or 16 - see
--res), PML 1a, subpixel averaging on.

Geometry: a W1 waveguide (one full missing row, along x at the waveguide line)
plus the s*-optimized L3 cavity (s*=0 from m2_2d_l3.py Q(s) scan) placed
N_rows rows to the side. The cavity is an L3 (remove the 3 holes nearest the
cavity centre in row j=N_rows); for even N_rows the cavity centres on x=0, for
odd N_rows on x=0.5 (that row's lattice offset) - a valid L3 either way.

No mirror symmetry is usable for the flux S21 runs: the eigenmode source
propagates along +x (breaks x-mirror) and the cavity sits on one side of the
guide (breaks y-mirror). The harminv 'probe' run excites only the localised
cavity, so it uses Mirror(X) when the cavity is x-centred (even N_rows).

Frequencies: nu=a/lambda; GHz via a_final_um=602.886947874021.

Subcommands (argv[2]):
  probe <N> [--res R]        excite the cavity directly, harminv ring-down ->
                             Q_loaded(N), nu0(N); writes s21_2d_probe_N<N>.json
  s21   <N> [--res R] [--tmax T]
                             normalisation run (waveguide only) + cavity run,
                             same eigenmode source + far flux monitor ->
                             S21(f)=flux/flux_norm; writes s21_2d_N<N>.json
Usage: python m2_2d_s21.py <project_root_wsl> <sub> <N_rows> [flags]
Deterministic - no RNG.
"""
import json
import math
import os
import sys

import numpy as np
try:
    import meep as mp            # not needed for the meep-free 'analyze' step
except Exception:
    mp = None

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
SUB = sys.argv[2] if len(sys.argv) > 2 else "probe"
N_ROWS = int(sys.argv[3]) if len(sys.argv) > 3 else 2


def flag(name, default):
    if name in sys.argv:
        return sys.argv[sys.argv.index(name) + 1]
    return default


RES = int(flag("--res", "20"))
TMAX = float(flag("--tmax", "0"))     # 0 => auto
EIG_BAND = int(flag("--eband", "1"))  # guided band to launch/measure

# ---- locked constants -------------------------------------------------------
SQRT3 = math.sqrt(3.0)
R_OVER_A = 0.30
EPS_EFF = 7.698245769014287
A_FINAL_UM = 602.886947874021
C_UM_GHZ = 299792.458
GAP = (0.2576086910560098, 0.32566092221171666)
S_STAR = 0.0                          # 2D-optimal end-hole shift (m2_2d_l3)

DPML = 1.0
SX = 24.0                             # cell width (x, propagation)
X_SRC = -9.0                          # eigenmode source x
X_MON = +9.0                          # transmitted-flux monitor x
JB = 4                                # crystal rows below the waveguide


def to_ghz(nu):
    return nu * C_UM_GHZ / A_FINAL_UM


def layout(N):
    """Return (cell_y, y_wg, y_cav, x_cav, JT, yc). Content centred in cell."""
    JT = N + 5                                    # 5 rows above the cavity
    y_top = JT * (SQRT3 / 2.0)
    y_bot = -JB * (SQRT3 / 2.0)
    yc = 0.5 * (y_top + y_bot)
    y_wg = 0.0 - yc                               # waveguide row (j=0) in cell
    y_cav = N * (SQRT3 / 2.0) - yc
    x_cav = 0.5 * (N % 2)
    cell_y = 2.0 * (max(abs(y_top - yc), abs(y_bot - yc)) + 0.9 + DPML)
    return cell_y, y_wg, y_cav, x_cav, JT, yc


CAP_WG = "--cap" in sys.argv            # fill the waveguide row (diagnostic)


def build(N, with_cavity):
    cell_y, y_wg, y_cav, x_cav, JT, yc = layout(N)
    holes = []
    xmax = SX / 2.0 + R_OVER_A + 1e-9
    imax = int(math.ceil(xmax)) + 2
    for j in range(-JB, JT + 1):
        y = j * (SQRT3 / 2.0) - yc
        off = 0.5 * (j % 2)
        for i in range(-imax, imax + 1):
            x = i + off
            if abs(x) > xmax:
                continue
            if j == 0 and not CAP_WG:         # W1 waveguide: whole row removed
                continue
            if with_cavity and j == N and i in (-1, 0, 1):   # L3 defect
                continue
            xc = x
            if with_cavity and j == N and i == 2:
                xc = x + S_STAR
            elif with_cavity and j == N and i == -2:
                xc = x - S_STAR
            holes.append(mp.Cylinder(radius=R_OVER_A, height=mp.inf,
                                     center=mp.Vector3(xc, y), material=mp.air))
    return holes, cell_y, y_wg, y_cav, x_cav


def run_until(sim, pt, tmax):
    decay = mp.stop_when_fields_decayed(50, mp.Hz, pt, 1e-4)

    def stopfn(s):
        if s.meep_time() > tmax:
            return True
        return decay(s)
    sim.run(until_after_sources=stopfn)


# =============================================================================
if SUB == "probe":
    N = N_ROWS
    holes, cell_y, y_wg, y_cav, x_cav = build(N, with_cavity=True)
    # single robust broadband pass at the cavity centre; harminv over the whole
    # tail. Cavity fundamental (isolated) sits at ~0.2699 with parity odd-x;
    # source at centre + Mirror(X,+1) when x-centred (even N, uncapped).
    fcen, df = 0.272, 0.035
    NU_ISO = 0.26986                      # capped-cavity fundamental (anchor)
    # cavity fundamental is odd about x=cav-centre (Hz node there): excite OFF
    # the node. Mirror(X,+1) (odd-x sector) is valid whenever the cavity is
    # x-centred (even N); MEEP mirrors the off-centre source automatically.
    syms = [mp.Mirror(mp.X, phase=+1)] if (N % 2 == 0) else []
    T = TMAX if TMAX > 0 else 4000.0
    sources = [mp.Source(mp.GaussianSource(fcen, fwidth=df), component=mp.Hz,
                         center=mp.Vector3(x_cav + 0.35, y_cav + 0.15))]
    sim = mp.Simulation(cell_size=mp.Vector3(SX, cell_y), geometry=holes,
                        default_material=mp.Medium(epsilon=EPS_EFF),
                        boundary_layers=[mp.PML(DPML)], sources=sources,
                        symmetries=syms, resolution=RES, eps_averaging=True)
    mon = mp.Vector3(x_cav + 0.30, y_cav + 0.20)
    h2 = mp.Harminv(mp.Hz, mon, fcen, df)
    sim.run(mp.after_sources(h2), until_after_sources=T)
    tau = 1.0
    allmodes = [{"nu": float(m.freq), "GHz": to_ghz(float(m.freq)),
                 "Q": float(m.Q), "amp": float(abs(m.amp)),
                 "err": float(abs(m.err))}
                for m in sorted(h2.modes, key=lambda m: m.freq)
                if GAP[0] <= m.freq <= GAP[1] and abs(m.amp) > 0]
    if not allmodes:
        print(f"N={N}: NO cavity mode found in gap")
        sys.exit(2)
    # cavity mode = the significant-amplitude mode nearest the anchor freq
    amax = max(mm["amp"] for mm in allmodes)
    sig = [mm for mm in allmodes if mm["amp"] >= 0.05 * amax]
    m0 = min(sig, key=lambda mm: abs(mm["nu"] - NU_ISO))
    Qr = m0["Q"]
    Qkey = "Q_i" if CAP_WG else "Q_loaded"
    out = {"task": "M2 2D S21 ladder: cavity harminv probe"
                   + (" (waveguide CAPPED -> Q_i)" if CAP_WG else " (coupled -> Q_loaded)"),
           "rng": "deterministic - no RNG", "meep_version": mp.__version__,
           "capped": CAP_WG,
           "params": {"N_rows": N, "res": RES, "cell": [SX, cell_y],
                      "y_wg": y_wg, "y_cav": y_cav, "x_cav": x_cav,
                      "s_star": S_STAR, "eps_eff": EPS_EFF,
                      "a_final_um": A_FINAL_UM, "until_after_sources": T,
                      "T_over_tau": T / tau},
           "result": {"nu0": m0["nu"], "f0_GHz": m0["GHz"],
                      Qkey: m0["Q"], "cav_mode_amp": m0["amp"],
                      "cav_mode_err": m0["err"], "anchor_nu_iso": NU_ISO,
                      "used_mirror_x": bool(N % 2 == 0 and not CAP_WG)},
           "all_gap_modes": allmodes}
    os.makedirs(f"{ROOT}/out/m2", exist_ok=True)
    suffix = "_cap" if CAP_WG else ""
    with open(f"{ROOT}/out/m2/s21_2d_probe_N{N}{suffix}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"N={N}{' CAP' if CAP_WG else ''}: nu0={m0['nu']:.6f} "
          f"({m0['GHz']:.3f} GHz) {Qkey}={m0['Q']:.1f} amp={m0['amp']:.1f} "
          f"(gap modes: "
          + ", ".join(f"{mm['nu']:.4f}/Q{mm['Q']:.0f}/a{mm['amp']:.1f}"
                      for mm in allmodes) + ")")
    print(f"saved out/m2/s21_2d_probe_N{N}.json")

elif SUB == "s21":
    N = N_ROWS
    fcen, df, nfreq = 0.28, 0.09, 400
    tmax = TMAX if TMAX > 0 else 40000.0

    def do_run(with_cavity):
        holes, cell_y, y_wg, y_cav, x_cav = build(N, with_cavity)
        src = mp.EigenModeSource(
            mp.GaussianSource(fcen, fwidth=df),
            center=mp.Vector3(X_SRC, y_wg), size=mp.Vector3(0, 3.0),
            direction=mp.X, eig_band=EIG_BAND, eig_match_freq=True,
            eig_parity=mp.NO_PARITY)
        sim = mp.Simulation(cell_size=mp.Vector3(SX, cell_y), geometry=holes,
                            default_material=mp.Medium(epsilon=EPS_EFF),
                            boundary_layers=[mp.PML(DPML)], sources=[src],
                            resolution=RES, eps_averaging=True)
        fr = mp.FluxRegion(center=mp.Vector3(X_MON, y_wg),
                           size=mp.Vector3(0, 3.0))
        rr = mp.FluxRegion(center=mp.Vector3(X_SRC + 2.0, y_wg),
                           size=mp.Vector3(0, 3.0))
        tr = sim.add_flux(fcen, df, nfreq, fr)
        rf = sim.add_flux(fcen, df, nfreq, rr)
        run_until(sim, mp.Vector3(X_MON, y_wg), tmax)
        freqs = np.array(mp.get_flux_freqs(tr))
        flux = np.array(mp.get_fluxes(tr))
        # mode-resolved forward amplitude of the coupling band at the far
        # monitor (transmission) and backward amplitude at the near monitor
        # (reflection). Total flux hides the notch if the cavity converts
        # band2<->band1; the mode-resolved coeffs isolate band EIG_BAND.
        rt = sim.get_eigenmode_coefficients(tr, [EIG_BAND],
                                            eig_parity=mp.NO_PARITY)
        rr_ = sim.get_eigenmode_coefficients(rf, [EIG_BAND],
                                             eig_parity=mp.NO_PARITY)
        pfwd = np.abs(rt.alpha[0, :, 0]) ** 2          # band fwd at far mon
        pinc = np.abs(rr_.alpha[0, :, 0]) ** 2         # band fwd at near mon
        pref = np.abs(rr_.alpha[0, :, 1]) ** 2         # band bwd at near mon
        return freqs, flux, pfwd, pinc, pref, sim.meep_time()

    f_n, flux_norm, p_n, pinc_n, pref_n, t_n = do_run(False)
    f_c, flux_cav, p_c, pinc_c, pref_c, t_c = do_run(True)
    with np.errstate(divide="ignore", invalid="ignore"):
        s21_flux = np.where(flux_norm > 0, flux_cav / flux_norm, np.nan)
        s21 = np.where(p_n > 0, p_c / p_n, np.nan)   # mode-resolved T (band)
        s11 = np.where(pinc_n > 0, pref_c / pinc_n, np.nan)  # mode-resolved R
    cell_y = layout(N)[0]
    out = {"task": "M2 2D S21 ladder (mode-resolved band-EIG transmission)",
           "rng": "deterministic - no RNG", "meep_version": mp.__version__,
           "params": {"N_rows": N, "res": RES, "cell": [SX, cell_y],
                      "s_star": S_STAR, "eps_eff": EPS_EFF, "eig_band": EIG_BAND,
                      "a_final_um": A_FINAL_UM, "fcen": fcen, "df": df,
                      "nfreq": nfreq, "tmax": tmax, "x_src": X_SRC,
                      "x_mon": X_MON, "t_norm": t_n, "t_cav": t_c},
           "freqs": [float(x) for x in f_c],
           "GHz": [to_ghz(float(x)) for x in f_c],
           "flux_norm": [float(x) for x in flux_norm],
           "flux_cav": [float(x) for x in flux_cav],
           "p_norm": [float(x) for x in p_n],
           "p_cav": [float(x) for x in p_c],
           "pinc_norm": [float(x) for x in pinc_n],
           "pref_cav": [float(x) for x in pref_c],
           "S21_totalflux": [None if np.isnan(v) else float(v) for v in s21_flux],
           "S11": [None if np.isnan(v) else float(v) for v in s11],
           "S21": [None if np.isnan(v) else float(v) for v in s21]}
    with open(f"{ROOT}/out/m2/s21_2d_N{N}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    imin = int(np.nanargmin(s21))
    print(f"N={N}: S21 min={np.nanmin(s21):.4f} at "
          f"nu={f_c[imin]:.5f} ({to_ghz(float(f_c[imin])):.3f} GHz)  "
          f"t_norm={t_n:.0f} t_cav={t_c:.0f}")
    print(f"saved out/m2/s21_2d_N{N}.json")

elif SUB == "diag":
    # drive the guide broadband; measure reflection (2-run subtraction),
    # transmission, and energy stored in a box around the cavity vs freq.
    N = N_ROWS
    fcen, df, nfreq = 0.28, 0.09, 300
    tmax = TMAX if TMAX > 0 else 8000.0
    _, _, _, y_cav0, x_cav0, _ = layout(N)  # placeholder (layout returns 6)

    def do(with_cavity):
        holes, cell_y, y_wg, y_cav, x_cav = build(N, with_cavity)
        src = mp.EigenModeSource(mp.GaussianSource(fcen, fwidth=df),
                                 center=mp.Vector3(X_SRC, y_wg),
                                 size=mp.Vector3(0, 3.0), direction=mp.X,
                                 eig_band=EIG_BAND, eig_match_freq=True,
                                 eig_parity=mp.NO_PARITY)
        sim = mp.Simulation(cell_size=mp.Vector3(SX, cell_y), geometry=holes,
                            default_material=mp.Medium(epsilon=EPS_EFF),
                            boundary_layers=[mp.PML(DPML)], sources=[src],
                            resolution=RES, eps_averaging=True)
        # reflection monitor between source and cavity
        refl = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
            center=mp.Vector3(X_SRC + 1.5, y_wg), size=mp.Vector3(0, 3.0)))
        tran = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
            center=mp.Vector3(X_MON, y_wg), size=mp.Vector3(0, 3.0)))
        return sim, refl, tran, y_wg

    sim, refl0, tran0, y_wg = do(False)
    run_until(sim, mp.Vector3(X_MON, y_wg), tmax)
    freqs = np.array(mp.get_flux_freqs(tran0))
    inc = np.array(mp.get_fluxes(tran0))
    refl0_data = sim.get_flux_data(refl0)

    sim, refl, tran, y_wg = do(True)
    sim.load_minus_flux_data(refl, refl0_data)
    run_until(sim, mp.Vector3(X_MON, y_wg), tmax)
    T = np.array(mp.get_fluxes(tran)) / inc
    R = -np.array(mp.get_fluxes(refl)) / inc
    out = {"task": "M2 2D S21 diag: R,T vs freq (reflection-subtracted)",
           "meep_version": mp.__version__, "params": {"N_rows": N, "res": RES,
           "fcen": fcen, "df": df, "nfreq": nfreq, "tmax": tmax},
           "freqs": [float(x) for x in freqs],
           "T": [float(x) for x in T], "R": [float(x) for x in R],
           "inc": [float(x) for x in inc]}
    with open(f"{ROOT}/out/m2/s21_2d_diag_N{N}.json", "w") as fh:
        json.dump(out, fh, indent=1)
    gap = (freqs > 0.258) & (freqs < 0.325)
    it = int(np.where(gap)[0][np.argmin(T[gap])])
    ir = int(np.where(gap)[0][np.argmax(R[gap])])
    print(f"N={N}: T-min in gap nu={freqs[it]:.5f} T={T[it]:.4f} R={R[it]:.4f} "
          f"R+T={R[it]+T[it]:.4f}")
    print(f"       R-max in gap nu={freqs[ir]:.5f} R={R[ir]:.4f} T={T[ir]:.4f}")
    print(f"saved out/m2/s21_2d_diag_N{N}.json")

elif SUB == "w1band":
    # W1 guided-mode dispersion via MPB. Rectangular supercell width 1 (guide
    # period along x=GammaK), height Msup*sqrt(3)/2 (Msup rows along GammaM),
    # centre row removed. TE bands vs kx in [0,0.5]. Guided modes = bands lying
    # inside the 2D gap (0.2576-0.3257).
    from meep import mpb
    Msup = int(flag("--msup", "11"))     # odd => a centre row to remove
    kint = int(flag("--kint", "12"))
    NB = int(flag("--nbands", "12"))
    geom = []
    jmax = Msup // 2
    for j in range(-jmax, jmax + 1):
        if j == 0:
            continue                      # remove centre row -> W1
        off = 0.5 * (j % 2)
        geom.append(mp.Cylinder(radius=R_OVER_A, material=mp.air,
                                center=mp.Vector3(off, j * (SQRT3 / 2.0))))
        geom.append(mp.Cylinder(radius=R_OVER_A, material=mp.air,
                                center=mp.Vector3(off - 1.0, j * (SQRT3 / 2.0))))
    lat = mp.Lattice(size=mp.Vector3(1, Msup * SQRT3 / 2.0))
    kpts = mp.interpolate(kint, [mp.Vector3(0, 0), mp.Vector3(0.5, 0)])
    ms = mpb.ModeSolver(geometry_lattice=lat, geometry=geom,
                        default_material=mp.Medium(epsilon=EPS_EFF),
                        k_points=kpts, resolution=RES, num_bands=NB)
    ms.run_te()
    allf = [[float(f) for f in row] for row in ms.all_freqs]
    kxs = [float(k.x) for k in kpts]
    out = {"task": "M2 2D W1 guided-band dispersion (MPB)",
           "meep_version": mp.__version__, "Msup": Msup, "res": RES,
           "gap": list(GAP), "kx": kxs, "te_bands": allf}
    with open(f"{ROOT}/out/m2/s21_2d_w1band.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"W1 bands (Msup={Msup}); gap=[{GAP[0]:.4f},{GAP[1]:.4f}]")
    print("kx    " + "  ".join(f"b{b+1}" for b in range(NB)))
    for i, k in enumerate(kxs):
        row = allf[i]
        marks = "".join("*" if GAP[0] <= f <= GAP[1] else " " for f in row)
        print(f"{k:.3f} " + " ".join(f"{f:.4f}" for f in row) + "  gap:" + marks)
    print(f"saved out/m2/s21_2d_w1band.json")

elif SUB == "field":
    # drive the guide (band EIG_BAND) broadband; DFT Hz over a box around the
    # cavity and integrate |Hz|^2 per freq -> cavity response spectrum. Reveals
    # whether/at-what-freq the guided mode excites the cavity.
    N = N_ROWS
    fcen, df, nfreq = 0.28, 0.09, 60
    tmax = TMAX if TMAX > 0 else 6000.0
    holes, cell_y, y_wg, y_cav, x_cav = build(N, with_cavity=True)
    src = mp.EigenModeSource(mp.GaussianSource(fcen, fwidth=df),
                             center=mp.Vector3(X_SRC, y_wg),
                             size=mp.Vector3(0, 3.0), direction=mp.X,
                             eig_band=EIG_BAND, eig_match_freq=True,
                             eig_parity=mp.NO_PARITY)
    sim = mp.Simulation(cell_size=mp.Vector3(SX, cell_y), geometry=holes,
                        default_material=mp.Medium(epsilon=EPS_EFF),
                        boundary_layers=[mp.PML(DPML)], sources=[src],
                        resolution=RES, eps_averaging=True)
    dft = sim.add_dft_fields([mp.Hz], fcen, df, nfreq,
                             center=mp.Vector3(x_cav, y_cav),
                             size=mp.Vector3(3.0, 3.0))
    tran = sim.add_flux(fcen, df, nfreq, mp.FluxRegion(
        center=mp.Vector3(X_MON, y_wg), size=mp.Vector3(0, 3.0)))
    run_until(sim, mp.Vector3(X_MON, y_wg), tmax)
    freqs = np.array(mp.get_flux_freqs(tran))
    inc = np.array(mp.get_fluxes(tran))
    Ucav = []
    for k in range(nfreq):
        H = np.asarray(sim.get_dft_array(dft, mp.Hz, k))
        Ucav.append(float(np.sum(np.abs(H) ** 2)))
    Ucav = np.array(Ucav)
    # normalise cavity energy by incident power to get coupling-response shape
    resp = np.where(inc > 1e-6, Ucav / inc, np.nan)
    ip = int(np.nanargmax(resp))
    print(f"N={N} band={EIG_BAND}: cavity response peak at nu={freqs[ip]:.5f} "
          f"({to_ghz(float(freqs[ip])):.3f} GHz) resp={resp[ip]:.3e} "
          f"(Ucav={Ucav[ip]:.3e} inc={inc[ip]:.3e})")
    for k in range(nfreq):
        if k % 4 == 0:
            print(f"  nu={freqs[k]:.5f} Ucav={Ucav[k]:.3e} inc={inc[k]:.3e} "
                  f"resp={resp[k]:.3e}")
    out = {"task": "M2 2D S21 field diag: cavity-box energy vs freq (driven)",
           "N_rows": N, "eig_band": EIG_BAND,
           "freqs": [float(x) for x in freqs],
           "Ucav": [float(x) for x in Ucav], "inc": [float(x) for x in inc]}
    with open(f"{ROOT}/out/m2/s21_2d_field_N{N}_b{EIG_BAND}.json", "w") as fh:
        json.dump(out, fh, indent=1)

elif SUB == "analyze":
    # meep-free: read L3 + ladder artifacts, decompose Q_w, fit exponential,
    # evaluate the standing-wave TCMT identities, write summary + figure.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def rd(p):
        return json.load(open(p))

    o2 = f"{ROOT}/out/m2"
    # ---- L3 Q(s) --------------------------------------------------------
    l3 = {}
    for tag, s in [("000", 0.0), ("010", 0.10), ("015", 0.15), ("020", 0.20)]:
        d = rd(f"{o2}/l3_2d_s{tag}.json")["result"]
        l3[s] = {"nu0": d["nu0"], "GHz": d["f0_GHz"], "Q": d["Q_2d"],
                 "parity_x": d["parity_x"], "parity_y": d["parity_y"]}
    s_star = max(l3, key=lambda s: l3[s]["Q"])

    # ---- ladder Q's -----------------------------------------------------
    Ns = [2, 3, 4, 5]
    lad = {}
    for N in Ns:
        c = rd(f"{o2}/s21_2d_probe_N{N}.json")["result"]
        cap = rd(f"{o2}/s21_2d_probe_N{N}_cap.json")["result"]
        Ql, Qi = c["Q_loaded"], cap["Q_i"]
        inv = 1.0 / Ql - 1.0 / Qi
        Qw = (1.0 / inv) if inv > 0 else float("inf")
        Tmin = (Qw / (Qi + Qw)) ** 2
        A = 2.0 * Qi * Qw / (Qi + Qw) ** 2
        lad[N] = {"nu0_coupled": c["nu0"], "GHz": c["f0_GHz"],
                  "Q_loaded": Ql, "Q_i": Qi, "Q_w": Qw,
                  "TCMT_Tmin_onres": Tmin, "TCMT_absorption_A": A,
                  "overcoupled": bool(Qw < Qi)}

    # ---- exponential fit ln(Q_w)=a*N+b (tunnelling regime N=3,4,5) ------
    Nfit = [3, 4, 5]
    x = np.array(Nfit, float)
    y = np.array([math.log(lad[N]["Q_w"]) for N in Nfit])
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    summary = {
        "task": "M2 2D exploratory L3 + side-coupled W1 S21 ladder (summary)",
        "rng": "deterministic - no RNG", "meep_version": rd(
            f"{o2}/l3_2d_s000.json")["meep_version"],
        "eps_eff": EPS_EFF, "a_final_um": A_FINAL_UM, "r_over_a": R_OVER_A,
        "gap_a_over_lambda": list(GAP),
        "L3": {"Q_of_s": {str(s): l3[s] for s in l3}, "s_star": s_star,
               "s_star_note": "2D-model optimum (max Q_2d); the 3D radiation-Q "
               "optimum ~0.15a is an AT-2 (out-of-scope) result"},
        "ladder": {str(N): lad[N] for N in Ns},
        "Qw_exponential_fit": {"form": "ln(Q_w) = alpha*N + beta",
                               "N_used": Nfit, "alpha": float(a),
                               "beta": float(b), "R2": r2,
                               "note": "N=2 excluded: its coupled mode is "
                               "pushed to nu0=0.2743 (out of guided-band "
                               "overlap), so it is anomalously weakly coupled"},
        "coupling_band": "L3 fundamental (Hz odd-x, even-y) couples to the "
        "2nd (even) W1 guided band; the 1st band is parity-forbidden",
        "TCMT_identity": "on-res T=(gamma_i/(gamma_i+gamma_w))^2=(Q_w/(Q_i+Q_w))^2; "
        "A=2 Q_i Q_w/(Q_i+Q_w)^2; verified algebraically per N in 'ladder'",
        "twoD_caveat": "The 2D W1 is MULTIMODE across the gap; the L3 "
        "mode-converts band2<->band1 and radiates, so a clean single-mode "
        "standing-wave notch/mirror does NOT form in 2D (mode-resolved "
        "band-2 reflection <0.07 on resonance). Q_w,Q_i are from harminv "
        "ring-down (coupled vs waveguide-capped). The clean TCMT S21 notch "
        "and T_min<0.05 mirror require the single-mode 3D slab W1 (AT-3, "
        "out of M2-2D scope).",
    }
    with open(f"{o2}/s21_2d_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)

    # ---- figure: mode-resolved S21 + TCMT(from harminv Q's) per N -------
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=False)
    for ax, N in zip(axes.ravel(), Ns):
        d = rd(f"{o2}/s21_2d_N{N}.json")
        nu = np.array(d["freqs"])
        T = np.array([np.nan if v is None else v for v in d["S21"]])
        pn = np.array(d["p_norm"])
        band = (nu > 0.266) & (nu < 0.315) & (pn > 0.15 * np.nanmax(pn))
        ax.plot(nu[band], T[band], ".", ms=3, color="#1f77b4",
                label="MEEP band-2 |S21|$^2$ (2D)")
        # TCMT from harminv Q's
        nu0 = lad[N]["nu0_coupled"]
        Qi, Qw = lad[N]["Q_i"], lad[N]["Q_w"]
        gi, gw = math.pi * nu0 / Qi, math.pi * nu0 / Qw
        xx = np.linspace(0.266, 0.315, 800)
        dom = 2 * math.pi * (xx - nu0)
        Tt = (dom ** 2 + gi ** 2) / (dom ** 2 + (gi + gw) ** 2)
        ax.plot(xx, Tt, "-", color="#d62728", lw=1.6,
                label="standing-wave TCMT (harminv $Q$)")
        ax.axvline(nu0, color="gray", ls=":", lw=0.8)
        ax.set_title(f"N_rows={N}: $Q_i$={Qi:.0f}, $Q_w$={Qw:.0f}, "
                     f"$\\nu_0$={nu0:.4f}")
        ax.set_xlabel(r"$\nu = a/\lambda$")
        ax.set_ylabel("transmission")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("M2 2D side-coupled L3+W1 ladder: mode-resolved S21 vs "
                 "standing-wave TCMT\n(2D W1 is multimode -> cavity mode-"
                 "converts, notch not realized in 2D; $Q$'s from harminv)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{ROOT}/figs/fig_m2_2d_s21_ladder.png", dpi=140)
    print("=== L3 Q(s) ===")
    for s in sorted(l3):
        print(f"  s={s:.2f}a  nu0={l3[s]['nu0']:.5f} ({l3[s]['GHz']:.2f} GHz) "
              f"Q_2d={l3[s]['Q']:.0f}  parity(x,y)=({l3[s]['parity_x']:+.2f},"
              f"{l3[s]['parity_y']:+.2f})")
    print(f"  s* = {s_star}a")
    print("=== ladder (harminv) ===")
    for N in Ns:
        L = lad[N]
        print(f"  N={N}: nu0={L['nu0_coupled']:.4f} Q_loaded={L['Q_loaded']:.0f} "
              f"Q_i={L['Q_i']:.0f} Q_w={L['Q_w']:.0f} "
              f"TCMT_Tmin={L['TCMT_Tmin_onres']:.4f} A={L['TCMT_absorption_A']:.4f}")
    print(f"=== ln(Q_w)={a:.4f}*N+{b:.4f} (N=3,4,5; R2={r2:.4f}) ===")
    print(f"saved {o2}/s21_2d_summary.json and figs/fig_m2_2d_s21_ladder.png")

else:
    print(f"unknown subcommand {SUB!r}")
    sys.exit(1)
