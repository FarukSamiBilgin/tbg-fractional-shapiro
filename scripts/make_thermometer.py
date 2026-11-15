"""T2 - build the inverse thermometer from the T1 calibration.

Input : out/thermometry_calibration*.json  (scripts/run_thermometry_calibration.py,
        one file per calibrated filling)
Output: out/thermometer.json, out/thermometer.npz, figs/fig_thermometer.png

What this does
--------------
T1 measured the forward map Gamma -> W_net (net soft half-step width, control
subtracted). A thermometer needs the inverse, with error bars:

    measured R = W_net / W_net_det   ->   Gamma   ->   T_e = Gamma * E_J / k_B

Steps:

  1. Monotone interpolation of R(Gamma) with a PCHIP spline in log(Gamma).
     PCHIP is shape-preserving, so it cannot introduce the overshoot a cubic
     spline would put into a saturating curve. Monotonicity is asserted, not
     assumed: if T1 reported violations beyond 1 combined SEM the build refuses
     to produce an inverse over the offending interval.

  2. Numerical inversion Gamma(R) by bisection on the spline.

  3. Error propagation. Two independent contributions:
       sigma_cal   - the calibration's own SEM on W_net at that Gamma
       sigma_meas  - the experiment's width-measurement precision (a parameter;
                     reported for 2 % and 5 % of W_net_det by default)
     combined in quadrature and divided by the local slope |dR/dGamma|:
       sigma_Gamma = sqrt(sigma_cal^2 + sigma_meas^2) / |dR/dGamma|
       sigma_T     = sigma_Gamma * E_J/k_B

  4. Per-filling temperature axes. E_J/k_B = hbar*I_c/(2 e k_B) with I_c(nu)
     read from junction.json:fingerprint (w_half_nA / w_half). Each filling maps
     the SAME reduced calibration onto a different absolute T_e window -- this is
     the gate-tunable part of the instrument.

  5. The free upper bound. Observing R at or above a given level immediately
     bounds Gamma from above, hence T_e. Reported explicitly because it needs no
     inversion, no fitting, and no absolute width calibration -- only the
     ability to see that the step retains a stated fraction of its width.

SCOPE LIMIT (stated in the output JSON, not only here)
------------------------------------------------------
The reduced-unit dynamics depend on Omega, and Omega(nu) runs 5.0 -> 13.9 across
the dome, so a filling is only genuinely calibrated when a T1 run exists at its
own Omega. This script picks up every out/thermometry_calibration*.json it finds
and matches each filling to its own calibration when one is present; fillings
without one fall back to the nu = -2.4 curve and are flagged `calibrated: false`
with the file they actually used named in `calibration_source`. Convert a row by
running  scripts/run_thermometry_calibration.py --nu <nu>.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "out"
FIGS = ROOT / "figs"

NU_REF = -2.4          # the filling T1 calibrates by default

# CODATA 2018
HBAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19
K_B = 1.380649e-23


def ej_over_kb(ic_nA):
    """E_J/k_B in kelvin for a critical current in nA:  hbar*I_c/(2 e k_B)."""
    return HBAR * (ic_nA * 1e-9) / (2.0 * E_CHARGE * K_B)


def pchip(x, y):
    """Shape-preserving piecewise cubic Hermite (Fritsch-Carlson slopes).

    Implemented locally so this script has no scipy dependency and so the slope
    rule is auditable: at any interior node where the data change direction the
    derivative is set to zero, which is what forbids overshoot in a saturating
    calibration curve.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n)
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            d[i] = 0.0
        else:
            w1, w2 = 2 * h[i] + h[i - 1], h[i] + 2 * h[i - 1]
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    # one-sided ends (Fritsch-Carlson endpoint rule, clipped for monotonicity)
    d[0] = ((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if d[0] * delta[0] <= 0:
        d[0] = 0.0
    elif abs(d[0]) > 3 * abs(delta[0]):
        d[0] = 3 * delta[0]
    d[-1] = ((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if d[-1] * delta[-1] <= 0:
        d[-1] = 0.0
    elif abs(d[-1]) > 3 * abs(delta[-1]):
        d[-1] = 3 * delta[-1]

    def f(xq, deriv=0):
        xq = np.atleast_1d(np.asarray(xq, float))
        k = np.clip(np.searchsorted(x, xq) - 1, 0, n - 2)
        t = (xq - x[k]) / h[k]
        h00 = 2 * t**3 - 3 * t**2 + 1
        h10 = t**3 - 2 * t**2 + t
        h01 = -2 * t**3 + 3 * t**2
        h11 = t**3 - t**2
        if deriv == 0:
            return y[k] * h00 + h[k] * d[k] * h10 + y[k + 1] * h01 + h[k] * d[k + 1] * h11
        dh00 = 6 * t**2 - 6 * t
        dh10 = 3 * t**2 - 4 * t + 1
        dh01 = -6 * t**2 + 6 * t
        dh11 = 3 * t**2 - 2 * t
        return (y[k] * dh00 / h[k] + d[k] * dh10
                + y[k + 1] * dh01 / h[k] + d[k + 1] * dh11)

    return f


def _at(f, u):
    """Scalar evaluation of a pchip callable (which always returns an array;
    numpy >= 2 refuses float() on a size-1 array)."""
    return float(np.asarray(f(u)).reshape(-1)[0])


def invert(f, target, lo, hi, iters=80):
    """Bisection for f(u) = target on a monotone f over [lo, hi]."""
    flo, fhi = _at(f, lo), _at(f, hi)
    if not (min(flo, fhi) <= target <= max(flo, fhi)):
        return None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if (_at(f, mid) - target) * (flo - target) > 0:
            lo, flo = mid, _at(f, mid)
        else:
            hi = mid
    return 0.5 * (lo + hi)


SLOPE_BANDWIDTH = 0.3      # Gaussian kernel width in ln(Gamma) for local_slope()


def local_slope(lg_query, lg_nodes, y_nodes, bw=SLOPE_BANDWIDTH):
    """Smooth dy/d(log Gamma) by weighted local linear regression.

    PCHIP is the right interpolant for the VALUE -- it is shape-preserving, so
    the inverse stays single-valued -- but its derivative is only C0 across the
    knots, and sigma_T is proportional to 1/|slope|. Differentiating the spline
    therefore stamps the knot spacing onto the reported resolution as ~30 %
    oscillations that are an artifact of the interpolant, not a property of the
    junction.

    A parametric fit was tried first and rejected: a three-parameter generalised
    logistic A/(1+(G/G0)^p) leaves chi2/dof = 4.9 with systematic 3-sigma
    residuals, so forcing that shape would bias the inversion rather than smooth
    it. Local regression assumes no global shape instead.

    Cost of the smoothing, measured on an analytic logistic sampled at this
    calibration's 14 nodes: at bw = 0.3 the recovered slope is biased by at most
    0.026 against a peak slope of 0.405, i.e. 6.4 %, and sigma_T is proportional
    to 1/|slope| so it inherits that 6.4 %. That is the price for suppressing a
    ~30 % interpolation artifact, and it is smooth and bounded where the
    artifact was neither. A local QUADRATIC fit was also tried, on the
    expectation that it would cut the derivative bias to O(h^2); at these
    bandwidths it measured the same bias as the linear fit and was dropped in
    favour of the simpler estimator.
    """
    lg_query = np.atleast_1d(np.asarray(lg_query, float))
    lg_nodes = np.asarray(lg_nodes, float)
    y_nodes = np.asarray(y_nodes, float)
    out = np.empty_like(lg_query)
    for i, x in enumerate(lg_query):
        w = np.exp(-0.5 * ((lg_nodes - x) / bw) ** 2)
        X = np.vstack([np.ones_like(lg_nodes), lg_nodes - x]).T
        XtW = X.T * w
        out[i] = np.linalg.solve(XtW @ X, XtW @ y_nodes)[1]
    return out


def load_calibrations():
    """Map nu -> (calibration dict, filename) over out/thermometry_calibration*.json.

    Smoke runs (--n-real < 20) write the same filenames as production runs, so
    they are skipped here rather than silently producing a thermometer whose
    error bars are ten times too wide.
    """
    found = {}
    for path in sorted(OUT.glob("thermometry_calibration*.json")):
        d = json.loads(path.read_text())
        if d.get("numerics", {}).get("n_real_signal", 0) < 20:
            print(f"  skipping {path.name}: smoke run "
                  f"(n_real={d['numerics']['n_real_signal']})")
            continue
        found[round(float(d["operating_point"]["nu"]), 3)] = (d, path.name)
    return found


def build_inverse(cal):
    """Monotone inverse map from one T1 calibration.

    Trims the saturated tail: past the first Gamma that is not below its
    predecessor by at least its own SEM, the estimator carries no temperature
    information, and including those points would flatten the spline and inflate
    the inferred resolution.
    """
    g = np.array([e["Gamma"] for e in cal["per_gamma"]], float)
    w = np.array([e["soft_net_mean"] for e in cal["per_gamma"]], float)
    sem = np.array([e["soft_net_sem"] for e in cal["per_gamma"]], float)
    w_det = cal["deterministic_reference"]["w_soft_net"]
    R, R_sem = w / w_det, sem / w_det

    keep = [0]
    for i in range(1, len(g)):
        if R[i] < R[keep[-1]] - R_sem[i]:
            keep.append(i)
        else:
            break
    keep = np.array(keep)
    trimmed = [float(x) for x in g[len(keep):]] if len(keep) < len(g) else []
    gk, Rk, Rsk = g[keep], R[keep], R_sem[keep]
    if len(gk) < 3:
        raise SystemExit(f"calibration at nu={cal['operating_point']['nu']} has "
                         f"only {len(gk)} usable points -- cannot invert")
    lg = np.log(gk)
    return {"gk": gk, "Rk": Rk, "Rsk": Rsk, "lg": lg,
            "f_R": pchip(lg, Rk), "f_sem": pchip(lg, Rsk),
            "w_det": w_det, "trimmed": trimmed,
            "violations": cal.get("monotonicity_violations", [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas-precision", type=float, nargs="+", default=[0.02, 0.05],
                    help="width-measurement precision as a fraction of W_net_det")
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    cals = load_calibrations()
    if not cals:
        raise SystemExit("no production calibration found -- run "
                         "scripts/run_thermometry_calibration.py first")
    if NU_REF not in cals:
        raise SystemExit(f"reference calibration (nu={NU_REF}) missing; extension "
                         "fillings fall back to it and cannot be built without it")
    jj = json.loads((ROOT / "junction.json").read_text())
    fp = jj["fingerprint"]

    print("=" * 78)
    print("T2 - inverse thermometer")
    print(f"calibrations found: "
          + ", ".join(f"nu={k} ({v[1]})" for k, v in sorted(cals.items())))
    print("-" * 78)

    # ---- per-filling absolute temperature axes -----------------------------
    nus = fp["nu"]
    ic_nA = [fp["w_half_nA"][i] / fp["w_half"][i] for i in range(len(nus))]
    # the dome is symmetric about nu_opt = -2.4; only this half was independently
    # simulated (junction.json fingerprint.symmetry_note), so only it is reported
    sel = sorted({i for i, nu in enumerate(nus) if nu <= NU_REF},
                 key=lambda i: nus[i], reverse=True)

    fillings = []
    dense_by_nu = []          # per-filling resolution curves for the figure
    inv_by_nu = {}            # each filling's own inverse map
    ref_table = None
    for i in sel:
        nu, ic = round(nus[i], 3), ic_nA[i]
        own = nu in cals
        cal, cal_name = cals[nu] if own else cals[NU_REF]
        inv = build_inverse(cal)
        inv_by_nu[nu] = inv
        if inv["violations"]:
            print(f"WARNING nu={nu}: T1 reported {len(inv['violations'])} "
                  "monotonicity violation(s) beyond 1 combined SEM; the inverse "
                  "is not single-valued there.")

        lg_dense = np.linspace(inv["lg"][0], inv["lg"][-1], 601)
        g_dense = np.exp(lg_dense)
        R_dense = inv["f_R"](lg_dense)
        # value from PCHIP (shape-preserving); slope from a smooth local linear
        # regression, because sigma_T ~ 1/|slope| and the spline derivative is
        # only C0 across knots -- see local_slope().
        slope = local_slope(lg_dense, inv["lg"], inv["Rk"]) / g_dense
        sem_dense = np.abs(inv["f_sem"](lg_dense))
        if nu == NU_REF:
            ref_table = (g_dense, R_dense, slope, sem_dense, inv)

        ejk = ej_over_kb(ic)
        T_dense = g_dense * ejk
        row = {
            "nu": nu,
            "Ic_nA": ic,
            "Omega": fp["Omega"][i],
            "EJ_over_kB_K": ejk,
            "calibrated": bool(own),
            "calibration_source": cal_name,
            "calibration_nu": cal["operating_point"]["nu"],
            "w_net_det": inv["w_det"],
            "gamma_range": [float(inv["gk"][0]), float(inv["gk"][-1])],
            "R_range": [float(inv["Rk"][-1]), float(inv["Rk"][0])],
            "trimmed_saturated_gammas": inv["trimmed"],
            "T_range_mK": [float(T_dense.min() * 1e3), float(T_dense.max() * 1e3)],
        }
        for pr in args.meas_precision:
            sig_R = np.hypot(sem_dense, pr)
            sig_g = sig_R / np.abs(slope)
            sig_T = sig_g * ejk
            # usable = where the temperature resolution beats 20 % of T itself
            ok = sig_T < 0.2 * T_dense
            row[f"sigma_T_mK_at_{pr:.0%}"] = {
                "best_mK": float(np.min(sig_T) * 1e3),
                "at_T_mK": float(T_dense[np.argmin(sig_T)] * 1e3),
                "usable_T_window_mK": ([float(T_dense[ok].min() * 1e3),
                                        float(T_dense[ok].max() * 1e3)]
                                       if ok.any() else None),
            }
        p0 = args.meas_precision[0]
        dense_by_nu.append({
            "nu": nu, "Ic_nA": ic, "calibrated": bool(own),
            "T_mK": T_dense * 1e3,
            "sigma_T_mK": (np.hypot(sem_dense, p0) / np.abs(slope)) * ejk * 1e3,
        })
        fillings.append(row)
        tag = "" if own else f"  [EXTRAPOLATED from nu={NU_REF}: Omega differs]"
        print(f"nu={nu:<5} Ic={ic:6.1f} nA  E_J/k_B={ejk:5.3f} K  "
              f"T window {row['T_range_mK'][0]:6.1f}..{row['T_range_mK'][1]:6.1f} mK"
              f"{tag}")
        for pr in args.meas_precision:
            d = row[f"sigma_T_mK_at_{pr:.0%}"]
            win = d["usable_T_window_mK"]
            print(f"      meas {pr:.0%}: best sigma_T = {d['best_mK']:.1f} mK "
                  f"(at {d['at_T_mK']:.0f} mK); usable "
                  + (f"{win[0]:.0f}..{win[1]:.0f} mK" if win else "none"))

    g_dense, R_dense, slope, sem_dense, inv_ref = ref_table
    gk, Rk, Rsk = inv_ref["gk"], inv_ref["Rk"], inv_ref["Rsk"]
    lg, f_R, w_det = inv_ref["lg"], inv_ref["f_R"], inv_ref["w_det"]
    trimmed = inv_ref["trimmed"]

    # ---- the free upper bound ----------------------------------------------
    print("-" * 78)
    # Each filling is bounded with ITS OWN calibration when it has one -- the
    # reference curve would give the wrong Gamma at a different Omega.
    bounds = []
    for level in (0.75, 0.50, 0.25):
        entry = {"R_level": level, "per_filling": {}}
        parts = []
        for row in fillings:
            nu = row["nu"]
            iv = inv_by_nu[nu]
            lg_star = invert(iv["f_R"], level, iv["lg"][0], iv["lg"][-1])
            if lg_star is None:
                entry["per_filling"][str(nu)] = {"Gamma_max": None}
                continue
            gs = float(np.exp(lg_star))
            t_max = gs * row["EJ_over_kB_K"] * 1e3
            entry["per_filling"][str(nu)] = {
                "Gamma_max": gs, "EJ_over_kBT_min": 1.0 / gs,
                "T_e_max_mK": t_max, "calibrated": row["calibrated"]}
            parts.append(f"nu={nu}: <{t_max:.0f} mK")
        ref = entry["per_filling"].get(str(NU_REF), {})
        entry["Gamma_max"] = ref.get("Gamma_max")
        entry["EJ_over_kBT_min"] = ref.get("EJ_over_kBT_min")
        bounds.append(entry)
        gm = ref.get("Gamma_max")
        head = (f"observe R >= {level:.2f}  =>  Gamma <= {gm:.4f} "
                f"(E_J/k_BT >= {1/gm:.1f})" if gm else
                f"observe R >= {level:.2f}  =>  outside the calibrated range")
        print(f"{head}  =>  " + ", ".join(parts))

    result = {
        "description": "T2 inverse thermometer: measured net soft half-step "
                       "width ratio R = W_net/W_net_det -> Gamma -> T_e.",
        "source_calibrations": "out/thermometry_calibration*.json",
        "scope_limit": ("A filling is only calibrated if a T1 run exists at its "
                        "own Omega. Rows with calibrated=false apply the "
                        "nu = -2.4 (Omega = 5) curve at a filling whose Omega "
                        "differs (Omega(nu) runs 5.0 -> 13.9); that is an "
                        "explicit extrapolation, not a measured calibration. "
                        "Run scripts/run_thermometry_calibration.py --nu <nu> to "
                        "convert a row to calibrated=true."),
        "estimator": ("W_soft = Int exp(-((v-v_target)/sigma_k)^2/2) d i_dc, "
                      "control-subtracted against a sinusoidal CPR at the same "
                      "Gamma; R = W_net / W_net_det."),
        "w_net_det_reference": w_det,
        "interpolant": "PCHIP (Fritsch-Carlson) in log(Gamma), shape-preserving",
        "slope_estimator": (f"weighted local linear regression in ln(Gamma), "
                            f"bandwidth {SLOPE_BANDWIDTH}; used only for "
                            "error propagation (sigma_T ~ 1/|dR/dGamma|). The "
                            "PCHIP derivative is C0 across knots and would print "
                            "the knot spacing onto the reported resolution."),
        "calibrations_used": {str(k): v[1] for k, v in sorted(cals.items())},
        "reference_calibration_points_used": int(len(gk)),
        "reference_trimmed_saturated_gammas": trimmed,
        "reference_gamma_range": [float(gk[0]), float(gk[-1])],
        "reference_R_range": [float(Rk[-1]), float(Rk[0])],
        "measurement_precisions": list(args.meas_precision),
        "fillings": fillings,
        "free_upper_bounds": bounds,
        "reference_inverse_table": {
            "note": f"nu = {NU_REF} curve; other fillings with their own "
                    "calibration use theirs, see fillings[].calibration_source",
            "Gamma": g_dense.tolist(),
            "R": R_dense.tolist(),
            "dR_dGamma": slope.tolist(),
            "sigma_R_calibration": sem_dense.tolist(),
        },
    }
    (OUT / "thermometer.json").write_text(json.dumps(result, indent=2,
                                                     allow_nan=False))
    np.savez(OUT / "thermometer.npz", gamma=g_dense, R=R_dense, slope=slope,
             sigma_R_cal=sem_dense, gamma_cal=gk, R_cal=Rk, R_sem_cal=Rsk,
             w_net_det=w_det,
             nu=np.array([f["nu"] for f in fillings]),
             Ic_nA=np.array([f["Ic_nA"] for f in fillings]),
             EJ_over_kB_K=np.array([f["EJ_over_kB_K"] for f in fillings]))

    if not args.no_fig:
        make_figure(g_dense, R_dense, gk, Rk, Rsk, dense_by_nu,
                    args.meas_precision)

    print("-" * 78)
    print(f"  json: {OUT/'thermometer.json'}")
    print(f"  npz : {OUT/'thermometer.npz'}")
    print("=" * 78)


def make_figure(g, R, gk, Rk, Rsk, dense_by_nu, precisions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGS.mkdir(exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax1.errorbar(gk, Rk, yerr=Rsk, fmt="o", ms=5, capsize=3, color="#1f3b73",
                 label="calibration (T1, $\\pm$SEM)", zorder=3)
    ax1.plot(g, R, "-", lw=1.8, color="#c1440e", label="PCHIP inverse map", zorder=2)
    ax1.set_xscale("log")
    ax1.set_xlabel(r"$\Gamma = k_BT/E_J$")
    ax1.set_ylabel(r"$R = W_{\rm net}/W_{\rm net}^{\rm det}$")
    ax1.set_title("Forward calibration (control-subtracted)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, frameon=False)

    cols = ["#1f3b73", "#2a7f62", "#c1440e", "#7d4a9e", "#b08900"]
    for d, col in zip(dense_by_nu, cols):
        ax2.plot(d["T_mK"], d["sigma_T_mK"], "-" if d["calibrated"] else "--",
                 lw=1.7, color=col,
                 label=(f"$\\nu$={d['nu']}, $I_c$={d['Ic_nA']:.0f} nA"
                        + ("" if d["calibrated"] else " (extrap.)")))
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel(r"$T_e$ (mK)")
    ax2.set_ylabel(r"$\sigma(T_e)$ (mK)")
    ax2.set_title(f"Resolution at {precisions[0]:.0%} width precision")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    p = FIGS / "fig_thermometer.png"
    fig.savefig(p, dpi=160)
    print(f"  fig : {p}")


if __name__ == "__main__":
    main()
