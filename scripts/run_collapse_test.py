"""D2 - the data-collapse test: turning the temperature family into a protocol.

D1 showed that the paper's counter-varying pair is a T = 0 feature, and observed
that the *family* of w(nu, T) curves might discriminate better than any single
one. This script makes that quantitative, because "the family looks distinctive"
is not a test.

The structural claim
--------------------
In the intrinsic picture the half-step is suppressed by phase diffusion whose
strength is Gamma = k_B T / E_J(nu), and E_J ~ I_c(nu) VARIES ACROSS THE DOME by
2.8x. So the thermal scale is filling-dependent, and the survival ratio

    y(nu, T) = w_measured(nu, T) / w_det(nu)

should be a function of Gamma ALONE: every filling collapses onto one curve when
plotted against k_B T / E_J(nu), and does NOT collapse when plotted against T.

An extrinsic sub-harmonic — trapped vortices, the origin found in twisted
cuprates (2512.01586) — has no reason to be governed by the junction's Josephson
energy. Its thermal scale is set by whatever pins the vortices, which is a
property of the film and its disorder, not of I_c(nu). The natural null is
therefore the mirror image: y collapses against T and not against Gamma.

The two hypotheses make OPPOSITE predictions about which axis collapses the data.
That is a much stronger statement than the shape of any one curve, and it is
measured by scanning two knobs the experiment has anyway: gate and temperature.

Method
------
For each (nu, T) on the grid, y is evaluated with that filling's own T1
calibration. Only points whose Gamma lies inside the calibrated range [0.005,
0.10] are used. For each candidate scaling variable (Gamma, or T) the points are
collapsed onto a common monotone curve (PCHIP through the binned medians) and the
RMS residual is reported. A collapse ratio

    Q_collapse = RMS(residual | T) / RMS(residual | Gamma)

far above 1 means the data distinguishes the hypotheses; near 1 means it does not.
Finally the required per-point measurement precision is derived: the hypotheses
separate when the residual difference exceeds the noise.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "out"
FIGS = ROOT / "figs"

HBAR, E_CHARGE, K_B = 1.054571817e-34, 1.602176634e-19, 1.380649e-23
NUS = [-2.4, -2.5, -2.6, -2.7, -2.8]
T_MK = [40, 50, 60, 75, 100, 125, 150, 200]


def nu_slug(nu):
    return "" if nu == -2.4 else "_nu_" + f"m{abs(nu):.1f}".replace(".", "_")


def main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "t2", ROOT / "scripts" / "make_thermometer.py")
    t2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t2)

    fp = json.loads((ROOT / "junction.json").read_text())["fingerprint"]
    inv, EJ = {}, {}
    for nu in NUS:
        cal = json.loads((OUT / f"thermometry_calibration{nu_slug(nu)}.json").read_text())
        inv[nu] = t2.build_inverse(cal)
        i = fp["nu"].index(nu)
        ic = fp["w_half_nA"][i] / fp["w_half"][i]
        EJ[nu] = HBAR * ic * 1e-9 / (2 * E_CHARGE * K_B)

    # ---- generate the intrinsic-model data set ---------------------------
    pts = []
    for nu in NUS:
        iv = inv[nu]
        for T in T_MK:
            g = (T * 1e-3) / EJ[nu]
            if not (np.exp(iv["lg"][0]) <= g <= np.exp(iv["lg"][-1])):
                continue
            y = float(np.asarray(iv["f_R"](np.log(g))).reshape(-1)[0])
            pts.append({"nu": nu, "T_mK": T, "Gamma": g, "y": y})
    print("=" * 78)
    print("D2 - collapse test")
    print(f"  {len(pts)} usable (nu, T) points out of {len(NUS)*len(T_MK)}; "
          f"the rest fall outside the calibrated Gamma range")
    print(f"  Gamma spans {min(p['Gamma'] for p in pts):.4f} .. "
          f"{max(p['Gamma'] for p in pts):.4f} "
          f"({max(p['Gamma'] for p in pts)/min(p['Gamma'] for p in pts):.1f}x)")
    print(f"  at fixed T, Gamma varies across the dome by "
          f"{EJ[-2.4]/EJ[-2.8]:.2f}x (the E_J lever)")

    def collapse_rms(xkey):
        """Collapse all points onto one monotone curve in log(x); RMS residual."""
        x = np.log(np.array([p[xkey] for p in pts], float))
        y = np.array([p["y"] for p in pts], float)
        o = np.argsort(x)
        x, y = x[o], y[o]
        # bin into ~8 bins and take medians -> the common curve
        nb = 8
        edges = np.linspace(x[0] - 1e-9, x[-1] + 1e-9, nb + 1)
        bx, by = [], []
        for i in range(nb):
            m = (x >= edges[i]) & (x < edges[i + 1])
            if m.sum() >= 2:
                bx.append(np.median(x[m]))
                by.append(np.median(y[m]))
        bx, by = np.array(bx), np.array(by)
        f = t2.pchip(bx, by)
        resid = y - np.asarray(f(np.clip(x, bx[0], bx[-1]))).reshape(-1)
        return float(np.sqrt(np.mean(resid**2))), float(np.max(np.abs(resid)))

    rms_g, max_g = collapse_rms("Gamma")
    rms_T, max_T = collapse_rms("T_mK")
    Q = rms_T / rms_g
    print("-" * 78)
    print(f"  collapse vs Gamma = k_B T/E_J(nu)  : RMS residual {rms_g:.5f} "
          f"(max {max_g:.5f})")
    print(f"  collapse vs T alone (extrinsic null): RMS residual {rms_T:.5f} "
          f"(max {max_T:.5f})")
    print(f"  Q_collapse = {Q:.2f}x")

    # ---- what precision does the test need? -------------------------------
    # The hypotheses separate when a per-point measurement error sigma_y is small
    # enough that the wrong-axis scatter is not absorbed into noise. Require the
    # wrong-axis RMS to exceed the noise by 3x.
    sigma_needed = rms_T / 3.0
    print(f"\n  To reject the wrong axis at 3 sigma per point, the survival ratio "
          f"y must be measured to better than {sigma_needed:.4f} "
          f"({sigma_needed*100:.1f} % of the deterministic width).")
    print(f"  For reference the T1 calibration's own SEM is 0.3-2.5 % of the net "
          f"width, so the numerics are not the limit -- the experiment is.")
    n_pts = len(pts)
    print(f"  With {n_pts} points the pooled separation is "
          f"{np.sqrt(n_pts)*rms_T/rms_g:.1f} sigma at that precision.")

    # ---- honest caveat: how universal is our own curve? -------------------
    # The five R_nu(Gamma) curves are NOT identical -- Gamma* runs 0.0180..0.0226
    # with Omega -- so even the intrinsic model does not collapse perfectly. That
    # residual is a prediction too, and it sets the floor of this test.
    gstars = []
    for nu in NUS:
        iv = inv[nu]
        u = t2.invert(iv["f_R"], 0.5, iv["lg"][0], iv["lg"][-1])
        gstars.append(float(np.exp(u)))
    print(f"\n  Floor of the test: the intrinsic curves are not exactly universal "
          f"either -- Gamma* runs {min(gstars):.4f}..{max(gstars):.4f} "
          f"({max(gstars)/min(gstars):.2f}x) with Omega(nu), which is why the "
          f"Gamma-collapse residual is {rms_g:.5f} rather than zero. That "
          f"residual is itself predicted and is the irreducible floor here.")

    result = {
        "description": "D2 collapse test: the intrinsic model predicts that the "
                       "survival ratio collapses against Gamma = k_B T/E_J(nu) "
                       "and not against T; an extrinsic sub-harmonic whose "
                       "thermal scale is not set by the junction's E_J predicts "
                       "the opposite.",
        "n_points_used": n_pts,
        "n_points_total": len(NUS) * len(T_MK),
        "EJ_lever_across_dome": EJ[-2.4] / EJ[-2.8],
        "collapse_vs_Gamma": {"rms_residual": rms_g, "max_residual": max_g},
        "collapse_vs_T": {"rms_residual": rms_T, "max_residual": max_T},
        "Q_collapse": Q,
        "sigma_y_needed_for_3sigma_per_point": sigma_needed,
        "gamma_star_by_filling": dict(zip(map(str, NUS), gstars)),
        "test_floor_note": "The five calibrations are not exactly universal "
                           "(Gamma* varies with Omega(nu)), so the Gamma-collapse "
                           "residual is nonzero by construction; that floor is "
                           "predicted, not fitted.",
        "points": pts,
    }
    (OUT / "collapse_test.json").write_text(json.dumps(result, indent=2))
    print(f"\n  json: {OUT/'collapse_test.json'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIGS.mkdir(exist_ok=True)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
        cols = {-2.4: "#1f3b73", -2.5: "#2a7f62", -2.6: "#c1440e",
                -2.7: "#7d4a9e", -2.8: "#b08900"}
        for nu in NUS:
            sel = [p for p in pts if p["nu"] == nu]
            if not sel:
                continue
            a1.plot([p["Gamma"] for p in sel], [p["y"] for p in sel], "o-",
                    color=cols[nu], ms=5, label=f"$\\nu$={nu}")
            a2.plot([p["T_mK"] for p in sel], [p["y"] for p in sel], "s--",
                    color=cols[nu], ms=5, label=f"$\\nu$={nu}")
        a1.set_xscale("log")
        a1.set_xlabel(r"$\Gamma = k_BT/E_J(\nu)$")
        a1.set_title(f"Intrinsic axis — collapses (RMS {rms_g:.4f})")
        a2.set_xlabel(r"$T$ (mK)")
        a2.set_title(f"Extrinsic axis — does not (RMS {rms_T:.4f})")
        for a in (a1, a2):
            a.set_ylabel(r"$w_{1/2}(T)\,/\,w_{1/2}^{\rm det}$")
            a.grid(alpha=0.3)
            a.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_collapse_test.png", dpi=160)
        print(f"  fig : {FIGS/'fig_collapse_test.png'}")
    except Exception as e:
        print(f"  (figure skipped: {e})")
    print("=" * 78)


if __name__ == "__main__":
    main()
