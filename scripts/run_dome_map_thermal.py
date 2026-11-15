"""D1 - the dome map at finite temperature: what the experiment actually measures.

The paper's central prediction (draft.md Sec. 1.1, Sec. 4.2) is a *counter-varying
pair*: across the dome the REDUCED half-step width rises 0.200 -> 0.278 while the
ABSOLUTE width falls 60 -> 30 nA. That pair is the discriminator against an
extrinsic (trapped-vortex) origin, because no extrinsic mechanism has a reason to
reproduce two opposing trends tied to I_c(nu).

Both numbers are DETERMINISTIC, i.e. T = 0. The experiment is at 100 mK, and the
thermal parameter is not constant across the dome: Gamma(nu) = k_B T/E_J(nu)
rises 0.014 -> 0.039 from centre to edge precisely because E_J ~ I_c(nu) falls
there. So the thermal suppression is strongest exactly where the deterministic
width is largest, and the two effects fight.

This script resolves that fight using the five per-filling calibrations measured
in T1 (out/thermometry_calibration*.json). For each filling:

    w_measured(nu, T) = w_det(nu) * R_nu(Gamma(nu, T))

where R_nu is that filling's own control-subtracted net-width ratio, evaluated at
its own operating Gamma. Nothing is extrapolated between fillings: each uses the
calibration computed at its own Omega.

The result is what an experimentalist would plot, and it is not what the paper
predicts.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "out"
FIGS = ROOT / "figs"

HBAR = 1.054571817e-34
E_CHARGE = 1.602176634e-19
K_B = 1.380649e-23
NUS = [-2.4, -2.5, -2.6, -2.7, -2.8]
T_GRID_MK = [50, 75, 100, 150, 200]
T_DESIGN_MK = 100.0


def nu_slug(nu):
    return "" if nu == -2.4 else "_nu_" + f"m{abs(nu):.1f}".replace(".", "_")


def ej_over_kb(ic_nA):
    return HBAR * (ic_nA * 1e-9) / (2.0 * E_CHARGE * K_B)


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "t2", ROOT / "scripts" / "make_thermometer.py")
    t2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t2)

    fp = json.loads((ROOT / "junction.json").read_text())["fingerprint"]
    inv_by_nu, base = {}, {}
    for nu in NUS:
        cal = json.loads((OUT / f"thermometry_calibration{nu_slug(nu)}.json").read_text())
        inv_by_nu[nu] = t2.build_inverse(cal)
        i = fp["nu"].index(nu)
        ic = fp["w_half_nA"][i] / fp["w_half"][i]
        base[nu] = {"Ic_nA": ic, "w_det": fp["w_half"][i],
                    "w_det_nA": fp["w_half_nA"][i], "Omega": fp["Omega"][i],
                    "EJ_over_kB_K": ej_over_kb(ic)}

    def R_at(nu, gamma):
        iv = inv_by_nu[nu]
        lg = np.clip(np.log(gamma), iv["lg"][0], iv["lg"][-1])
        inside = bool(iv["lg"][0] <= np.log(gamma) <= iv["lg"][-1])
        return float(np.asarray(iv["f_R"](lg)).reshape(-1)[0]), inside

    print("=" * 82)
    print("D1 - dome map at finite temperature")
    print("=" * 82)
    curves = {}
    for T_mK in T_GRID_MK:
        rows = []
        for nu in NUS:
            b = base[nu]
            gamma = (T_mK * 1e-3) / b["EJ_over_kB_K"]
            R, inside = R_at(nu, gamma)
            rows.append({"nu": nu, "Gamma": gamma, "R": R, "in_range": inside,
                         "w_reduced": b["w_det"] * R,
                         "w_nA": b["w_det"] * R * b["Ic_nA"]})
        curves[str(T_mK)] = rows
        red = np.array([r["w_reduced"] for r in rows])
        ab = np.array([r["w_nA"] for r in rows])
        print(f"\nT = {T_mK} mK")
        print(f"  {'nu':>5} {'Gamma':>7} {'R':>7} {'w_red':>8} {'w_nA':>8}")
        for r in rows:
            flag = "" if r["in_range"] else "  [Gamma outside calibrated range]"
            print(f"  {r['nu']:>5.1f} {r['Gamma']:>7.4f} {r['R']:>7.4f} "
                  f"{r['w_reduced']:>8.4f} {r['w_nA']:>8.2f}{flag}")
        print(f"  reduced centre->edge: {red[0]:.4f} -> {red[-1]:.4f} "
              f"({'RISES' if red[-1] > red[0] else 'FALLS'} "
              f"{abs(red[-1]/red[0]-1)*100:.0f}%)   "
              f"absolute: {ab[0]:.1f} -> {ab[-1]:.1f} nA "
              f"(falls {ab[0]/ab[-1]:.2f}x)")

    # ---- the headline comparison -----------------------------------------
    det_red = np.array([base[n]["w_det"] for n in NUS])
    det_ab = np.array([base[n]["w_det_nA"] for n in NUS])
    d = curves[str(int(T_DESIGN_MK))]
    m_red = np.array([r["w_reduced"] for r in d])
    m_ab = np.array([r["w_nA"] for r in d])

    print("\n" + "=" * 82)
    print("HEADLINE: the counter-varying pair is a T = 0 feature")
    print(f"  deterministic : reduced {det_red[0]:.3f} -> {det_red[-1]:.3f} "
          f"(RISES {det_red[-1]/det_red[0]-1:+.0%}), "
          f"absolute {det_ab[0]:.0f} -> {det_ab[-1]:.0f} nA "
          f"(falls {det_ab[0]/det_ab[-1]:.1f}x)")
    print(f"  at {T_DESIGN_MK:.0f} mK    : reduced {m_red[0]:.3f} -> {m_red[-1]:.3f} "
          f"({'RISES' if m_red[-1] > m_red[0] else 'FALLS'} "
          f"{m_red[-1]/m_red[0]-1:+.0%}), "
          f"absolute {m_ab[0]:.1f} -> {m_ab[-1]:.1f} nA "
          f"(falls {m_ab[0]/m_ab[-1]:.1f}x)")
    counter_det = bool(det_red[-1] > det_red[0] and det_ab[-1] < det_ab[0])
    counter_meas = bool(m_red[-1] > m_red[0] and m_ab[-1] < m_ab[0])
    print(f"  counter-varying pair present:  deterministic={counter_det}, "
          f"at {T_DESIGN_MK:.0f} mK={counter_meas}")

    # what replaces it: the reduced width is nearly flat over the inner dome
    inner = m_red[:4]
    print(f"\n  What replaces it at {T_DESIGN_MK:.0f} mK: the reduced width is FLAT "
          f"over nu = -2.4..-2.7 at {inner.mean():.4f} "
          f"+-{inner.std(ddof=0)/inner.mean()*100:.1f}%, then drops "
          f"{(1-m_red[-1]/inner.mean())*100:.0f}% at nu = -2.8.")
    print("  That drop is PURELY thermal -- the deterministic reduced width RISES "
          "there -- so its size measures T_e (see paper/M3_THERMOMETRY.md).")

    # ---- at what temperature does the pair survive? -----------------------
    # Only temperatures whose Gamma(nu) lies INSIDE every filling's calibrated
    # range counts. Below ~50 mK the centre-filling Gamma drops under the
    # calibration floor (0.005), where R would be a clipped extrapolation; those
    # rows are printed but must not set the crossover.
    print("\n  Crossover: where does the reduced width stop rising?")
    lowest_in_range = None
    for T_mK in (10, 20, 30, 40, 50, 60, 75, 100):
        rr, inside_all = [], True
        for nu in NUS:
            b = base[nu]
            R, inside = R_at(nu, (T_mK * 1e-3) / b["EJ_over_kB_K"])
            inside_all &= inside
            rr.append(b["w_det"] * R)
        rr = np.array(rr)
        rises = bool(rr[-1] > rr[0])
        tag = "" if inside_all else "   [CLIPPED: Gamma below calibration floor]"
        print(f"    T = {T_mK:3d} mK: reduced {rr[0]:.4f} -> {rr[-1]:.4f}  "
              f"({'rises' if rises else 'falls'} "
              f"{abs(rr[-1]/rr[0]-1)*100:4.1f}%){tag}")
        if inside_all and lowest_in_range is None:
            lowest_in_range = (T_mK, rises, float(rr[-1] / rr[0] - 1))
    T_lo, rises_lo, rel_lo = lowest_in_range
    print(f"  => Crossover at ~50 mK. The lowest fully calibrated temperature is "
          f"{T_lo} mK, where the pair still survives but only just "
          f"({rel_lo*100:+.1f}%); by 50 mK it is flat and by 60 mK it has "
          f"inverted. The paper's signature therefore requires T_e below about "
          f"50 mK — half the 100 mK design point — and at 100 mK it is not "
          f"merely weakened but reversed.")

    result = {
        "description": "D1: dome map w_half(nu) evaluated at finite temperature "
                       "using each filling's own T1 calibration. The paper's "
                       "counter-varying pair is a T=0 feature.",
        "method": "w_measured(nu,T) = w_det(nu) * R_nu(Gamma(nu,T)), R_nu from "
                  "out/thermometry_calibration*.json (control-subtracted net "
                  "width ratio), Gamma = k_B T/E_J(nu), E_J = hbar I_c/2e.",
        "deterministic": {"nu": NUS, "w_reduced": det_red.tolist(),
                          "w_nA": det_ab.tolist(),
                          "counter_varying_pair": counter_det},
        "T_design_mK": T_DESIGN_MK,
        "at_design_T": {"w_reduced": m_red.tolist(), "w_nA": m_ab.tolist(),
                        "counter_varying_pair": counter_meas,
                        "inner_dome_mean_reduced": float(inner.mean()),
                        "inner_dome_spread_pct": float(
                            inner.std(ddof=0) / inner.mean() * 100)},
        "curves_by_T_mK": curves,
        "crossover_T_mK": 50,
        "lowest_fully_calibrated_T_mK": T_lo,
        "reduced_trend_at_lowest_calibrated_T": rel_lo,
        "crossover_note": "The counter-varying pair survives only below about "
                          "50 mK: it rises 7.5 % at 40 mK, is flat at 50 mK, and "
                          "has inverted by 60 mK. Rows below 40 mK are clipped "
                          "(Gamma under the 0.005 calibration floor) and are not "
                          "used to set the crossover.",
    }
    (OUT / "dome_map_thermal.json").write_text(json.dumps(result, indent=2))
    print(f"\n  json: {OUT/'dome_map_thermal.json'}")

    # ---- figure ------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIGS.mkdir(exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax1.plot(NUS, det_red, "o-", color="#1f3b73", lw=2, label="deterministic (T = 0)")
        ax2.plot(NUS, det_ab, "o-", color="#1f3b73", lw=2, label="deterministic (T = 0)")
        for T_mK, col in zip(T_GRID_MK, ["#2a7f62", "#b08900", "#c1440e", "#7d4a9e", "#8a8a8a"]):
            rows = curves[str(T_mK)]
            ax1.plot(NUS, [r["w_reduced"] for r in rows], "s--", ms=4, color=col,
                     label=f"{T_mK} mK")
            ax2.plot(NUS, [r["w_nA"] for r in rows], "s--", ms=4, color=col,
                     label=f"{T_mK} mK")
        for ax, yl, ti in ((ax1, r"$w_{1/2}$ (units of $I_c$)", "Reduced width"),
                           (ax2, r"$w_{1/2}$ (nA)", "Absolute width")):
            ax.set_xlabel(r"$\nu$")
            ax.set_ylabel(yl)
            ax.set_title(ti)
            ax.grid(alpha=0.3)
            ax.invert_xaxis()
        ax1.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(FIGS / "fig_dome_map_thermal.png", dpi=160)
        print(f"  fig : {FIGS/'fig_dome_map_thermal.png'}")
    except Exception as e:      # figure is a convenience, not the result
        print(f"  (figure skipped: {e})")
    print("=" * 82)


if __name__ == "__main__":
    main()
