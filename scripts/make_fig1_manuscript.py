"""Figure 1 of the manuscript: mechanism -> observable, entirely from stored artifacts.

(a) The current-phase relation: sinusoidal control vs the measured Beenakker
    tau = 0.8 CPR (material.json), whose second harmonic c2/c1 = -0.187 is the
    mechanism of the half-integer step.
(b) The observable: deterministic reduced I-V staircase at the dome centre
    (out/fingerprint/nu_m2_4.npz, i_ac = i_ac_opt = 3.5) with the half-integer
    plateau at v = Omega/2 highlighted, overlaid with the 100 mK Langevin
    ensemble mean (out/thermometry_calibration.npz, Gamma = 0.014, 160
    realizations) on its window.
(c) The fingerprint: half-step width vs drive alpha for all five fillings
    (out/fingerprint/nu_*.json), reduced units.

No new physics is computed here; every curve is read from a committed artifact.
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT, FIGS = ROOT / "out", ROOT / "figs"

COLS = {-2.4: "#1f3b73", -2.5: "#2a7f62", -2.6: "#c1440e",
        -2.7: "#7d4a9e", -2.8: "#b08900"}


def slug(nu):
    return f"nu_{nu:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")


def cpr(phi, c):
    return sum(cm * np.sin((m + 1) * phi) for m, cm in enumerate(c))


def main():
    mat = json.loads((ROOT / "material.json").read_text())
    c = np.array(mat["cpr"]["harmonics_by_tau"]["0.8"], float)

    fp = np.load(OUT / "fingerprint" / "nu_m2_4.npz")
    th = np.load(OUT / "thermometry_calibration.npz")

    fig, (a, b, cx) = plt.subplots(1, 3, figsize=(13.2, 3.9))

    # (a) CPR ---------------------------------------------------------------
    phi = np.linspace(0, 2 * np.pi, 600)
    a.plot(phi / np.pi, np.sin(phi), "--", color="#8a8a8a", lw=1.6,
           label=r"sinusoidal, $c_m=\delta_{m1}$")
    a.plot(phi / np.pi, cpr(phi, c), "-", color="#1f3b73", lw=2.2,
           label=r"Beenakker $\tau=0.8$")
    a.axhline(0, color="k", lw=0.5)
    a.set_xlabel(r"$\varphi/\pi$")
    a.set_ylabel(r"$I(\varphi)/I_c$")
    a.set_title(r"(a) skewed CPR: $c_2/c_1=-0.187$")
    a.legend(fontsize=8, frameon=False)
    a.grid(alpha=0.25)

    # (b) staircase ---------------------------------------------------------
    iac = fp["i_ac"]
    k = int(np.argmin(np.abs(iac - 3.5)))
    idc, v = fp["i_dc"], fp["v"][k]
    b.plot(idc, v, "-", color="#1f3b73", lw=1.8, label="deterministic ($T=0$)")
    g_idx = int(np.argmin(np.abs(th["gamma_grid"] - 0.014)))
    b.plot(th["idc"], th["v_skewed"][g_idx].mean(axis=0), "-", color="#c1440e",
           lw=1.8, label=r"100 mK ensemble ($\Gamma=0.014$, $n=160$)")
    for vv, lab in ((2.5, r"$v=\Omega/2$"), (5.0, r"$v=\Omega$")):
        b.axhline(vv, color="#8a8a8a", lw=0.7, ls=":")
        b.text(1.08, vv + 0.09, lab, fontsize=8, color="#555555")
    b.set_xlim(1.0, 4.4)
    b.set_ylim(0.8, 6.2)
    b.set_xlabel(r"$i_{dc}=I_{dc}/I_c$")
    b.set_ylabel(r"$v=V/(I_cR_n)$")
    b.set_title(r"(b) half-integer step, $\nu=-2.4$, $\alpha=0.70$")
    b.legend(fontsize=8, frameon=False, loc="upper left")
    b.grid(alpha=0.25)

    # (c) width vs drive across the dome -----------------------------------
    for nu in COLS:
        d = json.loads((OUT / "fingerprint" / f"{slug(nu)}.json").read_text())
        cx.plot(d["alpha"], d["w_half"], "o-", ms=3.5, lw=1.6,
                color=COLS[nu], label=rf"$\nu={nu}$, $\Omega={d['Omega']:.1f}$")
    cx.set_xlabel(r"drive $\alpha = eV_{ac}/\hbar\omega$")
    cx.set_ylabel(r"$w_{1/2}$ (units of $I_c(\nu)$)")
    cx.set_title("(c) deterministic fingerprint across the dome")
    cx.set_xlim(0, 1.6)
    cx.legend(fontsize=7.5, frameon=False)
    cx.grid(alpha=0.25)

    fig.tight_layout()
    p = FIGS / "fig_ms1_mechanism.png"
    fig.savefig(p, dpi=170)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
