"""M2/M4 Fig. 2d -- collection-structure finding: Q_J(d_eff) closure.

Renders the Q_J junction-loading closure (figure_plan.md Fig. 2d, priority
#2 in "Outstanding plot work" -- the paper's sharpest device conclusion,
S5.3). Pure rendering of stored, already-computed numbers -- no physics is
re-derived and no RNG is involved anywhere in this closure (deterministic
algebra over measured cavity parameters).

Panel: Q_J(d_eff) = coefficient / d_eff^2 on log-log axes, with
  - the bare-vdW-stack point (d_eff = 100 nm -> Q_J = 1.26e8, i.e. only
    0.014% of the intrinsic dissipation reaches the junction),
  - the measured cavity Q band: Q_i_insitu = 1.461e4, Q_rad ceiling
    2.7-3.0e4, Q_mat = 5.21e4 (photonics.json:modes),
  - the loading-criterion band Q_J <= 2700-3000 (Q_rad,Q_mat >= 10 Q_J)
    and the strict Q_J <= 1975.66 (90%) line,
  - the resulting d_eff requirement window ~20.5-25.2 um.

Inputs (read-only):
  out/qj_closure.json  -- inputs, q_j, requirement, absolute_voltage_thread
  photonics.json        -- modes.Q_i_insitu_24a, modes.Q_rad_intrinsic, modes.Q_mat

Output:
  figs/fig_m2_qj_closure.png (dpi=200, Agg)
  figs/fig_m2_qj_closure.caption.txt
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
QJ_PATH = ROOT / "out" / "qj_closure.json"
PHOTONICS_PATH = ROOT / "photonics.json"
OUT_PNG = ROOT / "figs" / "fig_m2_qj_closure.png"
OUT_CAPTION = ROOT / "figs" / "fig_m2_qj_closure.caption.txt"


def main():
    qj = json.loads(QJ_PATH.read_text(encoding="utf-8"))
    ph = json.loads(PHOTONICS_PATH.read_text(encoding="utf-8"))

    coeff = qj["q_j"]["coefficient_Rn_eps0_eps_c_omega0_Vmode"]      # src: qj_closure.json:q_j
    d_eff_default_nm = qj["inputs"]["d_eff_nm"]                       # 100.0
    Q_J_default = qj["q_j"]["Q_J_at_d_eff"]                           # 1.2587e8
    frac_default = qj["requirement"]["default_100nm_fraction_to_junction"]

    Q_J_max_10x = qj["requirement"]["Q_J_max_allowed_10x"]            # 2700
    Q_J_max_10x_radhi = qj["requirement"]["Q_J_max_allowed_10x_radhi"]  # 3000
    Q_J_max_90pct = qj["requirement"]["Q_J_max_for_90pct"]             # 1975.66
    d_eff_min_10x_um = qj["requirement"]["d_eff_min_10x_um"]           # 21.59
    d_eff_min_10x_radhi_um = qj["requirement"]["d_eff_min_10x_radhi_um"]  # 20.48
    d_eff_min_90pct_um = qj["requirement"]["d_eff_min_90pct_um"]       # 25.24

    Q_i_insitu = ph["modes"]["Q_i_insitu_24a"]                        # src: photonics.json:modes
    Q_rad_lo, Q_rad_hi = ph["modes"]["Q_rad_intrinsic"]
    Q_mat = ph["modes"]["Q_mat"]

    V_ac_default_uV = qj["absolute_voltage_thread"]["V_ac_v03_default_100nm_uV"]
    V_ac_required_mV = qj["absolute_voltage_thread"]["V_ac_required_mV"]
    d_eff_0p6mV_um = qj["absolute_voltage_thread"]["d_eff_for_0p6mV_measured_um"]

    # ---- Q_J(d_eff) curve, d_eff in meters over a wide log range ----
    d_eff_um = np.logspace(-2, 3, 400)          # 0.01 um .. 1000 um
    d_eff_m = d_eff_um * 1e-6
    Q_J = coeff / d_eff_m ** 2

    fig, ax = plt.subplots(figsize=(9.2, 6.4))

    ax.plot(d_eff_um, Q_J, "-", color="C0", lw=1.8, zorder=3,
            label=r"$Q_J(d_{eff}) = R_n\varepsilon_0\varepsilon_c\omega_0 V_{mode}/d_{eff}^2$")

    # loading-criterion band: Q_rad, Q_mat >= 10 Q_J
    ax.axhspan(Q_J_max_10x, Q_J_max_10x_radhi, color="C2", alpha=0.18, zorder=0,
               label=rf"allowed band $Q_J\leq${Q_J_max_10x:.0f}-{Q_J_max_10x_radhi:.0f} "
                     r"(10$\times$ criterion, $Q_{rad}$ 27-30k)")
    ax.axhline(Q_J_max_90pct, color="C3", lw=1.3, ls="--", zorder=2,
               label=rf"strict 90%-to-junction line $Q_J\leq${Q_J_max_90pct:.0f}")

    # d_eff requirement window
    ax.axvspan(d_eff_min_10x_radhi_um, d_eff_min_90pct_um, color="0.6", alpha=0.15,
               zorder=0,
               label=rf"required $d_{{eff}}\approx${d_eff_min_10x_radhi_um:.1f}"
                     rf"-{d_eff_min_90pct_um:.1f} $\mu$m")
    ax.axvline(d_eff_min_10x_um, color="0.4", lw=1.0, ls=":", zorder=1)
    ax.axvline(d_eff_min_90pct_um, color="0.4", lw=1.0, ls=":", zorder=1)

    # cavity Q reference lines (context, not loading limits themselves)
    ax.axhline(Q_i_insitu, color="0.55", lw=1.0, ls="-.", zorder=1)
    ax.text(d_eff_um[-1] * 0.55, Q_i_insitu * 1.15,
            rf"$Q_i^{{in situ}}$={Q_i_insitu:.0f}", fontsize=7.6, color="0.4", ha="right")
    ax.axhline(Q_mat, color="0.7", lw=0.8, ls="-.", zorder=1)
    ax.text(d_eff_um[-1] * 0.55, Q_mat * 1.1,
            rf"$Q_{{mat}}$={Q_mat:.0f}", fontsize=7.2, color="0.55", ha="right")

    # bare-vdW-stack point
    ax.scatter([d_eff_default_nm / 1000.0], [Q_J_default], marker="*", s=280,
               color="C1", edgecolor="k", zorder=5)
    ax.annotate(
        rf"bare vdW stack: $d_{{eff}}$=100 nm"
        f"\n$Q_J$={Q_J_default:.2e}"
        f"\n({frac_default * 100:.3f}% of dissipation to junction)"
        f"\n$V_{{ac}}$={V_ac_default_uV:.2f} $\\mu$V at $P_{{in}}$=90 nW"
        f" (need {V_ac_required_mV*1000:.0f} $\\mu$V)",
        (d_eff_default_nm / 1000.0, Q_J_default),
        textcoords="offset points", xytext=(-15, -55), fontsize=7.4, color="C1",
        ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="C1", alpha=0.93),
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"effective junction gap $d_{eff}$ ($\mu$m, log scale)")
    ax.set_ylabel(r"$Q_J$ (log scale)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=7.3, framealpha=0.92)
    ax.set_xlim(d_eff_um[0], d_eff_um[-1])

    fig.suptitle(
        "M2/M4 Fig. 2d -- junction-loading closure: $Q_J(d_{eff})$ vs the "
        "cavity's intrinsic-loss budget\n"
        r"$R_n$=200 $\Omega$, $f_0$=145 GHz, $V_{mode}$=0.331 $a^3$"
        r"=$6.68\times10^{7}\,\mu m^3$; criterion $Q_{rad},Q_{mat}\geq10\,Q_J$"
        rf" $\Rightarrow d_{{eff}}\approx${d_eff_min_10x_radhi_um:.1f}-"
        rf"{d_eff_min_90pct_um:.1f} $\mu$m needed",
        fontsize=9.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    assert OUT_PNG.exists()

    # ---- caption ----
    caption_lines = [
        "Figure 2d -- Collection-structure finding: junction-loading Q_J "
        "closure (out/qj_closure.json, photonics.json:modes).",
        "",
        "Q_J(d_eff) = Rn eps0 eps_c omega0 V_mode / d_eff^2 (deterministic "
        f"closed form; coefficient = {coeff:.4e} m^2), plotted on log-log "
        "axes for the effective junction gap d_eff. The junction is a "
        "lumped ohmic loss channel loading the cavity in parallel with its "
        "radiative (Q_rad) and material (Q_mat) intrinsic losses: "
        "1/Q_i = 1/Q_rad + 1/Q_mat + 1/Q_J.",
        "",
        f"Bare-vdW-stack point (star): d_eff = {d_eff_default_nm:.0f} nm gives "
        f"Q_J = {Q_J_default:.4e}, i.e. only {frac_default*100:.4f}% of the "
        "intrinsic cavity dissipation reaches the junction -- the junction "
        "is a negligible loss channel at the bare stack thickness. At the "
        f"v0.3 reference drive P_in = 90 nW this puts only "
        f"V_ac = {V_ac_default_uV:.3f} uV across the junction, far short of "
        f"the V_ac = {V_ac_required_mV*1000:.0f} uV the drive design "
        f"requires (that voltage would need d_eff = {d_eff_0p6mV_um:.1f} um "
        "at the measured, band-tail-diluted field level).",
        "",
        f"Loading-criterion band (green, Q_J <= {Q_J_max_10x:.0f}-"
        f"{Q_J_max_10x_radhi:.0f}): requires Q_rad, Q_mat >= 10 Q_J (>=~90% "
        "of the intrinsic loss reaching the junction), using the measured "
        f"Q_rad_intrinsic band [{Q_rad_lo:.0f}, {Q_rad_hi:.0f}] and "
        f"Q_mat = {Q_mat:.0f}. The strict-90% line (red dashed, "
        f"Q_J <= {Q_J_max_90pct:.2f}) is the tighter, single-number "
        f"version of the same criterion (Q_i_EM_rad_mat_only = "
        f"{qj['requirement']['Q_i_EM_rad_mat_only']:.1f}).",
        "",
        f"Resulting d_eff window (grey band): {d_eff_min_10x_radhi_um:.2f}-"
        f"{d_eff_min_90pct_um:.2f} um ({d_eff_min_10x_um:.2f} um at the "
        "Q_rad low-end of the intrinsic-ceiling band). Meeting the loading "
        "criterion needs engineered antenna/electrode coupling (an "
        "effective gap roughly 200-250x the bare 100 nm vdW stack "
        "thickness), not the bare stack -- the sharpest device-design "
        "conclusion of the collection-structure analysis (S5.3).",
        "",
        f"Reference lines: in-situ cavity Q_i = {Q_i_insitu:.0f} "
        "(photonics.json:modes.Q_i_insitu_24a, N=3 capped ladder run) and "
        f"Q_mat = {Q_mat:.0f} (loss closure from the 16a res18 lossy/"
        "lossless pair) are shown for scale, not as loading limits "
        "themselves.",
    ]
    OUT_CAPTION.write_text("\n".join(caption_lines), encoding="utf-8")

    # ---- console report ----
    print("inputs:")
    print(f"  {QJ_PATH}")
    print(f"  {PHOTONICS_PATH} (modes.Q_i_insitu_24a, modes.Q_rad_intrinsic, modes.Q_mat)")
    print("RNG: none (deterministic closed-form closure over measured cavity parameters)")
    print("outputs:")
    print(f"  {OUT_PNG}  ({OUT_PNG.stat().st_size} bytes, dpi=200)")
    print(f"  {OUT_CAPTION}  ({OUT_CAPTION.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
