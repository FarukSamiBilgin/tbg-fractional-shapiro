"""M3 Fig. 3c -- production robustness result: survival curve W_1/2(Gamma).

Renders the A1 Gamma-scan (out/survival_curve.json) that has no plot yet
(figure_plan.md Fig. 3c, priority #3 in "Outstanding plot work"). Pure
rendering of stored, already-computed numbers -- no physics is re-derived
and no RNG is used here (the underlying noisy realizations were generated
by scripts/run_survival_curve.py and are frozen in the artifact).

Panel: survival fraction W_1/2^noisy / w_1/2^det (left axis, also read off
as the noisy half-step width in units of I_c on the right axis, since
survival = w_half_ensemble_mean / w_det_reference exactly) vs Gamma on a
log x-axis, with a top axis in E_J/k_BT = 1/Gamma. Marks:
  - the 50% survival line,
  - Gamma* = 0.028 <-> E_J/k_BT = 35.7 (the visibility threshold),
  - three labelled operating points: v0.3 production (Gamma=0.014),
    the Ic=100 nA equivalent (Gamma=0.042, insufficient), and the v0.2
    archive (Gamma=0.14),
  - the Gamma >~ 0.04 tail shaded as the unresolved detection floor,
    where the ensemble mean flattens at 0.050-0.075 Ic (2-3 cells of the
    i_dc grid step -- the discretization "width quantum").

Inputs (read-only):
  out/survival_curve.json  -- per_gamma[], headline{}, numerics{}

Output:
  figs/fig_m3_survival_curve.png (dpi=200, Agg)
  figs/fig_m3_survival_curve.caption.txt
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = pathlib.Path(__file__).resolve().parents[1]
SURVIVAL_PATH = ROOT / "out" / "survival_curve.json"
OUT_PNG = ROOT / "figs" / "fig_m3_survival_curve.png"
OUT_CAPTION = ROOT / "figs" / "fig_m3_survival_curve.caption.txt"


def main():
    d = json.loads(SURVIVAL_PATH.read_text(encoding="utf-8"))
    per_gamma = d["per_gamma"]                       # src: survival_curve.json:per_gamma
    hl = d["headline"]                                # src: survival_curve.json:headline
    numerics = d["numerics"]                          # src: survival_curve.json:numerics
    w_det = d["w_det_reference"]                      # src: survival_curve.json:w_det_reference

    gamma = [p["Gamma"] for p in per_gamma]
    surv = [p["survival_ensemble_mean"] for p in per_gamma]
    surv_std = [p["survival_per_real_std"] for p in per_gamma]
    frac_above = [p["frac_real_above_50pct"] for p in per_gamma]

    # sanity: survival_ensemble_mean == w_half_ensemble_mean / w_det_reference
    for p in per_gamma:
        lhs = p["survival_ensemble_mean"]
        rhs = p["w_half_ensemble_mean"] / w_det
        assert abs(lhs - rhs) < 1e-9, (lhs, rhs, p["Gamma"])

    gamma_star = hl["gamma_star_50pct_survival"]           # 0.028
    ej_kbt_star = hl["EJ_over_kBT_threshold"]               # 35.71
    prod = hl["production_v0p3"]                            # Gamma=0.014
    ic100 = hl["Ic100nA_equivalent"]                         # Gamma=0.042
    v02 = hl["v0p2_archive"]                                 # Gamma=0.14
    v_res = numerics["v_resolution"]                         # width quantum (Ic units)

    fig, ax = plt.subplots(figsize=(8.6, 6.0))

    # --- detection-floor tail: ensemble mean flattens at 0.050-0.075 Ic,
    # i.e. survival = (0.050..0.075)/0.2 = 0.25..0.375, once Gamma gtrsim 0.04
    floor_lo_g = 0.042
    ax.axvspan(floor_lo_g, max(gamma) * 1.15, color="0.55", alpha=0.14,
               hatch="///", zorder=0,
               label=r"$\Gamma \gtrsim 0.04$: unresolved detection floor"
                     "\n(ensemble mean pinned to 2-3 grid cells)")

    ax.axhline(0.5, color="0.35", lw=1.1, ls="--", zorder=1)
    ax.text(gamma[0], 0.515, "50% survival", fontsize=8, color="0.3", ha="left")

    ax.errorbar(gamma, surv, yerr=surv_std, fmt="o-", ms=6.5, lw=1.5,
                capsize=3.0, color="C0", ecolor="C0", elinewidth=1.0,
                markeredgecolor="k", zorder=3, label=r"$W_{1/2}^{noisy}/w_{1/2}^{det}(\Gamma)$"
                " (n_real=20 ensemble mean, $\\pm1\\sigma$)")

    ax.axvline(gamma_star, color="C3", lw=1.3, ls=":", zorder=2)
    ax.annotate(rf"$\Gamma^*$={gamma_star:.3f}"
                f"\n$\\leftrightarrow E_J/k_BT$={ej_kbt_star:.1f}",
                (gamma_star, 0.045), textcoords="offset points", xytext=(5, 0),
                fontsize=7.8, color="C3", ha="left", va="bottom")

    # three labelled operating points
    points = [
        (prod["Gamma"], prod["survival"], f"v0.3 production\n$\\Gamma$={prod['Gamma']:.3f}, "
         f"$I_c$={prod['Ic_nA']:.0f} nA\nsurvival={prod['survival']:.2f} (all 20 real. >50%)",
         "C2", (-95, -58)),
        (ic100["Gamma"], ic100["survival"], f"$I_c$=100 nA equiv.\n$\\Gamma$={ic100['Gamma']:.3f}\n"
         f"survival={ic100['survival']:.3f} (insufficient)", "C1", (10, 20)),
        (v02["Gamma"], v02["survival"], f"v0.2 archive\n$\\Gamma$={v02['Gamma']:.2f}, "
         f"$I_c$={v02['Ic_nA']:.0f} nA\nsurvival={v02['survival']:.3f}", "0.35", (12, -55)),
    ]
    for g, s, label, color, off in points:
        ax.scatter([g], [s], marker="*", s=230, color=color, edgecolor="k",
                   zorder=5)
        ax.annotate(label, (g, s), textcoords="offset points", xytext=off,
                    fontsize=7.3, color=color,
                    bbox=dict(boxstyle="round", fc="white", ec=color, alpha=0.92))

    ax.set_xscale("log")
    ax.set_xlabel(r"thermal parameter $\Gamma$  (log scale)")
    ax.set_ylabel(r"survival fraction  $W_{1/2}^{noisy}/w_{1/2}^{det}$")
    ax.set_ylim(-0.02, 0.95)
    ax.set_xlim(min(gamma) * 0.85, max(gamma) * 1.2)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=7.6, framealpha=0.92)

    # right axis: absolute noisy half-step width in Ic units (== survival * w_det)
    ax2 = ax.secondary_yaxis("right", functions=(lambda s: s * w_det, lambda w: w / w_det))
    ax2.set_ylabel(rf"$W_{{1/2}}^{{noisy}}$  (units of $I_c$; $w_{{1/2}}^{{det}}$={w_det:.3f})",
                   color="C4")
    ax2.tick_params(axis="y", colors="C4")

    # top axis: E_J/k_BT = 1/Gamma
    ax_top = ax.secondary_xaxis("top", functions=(lambda g: 1.0 / g, lambda e: 1.0 / e))
    ax_top.set_xlabel(r"visibility axis  $E_J/k_BT = 1/\Gamma$")
    ax_top.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_top.xaxis.set_minor_formatter(mticker.NullFormatter())

    fig.suptitle(
        "M3 Fig. 3c -- A1 production robustness: half-integer Shapiro-step "
        r"survival vs thermal parameter $\Gamma$" "\n"
        r"Beenakker $\tau$=0.8 CPR, $\nu=-2.4$, $\Omega$=5, $i_{ac}$=3.5; "
        rf"$w_{{1/2}}^{{det}}$={w_det:.3f} $I_c$ (stochastic-Heun, "
        "n_real=20/point)",
        fontsize=9.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    assert OUT_PNG.exists()

    # ---- caption ----
    seeds = d["seeds"]                                # src: survival_curve.json:seeds
    caption_lines = [
        "Figure 3c -- Survival curve W_1/2(Gamma), the A1 production "
        "robustness result (Beenakker tau=0.8 CPR, nu=-2.4, Omega=5, "
        "i_ac=3.5; out/survival_curve.json).",
        "",
        f"Main curve: ensemble-mean noisy half-integer (v=Omega/2) step "
        f"width, normalized by the deterministic width w_1/2^det={w_det:.4f} "
        f"Ic (stochastic-Heun Langevin integrator, spp=100, n_transient=150, "
        f"n_avg=300, n_real=20 realizations per Gamma; RK4 deterministic "
        f"reference spp=60), plotted vs Gamma on a log axis with error bars "
        f"= +/-1 std of the per-realization width divided by w_1/2^det. Top "
        f"axis: the visibility parameter E_J/k_BT = 1/Gamma. Right axis: "
        f"the same curve read off as an absolute noisy half-step width in "
        f"units of I_c (survival x w_1/2^det).",
        "",
        f"Gamma* = {gamma_star:.3f} <-> E_J/k_BT = {ej_kbt_star:.2f} marks "
        f"the 50%-survival visibility criterion (bracketed between the "
        f"measured Gamma={hl['gamma_star_bracket']['gamma_lo']:.3f} "
        f"(survival={hl['gamma_star_bracket']['survival_lo']:.3f}) and "
        f"Gamma={hl['gamma_star_bracket']['gamma_hi']:.3f} "
        f"(survival={hl['gamma_star_bracket']['survival_hi']:.3f}) grid "
        f"points): E_J/k_BT > 35.7 is required for the half-integer step to "
        "retain more than half of its deterministic width.",
        "",
        "Three labelled points (stars):",
        f"  - v0.3 production: Gamma={prod['Gamma']:.3f} (Ic={prod['Ic_nA']:.0f} nA, "
        f"T_op={prod['T_op_mK']:.0f} mK) -> survival={prod['survival']:.2f}, "
        "all 20 realizations individually above 50% "
        f"(frac_real_above_50pct={per_gamma[[p['Gamma'] for p in per_gamma].index(prod['Gamma'])]['frac_real_above_50pct']:.2f}).",
        f"  - Ic=100 nA equivalent: Gamma={ic100['Gamma']:.3f} -> "
        f"survival={ic100['survival']:.3f}, below the 50% criterion "
        f"({ic100['answer']}).",
        f"  - v0.2 archive: Gamma={v02['Gamma']:.2f} (Ic={v02['Ic_nA']:.0f} nA, "
        f"T_op={v02['T_op_mK']:.0f} mK) -> survival={v02['survival']:.3f}.",
        "",
        f"Shaded/hatched tail (Gamma >~ 0.042): the unresolved detection "
        f"floor. Beyond this Gamma the ensemble-mean width flattens and "
        f"becomes non-monotonic (0.050-0.075 Ic across Gamma=0.07..0.39, "
        f"including the v0.2 archive point at Gamma=0.14), because the "
        f"measured width has quantized to 2-3 cells of the i_dc grid step "
        f"(numerics.v_resolution={v_res:.5f} Ic; i_dc grid step=0.025 Ic) "
        "-- the residual apparent non-monotonicity there is this "
        "discretization \"width quantum\", not a physical recovery of "
        "visibility.",
        "",
        f"Seeds (seed_formula = 20260725000 + round(1000*Gamma)): "
        + ", ".join(f"Gamma={g}: seed {s}" for g, s in seeds.items()) + ".",
    ]
    OUT_CAPTION.write_text("\n".join(caption_lines), encoding="utf-8")

    # ---- console report ----
    print("inputs:")
    print(f"  {SURVIVAL_PATH}")
    print("RNG seeds inherited from data (seed_formula = 20260725000 + round(1000*Gamma)):")
    for g, s in seeds.items():
        print(f"  Gamma={g}: seed {s}")
    print("this script itself uses no RNG (pure rendering of stored ensemble statistics)")
    print("outputs:")
    print(f"  {OUT_PNG}  ({OUT_PNG.stat().st_size} bytes, dpi=200)")
    print(f"  {OUT_CAPTION}  ({OUT_CAPTION.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
