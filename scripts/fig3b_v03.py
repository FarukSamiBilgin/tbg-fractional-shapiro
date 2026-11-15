"""M3 TASK A3 -- Fig. 3b PAPER VERSION (contract v0.3, wide-junction rescale).

Regenerates the AT-3 fractional-fingerprint-vs-filling figure for the paper
under contract v0.3 (DECISION_contract_v0p3.md; paper/decisions.md
2026-07-25). IcRn = 60 uV is unchanged, so every REDUCED-unit deterministic
result (w_half(nu) in Ic units, Omega(nu), i_ac_opt(nu), the CPR shape) is
IDENTICAL to v0.2; only Gamma(nu) relabels (E_J x10 => Gamma(nu)/10 at
100 mK) and absolute currents/powers scale x10.

Panel (a): deterministic w_half(nu) -- left axis in units of Ic(nu)
(unchanged from v0.2), right axis in nA at v0.3 currents (x10 vs the v0.2
archive figure), with i_ac_opt(nu) annotations. Filled markers = the 5
independently simulated fillings (run_fingerprint_nu.py); open markers =
their dome-symmetric mirror copies (material.json nu_opt=-2.4 symmetry).

Panel (b): two-branch noisy survival overlay. For each of the 5 simulated
fillings, ensemble-mean noisy half-step width / deterministic half-step
width (survival_fraction, out/nu_survival.json) is plotted for:
  - v0.2 narrow-junction archive: Gamma_v02(nu) (0.140 dome-center .. 0.389
    edge), seed = 20260725100 + round(100*|nu|) + 0
  - v0.3 production (wide junction x10): Gamma_v03(nu) = Gamma_v02(nu)/10
    (0.014 .. 0.039), seed = 20260725100 + round(100*|nu|) + 1
Error bars are +/-1 std of the per-realization width (n_real=20) divided by
the deterministic width. Both branches trace the SAME i_ac_opt(nu) and
Omega(nu) (intrinsic dome physics, common to both -> shared shape along the
nu axis); the vertical gap between the two curves at fixed nu is the
thermal (Gamma) effect alone, since branches differ only in Gamma.

Inputs (read-only):
  out/fingerprint/nu_m2_[4-8].json  (deterministic AT-3 nu-scan)
  out/nu_survival.json              (A3 two-branch noisy nu-scan)
  out/survival_curve.json           (A1 Gamma-scan at nu=-2.4, cross-check only)
  junction.json                     (v0.3 contract; "fingerprint" block)

Output:
  figs/fig_m3_fingerprint_vs_nu_v03.png          (dpi=200, Agg)
  figs/fig_m3_fingerprint_vs_nu_v03.caption.txt  (sidecar caption text)

The archive figure figs/fig_m3_fingerprint_vs_nu.png (v0.2) is left in
place untouched.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
FP_DIR = ROOT / "out" / "fingerprint"
JUNCTION_PATH = ROOT / "junction.json"
NU_SURVIVAL_PATH = ROOT / "out" / "nu_survival.json"
SURVIVAL_CURVE_PATH = ROOT / "out" / "survival_curve.json"
OUT_PNG = ROOT / "figs" / "fig_m3_fingerprint_vs_nu_v03.png"
OUT_CAPTION = ROOT / "figs" / "fig_m3_fingerprint_vs_nu_v03.caption.txt"

NU_COMPUTED = [-2.4, -2.5, -2.6, -2.7, -2.8]


def main():
    junction = json.loads(JUNCTION_PATH.read_text(encoding="utf-8"))
    fp = junction["fingerprint"]
    assert fp.get("contract_version") == "0.3", "junction.json fingerprint block is not v0.3"

    nu_surv = json.loads(NU_SURVIVAL_PATH.read_text(encoding="utf-8"))
    surv_curve = json.loads(SURVIVAL_CURVE_PATH.read_text(encoding="utf-8"))

    # ---- cross-check: nu_survival's v0.3 branch at nu=-2.4 should agree
    # with survival_curve's dedicated Gamma=0.014 production point (both
    # measure the same physical ensemble: Omega=5, i_ac=3.5, Gamma=0.014,
    # at nu=-2.4; different seeds so not bit-identical, same regime). ----
    e24 = nu_surv["per_nu"]["-2.4"]["branches"]["v03"]
    ref = surv_curve["headline"]["production_v0p3"]
    print("cross-check (nu=-2.4, v0.3 branch, Gamma=0.014):")
    print(f"  nu_survival.json:   survival_fraction = {e24['survival_fraction']:.3f}")
    print(f"  survival_curve.json: survival (headline production_v0p3) = {ref['survival']:.3f}")

    nu_arr = fp["nu"]
    w_half = fp["w_half"]
    w_half_nA = fp["w_half_nA"]
    i_ac_opt = fp["i_ac_opt"]
    computed_set = set(NU_COMPUTED)
    is_computed = [n in computed_set for n in nu_arr]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13.2, 5.4))

    # ======================================================================
    # Panel (a): deterministic w_half(nu), Ic units (left) + nA (right, x10)
    # ======================================================================
    computed_nu = [n for n, c in zip(nu_arr, is_computed) if c]
    computed_w = [w for w, c in zip(w_half, is_computed) if c]
    mirror_nu = [n for n, c in zip(nu_arr, is_computed) if not c]
    mirror_w = [w for w, c in zip(w_half, is_computed) if not c]

    ax_a.plot(nu_arr, w_half, "-", color="0.6", lw=1.2, zorder=1)
    ax_a.scatter(computed_nu, computed_w, marker="o", s=70, color="C0",
                 edgecolor="k", zorder=3, label="simulated (5 fillings)")
    ax_a.scatter(mirror_nu, mirror_w, marker="o", s=70, facecolor="white",
                 edgecolor="C0", linewidth=1.6, zorder=3,
                 label=r"mirror copy (dome symmetric, $\nu_{opt}=-2.4$)")

    for n, w, iac in zip(nu_arr, w_half, i_ac_opt):
        ax_a.annotate(f"$i_{{ac}}$={iac:.2f}", (n, w), textcoords="offset points",
                       xytext=(0, 9), ha="center", fontsize=7.3, color="0.25")

    ax_a.set_xlabel(r"filling $\nu$")
    ax_a.set_ylabel(r"$w_{1/2}$  (units of $I_c(\nu)$)")
    ax_a.set_title("(a) deterministic half-step width vs filling")
    ax_a.grid(alpha=0.3)
    ax_a.legend(loc="upper center", fontsize=8, framealpha=0.9)
    ax_a.set_ylim(0, max(w_half) * 1.35)

    ax_a2 = ax_a.twinx()
    computed_wnA = [w for w, c in zip(w_half_nA, is_computed) if c]
    mirror_wnA = [w for w, c in zip(w_half_nA, is_computed) if not c]
    ax_a2.plot(nu_arr, w_half_nA, "--", color="C3", lw=1.0, alpha=0.6, zorder=1)
    ax_a2.scatter(computed_nu, computed_wnA, marker="^", s=55, color="C3",
                  edgecolor="k", zorder=3)
    ax_a2.scatter(mirror_nu, mirror_wnA, marker="^", s=55, facecolor="white",
                  edgecolor="C3", linewidth=1.6, zorder=3)
    ax_a2.set_ylabel(r"$w_{1/2}$  (nA, v0.3 currents $\times 10$)", color="C3")
    ax_a2.tick_params(axis="y", colors="C3")
    ax_a2.set_ylim(0, max(w_half_nA) * 1.35)

    # ======================================================================
    # Panel (b): two-branch noisy survival overlay
    # ======================================================================
    nu_plot = sorted(NU_COMPUTED)  # -2.8 .. -2.4, matches panel (a) direction
    branches = {"v02": {"color": "C1", "marker": "s",
                         "label": r"v0.2 narrow-junction archive: $\Gamma(\nu)$"},
                "v03": {"color": "C2", "marker": "o",
                        "label": r"v0.3 production: $\Gamma(\nu)/10$"}}

    data = {br: {"surv": [], "err": [], "gamma": []} for br in branches}
    for nu in nu_plot:
        entry = nu_surv["per_nu"][f"{nu:+.1f}"]
        w_det = entry["w_half_max_det"]
        for br in branches:
            b = entry["branches"][br]
            data[br]["surv"].append(b["survival_fraction"])
            data[br]["err"].append(b["w_half_perreal_std"] / w_det)
            data[br]["gamma"].append(b["Gamma"])

    for br, style in branches.items():
        ax_b.errorbar(nu_plot, data[br]["surv"], yerr=data[br]["err"],
                       fmt="-" + ("s" if style["marker"] == "s" else "o"),
                       ms=7, lw=1.6, capsize=3.5, color=style["color"],
                       ecolor=style["color"], elinewidth=1.0, alpha=0.95,
                       markeredgecolor="k", label=style["label"], zorder=3)

    # vertical connectors highlighting the thermal (Gamma) gap at fixed nu
    for i, nu in enumerate(nu_plot):
        y02 = data["v02"]["surv"][i]
        y03 = data["v03"]["surv"][i]
        ax_b.plot([nu, nu], [y02, y03], ":", color="0.55", lw=1.0, zorder=1)

    ax_b.axhline(0.5, color="0.7", lw=1.0, ls="--", zorder=0)
    ax_b.text(nu_plot[-1], 0.52, "50% survival", fontsize=7.5, color="0.45",
              ha="right", va="bottom")

    ax_b.set_xlabel(r"filling $\nu$")
    ax_b.set_ylabel(r"survival fraction  $W_{1/2}^{\rm noisy}\,/\,w_{1/2}^{\rm det}(\nu)$")
    ax_b.set_title("(b) thermal survival overlay: two Gamma branches")
    ax_b.grid(alpha=0.3)
    ax_b.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_b.set_ylim(0, 1.05)
    ax_b.text(
        0.02, 0.03,
        "shared curvature vs $\\nu$ (both branches): intrinsic $\\Omega(\\nu)$ effect\n"
        "vertical gap at fixed $\\nu$ (dotted): thermal ($\\Gamma$) effect alone",
        transform=ax_b.transAxes, fontsize=7.3, color="0.3", va="bottom", ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85),
    )

    fig.suptitle(
        "M3 Fig. 3b (paper, contract v0.3) -- AT-3 fractional fingerprint vs filling $\\nu$\n"
        r"Beenakker $\tau$=0.8 CPR; $I_cR_n$=60 $\mu$V invariant "
        r"$\Rightarrow$ reduced-unit results unchanged; $\Gamma$ relabels "
        r"$\times$0.1, currents/powers $\times$10",
        fontsize=10.3,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    assert OUT_PNG.exists()

    # ======================================================================
    # Caption sidecar
    # ======================================================================
    seed02 = {nu: nu_surv["per_nu"][f"{nu:+.1f}"]["branches"]["v02"]["seed"] for nu in nu_plot}
    seed03 = {nu: nu_surv["per_nu"][f"{nu:+.1f}"]["branches"]["v03"]["seed"] for nu in nu_plot}
    hl = nu_surv["headline_nu_-2.8"]

    caption_lines = [
        "Figure 3b (paper version, contract v0.3) -- Fractional Shapiro-step "
        "fingerprint vs filling nu.",
        "",
        "(a) Deterministic half-integer (v=Omega/2) step width w_1/2(nu) at "
        "its i_ac optimum (measured Beenakker tau=0.8 CPR, material.json v0.2 "
        "dome), in units of I_c(nu) (left axis, unchanged under v0.3: IcRn = "
        "60 uV is invariant, so the entire reduced-unit problem -- Omega(nu), "
        "i_ac_opt(nu), CPR shape, w_1/2(nu) -- is identical to the v0.2 "
        "archive) and in absolute nA at v0.3 currents (right axis, x10 vs "
        "the v0.2 archive figure: dome-center step 6.0 nA -> 60.0 nA). "
        "Filled circles are the 5 independently simulated fillings "
        "(nu = -2.4 .. -2.8, scripts/run_fingerprint_nu.py, deterministic "
        "RK4, spp=60, no RNG); open circles are their dome-symmetric mirror "
        "copies about nu_opt = -2.4 (material.json). Each point is "
        "annotated with its optimal drive amplitude i_ac_opt(nu).",
        "",
        "(b) Two-branch noisy survival overlay at each of the 5 simulated "
        "fillings' i_ac_opt(nu): ensemble-mean noisy half-step width divided "
        "by the deterministic width, W_1/2/w_1/2^det(nu) (stochastic-Heun "
        "Langevin, spp=100, n_real=20 realizations per point, out/"
        "nu_survival.json). Both branches use the SAME Omega(nu) and "
        "i_ac_opt(nu) (the intrinsic, Gamma-independent dome physics) and "
        "differ ONLY in the thermal parameter Gamma, so their shared shape "
        "along the nu axis isolates the intrinsic Omega(nu) suppression "
        "while the vertical gap between them at fixed nu (dotted "
        "connectors) isolates the thermal suppression:",
        f"  - v0.2 narrow-junction archive branch: Gamma_v0.2(nu) "
        f"(0.140 at nu=-2.4 rising to 0.389 at nu=-2.8), "
        f"seeds = 20260725100 + round(100*|nu|) + 0 -> "
        + ", ".join(f"nu={nu:+.1f}: seed {seed02[nu]}" for nu in nu_plot) + ".",
        f"  - v0.3 production branch: Gamma_v0.3(nu) = Gamma_v0.2(nu)/10 "
        f"(0.014 at nu=-2.4 rising to 0.039 at nu=-2.8), "
        f"seeds = 20260725100 + round(100*|nu|) + 1 -> "
        + ", ".join(f"nu={nu:+.1f}: seed {seed03[nu]}" for nu in nu_plot) + ".",
        "Error bars are +/-1 standard deviation of the per-realization step "
        "width across the n_real=20 realizations, normalized by w_1/2^det(nu).",
        "",
        "Caveat (Omega=13.9 edge point, nu=-2.8): Omega(nu) = f_drive / "
        "f_c(nu) is computed holding the normal-state resistance R_n FIXED "
        "at its dome-center (nu=-2.4) contract value across the whole "
        "nu-scan, while only I_c(nu) varies with filling per the measured "
        "dome (material.json); a real device's R_n would generically also "
        "vary with filling/junction width away from the dome center, which "
        "this Rn(nu)-fixed treatment does not capture. The nu=-2.8 point, "
        "with the largest Omega excursion in the scan (Omega=13.9 vs 5.0 at "
        "the dome center), is the most exposed to this approximation and "
        "should be read with that caveat; its large apparent w_1/2 (0.278 "
        f"Ic, {hl['w_half_max_det']:.4f} reduced) and its thermal-branch gap "
        f"(v0.2 survival {hl['v02_survival_fraction']:.2f} vs v0.3 survival "
        f"{hl['v03_survival_fraction']:.2f}) are measured self-consistently "
        "within this convention but are the least directly transferable "
        "to a fixed-geometry device sweep.",
        "",
        "Cross-check: the v0.3 branch at nu=-2.4 (Omega=5, i_ac=3.5, "
        "Gamma=0.014) independently reproduces the dedicated Gamma-scan "
        "production point (out/survival_curve.json, task A1) within "
        "realization-to-realization noise: "
        f"nu_survival.json survival_fraction={e24['survival_fraction']:.3f} "
        f"vs survival_curve.json production_v0p3 survival={ref['survival']:.3f} "
        "(different seeds, same Omega/i_ac/Gamma regime).",
    ]
    OUT_CAPTION.write_text("\n".join(caption_lines), encoding="utf-8")

    # ======================================================================
    # Console report
    # ======================================================================
    print("\ninputs:")
    print(f"  {JUNCTION_PATH} (fingerprint block, contract_version={fp['contract_version']})")
    for nuc in NU_COMPUTED:
        print(f"  {FP_DIR / f'nu_m2_{int(round(abs(nuc)*10)) - 20}.json'} (nu={nuc:+.1f})")
    print(f"  {NU_SURVIVAL_PATH}")
    print(f"  {SURVIVAL_CURVE_PATH} (cross-check only)")
    print("deterministic panel (a): yes (RK4, spp=60, no RNG)")
    print("stochastic panel (b): stochastic-Heun, spp=100, n_real=20, seeds logged above/in caption")
    print("\noutputs:")
    print(f"  {OUT_PNG}  ({OUT_PNG.stat().st_size} bytes, dpi=200)")
    print(f"  {OUT_CAPTION}  ({OUT_CAPTION.stat().st_size} bytes)")
    print(f"  (archive figure left untouched: {ROOT / 'figs' / 'fig_m3_fingerprint_vs_nu.png'})")


if __name__ == "__main__":
    main()
