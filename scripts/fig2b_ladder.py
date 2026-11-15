"""M2 Fig. 2b -- 3D S21 + ring-down coupling-ladder plot.

Renders the AT-3/AT-4 side-coupled W1+L3 ladder result (figure_plan.md
Fig. 2b, priority #4 in "Outstanding plot work" -- data complete, plot
missing). Pure rendering of stored, already-computed MEEP outputs -- no
physics is re-derived and no RNG is involved (rng: "deterministic - no
RNG" per every source artifact's own header).

Four panels:
  (a) S21(nu) near the cavity line for the res18 N=3 anchor run, zoomed to
      the measured dip (only 5.8% deep -- band-tail/Fano dilution, NOT an
      idealized clean notch) with the off-resonance floor noted.
  (b) Ring-down Q convergence vs cumulative simulated periods for
      N=2/3/4 (the harminv stationarity check behind each headline Q).
  (c) Q_loaded(N) with the Q_i anchor and the N=2 near-critical anomaly
      flagged by its mode-pulling signature |nu0-nu_cavity|.
  (d) Q_w(N) on a log axis with the 2-point exponential fit (N=3,4) and
      the N*=6.65~=7 extrapolation for A=1% (Qw=200 Q_i), explicitly
      flagged as a design estimate outside the simulated N range.

Inputs (read-only):
  photonics.json                                                  -- ladder, tcmt_check, modes, geometry
  out/m2/s21_3d_lad18_s21_N3_s020_rdef115_res18.json               -- S21(nu) spectrum, N=3 anchor
  out/m2/s21_3d_l18_harminv_N{2,3,4}_s020_rdef115_res18.json       -- ring-down Q convergence (q_trend_clean)

Output:
  figs/fig_m2_ladder.png (dpi=200, Agg)
  figs/fig_m2_ladder.caption.txt
"""
import json
import math
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
PHOTONICS_PATH = ROOT / "photonics.json"
S21_N3_PATH = ROOT / "out" / "m2" / "s21_3d_lad18_s21_N3_s020_rdef115_res18.json"
HARMINV_PATHS = {
    2: ROOT / "out" / "m2" / "s21_3d_l18_harminv_N2_s020_rdef115_res18.json",
    3: ROOT / "out" / "m2" / "s21_3d_l18_harminv_N3_s020_rdef115_res18.json",
    4: ROOT / "out" / "m2" / "s21_3d_l18_harminv_N4_s020_rdef115_res18.json",
}
OUT_PNG = ROOT / "figs" / "fig_m2_ladder.png"
OUT_CAPTION = ROOT / "figs" / "fig_m2_ladder.caption.txt"

C_LIGHT_M_S = 2.99792458e8


def main():
    ph = json.loads(PHOTONICS_PATH.read_text(encoding="utf-8"))
    s21 = json.loads(S21_N3_PATH.read_text(encoding="utf-8"))
    harminv = {N: json.loads(p.read_text(encoding="utf-8")) for N, p in HARMINV_PATHS.items()}

    a_device_m = ph["geometry"]["a_um_precise"] * 1e-6         # src: photonics.json:geometry.a_um_precise
    nu_cavity = ph["modes"]["nu0_a_over_lambda_precise"]        # src: photonics.json:modes
    ladder = ph["ladder"]                                       # src: photonics.json:ladder
    tcmt = ph["tcmt_check"]                                     # src: photonics.json:tcmt_check

    def nu_to_ghz(nu):
        return np.asarray(nu) * C_LIGHT_M_S / a_device_m / 1e9

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 11.5))
    ax_s21, ax_ring, ax_qload, ax_qw = axes.ravel()

    # ======================================================================
    # (a) S21(nu) near the cavity line, N=3 res18 anchor
    # ======================================================================
    sp = s21["spectra"]
    nu = np.asarray(sp["freqs"])
    S21b = np.asarray(sp["S21_band"])
    S21f = np.asarray(sp["S21_flux"])
    window = (nu > 0.278) & (nu < 0.290)

    ax_s21.plot(nu[window], S21b[window], "-", color="#1f77b4", lw=1.4,
                label="S21 band-power (MEEP, N=3 anchor)")
    ax_s21.plot(nu[window], S21f[window], "--", color="#2ca02c", lw=1.1,
                alpha=0.85, label="S21 flux (cross-check)")
    fitb = tcmt["Qw_N3_s21fit"][0]
    nu0_fit = s21["tcmt_fit_band"]["nu0"]
    Tmin_fit = s21["tcmt_fit_band"]["T_min_fit"]
    ax_s21.axvline(nu0_fit, color="0.4", lw=0.9, ls=":")
    ax_s21.axhline(Tmin_fit, color="C3", lw=0.9, ls="--")
    ax_s21.annotate(
        f"dip depth {(1 - Tmin_fit) * 100:.1f}%\n(band-tail/Fano diluted;\n"
        "NOT an idealized clean notch)",
        (nu0_fit, Tmin_fit), textcoords="offset points", xytext=(45, -60),
        fontsize=7.4, color="C3", ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="C3", alpha=0.9),
    )
    ax_s21.text(
        0.985, 0.04,
        f"off-res min T ($\\pm$5 GHz)="
        f"{tcmt['offres_99p5_note'].split('=')[-1].strip().rstrip('.')}"
        "\n(brief's >=99.5% target NOT met\n-- band-tail artifact)",
        transform=ax_s21.transAxes, fontsize=6.8, color="0.3", va="bottom", ha="right",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.88),
    )
    ax_s21.set_xlabel(r"$\nu = a/\lambda$  (reduced frequency)")
    ax_s21.set_ylabel("transmission")
    ax_s21.set_title("(a) S21 near cavity line, res18 N=3 anchor", pad=32)
    ax_s21.grid(alpha=0.3)
    ax_s21.legend(loc="lower left", fontsize=7.3, framealpha=0.9)

    ax_s21_top = ax_s21.secondary_xaxis("top", functions=(nu_to_ghz, lambda f: f * 1e9 * a_device_m / C_LIGHT_M_S))
    ax_s21_top.set_xlabel(r"$f$ (GHz), device $a$=586.735 $\mu$m rescale")

    # ======================================================================
    # (b) ring-down Q convergence vs cumulative periods, N=2/3/4
    # ======================================================================
    colors = {2: "#d62728", 3: "#1f77b4", 4: "#2ca02c"}
    for N, color in colors.items():
        trend = harminv[N]["result"]["q_trend_clean"]
        periods = [t["cum_periods"] for t in trend]
        Qs = [t["Q"] for t in trend]
        marker = "s" if N == 2 else ("o" if N == 3 else "^")
        ax_ring.plot(periods, Qs, "-", marker=marker, ms=5, color=color, lw=1.3,
                     label=f"N={N} (headline Q={ladder['Q_loaded'][f'N{N}']['value']:.0f})")
    ax_ring.set_xlabel("cumulative ring-down periods")
    ax_ring.set_ylabel(r"$Q_{loaded}$ (harminv, windowed)")
    ax_ring.set_title("(b) ring-down Q convergence (stationarity check)")
    ax_ring.grid(alpha=0.3)
    ax_ring.legend(loc="center right", fontsize=8, framealpha=0.9)

    # ======================================================================
    # (c) Q_loaded(N) with Q_i anchor and the N=2 near-critical anomaly
    # ======================================================================
    Ns_all = [2, 3, 4]
    Qloaded = [ladder["Q_loaded"][f"N{N}"]["value"] for N in Ns_all]
    nu0_N = {N: ladder["Q_loaded"][f"N{N}"]["nu0_a_over_lambda"] for N in Ns_all}
    Qi_anchor = ladder["Q_i_anchor"]["value"]

    ax_qload.plot(Ns_all, Qloaded, ":", color="0.5", lw=1.2, zorder=1)
    for N, Ql in zip(Ns_all, Qloaded):
        is_anom = (N == 2)
        ax_qload.scatter([N], [Ql], marker=("x" if is_anom else "o"), s=140,
                         color=("C3" if is_anom else "C0"),
                         edgecolor="k", linewidth=1.6, zorder=3)
    ax_qload.axhline(Qi_anchor, color="0.35", lw=1.1, ls="-.", zorder=0,
                      label=rf"$Q_i$ anchor = {Qi_anchor:.1f}")
    dnu2 = abs(nu0_N[2] - nu_cavity)
    dnu4 = abs(nu0_N[4] - nu_cavity)
    ax_qload.annotate(
        f"N=2: near-critical\n"
        f"$Q_{{loaded}}$={Qloaded[0]:.0f} ~ $Q_i$\n"
        f"mode-pulled: $|\\Delta\\nu|$={dnu2:.1e}\n"
        f"(N=4 sits within {dnu4:.1e}) -> excluded",
        (2, Qloaded[0]), textcoords="offset points", xytext=(50, -20),
        fontsize=7.3, color="C3", ha="left", va="top",
        bbox=dict(boxstyle="round", fc="white", ec="C3", alpha=0.92),
    )
    ax_qload.set_xticks(Ns_all)
    ax_qload.set_xlim(1.7, 4.5)
    ax_qload.set_xlabel(r"$N_{rows}$ (side-coupled separation)")
    ax_qload.set_ylabel(r"$Q_{loaded}$")
    ax_qload.set_title("(c) $Q_{loaded}(N)$: the N=2 anomaly", pad=12)
    ax_qload.grid(alpha=0.3)
    ax_qload.legend(loc="center right", fontsize=8, framealpha=0.9)

    # ======================================================================
    # (d) Q_w(N) exponential fit + N* extrapolation (design estimate)
    # ======================================================================
    fit = ladder["exponential_fit"]
    sigma, b = fit["sigma"], fit["b"]
    N_star = fit["N_star_for_A1pct"]
    Qw_target = fit["Qw_A1pct_target"]

    Qw_used_N = fit["usable_N"]          # [3, 4]
    Qw_used = [ladder["Q_w"][f"N{N}"]["value"] for N in Qw_used_N]
    Qw_N2 = ladder["Q_w"]["N2"]["value"]
    Qw_N5 = ladder["N5"]["Q_w_extrapolated"]

    ax_qw.scatter(Qw_used_N, Qw_used, marker="o", s=110, color="C0",
                  edgecolor="k", zorder=4, label="N=3,4 (used in fit)")
    ax_qw.scatter([2], [Qw_N2], marker="x", s=140, color="C3", linewidth=2,
                  zorder=4, label="N=2 (excluded, near-critical)")
    ax_qw.scatter([5], [Qw_N5], marker="^", s=110, facecolor="white",
                  edgecolor="C2", linewidth=1.8, zorder=4,
                  label="N=5 (extrapolated, not simulated)")

    N_line = np.linspace(1.5, N_star + 0.6, 200)
    Qw_line = np.exp(sigma * N_line + b)
    ax_qw.plot(N_line, Qw_line, "-", color="0.4", lw=1.4, zorder=2,
               label=rf"fit: $\ln Q_w={sigma:.3f}N+{b:.3f}$ (2 pts, $R^2$=1.0 trivial)")

    ax_qw.axhline(Qw_target, color="C1", lw=1.0, ls="--", zorder=1)
    ax_qw.scatter([N_star], [Qw_target], marker="*", s=260, color="gold",
                  edgecolor="k", zorder=5)
    ax_qw.annotate(
        f"$N^*$={N_star:.2f}$\\approx$7 rows\n"
        f"for $Q_w$=200$Q_i$={Qw_target:.2e} (A=1%)\n"
        "design estimate -- extrapolated beyond\nsimulated N in [3,4]",
        (N_star, Qw_target), textcoords="offset points", xytext=(-205, -25),
        fontsize=7.3, color="0.25", ha="left",
        bbox=dict(boxstyle="round", fc="white", ec="0.5", alpha=0.93),
    )
    ax_qw.set_yscale("log")
    ax_qw.set_xlabel(r"$N_{rows}$ (side-coupled separation)")
    ax_qw.set_ylabel(r"$Q_w$ (log scale)")
    ax_qw.set_title("(d) $Q_w(N)$ exponential coupling law", pad=12)
    ax_qw.grid(alpha=0.3, which="both")
    ax_qw.legend(loc="lower right", fontsize=7.2, framealpha=0.9)

    fig.suptitle(
        "M2 Fig. 2b -- side-coupled W1+L3 ladder: S21, ring-down Q "
        "convergence, and the coupling ladder ($Q_{loaded}$, $Q_w$)\n"
        r"res18 subpixel, 24a cell, 8 outer mirror rows; harminv ring-down "
        "is the primary Q method, S21 TCMT fit the cross-check",
        fontsize=10.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(hspace=0.38, wspace=0.26)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    assert OUT_PNG.exists()

    # ---- caption ----
    caption_lines = [
        "Figure 2b -- S21 + ring-down coupling-ladder plot (photonics.json:"
        "ladder, tcmt_check; out/m2/s21_3d_lad18_s21_N3_...json; out/m2/"
        "s21_3d_l18_harminv_N{2,3,4}_...json).",
        "",
        "(a) S21(nu) near the cavity line for the res18 N=3 anchor run "
        "(band-power and flux routes, independent TCMT fits). The measured "
        f"dip is only {(1 - Tmin_fit) * 100:.1f}% deep "
        f"(fit dip_depth={s21['tcmt_fit_band']['dip_depth_fit']:.4f}), far "
        "shallower than a naive lossless single-mode two-port notch "
        "(~81%), because the cavity line sits in the guided-band tail "
        "(band-tail/Fano dilution). The off-resonance transmission floor "
        f"beyond +/-5 GHz measures "
        f"{tcmt['offres_99p5_note'].split('=')[-1].strip().rstrip('.')} "
        "-- the brief's >=99.5% off-resonance target is not met, again a "
        "band-tail artifact rather than excess loss. Top axis: GHz at the "
        "device lattice constant a=586.735 um (photonics.json:geometry."
        "a_um_precise); the raw MEEP nu-axis is scale-invariant.",
        "",
        "(b) Ring-down Q convergence: the harminv-fitted Q vs cumulative "
        "simulated ring-down periods for N=2/3/4, i.e. the stationarity "
        "check underlying each headline Q_loaded (photonics.json requires "
        "q_stationary_last2_within_10pct=true for a 'clean' result; all "
        "three N pass). This is the ring-down method itself (brief Sec.5 "
        "primary Q technique), shown so the reader can see each Q has "
        "converged rather than being read off a single noisy window.",
        "",
        "(c) Q_loaded(N) at N=2/3/4: "
        + ", ".join(f"N={N}: {ladder['Q_loaded'][f'N{N}']['value']:.1f}"
                     for N in Ns_all)
        + f", against the N-independent in-situ Q_i anchor = {Qi_anchor:.1f} "
        "(24a-cell capped run). N=2 is flagged as near-critical "
        "(Q_loaded within ~6% of Q_i) and excluded from the exponential "
        "fit; the refined exclusion criterion also requires a "
        "mode-distortion signature |nu0(N) - nu_cavity| "
        "(nu_cavity=photonics.json:modes.nu0_a_over_lambda_precise): "
        f"N=2 pulls the line by {dnu2:.2e} while N=4 sits within "
        f"{dnu4:.2e} of the isolated-cavity value, so N=4's proximity to "
        "Q_i is the expected exponential-decoupling trend, not an "
        "anomaly. This mirrors the 2D exploratory ladder's own N=2 "
        "anomaly (figs/fig_m2_2d_s21_ladder.png) -- a 2D-predicted, "
        "3D-confirmed feature of this geometry.",
        "",
        "(d) Q_w(N) = the coupling-only Q backed out via "
        "1/Q_w = 1/Q_loaded - 1/Q_i_anchor: "
        + ", ".join(f"N={N}: {ladder['Q_w'][f'N{N}']['value']:.1f}" for N in [2, 3, 4])
        + f", N=5 extrapolated={Qw_N5:.1f}. The 2-point exponential fit over "
        f"the non-anomalous N=3,4 points, ln Q_w = {sigma:.5f}*N + {b:.5f} "
        "(R2=1.0 -- trivial for 2 points, NOT a validated statistical fit), "
        f"extrapolates to N* = {N_star:.3f} (~7 rows) for the A=1% design "
        f"target Q_w = 200 Q_i = {Qw_target:.6e}. N*=7 sits outside the "
        "simulated N in {3,4} and must be read as a design estimate, not "
        "a measured result.",
    ]
    OUT_CAPTION.write_text("\n".join(caption_lines), encoding="utf-8")

    # ---- console report ----
    print("inputs:")
    print(f"  {PHOTONICS_PATH} (ladder, tcmt_check, modes, geometry)")
    print(f"  {S21_N3_PATH}")
    for N, p in HARMINV_PATHS.items():
        print(f"  {p}")
    print('RNG: none -- every source artifact header states rng="deterministic - no RNG"')
    print("outputs:")
    print(f"  {OUT_PNG}  ({OUT_PNG.stat().st_size} bytes, dpi=200)")
    print(f"  {OUT_CAPTION}  ({OUT_CAPTION.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
