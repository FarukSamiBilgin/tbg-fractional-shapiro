"""M2 Fig. 2a -- device-a TE-like photonic-crystal-slab band diagram.

Pure rendering of the stored MPB band table (figure_plan.md Fig. 2a: "needs a
dedicated MPB re-run that dumps the full per-band k-point table"). No physics is
re-derived here and no RNG is involved -- every number plotted comes straight
from out/m2/fig2a_bands.json, produced by scripts/m2_fig2a_bands.py inside the
m2env WSL distro (rng: "deterministic - no RNG").

Inputs (read-only):
  out/m2/fig2a_bands.json  -- 8-band x 52-k-point table (a/lambda and GHz),
                              air light line, gap edges, res-30 convergence,
                              consistency block vs photonics.json
  photonics.json           -- geometry.a_um_precise, modes.cavity_line_gap_GHz,
                              modes.f0_GHz  (labels only)

Output:
  figs/fig_m2_bands_device.png (dpi=200, Agg)
  figs/fig_m2_bands_device.caption.txt
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
BANDS_PATH = ROOT / "out" / "m2" / "fig2a_bands.json"
PHOTONICS_PATH = ROOT / "photonics.json"
OUT_PNG = ROOT / "figs" / "fig_m2_bands_device.png"
OUT_CAPTION = ROOT / "figs" / "fig_m2_bands_device.caption.txt"

GREEK = {"Gamma": r"$\Gamma$", "M": "M", "K": "K"}


def main():
    d = json.loads(BANDS_PATH.read_text(encoding="utf-8"))
    ph = json.loads(PHOTONICS_PATH.read_text(encoding="utf-8"))

    cfg = d["config"]
    kp = d["kpath"]
    sc = d["scale"]
    gap = d["gap"]
    cons = d["consistency_vs_photonics_json"]
    conv = d["convergence_res30"]

    x = np.asarray(kp["abscissa_2pi_over_a"], float)
    corner_i = kp["corner_indices"]
    corner_names = kp["corners"]
    bands = np.asarray(d["bands"]["a_over_lambda"], float)      # (8, nk)
    fll = np.asarray(d["bands"]["f_light_line_a_over_lambda"], float)
    nb, nk = bands.shape

    ghz_per_nu = sc["GHz_per_unit_nu"]
    a_um = sc["a_um"]
    nu_cav = sc["cavity_line_a_over_lambda"]
    f_cav = sc["cavity_line_GHz"]
    lo, hi = gap["edges_a_over_lambda"]
    lo_g, hi_g = gap["edges_GHz"]

    ymax = 0.62

    fig, ax = plt.subplots(figsize=(9.6, 6.6))

    # ---- light cone: everything above the air light line is leaky ----------
    ax.fill_between(x, fll, ymax, color="0.82", alpha=0.85, lw=0, zorder=0)
    ax.plot(x, fll, color="0.35", lw=1.4, ls="-", zorder=3)
    ax.text(x[-1] * 0.035, 0.578, "light cone (air):\nleaky modes",
            ha="left", va="center", fontsize=9.5, color="0.30", zorder=6)

    # ---- TE-like guided gap -------------------------------------------------
    ax.axhspan(lo, hi, color="#f2c14e", alpha=0.30, lw=0, zorder=1)
    ax.axhline(lo, color="#b8860b", lw=1.0, ls="-", zorder=4)
    ax.axhline(hi, color="#b8860b", lw=1.0, ls="-", zorder=4)

    # ---- bands: guided part solid, above-light-line part faded --------------
    for b in range(nb):
        yb = bands[b]
        guided = yb < fll
        ax.plot(x, yb, color="#8fa6c4", lw=1.0, alpha=0.75, zorder=2)
        yg = np.where(guided, yb, np.nan)
        color = "#1f4e79" if b < 2 else "#5b8db8"
        ax.plot(x, yg, color=color, lw=2.2 if b < 2 else 1.6, zorder=5)

    # ---- 145 GHz cavity line ------------------------------------------------
    ax.axhline(nu_cav, color="#c0392b", lw=1.8, ls="--", zorder=6)
    ax.text(x[-1] * 0.035, nu_cav + 0.005,
            f"145.000 GHz cavity line  ($\\nu_0$ = {nu_cav:.6f})",
            fontsize=9.5, color="#c0392b", va="bottom", ha="left", zorder=7)

    # gap-width arrow (kept clear of the labels) and the numbers box
    ax.annotate("", xy=(x[-1] * 0.012, lo), xytext=(x[-1] * 0.012, hi),
                arrowprops=dict(arrowstyle="<->", color="#b8860b", lw=1.2),
                zorder=7)
    box = (
        "TE-like guided gap (device $a$)\n"
        f"upper edge  $\\nu$ = {hi:.6f} at M   = {hi_g:.3f} GHz\n"
        f"lower edge  $\\nu$ = {lo:.6f} at K   = {lo_g:.3f} GHz\n"
        f"width          {hi - lo:.6f}              = "
        f"{hi_g - lo_g:.3f} GHz\n"
        f"cavity-line margins  +{f_cav - lo_g:.2f} GHz / "
        f"+{hi_g - f_cav:.2f} GHz"
    )
    ax.text(x[-1] * 0.30, 0.155, box, fontsize=9, color="#5c4600",
            va="center", ha="left", zorder=8, family="DejaVu Sans Mono",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#fdf6e3",
                      edgecolor="#b8860b", alpha=0.95))

    # ---- axes ---------------------------------------------------------------
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0.0, ymax)
    ax.set_xticks([x[i] for i in corner_i])
    ax.set_xticklabels([GREEK.get(n, n) for n in corner_names], fontsize=12)
    for i in corner_i[1:-1]:
        ax.axvline(x[i], color="0.6", lw=0.7, ls=":", zorder=1)
    ax.set_ylabel(r"frequency  $\nu = a/\lambda$   (MPB units $c/a$)")
    ax.set_xlabel("in-plane wavevector along $\\Gamma$-M-K-$\\Gamma$")

    axr = ax.secondary_yaxis(
        "right", functions=(lambda v: v * ghz_per_nu,
                            lambda f: f / ghz_per_nu))
    axr.set_ylabel(f"frequency (GHz) at the device $a$ = {a_um:.4f} $\\mu$m")

    ax.set_title(
        "Fig. 2a  TE-like slab band structure at the device lattice constant\n"
        f"triangular air holes, $r/a$ = {cfg['r_over_a']}, "
        f"$t/a$ = {cfg['t_over_a']}, $\\epsilon$ = {cfg['eps_core']}; "
        f"MPB run_zeven, res {cfg['res_inplane']}, "
        f"{cfg['num_bands']} bands, {kp['n_kpoints']} k-points",
        fontsize=11)

    # legend proxies
    handles = [
        plt.Line2D([], [], color="#1f4e79", lw=2.2,
                   label="guided bands 1-2 (below light line)"),
        plt.Line2D([], [], color="#5b8db8", lw=1.6,
                   label="guided bands 3-8"),
        plt.Line2D([], [], color="#8fa6c4", lw=1.0, alpha=0.75,
                   label="same bands above the light line (leaky)"),
        plt.Line2D([], [], color="0.35", lw=1.4, label="air light line"),
        plt.Line2D([], [], color="#c0392b", lw=1.8, ls="--",
                   label="145 GHz cavity line"),
        matplotlib.patches.Patch(facecolor="#f2c14e", alpha=0.30,
                                 label="TE-like gap"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.0))

    fig.tight_layout(rect=(0, 0.085, 1, 1))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    plt.close(fig)

    write_caption(d, ph, cfg, kp, sc, gap, cons, conv)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_CAPTION}")


def write_caption(d, ph, cfg, kp, sc, gap, cons, conv):
    lo, hi = gap["edges_a_over_lambda"]
    lo_g, hi_g = gap["edges_GHz"]
    ref = cons["reference_gap_GHz"]
    dev = cons["deviation_GHz"]
    devp = cons["deviation_pct"]
    xc = d["cross_check_at0_slab3d"]
    tot = sum(t["seconds"] for t in d["timings_seconds"])

    text = f"""\
Figure 2a -- TE-like photonic-crystal-slab band structure at the device lattice constant (out/m2/fig2a_bands.json; scripts/m2_fig2a_bands.py; photonics.json:geometry, modes).

3D MPB band structure of the BULK triangular air-hole lattice in the HR-Si slab at the locked device geometry: r/a = {cfg['r_over_a']}, slab thickness t/a = {cfg['t_over_a']}, eps_core = {cfg['eps_core']}, air cladding, supercell height {cfg['supercell_z_a']} a in z, solved with run_zeven (TE-like: E in-plane, even about the mid-slab z = 0 mirror plane) at in-plane resolution {cfg['res_inplane']} px/a ({cfg['res_z_px']} px across the z supercell), {cfg['num_bands']} bands on {kp['n_kpoints']} k-points along Gamma-M-K-Gamma ({cfg['n_interp_per_seg']} interpolated points per segment). Left axis is the MPB reduced frequency nu = a/lambda (units c/a); the right axis is the same quantity in GHz at the DEVICE lattice constant a = {sc['a_um']} um (photonics.json:geometry.a_um_precise, the cavity-line-fixed a; {sc['GHz_per_unit_nu']:.4f} GHz per unit nu). Note this is a different a from the AT-0 bulk-midgap-centred a = 638.0132 um used in earlier M2 band summaries -- the reduced-frequency band table is scale free, so only the GHz axis moves.

Grey shading is the air light cone f_LL(k) = |k_cart|/(2 pi); modes above it are leaky and are drawn faded. Dark blue are the two lowest guided TE-like bands (band 1 = dielectric band, band 2 = air band) restricted to k-points where they lie below the light line; medium blue are the higher guided bands 3-8.

Gap edges (max of band 1 over guided k / min of band 2 over guided k): lower {lo:.6f} a/lambda at K, upper {hi:.6f} a/lambda at M, i.e. [{lo_g:.4f}, {hi_g:.4f}] GHz at the device a, gap width {hi - lo:.6f} a/lambda = {hi_g - lo_g:.3f} GHz. The 145.000 GHz cavity line (nu_0 = {sc['cavity_line_a_over_lambda']:.6f}, photonics.json:modes.f0_GHz) sits inside the gap with margins +{gap['cavity_margin_lower_GHz']:.2f} GHz above the lower edge and +{gap['cavity_margin_upper_GHz']:.2f} GHz below the upper edge.

Consistency (this is a re-measurement of an already-locked number, not a new one): photonics.json:modes.cavity_line_gap_GHz = [{ref[0]}, {ref[1]}] GHz. Measured here: [{lo_g:.4f}, {hi_g:.4f}] GHz, deviation ({dev[0]:+.4f}, {dev[1]:+.4f}) GHz = ({devp[0]:+.4f}%, {devp[1]:+.4f}%), inside the {cons['tolerance_pct']}% MPB internal-convergence tolerance the project uses; the residual is at the level of the 0.1 GHz rounding with which photonics.json quotes those edges. The gap edges also reproduce the AT-0 step-2 run (out/m2/at0_slab3d.json, {xc['at0_n_interp_per_seg']} interpolated points per segment) to {max(xc['abs_diff_a_over_lambda']):.1e} in a/lambda -- both k-paths contain the K and M corners exactly, where the edges sit, so this is solver repeatability rather than an independent check.

Convergence: repeating the two edge k-points at resolution {conv['resolution']} (x1.5) gives lower edge {conv['lo_edge_res30']:.6f} and upper edge {conv['hi_edge_res30']:.6f} a/lambda, i.e. edge drifts of {conv['lo_drift_rel_pct']:.4f}% (lower) and {conv['hi_drift_rel_pct']:.4f}% (upper) and a midgap drift of {conv['midgap_drift_rel_pct']:.4f}% ({conv['midgap_drift_GHz']:+.4f} GHz). Both edges move down slightly with resolution; the lower edge carries the larger residual grid bias ({conv['lo_drift_rel_pct']:.4f}%, i.e. {abs(conv['lo_edge_res30'] - conv['edge_kpts']['lo_edge_res20']) * d['scale']['GHz_per_unit_nu']:.3f} GHz, right at the {cons['tolerance_pct']}% convergence tolerance), and the midgap is converged to {conv['midgap_drift_rel_pct']:.4f}%. The res-20 numbers are the ones carried by photonics.json and are quoted here unchanged; this residual is the dominant uncertainty on the plotted edges, larger than the {abs(dev[0]):.4f}/{abs(dev[1]):.4f} GHz agreement with photonics.json.

Run: MEEP/MPB {d['versions']['meep_mpb']}, Python {d['versions']['python']}, serial, inside the m2env WSL distro (conda env mp). Deterministic -- no RNG. Total solver time {tot:.1f} s ({d['timings_seconds'][0]['seconds']:.1f} s for the {kp['n_kpoints']}-point res-{cfg['res_inplane']} path, {d['timings_seconds'][1]['seconds']:.1f} s for the 2 res-{conv['resolution']} edge k-points). Checkpoint/resume state: out/m2/fig2a_bands_ck.json.
"""
    OUT_CAPTION.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
