"""M2 Fig. -- device-specific cavity-line fabrication tolerance at a = 586.74 um.

Pure rendering of out/m2/device_tolerance.json.  No physics is re-derived here;
every number plotted was measured by scripts/m2_device_tolerance.py (cavity
line, 3D FDTD) or reused from out/m2/tolerance.json (bulk gap edges, MPB) or is
analytic (the a-axis, Maxwell scale invariance).  Deterministic - no RNG.

Panel (a)  what moves: signed shift of the CAVITY LINE and of the LOWER GAP
           EDGE for each fabrication perturbation.  The margin only erodes when
           the edge rises faster than the line, so the two bars together are the
           physical content.
Panel (b)  the deliverable: absolute margin (cavity line minus lower gap edge)
           per case, against margin = 0 (the cavity line falls out of the gap)
           and the unperturbed baseline margin.

Input   : out/m2/device_tolerance.json   (tolerates a partial file: cases that
          have not finished yet are simply absent)
Outputs : figs/fig_m2_device_tolerance.png (dpi=200, Agg)
          figs/fig_m2_device_tolerance.caption.txt
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "out" / "m2" / "device_tolerance.json"
OUT_PNG = ROOT / "figs" / "fig_m2_device_tolerance.png"
OUT_CAPTION = ROOT / "figs" / "fig_m2_device_tolerance.caption.txt"

LABEL = {
    "r_hi": r"$r/a$  $+2\%$",
    "r_lo": r"$r/a$  $-2\%$",
    "t_hi": r"$t/a$  $+5\%$",
    "t_lo": r"$t/a$  $-5\%$",
    "stack": "worst $(r,t)$ stack",
}
PLAIN = {"r_hi": "r/a +2%", "r_lo": "r/a -2%", "t_hi": "t/a +5%",
         "t_lo": "t/a -5%", "stack": "worst (r,t) stack"}


def main():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    base = d["baseline"]
    perts = d["perturbations"]
    a_err = d["a_error_analytic"]
    consts = d["constants"]
    a_dev = consts["a_device_um"]
    f_op = consts["f_operating_GHz"]
    dafrac = consts["box"]["da_frac"]

    f0 = base["f0_GHz"]
    m0 = base["margin_to_lo_GHz"]

    # ---- assemble the row list (measured cases + the analytic a rows) -------
    rows = []          # (key, label, df_cav, df_edge, margin)
    for p in perts:
        rows.append((p["case"], LABEL.get(p["case"], p["case"]),
                     p["shift_GHz"], p["edge_shift_GHz"], p["margin_GHz"]))
    for sgn, key in ((-1, "a_minus_2pct"), (+1, "a_plus_2pct")):
        df = a_err["cavity_line_shift_GHz"][key]
        rows.append((key,
                     rf"$a$  ${'+' if sgn > 0 else '-'}{dafrac * 100:.0f}\%$"
                     "\n(analytic)",
                     df, df, a_err["margin_GHz"][key]))
        PLAIN[key] = f"a {'+' if sgn > 0 else '-'}{dafrac*100:.0f}% (analytic)"
        LABEL.setdefault(key, key)

    stack = next((p for p in perts if p["case"] == "stack"), None)
    verdict = d.get("verdict")
    box = d.get("box_scan")
    wcs = d.get("worst_case_stack") or {}
    if verdict:
        wc = verdict["worst_measured_rt_case"]
        wrow = next((p for p in perts if p["case"] == wc), None)
        f_worst = ((wrow["f0_GHz"] if wrow else f0)
                   / (1 + consts["box"]["da_frac"]))
        rows.append(("full_box",
                     "FULL BOX worst\n" + rf"$r-2\%$, $t$ nom, $a+2\%$",
                     f_worst - f0, float("nan"), verdict["min_margin_GHz"]))
        PLAIN["full_box"] = "full box worst (r -2%, t nominal, a +2%)"

    y = np.arange(len(rows))[::-1]
    labels = [r[1] for r in rows]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.4, 5.9),
                                   gridspec_kw={"width_ratios": [1.0, 1.0]})

    # ---------------- panel (a): what moves ---------------------------------
    h = 0.34
    dcav = np.array([r[2] for r in rows], float)
    dedge = np.array([r[3] for r in rows], float)
    axA.barh(y + h / 2, dcav, height=h, color="C0", edgecolor="k", lw=0.5,
             label="cavity line shift  (MEASURED, 3D FDTD)")
    axA.barh(y - h / 2, dedge, height=h, color="C3", edgecolor="k", lw=0.5,
             label="lower gap-edge shift  (bulk study, rescaled)")
    axA.axvline(0.0, color="k", lw=1.0)
    for i, (key, _lab, dc, de, _m) in enumerate(rows):
        yy = y[i]
        axA.text(dc + (0.06 if dc >= 0 else -0.06), yy + h / 2, f"{dc:+.2f}",
                 va="center", ha="left" if dc >= 0 else "right", fontsize=7.2,
                 color="C0")
        if np.isfinite(de):
            axA.text(de + (0.06 if de >= 0 else -0.06), yy - h / 2, f"{de:+.2f}",
                     va="center", ha="left" if de >= 0 else "right", fontsize=7.2,
                     color="C3")
    axA.set_yticks(y)
    axA.set_yticklabels(labels, fontsize=8.2)
    axA.set_xlabel("frequency shift (GHz) at the device $a$ = "
                   rf"{a_dev:.2f} $\mu$m")
    axA.set_title("(a) what moves under the fabrication box", fontsize=9.6)
    axA.grid(axis="x", alpha=0.3)
    axA.legend(loc="lower right", fontsize=7.3, framealpha=0.93)
    xl = np.nanmax(np.abs(np.concatenate([dcav, dedge]))) * 1.45 + 0.2
    axA.set_xlim(-xl, xl)
    # one empty row at the bottom of both panels: keeps the legends and the
    # verdict box off the bars.
    ylim = (y.min() - 1.45, y.max() + 0.55)
    axA.set_ylim(*ylim)

    # ---------------- panel (b): the margin ---------------------------------
    marg = np.array([r[4] for r in rows], float)
    cols = ["C2" if m > 0 else "C3" for m in marg]
    axB.barh(y, marg, height=0.55, color=cols, edgecolor="k", lw=0.5, alpha=0.85)
    axB.axvline(0.0, color="C3", lw=1.8,
                label="margin = 0: cavity line leaves the gap")
    axB.axvline(m0, color="0.35", lw=1.2, ls="--",
                label=rf"unperturbed margin {m0:+.2f} GHz")
    for i, m in enumerate(marg):
        axB.text(m + 0.12, y[i], f"{m:+.2f}", va="center", ha="left",
                 fontsize=7.6, fontweight="bold")
    axB.set_yticks(y)
    axB.set_yticklabels([])
    axB.set_xlabel("margin = cavity line $-$ lower gap edge (GHz)")
    axB.set_title("(b) deliverable: lower-edge margin per case", fontsize=9.6)
    axB.grid(axis="x", alpha=0.3)
    axB.set_xlim(0.0, max(marg.max(), m0) * 1.30)
    axB.set_ylim(*ylim)
    axB.legend(loc="lower right", fontsize=7.3, framealpha=0.93)

    if verdict:
        axB.text(0.03, 0.03,
                 f"worst margin over the full box: {verdict['min_margin_GHz']:+.2f} GHz\n"
                 f"cavity line stays in gap: {verdict['in_gap']}\n"
                 f"binding axis: {PLAIN.get(verdict['binding_axis'], verdict['binding_axis'])}",
                 transform=axB.transAxes, fontsize=8.0, va="bottom", ha="left",
                 bbox=dict(boxstyle="round", fc="white",
                           ec=("C2" if verdict["in_gap"] else "C3"), alpha=0.95))

    fig.suptitle(
        "M2 -- device-specific CAVITY-LINE fabrication tolerance at the device "
        rf"$a$ = {a_dev:.2f} $\mu$m (closing report S5.3 open item)"
        "\n"
        rf"L3 $s$=0.20 $r_{{def}}$=1.15, res 18 + subpixel, lossless 16$a$ "
        rf"ring-down;  baseline line {f0:.3f} GHz, lower edge "
        rf"{base['lo_edge_GHz']:.3f} GHz, margin {m0:+.3f} GHz",
        fontsize=9.8)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    assert OUT_PNG.exists()

    # ---------------- caption ------------------------------------------------
    lines = [
        "Figure -- Device-specific cavity-line fabrication tolerance at the "
        f"device lattice constant a = {a_dev:.3f} um "
        "(out/m2/device_tolerance.json).",
        "",
        "The +-2%/+-5%/+-2% (r/a, t/a, a) box had only ever been evaluated on "
        "the BULK gap at the AT-0 midgap-centred a = "
        f"{consts['a_bulk_study_um']:.2f} um, where 145 GHz sits mid-gap with "
        "~+-15 GHz margins. The DEVICE uses the cavity-line a, where the "
        "cavity line sits only "
        f"{m0:+.2f} GHz above the lower gap edge - so the lower edge, not the "
        "upper, is the binding constraint. This figure closes that item.",
        "",
        f"Baseline (unperturbed): cavity line nu0 = {base['nu0']:.7f} -> "
        f"{f0:.3f} GHz, lower TE gap edge {base['lo_edge_GHz']:.3f} GHz, upper "
        f"edge {base['hi_edge_GHz']:.3f} GHz; margin to the lower edge "
        f"{m0:+.3f} GHz. The re-run reproduces the stored cavity line to "
        f"{abs(base['dev_vs_stored_deep_GHz']):.4f} GHz "
        f"({abs(base['dev_vs_stored_deep_nu']):.2e} in nu).",
        "",
        "Panel (a): signed shift of the cavity line (blue, MEASURED by new 3D "
        "FDTD runs at res 18 + subpixel averaging in the established isolated-"
        "L3 16a supercell) and of the lower gap edge (red, REUSED from the "
        "bulk tolerance study out/m2/tolerance.json - MPB run_zeven, res 20, "
        "K/M edge k-points - transferred to the device a by Maxwell scale "
        "invariance f = nu c / a; no bulk band structure was re-run). The two "
        "bars move in the same direction on every axis: the perturbations "
        "blue- or red-shift the whole photonic structure together, so the "
        "margin erodes only by the DIFFERENCE of the two bars.",
        "",
        "Panel (b): the deliverable. Margin = cavity line - lower gap edge, "
        "per case. Green bars are in-gap (margin > 0); the red vertical line "
        "at 0 is the failure boundary.",
    ]
    for p in perts:
        lines.append(
            f"  {PLAIN.get(p['case'], p['case']):>26}: line {p['f0_GHz']:.3f} GHz "
            f"({p['shift_GHz']:+.3f}), edge {p['lo_edge_GHz']:.3f} GHz "
            f"({p['edge_shift_GHz']:+.3f}), margin {p['margin_GHz']:+.3f} GHz "
            f"({p['margin_shift_GHz']:+.3f} vs baseline), in_gap={p['in_gap']}")
    lines += [
        "",
        "The a-axis is analytic, not simulated: nu = a/lambda depends only on "
        "(r/a, t/a, eps), so a -> a(1+da) sends every frequency to f/(1+da). A "
        f"+-{dafrac*100:.0f}% error in a therefore moves the {f_op:.0f} GHz "
        f"line by "
        f"{a_err['cavity_line_shift_GHz']['a_plus_2pct']:+.2f} / "
        f"{a_err['cavity_line_shift_GHz']['a_minus_2pct']:+.2f} GHz, but moves "
        "the gap edges by the SAME factor - so it changes the margin only by "
        f"{a_err['margin_shift_GHz']['a_plus_2pct']:+.2f} / "
        f"{a_err['margin_shift_GHz']['a_minus_2pct']:+.2f} GHz. The a-error is "
        "a cavity/qubit DETUNING axis, not an in-gap axis.",
    ]
    if verdict and stack is not None:
        lines += [
            "",
            "KEY STRUCTURAL RESULT: the margin is NOT monotonic across the box. "
            "Three of the four single perturbations (r +2%, t +5%, t -5%) move "
            "the cavity line MORE than they move the lower gap edge and "
            "therefore INCREASE the margin; only r -2% erodes it. The L3 "
            "donor mode is derived from, and tracks, the dielectric band edge "
            "that forms the lower gap boundary, so the mode's binding depth in "
            "the gap is far more robust than either frequency separately. "
            "Consequently the box minimum is NOT at a corner: on the t axis "
            "the measured margin shifts are "
            f"{next(p['margin_shift_GHz'] for p in perts if p['case'] == 't_lo'):+.4f} "
            f"GHz at t -5% and "
            f"{next(p['margin_shift_GHz'] for p in perts if p['case'] == 't_hi'):+.4f} "
            "GHz at t +5%, both positive, so a quadratic through the three "
            f"measured t points puts the in-box t minimum at "
            f"{box['t_axis']['argmin_pct']:+.2f}% "
            f"({box['t_axis']['min_margin_shift_GHz']:+.4f} GHz) - i.e. "
            "essentially at NOMINAL thickness. On the r axis the response is "
            f"concave, so its minimum is the endpoint r -2% "
            f"({box['r_axis']['min_margin_shift_GHz']:+.4f} GHz).",
            "",
            "Worst case over the full box is therefore r/a -2% at NOMINAL t/a, "
            "with the analytic a +2% factor applied: minimum margin "
            f"{verdict['min_margin_GHz']:+.3f} GHz "
            f"({verdict['min_margin_rt_only_GHz']:+.3f} GHz on (r,t) alone), "
            f"i.e. only {verdict['margin_erosion_GHz']:.3f} GHz "
            f"({verdict['margin_erosion_pct_of_baseline']:.1f}%) of the "
            f"{m0:.3f} GHz baseline margin is consumed. The cavity line stays "
            f"inside the gap across the whole box: {verdict['in_gap']}. "
            f"Binding axis: {PLAIN.get(verdict['binding_axis'], verdict['binding_axis'])}.",
            "",
            "The 'worst (r,t) stack' row is a CROSS-CHECK, not the worst point: "
            f"it is the corner {stack['params']['stack_from']} that linear "
            "superposition of the single-axis margin shifts predicts to be the "
            "worst corner, run directly to test that superposition. Predicted "
            f"margin shift {wcs['predicted_margin_shift_linear_GHz']:+.4f} GHz, "
            f"measured {wcs['measured_margin_shift_GHz']:+.4f} GHz, residual "
            f"{wcs['superposition_residual_GHz']:+.4f} GHz - superposition "
            "holds to ~0.02 GHz over the full box, but its worst CORNER "
            f"({stack['margin_GHz']:+.3f} GHz) is milder than the worst EDGE "
            f"point r -2% ({verdict['worst_measured_rt_margin_GHz']:+.3f} GHz). "
            "A corner-only worst-case search - the method used for the bulk-gap "
            "box in out/m2/tolerance.json - would have missed the true minimum "
            "here.",
        ]
    else:
        lines += ["", "NOTE: partial data - the worst-case stack has not "
                      "finished; verdict pending."]
    lines += [
        "",
        "Caveat stated for the record: the 16a supercell's Q ~ 8.6e3 is a known "
        "lateral-leakage artifact (physical ceiling 2.7-3.0e4, decisions.md "
        "2026-07-24 AT-2 CLOSED). It is irrelevant to this figure, which "
        "reports only FREQUENCY shifts; the mode frequency is converged in this "
        "cell, and it is the same cell that defined the stored nu0 fixing the "
        "device a, so every case is differenced against its own baseline.",
    ]
    OUT_CAPTION.write_text("\n".join(lines), encoding="utf-8")

    print(f"input : {SRC}  (status={d['status']})")
    print("RNG   : none (deterministic)")
    print(f"output: {OUT_PNG}  ({OUT_PNG.stat().st_size} bytes, dpi=200)")
    print(f"output: {OUT_CAPTION}  ({OUT_CAPTION.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
