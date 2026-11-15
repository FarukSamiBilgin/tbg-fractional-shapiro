"""M2 Fig. 2c -- |E|^2 mode map of the tuned L3 cavity, junction site marked.

Renders the stored steady-state field map from the isolated 3D L3 cavity
run (figure_plan.md Fig. 2c, priority #4 in "Outstanding plot work" --
field data stored, plot missing). Pure rendering of a stored MEEP
mid-slab field slice -- no physics is re-derived and no RNG is involved
(rng: "deterministic - no RNG" per the source artifact's own header).

Inputs (read-only):
  out/m2/l3_3d_final_s020_rdef115_res18_Efield.npz  -- x, y, E2_mid, eps_mid (mid-slab z-index)
  photonics.json                                     -- modes.V_mode_a3/um3, geometry.a_um_precise

Output:
  figs/fig_m2_modemap.png (dpi=200, Agg)
  figs/fig_m2_modemap.caption.txt
"""
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ROOT = pathlib.Path(__file__).resolve().parents[1]
FIELD_PATH = ROOT / "out" / "m2" / "l3_3d_final_s020_rdef115_res18_Efield.npz"
PHOTONICS_PATH = ROOT / "photonics.json"
OUT_PNG = ROOT / "figs" / "fig_m2_modemap.png"
OUT_CAPTION = ROOT / "figs" / "fig_m2_modemap.caption.txt"


def main():
    d = np.load(FIELD_PATH)
    x = d["x"]                      # src: l3_3d_final_..._Efield.npz:x (units of lattice constant a)
    y = d["y"]                      # src: l3_3d_final_..._Efield.npz:y
    E2 = d["E2_mid"]                # src: l3_3d_final_..._Efield.npz:E2_mid  (|E|^2, mid-slab z)
    eps = d["eps_mid"]              # src: l3_3d_final_..._Efield.npz:eps_mid
    z_index = int(d["z_index"])
    a_um_run = float(d["a_um"])     # a used INSIDE this MEEP run (AT-0/AT-1 bulk-midgap a)
    fcen = float(d["fcen"])
    res = int(d["res"])

    ph = json.loads(PHOTONICS_PATH.read_text(encoding="utf-8"))
    a_um_device = ph["geometry"]["a_um_precise"]                 # src: photonics.json:geometry.a_um_precise
    V_mode_a3 = ph["modes"]["V_mode_a3"]                          # src: photonics.json:modes
    V_mode_um3 = ph["modes"]["V_mode_um3"]

    E2_norm = E2 / E2.max()
    vmin = max(E2_norm[E2_norm > 0].min(), 1e-4)

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    mesh = ax.pcolormesh(x, y, E2_norm.T, shading="auto", cmap="inferno",
                          norm=mcolors.LogNorm(vmin=vmin, vmax=1.0))
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(r"$|E|^2$ / max$|E|^2$  (log scale, mid-slab)")

    # lattice/hole outline from the binary eps map (air holes vs slab eps_core)
    ax.contour(x, y, eps.T, levels=[(1.0 + ph["geometry"]["eps_core"]) / 2.0],
                colors="C0", linewidths=0.7, alpha=0.75)

    # junction site: cavity centre
    ax.scatter([0], [0], marker="+", s=260, color="cyan", linewidth=2.2, zorder=5,
               label="junction site (cavity centre)")

    ax.set_xlabel(r"$x$ (units of lattice constant $a$)")
    ax.set_ylabel(r"$y$ (units of lattice constant $a$)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.set_title(
        f"M2 Fig. 2c -- tuned L3 cavity $|E|^2$ mode map "
        f"(mid-slab, z-index={z_index}, res={res})\n"
        rf"$V_{{mode}}$={V_mode_a3:.4f} $a^3$=${V_mode_um3:.3e}\,\mu m^3$ "
        f"at device $a$={a_um_device:.2f} $\\mu$m (scale-invariant in "
        "units of $a$)",
        fontsize=9.6,
    )
    fig.tight_layout(rect=(0, 0, 0.98, 1))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=200)
    assert OUT_PNG.exists()

    # ---- caption ----
    caption_lines = [
        "Figure 2c -- |E|^2 mode map of the tuned L3 cavity, junction site "
        "marked (out/m2/l3_3d_final_s020_rdef115_res18_Efield.npz; "
        "photonics.json:modes, geometry).",
        "",
        f"Mid-slab (z-index={z_index} of the res={res} grid) steady-state "
        "|E|^2 field intensity, normalized to its own maximum and shown on "
        "a log color scale to expose both the strongly-confined cavity "
        "mode and its weaker in-plane leakage into the surrounding lattice "
        "(hole outline overlaid in blue, contoured at the air/slab "
        f"eps midpoint, eps_core={ph['geometry']['eps_core']}). x and y "
        "are in units of the lattice constant a (MEEP normalized units, "
        "scale-invariant): the field pattern shown is unchanged whether "
        f"a is set to the value used inside this run "
        f"(a={a_um_run:.4f} um, the AT-0/AT-1 bulk-midgap lattice constant "
        "used for this isolated-cavity MEEP run) or rescaled to the "
        f"final device value a={a_um_device:.4f} um "
        "(photonics.json:geometry.a_um_precise, the cavity-line-fixed "
        "lattice constant). The junction site (cyan cross) sits at the "
        "cavity centre (x=y=0), coincident with the field maximum.",
        "",
        f"Mode volume: V_mode = {V_mode_a3:.4f} a^3 = {V_mode_um3:.6e} "
        r"um^3 at the device a (V_mode convention: "
        "integral(eps|E|^2 dV) / max(eps|E|^2); "
        f"photonics.json:modes.V_mode_convention). Run parameters: "
        f"drive fcen={fcen:.3f} (reduced frequency), isolated 3D L3 "
        "cavity (s=0.20, rdef=1.15), from the AT-2 lossy res18 winner "
        "(out/m2/l3_3d_final_s020_rdef115_res18.json).",
    ]
    OUT_CAPTION.write_text("\n".join(caption_lines), encoding="utf-8")

    # ---- console report ----
    print("inputs:")
    print(f"  {FIELD_PATH}")
    print(f"  {PHOTONICS_PATH} (modes.V_mode_a3/um3, geometry.a_um_precise, geometry.eps_core)")
    print('RNG: none (rng="deterministic - no RNG" in the source MEEP run)')
    print("outputs:")
    print(f"  {OUT_PNG}  ({OUT_PNG.stat().st_size} bytes, dpi=200)")
    print(f"  {OUT_CAPTION}  ({OUT_CAPTION.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
