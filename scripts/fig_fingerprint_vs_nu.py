"""M3 Deliverable 3, Figure: AT-3 fractional fingerprint vs filling nu (paper Fig 3b).

Reads the "fingerprint" block of junction.json (nu, w_half, w_half_nA,
alpha_opt, Omega, Gamma) and the per-nu alpha-sweep curves saved by
run_fingerprint_nu.py (out/fingerprint/nu_m2_[4-8].json, the 5 independently
computed fillings; the *_smoke files are ignored).

Panel (a): half-integer step width w_half(nu) at its alpha-optimum, in units
of Ic(nu) (left axis) and in nA (right, twin axis); each point annotated with
Gamma(nu). Mirror-copied points (nu in [-2.3,-2.0], identical to their
symmetric twin -2.5..-2.8 per material.json's nu_opt=-2.4 dome symmetry) are
drawn as open markers to distinguish them from the 5 simulated points.

Panel (b): w_half vs alpha = i_ac/Omega for all five independently simulated
fillings, labelled by nu.

Output: figs/fig_m3_fingerprint_vs_nu.png, dpi=200, Agg backend.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
FP_DIR = ROOT / "out" / "fingerprint"
JUNCTION_PATH = ROOT / "junction.json"
OUT_PATH = ROOT / "figs" / "fig_m3_fingerprint_vs_nu.png"

NU_COMPUTED = [-2.4, -2.5, -2.6, -2.7, -2.8]


def slug_for(nu: float) -> str:
    return f"nu_{nu:+.1f}".replace("+", "p").replace("-", "m").replace(".", "_")


def main():
    junction = json.loads(JUNCTION_PATH.read_text())
    fp = junction["fingerprint"]
    nu = fp["nu"]
    w_half = fp["w_half"]
    w_half_nA = fp["w_half_nA"]
    gamma = fp["Gamma"]

    computed_set = set(NU_COMPUTED)
    is_computed = [n in computed_set for n in nu]

    curves = {}
    for nuc in NU_COMPUTED:
        path = FP_DIR / f"{slug_for(nuc)}.json"
        curves[nuc] = json.loads(path.read_text())

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- panel (a): w_half(nu), left axis = Ic units, right axis = nA ---
    nu_arr = nu
    computed_nu = [n for n, c in zip(nu_arr, is_computed) if c]
    computed_w = [w for w, c in zip(w_half, is_computed) if c]
    mirror_nu = [n for n, c in zip(nu_arr, is_computed) if not c]
    mirror_w = [w for w, c in zip(w_half, is_computed) if not c]

    ax_a.plot(nu_arr, w_half, "-", color="0.6", lw=1.2, zorder=1)
    ax_a.scatter(computed_nu, computed_w, marker="o", s=70, color="C0",
                 edgecolor="k", zorder=3, label="simulated (run_fingerprint_nu.py)")
    ax_a.scatter(mirror_nu, mirror_w, marker="o", s=70, facecolor="white",
                 edgecolor="C0", linewidth=1.6, zorder=3,
                 label="mirror copy (dome symmetric about $\\nu_{opt}=-2.4$)")

    for n, w, g in zip(nu_arr, w_half, gamma):
        ax_a.annotate(f"$\\Gamma$={g:.3f}", (n, w), textcoords="offset points",
                       xytext=(0, 9), ha="center", fontsize=7.5, color="0.25")

    ax_a.set_xlabel(r"filling $\nu$")
    ax_a.set_ylabel(r"$w_{1/2}$  (units of $I_c(\nu)$)")
    ax_a.set_title("(a) half-integer step width vs filling")
    ax_a.grid(alpha=0.3)
    ax_a.legend(loc="upper center", fontsize=8, framealpha=0.9)
    ax_a.set_ylim(0, max(w_half) * 1.35)

    # w_half_nA = w_half * Ic(nu) is not a fixed linear rescale of w_half since
    # Ic(nu) varies across the dome, so the nA values are plotted as their own
    # series on the twin axis rather than via a shared linear map.
    ax_a2 = ax_a.twinx()
    ax_a2.scatter(computed_nu, [w for w, c in zip(w_half_nA, is_computed) if c],
                  marker="^", s=55, color="C3", edgecolor="k", zorder=3)
    ax_a2.scatter(mirror_nu, [w for w, c in zip(w_half_nA, is_computed) if not c],
                  marker="^", s=55, facecolor="white", edgecolor="C3",
                  linewidth=1.6, zorder=3)
    ax_a2.plot(nu_arr, w_half_nA, "--", color="C3", lw=1.0, alpha=0.6, zorder=1)
    ax_a2.set_ylabel(r"$w_{1/2}$  (nA)", color="C3")
    ax_a2.tick_params(axis="y", colors="C3")
    ax_a2.set_ylim(0, max(w_half_nA) * 1.35)

    # --- panel (b): w_half vs alpha for the 5 simulated fillings ---
    cmap = plt.get_cmap("viridis")
    for i, nuc in enumerate(NU_COMPUTED):
        data = curves[nuc]
        alpha = data["alpha"]
        wh = data["w_half"]
        color = cmap(i / (len(NU_COMPUTED) - 1))
        ax_b.plot(alpha, wh, "-o", ms=3.5, lw=1.4, color=color,
                  label=f"$\\nu$={nuc:+.1f}  ($\\Gamma$={data['Gamma']:.3f})")
        ax_b.scatter([data["alpha_opt"]], [data["w_half_max"]], marker="*",
                     s=140, color=color, edgecolor="k", zorder=5)

    ax_b.set_xlabel(r"$\alpha = i_{ac}/\Omega(\nu)$")
    ax_b.set_ylabel(r"$w_{1/2}$  (units of $I_c(\nu)$)")
    ax_b.set_title("(b) half-step width vs drive amplitude, 5 simulated fillings")
    ax_b.grid(alpha=0.3)
    ax_b.legend(loc="upper right", fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        "M3 Deliverable 3 -- AT-3 fractional fingerprint vs filling $\\nu$ "
        "(paper Fig. 3b)\nBeenakker $\\tau$=0.8 CPR ($c_2/c_1=-0.187$), "
        "amplitude via $I_c(\\nu)$ (material.json v0.2)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=200)
    print("inputs:")
    print(f"  {JUNCTION_PATH} (fingerprint block)")
    for nuc in NU_COMPUTED:
        print(f"  {FP_DIR / (slug_for(nuc) + '.json')}")
    print("deterministic: yes (RK4, spp=60, no RNG; figure itself has no randomness)")
    print(f"output: {OUT_PATH}  ({OUT_PATH.stat().st_size} bytes, dpi=200)")


if __name__ == "__main__":
    main()
