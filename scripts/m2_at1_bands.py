"""M2 AT-1: MPB 2D bands of the triangular air-hole lattice vs the PWE table.

Runs INSIDE the m2env WSL distro (conda env mp, meep.mpb). Mirrors
pwe_reference.py conventions: TE = (Ex, Ey, Hz) carries the gap; gap edges =
max(band1) / min(band2) along the Gamma-M-K-Gamma path; a/lambda units.
Acceptance (M2 brief AT-1): both edges within 3% of the PWE table for
eps = 11.68 and eps_eff = 8.12 at r/a = 0.30; TM shows no comparable gap.

Usage: python m2_at1_bands.py <project_root_wsl_path>
Deterministic — no RNG.
"""
import json
import math
import sys

import meep as mp
from meep import mpb

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
R_OVER_A = 0.30
NUM_BANDS = 4
K_INTERP = 16

targets = json.load(open(f"{ROOT}/photonics_targets.json"))
TABLE = {
    11.68: targets["scan"]["bulk eps=11.68"]["0.3"],
    8.12: targets["scan"]["slab n_eff=2.85 (eps=8.12)"]["0.3"],
}

lattice = mp.Lattice(
    size=mp.Vector3(1, 1),
    basis1=mp.Vector3(math.sqrt(3) / 2, 0.5),
    basis2=mp.Vector3(math.sqrt(3) / 2, -0.5),
)
k_points = mp.interpolate(K_INTERP, [
    mp.Vector3(),                    # Gamma
    mp.Vector3(y=0.5),               # M
    mp.Vector3(-1 / 3, 1 / 3),       # K
    mp.Vector3(),                    # Gamma
])


def gap_edges(eps_bg, resolution):
    ms = mpb.ModeSolver(
        geometry_lattice=lattice,
        geometry=[mp.Cylinder(radius=R_OVER_A, material=mp.air)],
        default_material=mp.Medium(epsilon=eps_bg),
        k_points=k_points,
        resolution=resolution,
        num_bands=NUM_BANDS,
    )
    ms.run_te()
    te = [list(f) for f in ms.all_freqs]
    ms.run_tm()
    tm = [list(f) for f in ms.all_freqs]
    te1 = float(max(f[0] for f in te))
    te2 = float(min(f[1] for f in te))
    tm1 = float(max(f[0] for f in tm))
    tm2 = float(min(f[1] for f in tm))
    return {"te_lo": te1, "te_hi": te2,
            "tm_lo": tm1, "tm_hi": tm2,
            "te_bands": [[float(v) for v in f] for f in te],
            "tm_bands": [[float(v) for v in f] for f in tm]}


result = {"r_over_a": R_OVER_A, "k_interp": K_INTERP, "num_bands": NUM_BANDS,
          "cases": {}, "rng": "deterministic - no RNG"}
ok_all = True
for eps, (pwe_lo, pwe_hi) in TABLE.items():
    r32 = gap_edges(eps, 32)
    r48 = gap_edges(eps, 48)
    dev_lo = abs(r32["te_lo"] - pwe_lo) / pwe_lo
    dev_hi = abs(r32["te_hi"] - pwe_hi) / pwe_hi
    mid32 = 0.5 * (r32["te_lo"] + r32["te_hi"])
    mid48 = 0.5 * (r48["te_lo"] + r48["te_hi"])
    tm_gap = max(0.0, r32["tm_hi"] - r32["tm_lo"])
    te_gap = r32["te_hi"] - r32["te_lo"]
    dev_lo, dev_hi = float(dev_lo), float(dev_hi)
    ok = bool(dev_lo <= 0.03 and dev_hi <= 0.03)
    ok_all = bool(ok_all and ok)
    result["cases"][str(eps)] = {
        "pwe_edges": [pwe_lo, pwe_hi],
        "mpb_te_edges_res32": [r32["te_lo"], r32["te_hi"]],
        "mpb_te_edges_res48": [r48["te_lo"], r48["te_hi"]],
        "rel_dev_lo": dev_lo, "rel_dev_hi": dev_hi,
        "midgap_res32": mid32, "midgap_res48": mid48,
        "midgap_convergence": abs(mid48 - mid32) / mid32,
        "tm_gap": tm_gap, "te_gap": te_gap,
        "tm_over_te_gap": (tm_gap / te_gap) if te_gap > 0 else None,
        "within_3pct": ok,
        "te_bands_res32": r32["te_bands"],
        "tm_bands_res32": r32["tm_bands"],
    }
    print(f"eps={eps}: PWE [{pwe_lo:.4f},{pwe_hi:.4f}] "
          f"MPB [{r32['te_lo']:.4f},{r32['te_hi']:.4f}] "
          f"dev=({100*dev_lo:.2f}%,{100*dev_hi:.2f}%) "
          f"TM gap={tm_gap:.4f} -> {'PASS' if ok else 'FAIL'}")

result["AT1_pass"] = ok_all
with open(f"{ROOT}/out/m2/at1_mpb.json", "w") as fh:
    json.dump(result, fh, indent=1)
print("AT-1 overall:", "PASS" if ok_all else "FAIL")
print(f"saved {ROOT}/out/m2/at1_mpb.json")
