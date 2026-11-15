"""M3 deliverable 5: assemble the export file junction.json (brief Sec. 4.5).

Reads (read-only, produced by other M3 agents):
  out/shapiro_map.npz        (d1: full Shapiro dV/dI map, deterministic)
  out/shapiro_convergence.json (d1: spp/averaging convergence study)
  out/noise_ensemble.npz     (d2: Langevin noise ensemble raw arrays)
  out/noise_summary.json     (d2: measured half-step / survival numbers)
  out/above_gap.json         (d4: Tien-Gordon above-gap I-V vs P_abs)
  junction.CONTRACT           (frozen physics contract, junction/__init__.py)

Writes:
  <root>/junction.json  (schema per M3_BRIEF.md Sec. 4.5 / kickoff spec)

`fingerprint` (d3, CPR-from-M1) is left as pending_M1: true because
material.json has not been delivered by M1 yet -- no cpr_harmonics[] vs
filling nu exist to recompute AT-3 against. This is documented, not
fabricated.

No physics is (re)computed here beyond loading arrays already produced and
verified by the other M3 deliverable scripts; this script only assembles and
validates the export.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junction import CONTRACT  # noqa: E402

OUT = ROOT / "out"


def to_list(arr):
    """np.ndarray -> nested python list (JSON-serializable), 0-d -> scalar."""
    a = np.asarray(arr)
    if a.ndim == 0:
        return a.item()
    return a.tolist()


def main():
    # ---------------------------------------------------------------
    # Load inputs
    # ---------------------------------------------------------------
    shapiro = np.load(OUT / "shapiro_map.npz", allow_pickle=True)
    conv = json.loads((OUT / "shapiro_convergence.json").read_text())
    noise_npz = np.load(OUT / "noise_ensemble.npz", allow_pickle=True)
    noise_summary = json.loads((OUT / "noise_summary.json").read_text())
    above_gap = json.loads((OUT / "above_gap.json").read_text())

    # ---------------------------------------------------------------
    # contract block: frozen physics contract + numerics metadata
    # ---------------------------------------------------------------
    contract_block = dict(CONTRACT)
    contract_block["numerics"] = {
        "deterministic": {
            "integrator": "RK4 (vectorized fixed-step)",
            "spp": int(shapiro["spp"]),
            "n_transient_periods": int(shapiro["n_transient"]),
            "n_avg_periods": int(shapiro["n_avg"]),
            "v_resolution": float(shapiro["v_resolution"]),
            "plateau_tol": float(shapiro["tol"]),
            "tol_derivation": "tol = 1.5 * v_resolution, v_resolution = 2*pi/tau_avg = Omega/n_avg",
        },
        "noisy": {
            "integrator": "stochastic-Heun (additive Langevin noise)",
            "spp": int(noise_npz["spp"]),
            "n_transient_periods": int(noise_npz["n_transient"]),
            "n_avg_periods": int(noise_npz["n_avg"]),
            "v_resolution": float(noise_npz["v_resolution"]),
            "plateau_tol": float(noise_npz["tol"]),
            "n_realizations": int(noise_npz["n_real"]),
        },
        "convergence_study": {
            "description": conv["description"],
            "grid": conv["grid"],
            "variants": conv["variants"],
            "pairs": conv["pairs"],
            "tolerance_1000T": conv["tolerance_1000T"],
            "tolerance_1000T_derivation": conv["tolerance_1000T_derivation"],
        },
    }

    # ---------------------------------------------------------------
    # shapiro_map block (d1): production map = skewed CPR; sin is a
    # reference/control map and is only referenced, not inlined again
    # (it already lives in out/shapiro_map.npz).
    # ---------------------------------------------------------------
    shapiro_map_block = {
        "i_ac": to_list(shapiro["i_ac"]),
        "i_dc": to_list(shapiro["i_dc"]),
        "dVdI": to_list(shapiro["dvdi_skew"]),
        "v": to_list(shapiro["v_skew"]),
        "cpr": "skewed [1, -0.25] normalized (c = %s, renorm = %.7f)" % (
            to_list(shapiro["cpr_skew"]), float(CONTRACT["skew_renorm_factor"]),
        ),
        "grid_resolution": "%d x %d" % (len(shapiro["i_ac"]), len(shapiro["i_dc"])),
        "control": {
            "description": "sinusoidal-CPR (AT-1) control map, not re-inlined here",
            "npz_file": "out/shapiro_map.npz",
            "npz_keys": ["v_sin", "dvdi_sin", "cpr_sin"],
        },
        "seed_status": str(shapiro["seed_status"]),
    }

    # ---------------------------------------------------------------
    # fingerprint block (d3): requires material.json from M1, which has
    # not been delivered -- left pending per brief instructions.
    # ---------------------------------------------------------------
    fingerprint_block = {
        "nu": [],
        "w_half": [],
        "i_ac_opt": [],
        "pending_M1": True,
        "note": (
            "requires material.json from M1; contract-CPR values: "
            "w_half=0.250 at i_ac=4.0"
        ),
    }

    # ---------------------------------------------------------------
    # noise block (d2)
    # ---------------------------------------------------------------
    runs = noise_summary["runs"]
    noise_block = {
        "Gamma": float(noise_npz["Gamma_a"]),
        "i_ac": float(noise_npz["i_ac"]),
        "i_dc": to_list(noise_npz["i_dc"]),
        "staircase_rounded": to_list(noise_npz["v_mean_a"]),
        "staircase_std": to_list(noise_npz["v_std_a"]),
        "w_half_surviving": runs["a"]["w_half_ensemble_mean"],
        "survival_fraction": noise_summary["survival_fraction"]["PAPER_NUMBER_a_G0.140"],
        "Gamma_clean": float(noise_npz["Gamma_b"]),
        "staircase_clean": to_list(noise_npz["v_mean_b"]),
        "staircase_clean_std": to_list(noise_npz["v_std_b"]),
        "w_half_clean": runs["b"]["w_half_ensemble_mean"],
        "seeds": {
            "a_skew_G0.140": int(noise_npz["seed_a"]),
            "b_skew_G0.014": int(noise_npz["seed_b"]),
            "c_sin_G0.140": int(noise_npz["seed_c"]),
        },
        "n_realizations": int(noise_npz["n_real"]),
        "w_det_reference": noise_summary["w_det_reference"],
        "w_det_on_grid": noise_summary["w_det_on_grid"],
        "staircase_det": to_list(noise_npz["v_det"]),
        "sin_control": {
            "description": "sinusoidal-CPR control run (c), Gamma=0.140 -- thermal floor sanity check",
            "Gamma": float(noise_npz["Gamma_c"]),
            "seed": int(noise_npz["seed_c"]),
            "staircase_rounded": to_list(noise_npz["v_mean_c"]),
            "staircase_std": to_list(noise_npz["v_std_c"]),
            "w_half": runs["c"]["w_half_ensemble_mean"],
        },
        "dvdi_rounding": noise_summary["dvdi_rounding"],
    }

    # ---------------------------------------------------------------
    # above_gap block (d4): brief schema names the current-map key "IV"
    # (producer script stores it as "I"; renamed here to match schema).
    # ---------------------------------------------------------------
    above_gap_block = {
        "P_abs": above_gap["P_abs"],
        "V": above_gap["V"],
        "IV": above_gap["I"],
        "Delta_of_P": above_gap["Delta_of_P"],
        "params": above_gap["params"],
    }

    # ---------------------------------------------------------------
    # Assemble + write
    # ---------------------------------------------------------------
    export = {
        "contract": contract_block,
        "shapiro_map": shapiro_map_block,
        "fingerprint": fingerprint_block,
        "noise": noise_block,
        "above_gap": above_gap_block,
    }

    out_path = ROOT / "junction.json"
    out_path.write_text(json.dumps(export, indent=2))

    # ---------------------------------------------------------------
    # Validate
    # ---------------------------------------------------------------
    reloaded = json.loads(out_path.read_text())
    required_blocks = ["contract", "shapiro_map", "fingerprint", "noise", "above_gap"]
    for b in required_blocks:
        assert b in reloaded, f"missing block: {b}"
    print("blocks present:", required_blocks)

    def nonempty(name, val):
        ok = len(val) > 0
        print(f"  {name}: len={len(val)} nonempty={ok}")
        assert ok, f"{name} is empty"

    print("contract:")
    print(f"  keys={list(reloaded['contract'].keys())}")
    assert reloaded["contract"]["Omega"] == 5.0
    assert reloaded["contract"]["Ic_nA"] == 30.0
    assert reloaded["contract"]["Gamma_noise_100mK"] == 0.140

    print("shapiro_map:")
    nonempty("i_ac", reloaded["shapiro_map"]["i_ac"])
    nonempty("i_dc", reloaded["shapiro_map"]["i_dc"])
    nonempty("dVdI", reloaded["shapiro_map"]["dVdI"])
    assert len(reloaded["shapiro_map"]["dVdI"]) == len(reloaded["shapiro_map"]["i_ac"])
    assert len(reloaded["shapiro_map"]["dVdI"][0]) == len(reloaded["shapiro_map"]["i_dc"])

    print("fingerprint (pending M1):")
    print(f"  pending_M1={reloaded['fingerprint']['pending_M1']}")
    print(f"  nu={reloaded['fingerprint']['nu']} w_half={reloaded['fingerprint']['w_half']}"
          f" i_ac_opt={reloaded['fingerprint']['i_ac_opt']}")
    assert reloaded["fingerprint"]["pending_M1"] is True

    print("noise:")
    nonempty("i_dc", reloaded["noise"]["i_dc"])
    nonempty("staircase_rounded", reloaded["noise"]["staircase_rounded"])
    nonempty("staircase_std", reloaded["noise"]["staircase_std"])
    nonempty("staircase_clean", reloaded["noise"]["staircase_clean"])
    print(f"  Gamma={reloaded['noise']['Gamma']} w_half_surviving={reloaded['noise']['w_half_surviving']}"
          f" survival_fraction={reloaded['noise']['survival_fraction']}")
    print(f"  Gamma_clean={reloaded['noise']['Gamma_clean']} w_half_clean={reloaded['noise']['w_half_clean']}")
    print(f"  seeds={reloaded['noise']['seeds']}")

    print("above_gap:")
    nonempty("P_abs", reloaded["above_gap"]["P_abs"])
    nonempty("V", reloaded["above_gap"]["V"])
    nonempty("IV", reloaded["above_gap"]["IV"])
    nonempty("Delta_of_P", reloaded["above_gap"]["Delta_of_P"])
    assert len(reloaded["above_gap"]["IV"]) == len(reloaded["above_gap"]["P_abs"])
    assert len(reloaded["above_gap"]["IV"][0]) == len(reloaded["above_gap"]["V"])

    size_bytes = out_path.stat().st_size
    print(f"\nWrote {out_path} ({size_bytes} bytes = {size_bytes / 1024:.1f} KiB)")

    print("\nHeadline numbers embedded:")
    print(f"  contract: Ic=30nA Rn=2kOhm Omega=5.0 f_c=29GHz drive=145GHz Gamma(100mK)=0.140")
    print(f"  shapiro_map grid: {reloaded['shapiro_map']['grid_resolution']}")
    print(f"  noise w_half_surviving(Gamma=0.140)={reloaded['noise']['w_half_surviving']}"
          f" survival_fraction={reloaded['noise']['survival_fraction']}")
    print(f"  noise w_half_clean(Gamma=0.014)={reloaded['noise']['w_half_clean']}")
    print(f"  noise w_det_reference={reloaded['noise']['w_det_reference']}")
    print(f"  above_gap n_P={len(reloaded['above_gap']['P_abs'])} n_V={len(reloaded['above_gap']['V'])}"
          f" Delta0_eV={reloaded['above_gap']['params']['Delta0_eV']}")
    print(f"  fingerprint pending_M1={reloaded['fingerprint']['pending_M1']} (note: {reloaded['fingerprint']['note']})")


if __name__ == "__main__":
    main()
