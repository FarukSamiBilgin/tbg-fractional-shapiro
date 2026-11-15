"""M4 milestone-1 driver: export integration.json + figs/fig_m4_power_chain.png.

Closes the loop between the M2 photonic cavity (via the locked TCMT/power-
budget relations in ``integration/__init__.py``) and the M3 junction export
``junction.json``, using only the reusable modules already validated in
``integration/`` (power_chain.py, backaction.py, thermal.py) -- this script
performs no new physics, it assembles and exports what those modules compute
(M4_BRIEF_DRAFT.md deliverable 5, Sec. 5 "out of scope": no EM/junction-dynamics
recomputation).

Deterministic; no RNG. Windows/Agg-safe (matplotlib backend set before pyplot
import). Run via::

    python scripts/run_integration.py

Outputs
-------
* ``integration.json``       -- power_chain, operating_points, backaction,
                                 thermal, pending_M2, params, provenance blocks.
* ``figs/fig_m4_power_chain.png`` -- 3-panel summary figure (dpi=200).
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integration import (  # noqa: E402
    A_TARGET,
    E_CHARGE,
    F0_HZ,
    H_PLANCK,
    HBAR,
    IC_CONTRACT_A,
    PENDING_M2,
    P_IN_REF_W,
    P_J_REF_W,
    QW_OVER_QI,
    RN_OHM,
    V_PHOTON,
)
from integration.power_chain import (  # noqa: E402
    absorption_from_Q,
    alpha_from_pin,
    operating_points,
    v_ac_from_pj,
)
from integration.backaction import (  # noqa: E402
    delta_vs_load,
    detuning_fraction,
    threshold_load,
)
from integration.thermal import (  # noqa: E402
    P_FRIDGE_DEFAULT_W,
    chip_budget,
    te_from_power,
)

JUNCTION_JSON = ROOT / "junction.json"
OUT_JSON = ROOT / "integration.json"
FIG_PATH = ROOT / "figs" / "fig_m4_power_chain.png"

# --------------------------------------------------------------------------- #
# Free-parameter defaults (documented, M4_BRIEF_DRAFT.md Sec. 2)
# --------------------------------------------------------------------------- #
# Back-action: reduced/normalized units (eta_abs=tau_rec=n0=Delta0=1). The
# reduced gap Delta/Delta0 depends on these only through the dimensionless
# load L = eta_abs*P_abs*tau_rec/(n0*Delta0) (verified to 1e-12 in
# tests/test_backaction_thermal.py), so this choice is representative of any
# (eta_abs, tau_rec, n0, Delta0) at the same L.
BACKACTION_DEFAULTS = dict(eta_abs=1.0, tau_rec=1.0, n0=1.0, Delta0=1.0)
Q_VALUES = (1.0e3, 1.0e4, 1.0e5)

# Thermal: graphene-like electron-phonon cooling, P = Sigma_V*(Te^delta - Tph^delta).
SIGMA_V_W_PER_K5 = 1.0e-8   # W/K^5, free parameter (lumped Sigma_eph * volume);
# v0.3: junction volume x10 -> Sigma_V x10 (DECISION_contract_v0p3.md), so
# T_e(900 pW, 1e-8) = T_e(90 pW, 1e-9) = 0.618 K — self-heating risk unchanged.
DELTA_EXP = 5               # clean-graphene acoustic-phonon exponent (default)
T_PHONON_K = 0.1            # 100 mK operating point


def _to_jsonable(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


# --------------------------------------------------------------------------- #
# Block builders
# --------------------------------------------------------------------------- #
def build_power_chain():
    """Locked constants, the 9 nW reference operating point, and the TCMT
    absorption identity at the locked Q_w/Q_i = 200 target, with formulas as
    strings + sources (M4_BRIEF_DRAFT.md Sec. 5, deliverable 5)."""
    alpha_9nW = float(alpha_from_pin(P_IN_REF_W))
    V_ac_mV = float(v_ac_from_pj(P_J_REF_W, RN_OHM)) / 1e-3
    A_exact_at_200 = float(absorption_from_Q(1.0, QW_OVER_QI))
    approx = 2.0 / QW_OVER_QI
    A_rel_err = abs(A_exact_at_200 - approx) / approx

    return {
        "locked_constants": {
            "E_CHARGE_C": E_CHARGE,
            "H_PLANCK_Js": H_PLANCK,
            "HBAR_Js": HBAR,
            "F0_Hz": F0_HZ,
            "V_PHOTON_mV": V_PHOTON / 1e-3,
            "RN_OHM": RN_OHM,
            "IC_CONTRACT_nA": IC_CONTRACT_A / 1e-9,
            "A_TARGET": A_TARGET,
            "QW_OVER_QI": QW_OVER_QI,
            "P_IN_REF_nW": P_IN_REF_W / 1e-9,
            "P_J_REF_pW": P_J_REF_W / 1e-12,
        },
        "alpha_9nW": alpha_9nW,
        "V_ac_mV": V_ac_mV,
        "A_exact_at_200": A_exact_at_200,
        "A_exact_at_200_closed_form": "400/201^2 = 0.00990075",
        "A_rel_err_vs_1pct_approx": A_rel_err,
        "P_J_ref_pW": P_J_REF_W / 1e-12,
        "formulas": {
            "tcmt_on_resonance": {
                "T": "(gamma_i/(gamma_i+gamma_w))**2",
                "R": "(gamma_w/(gamma_i+gamma_w))**2",
                "A": "2*gamma_i*gamma_w/(gamma_i+gamma_w)**2 = 2*Q_i*Q_w/(Q_i+Q_w)**2",
                "source": "M2_BRIEF.md Sec. 2 (lines 41-46); M4_BRIEF_DRAFT.md Sec. 1 line 19",
            },
            "absorption_target": {
                "formula": "A=1.0% => Q_w = 200*Q_i (undercoupled A ~ 2*Q_i/Q_w); exact A(200) = 400/201^2 = 0.99007%",
                "source": "M2_BRIEF.md line 46; M4_BRIEF_DRAFT.md Sec. 1 line 20",
            },
            "power_budget": {
                "formula": "P_in -> P_J = A*P_in -> V_ac = sqrt(2*P_J*Rn) -> alpha = e*V_ac/(hbar*omega)",
                "reference_point": "P_in=9 nW -> P_J=90 pW -> V_ac=0.6 mV -> alpha=1.0",
                "source": "M2_BRIEF.md lines 52-53; M4_BRIEF_DRAFT.md Sec. 1 line 21",
            },
            "photon_voltage": {
                "formula": "V_PHOTON = h*f0/e = hbar*omega/e",
                "value_mV_at_145GHz": V_PHOTON / 1e-3,
                "source": "M3 contract (hbar/2e = 300 uV); M4_BRIEF_DRAFT.md Sec. 1 line 22",
            },
            "drive_mapping": {
                "formula": "alpha = e*V_ac/(hbar*omega) ~ i_ac/Omega  =>  i_ac = alpha*Omega",
                "source": "M3_BRIEF.md Sec. 2 line 34; M4_BRIEF_DRAFT.md Sec. 1 line 23",
            },
            "junction_scaling": {
                "formula": "Omega(nu) = 5*30/Ic(nu) [nA];  Gamma(nu) = 0.14*30/Ic(nu) [nA]",
                "source": "M3_BRIEF.md contract + junction.json; M4_BRIEF_DRAFT.md Sec. 1 line 24",
            },
        },
    }


def build_operating_points():
    """Per-filling operating-point table, extended with Ic(nu) (M4_BRIEF_DRAFT.md
    deliverable 2). Ic(nu) is derived from the locked relation
    Omega(nu) = 5*30/Ic(nu) [nA] (M4_BRIEF_DRAFT.md Sec. 1 line 24) using the
    contract reference (Omega_ref=5.0, Ic_ref=30 nA) already exported in
    junction.json; it is cross-checked against the independent M3 export ratio
    w_half_nA/w_half."""
    rows = operating_points(str(JUNCTION_JSON))

    with open(JUNCTION_JSON, "r", encoding="utf-8") as fh:
        contract = json.load(fh)["contract"]
    Omega_ref = float(contract["Omega"])          # 5.0
    Ic_ref_nA = float(contract["Ic_nA"])           # 30.0 nA

    nu, Ic_nA, Omega, Gamma = [], [], [], []
    alpha_opt, i_ac_opt, w_half_Ic, w_half_nA = [], [], [], []
    P_in_opt_nW, P_J_opt_pW, V_ac_opt_mV = [], [], []

    for row in rows:
        Ic_here = Ic_ref_nA * Omega_ref / row["Omega"]
        # cross-check against the independently exported ratio (M3 fingerprint)
        Ic_cross = row["w_half_nA"] / row["w_half"]
        assert abs(Ic_here - Ic_cross) / Ic_cross < 1e-9, (
            f"Ic(nu={row['nu']}) mismatch: {Ic_here} vs {Ic_cross}"
        )
        nu.append(row["nu"])
        Ic_nA.append(Ic_here)
        Omega.append(row["Omega"])
        Gamma.append(row["Gamma"])
        alpha_opt.append(row["alpha_opt"])
        i_ac_opt.append(row["i_ac_opt"])
        w_half_Ic.append(row["w_half"])
        w_half_nA.append(row["w_half_nA"])
        P_in_opt_nW.append(row["P_in_opt_nW"])
        P_J_opt_pW.append(row["P_J_opt_pW"])
        V_ac_opt_mV.append(row["V_ac_opt_mV"])

    return {
        "nu": nu,
        "Ic_nA": Ic_nA,
        "Omega": Omega,
        "Gamma": Gamma,
        "alpha_opt": alpha_opt,
        "i_ac_opt": i_ac_opt,
        "w_half_Ic": w_half_Ic,
        "w_half_nA": w_half_nA,
        "P_in_opt_nW": P_in_opt_nW,
        "P_J_opt_pW": P_J_opt_pW,
        "V_ac_opt_mV": V_ac_opt_mV,
        "source": "junction.json fingerprint block (M3 export) + locked power chain "
                  "(M4_BRIEF_DRAFT.md Sec. 1 line 24, Sec. 3 AT-M4-3); Ic(nu) derived, "
                  "cross-checked to <1e-9 rel against w_half_nA/w_half",
    }


def build_backaction():
    """Back-action envelope over a log-spaced load grid, plus the per-Q
    half-linewidth threshold loads (AT-M4-4, M4_BRIEF_DRAFT.md deliverable 3)."""
    L_report = np.array([1e-4, 1e-3, 1e-2, 0.1])
    L_grid = np.sort(np.unique(np.concatenate([np.logspace(-6, -0.5, 56), L_report])))

    d_over_d0 = delta_vs_load(L_grid, **BACKACTION_DEFAULTS)
    detuning = detuning_fraction(L_grid, **BACKACTION_DEFAULTS)
    thr = {f"{Q:.0e}": float(threshold_load(Q)) for Q in Q_VALUES}

    return {
        "L_grid": L_grid,
        "Delta_over_Delta0": d_over_d0,
        "detuning_fraction": detuning,
        "threshold_load_by_Q": thr,
        "L_report_points": {
            f"L_{L:g}": {
                "Delta_over_Delta0": float(delta_vs_load(L, **BACKACTION_DEFAULTS)),
                "detuning_fraction": float(detuning_fraction(L, **BACKACTION_DEFAULTS)),
            }
            for L in L_report
        },
        "reduced_units_note": "eta_abs=tau_rec=n0=Delta0=1 (reduced units); the reduced "
                               "gap depends on these only through L (L-invariance verified "
                               "to 1e-12 in tests/test_backaction_thermal.py)",
        "source": "junction.tien_gordon.steady_state_delta (M3 D4, reused via "
                  "integration.backaction); M4_BRIEF_DRAFT.md AT-M4-4",
    }


def build_thermal():
    """Chip-level and junction-local thermal budget (M4_BRIEF_DRAFT.md
    deliverable 4)."""
    P_grid = np.logspace(-13, -8, 60)   # 0.1 pW .. 10 nW, spans P_J_REF_W = 90 pW
    Te = te_from_power(P_grid, SIGMA_V_W_PER_K5, T_PHONON_K, delta_exp=DELTA_EXP)
    cb = chip_budget(P_J_REF_W, P_fridge=P_FRIDGE_DEFAULT_W)

    return {
        "P_grid_W": P_grid,
        "T_e_K": Te,
        "params": {
            "Sigma_V_W_per_K5": SIGMA_V_W_PER_K5,
            "delta_exp": DELTA_EXP,
            "T_phonon_K": T_PHONON_K,
        },
        "chip_budget": {
            "P_J_pW": P_J_REF_W / 1e-12,
            "P_fridge_uW": P_FRIDGE_DEFAULT_W / 1e-6,
            "ratio": cb.ratio,
            "ok": cb.ok,
        },
        "cooling_law": "P = Sigma_V*(Te^delta - Tph^delta)  =>  Te = (P/Sigma_V + Tph^delta)^(1/delta)",
        "source": "integration.thermal (graphene-like e-ph cooling, parametric); "
                  "M4_BRIEF_DRAFT.md Sec. 2, 4",
    }


def build_pending_m2():
    return [
        {
            "name": name,
            "note": "not available: photonics.json does not exist (M2 not run); "
                     "any absolute value depending on this stays parametric",
        }
        for name in PENDING_M2
    ]


def build_params():
    return {
        "photonics_pending_M2": list(PENDING_M2),
        "quasiparticle_backaction": {
            **BACKACTION_DEFAULTS,
            "Q_values": list(Q_VALUES),
            "note": "reduced units (M3 D4 free parameters eta_abs, tau_rec, n0, "
                     "Delta0); L-invariant so absolute values are representative",
        },
        "thermal": {
            "Sigma_V_W_per_K5": SIGMA_V_W_PER_K5,
            "delta_exp": DELTA_EXP,
            "T_phonon_K": T_PHONON_K,
            "P_fridge_W": P_FRIDGE_DEFAULT_W,
            "note": "Sigma_V is the lumped Sigma_eph * junction-electron-volume "
                     "product (both free, undetermined); delta_exp default 5 "
                     "(clean graphene); P_fridge order-of-magnitude 500 uW at "
                     "the mixing-chamber stage",
        },
    }


def build_provenance():
    return {
        "junction_json": {
            "path": "junction.json",
            "version_note": "M3 export v1; fingerprint.pending_M1 = False "
                             "(material.json v0.2 consumed); 9 fillings "
                             "(5 independently simulated, 4 dome-symmetric "
                             "mirror copies) -- see junction.json fingerprint.symmetry_note",
        },
        "M4_BRIEF_DRAFT_status": "DRAFT authored by the M3/M4 agent on 2026-07-23; "
                                   "no ratified M4_BRIEF.md exists. Every locked number "
                                   "used here is inherited verbatim from the M2 or M3 "
                                   "briefs (M4_BRIEF_DRAFT.md Sec. 1); nothing new is "
                                   "locked. To be ratified via paper/decisions.md.",
        "generator": "scripts/run_integration.py",
    }


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def make_figure(power_chain, ops, backaction, thermal):
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(17, 5.0))

    # --- panel (a): P_in vs alpha, quadratic curve + the five nu operating points ---
    alpha_curve = np.linspace(0.0, 1.0, 200)
    P_in_curve_nW = power_chain["locked_constants"]["P_IN_REF_nW"] * alpha_curve ** 2
    ref_nW = power_chain["locked_constants"]["P_IN_REF_nW"]
    ax_a.plot(alpha_curve, P_in_curve_nW, "-", color="0.3", lw=1.6,
              label=rf"$P_{{in}}={ref_nW:.0f}\,\mathrm{{nW}}\cdot\alpha^2$")

    # Mark the 5 independently-simulated fillings (M3 report NU_COMPUTED =
    # [-2.4,-2.5,-2.6,-2.7,-2.8]; junction.json fingerprint.symmetry_note); each
    # carries its dome-symmetric mirror twin's label since the mirror copy is
    # bit-identical in (alpha, P_in). Two of the five coincide exactly on this
    # 2D (alpha, P_in) projection (nu=-2.4 & -2.6 both alpha=0.7; nu=-2.7 & -2.8
    # both alpha=0.9) -- both markers are drawn at their true physical location
    # and the labels are stacked so both remain legible.
    nu_arr = ops["nu"]
    alpha_arr = ops["alpha_opt"]
    pin_arr = ops["P_in_opt_nW"]
    by_nu = {n: (a, p) for n, a, p in zip(nu_arr, alpha_arr, pin_arr)}
    NU_COMPUTED = [-2.4, -2.5, -2.6, -2.7, -2.8]
    MIRROR_OF = {-2.5: -2.3, -2.6: -2.2, -2.7: -2.1, -2.8: -2.0}
    cmap = plt.get_cmap("viridis")
    label_offsets_by_key = {}
    for i, nu_c in enumerate(NU_COMPUTED):
        a_i, p_i = by_nu[nu_c]
        color = cmap(i / (len(NU_COMPUTED) - 1))
        label = f"{nu_c:+.1f}"
        if nu_c in MIRROR_OF:
            label += f"/{MIRROR_OF[nu_c]:+.1f}"
        ax_a.scatter([a_i], [p_i], marker="o", s=90, color=color,
                     edgecolor="k", zorder=5)
        key = round(a_i, 6)
        stack = label_offsets_by_key.setdefault(key, 0)
        ax_a.annotate(rf"$\nu$={label}", (a_i, p_i), textcoords="offset points",
                      xytext=(7, -4 + 11 * stack), fontsize=7.5, color=color)
        label_offsets_by_key[key] += 1
    p_opts = list(ops["P_in_opt_nW"])
    ax_a.axhspan(min(p_opts), max(p_opts), color="C0", alpha=0.08, zorder=0,
                 label=rf"$P_{{in,opt}}$ band {min(p_opts):.1f}-{max(p_opts):.1f} nW")
    ax_a.set_xlabel(r"drive strength $\alpha=eV_{ac}/\hbar\omega$")
    ax_a.set_ylabel(r"$P_{in}$ (nW)")
    ax_a.set_title("(a) power chain: $P_{in}$ vs $\\alpha$")
    ax_a.grid(alpha=0.3)
    ax_a.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    # --- panel (b): Delta/Delta0, detuning vs load L (log x), Q thresholds ---
    L = np.asarray(backaction["L_grid"])
    ax_b.plot(L, backaction["Delta_over_Delta0"], "-", color="C0", lw=1.6,
              label=r"$\Delta/\Delta_0$")
    ax_b.plot(L, backaction["detuning_fraction"], "--", color="C3", lw=1.6,
              label=r"detuning fraction $1-\Delta/\Delta_0$")
    ax_b.set_xscale("log")
    for Q_str, L_thr in backaction["threshold_load_by_Q"].items():
        Q_val = float(Q_str)
        half_lw = 1.0 / (2.0 * Q_val)
        ax_b.axvline(L_thr, color="0.5", lw=0.9, ls=":")
        ax_b.axhline(half_lw, color="0.5", lw=0.6, ls=":")
        ax_b.annotate(f"Q={Q_str}", (L_thr, half_lw), textcoords="offset points",
                      xytext=(4, 6), fontsize=7, color="0.3", rotation=90)
    ax_b.set_xlabel(r"load $L=\eta_{abs}P_{abs}\tau_{rec}/(n_0\Delta_0)$")
    ax_b.set_ylabel("reduced gap / detuning fraction")
    ax_b.set_title("(b) back-action envelope vs load")
    ax_b.grid(alpha=0.3, which="both")
    ax_b.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    # --- panel (c): T_e vs P, P_J = 90 pW marked ---
    P_grid = np.asarray(thermal["P_grid_W"])
    Te = np.asarray(thermal["T_e_K"])
    ax_c.plot(P_grid, Te, "-", color="C2", lw=1.6)
    ax_c.set_xscale("log")
    P_J_here = thermal["chip_budget"]["P_J_pW"] * 1e-12
    Te_at_PJ = float(te_from_power(P_J_here, SIGMA_V_W_PER_K5, T_PHONON_K, delta_exp=DELTA_EXP))
    ax_c.scatter([P_J_here], [Te_at_PJ], marker="*", s=180, color="C3",
                 edgecolor="k", zorder=5,
                 label=rf"$P_J$={thermal['chip_budget']['P_J_pW']:.0f} pW $\to$ $T_e$={Te_at_PJ:.3f} K")
    ax_c.axhline(T_PHONON_K, color="0.5", lw=0.9, ls=":", label=r"$T_{ph}$=100 mK")
    ax_c.set_xlabel(r"dissipated power $P$ (W)")
    ax_c.set_ylabel(r"$T_e$ (K)")
    ax_c.set_title("(c) junction-local $T_e$ vs $P$ (parametric)")
    ax_c.grid(alpha=0.3, which="both")
    ax_c.legend(loc="upper left", fontsize=7.5, framealpha=0.9)

    fig.suptitle(
        "M4 milestone 1 -- power chain, back-action envelope, thermal budget\n"
        "(power_chain/operating_points locked; back-action/thermal parametric, pending_M2)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    power_chain = build_power_chain()
    ops = build_operating_points()
    backaction = build_backaction()
    thermal = build_thermal()
    pending_m2 = build_pending_m2()
    params = build_params()
    provenance = build_provenance()

    export = {
        "power_chain": power_chain,
        "operating_points": ops,
        "backaction": backaction,
        "thermal": thermal,
        "pending_M2": pending_m2,
        "params": params,
        "provenance": provenance,
    }
    export = _to_jsonable(export)

    OUT_JSON.write_text(json.dumps(export, indent=2), encoding="utf-8")

    # Round-trip validation.
    with open(OUT_JSON, "r", encoding="utf-8") as fh:
        reloaded = json.load(fh)
    assert set(reloaded.keys()) == {
        "power_chain", "operating_points", "backaction", "thermal",
        "pending_M2", "params", "provenance",
    }
    for key in ("nu", "Ic_nA", "Omega", "Gamma", "alpha_opt", "i_ac_opt",
                "w_half_Ic", "w_half_nA", "P_in_opt_nW", "P_J_opt_pW", "V_ac_opt_mV"):
        assert len(reloaded["operating_points"][key]) == 9, key
    assert len(reloaded["backaction"]["L_grid"]) > 0
    assert len(reloaded["backaction"]["L_grid"]) == len(reloaded["backaction"]["Delta_over_Delta0"])
    assert len(reloaded["backaction"]["L_grid"]) == len(reloaded["backaction"]["detuning_fraction"])
    assert set(reloaded["backaction"]["threshold_load_by_Q"].keys()) == {"1e+03", "1e+04", "1e+05"}
    assert len(reloaded["thermal"]["P_grid_W"]) > 0
    assert len(reloaded["thermal"]["P_grid_W"]) == len(reloaded["thermal"]["T_e_K"])
    assert len(reloaded["pending_M2"]) == len(PENDING_M2)

    make_figure(power_chain, ops, backaction, thermal)

    print(f"wrote {OUT_JSON}  ({OUT_JSON.stat().st_size} bytes)")
    print(f"wrote {FIG_PATH}  ({FIG_PATH.stat().st_size} bytes, dpi=200)")
    print("block sizes:")
    print(f"  operating_points.nu           : {len(reloaded['operating_points']['nu'])}")
    print(f"  backaction.L_grid             : {len(reloaded['backaction']['L_grid'])}")
    print(f"  thermal.P_grid_W              : {len(reloaded['thermal']['P_grid_W'])}")
    print(f"  pending_M2                    : {len(reloaded['pending_M2'])}")
    print("key numbers:")
    print(f"  alpha(9nW)                    : {power_chain['alpha_9nW']:.6f}")
    print(f"  V_ac (mV)                     : {power_chain['V_ac_mV']:.4f}")
    print(f"  A_exact at Qw/Qi=200          : {power_chain['A_exact_at_200']:.8f}")
    print(f"  threshold_load_by_Q           : {backaction['threshold_load_by_Q']}")
    print(f"  chip_budget ratio / ok        : {thermal['chip_budget']['ratio']:.3e} / {thermal['chip_budget']['ok']}")


if __name__ == "__main__":
    main()
