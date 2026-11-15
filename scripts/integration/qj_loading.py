"""M4 junction-loading closure: the lumped Josephson-junction Q_J channel.

This is the last ``pending_M2`` item of the M4 integration milestone
(M4_REPORT.md Open item #2). M2 has now delivered ``photonics.json`` (the
cavity mode volume ``V_mode``, the intrinsic ``Q_rad``/``Q_mat``, the in-situ
``Q_i`` and the cavity-line field-transfer coefficient ``E/sqrt(W)``), so the
junction loading contribution to ``1/Q_i`` can finally be computed.

Physics (M2_BRIEF.md Sec. 2, lines 54-56)::

    1/Q_i_total = 1/Q_rad + 1/Q_mat + 1/Q_J        (loss-channel sum)
    requirement:  Q_rad, Q_mat >= 10 * Q_J          (>= ~90% of the intrinsic
                                                     dissipation reaches the
                                                     junction)

The junction is a lumped resistive load (normal-state resistance ``Rn``) at
the cavity centre, driven by the AC voltage the cavity near-field develops
across it::

    V_ac   = E_center * d_eff        (d_eff = junction effective length, FREE
                                      parameter; default 100 nm ~ a vdW stack)
    P_J    = V_ac**2 / (2*Rn)        (time-averaged dissipation, v0.3 Rn=200 Ohm)
    Q_J    = omega0 * U / P_J        (Q of the junction loss channel)

with ``U`` the total time-averaged EM energy stored in the cavity mode.


Two results, and why they are different questions
--------------------------------------------------
Q_J = omega0 * U / P_J is a *ratio* of the stored energy to the junction
dissipation.  Both ``U`` and ``P_J = E_center**2 d_eff**2/(2 Rn)`` scale with
the SAME field amplitude, so the field amplitude (and therefore the drive
power *and* any Fano/band-tail dilution of the on-resonance build-up)
**cancels exactly**.  Using the mode relation for the total time-averaged
energy of a cavity mode with peak (DFT-amplitude) centre field ``E_c``::

    <U> = (1/2) * eps0 * eps_c * V_mode * |E_c|**2                (equation *)

(``V_mode = integral(eps|E|^2 dV)/max(eps|E|^2)``, and the junction sits at the
``max(eps|E|^2)`` point in the eps_c = 11.68 silicon, so
``max(eps|E|^2) = eps_c |E_c|^2``; ``<U> = 2<U_E>`` at resonance).  Substituting
into ``Q_J = omega0 <U>/(|E_c|^2 d_eff^2/(2 Rn))`` the ``|E_c|^2`` cancels:

    Q_J(d_eff) = Rn * eps0 * eps_c * omega0 * V_mode / d_eff**2   (equation **)

**This is the closure result.**  It is drive-independent and
dilution-independent, needing only the *geometric* mode volume ``V_mode`` (an
intrinsic, isolated-cavity quantity, exactly like ``Q_rad``/``Q_mat``), so it
is an apples-to-apples partner for the ``Q_rad, Q_mat >= 10 Q_J`` requirement.

The field-transfer coefficient ``E/sqrt(W)`` and the CMT energy
``U = A P_in Q_i/omega0`` answer a *separate* question -- the absolute AC
voltage a given input power develops at the junction (the power-budget /
A = 1% consistency thread).  They are computed here too, with the full MEEP ->
SI unit conversion documented, but they do NOT enter ``Q_J`` (the amplitude
cancels).


MEEP -> SI unit conversion (documented factor by factor)
--------------------------------------------------------
MEEP normalises eps0 = mu0 = c = 1 and measures length in the lattice constant
``a`` (here the device ``a = 586.74 um``, the cavity-line rescale).  Two
conversions are needed and both are derived from ONE fact: the MEEP unit of
power corresponding to a unit MEEP field is ``P_unit = eps0 c a^2 E_unit^2``
(Poynting flux ``E x H`` over an ``a^2`` area, with the vacuum relation
``H_SI = eps0 c E_SI`` fixing ``E_unit^2/P_unit = 1/(eps0 c a^2)``).

1. Field per sqrt(power).  ``E/sqrt(W)`` is exported as the MEEP centre field
   divided by sqrt(MEEP incident flux).  Multiplying by ``E_unit/sqrt(P_unit)``
   converts it to SI::

       (E_SI/sqrt(P_SI)) = (E_meep/sqrt(F_meep)) * sqrt(1/(eps0 c a^2))
                         = (E_meep/sqrt(F_meep)) * sqrt(Z0)/a
                         = (E_meep/sqrt(F_meep)) * sqrt(Z0/a^2)

   with ``Z0 = 1/(eps0 c) = sqrt(mu0/eps0) = 376.730 Ohm``.  The two forms
   ``sqrt(1/(eps0 c a^2))`` and ``sqrt(Z0/a^2)`` are identical and are both
   checked in the tests.  Units: (V/m)/sqrt(W), since
   ``[Z0/a^2] = (V/A)/m^2`` and ``sqrt(V/(A m^2)) = (V/m)/sqrt(V A) =
   (V/m)/sqrt(W)``.

2. Energy per power.  ``energy_unit/P_unit = a/c`` (energy = power x
   light-crossing time ``a/c``), so a MEEP field-energy-integral per MEEP
   incident flux converts to Joules-per-Watt by multiplying by ``a/c``.

The conversion is medium-independent (it is a pure unit change); all medium /
field-profile physics is already inside the dimensionless MEEP numbers.


Fano / band-tail dilution (why the field-U vs CMT-U cross-check is loose)
-------------------------------------------------------------------------
The res18 N=3 S21 run sits in the W1 guided-band tail (decisions 2026-07-24/25):
the measured on-resonance absorbed fraction is only ~5.8% (two independent
TCMT fits, R^2 ~ 0.45) and the field build-up is a corresponding factor below a
clean, on-band resonance.  Consequently the field-transfer stored energy is
~1-2x10^1 times below the clean design-point CMT energy
``U = A P_in Q_i/omega0``.  This is a documented, expected discrepancy -- and it
does NOT touch the Q_J closure (equation ** has no field amplitude in it).  The
CMT cross-check test therefore uses a Fano-justified tolerance and its tight
20% partner is the mutual consistency of the two independent S21 TCMT fits
(the brief's own AT-3 criterion).

Sources: M2_BRIEF.md Sec. 2; paper/decisions.md 2026-07-23/24/25;
photonics.json (modes, transfer, tcmt_check); junction.json contract
(Rn = 200 Ohm, v0.3); integration/__init__.py locked constants.
"""
import json
import math
import pathlib

from integration import RN_OHM, F0_HZ, A_TARGET

# --------------------------------------------------------------------------- #
# SI constants (CODATA; eps0/c match integration/__init__ h,e conventions)
# --------------------------------------------------------------------------- #
EPS0 = 8.8541878128e-12          # F/m   vacuum permittivity
C_LIGHT = 299792458.0            # m/s   speed of light
MU0 = 4.0e-7 * math.pi           # H/m   vacuum permeability
Z0 = math.sqrt(MU0 / EPS0)       # Ohm   vacuum impedance = 376.730...

OMEGA0_RAD = 2.0 * math.pi * F0_HZ   # rad/s   angular drive frequency

#: default junction effective length (typical van-der-Waals stack thickness).
D_EFF_DEFAULT_M = 100.0e-9

ROOT = pathlib.Path(__file__).resolve().parents[1]
PHOTONICS_JSON = ROOT / "photonics.json"
S21_JSON = ROOT / "out" / "m2" / "s21_3d_lad18_s21_N3_s020_rdef115_res18.json"
INTEGRATION_JSON = ROOT / "integration.json"
QJ_CLOSURE_JSON = ROOT / "out" / "qj_closure.json"

__all__ = [
    "meep_field_per_sqrtW_to_SI",
    "meep_conv_factor",
    "stored_energy_cmt",
    "stored_energy_from_center_field",
    "stored_energy_from_boxint",
    "q_junction",
    "q_junction_via_U",
    "d_eff_for_qj",
    "v_ac_from_center_field",
    "p_j_from_center_field",
    "loading_split_qi",
    "fraction_to_junction",
    "closure",
    "main",
]


# --------------------------------------------------------------------------- #
# 1. MEEP -> SI unit conversion (see module docstring, factor by factor)
# --------------------------------------------------------------------------- #
def meep_conv_factor(a_m):
    """Field-per-sqrt(power) MEEP->SI factor ``sqrt(1/(eps0 c a^2))`` [(V/m)/sqrt(W)].

    Equal to ``sqrt(Z0)/a = sqrt(Z0/a^2)`` (checked independently in tests).
    ``a_m`` is the lattice constant in metres.
    """
    return math.sqrt(1.0 / (EPS0 * C_LIGHT * a_m * a_m))


def meep_field_per_sqrtW_to_SI(e_per_sqrtW_meep, a_m):
    """Convert the exported MEEP ``E/sqrt(W)`` to SI ``(V/m)/sqrt(W)``."""
    return e_per_sqrtW_meep * meep_conv_factor(a_m)


# --------------------------------------------------------------------------- #
# 2. Stored energy (three routes: CMT design point, and two field routes)
# --------------------------------------------------------------------------- #
def stored_energy_cmt(P_in_W, A, Q_i, omega0=OMEGA0_RAD):
    """Clean design-point stored energy ``U = A P_in Q_i / omega0`` [J].

    The brief's AT-4 relation (M2_BRIEF.md line 76): the intrinsically
    dissipated power ``A P_in`` equals the energy decay ``U * omega0/Q_i`` in
    steady state.  ``A`` is the absorbed fraction (design 0.99%), ``Q_i`` the
    intrinsic quality factor.
    """
    return A * P_in_W * Q_i / omega0


def stored_energy_from_center_field(E_center_SI, V_mode_m3, eps_c):
    """Total time-averaged EM energy from the peak centre field (equation *).

    ``<U> = (1/2) eps0 eps_c V_mode |E_c|^2`` -- the mode relation used to
    derive Q_J.  ``E_center_SI`` is the peak (DFT-amplitude) field in V/m.
    """
    return 0.5 * EPS0 * eps_c * V_mode_m3 * E_center_SI * E_center_SI


def stored_energy_from_boxint(boxint_per_flux_meep, a_m, eps_c):
    """Total time-averaged stored energy per Watt from the MEEP box |E|^2 integral.

    ``<U>/P_in = (1/2) eps_c * (boxint/inc_flux)_meep * (a/c)`` [J/W]: the MEEP
    time-averaged energy density ``(1/2) eps|E|^2`` integrated over the cavity
    box, per MEEP incident flux, times the energy/power unit ``a/c``.  ``eps_c``
    approximates the box as filled with the eps_c dielectric (the note in the
    S21 artifact: "multiply by eps and 1/2 for stored energy").
    """
    return 0.5 * eps_c * boxint_per_flux_meep * (a_m / C_LIGHT)


# --------------------------------------------------------------------------- #
# 3. Junction loading: V_ac, P_J, Q_J  (equations ** and the recipe)
# --------------------------------------------------------------------------- #
def v_ac_from_center_field(E_center_SI, d_eff_m):
    """AC voltage across the junction ``V_ac = E_center * d_eff`` [V] (amplitude)."""
    return E_center_SI * d_eff_m


def p_j_from_center_field(E_center_SI, d_eff_m, Rn=RN_OHM):
    """Junction dissipation ``P_J = V_ac^2/(2 Rn)`` [W] (time-averaged)."""
    V_ac = v_ac_from_center_field(E_center_SI, d_eff_m)
    return V_ac * V_ac / (2.0 * Rn)


def q_junction(d_eff_m, V_mode_m3, eps_c, Rn=RN_OHM, omega0=OMEGA0_RAD):
    """Junction-loading quality factor (equation **, drive/dilution-independent).

    ``Q_J = Rn eps0 eps_c omega0 V_mode / d_eff^2``.  Derived from
    ``Q_J = omega0 <U>/P_J`` with ``<U>`` and ``P_J`` sharing the same centre
    field, which cancels.  Larger ``d_eff`` (a longer effective collection
    length -- e.g. antenna electrodes feeding the junction) => smaller Q_J =>
    a stronger junction load.
    """
    return Rn * EPS0 * eps_c * omega0 * V_mode_m3 / (d_eff_m * d_eff_m)


def q_junction_via_U(U_J, E_center_SI, d_eff_m, Rn=RN_OHM, omega0=OMEGA0_RAD):
    """Literal ``Q_J = omega0 U / P_J`` (used to show equivalence with equation **).

    With ``U`` and ``E_center_SI`` taken *consistently* (both clean, or both the
    same measured run) this returns exactly :func:`q_junction`.  Mixing a clean
    ``U`` with a diluted ``E_center`` (or vice-versa) double-counts the
    dilution -- documented in the report; not the closure value.
    """
    return omega0 * U_J / p_j_from_center_field(E_center_SI, d_eff_m, Rn)


def d_eff_for_qj(Q_J_target, V_mode_m3, eps_c, Rn=RN_OHM, omega0=OMEGA0_RAD):
    """Effective length giving a target Q_J (inverse of :func:`q_junction`) [m]."""
    return math.sqrt(Rn * EPS0 * eps_c * omega0 * V_mode_m3 / Q_J_target)


# --------------------------------------------------------------------------- #
# 4. Loss-channel sum and the A = 1% (>= ~90% to junction) consistency check
# --------------------------------------------------------------------------- #
def loading_split_qi(Q_rad, Q_mat, Q_J):
    """Total intrinsic Q from the loss-channel sum ``1/Q_i = sum 1/Q_x``."""
    return 1.0 / (1.0 / Q_rad + 1.0 / Q_mat + 1.0 / Q_J)


def fraction_to_junction(Q_J, Q_rad, Q_mat):
    """Fraction of the intrinsic dissipation absorbed by the junction.

    ``(1/Q_J) / (1/Q_rad + 1/Q_mat + 1/Q_J)`` -- the brief wants this >= ~90%,
    which is the ``Q_rad, Q_mat >= 10 Q_J`` requirement stated as a power split.
    """
    inv_J = 1.0 / Q_J
    return inv_J / (1.0 / Q_rad + 1.0 / Q_mat + inv_J)


# --------------------------------------------------------------------------- #
# 5. Input readers (idempotent: re-read the live artifacts every call)
# --------------------------------------------------------------------------- #
def _read_photonics(path=PHOTONICS_JSON):
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    geom, modes, transfer = d["geometry"], d["modes"], d["transfer"]
    return {
        "a_um": float(geom.get("a_um_precise", geom["a_um"])),
        "eps_core": float(geom["eps_core"]),
        "V_mode_a3": float(modes["V_mode_a3"]),
        "f0_GHz": float(modes["f0_GHz"]),
        "Q_i_insitu": float(modes["Q_i_insitu_24a"]),
        "Q_rad_intrinsic": [float(x) for x in modes["Q_rad_intrinsic"]],
        "Q_mat": float(modes["Q_mat"]),
        "E_per_sqrtW_N3": float(transfer["E_per_sqrtW"]["N3"]["value"]),
        "ladder_pending": d["_meta"]["measured_vs_pending"]["pending_run"],
    }


def _read_s21_fits(path=S21_JSON):
    """Field-transfer peak + the two independent S21 TCMT fits (band, flux)."""
    with open(path, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    ft, sp = d["field_transfer"], d["spectra"]
    ip = int(ft["peak_index"])
    return {
        "peak_index": ip,
        "peak_nu": float(ft["peak_nu"]),
        "E_over_sqrtW_peak": float(ft["E_over_sqrtW"][ip]),
        "E2_boxint_peak": float(ft["E2_boxint"][ip]),
        "inc_flux_peak": float(sp["inc_flux"][ip]),
        "R_peak": float(sp["R"][ip]),
        "A_onres_band": float(d["tcmt_fit_band"]["A_onres"]),
        "A_onres_flux": float(d["tcmt_fit_flux"]["A_onres"]),
        "Q_w_fit_band": float(d["tcmt_fit_band"]["Q_w"]),
        "Q_w_fit_flux": float(d["tcmt_fit_flux"]["Q_w"]),
        "R2_band": float(d["tcmt_fit_band"]["R2"]),
        "R2_flux": float(d["tcmt_fit_flux"]["R2"]),
        "dip_depth_fit_band": float(d["tcmt_fit_band"]["dip_depth_fit"]),
    }


# --------------------------------------------------------------------------- #
# 6. Full closure
# --------------------------------------------------------------------------- #
def closure(d_eff_m=D_EFF_DEFAULT_M, photonics_path=PHOTONICS_JSON,
            s21_path=S21_JSON, P_in_ref_W=1.0):
    """Compute the whole Q_J junction-loading closure. Returns a plain dict.

    ``P_in_ref_W`` (default 1 W) sets the reference for the absolute-voltage
    thread only; every Q_J / d_eff result is P_in-independent.
    """
    ph = _read_photonics(photonics_path)
    s21 = _read_s21_fits(s21_path)

    a_m = ph["a_um"] * 1e-6
    eps_c = ph["eps_core"]
    V_mode_m3 = ph["V_mode_a3"] * a_m ** 3
    omega0 = OMEGA0_RAD
    Rn = RN_OHM
    A_design = 0.0099                 # task-specified design absorbed fraction
    Q_i = ph["Q_i_insitu"]            # 1.461e4 (task value, in-situ 24a)
    Q_rad_lo, Q_rad_hi = ph["Q_rad_intrinsic"]
    Q_mat = ph["Q_mat"]

    # ---- MEEP -> SI field ------------------------------------------------- #
    conv = meep_conv_factor(a_m)
    E_per_sqrtW_SI = meep_field_per_sqrtW_to_SI(ph["E_per_sqrtW_N3"], a_m)

    # ---- stored energy: CMT design point + measured field (per W) --------- #
    U_cmt = stored_energy_cmt(P_in_ref_W, A_design, Q_i, omega0)     # J (per P_in_ref)
    E_center_SI = E_per_sqrtW_SI * math.sqrt(P_in_ref_W)             # V/m at P_in_ref
    U_field_center = stored_energy_from_center_field(E_center_SI, V_mode_m3, eps_c)
    boxint_per_flux = s21["E2_boxint_peak"] / s21["inc_flux_peak"]
    U_field_boxint = stored_energy_from_boxint(boxint_per_flux, a_m, eps_c) * P_in_ref_W
    dilution_center = U_cmt / U_field_center
    dilution_boxint = U_cmt / U_field_boxint

    # ---- Q_J closure (equation **) ---------------------------------------- #
    Q_J = q_junction(d_eff_m, V_mode_m3, eps_c, Rn, omega0)
    # equivalence check: same value via omega0 U/P_J with the CMT-consistent field
    E_c_clean = math.sqrt(2.0 * U_cmt / (EPS0 * eps_c * V_mode_m3))  # per P_in_ref
    Q_J_via_U = q_junction_via_U(U_cmt, E_c_clean, d_eff_m, Rn, omega0)

    # ---- requirement: Q_rad, Q_mat >= 10 Q_J  <=> Q_J <= min/10 ----------- #
    Q_J_max_allowed = min(Q_rad_lo, Q_mat) / 10.0        # binding = 2700 (Q_rad_lo)
    d_eff_min_10x = d_eff_for_qj(Q_J_max_allowed, V_mode_m3, eps_c, Rn, omega0)
    d_eff_min_10x_radhi = d_eff_for_qj(min(Q_rad_hi, Q_mat) / 10.0,
                                       V_mode_m3, eps_c, Rn, omega0)
    # strict >= 90% to junction: Q_J <= Q_i_EM/9
    Q_i_EM = 1.0 / (1.0 / Q_rad_lo + 1.0 / Q_mat)
    Q_J_max_90pct = Q_i_EM / 9.0
    d_eff_min_90pct = d_eff_for_qj(Q_J_max_90pct, V_mode_m3, eps_c, Rn, omega0)

    default_in_range = Q_J <= Q_J_max_allowed
    frac_default = fraction_to_junction(Q_J, Q_rad_lo, Q_mat)
    Q_i_total_default = loading_split_qi(Q_rad_lo, Q_mat, Q_J)

    # ---- absolute-voltage thread (uses the measured, diluted field) ------- #
    # v0.3 reference input power 90 nW; what V_ac does 100 nm see, vs the 0.6 mV
    # the drive design requires?
    P_in_v03 = 90.0e-9
    E_center_v03 = E_per_sqrtW_SI * math.sqrt(P_in_v03)
    V_ac_v03_default = v_ac_from_center_field(E_center_v03, d_eff_m)
    V_ac_required_mV = 0.6
    d_eff_for_0p6mV_measured = (V_ac_required_mV * 1e-3) / E_center_v03
    E_center_v03_clean = math.sqrt(2.0 * stored_energy_cmt(P_in_v03, A_design, Q_i, omega0)
                                   / (EPS0 * eps_c * V_mode_m3))
    d_eff_for_0p6mV_clean = (V_ac_required_mV * 1e-3) / E_center_v03_clean

    # ---- A = 1% / dip consistency (two independent S21 TCMT fits) --------- #
    A_meas_band, A_meas_flux = s21["A_onres_band"], s21["A_onres_flux"]
    fits_rel_err = abs(A_meas_band - A_meas_flux) / A_meas_flux
    Qw_fits_rel_err = abs(s21["Q_w_fit_band"] - s21["Q_w_fit_flux"]) / s21["Q_w_fit_flux"]

    return {
        "inputs": {
            "a_um": ph["a_um"], "a_m": a_m, "eps_core": eps_c,
            "V_mode_a3": ph["V_mode_a3"], "V_mode_m3": V_mode_m3,
            "f0_GHz": ph["f0_GHz"], "omega0_rad_s": omega0,
            "Rn_Ohm": Rn, "A_design": A_design, "Q_i_insitu": Q_i,
            "Q_rad_intrinsic": [Q_rad_lo, Q_rad_hi], "Q_mat": Q_mat,
            "E_per_sqrtW_N3_meep": ph["E_per_sqrtW_N3"],
            "d_eff_m": d_eff_m, "d_eff_nm": d_eff_m * 1e9,
            "P_in_ref_W": P_in_ref_W,
        },
        "unit_conversion": {
            "conv_factor_Vm_per_sqrtW": conv,
            "conv_factor_via_Z0_over_a2": math.sqrt(Z0 / (a_m * a_m)),
            "Z0_Ohm": Z0,
            "E_center_SI_per_sqrtW_Vm": E_per_sqrtW_SI,
            "energy_per_power_unit_a_over_c_s": a_m / C_LIGHT,
            "note": "E_SI/sqrt(P_SI) = E_meep/sqrt(F_meep) * sqrt(1/(eps0 c a^2)) = *sqrt(Z0/a^2); U_SI/P_SI = (U/F)_meep * a/c.",
        },
        "stored_energy_per_ref": {
            "P_in_ref_W": P_in_ref_W,
            "U_cmt_J": U_cmt,
            "U_field_center_J": U_field_center,
            "U_field_boxint_J": U_field_boxint,
            "dilution_cmt_over_field_center": dilution_center,
            "dilution_cmt_over_field_boxint": dilution_boxint,
            "note": "U_cmt = A P_in Q_i/omega0 (clean design point). Field routes "
                    "are the measured, band-tail-diluted N=3 build-up; U_field << "
                    "U_cmt is the documented Fano dilution and does NOT enter Q_J.",
        },
        "q_j": {
            "formula": "Q_J = Rn eps0 eps_c omega0 V_mode / d_eff^2",
            "Q_J_at_d_eff": Q_J,
            "Q_J_via_omega0_U_over_PJ_clean": Q_J_via_U,
            "equivalence_rel_err": abs(Q_J - Q_J_via_U) / Q_J,
            "coefficient_Rn_eps0_eps_c_omega0_Vmode": Rn * EPS0 * eps_c * omega0 * V_mode_m3,
        },
        "requirement": {
            "criterion": "Q_rad, Q_mat >= 10 Q_J  (>= ~90% of intrinsic loss to junction)",
            "Q_J_max_allowed_10x": Q_J_max_allowed,
            "Q_J_max_allowed_10x_radhi": min(Q_rad_hi, Q_mat) / 10.0,
            "Q_J_max_for_90pct": Q_J_max_90pct,
            "Q_i_EM_rad_mat_only": Q_i_EM,
            "d_eff_min_10x_m": d_eff_min_10x,
            "d_eff_min_10x_um": d_eff_min_10x * 1e6,
            "d_eff_min_10x_radhi_um": d_eff_min_10x_radhi * 1e6,
            "d_eff_min_90pct_um": d_eff_min_90pct * 1e6,
            "default_100nm_in_range": bool(default_in_range),
            "default_100nm_Q_J": Q_J,
            "default_100nm_fraction_to_junction": frac_default,
            "default_100nm_Q_i_total": Q_i_total_default,
            "verdict": (
                "The default d_eff = 100 nm (bare vdW stack) is NOT in range: "
                "Q_J = %.3g >> %.0f, so the junction is a negligible loss channel "
                "(only %.3g%% of the intrinsic dissipation reaches it). Meeting "
                "Q_rad,Q_mat >= 10 Q_J needs d_eff >= ~%.1f um -- i.e. engineered "
                "antenna/electrode coupling, not the bare stack."
                % (Q_J, Q_J_max_allowed, frac_default * 100.0, d_eff_min_10x * 1e6)
            ),
        },
        "absolute_voltage_thread": {
            "P_in_v03_W": P_in_v03,
            "E_center_v03_measured_Vm": E_center_v03,
            "V_ac_v03_default_100nm_uV": V_ac_v03_default / 1e-6,
            "V_ac_required_mV": V_ac_required_mV,
            "d_eff_for_0p6mV_measured_um": d_eff_for_0p6mV_measured * 1e6,
            "d_eff_for_0p6mV_clean_um": d_eff_for_0p6mV_clean * 1e6,
            "note": "At the v0.3 reference P_in = 90 nW the measured (diluted) cavity "
                    "field puts only ~%.2f uV across a 100 nm junction vs the 0.6 mV "
                    "the drive design requires -- consistent with the Q_J closure: "
                    "the bare stack is far too short."
                    % (V_ac_v03_default / 1e-6),
        },
        "cmt_cross_check": {
            "U_cmt_J": U_cmt,
            "U_field_boxint_J": U_field_boxint,
            "U_field_less_than_cmt": bool(U_field_boxint < U_cmt),
            "dilution_factor": dilution_boxint,
            "dilution_justification": (
                "band-tail/Fano: measured on-resonance absorbed fraction ~%.1f%% "
                "(TCMT fit R^2~%.2f) vs a clean on-band resonance; the field build-up "
                "and hence the stored energy are diluted by ~1-2x10^1."
                % (A_meas_band * 100.0, s21["R2_band"])
            ),
            "A_onres_fit_band": A_meas_band,
            "A_onres_fit_flux": A_meas_flux,
            "two_fit_A_rel_err": fits_rel_err,
            "two_fit_Qw_rel_err": Qw_fits_rel_err,
            "tight_20pct_check": "independent band-power vs flux S21 TCMT fits (brief AT-3 consistency)",
        },
        "provenance": {
            "photonics_json": str(pathlib.Path(photonics_path).name),
            "s21_json": str(pathlib.Path(s21_path).name),
            "ladder_pending_run": ph["ladder_pending"],
            "auto_fill_note": "Q_J closure is N_rows-independent (uses V_mode, not "
                              "per-N Q_w); it re-reads photonics.json every run, so "
                              "the pending N2/N4 in-situ ring-downs auto-propagate "
                              "idempotently when photonics.json fills them in.",
            "generator": "integration/qj_loading.py",
        },
    }


def _integration_json_block(cl):
    """The compact, ready-to-merge integration.json 'qj_closure' block."""
    return {
        "resolves_pending_M2": "Q_J (M4_REPORT.md open item #2)",
        "d_eff_default_nm": cl["inputs"]["d_eff_nm"],
        "Q_J_default_100nm": cl["q_j"]["Q_J_at_d_eff"],
        "Q_J_formula": cl["q_j"]["formula"],
        "d_eff_min_um_for_10x": cl["requirement"]["d_eff_min_10x_um"],
        "d_eff_min_um_for_90pct": cl["requirement"]["d_eff_min_90pct_um"],
        "default_100nm_in_range": cl["requirement"]["default_100nm_in_range"],
        "fraction_to_junction_100nm": cl["requirement"]["default_100nm_fraction_to_junction"],
        "Q_i_total_100nm": cl["requirement"]["default_100nm_Q_i_total"],
        "U_cmt_J_per_W": cl["stored_energy_per_ref"]["U_cmt_J"],
        "dilution_cmt_over_field": cl["cmt_cross_check"]["dilution_factor"],
        "source": "out/qj_closure.json (integration/qj_loading.py)",
    }


def _merge_into_integration_json(cl, path=INTEGRATION_JSON):
    """Idempotently add a namespaced 'qj_closure' key to integration.json.

    run_integration.py (unowned) is NOT touched; the key is namespaced so it
    never collides with that script's output. A bare re-run of run_integration.py
    would drop this key -- re-run qj_loading.py to restore it (idempotent).
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return False
    data["qj_closure"] = _integration_json_block(cl)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    return True


def main(d_eff_m=D_EFF_DEFAULT_M, write=True, update_integration=True):
    """Compute the closure, write out/qj_closure.json, merge integration.json."""
    cl = closure(d_eff_m=d_eff_m)
    payload = dict(cl)
    payload["_meta"] = {
        "module": "M4 junction-loading closure (Q_J)",
        "task": "last pending_M2 item; 1/Q_i = 1/Q_rad + 1/Q_mat + 1/Q_J",
        "deterministic": True,
        "idempotent": True,
        "generator": "integration/qj_loading.py",
    }
    payload["integration_json_update"] = {
        "note": "delivered here (out/qj_closure.json) so integration.json is updated "
                "WITHOUT editing run_integration.py (unowned); main() also merges this "
                "block into integration.json under the namespaced key 'qj_closure'.",
        "qj_closure": _integration_json_block(cl),
    }
    if write:
        QJ_CLOSURE_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(QJ_CLOSURE_JSON, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
    if update_integration:
        _merge_into_integration_json(cl)

    r = cl["requirement"]
    print("=== M4 Q_J junction-loading closure ===")
    print("Q_J formula : %s" % cl["q_j"]["formula"])
    print("Q_J(100 nm) : %.4g   (max allowed = %.0f  =>  in range: %s)"
          % (r["default_100nm_Q_J"], r["Q_J_max_allowed_10x"],
             r["default_100nm_in_range"]))
    print("frac to J   : %.3g %%   Q_i_total(100nm) = %.4g"
          % (r["default_100nm_fraction_to_junction"] * 100.0,
             r["default_100nm_Q_i_total"]))
    print("d_eff_min   : %.2f um (10x rule) / %.2f um (>=90%%)"
          % (r["d_eff_min_10x_um"], r["d_eff_min_90pct_um"]))
    print("U_cmt/W     : %.4g J ; field-U dilution ~%.1fx (Fano/band-tail)"
          % (cl["stored_energy_per_ref"]["U_cmt_J"],
             cl["cmt_cross_check"]["dilution_factor"]))
    print("verdict     : %s" % r["verdict"])
    return payload


if __name__ == "__main__":
    main()
