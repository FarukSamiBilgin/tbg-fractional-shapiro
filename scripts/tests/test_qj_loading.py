"""M4 junction-loading closure tests (integration/qj_loading.py).

Covers the last pending_M2 item: 1/Q_i = 1/Q_rad + 1/Q_mat + 1/Q_J with the
requirement Q_rad, Q_mat >= 10 Q_J.

Groups (per the task):
  * unit discipline  -- MEEP -> SI field factor equals the independent
                        sqrt(Z0/a^2), documented factor by factor.
  * sum / limit      -- loss-channel sum is exact; Q_J(d_eff) is monotone, has
                        the right 1/d_eff^2 scaling, and d_eff_for_qj inverts it;
                        fraction_to_junction is a proper fraction hitting exactly
                        90% at Q_J = Q_i_EM/9.
  * Q_J internal     -- the geometric formula (**) equals omega0*U/P_J with a
                        CMT-consistent field to machine precision.
  * CMT cross-check  -- within 20% (Fano-diluted, tolerance justified): the two
                        INDEPENDENT S21 TCMT fits (band vs flux) agree to <20%
                        (brief AT-3 consistency); the field-transfer stored
                        energy is below the clean-CMT bound with a dilution
                        consistent with the ~5.8% measured absorbed fraction.
  * closure verdict  -- the default 100 nm is NOT in range; d_eff must be
                        ~20-25 um; artifacts written and round-trip.

All deterministic, no RNG. Reads the read-only M2 artifacts photonics.json and
out/m2/s21_3d_lad18_...res18.json.
"""
import json
import math
import pathlib

import pytest

from integration import RN_OHM, F0_HZ
from integration.qj_loading import (
    EPS0,
    C_LIGHT,
    Z0,
    OMEGA0_RAD,
    D_EFF_DEFAULT_M,
    QJ_CLOSURE_JSON,
    INTEGRATION_JSON,
    meep_conv_factor,
    meep_field_per_sqrtW_to_SI,
    stored_energy_cmt,
    stored_energy_from_center_field,
    q_junction,
    q_junction_via_U,
    d_eff_for_qj,
    fraction_to_junction,
    loading_split_qi,
    v_ac_from_center_field,
    p_j_from_center_field,
    closure,
    main,
)

# A representative device point (matches photonics.json / the task).
A_UM = 586.735322
A_M = A_UM * 1e-6
EPS_C = 11.68
V_MODE_A3 = 0.3307
V_MODE_M3 = V_MODE_A3 * A_M ** 3
Q_RAD_LO = 27000.0
Q_MAT = 52075.0


@pytest.fixture(scope="module")
def cl():
    return closure()


# --------------------------------------------------------------------------- #
# Unit discipline: MEEP -> SI
# --------------------------------------------------------------------------- #
def test_meep_conv_equals_Z0_over_a2():
    """sqrt(1/(eps0 c a^2)) == sqrt(Z0/a^2) == sqrt(Z0)/a (three identical forms)."""
    conv = meep_conv_factor(A_M)
    # conv is built from 1/(eps0 c a^2); the Z0 forms use Z0 = sqrt(mu0/eps0).
    # Post-2019 SI: mu0 is no longer exactly 4*pi*1e-7, so 1/(eps0 c) and
    # sqrt(mu0/eps0) differ at the 9th digit -> compare at 1e-8.
    assert conv == pytest.approx(math.sqrt(Z0) / A_M, rel=1e-8)
    assert conv == pytest.approx(math.sqrt(Z0 / A_M ** 2), rel=1e-8)
    # sqrt(Z0)/a and sqrt(Z0/a^2) are algebraically identical -> exact.
    assert math.sqrt(Z0) / A_M == pytest.approx(math.sqrt(Z0 / A_M ** 2), rel=1e-12)
    assert Z0 == pytest.approx(1.0 / (EPS0 * C_LIGHT), rel=1e-8)


def test_meep_field_to_SI_scales_linearly():
    """E_SI/sqrt(W) is linear in the MEEP number and ~3.46e4 (V/m)/sqrt(W) at 1.046."""
    e_si = meep_field_per_sqrtW_to_SI(1.046, A_M)
    assert e_si == pytest.approx(1.046 * meep_conv_factor(A_M), rel=1e-12)
    assert e_si == pytest.approx(3.46e4, rel=0.02)          # sanity magnitude
    assert meep_field_per_sqrtW_to_SI(2.092, A_M) == pytest.approx(2.0 * e_si, rel=1e-12)


def test_conv_factor_a_scaling():
    """Field factor scales as 1/a (scale invariance of Maxwell)."""
    assert meep_conv_factor(2.0 * A_M) == pytest.approx(meep_conv_factor(A_M) / 2.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# Sum / limit checks
# --------------------------------------------------------------------------- #
def test_loss_channel_sum_exact():
    """1/Q_i_total = 1/Q_rad + 1/Q_mat + 1/Q_J to machine precision."""
    Q_J = 3.0e3
    Q_i = loading_split_qi(Q_RAD_LO, Q_MAT, Q_J)
    lhs = 1.0 / Q_i
    rhs = 1.0 / Q_RAD_LO + 1.0 / Q_MAT + 1.0 / Q_J
    assert lhs == pytest.approx(rhs, rel=1e-12)


def test_qj_scales_as_inverse_d_eff_squared():
    """Q_J ~ 1/d_eff^2 (equation **): halving d_eff quadruples Q_J."""
    q1 = q_junction(1.0e-7, V_MODE_M3, EPS_C)
    q2 = q_junction(0.5e-7, V_MODE_M3, EPS_C)
    assert q2 == pytest.approx(4.0 * q1, rel=1e-12)


def test_qj_monotone_decreasing_in_d_eff():
    ds = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    qs = [q_junction(d, V_MODE_M3, EPS_C) for d in ds]
    assert all(qs[i] > qs[i + 1] for i in range(len(qs) - 1))
    # limits: tiny d_eff -> huge Q_J (negligible load); large d_eff -> tiny Q_J
    assert q_junction(1e-9, V_MODE_M3, EPS_C) > 1e10
    assert q_junction(1e-2, V_MODE_M3, EPS_C) < 1.0            # d_eff = 1 cm


def test_d_eff_for_qj_is_inverse():
    """d_eff_for_qj inverts q_junction exactly."""
    for Q_target in (2.7e3, 5.2e3, 1.0e4, 1.259e8):
        d = d_eff_for_qj(Q_target, V_MODE_M3, EPS_C)
        assert q_junction(d, V_MODE_M3, EPS_C) == pytest.approx(Q_target, rel=1e-10)


def test_fraction_to_junction_is_proper_fraction():
    for Q_J in (1e2, 1e3, 1e4, 1e6, 1e8):
        f = fraction_to_junction(Q_J, Q_RAD_LO, Q_MAT)
        assert 0.0 < f < 1.0
    # monotone: smaller Q_J (stronger load) -> larger fraction
    assert fraction_to_junction(1e2, Q_RAD_LO, Q_MAT) > fraction_to_junction(1e6, Q_RAD_LO, Q_MAT)


def test_fraction_hits_90pct_at_Qi_EM_over_9():
    """Exactly 90% reaches the junction when Q_J = Q_i_EM/9 (definition check)."""
    Q_i_EM = 1.0 / (1.0 / Q_RAD_LO + 1.0 / Q_MAT)
    f = fraction_to_junction(Q_i_EM / 9.0, Q_RAD_LO, Q_MAT)
    assert f == pytest.approx(0.9, rel=1e-9)


# --------------------------------------------------------------------------- #
# Q_J internal consistency: (**) == omega0 U / P_J
# --------------------------------------------------------------------------- #
def test_geometric_qj_equals_omega0_U_over_PJ():
    """Q_J = Rn eps0 eps_c omega0 V_mode/d_eff^2 equals omega0*U/P_J for a
    field consistent with U (the |E_c|^2 cancellation), any U, any d_eff."""
    for U in (1e-12, 1e-10, 3.3e-9):
        E_c = math.sqrt(2.0 * U / (EPS0 * EPS_C * V_MODE_M3))   # <U>=(1/2)eps0 eps_c Vmode E_c^2
        for d in (5e-8, 1e-7, 2e-5):
            geo = q_junction(d, V_MODE_M3, EPS_C)
            via = q_junction_via_U(U, E_c, d)
            assert geo == pytest.approx(via, rel=1e-9)


def test_pj_and_vac_definitions():
    """P_J = V_ac^2/(2 Rn) with V_ac = E_center d_eff (amplitude, time-averaged)."""
    E_c, d = 3.46e4, 1e-7
    V_ac = v_ac_from_center_field(E_c, d)
    assert V_ac == pytest.approx(E_c * d, rel=1e-12)
    assert p_j_from_center_field(E_c, d) == pytest.approx(V_ac ** 2 / (2.0 * RN_OHM), rel=1e-12)


# --------------------------------------------------------------------------- #
# CMT cross-check within 20% (Fano-diluted; tolerance justified)
# --------------------------------------------------------------------------- #
def test_U_cmt_formula_exact(cl):
    """U_cmt reproduces A*P_in*Q_i/omega0 exactly (A=0.0099, Q_i=1.461e4)."""
    inp = cl["inputs"]
    U = stored_energy_cmt(inp["P_in_ref_W"], inp["A_design"], inp["Q_i_insitu"], OMEGA0_RAD)
    assert cl["stored_energy_per_ref"]["U_cmt_J"] == pytest.approx(U, rel=1e-12)
    assert U == pytest.approx(0.0099 * 1.0 * 1.461e4 / (2 * math.pi * F0_HZ), rel=2e-3)


def test_two_independent_s21_fits_agree_within_20pct(cl):
    """The TIGHT 20% CMT cross-check: two independent S21 TCMT fits (band-power
    vs flux) give the same absorbed fraction and Q_w to <20% (brief AT-3
    consistency). They actually agree to ~2-3%."""
    cc = cl["cmt_cross_check"]
    assert cc["two_fit_A_rel_err"] < 0.20
    assert cc["two_fit_Qw_rel_err"] < 0.20
    assert cc["A_onres_fit_band"] == pytest.approx(cc["A_onres_fit_flux"], rel=0.20)


def test_field_U_below_clean_cmt_with_justified_dilution(cl):
    """Field-transfer stored energy < clean-CMT bound; the dilution factor is in
    the band-tail range justified by the ~5.8% measured absorbed fraction (a
    clean on-band resonance would build up far more). Not 20% -- Fano-diluted,
    and it does NOT enter Q_J (amplitude cancels)."""
    cc = cl["cmt_cross_check"]
    assert cc["U_field_less_than_cmt"] is True
    assert 3.0 < cc["dilution_factor"] < 40.0
    # measured on-resonance absorbed fraction is the ~5.8% band-tail value, not 99%
    assert cc["A_onres_fit_band"] == pytest.approx(0.058, abs=0.01)


# --------------------------------------------------------------------------- #
# Closure verdict + artifacts
# --------------------------------------------------------------------------- #
def test_default_100nm_not_in_range(cl):
    """The headline finding: bare 100 nm gives Q_J ~ 1.3e8 >> 2.7e3, far outside
    the Q_rad,Q_mat >= 10 Q_J window; <<1% of intrinsic loss reaches it."""
    r = cl["requirement"]
    assert r["default_100nm_in_range"] is False
    assert r["default_100nm_Q_J"] > 1e7
    assert r["default_100nm_Q_J"] > 100.0 * r["Q_J_max_allowed_10x"]
    assert r["default_100nm_fraction_to_junction"] < 0.01       # <1%, nowhere near 90%


def test_d_eff_window_is_tens_of_microns(cl):
    """Meeting the requirement needs d_eff ~ 20-25 um (engineered coupling)."""
    r = cl["requirement"]
    assert 15.0 < r["d_eff_min_10x_um"] < 35.0
    assert r["d_eff_min_90pct_um"] >= r["d_eff_min_10x_um"]      # 90% is stricter
    # 100 nm is ~2 orders of magnitude too short
    assert r["d_eff_min_10x_um"] * 1e-6 / D_EFF_DEFAULT_M > 100.0


def test_qj_default_matches_closure_field(cl):
    """The default-d_eff Q_J in the closure matches a direct q_junction call."""
    q = q_junction(D_EFF_DEFAULT_M, V_MODE_M3, EPS_C)
    assert cl["q_j"]["Q_J_at_d_eff"] == pytest.approx(q, rel=1e-6)
    assert cl["q_j"]["equivalence_rel_err"] < 1e-9


def test_absolute_voltage_thread_consistent(cl):
    """Sanity of the absolute-voltage thread: 100 nm sees ~1 uV at 90 nW vs the
    0.6 mV the drive needs; the d_eff to reach 0.6 mV is tens of um."""
    av = cl["absolute_voltage_thread"]
    assert av["V_ac_v03_default_100nm_uV"] == pytest.approx(1.04, rel=0.1)
    assert 5.0 < av["d_eff_for_0p6mV_clean_um"] < 100.0
    assert 5.0 < av["d_eff_for_0p6mV_measured_um"] < 200.0


def test_closure_is_idempotent():
    """Re-reading the live artifacts twice yields identical Q_J (idempotent)."""
    a = closure()["q_j"]["Q_J_at_d_eff"]
    b = closure()["q_j"]["Q_J_at_d_eff"]
    assert a == b


def test_artifacts_written_and_roundtrip():
    """main() writes out/qj_closure.json and merges integration.json['qj_closure']."""
    payload = main(write=True, update_integration=True)
    assert QJ_CLOSURE_JSON.exists()
    with open(QJ_CLOSURE_JSON, "r", encoding="utf-8") as fh:
        disk = json.load(fh)
    assert disk["q_j"]["Q_J_at_d_eff"] == pytest.approx(payload["q_j"]["Q_J_at_d_eff"], rel=1e-12)
    assert "integration_json_update" in disk
    # integration.json carries the namespaced closure block, other keys intact
    with open(INTEGRATION_JSON, "r", encoding="utf-8") as fh:
        ij = json.load(fh)
    assert "qj_closure" in ij
    assert "power_chain" in ij and "operating_points" in ij     # untouched
    assert ij["qj_closure"]["default_100nm_in_range"] is False
