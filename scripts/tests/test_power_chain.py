"""M4 acceptance tests AT-M4-1..3 (M4_BRIEF_DRAFT.md Sec. 3) for the
integration power chain and standing-wave TCMT.

All deterministic, no RNG. Numbers are the locked constants
(``integration/__init__.py``) and the M3 export ``junction.json`` fingerprint;
nothing new is measured (M4_BRIEF_DRAFT.md line 35, "no new measured numbers").

* AT-M4-1  budget closes: alpha(90 nW) = 1.000 +- 0.002, V_ac = 0.600 mV +- 0.001,
           P_J = 900 pW from A_TARGET*P_IN_REF_W.
* AT-M4-2  TCMT identities: T+R+A = 1 to 1e-12; gamma_i -> 0 => on-res T < 1e-6;
           |A_exact - 2*Q_i/Q_w|/(2*Q_i/Q_w) < 1% at Q_w/Q_i = 200; off-resonance
           |t|^2+|r|^2 <= 1 with equality iff gamma_i = 0.
* AT-M4-3  drive mapping: |i_ac_opt - alpha_opt*Omega| < 1e-9 for every nu;
           P_in_opt(nu=-2.4) = 44.1 nW +- 0.2 nW; P_in_opt ~ alpha_opt^2.

Contract v0.3 (DECISION_contract_v0p3.md; paper/decisions.md 2026-07-25):
the junction was widened x10 (Rn 2 kOhm -> 200 Ohm, Ic 30 -> 300 nA). IcRn =
60 uV is invariant so V_ac = 0.600 mV and alpha = 1.0005 are UNCHANGED; the
power budget scales x10: P_IN_REF_W 9 -> 90 nW and P_J_REF_W 90 -> 900 pW. The
hardcoded budget numbers below are updated accordingly.
"""
import pathlib

import numpy as np
import pytest

from integration import (
    A_TARGET,
    P_IN_REF_W,
    P_J_REF_W,
    QW_OVER_QI,
    RN_OHM,
    V_PHOTON,
)
from integration.power_chain import (
    absorption_from_Q,
    alpha_from_pin,
    alpha_from_vac,
    i_ac_from_alpha,
    on_resonance_TRA,
    operating_points,
    pin_for_alpha,
    tcmt_amplitudes,
    tcmt_power,
    v_ac_from_pj,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
JUNCTION_JSON = ROOT / "junction.json"


# --------------------------------------------------------------------------- #
# AT-M4-1  (power budget closes)
# --------------------------------------------------------------------------- #
def test_at_m4_1_alpha_from_90nW():
    """alpha_from_pin(90 nW) = 1.000 +- 0.002 (v0.3; V_ac preserved => alpha
    unchanged from the v0.2 9 nW reference)."""
    alpha = float(alpha_from_pin(P_IN_REF_W))
    assert alpha == pytest.approx(1.000, abs=0.002)


def test_at_m4_1_v_ac_is_600uV():
    """V_ac = sqrt(2*P_J*Rn) = 0.600 mV +- 0.001 at the 900 pW v0.3 reference
    (sqrt(2*900pW*200 Ohm) = 0.6 mV; the v0.2 value sqrt(2*90pW*2kOhm) is the
    same 0.6 mV -- IcRn/Rn budget preserved)."""
    V_ac_mV = float(v_ac_from_pj(A_TARGET * P_IN_REF_W, RN_OHM)) / 1e-3
    assert V_ac_mV == pytest.approx(0.600, abs=0.001)


def test_at_m4_1_pj_is_900pW_exact():
    """P_J = A_TARGET*P_IN_REF_W = 900 pW exactly (v0.3; matches P_J_REF_W)."""
    P_J = A_TARGET * P_IN_REF_W
    assert P_J == P_J_REF_W                        # identical product, bit-exact
    assert P_J / 1e-12 == pytest.approx(900.0, rel=1e-12)


def test_at_m4_1_chain_roundtrip():
    """pin_for_alpha is the exact inverse of alpha_from_pin (self-consistency)."""
    for P_in in (1e-9, 44.1e-9, 90e-9, 200e-9):
        alpha = alpha_from_pin(P_in)
        assert float(pin_for_alpha(alpha)) == pytest.approx(P_in, rel=1e-12)
    # V_PHOTON is the photon voltage that makes alpha dimensionless
    assert float(alpha_from_vac(V_PHOTON)) == pytest.approx(1.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# AT-M4-2  (TCMT identities)
# --------------------------------------------------------------------------- #
def test_at_m4_2_energy_conservation():
    """On resonance T + R + A = 1 to 1e-12 over a fixed (gamma_i, gamma_w) grid."""
    gamma_i = np.array([0.0, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0])
    gamma_w = np.array([0.05, 0.2, 1.0, 7.0, 50.0, 200.0])
    GI, GW = np.meshgrid(gamma_i, gamma_w, indexing="ij")
    T, R, A = on_resonance_TRA(GI, GW)
    assert np.max(np.abs(T + R + A - 1.0)) < 1e-12


def test_at_m4_2_lossless_mirror():
    """gamma_i -> 0: on-resonance T -> 0 (narrowband mirror), T < 1e-6."""
    gamma_w = np.array([0.05, 0.2, 1.0, 7.0, 50.0, 200.0])
    gamma_i = 1e-9 * gamma_w                       # deterministic vanishing loss
    T, R, A = on_resonance_TRA(gamma_i, gamma_w)
    assert np.max(T) < 1e-6
    assert np.allclose(R, 1.0, atol=1e-6)          # fully reflected


def test_at_m4_2_undercoupled_approx_at_200():
    """|A_exact - 2*Q_i/Q_w|/(2*Q_i/Q_w) < 1% at Q_w/Q_i = 200; A_exact = 400/201^2."""
    Q_i = np.array([1.0e3, 5.0e3, 2.0e4, 8.0e4])
    Q_w = QW_OVER_QI * Q_i                          # locked target Q_w = 200*Q_i
    A_exact = absorption_from_Q(Q_i, Q_w)
    approx = 2.0 * Q_i / Q_w                         # = 2/200 = 0.01
    rel = np.abs(A_exact - approx) / approx
    assert np.all(rel < 0.01)                        # 0.9925% < 1%
    assert np.all(A_exact == pytest.approx(400.0 / 201.0 ** 2, rel=1e-12))


def test_at_m4_2_offres_inequality_and_equality():
    """Off resonance |t|^2+|r|^2 <= 1, strict iff gamma_i > 0, equality iff gamma_i = 0."""
    delta = np.array([-5.0, -1.0, -0.3, 0.3, 1.0, 5.0, 20.0])  # nonzero detunings
    gamma_w = 1.0
    # (a) lossy cavity: strict inequality everywhere off resonance
    gamma_i = 0.37
    t, r = tcmt_amplitudes(delta, gamma_i, gamma_w)
    s_lossy = np.abs(t) ** 2 + np.abs(r) ** 2
    assert np.all(s_lossy <= 1.0 + 1e-15)
    assert np.all(s_lossy < 1.0 - 1e-9)              # strictly below 1
    # analytic absorbed fraction matches 1 - (|t|^2+|r|^2)
    A_analytic = 2.0 * gamma_i * gamma_w / (delta ** 2 + (gamma_i + gamma_w) ** 2)
    assert np.allclose(1.0 - s_lossy, A_analytic, atol=1e-14)
    # (b) lossless cavity gamma_i = 0: equality to machine precision
    t0, r0 = tcmt_amplitudes(delta, 0.0, gamma_w)
    s_lossless = np.abs(t0) ** 2 + np.abs(r0) ** 2
    assert np.allclose(s_lossless, 1.0, atol=1e-14)


# --------------------------------------------------------------------------- #
# AT-M4-3  (drive-mapping consistency, consumes junction.json fingerprint)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def ops():
    return operating_points(str(JUNCTION_JSON))


def test_at_m4_3_i_ac_self_consistency(ops):
    """M3 export self-consistency: |i_ac_opt - alpha_opt*Omega| < 1e-9 for every nu."""
    assert len(ops) == 9
    for op in ops:
        pred = float(i_ac_from_alpha(op["alpha_opt"], op["Omega"]))
        assert abs(op["i_ac_opt"] - pred) < 1e-9


def test_at_m4_3_pin_opt_spot_value(ops):
    """P_in_opt(nu = -2.4) = 90 * 0.7^2 = 44.1 nW +- 0.2 nW (v0.3; x10 of the
    v0.2 4.41 nW, P_IN_REF_W 9 -> 90 nW)."""
    op = next(o for o in ops if abs(o["nu"] - (-2.4)) < 1e-9)
    assert op["alpha_opt"] == pytest.approx(0.7, abs=1e-9)
    assert op["P_in_opt_nW"] == pytest.approx(44.1, abs=0.2)


def test_at_m4_3_pin_opt_scales_as_alpha_squared(ops):
    """P_in_opt ~ alpha_opt^2: the ratio is constant (= 9 nW) across the dome."""
    ratios = np.array([o["P_in_opt_nW"] / o["alpha_opt"] ** 2 for o in ops])
    assert np.allclose(ratios, P_IN_REF_W / 1e-9, rtol=1e-9)   # 90.0 nW (v0.3)
    # pairwise ratio identity: P_in_opt(i)/P_in_opt(j) == (alpha_i/alpha_j)^2
    for a in ops:
        for b in ops:
            lhs = a["P_in_opt_nW"] / b["P_in_opt_nW"]
            rhs = (a["alpha_opt"] / b["alpha_opt"]) ** 2
            assert lhs == pytest.approx(rhs, rel=1e-12)


def test_at_m4_3_derived_power_columns_consistent(ops):
    """P_J_opt = 1% of P_in_opt and V_ac_opt = sqrt(2*P_J_opt*Rn) for every nu."""
    for op in ops:
        assert op["P_J_opt_pW"] == pytest.approx(
            A_TARGET * op["P_in_opt_nW"] * 1e3, rel=1e-12
        )  # nW * 1e3 = pW; times A_TARGET
        V_ac_mV = float(v_ac_from_pj(op["P_J_opt_pW"] * 1e-12, RN_OHM)) / 1e-3
        assert op["V_ac_opt_mV"] == pytest.approx(V_ac_mV, rel=1e-12)
