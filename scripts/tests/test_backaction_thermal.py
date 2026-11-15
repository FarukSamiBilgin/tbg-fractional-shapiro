"""Tests for the M4 back-action envelope and thermal budget.

Covers AT-M4-4 (quasiparticle back-action) and the thermal deliverable of
M4_BRIEF_DRAFT.md Sec. 3-4. Deterministic; no RNG. Every asserted number is
computed here from the reused M3 steady state or the closed-form thermal law.
"""
import numpy as np
import pytest

from junction.tien_gordon import steady_state_delta
from integration import P_J_REF_W
from integration.backaction import (
    load, delta_vs_load, detuning_fraction, threshold_load,
    _threshold_closed_form,
)
from integration.thermal import te_from_power, power_from_te, chip_budget

Q_SET = (1.0e3, 1.0e4, 1.0e5)
L_REPORT = (1e-4, 1e-3, 1e-2, 0.1)


# ============================================================ AT-M4-4
def test_delta_at_zero_load_is_exactly_delta0():
    """Delta(L=0) = Delta0 exactly, i.e. the reduced gap is exactly 1.0."""
    assert delta_vs_load(0.0) == 1.0
    # array form: the L=0 entry is exactly 1.0 too
    d = delta_vs_load(np.array([0.0, 1e-3, 1e-2]))
    assert d[0] == 1.0


def test_delta_strictly_decreasing_on_log_grid():
    """Delta/Delta0 strictly decreasing along a log-spaced load grid."""
    L = np.logspace(-6, -1, 60)
    d = delta_vs_load(L)
    assert np.all(np.diff(d) < 0.0)
    assert np.all((d > 0.0) & (d <= 1.0))


def test_load_invariance_fixed_L_different_eta_P():
    """Different (eta_abs, P_abs) at fixed L give identical Delta to 1e-12.

    Holds L constant while varying eta_abs, tau_rec, n0 and Delta0 (each of
    which forces a different P_abs = L*n0*Delta0/(eta_abs*tau_rec)); the reduced
    gap must be identical. Also checks the raw steady_state_delta directly with
    two explicit (eta_abs, P_abs) pairs at the same load.
    """
    param_sets = [
        dict(eta_abs=1.0, tau_rec=1.0, n0=1.0, Delta0=1.0),
        dict(eta_abs=3.7, tau_rec=1.0, n0=1.0, Delta0=1.0),
        dict(eta_abs=1.0, tau_rec=0.25, n0=1.0, Delta0=1.0),
        dict(eta_abs=2.0, tau_rec=5.0, n0=3.0, Delta0=7.0),
        dict(eta_abs=0.1, tau_rec=13.0, n0=0.5, Delta0=0.2998e-3),
    ]
    for L in (1e-4, 1e-3, 1e-2, 0.1, 0.5):
        ref = delta_vs_load(L, **param_sets[0])
        for ps in param_sets[1:]:
            got = delta_vs_load(L, **ps)
            assert abs(got - ref) < 1e-12, (L, ps, got, ref)

        # direct steady_state_delta: two (eta_abs, P_abs) pairs, same L
        eta_a, eta_b = 1.0, 4.0
        n0, tau, D0 = 1.0, 1.0, 1.0
        P_a = L * n0 * D0 / (eta_a * tau)
        P_b = L * n0 * D0 / (eta_b * tau)
        d_a = steady_state_delta(eta_a, P_a, tau, n0, D0,
                                 tol=1e-14, max_iter=5000) / D0
        d_b = steady_state_delta(eta_b, P_b, tau, n0, D0,
                                 tol=1e-14, max_iter=5000) / D0
        assert abs(d_a - d_b) < 1e-12
        assert abs(load(eta_a, P_a, tau, n0, D0) - L) < 1e-15
        assert abs(load(eta_b, P_b, tau, n0, D0) - L) < 1e-15


def test_detuning_identity_to_1e_12():
    """detuning_fraction(L) == 1 - Delta(L)/Delta0 to 1e-12 on a grid."""
    L = np.logspace(-6, -0.5, 40)
    lhs = detuning_fraction(L)
    rhs = 1.0 - delta_vs_load(L)
    assert np.max(np.abs(lhs - rhs)) < 1e-12
    # reduced-form definition also matches the frequency-space form
    # (f0 - 2*Delta/h)/f0 with f0 = 2*Delta0/h  ->  1 - Delta/Delta0
    assert detuning_fraction(0.0) == 0.0


def test_threshold_load_monotone_decreasing_in_Q():
    """threshold_load is strictly decreasing in Q over {1e3, 1e4, 1e5}."""
    thr = np.array([threshold_load(Q) for Q in Q_SET])
    assert np.all(np.diff(thr) < 0.0)
    # each threshold reproduces the load whose detuning equals 1/(2Q)
    for Q, L in zip(Q_SET, thr):
        assert detuning_fraction(L) == pytest.approx(1.0 / (2.0 * Q), rel=1e-9)


def test_threshold_load_matches_closed_form():
    """Root-found threshold agrees with the closed form (4/Q^2)(1-1/(2Q))."""
    for Q in Q_SET:
        got = threshold_load(Q)
        exp = _threshold_closed_form(Q)
        assert got == pytest.approx(exp, rel=1e-8)


def test_detuning_exceeds_linewidth_above_threshold():
    """Above the threshold load the detuning does exceed the half-linewidth."""
    for Q in Q_SET:
        L_thr = threshold_load(Q)
        assert detuning_fraction(1.01 * L_thr) > 1.0 / (2.0 * Q)
        assert detuning_fraction(0.99 * L_thr) < 1.0 / (2.0 * Q)


# ============================================================ thermal
SIGMA_V = 1.0e-9      # W/K^5, free parameter (documented)
T_PH = 0.1            # K, 100 mK phonon bath


def test_te_at_zero_power_is_exactly_tph():
    """T_e(P=0) = T_ph exactly, scalar and array."""
    assert te_from_power(0.0, SIGMA_V, T_PH) == T_PH
    Te = te_from_power(np.array([0.0, 1e-12, 1e-9]), SIGMA_V, T_PH)
    assert Te[0] == T_PH


def test_te_monotone_increasing_in_power():
    """T_e strictly increasing in P."""
    P = np.linspace(0.0, 1e-8, 50)
    Te = te_from_power(P, SIGMA_V, T_PH)
    assert np.all(np.diff(Te) > 0.0)
    assert np.all(Te >= T_PH)


def test_power_te_roundtrip_to_1e_12():
    """Closed-form inversion round-trips P -> T_e -> P to 1e-12 (relative)."""
    # loads chosen so P/Sigma_V is comparable to or above the phonon floor
    # T_ph^delta (=1e-5 here), keeping the inverse well conditioned.
    P = np.array([1e-13, 9e-11, 1e-9, 1e-7, 1e-5])
    Te = te_from_power(P, SIGMA_V, T_PH)
    P_back = power_from_te(Te, SIGMA_V, T_PH)
    assert np.allclose(P_back, P, rtol=1e-12, atol=0.0)
    # exact closure at P = 0
    assert power_from_te(te_from_power(0.0, SIGMA_V, T_PH), SIGMA_V, T_PH) == 0.0


@pytest.mark.parametrize("delta_exp", [3, 4, 5, 6])
def test_roundtrip_various_exponents(delta_exp):
    """Round-trip holds for other cooling exponents too."""
    P = np.array([1e-12, 1e-10, 1e-8])
    Te = te_from_power(P, SIGMA_V, T_PH, delta_exp=delta_exp)
    P_back = power_from_te(Te, SIGMA_V, T_PH, delta_exp=delta_exp)
    assert np.allclose(P_back, P, rtol=1e-12, atol=0.0)


def test_chip_budget_at_locked_operating_point():
    """P_J vs 500 uW fridge, ok True.

    Contract v0.3 (DECISION_contract_v0p3.md; paper/decisions.md 2026-07-25):
    the junction was widened x10, so the reference junction power scaled
    P_J_REF_W = A_TARGET*P_IN_REF_W: 90 pW -> 900 pW and the chip-budget ratio
    900 pW / 500 uW = 1.8e-6 (was 1.8e-7). Still ok (six orders of margin).
    The chip_budget() function itself is unchanged; only the reference number
    here tracks the sanctioned contract change.
    """
    # function behaviour with an explicit input (unchanged by the contract):
    cb = chip_budget(90.0e-12)
    assert cb.ratio == pytest.approx(1.8e-7, rel=1e-9)
    assert cb.ok is True
    # consistency with the v0.3 locked reference power P_J_REF_W = 900 pW
    assert P_J_REF_W == pytest.approx(900.0e-12, rel=1e-12)
    cb2 = chip_budget(P_J_REF_W)
    assert cb2.ratio == pytest.approx(1.8e-6, rel=1e-9)
    assert cb2.ok is True


def test_chip_budget_ok_flips_when_load_exceeds_fridge():
    """ok is False once the junction load exceeds the fridge cooling power."""
    assert chip_budget(1.0e-3, P_fridge=500e-6).ok is False
    assert chip_budget(499e-6, P_fridge=500e-6).ok is True
