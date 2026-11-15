"""Tests for the Tien-Gordon above-gap module (M3_BRIEF.md Sec. 4.4).

Deterministic; no RNG. Covers the six cases mandated by the brief:
 (i)   Bessel sum rule sum_n J_n^2(alpha) = 1 to 1e-10 at the contract nmax;
 (ii)  alpha = 0 recovers iqp0_bcs exactly;
 (iii) ohmic asymptote within 2% at eV = 10*Delta;
 (iv)  sub-gap current ~ 0 for eV < 2*Delta at T << Delta (alpha = 0);
 (v)   photon-assisted step at eV = 2*Delta - hbar*omega for alpha = 0.5;
 (vi)  P_abs = 0 -> Delta = Delta0, and Delta strictly decreasing in P_abs.
"""
import numpy as np
import pytest

from junction.tien_gordon import (
    DELTA0, HW_DRIVE, KB_EV_K,
    bcs_dos, bessel_weights, default_nmax, iqp0_bcs,
    steady_state_delta, tien_gordon_iv, evolve_nqp,
)

RN = 2000.0            # ohm (contract normal-state resistance)
T_LOW = 0.05           # K, kT/Delta0 ~ 0.014 << 1


# ---------------------------------------------------------------- (i)
@pytest.mark.parametrize("alpha", [0.5, 2.0, 5.0])
def test_bessel_sum_rule(alpha):
    """sum_n J_n(alpha)^2 = 1 to 1e-10 at the contract truncation nmax."""
    n, w = bessel_weights(alpha)
    assert n.max() == default_nmax(alpha)
    assert abs(w.sum() - 1.0) < 1e-10


# ---------------------------------------------------------------- (ii)
def test_alpha0_recovers_iqp0():
    """alpha = 0 collapses the sideband sum to iqp0_bcs exactly."""
    V = np.linspace(0.5 * DELTA0, 5.0 * DELTA0, 25)
    I_tg = tien_gordon_iv(V, 0.0, DELTA0, T_LOW, RN)
    I_qp = iqp0_bcs(V, DELTA0, T_LOW, RN)
    assert np.allclose(I_tg, I_qp, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------- (iii)
def test_ohmic_asymptote():
    """I*Rn -> V within 2% at eV = 10*Delta (DOS -> 1 well above the gap)."""
    V = 10.0 * DELTA0                      # bias energy 10*Delta (eV) == V (volts)
    I = iqp0_bcs(V, DELTA0, T_LOW, RN)
    dev = abs(I * RN / V - 1.0)
    assert dev < 0.02


# ---------------------------------------------------------------- (iv)
def test_subgap_current_vanishes():
    """Sub-gap current is negligible vs an above-gap reference (alpha = 0)."""
    I_ref = iqp0_bcs(3.0 * DELTA0, DELTA0, T_LOW, RN)          # above gap
    for r in (0.5, 1.0, 1.5, 1.9):
        I_sub = iqp0_bcs(r * DELTA0, DELTA0, T_LOW, RN)
        assert 0.0 <= I_sub < 0.02 * I_ref


# ---------------------------------------------------------------- (v)
def test_photon_assisted_step():
    """alpha = 0.5 opens photon-assisted current at eV = 2*Delta - hbar*omega.

    Uses hw_test = Delta so the first sideband threshold sits at eV = Delta,
    a clean interior point. Below 2*Delta the un-driven current is zero, so any
    current there is photon-assisted; it switches on sharply at eV = Delta.
    """
    hw = DELTA0
    thr = 2.0 * DELTA0 - hw                 # = Delta0
    V_below = 0.8 * thr
    V_above = 1.2 * thr
    I_below = tien_gordon_iv(V_below, 0.5, DELTA0, T_LOW, RN, hw=hw)
    I_above = tien_gordon_iv(V_above, 0.5, DELTA0, T_LOW, RN, hw=hw)
    # sharp step at the sideband threshold
    assert I_above > 20.0 * max(I_below, 1e-30)
    # the current below 2*Delta is purely photon-assisted: negligible without
    # drive, but the drive lifts it by more than an order of magnitude.
    I_nodrive = iqp0_bcs(V_above, DELTA0, T_LOW, RN)            # < 2*Delta => ~0
    I_ohmic = iqp0_bcs(3.0 * DELTA0, DELTA0, T_LOW, RN)
    assert I_nodrive < 0.02 * I_ohmic
    assert I_above > 0.0
    assert I_above > 10.0 * I_nodrive                          # drive-induced


# ---------------------------------------------------------------- (vi)
def test_pairbreaking_monotonic():
    """P_abs = 0 -> Delta = Delta0; Delta strictly decreasing in P_abs."""
    assert steady_state_delta(1.0, 0.0, 1.0, 1.0, DELTA0) == DELTA0
    P = np.linspace(0.0, 1.0, 12) * DELTA0
    Delta = steady_state_delta(1.0, P, 1.0, 1.0, DELTA0)
    assert Delta[0] == DELTA0
    assert np.all(np.diff(Delta) < 0.0)
    assert np.all(Delta > 0.0)


# ------------------------------------------------- supporting sanity checks
def test_dos_limits():
    """DOS -> 1 far above the gap and is small (Dynes floor) inside the gap."""
    assert abs(bcs_dos(50.0 * DELTA0, DELTA0) - 1.0) < 1e-3
    assert bcs_dos(0.0, DELTA0) < 1e-2


def test_iqp_odd_in_V():
    """Quasiparticle I-V is odd: I(-V) = -I(V)."""
    V = np.array([1.5, 2.5, 5.0]) * DELTA0
    assert np.allclose(iqp0_bcs(-V, DELTA0, T_LOW, RN),
                       -iqp0_bcs(V, DELTA0, T_LOW, RN), rtol=1e-9)


def test_evolve_reaches_steady_state():
    """Transient RK4 evolve converges to the self-consistent steady state."""
    eta, P, tau_rec, n0 = 1.0, 0.5 * DELTA0, 1.0, 1.0
    _, nq, Delta = evolve_nqp(eta, P, tau_rec, n0, DELTA0, dtau=1e-2, n_steps=20000)
    Delta_ss = steady_state_delta(eta, P, tau_rec, n0, DELTA0)
    assert abs(Delta[-1] - Delta_ss) < 1e-4 * DELTA0
