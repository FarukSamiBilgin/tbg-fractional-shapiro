"""M3 acceptance tests AT-1..AT-3 (M3_BRIEF.md Sec. 3) + analytic unit test.

Every test re-runs the reference numerics (RK4, spp=60, 150 T transient,
300 T averaging, plateau tolerance 0.025) on the reference grids and compares
both against the brief targets and against `junction_reference.json`.
Reproduction tolerances are one bias-grid cell of the width detector.
"""
import json
import pathlib

import numpy as np
import pytest
from scipy.special import j0

from junction.analysis import step_width, width_curve
from junction.cpr import C_SIN, C_SKEW, SKEW_RENORM
from junction.rsj import OMEGA, grid_run, run_rsj

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOL = 0.025  # plateau tolerance |v - v_step| (brief Sec. 3)


@pytest.fixture(scope="session")
def ref():
    with open(ROOT / "junction_reference.json") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def stageA():
    """Benchmark staircases, sinusoidal CPR (reference stage A grid)."""
    idc = np.arange(0.0, 13.0 + 1e-9, 0.05)
    iac_vals = np.array([0.0, 4.0, 9.2])
    v = grid_run(idc, iac_vals, C_SIN)
    return idc, iac_vals, v


@pytest.fixture(scope="session")
def stageB():
    """Half-integer step map, skewed vs sinusoidal CPR (reference stage B grid)."""
    idc = np.arange(1.0, 8.0 + 1e-9, 0.025)
    iac = np.arange(0.0, 12.0 + 1e-9, 0.5)
    v_skew = grid_run(idc, iac, C_SKEW)
    v_sin = grid_run(idc, iac, C_SIN)
    return idc, iac, v_skew, v_sin


@pytest.fixture(scope="session")
def stageC():
    """Step-0 width vs i_ac, sinusoidal CPR (reference stage C grid)."""
    idc = np.arange(0.0, 1.4 + 1e-9, 0.02)
    iac = np.arange(0.0, 14.0 + 1e-9, 0.35)
    v = grid_run(idc, iac, C_SIN)
    return idc, iac, v


# ---------------------------------------------------------------- unit tests
def test_analytic_iv_no_drive():
    """Brief Sec. 5 unit test (i): i_ac = 0 => v = sqrt(i_dc^2 - 1) for i_dc > 1."""
    idc = np.array([1.05, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0])
    v = run_rsj(idc, np.zeros_like(idc), C_SIN)
    # v-resolution of the 300 T averaging window is 2*pi/tau_avg ~ 0.017
    assert np.allclose(v, np.sqrt(idc**2 - 1.0), atol=0.02)


def test_cpr_normalization(ref):
    """Skew CPR renormalization matches the reference contract."""
    assert np.allclose(C_SKEW, ref["contract"]["cpr_skew"], atol=1e-9)
    assert abs(SKEW_RENORM - ref["contract"]["skew_renorm_factor"]) < 1e-9
    assert np.allclose(C_SIN, [1.0])


# ---------------------------------------------------------------- AT-1
def test_at1_integer_steps_only(stageA, stageB, ref):
    """AT-1: sinusoidal CPR shows only integer steps; half-step detector <= 0.05."""
    idcA, iacA, vA = stageA
    # reproduction of the reference staircases (v-resolution ~ 0.017)
    vA_ref = np.asarray(ref["stageA"]["v"], float)
    assert np.allclose(vA, vA_ref, atol=0.02)
    # plateaus at v = n*Omega exist under drive (i_ac = 4)
    k4 = 1  # iacA = [0, 4, 9.2]
    assert step_width(vA[k4], idcA, 1.0 * OMEGA, TOL) >= 0.1
    assert step_width(vA[k4], idcA, 2.0 * OMEGA, TOL) >= 0.1
    # half-step detector on the sinusoidal control stays at the tolerance floor
    idcB, iacB, _, vB_sin = stageB
    w_half_sin = width_curve(vB_sin, idcB, 0.5 * OMEGA, TOL)
    assert w_half_sin.max() <= 0.05 + 1e-12
    assert np.allclose(w_half_sin, ref["stageB"]["w_half_sin"], atol=0.026)


# ---------------------------------------------------------------- AT-2
def test_at2_bessel_tien_gordon_limit(stageC, ref):
    """AT-2: step-0 width follows 2|J0(i_ac/Omega)|; first zero at 12.0 +- 0.4."""
    idcC, iacC, vC = stageC
    w0 = 2.0 * width_curve(vC, idcC, 0.0, TOL)
    w0_theory = 2.0 * np.abs(j0(iacC / OMEGA))
    # first J0 zero located from the minimum above i_ac = 9
    k_lo = np.searchsorted(iacC, 9.0)
    k_zero = k_lo + int(np.argmin(w0[k_lo:]))
    iac_zero = iacC[k_zero]
    assert abs(iac_zero - 12.0) <= 0.4
    # RMS deviation from the Bessel envelope over i_ac in [0, 14]
    rms = float(np.sqrt(np.mean((w0 - w0_theory) ** 2)))
    assert rms <= 0.1
    # reproduction of the reference curve (grid quantum 2*0.02 = 0.04)
    assert np.allclose(w0, ref["stageC"]["w0_numeric"], atol=0.045)
    assert abs(iac_zero - ref["stageC"]["iac_J0zero_numeric"]) <= 0.35 + 1e-9


# ---------------------------------------------------------------- AT-3
def test_at3_fractional_fingerprint(stageB, ref):
    """AT-3: skewed CPR (c2/c1 = -0.25) half-step 0.250 +- 0.03 at i_ac ~ 4,
    >= 5x the sinusoidal control floor; second lobe near i_ac = 9-10."""
    idcB, iacB, vB_skew, vB_sin = stageB
    w_half_skew = width_curve(vB_skew, idcB, 0.5 * OMEGA, TOL)
    w_half_sin = width_curve(vB_sin, idcB, 0.5 * OMEGA, TOL)
    k_best = int(np.argmax(w_half_skew))
    w_max = w_half_skew[k_best]
    assert abs(w_max - 0.250) <= 0.03
    assert abs(iacB[k_best] - 4.0) <= 0.5
    # contrast vs the sinusoidal tolerance floor
    floor = w_half_sin.max()
    assert w_max >= 5.0 * floor - 1e-9
    # second lobe: node (minimum) near i_ac ~ 9.5, width recovering to ~0.15
    # by i_ac = 12 (see paper/decisions.md: brief text says "near 9-10", the
    # reference artifact places the node at 9.5 and the lobe beyond it)
    node_win = (iacB >= 8.0) & (iacB <= 11.0)
    k_node = np.where(node_win)[0][int(np.argmin(w_half_skew[node_win]))]
    assert abs(iacB[k_node] - 9.5) <= 1.0
    assert w_half_skew[k_node] <= 0.05
    lobe2 = w_half_skew[(iacB > iacB[k_node]) & (iacB <= 12.0)]
    assert lobe2.max() >= 0.125
    # reproduction of the reference curves (grid quantum 0.025)
    assert np.allclose(w_half_skew, ref["stageB"]["w_half_skew"], atol=0.026)
    assert abs(w_max - ref["stageB"]["w_half_max"]) <= 0.026
