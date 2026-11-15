"""Quasiparticle back-action envelope (M4_BRIEF_DRAFT.md AT-M4-4, deliverable 3).

Closes the loop from absorbed cavity power to the driven-junction gap: the
absorbed power breaks Cooper pairs, the quasiparticle density rises, the gap
Delta is suppressed, and because the 145 GHz drive is locked to 2*Delta0/h the
suppression detunes the drive from twice the gap. This module maps the whole
chain onto a single dimensionless load and exposes the reduced gap and the
drive-detuning threshold parametrically.

Reuse, do not reimplement
-------------------------
The Owen-Scalapino linearized pair-breaking kinetics live in
`junction.tien_gordon.steady_state_delta` (validated by the M3 test suite).
This module *imports* that steady state and only re-parameterizes its inputs;
it never re-solves the kinetics itself.

Dimensionless load
------------------
    L = eta_abs * P_abs * tau_rec / (n0 * Delta0).

The tien_gordon steady state solves, with x = n_qp / n0 and d = Delta / Delta0,

    n_qp* = sqrt(eta_abs * P_abs * n0 * tau_rec / Delta),
    Delta = Delta0 * (1 - n_qp* / (4 n0)),

which in reduced variables is exactly

    x^2 = L / d ,     d = 1 - x / 4 .

Both reduced equations depend on the inputs only through the product L, so the
reduced gap d = Delta/Delta0 is a function of L alone. `delta_vs_load` maps a
requested L onto the steady-state arguments using reference units (any choice
of the individual knobs that reproduces L gives the same d), and the
L-invariance is verified numerically in the test suite to 1e-12.

Detuning
--------
The drive frequency is locked to f0 = 2*Delta0/h. After suppression the pair
resonance sits at 2*Delta(L)/h, so the fractional detuning is

    detuning_fraction(L) = (f0 - 2*Delta(L)/h) / f0 = 1 - Delta(L)/Delta0,

i.e. it equals the reduced gap suppression. The drive falls outside the loaded
half-linewidth once this exceeds 1/(2Q); `threshold_load(Q)` returns the load
at that boundary. Q is a free parameter here (no locked Q exists until M2
delivers photonics.json) and the threshold is evaluated for a caller-supplied
set of Q values.
"""
import pathlib
import sys

import numpy as np
from scipy.optimize import brentq

# Make the read-only M3 junction package importable (documented import pattern).
_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from junction.tien_gordon import steady_state_delta  # noqa: E402

# Convergence controls for the reused steady state. Tightened well below the
# 1e-12 back-action acceptance target so the reduced gap is fixed by L to
# machine precision regardless of the absolute (eta_abs, P_abs, ...) scale,
# which is what makes the L-invariance exact to 1e-12.
_STEADY_TOL = 1e-14
_STEADY_MAXITER = 5000


def load(eta_abs, P_abs, tau_rec, n0, Delta0):
    """Dimensionless back-action load L = eta_abs*P_abs*tau_rec/(n0*Delta0).

    This is the single scalar that controls the reduced gap; supplied here so
    callers document the definition rather than open-coding it. Vectorized over
    any array-valued argument via numpy broadcasting.
    """
    return eta_abs * np.asarray(P_abs, float) * tau_rec / (n0 * Delta0)


def delta_vs_load(L_grid, eta_abs=1.0, tau_rec=1.0, n0=1.0, Delta0=1.0):
    """Self-consistent reduced gap Delta/Delta0 as a function of the load L.

    Maps each requested L onto the `steady_state_delta` arguments by inverting
    the load definition, P_abs = L * n0 * Delta0 / (eta_abs * tau_rec), then
    calls the reused steady state and returns Delta/Delta0. The extra knobs
    (eta_abs, tau_rec, n0, Delta0) default to reference units and exist only so
    the L-invariance can be exercised with different (eta_abs, P_abs, ...) at
    fixed L; the result depends on those knobs only through L.

    L = 0 returns exactly 1.0. The reduced gap decreases monotonically with L.
    Scalar in -> float out; array in -> ndarray out.
    """
    L = np.asarray(L_grid, dtype=float)
    scalar = (L.ndim == 0)
    P_abs = L * n0 * Delta0 / (eta_abs * tau_rec)
    Delta = steady_state_delta(eta_abs, P_abs, tau_rec, n0, Delta0,
                               tol=_STEADY_TOL, max_iter=_STEADY_MAXITER)
    ratio = np.asarray(Delta, dtype=float) / Delta0
    return float(ratio) if scalar else ratio


def detuning_fraction(L_grid, eta_abs=1.0, tau_rec=1.0, n0=1.0, Delta0=1.0):
    """Fractional drive detuning (f0 - 2*Delta(L)/h)/f0 = 1 - Delta(L)/Delta0.

    Because f0 = 2*Delta0/h is locked, the fractional detuning reduces exactly
    to the gap suppression 1 - Delta/Delta0. Same shape semantics as
    `delta_vs_load`.
    """
    return 1.0 - delta_vs_load(L_grid, eta_abs, tau_rec, n0, Delta0)


def _threshold_closed_form(Q):
    """Closed-form threshold load, for cross-checking the root find.

    At threshold the detuning equals 1/(2Q), so the reduced gap is
    d = 1 - 1/(2Q) and x = 4(1 - d) = 2/Q. With L = x^2 * d this gives

        L_thr = (2/Q)^2 * (1 - 1/(2Q)) = (4/Q^2) * (1 - 1/(2Q)).
    """
    half_lw = 1.0 / (2.0 * Q)
    return (4.0 / Q ** 2) * (1.0 - half_lw)


def threshold_load(Q):
    """Smallest load L at which the detuning exceeds the half-linewidth 1/(2Q).

    The detuning is continuous and strictly increasing from 0, so the smallest
    L that *exceeds* 1/(2Q) is the L at which it *equals* 1/(2Q). That boundary
    is found by a bracketed root of detuning_fraction(L) - 1/(2Q), which is
    built on the reused steady state. Q is a free parameter (no locked Q); pass
    any positive value. `threshold_load` is strictly decreasing in Q.
    """
    Q = float(Q)
    target = 1.0 / (2.0 * Q)

    def g(L):
        return detuning_fraction(L) - target

    # g(0) = -target < 0; expand the upper bracket until the detuning exceeds
    # the target, then locate the crossing to machine precision.
    L_hi = 1e-12
    g_lo = g(0.0)
    while g(L_hi) < 0.0:
        L_hi *= 2.0
        if L_hi > 1.0:
            raise ValueError(
                f"detuning never reaches 1/(2Q) for Q={Q!r} within L<=1 "
                "(gap suppression saturates before the linewidth is exceeded)"
            )
    assert g_lo < 0.0  # sanity: zero load gives zero detuning
    # rtol must stay above scipy's floor of 4*eps (~8.88e-16); this drives the
    # root to ~machine-precision relative accuracy on the (tiny) threshold L.
    return brentq(g, 0.0, L_hi, xtol=1e-30, rtol=1.0e-15, maxiter=200)


def backaction_envelope(L_grid, Q_values):
    """Assemble the exportable back-action envelope over a load grid.

    Returns a dict with the reduced gap, the fractional detuning, and the
    per-Q threshold loads (Q a free parameter). Deterministic; no RNG.
    """
    L = np.asarray(L_grid, dtype=float)
    return {
        "L": L,
        "delta_over_delta0": delta_vs_load(L),
        "detuning_fraction": detuning_fraction(L),
        "Q_values": np.asarray(Q_values, dtype=float),
        "threshold_load": np.array([threshold_load(Q) for Q in Q_values]),
    }
