"""Thermal budget (M4_BRIEF_DRAFT.md deliverable 4).

Two independent thermal checks for the driven junction:

1. Junction-local electron temperature under Joule/absorbed load, using the
   graphene-like electron-phonon cooling law
       P = Sigma_V * (T_e^delta - T_ph^delta),
   which inverts in closed form to
       T_e = (P / Sigma_V + T_ph^delta)^(1/delta).

2. Chip-level budget: the dissipated junction power P_J against the fridge
   cooling power at the operating stage.

Free parameters (not locked by any brief)
-----------------------------------------
* Sigma_V -- the lumped electron-phonon coupling times the junction electron
  volume, in W / K^delta. This is the product Sigma * Volume that sets the
  cooling strength; it is a FREE PARAMETER pending experimental calibration and
  must be supplied by the caller. The curves here are parametric in Sigma_V and
  make no locked quantitative claim.
* delta_exp -- the electron-phonon cooling exponent. Default 5 (clean-graphene,
  acoustic-phonon / deformation-potential limit); disordered graphene can flatten
  toward 3-4. Exposed so the exponent can be varied; also parametric.
* P_fridge -- fridge cooling power at the operating stage. Default 500 uW is an
  order-of-magnitude figure for a dilution-refrigerator mixing chamber near
  100 mK (documented as such in the brief), not a locked number.

Only the P_J = 90 pW load (locked via integration.P_J_REF_W = A_TARGET * P_in)
and the physical constants come from upstream; everything thermal is parametric.
Deterministic; no RNG.
"""
from collections import namedtuple

import numpy as np

# Default fridge cooling power (order-of-magnitude, documented in the brief).
P_FRIDGE_DEFAULT_W = 500.0e-6

ChipBudget = namedtuple("ChipBudget", ["ratio", "ok"])


def te_from_power(P, Sigma_V, T_ph, delta_exp=5):
    """Electron temperature from dissipated power (graphene-like e-ph cooling).

    Inverts P = Sigma_V*(T_e^delta - T_ph^delta):

        T_e = (P / Sigma_V + T_ph^delta)^(1/delta).

    Sigma_V (W/K^delta) and delta_exp are free parameters (see module docstring).
    P = 0 returns T_ph exactly (thermal equilibrium, by definition); T_e is
    strictly increasing in P. Requires P >= 0. Scalar in -> float out; array in
    -> ndarray out.
    """
    P = np.asarray(P, dtype=float)
    scalar = (P.ndim == 0)
    inside = P / Sigma_V + float(T_ph) ** delta_exp
    Te = inside ** (1.0 / delta_exp)
    # At zero load the electron and phonon baths are in equilibrium exactly,
    # independent of any round-off in the power/root of the general formula.
    Te = np.where(P == 0.0, float(T_ph), Te)
    return float(Te) if scalar else Te


def power_from_te(T_e, Sigma_V, T_ph, delta_exp=5):
    """Dissipated power required to hold electrons at T_e (inverse of te_from_power).

        P = Sigma_V * (T_e^delta - T_ph^delta).

    Exact left inverse of `te_from_power` in exact arithmetic; the numerical
    round-trip P -> T_e -> P closes to ~1e-12 relative for loads where
    P/Sigma_V is at least comparable to the phonon floor T_ph^delta.
    """
    T_e = np.asarray(T_e, dtype=float)
    scalar = (T_e.ndim == 0)
    P = Sigma_V * (T_e ** delta_exp - float(T_ph) ** delta_exp)
    return float(P) if scalar else P


def chip_budget(P_J, P_fridge=P_FRIDGE_DEFAULT_W):
    """Chip-level thermal budget: dissipated junction power vs fridge cooling.

    Returns a ChipBudget(ratio, ok) namedtuple where ratio = P_J / P_fridge and
    ok is True when the junction load sits below the available cooling power
    (ratio < 1). At the locked operating point P_J = 90 pW against the default
    500 uW fridge the ratio is 1.8e-7 -- a seven-order-of-magnitude margin.
    """
    ratio = float(P_J) / float(P_fridge)
    return ChipBudget(ratio=ratio, ok=bool(ratio < 1.0))
