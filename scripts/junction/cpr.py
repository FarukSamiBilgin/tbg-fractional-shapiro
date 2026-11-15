"""Current-phase relation (CPR) helpers.

Normalization rule (M3_BRIEF.md Sec. 2): rescale raw harmonics so that
max_phi sum_m c_m sin(m phi) = 1, i.e. current is in units of the true Ic.
"""
import numpy as np


def cpr(phi, c):
    """I(phi)/Ic = sum_m c_m sin(m phi), m starting at 1."""
    f = np.zeros_like(phi)
    for m, cm in enumerate(c, start=1):
        if cm != 0.0:
            f += cm * np.sin(m * phi)
    return f


def normalize_cpr(c_raw):
    """Rescale harmonic coefficients so that max_phi I(phi) = 1.

    Returns (c_normalized, renorm_factor). Grid of 40001 points over
    [0, 2pi] matches the reference implementation exactly.
    """
    c_raw = np.asarray(c_raw, float)
    phi = np.linspace(0.0, 2.0 * np.pi, 40001)
    fmax = cpr(phi, c_raw).max()
    return c_raw / fmax, fmax


C_SIN, _ = normalize_cpr([1.0])                  # sinusoidal benchmark CPR
C_SKEW, SKEW_RENORM = normalize_cpr([1.0, -0.25])  # forward-skewed CPR (high-transparency TBG link)
