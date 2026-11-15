"""Beenakker short-junction current-phase relation harmonics (feeds M3).

Faithful port of m1_reference.beenakker_harmonics: 8192-point phi grid,
Andreev-bound-state occupation via a tanh(EA / 2kT) factor, and M3's
max-normalization convention (scale so max_phi sum_m c_m sin((m+1) phi) = 1).
"""

import numpy as np

from m1.bm import KB_MEV_K


def beenakker_harmonics(tau, Delta_meV=0.304, T_K=0.1, nH=5):
    """Fourier sine harmonics c_1..c_nH of the Beenakker CPR at transparency tau.

    I(phi) = tau sin(phi) / sqrt(1 - tau sin^2(phi/2)) * tanh(EA / 2 kB T),
    EA(phi) = Delta sqrt(1 - tau sin^2(phi/2)).
    Coefficients c_m = 2 <I sin((m+1) phi)> are max-normalized so the
    reconstructed CPR peaks at 1 (M3 convention).
    """
    phi = np.linspace(0, 2 * np.pi, 8192, endpoint=False)
    EA = Delta_meV * np.sqrt(1.0 - tau * np.sin(phi / 2.0) ** 2)
    I = tau * np.sin(phi) / np.sqrt(1.0 - tau * np.sin(phi / 2.0) ** 2)
    I *= np.tanh(EA / (2.0 * KB_MEV_K * T_K))
    c = [2.0 * np.mean(I * np.sin((m + 1) * phi)) for m in range(nH)]
    c = np.array(c)
    # normalize so that max_phi sum_m c_m sin((m+1) phi) = 1  (M3 convention)
    f = sum(cm * np.sin((m + 1) * phi) for m, cm in enumerate(c))
    return c / f.max()
