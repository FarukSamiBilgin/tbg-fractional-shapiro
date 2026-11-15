"""M1 material module: Bistritzer-MacDonald bands, Beenakker CPR, SC dome.

Faithful package port of m1_reference.py. Produces the numbers locked in
material.json (v0.2) and consumed by M3/M4.
"""

from m1.bm import (
    KB_MEV_K, A_G, KD, WAB, WAA, RATIO,
    geometry, basis, hamiltonian, flat_band_width, band_path, dirac_flat_eigs,
)
from m1.cpr import beenakker_harmonics
from m1.dome import (
    TC_MAX, NU_OPT, NU_HW,
    Tc_of_nu, Delta_of, nu_grid, build_nu_table, build_material,
)

__all__ = [
    "KB_MEV_K", "A_G", "KD", "WAB", "WAA", "RATIO",
    "geometry", "basis", "hamiltonian", "flat_band_width", "band_path",
    "dirac_flat_eigs", "beenakker_harmonics",
    "TC_MAX", "NU_OPT", "NU_HW",
    "Tc_of_nu", "Delta_of", "nu_grid", "build_nu_table", "build_material",
]
