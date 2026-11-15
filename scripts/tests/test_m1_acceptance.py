"""M1 acceptance tests AT-1..AT-4 (M1_BRIEF.md Sec. 3).

Deterministic (no RNG). Every test recomputes the band/CPR/dome numerics from
the m1 package port and compares against both the brief targets and the
provided material.json (v0.2, ground truth). Session-scoped fixtures share the
one expensive piece of work -- the W(theta) magic-angle scan.
"""
import json
import pathlib

import numpy as np
import pytest

from m1 import bm, cpr, dome

ROOT = pathlib.Path(__file__).resolve().parents[1]

# brief AT-1 scan grid
THETAS = np.arange(0.90, 1.201, 0.025)
HBV_MAIN = 620.0
HBV_HI = 658.0    # companion -> magic angle 0.975 deg
HBV_LO = 601.0    # companion -> magic angle 1.075 deg
THETA_DESIGN = 1.025


@pytest.fixture(scope="session")
def mat():
    """Provided material.json -- ground truth for acceptance."""
    with open(ROOT / "material.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def scan():
    """W(theta) over the AT-1 grid for the three hbv candidates (N=3, 7x7)."""
    out = {}
    for hbv in (HBV_MAIN, HBV_HI, HBV_LO):
        out[hbv] = np.array([bm.flat_band_width(th, hbv, N=3, grid=7)[0]
                             for th in THETAS])
    return out


@pytest.fixture(scope="session")
def magic_bandwidths():
    """W at theta_design for N=3/N=4 and the +-0.05 deg disorder window."""
    Wm_N3, _ = bm.flat_band_width(THETA_DESIGN, HBV_MAIN, N=3)
    Wm_N4, _ = bm.flat_band_width(THETA_DESIGN, HBV_MAIN, N=4)
    Wq, _ = bm.flat_band_width(THETA_DESIGN - 0.05, HBV_MAIN, N=3)
    Wp, _ = bm.flat_band_width(THETA_DESIGN + 0.05, HBV_MAIN, N=3)
    return {"N3": Wm_N3, "N4": Wm_N4, "minus": Wq, "plus": Wp}


# ---------------------------------------------------------------- AT-1
def test_at1_magic_angle_dip(scan):
    """AT-1: bandwidth minimum lands at theta_design with W_min = 1.8 +- 0.3."""
    W = scan[HBV_MAIN]
    theta_min = THETAS[np.argmin(W)]
    assert abs(theta_min - THETA_DESIGN) <= 0.025
    assert abs(W.min() - 1.8) <= 0.3


def test_at1_companion_magic_angles(scan):
    """AT-1 companions: hbv=658 -> 0.975 deg, hbv=601 -> 1.075 deg (+-0.025)."""
    assert abs(THETAS[np.argmin(scan[HBV_HI])] - 0.975) <= 0.025
    assert abs(THETAS[np.argmin(scan[HBV_LO])] - 1.075) <= 0.025


def test_at1_dirac_touch():
    """AT-1: at K_m=(0,0), theta=1.025, N=4 the two flat bands straddle E=0
    and touch to < 0.1 meV."""
    lo, hi = bm.dirac_flat_eigs(THETA_DESIGN, HBV_MAIN, N=4, k=(0.0, 0.0))
    assert lo <= 0.0 <= hi                 # straddle
    assert abs(lo) < 0.1 and abs(hi) < 0.1  # touch to < 0.1 meV


# ---------------------------------------------------------------- AT-2
def test_at2_disorder_sensitivity(magic_bandwidths, mat):
    """AT-2: W(1.025 +- 0.05) reproduces material.json W_pm_0p05deg_meV within
    15% -- and, being the same algorithm, to 1e-6 after the port's own rounding."""
    Wq, Wp = magic_bandwidths["minus"], magic_bandwidths["plus"]
    ref_minus, ref_plus = mat["flat_band"]["W_pm_0p05deg_meV"]  # [7.408, 7.138]
    # ~4x broadening lands in the 7.1-7.4 meV window
    assert 7.1 <= Wq <= 7.5 and 7.1 <= Wp <= 7.5
    # within 15% of the provided numbers
    assert abs(Wq - ref_minus) / ref_minus <= 0.15
    assert abs(Wp - ref_plus) / ref_plus <= 0.15
    # same algorithm -> port's rounded values match json to 1e-6
    assert abs(round(float(Wq), 3) - ref_minus) < 1e-6
    assert abs(round(float(Wp), 3) - ref_plus) < 1e-6


# ---------------------------------------------------------------- AT-3
def test_at3_cpr_ratios():
    """AT-3: max-normalized Beenakker c2/c1 and c3/c1 at tau = 0.5/0.8/0.95."""
    targets = {0.5: (-0.085, 0.011), 0.8: (-0.187, 0.053), 0.95: (-0.299, 0.138)}
    for tau, (r21, r31) in targets.items():
        c = cpr.beenakker_harmonics(tau)
        assert abs(c[1] / c[0] - r21) <= 0.005
        assert abs(c[2] / c[0] - r31) <= 0.005


def test_at3_cpr_full_coefficients(mat):
    """AT-3: full coefficient lists match material.json harmonics_by_tau to 1e-9."""
    for tau in (0.5, 0.8, 0.95):
        c = cpr.beenakker_harmonics(tau)
        ref = np.asarray(mat["cpr"]["harmonics_by_tau"][str(tau)], float)
        assert c.shape == ref.shape
        assert np.max(np.abs(c - ref)) < 1e-9


# ---------------------------------------------------------------- AT-4
def test_at4_nu_table_row_for_row(mat):
    """AT-4: dome/nu_table builder regenerates material.json nu_table (1e-6)."""
    built = dome.build_nu_table()
    ref = mat["nu_table"]
    assert len(built) == len(ref)
    for row, ref_row in zip(built, ref):
        # v0.3 adds cpr_harmonics per SC row (brief production item 4);
        # the v0.2 builder schema must remain a subset (decisions.md 2026-07-23)
        assert set(row.keys()) <= set(ref_row.keys())
        for key in ("nu", "Tc_K", "Delta_meV", "Ic_nA"):
            assert abs(row[key] - ref_row[key]) < 1e-6
        if ref_row["Ic_nA"] > 0:
            assert "cpr_harmonics" in ref_row
            assert np.allclose(ref_row["cpr_harmonics"],
                               mat["cpr"]["harmonics_by_tau"]["0.8"], atol=1e-12)


def test_at4_schema_keys(magic_bandwidths, mat):
    """AT-4: regenerated material dict matches provided material.json key-for-key."""
    mb = magic_bandwidths
    built = dome.build_material(
        mat["bm_params"]["theta_design_deg"], HBV_MAIN,
        mb["N3"], mb["N4"], mb["minus"], mb["plus"])

    def keyset(d, prefix=""):
        ks = set()
        if isinstance(d, dict):
            for k, v in d.items():
                ks.add(prefix + k)
                ks |= keyset(v, prefix + k + ".")
        return ks

    # v0.3 evolves the schema additively (version bump, remote_band_gap,
    # fine_theta_scan, per-row cpr_harmonics — brief production items 1-4).
    # AT-4 therefore checks that the v0.2 builder schema survives as a strict
    # subset and that the sanctioned v0.3 additions are present.
    built_keys = keyset(built)
    mat_keys = keyset(mat)
    extra = {k for k in built_keys - mat_keys if not k.startswith("nu_table.")}
    assert not extra, f"v0.2 keys missing from material.json: {extra}"
    for required in ("remote_band_gap", "fine_theta_scan", "flat_band.W_v02_7x7"):
        assert any(k == required or k.startswith(required + ".")
                   for k in mat_keys), f"v0.3 addition missing: {required}"
    # every v0.2 top-level key still present, in the same relative order
    idx = {k: i for i, k in enumerate(mat.keys())}
    positions = [idx[k] for k in built.keys() if k in idx]
    assert positions == sorted(positions)
    assert mat["contract_version"] == "0.3"
