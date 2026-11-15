"""M2 AT-1 with the revised, measured pass criterion (paper/decisions.md
2026-07-23 "PWE table truncation bias"): the locked PWE table (N=5, 121 plane
waves) carries 2-6% truncation error, largest on the upper gap edge, so the
brief's naive "MPB within 3% of the table" cannot hold for a converged MPB.
Revised AT-1:
  (a) MPB is internally converged (res 32 -> 48 midgap drift < 0.1%),
  (b) PWE edges drift monotonically toward MPB as N grows, with the N=13
      residual < 2.5% on both edges,
  (c) the N=5 lower edge is within 3% of MPB (the table part that survives),
  (d) TM shows no comparable gap (TM gap < 5% of the TE gap).
Artifact-based: reads out/m2/at1_mpb.json (MPB run inside the m2env WSL
distro) and out/m2/pwe_convergence.json; skipped if the artifacts are absent.
"""
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
AT1 = ROOT / "out" / "m2" / "at1_mpb.json"
CONV = ROOT / "out" / "m2" / "pwe_convergence.json"

pytestmark = pytest.mark.skipif(
    not (AT1.exists() and CONV.exists()),
    reason="M2 AT-1 artifacts not present (MPB run not done on this machine)")


@pytest.fixture(scope="module")
def at1():
    return json.loads(AT1.read_text())


@pytest.fixture(scope="module")
def conv():
    return json.loads(CONV.read_text())


@pytest.mark.parametrize("eps", ["11.68", "8.12"])
def test_mpb_internally_converged(at1, eps):
    assert at1["cases"][eps]["midgap_convergence"] < 1e-3


@pytest.mark.parametrize("eps", ["11.68", "8.12"])
def test_pwe_drifts_monotonically_toward_mpb(conv, eps):
    case = conv["cases"][eps]
    edges = np.asarray(case["pwe_edges_by_N"], float)   # (nN, 2)
    mpb_lo, mpb_hi = case["mpb_edges_res48"]
    # distances to the MPB result must decrease with N on both edges
    d_lo = np.abs(edges[:, 0] - mpb_lo)
    d_hi = np.abs(edges[:, 1] - mpb_hi)
    assert np.all(np.diff(d_lo) < 0)
    assert np.all(np.diff(d_hi) < 0)
    # N=13 residuals below 2.5%
    assert d_lo[-1] / mpb_lo < 0.025
    assert d_hi[-1] / mpb_hi < 0.025


@pytest.mark.parametrize("eps", ["11.68", "8.12"])
def test_n5_lower_edge_within_3pct(at1, eps):
    case = at1["cases"][eps]
    assert case["rel_dev_lo"] <= 0.03


@pytest.mark.parametrize("eps", ["11.68", "8.12"])
def test_tm_has_no_comparable_gap(at1, eps):
    case = at1["cases"][eps]
    assert case["tm_gap"] < 0.05 * case["te_gap"]
