"""T1/T2 thermometry acceptance tests.

Covers the two new artifacts:
  out/thermometry_calibration.json  (scripts/run_thermometry_calibration.py)
  out/thermometer.json              (scripts/make_thermometer.py)

and the two pieces of machinery they rest on that are not otherwise exercised
by the M3 suite: the soft (kernel-integral) width estimator and the local
shape-preserving PCHIP interpolant used to invert the calibration.

Artifact-dependent tests skip when the artifact has not been generated, so the
file is safe to run on a clean checkout.
"""
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def _load(script_name):
    """Import a scripts/*.py module by path (scripts/ is not a package)."""
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def t1mod():
    return _load("run_thermometry_calibration.py")


@pytest.fixture(scope="session")
def t2mod():
    return _load("make_thermometer.py")


@pytest.fixture(scope="session")
def cal():
    p = OUT / "thermometry_calibration.json"
    if not p.exists():
        pytest.skip("out/thermometry_calibration.json not generated")
    d = json.loads(p.read_text())
    # a --n-real 2 plumbing run writes the same filename; its SEMs are ~10x the
    # production ones and would fail the precision tests for the wrong reason.
    if d["numerics"]["n_real_signal"] < 20:
        pytest.skip("calibration artifact is a smoke run (n_real < 20)")
    return d


@pytest.fixture(scope="session")
def thermo():
    p = OUT / "thermometer.json"
    if not p.exists():
        pytest.skip("out/thermometer.json not generated")
    return json.loads(p.read_text())


# ---------------------------------------------------------------- estimator
def test_soft_width_matches_analytic_transit(t1mod):
    """On a pure linear transit through v_target the kernel integral is exactly
    sigma_k*sqrt(2*pi)/slope. This is the estimator's irreducible floor, so its
    value has to be right, not merely stable."""
    sk = t1mod.SIGMA_K
    idc = np.linspace(-4.0, 4.0, 40001)
    for slope in (0.5, 1.0, 2.5):
        v = t1mod.V_TARGET + slope * idc
        got = t1mod.soft_width(v, idc)
        want = sk * np.sqrt(2 * np.pi) / slope
        assert abs(got - want) / want < 1e-6


def test_production_grid_resolves_the_kernel(t1mod):
    """The soft estimator is a quadrature: it is only meaningful if the i_dc
    grid samples the kernel. On the ohmic branch (slope ~ 1 in v per i_dc) the
    kernel is sigma_k wide in i_dc, so require at least 5 grid points across
    it. This is the condition the next test then verifies numerically."""
    assert t1mod.SIGMA_K / t1mod.IDC_STEP >= 5.0


def test_soft_width_is_grid_convergent(t1mod):
    """The soft estimator must converge under grid refinement -- the whole
    reason it replaces the plateau count, which quantises into multiples of the
    i_dc step and cannot converge. Uses a transit of unit slope, i.e. the real
    ohmic regime the production grid was sized for."""
    v_of = lambda i: t1mod.V_TARGET + np.tanh((i - 2.6) / 0.4) * 0.4
    vals = []
    steps = (0.01, 0.005, 0.0025, 0.00125)
    for step in steps:
        idc = np.arange(2.1, 3.1 + 1e-9, step)
        vals.append(t1mod.soft_width(v_of(idc), idc))
    vals = np.array(vals)
    d = np.abs(np.diff(vals))
    # Trapezoidal quadrature of a smooth Gaussian is spectrally accurate, so on
    # a resolved transit the estimator is converged to round-off at every step
    # tried here. The assertion is deliberately at 1e-9 rather than at machine
    # epsilon: it must catch a genuinely unconverged estimator, not police the
    # last bit of floating-point noise.
    assert np.all(d < 1e-9), f"refinement deltas {d}"
    assert abs(vals[0] - vals[-1]) < 1e-9


def test_soft_width_zero_far_from_target(t1mod):
    idc = np.linspace(2.1, 3.1, 201)
    v = np.full_like(idc, t1mod.V_TARGET + 1.0)     # 40 sigma_k away
    assert t1mod.soft_width(v, idc) < 1e-12


def test_soft_width_is_window_invariant(t1mod):
    """The reference filling uses a pinned legacy i_dc window while every other
    filling gets one centred by build_config(). Those windows differ by ~0.04,
    so the estimator has to be insensitive to the choice or the two families of
    calibration would not be comparable.

    The invariance is good to ~0.1 %, not to round-off: v(i_dc) has kinks at the
    plateau edges, so shifting the grid relative to those kinks moves the
    quadrature slightly (unlike the smooth-transit case, where it is exact to
    round-off). 0.1 % sits an order of magnitude below the calibration's own SEM
    (~1 %) and well below the 2 % measurement precision the thermometer quotes,
    so it is a bounded systematic rather than a defect -- but it is asserted at
    its true size, not at zero."""
    from junction.cpr import normalize_cpr
    from junction.rsj import run_rsj

    mat = json.loads((ROOT / "material.json").read_text())
    c, _ = normalize_cpr(mat["cpr"]["harmonics_by_tau"]["0.8"])
    c = np.asarray(c, float)

    vals = []
    for lo, hi in ((2.10, 3.10), (2.1375, 3.1375), (2.30, 2.95)):
        idc = np.arange(lo, hi + 1e-9, t1mod.IDC_STEP)
        v = run_rsj(idc, np.full_like(idc, t1mod.I_AC), c)
        vals.append(t1mod.soft_width(v, idc))
    vals = np.array(vals)
    rel = np.ptp(vals) / vals.mean()
    assert rel < 5e-3, f"window-dependent at {rel:.2%}: {vals}"


# ---------------------------------------------------------------- nu extension
def test_build_config_reproduces_reference_filling(t1mod):
    """--nu -2.4 must regenerate the pinned production configuration exactly,
    otherwise the extension runs would silently invalidate the blocks already
    computed for the reference filling."""
    cfg = t1mod.build_config(t1mod.NU_REF)
    assert np.allclose(cfg["idc"], t1mod.IDC)
    assert cfg["i_ac"] == t1mod.I_AC
    assert cfg["omega"] == 5.0
    assert cfg["n_transient"] == t1mod.N_TRANSIENT
    assert cfg["n_avg"] == t1mod.N_AVG
    assert cfg["v_target"] == t1mod.V_TARGET
    assert cfg["is_reference"] is True


def test_extension_seeds_never_collide(t1mod):
    """Every (branch, Gamma, nu) must get its own noise realisation."""
    seen = {}
    for nu in (-2.4, -2.5, -2.6, -2.7, -2.8):
        for g in t1mod.GAMMAS:
            for br in ("skewed", "sin"):
                s = t1mod.gamma_seed(g, br, nu)
                assert s not in seen, f"{(g, br, nu)} collides with {seen[s]}"
                seen[s] = (g, br, nu)


@pytest.mark.parametrize("nu", [-2.5, -2.6, -2.7, -2.8])
def test_extension_window_brackets_the_step(t1mod, nu):
    """The derived window must contain that filling's deterministic half-step
    with at least 4 sigma_k of margin on both sides, else the kernel integral is
    truncated and the calibration is biased."""
    fp = ROOT / "out" / "fingerprint" / f"{t1mod.nu_slug(nu)}.npz"
    if not fp.exists():
        pytest.skip(f"{fp} not generated")
    cfg = t1mod.build_config(nu)
    d = np.load(fp)
    iac, idc, v = d["i_ac"], d["i_dc"], d["v"]
    k = int(np.argmin(np.abs(iac - cfg["i_ac"])))
    vv = v[k] if v.shape[0] == len(iac) else v[:, k]
    on = np.abs(vv - cfg["v_target"]) < t1mod.TOL
    margin = 4 * t1mod.SIGMA_K
    assert cfg["idc_lo"] < idc[on].min() - margin
    assert cfg["idc_hi"] > idc[on].max() + margin


@pytest.mark.parametrize("nu", [-2.5, -2.6, -2.7, -2.8])
def test_extension_holds_v_resolution(t1mod, nu):
    """n_avg scales with Omega precisely so the v-resolution stays put and the
    fixed tolerance 0.025 keeps meaning 1.5 resolutions at every filling."""
    if not (ROOT / "out" / "fingerprint" / f"{t1mod.nu_slug(nu)}.json").exists():
        pytest.skip("fingerprint not generated")
    cfg = t1mod.build_config(nu)
    assert abs(cfg["v_res"] - 5.0 / 300.0) < 5e-4
    assert 1.4 < t1mod.TOL / cfg["v_res"] < 1.6


# ---------------------------------------------------------------- interpolant
def test_pchip_matches_scipy(t2mod):
    """The local PCHIP must agree with the reference implementation in value and
    first derivative -- the derivative is what sets the propagated sigma_T."""
    scipy_interp = pytest.importorskip("scipy.interpolate")
    rng = np.random.default_rng(20260804)
    for _ in range(5):
        x = np.unique(np.sort(rng.uniform(-5.0, 1.0, 9)))
        y = np.sort(rng.uniform(0.0, 1.0, len(x)))[::-1]
        f = t2mod.pchip(x, y)
        g = scipy_interp.PchipInterpolator(x, y)
        xq = np.linspace(x[0], x[-1], 401)
        assert np.max(np.abs(f(xq) - g(xq))) < 1e-12
        assert np.max(np.abs(f(xq, deriv=1) - g(xq, nu=1))) < 1e-10


def test_pchip_preserves_monotonicity(t2mod):
    """A monotone-decreasing calibration must not acquire a local rise; a plain
    cubic spline would overshoot on a saturating curve and make the inverse
    multi-valued."""
    x = np.log(np.array([0.005, 0.007, 0.010, 0.014, 0.021, 0.028, 0.042, 0.070]))
    y = np.array([1.00, 0.92, 0.80, 0.60, 0.42, 0.30, 0.14, 0.03])
    f = t2mod.pchip(x, y)
    xq = np.linspace(x[0], x[-1], 2001)
    assert np.all(np.diff(f(xq)) <= 1e-12)


def test_invert_round_trips(t2mod):
    x = np.log(np.array([0.005, 0.010, 0.020, 0.040, 0.080]))
    y = np.array([1.0, 0.75, 0.5, 0.25, 0.05])
    f = t2mod.pchip(x, y)
    for target in (0.9, 0.6, 0.3, 0.1):
        u = t2mod.invert(f, target, x[0], x[-1])
        assert u is not None
        assert abs(t2mod._at(f, u) - target) < 1e-8
    assert t2mod.invert(f, 5.0, x[0], x[-1]) is None      # outside range


# ---------------------------------------------------------------- physics
def test_ej_over_kb_matches_contract(t2mod):
    """M3_BRIEF Sec. 1: E_J/k_B = 0.715 K at Ic = 30 nA; contract v0.3 scales
    it x10 with the x10 junction widening."""
    assert abs(t2mod.ej_over_kb(30.0) - 0.715) < 5e-4
    assert abs(t2mod.ej_over_kb(300.0) - 7.15) < 5e-3


# ---------------------------------------------------------------- artifacts
def test_calibration_reproduces_at1_floor(cal):
    """The sinusoidal control's HARD width on the refined grid is the AT-1
    artifact floor and must still satisfy the brief's <= 0.05 bound."""
    assert cal["deterministic_reference"]["w_hard_sinusoidal"] <= 0.05


def test_calibration_reproduces_fingerprint_width(cal):
    """The deterministic skewed hard width on the refined grid must reproduce
    the 0.200 of out/fingerprint/nu_m2_4.json to within one coarse (0.025)
    grid cell -- the refinement corrects the plateau count, it does not move
    the physics."""
    assert abs(cal["deterministic_reference"]["w_hard_skewed"] - 0.200) <= 0.025


def test_control_subtraction_removes_the_floor(cal):
    """The net deterministic width must be materially smaller than the raw soft
    width: if it were not, the control would not be subtracting anything."""
    d = cal["deterministic_reference"]
    assert d["w_soft_net"] == pytest.approx(d["w_soft_skewed"] - d["w_soft_sinusoidal"])
    assert 0.25 < d["w_soft_sinusoidal"] / d["w_soft_skewed"] < 0.45


def test_calibration_is_monotone_and_invertible(cal):
    """The forward map has to be single-valued to be inverted. Rises within one
    combined SEM are noise and are tolerated; anything larger is reported by T1
    and must not be present in the range the thermometer claims to use."""
    g = np.array([e["Gamma"] for e in cal["per_gamma"]])
    w = np.array([e["soft_net_mean"] for e in cal["per_gamma"]])
    sem = np.array([e["soft_net_sem"] for e in cal["per_gamma"]])
    assert np.all(np.diff(g) > 0)
    rises = np.diff(w) > np.hypot(sem[:-1], sem[1:])
    assert not rises.any(), (
        f"non-monotone at Gamma {[float(x) for x in g[1:][rises]]}")


def test_calibration_decays_to_zero(cal):
    """The point of the control subtraction is a PHYSICAL zero: at the top of
    the Gamma grid the net width must be consistent with zero, so the
    calibration has an endpoint rather than an empirical floor."""
    last = cal["per_gamma"][-1]
    assert last["soft_net_mean"] < 3.0 * last["soft_net_sem"] + 0.02 * \
        cal["deterministic_reference"]["w_soft_net"]


def test_calibration_sems_are_small_enough_to_invert(cal):
    """A calibration whose own scatter rivals its dynamic range cannot be
    inverted usefully. Require every SEM below 5 % of the deterministic net."""
    det = cal["deterministic_reference"]["w_soft_net"]
    for e in cal["per_gamma"]:
        assert e["soft_net_sem"] < 0.05 * det, f"Gamma={e['Gamma']}"


def test_thermometer_scope_limit_is_declared(thermo):
    """A filling may only claim calibrated=true if the calibration it actually
    used was computed at its own filling. Silently presenting an Omega = 5 curve
    as a calibration for Omega = 13.9 is the failure mode this test prevents."""
    assert "Omega" in thermo["scope_limit"]
    by_nu = {f["nu"]: f for f in thermo["fillings"]}
    assert by_nu[-2.4]["calibrated"] is True
    for nu, f in by_nu.items():
        used_own = abs(f["calibration_nu"] - nu) < 1e-9
        assert f["calibrated"] is used_own, (
            f"nu={nu} claims calibrated={f['calibrated']} but used a curve "
            f"computed at nu={f['calibration_nu']}")


def test_thermometer_free_bound_is_ordered(thermo):
    """Requiring a larger surviving fraction must bound Gamma more tightly, at
    every filling -- each of which is bounded with its own calibration."""
    b = [x for x in thermo["free_upper_bounds"] if x["Gamma_max"] is not None]
    assert [x["R_level"] for x in b] == sorted([x["R_level"] for x in b],
                                               reverse=True)
    assert [x["Gamma_max"] for x in b] == sorted([x["Gamma_max"] for x in b])
    for nu in b[0]["per_filling"]:
        gs = [x["per_filling"][nu]["Gamma_max"] for x in b
              if x["per_filling"].get(nu, {}).get("Gamma_max") is not None]
        assert gs == sorted(gs), f"nu={nu} bounds not ordered: {gs}"


def test_thermometer_temperature_windows_scale_with_Ic(thermo):
    """The gate-tunable claim in one line: lower Ic -> lower E_J -> the same
    reduced Gamma window maps to a lower absolute temperature window."""
    rows = sorted(thermo["fillings"], key=lambda f: f["Ic_nA"])
    tops = [f["T_range_mK"][1] for f in rows]
    assert tops == sorted(tops)
    assert rows[0]["T_range_mK"][1] < rows[-1]["T_range_mK"][1] / 2


# ---------------------------------------------------------------- slope estimator
def test_local_slope_recovers_an_analytic_derivative(t2mod):
    """On a curve whose derivative is known, the local linear regression must
    return it -- smoothing is only acceptable if it is not also biasing."""
    lg = np.log(np.geomspace(0.005, 0.10, 14))
    y = 0.9 / (1.0 + np.exp(1.8 * (lg - np.log(0.02))))
    got = t2mod.local_slope(lg, lg, y)
    want = -0.9 * 1.8 * np.exp(1.8 * (lg - np.log(0.02))) / \
        (1.0 + np.exp(1.8 * (lg - np.log(0.02)))) ** 2
    # Interior nodes only: the regression is one-sided at the ends. The bound is
    # the measured smoothing bias at the production bandwidth (0.026 against a
    # peak slope of 0.405, i.e. 6.4 %), documented in local_slope() -- asserted
    # at its true size so a regression that made it worse would be caught.
    assert np.max(np.abs(got[2:-2] - want[2:-2])) < 0.03


def test_local_slope_is_smoother_than_the_spline_derivative(t2mod):
    """The reason it exists: sigma_T ~ 1/|slope|, and the PCHIP derivative is
    only C0 across knots, so differentiating the spline prints the knot spacing
    onto the reported resolution."""
    cal_path = OUT / "thermometry_calibration.json"
    if not cal_path.exists():
        pytest.skip("calibration not generated")
    cal = json.loads(cal_path.read_text())
    inv = t2mod.build_inverse(cal)
    lgd = np.linspace(inv["lg"][0], inv["lg"][-1], 601)
    spline_d = inv["f_R"](lgd, deriv=1)
    local_d = t2mod.local_slope(lgd, inv["lg"], inv["Rk"])
    rough = lambda a: np.abs(np.diff(a, 2)).mean()
    assert rough(local_d) < rough(spline_d) / 10
    # and it must not have flattened the curve away: same overall drop
    assert 0.6 < np.trapezoid(local_d, lgd) / np.trapezoid(spline_d, lgd) < 1.4


def test_slope_never_vanishes_inside_the_usable_range(t2mod, thermo):
    """A zero slope means an infinite sigma_T -- the inverse would be undefined
    there. The trimmed calibration must keep the map strictly decreasing."""
    slope = np.array(thermo["reference_inverse_table"]["dR_dGamma"])
    assert np.all(slope < 0)


def test_output_tag_is_stable_and_unique(t1mod):
    """Regression: the artifact suffix was once a local named `tag` that a later
    loop variable of the same name overwrote, so runs wrote files called
    `thermometry_calibration[   6.7s].json`. The reference filling must keep the
    unsuffixed production names and every other filling must get its own."""
    seen = {}
    for nu in (-2.4, -2.5, -2.6, -2.7, -2.8):
        cfg = {"nu": nu, "is_reference": abs(nu - t1mod.NU_REF) < 1e-9}
        tag = t1mod.output_tag(cfg)
        assert not any(ch in tag for ch in "[]. s"), f"unsafe suffix {tag!r}"
        assert tag not in seen, f"nu={nu} collides with nu={seen.get(tag)}"
        seen[tag] = nu
    assert seen[""] == t1mod.NU_REF
