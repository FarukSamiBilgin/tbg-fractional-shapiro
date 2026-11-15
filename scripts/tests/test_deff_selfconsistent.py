"""C1 acceptance tests — self-consistent collection-structure requirement.

The result under test: once the junction's loading back-action on the cavity is
included, V_ac(d_eff) has a maximum at Q_J = Q_EM, and that maximum is a factor
of two below the linear (fixed-field) estimate. Both facts are provable in the
weak-coupling limit, so they are asserted against the closed forms and not only
against the stored numbers.
"""
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


@pytest.fixture(scope="session")
def sc():
    p = OUT / "deff_selfconsistent.json"
    if not p.exists():
        pytest.skip("out/deff_selfconsistent.json not generated")
    return json.loads(p.read_text())


@pytest.fixture(scope="session")
def qj():
    return json.loads((OUT / "qj_closure.json").read_text())


@pytest.fixture(scope="session")
def integ():
    return json.loads((ROOT / "integration.json").read_text())


def test_absorption_formula_matches_the_locked_contract(integ):
    """A must be the repo's own on-resonance TCMT expression, and reproduce the
    contract's exact design value 400/201^2 at Q_w = 200 Q_i."""
    f = integ["power_chain"]["formulas"]["tcmt_on_resonance"]["A"]
    assert "2*Q_i*Q_w/(Q_i+Q_w)**2" in f.replace(" ", "")
    Qi = 14611.0
    Qw = 200.0 * Qi
    assert 2 * Qi * Qw / (Qi + Qw) ** 2 == pytest.approx(400 / 201**2, rel=1e-12)


def test_optimum_sits_at_impedance_match(sc, qj):
    """V ~ d*Q_i(d) in the weak-coupling limit, so dV/dd = 0 exactly where the
    junction load equals the intrinsic load: Q_J = Q_EM, i.e. d_opt = sqrt(C/Q_EM).
    """
    C = qj["q_j"]["coefficient_Rn_eps0_eps_c_omega0_Vmode"]
    Q_EM = qj["inputs"]["Q_i_insitu"]
    d_analytic = np.sqrt(C / Q_EM)
    for branch in ("clean", "measured"):
        d_num = sc[branch]["d_eff_opt_um"] * 1e-6
        assert d_num == pytest.approx(d_analytic, rel=5e-3), branch
        # and Q_J there equals Q_EM
        assert sc[branch]["Q_J_at_opt"] == pytest.approx(Q_EM, rel=5e-3)


def test_half_the_power_lands_on_the_junction_at_the_optimum(sc):
    """Q_J = Q_EM means the junction and the rest of the cavity dissipate
    equally — the physical content of the impedance match."""
    for branch in ("clean", "measured"):
        assert sc[branch]["frac_power_to_junction_at_opt"] == pytest.approx(0.5, abs=0.01)


def test_the_shortfall_is_exactly_a_factor_two(sc):
    """At the optimum Q_i = Q_EM/2, and V ~ d*Q_i, so the self-consistent maximum
    is half the fixed-field estimate evaluated at the same d_eff. The clean
    branch's shortfall against the 0.6 mV requirement is therefore 2.00, not an
    empirical near-miss."""
    assert sc["clean"]["shortfall_factor"] == pytest.approx(2.0, rel=0.02)
    assert not sc["clean"]["meets_requirement"]
    assert not sc["measured"]["meets_requirement"]


def test_measured_branch_shortfall_is_the_dilution(sc, qj):
    """The two branches differ only by the measured Fano energy dilution, so
    their shortfalls differ by sqrt of it."""
    dil = qj["stored_energy_per_ref"]["dilution_cmt_over_field_center"]
    ratio = sc["measured"]["shortfall_factor"] / sc["clean"]["shortfall_factor"]
    assert ratio == pytest.approx(np.sqrt(dil), rel=0.02)


def test_required_power_reproduces_the_contract_thermal_load(sc):
    """The consistency that makes the correction believable: raising P_in until
    V_ac reaches 0.6 mV at the optimum puts exactly the contract's assumed
    900 pW on the junction. The contract's thermal number was right; only P_in
    and d_eff were wrong."""
    row = sc["power_to_close"]["clean"]
    assert row["P_J_W"] * 1e12 == pytest.approx(sc["contract_P_J_pW"], rel=0.01)
    assert row["factor_over_contract"] == pytest.approx(4.0, rel=0.05)


def test_loading_criterion_is_past_the_optimum(sc, qj):
    """The Q_rad,Q_mat >= 10 Q_J criterion of qj_closure.json asks for
    d_eff >= 21.6 um, which is beyond the voltage optimum and therefore delivers
    LESS drive, not more. The two criteria conflict and this test pins that."""
    d_load = sc["loading_requirement"]["d_eff_min_um"]
    assert d_load > sc["clean"]["d_eff_opt_um"]
    assert sc["loading_requirement"]["V_ac_clean_V"] < sc["clean"]["V_ac_max_V"]


def test_linear_estimate_is_optimistic_everywhere(sc):
    """Neither linear d_eff target from qj_closure.json delivers what it claims
    once back-action is included."""
    chk = sc["linear_estimate_check"]
    assert chk["clean_selfconsistent_V_at_that_d_eff"] < 0.6e-3
    assert chk["measured_selfconsistent_V_at_that_d_eff"] < 0.6e-3


# ------------------------------------------------------- the closing lever
def test_coupling_lever_scaling_is_inverse_sqrt_Qw(sc):
    """V_max ~ 1/sqrt(Q_w): halving the required voltage target should quadruple
    the admissible Q_w. Checked between the alpha = 1 and alpha = 0.70 rows at
    the same dilution, whose targets differ by 0.70."""
    rows = {(r["D"], round(r["alpha"], 2)): r for r in sc["coupling_ladder_lever"]["rows"]}
    a1 = rows[(1.0, 1.0)]
    a07 = rows[(1.0, 0.7)]
    assert a07["Qw_over_QEM"] / a1["Qw_over_QEM"] == pytest.approx(1 / 0.7**2, rel=0.05)


def test_design_closes_at_the_fingerprint_optimum(sc):
    """The load-bearing result: with the band tail removed (D = 1) and the drive
    at the half-step's own optimum alpha = 0.70 rather than the contract's
    alpha = 1, the requirement is met at the contract's P_in with the absorption
    still at its designed ~1 % and the junction load HALF the contract's 900 pW."""
    rows = {(r["D"], round(r["alpha"], 2)): r for r in sc["coupling_ladder_lever"]["rows"]}
    r = rows[(1.0, 0.7)]
    assert r["verdict"] == "OK"
    assert r["A"] == pytest.approx(0.01, abs=0.003)        # design premise intact
    assert r["P_J_W"] * 1e12 < 0.6 * sc["contract_P_J_pW"]  # thermal headroom
    assert abs(r["delta_rows"]) < 1.0                       # sub-one-row change


def test_alpha_opt_comes_from_the_fingerprint_not_a_guess(sc):
    """alpha = 0.70 is not a relaxation invented to make the budget work; it is
    junction.json's own measured half-step optimum at the dome centre."""
    jj = json.loads((ROOT / "junction.json").read_text())["fingerprint"]
    assert sc["coupling_ladder_lever"]["alpha_opt_used"] == pytest.approx(
        jj["alpha_opt"][jj["nu"].index(-2.4)])


def test_band_tail_must_be_essentially_removed(sc):
    """No ladder setting rescues a diluted cavity: at D >= 4 every row either
    pushes absorption past 5 % (destroying the '99 % passes by' premise) or the
    junction past its thermal budget."""
    for r in sc["coupling_ladder_lever"]["rows"]:
        if r["D"] >= 4.0:
            assert r["verdict"] != "OK", r
