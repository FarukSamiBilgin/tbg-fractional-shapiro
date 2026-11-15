"""C1 - self-consistent V_ac(d_eff): the collection-structure requirement, with
cavity back-action included.

Why this exists
---------------
`out/qj_closure.json` reports two separate d_eff requirements:

  * loading   : Q_J <= 2700  ->  d_eff >= 21.6 um   (junction must actually be
                the dominant loss channel)
  * voltage   : V_ac = 0.6 mV -> d_eff = 57.8 um    (measured/diluted field)
                                  d_eff =  9.3 um    (clean CMT field)

The voltage numbers there are a LINEAR extrapolation: V_ac = E_center * d_eff at
the field the cavity has when the junction is a negligible load. That cannot be
the whole story, because the two requirements are the same physics seen twice.
Coupling the junction in harder (larger d_eff) loads the cavity:

    Q_J(d_eff) = C / d_eff^2,      C = Rn eps0 eps_c omega0 V_mode
    1/Q_i      = 1/Q_EM + 1/Q_J          (junction adds a loss channel)
    A          = 2 Q_L^2 / (Q_w Q_i),    1/Q_L = 1/Q_i + 2/Q_w
    U          = A P_in Q_i / omega0     (steady state: absorbed = dissipated)
    E          = sqrt(2U / (eps0 eps_c V_mode))
    V_ac       = E d_eff

so E itself falls as d_eff grows. V_ac ~ d_eff sqrt(Q_i(d_eff) A(d_eff)) has a
maximum: past some d_eff the junction over-damps the cavity and the voltage it
sees goes DOWN despite the stronger coupling. This script locates that maximum
and asks whether 0.6 mV is reachable at all.

Everything is read from the stored artifacts; no new electromagnetics is done.
The mode volume, Q's, the CMT/field dilution factor and the unit conversion all
come from out/qj_closure.json and photonics.json as measured.

Two field branches are carried throughout, exactly as qj_closure.json does:

  clean    - the CMT design point, U = A P_in Q_i / omega0
  measured - the same, divided by the measured Fano dilution (the N = 3 build-up
             is band-tail diluted by 38.4x in energy relative to CMT). This is a
             property of the present W1 alignment, not a law of nature; the
             band-tail fix is a separate open item. Both branches are reported
             because the answer differs qualitatively between them.
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "out"

EPS0 = 8.8541878128e-12


def load():
    qj = json.loads((OUT / "qj_closure.json").read_text())
    integ = json.loads((ROOT / "integration.json").read_text())
    inp = qj["inputs"]
    return qj, integ, inp


def main():
    qj, integ, inp = load()

    # ---- measured inputs -------------------------------------------------
    C = qj["q_j"]["coefficient_Rn_eps0_eps_c_omega0_Vmode"]   # Q_J = C/d_eff^2
    omega0 = inp["omega0_rad_s"]
    eps_c = inp["eps_core"]
    V_mode = inp["V_mode_m3"]
    Q_EM = inp["Q_i_insitu"]              # rad+mat+in-situ, junction negligible
    A_design = inp["A_design"]
    P_in = qj["absolute_voltage_thread"]["P_in_v03_W"]
    V_req = qj["absolute_voltage_thread"]["V_ac_required_mV"] * 1e-3
    dilution = qj["stored_energy_per_ref"]["dilution_cmt_over_field_center"]

    # On-resonance TCMT absorption, taken verbatim from the repo's own locked
    # formula (integration.json:power_chain.formulas.tcmt_on_resonance):
    #     A = 2 gamma_i gamma_w / (gamma_i + gamma_w)^2 = 2 Q_i Q_w / (Q_i + Q_w)^2
    # Q_w is a LOCKED design constant (QW_OVER_QI = 200), not something to infer:
    # the coupling ladder is fixed hardware, so loading the cavity with the
    # junction changes Q_i and therefore A, at fixed Q_w.
    Q_w = integ["power_chain"]["locked_constants"]["QW_OVER_QI"] * Q_EM

    def A_of_Qi(Qi):
        return 2.0 * Qi * Q_w / (Qi + Q_w) ** 2

    A_check = A_of_Qi(Q_EM)

    print("=" * 76)
    print("C1 - self-consistent V_ac(d_eff) with cavity back-action")
    print(f"  C = Rn eps0 eps_c omega0 V_mode = {C:.6e}   (Q_J = C/d_eff^2)")
    print(f"  Q_EM (junction negligible)      = {Q_EM:.1f}")
    print(f"  A_design                        = {A_design:.5f}")
    print(f"  Q_w = 200*Q_EM (locked)         = {Q_w:.4e}")
    print(f"  A(Q_EM) from the locked formula = {A_check:.6f}   "
          f"[contract A_TARGET = 0.01, exact 400/201^2 = 0.0099007]")
    print(f"  P_in = {P_in*1e9:.1f} nW,  V_ac required = {V_req*1e3:.3f} mV")
    print(f"  measured Fano dilution (energy) = {dilution:.2f}x")
    print("=" * 76)

    # ---- self-consistent sweep -------------------------------------------
    d = np.logspace(-7, -3, 40001)         # 0.1 um .. 1 mm, ~0.023 % spacing so
                                           # the argmax locates Q_J = Q_EM to
                                           # better than 0.1 %
    Q_J = C / d**2
    Q_i = 1.0 / (1.0 / Q_EM + 1.0 / Q_J)
    A = A_of_Qi(Q_i)
    U_clean = A * P_in * Q_i / omega0
    E_clean = np.sqrt(2.0 * U_clean / (EPS0 * eps_c * V_mode))
    V_clean = E_clean * d
    V_meas = V_clean / np.sqrt(dilution)   # field scales as sqrt(energy)

    # fraction of the intrinsic dissipation that lands on the junction
    frac_J = Q_i / Q_J

    def report(name, V):
        k = int(np.argmax(V))
        print(f"\n--- {name} branch ---")
        print(f"  V_ac peaks at d_eff = {d[k]*1e6:8.2f} um  ->  V_ac = "
              f"{V[k]*1e3:.4f} mV   (Q_J = {Q_J[k]:.3g}, Q_i = {Q_i[k]:.1f}, "
              f"P_J/P_diss = {frac_J[k]*100:.1f}%)")
        if V[k] >= V_req:
            # first crossing of the requirement
            lo = np.where(V >= V_req)[0]
            print(f"  REACHES the 0.600 mV requirement: d_eff in "
                  f"[{d[lo[0]]*1e6:.2f}, {d[lo[-1]]*1e6:.2f}] um")
        else:
            print(f"  DOES NOT reach 0.600 mV: short by "
                  f"{V_req/V[k]:.2f}x at the optimum")
        return k

    k_clean = report("clean (CMT)", V_clean)
    k_meas = report("measured (band-tail diluted)", V_meas)

    # ---- what the linear estimate said ------------------------------------
    lin_clean = qj["absolute_voltage_thread"]["d_eff_for_0p6mV_clean_um"]
    lin_meas = qj["absolute_voltage_thread"]["d_eff_for_0p6mV_measured_um"]
    print("\n--- versus the linear (no back-action) estimate in qj_closure.json ---")
    print(f"  clean   : linear said {lin_clean:.2f} um")
    print(f"  measured: linear said {lin_meas:.2f} um")

    # linear prediction evaluated self-consistently
    for nm, dl, V in (("clean", lin_clean * 1e-6, V_clean),
                      ("measured", lin_meas * 1e-6, V_meas)):
        Vl = np.interp(dl, d, V)
        print(f"  at d_eff = {dl*1e6:6.2f} um the self-consistent {nm} V_ac is "
              f"{Vl*1e3:.4f} mV ({Vl/V_req*100:.1f}% of requirement)")

    # ---- the loading requirement, for reference ---------------------------
    d_load = np.sqrt(C / qj["requirement"]["Q_J_max_allowed_10x"])
    print(f"\n--- loading requirement ---")
    print(f"  Q_J <= 2700 needs d_eff >= {d_load*1e6:.2f} um; at that d_eff the "
          f"self-consistent V_ac is {np.interp(d_load, d, V_clean)*1e3:.4f} mV "
          f"(clean) / {np.interp(d_load, d, V_meas)*1e3:.4f} mV (measured)")

    # ---- what P_in would be needed, and what it costs thermally -----------
    # V_ac ~ sqrt(P_in) at fixed d_eff, so the shortfall squares into power.
    print("\n--- closing the gap with input power ---")
    pj_rows = {}
    for nm, V, k in (("clean", V_clean, k_clean), ("measured", V_meas, k_meas)):
        need = (V_req / V[k]) ** 2
        P_need = P_in * need
        # at the optimum the cavity dissipates A*P_in and frac_J of it lands on
        # the junction; both are P_in-linear, so scale directly.
        P_diss = A[k] * P_need
        P_J = frac_J[k] * P_diss
        pj_rows[nm] = {"P_in_required_W": P_need, "factor_over_contract": need,
                       "P_diss_W": P_diss, "P_J_W": P_J,
                       "d_eff_um": float(d[k] * 1e6)}
        print(f"  {nm:9s}: P_in = {P_need*1e9:9.1f} nW ({need:6.1f}x contract) "
              f"-> P_diss = {P_diss*1e12:8.1f} pW, P_J = {P_J*1e12:8.1f} pW "
              f"at d_eff = {d[k]*1e6:.2f} um")
    P_J_contract = json.loads((ROOT / "integration.json").read_text())[
        "power_chain"]["locked_constants"]["P_J_REF_pW"]
    print(f"  contract P_J_REF = {P_J_contract:.1f} pW  (the thermal analysis' "
          f"assumed junction load)")

    # ---- how much band-tail recovery is required --------------------------
    # At fixed P_in, V_ac ~ 1/sqrt(D) with D the Fano ENERGY dilution, so
    # restoring V_ac costs P_in ~ D, and since P_J is P_in-linear at fixed
    # d_eff, P_J ~ D as well. The whole band-tail question therefore collapses
    # to one proportionality, with no electromagnetics needed:
    #     P_in(D) = P_in(1) * D ,  P_J(D) = P_J(1) * D
    # P_J(1) comes out at exactly the contract's P_J_REF, and that is forced,
    # not coincidental: the contract set P_J = A0*P_in0 assuming ALL absorbed
    # power reaches the junction; self-consistently only 50 % does, but A is
    # halved and P_in is 4x, and 4 * (A0/2) * 0.5 = A0.
    print("\n--- band-tail recovery requirement (D = Fano energy dilution) ---")
    base = pj_rows["clean"]
    dil_rows = []
    for D in (1.0, 2.0, 4.0, 10.0, 23.1, dilution):
        row = {"D": D, "P_in_nW": base["P_in_required_W"] * D * 1e9,
               "P_J_pW": base["P_J_W"] * D * 1e12}
        dil_rows.append(row)
        tag = ""
        if abs(D - dilution) < 1e-9:
            tag = "  <- measured today (centre-field route)"
        elif abs(D - 23.1) < 0.05:
            tag = "  <- measured, box-integral route"
        elif D == 1.0:
            tag = "  <- complete fix; saturates the contract's 900 pW exactly"
        print(f"  D = {D:6.2f}: P_in = {row['P_in_nW']:9.1f} nW, "
              f"P_J = {row['P_J_pW']:9.1f} pW{tag}")
    D_max_thermal = P_J_contract / (base["P_J_W"] * 1e12)
    print(f"  => tolerable D at the contract's 900 pW budget: D <= "
          f"{D_max_thermal:.2f}  (i.e. the band tail must be removed ENTIRELY)")
    print(f"     current D = {dilution:.1f} exceeds it by {dilution/D_max_thermal:.1f}x")

    # ---- the lever that actually closes it: the coupling ladder -----------
    # V_max^2 ~ P_in * C * Q_EM / (2 Q_w) with Q_w free, so V_max ~ 1/sqrt(Q_w):
    # coupling the CAVITY harder to the bus is a far cheaper lever than raising
    # P_in. Q_w is set by the ladder row count through the measured law
    # ln Q_w = sigma*N + const (photonics.json:ladder.exponential_fit), so a
    # factor 4 in Q_w is ln(4)/sigma ~ 0.96 rows. Note Q_EM cancels if Q_w is
    # locked to a FIXED RATIO 200*Q_EM, which is why improving the cavity Q
    # alone buys nothing here -- the ratio, not the Q, is the design variable.
    #
    # The drive target is also over-specified: the contract asks for alpha = 1,
    # but junction.json's own fingerprint puts the half-step optimum at
    # alpha_opt = 0.70 at the dome centre. Driving at the optimum instead of at
    # alpha = 1 costs nothing in signal and saves (0.7)^2 = 51 % of the power.
    sigma_ladder = json.loads((ROOT / "photonics.json").read_text())[
        "ladder"]["exponential_fit"]["sigma"]
    fp = json.loads((ROOT / "junction.json").read_text())["fingerprint"]
    alpha_opt = fp["alpha_opt"][fp["nu"].index(-2.4)]
    V_photon = integ["power_chain"]["locked_constants"]["V_PHOTON_mV"] * 1e-3

    def solve_ratio(D, alpha):
        """Q_w/Q_EM (and the row change) that makes V_max/sqrt(D) = alpha*V_photon."""
        target = alpha * V_photon
        r = 200.0
        for _ in range(80):
            QJ_ = C / d**2
            Qi_ = 1.0 / (1.0 / Q_EM + 1.0 / QJ_)
            A_ = 2.0 * Qi_ * (r * Q_EM) / (Qi_ + r * Q_EM) ** 2
            U_ = A_ * P_in * Qi_ / omega0
            V_ = np.sqrt(2.0 * U_ / (EPS0 * eps_c * V_mode)) * d
            k_ = int(np.argmax(V_))
            r *= ((V_[k_] / np.sqrt(D)) / target) ** 2
        return (r, np.log(r / 200.0) / sigma_ladder, float(A_[k_]),
                float(Qi_[k_] / QJ_[k_] * A_[k_] * P_in), float(d[k_]))

    print("\n--- the coupling-ladder lever (V_max ~ 1/sqrt(Q_w)) ---")
    print(f"  alpha = 1 needs {V_photon*1e3:.4f} mV; the fingerprint's own "
          f"optimum alpha_opt(nu=-2.4) = {alpha_opt:.2f} needs "
          f"{alpha_opt*V_photon*1e3:.4f} mV")
    print(f"  {'D':>6} {'alpha':>6} {'Qw/Q_EM':>9} {'dN rows':>9} {'A(%)':>7} "
          f"{'P_J(pW)':>9} {'d_opt(um)':>10}  verdict")
    lever = []
    for D in (1.0, 4.0, 10.0, dilution):
        for al in (1.0, alpha_opt):
            r, dN, A_r, P_J_r, d_r = solve_ratio(D, al)
            ok = ("OK" if (A_r < 0.05 and P_J_r <= 1.1e-9)
                  else ("absorption > 5%" if A_r >= 0.05 else "P_J over budget"))
            lever.append({"D": D, "alpha": al, "Qw_over_QEM": r,
                          "delta_rows": dN, "A": A_r, "P_J_W": P_J_r,
                          "d_eff_um": d_r * 1e6, "verdict": ok})
            print(f"  {D:6.1f} {al:6.2f} {r:9.2f} {dN:9.2f} {A_r*100:7.2f} "
                  f"{P_J_r*1e12:9.1f} {d_r*1e6:10.2f}  {ok}")

    result = {
        "description": "C1 self-consistent V_ac(d_eff) including the junction's "
                       "loading back-action on the cavity. Supersedes the linear "
                       "d_eff_for_0p6mV entries of out/qj_closure.json, which "
                       "hold E fixed at its unloaded value.",
        "model": {
            "Q_J": "C/d_eff^2, C from qj_closure.json:q_j",
            "Q_i": "1/(1/Q_EM + 1/Q_J)",
            "A": "2 Q_L^2/(Q_w Q_i), 1/Q_L = 1/Q_i + 2/Q_w",
            "U": "A P_in Q_i / omega0",
            "E": "sqrt(2U/(eps0 eps_c V_mode))",
            "V_ac": "E d_eff",
            "measured_branch": "clean field divided by sqrt(Fano energy dilution)",
        },
        "inputs": {"C": C, "Q_EM": Q_EM, "A_design": A_design, "Q_w_implied": Q_w,
                   "P_in_W": P_in, "V_ac_required_V": V_req,
                   "energy_dilution": dilution},
        "clean": {
            "d_eff_opt_um": float(d[k_clean] * 1e6),
            "V_ac_max_V": float(V_clean[k_clean]),
            "meets_requirement": bool(V_clean[k_clean] >= V_req),
            "shortfall_factor": float(V_req / V_clean[k_clean]),
            "Q_J_at_opt": float(Q_J[k_clean]),
            "frac_power_to_junction_at_opt": float(frac_J[k_clean]),
        },
        "measured": {
            "d_eff_opt_um": float(d[k_meas] * 1e6),
            "V_ac_max_V": float(V_meas[k_meas]),
            "meets_requirement": bool(V_meas[k_meas] >= V_req),
            "shortfall_factor": float(V_req / V_meas[k_meas]),
            "Q_J_at_opt": float(Q_J[k_meas]),
            "frac_power_to_junction_at_opt": float(frac_J[k_meas]),
        },
        "linear_estimate_check": {
            "clean_linear_um": lin_clean,
            "clean_selfconsistent_V_at_that_d_eff":
                float(np.interp(lin_clean * 1e-6, d, V_clean)),
            "measured_linear_um": lin_meas,
            "measured_selfconsistent_V_at_that_d_eff":
                float(np.interp(lin_meas * 1e-6, d, V_meas)),
        },
        "power_to_close": pj_rows,
        "contract_P_J_pW": P_J_contract,
        "dilution_requirement": {
            "law": "P_in ~ D and P_J ~ D at fixed d_eff and fixed V_ac target",
            "rows": dil_rows,
            "D_max_at_contract_thermal_budget": D_max_thermal,
            "current_D_centre_field": dilution,
            "exceeds_by": dilution / D_max_thermal,
            "note": "P_J(D=1) equals the contract P_J_REF by construction, not "
                    "coincidence: the contract assumed all absorbed power reaches "
                    "the junction; self-consistently 50 % does, with A halved and "
                    "P_in 4x, and 4*(A0/2)*0.5 = A0.",
        },
        "coupling_ladder_lever": {
            "scaling": "V_max^2 ~ P_in C Q_EM/(2 Q_w); V_max ~ 1/sqrt(Q_w). Q_EM "
                       "cancels when Q_w is locked to a fixed RATIO of it, so the "
                       "ratio is the design variable, not the cavity Q.",
            "row_law": "ln Q_w = sigma*N + const, sigma from "
                       "photonics.json:ladder.exponential_fit",
            "alpha_opt_used": alpha_opt,
            "alpha_opt_source": "junction.json:fingerprint.alpha_opt at nu=-2.4",
            "V_photon_V": V_photon,
            "rows": lever,
        },
        "loading_requirement": {
            "d_eff_min_um": float(d_load * 1e6),
            "V_ac_clean_V": float(np.interp(d_load, d, V_clean)),
            "V_ac_measured_V": float(np.interp(d_load, d, V_meas)),
        },
    }
    (OUT / "deff_selfconsistent.json").write_text(json.dumps(result, indent=2))
    np.savez(OUT / "deff_selfconsistent.npz", d_eff=d, Q_J=Q_J, Q_i=Q_i, A=A,
             V_ac_clean=V_clean, V_ac_measured=V_meas, frac_J=frac_J)
    print(f"\n  json: {OUT/'deff_selfconsistent.json'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
