"""M4 integration & thermal package — TBG-JJ / photonic-cavity paper.

Locked constants inherited from the M2/M3 briefs (see M4_BRIEF_DRAFT.md
Sec. 1 for sources). Everything else in this package is an explicit free
parameter until M2 delivers photonics.json.
"""

E_CHARGE = 1.602176634e-19   # C
H_PLANCK = 6.62607015e-34    # J s
HBAR = H_PLANCK / (2.0 * 3.141592653589793)

F0_HZ = 145.0e9                                # drive = 2*Delta/h (M3 contract)
V_PHOTON = H_PLANCK * F0_HZ / E_CHARGE         # hbar*omega/e = 0.59967 mV

# --- Contract v0.3 (wide-junction rescale; DECISION_contract_v0p3.md,
# paper/decisions.md 2026-07-25). The junction is widened x10: Rn 2 kOhm ->
# 200 Ohm and Ic 30 -> 300 nA. IcRn = 60 uV is INVARIANT, so V_PHOTON, F0,
# A_TARGET, QW_OVER_QI and the whole dimensionless problem are unchanged. The
# power budget scales x10 (Rn/10 but P_J = V_ac^2/2Rn and V_ac is held): with
# P_IN_REF_W = 90 nW, A = 1%, P_J = 900 pW and V_ac = sqrt(2*900pW*200 Ohm) =
# 0.6 mV is PRESERVED, so alpha = e*V_ac/(hbar*omega) = 1.0005 is unchanged.
RN_OHM = 200.0                                 # v0.3 (was 2000.0)
IC_CONTRACT_A = 300.0e-9                        # v0.3 (was 30.0e-9; dome center)

A_TARGET = 0.01                                # absorbed fraction (M2 brief)
QW_OVER_QI = 200.0                             # Q_w = 200*Q_i for A = 1% (M2 brief)
P_IN_REF_W = 90.0e-9                            # v0.3 power budget (was 9.0e-9)
P_J_REF_W = A_TARGET * P_IN_REF_W              # 900 pW (v0.3; was 90 pW)

PENDING_M2 = ("Q_i", "V_mode_um3", "E_per_sqrtW", "Q_w_per_N_rows")
