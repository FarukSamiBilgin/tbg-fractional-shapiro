"""M3 junction-dynamics package — magic-angle TBG Josephson junction
embedded in a sub-THz photonic-crystal cavity.

Physics contract v0.1 (M3_BRIEF.md Sec. 1, immutable):
  Ic = 30 nA, Rn = 2 kOhm, C ~ 0.1 fF -> beta_c ~ 0.04 (overdamped, RSJ valid)
  f_c = 2e*Ic*Rn/h = 29.0 GHz ; drive f = 145 GHz -> Omega = 5.0
  T_op = 100 mK -> Gamma = k_B T / E_J = 0.140
"""
from .cpr import cpr, normalize_cpr, C_SIN, C_SKEW, SKEW_RENORM
from .rsj import OMEGA, run_rsj, grid_run
from .noise import run_rsj_langevin, ensemble_staircase
from .analysis import step_width, width_curve, dvdi

CONTRACT = {
    "Omega": 5.0, "Ic_nA": 30.0, "Rn_kOhm": 2.0, "f_c_GHz": 29.0,
    "f_drive_GHz": 145.0, "V_step_uV": 300.0, "IcRn_uV": 60.0,
    "beta_c": 0.04, "Gamma_noise_100mK": 0.140, "EJ_over_kB_K": 0.715,
    "cpr_sin": list(C_SIN), "cpr_skew": list(C_SKEW),
    "skew_renorm_factor": SKEW_RENORM,
}
