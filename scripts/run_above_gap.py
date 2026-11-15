"""Deliverable 4 driver: Tien-Gordon above-gap I-V vs absorbed power.

Produces the photon-assisted quasiparticle I-V family of the driven TBG
junction for a log-spaced grid of absorbed cavity power P_abs (including 0),
with the gap Delta(P_abs) taken from the Owen-Scalapino steady state. Writes
out/above_gap.json and figs/fig_m3_above_gap.png.

Deterministic: no RNG is used anywhere (metadata records 'deterministic').

Gap note: Delta0 is the contract value h*72.5 GHz (= 2*Delta0/h at 145 GHz),
to be superseded by M1 material.json. The 145 GHz drive photon energy equals
2*Delta0 exactly (contract), so the sideband spacing hbar*omega = 2*Delta0.

Fixed physics choices (documented):
  alpha    = 1.0    representative ac-drive strength eV_ac/hbar*omega
  T        = 0.1 K  operating temperature (100 mK, contract)
  Rn       = 2 kOhm normal-state resistance (contract)
  eta_abs=1, n0=1, tau_rec=1  reference pair-breaking units (P_abs in eV)
"""
import json
import pathlib
import sys
import time

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from junction.tien_gordon import (
    DELTA0, HW_DRIVE, default_nmax, iqp0_bcs, steady_state_delta, tien_gordon_iv,
)

OUT = ROOT / "out"
FIGS = ROOT / "figs"
OUT.mkdir(exist_ok=True)
FIGS.mkdir(exist_ok=True)

# ------------------------------------------------------------------ params
ALPHA = 1.0
T_OP = 0.1            # K
RN = 2000.0          # ohm
ETA_ABS = 1.0
N0 = 1.0
TAU_REC = 1.0
DYNES = 3.0e-3
NV = 240

# absorbed-power grid: 0 plus five log-spaced points up to 1*Delta0 (eV).
P_ABS = np.concatenate([[0.0], np.logspace(-2.0, 0.0, 5) * DELTA0])
# bias grid 0 .. 6*Delta0 (bias energy in eV == voltage in V numerically)
V = np.linspace(0.0, 6.0 * DELTA0, NV)


def main():
    t0 = time.time()
    Delta_of_P = steady_state_delta(ETA_ABS, P_ABS, TAU_REC, N0, DELTA0)
    Delta_of_P = np.atleast_1d(Delta_of_P)

    I = np.empty((len(P_ABS), NV))
    for k, (P, D) in enumerate(zip(P_ABS, Delta_of_P)):
        I[k] = tien_gordon_iv(V, ALPHA, D, T_OP, RN, hw=HW_DRIVE, dynes=DYNES)
        print(f"  P_abs/Delta0={P/DELTA0:7.4f}  Delta/Delta0={D/DELTA0:.4f}  "
              f"I(6D)={I[k, -1]*1e9:8.2f} nA")

    # ------------------------------------------------ diagnostics (measured)
    # onset (P=0): voltage of maximum dI/dV on a fine grid
    Vf = np.linspace(0.01 * DELTA0, 4.0 * DELTA0, 4000)
    If = iqp0_bcs(Vf, DELTA0, T_OP, RN, dynes=DYNES)
    onset_ratio = float(Vf[np.argmax(np.gradient(If, Vf))] / DELTA0)  # ~2
    # ohmic deviation at eV = 10*Delta0 (P=0)
    I10 = float(iqp0_bcs(10.0 * DELTA0, DELTA0, T_OP, RN, dynes=DYNES))
    ohmic_dev = abs(I10 * RN / (10.0 * DELTA0) - 1.0)
    supp_max = float(1.0 - Delta_of_P[-1] / DELTA0)
    runtime = time.time() - t0

    params = {
        "alpha": ALPHA, "T_K": T_OP, "Rn_ohm": RN,
        "eta_abs": ETA_ABS, "n0": N0, "tau_rec": TAU_REC, "dynes": DYNES,
        "Delta0_eV": DELTA0, "hw_drive_eV": HW_DRIVE, "nmax": default_nmax(ALPHA),
        "P_abs_units": "eV (reference units eta_abs=n0=tau_rec=1)",
        "V_units": "V (bias energy in eV == V numerically)",
        "I_units": "A", "rng": "deterministic",
        "Delta0_note": "contract value h*72.5GHz, to be superseded by M1 material.json",
        "diagnostics": {
            "onset_eV_over_Delta": onset_ratio,
            "ohmic_dev_at_10Delta": ohmic_dev,
            "Delta_suppression_at_maxP": supp_max,
        },
    }
    payload = {
        "P_abs": P_ABS.tolist(),
        "V": V.tolist(),
        "I": I.tolist(),
        "Delta_of_P": Delta_of_P.tolist(),
        "params": params,
    }
    (OUT / "above_gap.json").write_text(json.dumps(payload))

    # ------------------------------------------------------------- figure
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.4))
    cmap = plt.get_cmap("viridis")
    for k, (P, D) in enumerate(zip(P_ABS, Delta_of_P)):
        col = cmap(k / max(1, len(P_ABS) - 1))
        lbl = "P=0" if P == 0.0 else rf"$P/\Delta_0$={P/DELTA0:.2g}"
        ax0.plot(V * 1e3, I[k] * 1e9, color=col, lw=1.7, label=lbl)
    ax0.axvline(2.0 * DELTA0 * 1e3, color="k", ls="--", lw=0.8,
                label=r"$eV=2\Delta_0$")
    ax0.set_xlabel("bias voltage V (mV)")
    ax0.set_ylabel("I (nA)")
    ax0.set_title(rf"Tien-Gordon above-gap I-V ($\alpha$={ALPHA}, T={T_OP} K)")
    ax0.legend(fontsize=7, loc="upper left")
    ax0.grid(alpha=0.3)

    ax1.plot(P_ABS / DELTA0, Delta_of_P / DELTA0, "o-", color="C3")
    ax1.set_xlabel(r"$P_{\rm abs}/\Delta_0$  (reference units)")
    ax1.set_ylabel(r"$\Delta/\Delta_0$")
    ax1.set_title("Owen-Scalapino gap suppression")
    ax1.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGS / "fig_m3_above_gap.png", dpi=200)
    plt.close(fig)

    print(f"\nonset eV/Delta         = {onset_ratio:.4f}")
    print(f"ohmic dev @10Delta     = {ohmic_dev*100:.3f}%")
    print(f"Delta suppression @maxP= {supp_max*100:.3f}%")
    print(f"runtime                = {runtime:.2f} s")
    print(f"wrote {OUT / 'above_gap.json'}")
    print(f"wrote {FIGS / 'fig_m3_above_gap.png'}")


if __name__ == "__main__":
    main()
