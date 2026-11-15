"""Bistritzer-MacDonald continuum model of magic-angle TBG (M1 band module).

Faithful port of m1_reference.py: same grids, same conventions, same
assembly (build kinetic blocks once, add ONLY the interlayer h.c. explicitly
-- a blanket H+H^dagger double-counts the kinetic term and destroys the
flat band).

Single valley, small-angle BM Hamiltonian, basis |layer, g>, g = m*b1 + n*b2:
  H_11 = hbar vF sigma.(k+g),         H_22 = hbar vF sigma.(k+g+q0),
  <2, g+delta|H|1, g> = T_j for delta in {0, b1, b2}, j = 0,1,2,
  T_j = wAA*s0 + wAB[cos(2 pi j/3) sx + sin(2 pi j/3) sy].
Flat bands = eigenvalue indices 2*Ng-1, 2*Ng. Dirac cones at K_m(0,0),
K'_m(0, k_theta).
"""

import numpy as np

KB_MEV_K = 0.08617          # meV / K
A_G = 0.246                 # graphene lattice constant, nm
KD = 4.0 * np.pi / (3.0 * A_G)   # Dirac momentum, 1/nm

S0 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], dtype=complex)
SY = np.array([[0, -1j], [1j, 0]], dtype=complex)

WAB = 110.0                 # meV (contract)
RATIO = 0.8                 # wAA/wAB corrugation (contract)
WAA = RATIO * WAB


def geometry(theta_deg):
    """Twist geometry: k_theta, moire q-vectors, reciprocal basis b1, b2."""
    th = np.deg2rad(theta_deg)
    ktheta = 2.0 * KD * np.sin(th / 2.0)
    q0 = ktheta * np.array([0.0, -1.0])
    q1 = ktheta * np.array([np.sqrt(3) / 2.0, 0.5])
    q2 = ktheta * np.array([-np.sqrt(3) / 2.0, 0.5])
    b1, b2 = q1 - q0, q2 - q0
    return ktheta, (q0, q1, q2), b1, b2


def basis(N):
    """Momentum-shell basis: integer (m, n) with m, n in [-N, N]. Shape (Ng, 2)."""
    idx = np.arange(-N, N + 1)
    mm, nn = np.meshgrid(idx, idx, indexing="ij")
    return np.stack([mm.ravel(), nn.ravel()], axis=1)


# interlayer tunneling matrices T0, T1, T2
T_MATS = [WAA * S0 + WAB * (np.cos(2 * np.pi * j / 3) * SX
                            + np.sin(2 * np.pi * j / 3) * SY)
          for j in range(3)]
# layer-1 site g couples to layer-2 site g' with g'-g = {0, b1, b2} -> T0,T1,T2
HOPS = [(0, 0), (1, 0), (0, 1)]


def hamiltonian(k, mn, hbv, ktheta, qs, b1, b2):
    """Assemble BM Hamiltonian at moire momentum k.

    Kinetic blocks are written once; only the interlayer coupling adds its
    explicit h.c. partner (never a blanket H + H^dagger).
    """
    Ng = len(mn)
    gs = mn[:, 0][:, None] * b1[None, :] + mn[:, 1][:, None] * b2[None, :]
    dim = 4 * Ng                    # 2 layers x 2 sublattices x Ng
    H = np.zeros((dim, dim), dtype=complex)
    key = {(int(m), int(n)): i for i, (m, n) in enumerate(mn)}
    o = 2 * Ng
    for i in range(Ng):
        p1 = k + gs[i]                       # layer 1 momentum
        p2 = k + gs[i] + qs[0]               # layer 2 momentum
        H[2*i:2*i+2, 2*i:2*i+2] = hbv * (p1[0] * SX + p1[1] * SY)
        H[o+2*i:o+2*i+2, o+2*i:o+2*i+2] = hbv * (p2[0] * SX + p2[1] * SY)
        for j, (dm, dn) in enumerate(HOPS):
            tgt = key.get((int(mn[i, 0] + dm), int(mn[i, 1] + dn)))
            if tgt is not None:
                blk = T_MATS[j]
                H[o+2*tgt:o+2*tgt+2, 2*i:2*i+2] = blk           # layer2 <- layer1
                H[2*i:2*i+2, o+2*tgt:o+2*tgt+2] = blk.conj().T  # h.c.
    return H


def flat_band_width(theta_deg, hbv, N=3, grid=7):
    """Flat-band width W over a grid x grid moire BZ mesh.

    Returns (W, (lo_min, lo_max, hi_min, hi_max)) where the flat bands are
    eigenvalue indices 2*Ng-1 (lower) and 2*Ng (upper).
    """
    ktheta, qs, b1, b2 = geometry(theta_deg)
    mn = basis(N)
    Ng = len(mn)
    lo, hi = [], []
    for i in range(grid):
        for j in range(grid):
            k = (i / grid) * b1 + (j / grid) * b2
            ev = np.linalg.eigvalsh(hamiltonian(k, mn, hbv, ktheta, qs, b1, b2))
            lo.append(ev[2 * Ng - 1])        # lower flat band
            hi.append(ev[2 * Ng])            # upper flat band
    lo, hi = np.array(lo), np.array(hi)
    return hi.max() - lo.min(), (lo.min(), lo.max(), hi.min(), hi.max())


def band_path(theta_deg, hbv, N=4, npts=30, nbands=10):
    """Band structure along K_m -> Gamma_m -> M_m -> K'_m.

    Returns (x, bands, ticks, labels) with bands centered on the flat pair.
    """
    ktheta, qs, b1, b2 = geometry(theta_deg)
    mn = basis(N)
    Ng = len(mn)
    K = np.array([0.0, 0.0])
    Kp = -qs[0]                                    # (0, +ktheta)
    G = np.array([np.sqrt(3) * ktheta / 2.0, ktheta / 2.0])
    M = 0.5 * (K + Kp)
    segs = [(K, G), (G, M), (M, Kp)]
    ks = []
    for A, B in segs:
        for t in np.linspace(0, 1, npts, endpoint=False):
            ks.append(A + t * (B - A))
    ks.append(Kp)
    ks = np.array(ks)
    x = np.concatenate([[0.0],
                        np.cumsum(np.linalg.norm(np.diff(ks, axis=0), axis=1))])
    half = nbands // 2
    bands = np.empty((len(ks), 2 * half))
    for i, k in enumerate(ks):
        ev = np.linalg.eigvalsh(hamiltonian(k, mn, hbv, ktheta, qs, b1, b2))
        bands[i] = ev[2 * Ng - half: 2 * Ng + half]
    ticks = [0.0]
    for A, B in segs:
        ticks.append(ticks[-1] + np.linalg.norm(B - A))
    return x, bands, ticks, [r"$K_m$", r"$\Gamma_m$", r"$M_m$", r"$K'_m$"]


def dirac_flat_eigs(theta_deg, hbv, N=4, k=(0.0, 0.0)):
    """The two flat-band eigenvalues (indices 2*Ng-1, 2*Ng) at momentum k.

    At the moire Dirac point K_m = (0, 0) these straddle/touch E = 0.
    """
    ktheta, qs, b1, b2 = geometry(theta_deg)
    mn = basis(N)
    Ng = len(mn)
    ev = np.linalg.eigvalsh(
        hamiltonian(np.asarray(k, dtype=float), mn, hbv, ktheta, qs, b1, b2))
    return ev[2 * Ng - 1], ev[2 * Ng]
