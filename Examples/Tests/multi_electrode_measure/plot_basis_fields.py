#!/usr/bin/env python3
"""Plot raw vs resolved reconstructed potential from the basis-consistency dump.

Usage: python plot_basis_fields.py <basis_fields_dir> [out.png]
Expects <dir>/raw.h5 and <dir>/resolved.h5 (phi[z,r] + grid metadata) as written
by inputs_rz_basis_check.py. Shows phi_raw, phi_resolved, their difference, and
the z-averaged radial profiles (raw vs resolved) against the electrode radii.
"""
import sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "basis_fields"
out = sys.argv[2] if len(sys.argv) > 2 else f"{d}/basis_compare.png"

# electrode radii (concentric RZ test): inner rod, middle shell, grounded outer
R_INNER, R_SH1, R_SH2, R_OUTER = 0.005, 0.020, 0.025, 0.050


def _conductor(r):
    """True on conductor cells (inner rod, shell, outer) -- E must be 0 there."""
    return (r < R_INNER) | ((r > R_SH1) & (r < R_SH2)) | (r > R_OUTER)


def load(tag):
    """Reconstruct phi(z,r) from Er with E=0 ENFORCED inside conductors, referenced
    at the grounded outer surface. Integrating the raw Er through the conductor
    interiors (where solve_poisson_efield leaves spurious, inert E) produces
    unphysical kinks -- masking it out is the physically correct reconstruction."""
    with h5py.File(f"{d}/{tag}.h5") as h:
        Er = np.asarray(h["Er"])                # [z, r]
        dz, dr = h.attrs["gridSpacing"]
        z0, r0 = h.attrs["gridGlobalOffset"]
        plasma = int(h.attrs.get("plasma", 0))
    nz, nr = Er.shape
    r = r0 + (np.arange(nr) + 0.5) * dr
    z = z0 + (np.arange(nz) + 0.5) * dz
    E = Er.copy()
    E[:, _conductor(r)] = 0.0                    # E=0 in conductors (kink-free)
    i_ground = int(np.argmin(np.abs(r - R_OUTER)))  # phi=0 at the grounded outer surface
    suffix = np.cumsum((E * dr)[:, ::-1], axis=1)[:, ::-1]
    phi = suffix - suffix[:, [i_ground]]
    return phi, r, z, plasma


pr, r, z, plasma = load("raw")
pv, _, _, _ = load("resolved")
kV = 1e-3
ext = [r[0], r[-1], z[0], z[-1]]

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
vmax = max(np.abs(pv).max(), 1.0) * kV
im0 = ax[0].imshow(pr * kV, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
ax[0].set_title("phi_raw (WarpX native init field) [kV]")
fig.colorbar(im0, ax=ax[0], shrink=0.8)
im1 = ax[1].imshow(pv * kV, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
ax[1].set_title("phi_resolved (corrector's own solve) [kV]")
fig.colorbar(im1, ax=ax[1], shrink=0.8)
for a in ax[:2]:
    a.set_xlabel("r [m]"); a.set_ylabel("z [m]")

ax[2].plot(r, np.nanmean(pr, axis=0) * kV, label="raw", color="C3")
ax[2].plot(r, np.nanmean(pv, axis=0) * kV, label="resolved", color="C0")
for rr, lbl in [(R_INNER, "inner (-1000 V)"), (R_SH1, ""), (R_SH2, "shell (-400 V)"),
                (R_OUTER, "ground")]:
    ax[2].axvline(rr, ls="--", lw=0.8, color="grey", alpha=0.6)
    if lbl:
        ax[2].text(rr, ax[2].get_ylim()[1], lbl, rotation=90, va="top", fontsize=7)
ax[2].set_title("z-averaged phi(r) [kV]")
ax[2].set_xlabel("r [m]"); ax[2].set_ylabel("phi [kV]")
ax[2].grid(True, alpha=0.3); ax[2].legend()

fig.suptitle(f"Basis check (plasma={'on' if plasma else 'off'}): "
             "raw init field vs corrector re-solve")
fig.tight_layout()
fig.savefig(out, dpi=130)
print(f"wrote {out}")
