#!/usr/bin/env python3
"""Upstream cut-edge sign test: verify the native WarpX electrostatic solver's
E-field sign convention at EB cut edges, using ONLY upstream features (no custom
SolvePoissonEfield, no corrector code).

Uses a simple coaxial geometry (inner conductor at V0, outer grounded) where
E_r is known analytically: E_r(r) = V0 / (r * ln(r_o/r_i)), pointing radially
outward for V0 < 0 (i.e. E_r < 0 at all fluid cells for negative potential).

Compares the native init field against the analytic expectation at the cut edges
to verify the sign is correct. If the native solver's cut-edge sign is correct,
our SolvePoissonEfield's flip is confirmed as a bug in our EB-aware E path.

This script uses NO custom pywarpx extensions beyond the standard PICMI +
callback interface that ships with WarpX. It can be run on a clean `development`
worktree.

Usage:
    python inputs_rz_upstream_sign.py

Implementation-report Section 17.3, item 3 (upstream isolation).
"""

import numpy as np

from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve

r_i, r_o = 0.010, 0.050
Lr, Lz, nr, nz = 0.060, 0.020, 120, 16
V0 = -1000.0

eb_implicit = f"((x*x)-{r_i}*{r_i})*((x*x)-{r_o}*{r_o})"
potential_expression = f"{V0}*(x*x<{((r_i + r_o) / 2)}**2)"

grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz], n_azimuthal_modes=1,
    lower_bound=[0.0, -Lz / 2], upper_bound=[Lr, Lz / 2],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["neumann", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    warpx_blocking_factor=8, warpx_max_grid_size=256,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=eb_implicit, potential=potential_expression,
    cover_multiple_cuts=True)

sim = picmi.Simulation(solver=solver,
                       warpx_embedded_boundary=embedded_boundary,
                       particle_shape="linear", max_steps=0)


def _check_upstream_sign():
    from pywarpx._libwarpx import libwarpx  # noqa: PLC0415
    wx = libwarpx.libwarpx_so.get_instance()
    mfr = wx.multifab_register()
    geom = wx.Geom(lev=0).data()
    dr = geom.CellSize()[0]
    r0 = geom.ProbLo()[0]

    mf = mfr.get("Efield_fp", dir=libwarpx.libwarpx_so.Direction(0), level=0)
    dom = geom.Domain().convert(mf.box_array().ix_type())
    lo, hi = dom.small_end, dom.big_end
    arr = mf[lo[0]:hi[0] + 1, :]
    Er = arr.get() if hasattr(arr, "get") else np.asarray(arr)

    nr_cells = Er.shape[0]
    r_c = r0 + (np.arange(nr_cells) + 0.5) * dr
    iz_mid = Er.shape[1] // 2

    # analytic: E_r = V0 / (r * ln(r_o / r_i)) for r_i < r < r_o
    Er_analytic = np.where(
        (r_c > r_i) & (r_c < r_o),
        V0 / (r_c * np.log(r_o / r_i)),
        0.0
    )

    # print comparison around the inner cut edge
    ir_i = int(np.round(r_i / dr))
    print("\n=== UPSTREAM NATIVE SOLVER CUT-EDGE SIGN CHECK ===")
    print(f"Coaxial geometry: r_i={r_i*1e3:.1f} mm, r_o={r_o*1e3:.1f} mm, V0={V0:.0f} V")
    print(f"Analytic: E_r = {V0:.0f} / (r * ln({r_o/r_i:.3f})) -- should be negative throughout\n")

    print(f"{'r[mm]':>8s}  {'Er_native':>14s}  {'Er_analytic':>14s}  {'match?':>8s}")
    for ir in range(max(0, ir_i - 2), min(nr_cells, ir_i + 5)):
        r_mm = r_c[ir] * 1e3
        en = Er[ir, iz_mid]
        ea = Er_analytic[ir]
        if abs(ea) < 1.0:
            status = "covered" if abs(en) < 1.0 else "nonzero!"
        else:
            status = "OK" if np.sign(en) == np.sign(ea) else "FLIPPED!"
        print(f"{r_mm:8.3f}  {en:14.1f}  {ea:14.1f}  {status:>8s}")

    # same for the outer cut edge
    ir_o = int(np.round(r_o / dr))
    print()
    for ir in range(max(0, ir_o - 3), min(nr_cells, ir_o + 3)):
        r_mm = r_c[ir] * 1e3
        en = Er[ir, iz_mid]
        ea = Er_analytic[ir]
        if abs(ea) < 1.0:
            status = "covered" if abs(en) < 1.0 else "nonzero!"
        else:
            status = "OK" if np.sign(en) == np.sign(ea) else "FLIPPED!"
        print(f"{r_mm:8.3f}  {en:14.1f}  {ea:14.1f}  {status:>8s}")

    # overall verdict
    fluid = (r_c > r_i + dr) & (r_c < r_o - dr)
    sign_ok = np.all(np.sign(Er[fluid, iz_mid]) == np.sign(Er_analytic[fluid]))
    print(f"\nBulk fluid cells sign-consistent: {sign_ok}")

    cut_inner = (r_c > r_i - dr) & (r_c < r_i + 2 * dr)
    cut_outer = (r_c > r_o - 2 * dr) & (r_c < r_o + dr)
    for label, mask in [("inner", cut_inner), ("outer", cut_outer)]:
        nonzero_mask = mask & (np.abs(Er_analytic) > 1.0)
        if np.any(nonzero_mask):
            ok = np.all(np.sign(Er[nonzero_mask, iz_mid]) == np.sign(Er_analytic[nonzero_mask]))
            print(f"Cut edges at {label} electrode: sign-correct = {ok}")

    print("\nIf ALL signs are correct here, the native (upstream) solver is fine")
    print("and the flip in SolvePoissonEfield is in our EB-aware E path.\n", flush=True)


installafterInitEsolve(_check_upstream_sign)
sim.step(0)

import sys  # noqa: E402
sys.stdout.flush()
import os; os._exit(0)  # noqa: E702
