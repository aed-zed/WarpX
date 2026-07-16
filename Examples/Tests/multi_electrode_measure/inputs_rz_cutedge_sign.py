#!/usr/bin/env python3
"""Cut-edge sign attribution: isolate whether the sign-flipped E_r at cut
edges comes from the EB-aware E computation inside computePhi or from the
plain computeE path.

Runs three solves on the same geometry/charge:
  (a) E_native  -- WarpX's init field (the "ground truth" from the native
                   electrostatic init path),
  (b) E_eb      -- our SolvePoissonEfield with the default EB-aware gradient,
  (c) E_plain   -- our SolvePoissonEfield with force_plain_gradient=True
                   (computePhi then computeE, even with EB enabled).

At the cut edges, (a) and (b) have been shown to have equal magnitude but
opposite sign. If (c) agrees with (a), the flip is in the EB-aware E path
(our code to fix); if (c) agrees with (b), the flip is upstream in computeE.

Prints an E_r table at the inner-electrode cut edge (r ~ r_i) for quick
comparison, and writes the full 2D fields to HDF5 for plotting.

Usage (vacuum, no plasma -- cleanest):
    POISSON_TEST_PLASMA=0 python inputs_rz_cutedge_sign.py

Implementation-report Section 17.3, item 3 (in-branch bisection).
"""

import os
import numpy as np

from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve

# --- geometry (same as basis check) ----------------------------------------
r_i, r_m1, r_m2, r_o = 0.005, 0.020, 0.025, 0.050
Lr, Lz, nr, nz = 0.060, 0.020, 120, 16
V1, V2 = -1000.0, -400.0
r_mid1, r_mid2 = 0.0125, 0.0375
use_plasma = os.environ.get("POISSON_TEST_PLASMA", "0") != "0"
n0 = 1.0e14

eb_implicit = (f"((x*x)-{r_i}*{r_i})*((x*x)-{r_m1}*{r_m1})"
               f"*((x*x)-{r_m2}*{r_m2})*((x*x)-{r_o}*{r_o})")
potential_expression = (
    f"{V1}*(x*x<{r_mid1}**2) + {V2}*(x*x>{r_mid1}**2)*(x*x<{r_mid2}**2)")

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
if use_plasma:
    from scipy.constants import e  # noqa: PLC0415
    dist = picmi.AnalyticDistribution(
        density_expression=f"{n0}*(x*x>{r_i}**2)*(x*x<{r_o}**2)",
        rms_velocity=[0.0, 0.0, 0.0],
        lower_bound=[r_i, None, -Lz / 2], upper_bound=[r_o, None, Lz / 2])
    ions = picmi.Species(name="ions", particle_type="H", charge_state=1,
                         mass=1.6726e-27, initial_distribution=dist)
    sim.add_species(ions, layout=picmi.PseudoRandomLayout(
        n_macroparticles_per_cell=8, grid=grid))


OUT = os.environ.get("CUTEDGE_OUT", "cutedge_sign")


def _get_libwarpx():
    from pywarpx._libwarpx import libwarpx  # noqa: PLC0415
    return libwarpx


def _Direction(comp):
    return _get_libwarpx().libwarpx_so.Direction(comp)


def _read_Er():
    """Read E_r as [r, z] numpy array."""
    lib = _get_libwarpx()
    wx = lib.libwarpx_so.get_instance()
    mfr = wx.multifab_register()
    geom = wx.Geom(lev=0).data()
    mf = mfr.get("Efield_fp", dir=_Direction(0), level=0)
    dom = geom.Domain().convert(mf.box_array().ix_type())
    lo, hi = dom.small_end, dom.big_end
    arr = mf[lo[0]:hi[0] + 1, :]
    return arr.get() if hasattr(arr, "get") else np.asarray(arr)


def _save_Er(tag, Er):
    """Dump E_r to HDF5."""
    import h5py  # noqa: PLC0415
    wx = _get_libwarpx().libwarpx_so.get_instance()
    geom = wx.Geom(lev=0).data()
    dr, dz = geom.CellSize()[0], geom.CellSize()[1]
    r0, z0 = geom.ProbLo()[0], geom.ProbLo()[1]
    os.makedirs(OUT, exist_ok=True)
    with h5py.File(os.path.join(OUT, f"Er_{tag}.h5"), "w") as h:
        h.attrs["gridSpacing"] = [dr, dz]
        h.attrs["gridGlobalOffset"] = [r0, z0]
        h.create_dataset("Er", data=Er, compression="gzip")
    print(f"[cutedge] wrote {OUT}/Er_{tag}.h5", flush=True)


def _print_table(r_c, Er_dict, label, iz_mid):
    """Print E_r(r) at a fixed z slice around an electrode surface."""
    inner_ir = int(np.round(r_i / (r_c[1] - r_c[0])))
    r_lo = max(0, inner_ir - 2)
    r_hi = min(len(r_c), inner_ir + 4)

    tags = list(Er_dict.keys())
    hdr = f"{'r[mm]':>8s}"
    for t in tags:
        hdr += f"  {t:>14s}"
    print(f"\n--- {label} (z-index {iz_mid}) ---")
    print(hdr)
    for ir in range(r_lo, r_hi):
        row = f"{r_c[ir]*1e3:8.3f}"
        for t in tags:
            row += f"  {Er_dict[t][ir, iz_mid]:14.1f}"
        print(row)


def _cutedge_check():
    lib = _get_libwarpx()
    wx = lib.libwarpx_so.get_instance()
    geom = wx.Geom(lev=0).data()
    dr = geom.CellSize()[0]
    r0 = geom.ProbLo()[0]

    # (a) native init field
    Er_native = _read_Er().copy()
    _save_Er("native", Er_native)

    # (b) EB-aware solve (default)
    wx.solve_poisson_efield(force_plain_gradient=False)
    Er_eb = _read_Er().copy()
    _save_Er("eb_aware", Er_eb)

    # (c) plain gradient (computePhi + computeE, bypassing EB E-computation)
    wx.solve_poisson_efield(force_plain_gradient=True)
    Er_plain = _read_Er().copy()
    _save_Er("plain_grad", Er_plain)

    # --- comparison table around the inner electrode cut edge ---
    nr = Er_native.shape[0]
    r_c = r0 + (np.arange(nr) + 0.5) * dr
    iz_mid = Er_native.shape[1] // 2

    Er_dict = {"native": Er_native, "eb_aware": Er_eb, "plain_grad": Er_plain}
    _print_table(r_c, Er_dict, "inner electrode (r_i)", iz_mid)

    # also check middle electrode inner surface
    def _print_range(r_center, label):
        ir_c = int(np.round(r_center / dr))
        r_lo2 = max(0, ir_c - 2)
        r_hi2 = min(nr, ir_c + 4)
        hdr = f"{'r[mm]':>8s}"
        for t in Er_dict:
            hdr += f"  {t:>14s}"
        print(f"\n--- {label} (z-index {iz_mid}) ---")
        print(hdr)
        for ir in range(r_lo2, r_hi2):
            row = f"{r_c[ir]*1e3:8.3f}"
            for t in Er_dict:
                row += f"  {Er_dict[t][ir, iz_mid]:14.1f}"
            print(row)

    _print_range(r_m1, "middle electrode inner surface (r_m1)")
    _print_range(r_m2, "middle electrode outer surface (r_m2)")
    _print_range(r_o, "outer electrode (r_o)")

    # summary: at the cut edge(s), does plain_grad agree with native or eb_aware?
    print("\n=== CUT-EDGE SIGN SUMMARY ===")
    print("If plain_grad agrees with native  -> flip is in the EB-aware E path")
    print("If plain_grad agrees with eb_aware -> flip is in computeE (upstream)")

    # find cut edges where native and eb_aware disagree in sign
    sign_diff = np.sign(Er_native) != np.sign(Er_eb)
    nonzero = (np.abs(Er_native) > 1.0) & (np.abs(Er_eb) > 1.0)
    cut_edges = sign_diff & nonzero
    n_cut = int(np.sum(cut_edges))
    print(f"\nFound {n_cut} cut-edge cells with sign flip between native and eb_aware")

    if n_cut > 0:
        agrees_native = np.sum(np.sign(Er_plain[cut_edges]) == np.sign(Er_native[cut_edges]))
        agrees_eb = np.sum(np.sign(Er_plain[cut_edges]) == np.sign(Er_eb[cut_edges]))
        print(f"  plain_grad agrees with native:   {agrees_native}/{n_cut}")
        print(f"  plain_grad agrees with eb_aware:  {agrees_eb}/{n_cut}")
        if agrees_native > agrees_eb:
            print("  -> CONCLUSION: flip is in the EB-aware E path (our code)")
        elif agrees_eb > agrees_native:
            print("  -> CONCLUSION: flip is in computeE (upstream WarpX bug)")
        else:
            print("  -> INCONCLUSIVE: mixed agreement")

    print(flush=True)


installafterInitEsolve(_cutedge_check)
sim.step(0)

import sys  # noqa: E402
sys.stdout.flush()
os._exit(0)
