#!/usr/bin/env python3
"""Basis-consistency diagnostic: is the step-0 per-electrode offset a
representation mismatch, or a capacitance-matrix (resolution) error?

At init we measure V = C^{-1}(Q - Q_g) TWICE on the SAME state, same plasma:
  (a) V_raw       -- on WarpX's native initial field (what the corrector
                     normally sees at step 0),
  (b) V_resolved  -- immediately after warpx.solve_poisson_efield(), i.e. the
                     field re-expressed in the corrector's OWN solve basis.

Interpretation:
  * V_raw off  AND V_resolved exact  -> representation mismatch: WarpX's init
    field is not bit-identical to the corrector's solve basis; the one-line
    re-solve is the cure (the step-0 offset is cosmetic).
  * BOTH off                          -> the capacitance matrix itself is
    inaccurate (under-resolved features) -> resolving the in-between regions is
    the real lever, and a re-solve won't help.

Run vacuum (POISSON_TEST_PLASMA=0) and plasma (=1): if V_raw is off only with
plasma, the mismatch is the plasma-charge representation (init field lacks the
screening that Q_g subtracts); if off in vacuum too, it is the EB/solver path.
"""

import os
import numpy as np

from scipy.constants import e

from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve
from pywarpx.multi_electrode_logger import MultiElectrodePotentialLogger

# --- concentric 3-conductor RZ geometry (as in inputs_rz.py) ---------------
r_i, r_m1, r_m2, r_o = 0.005, 0.020, 0.025, 0.050
Lr, Lz, nr, nz = 0.060, 0.020, 120, 16
V1, V2 = -1000.0, -400.0
r_mid1, r_mid2 = 0.0125, 0.0375
use_plasma = os.environ.get("POISSON_TEST_PLASMA", "1") != "0"
n0 = 1.0e14

eb_implicit = (f"((x*x)-{r_i}*{r_i})*((x*x)-{r_m1}*{r_m1})"
               f"*((x*x)-{r_m2}*{r_m2})*((x*x)-{r_o}*{r_o})")
potential_expression = f"{V1}*(x*x<{r_mid1}**2) + {V2}*(x*x>{r_mid1}**2)*(x*x<{r_mid2}**2)"

grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz], n_azimuthal_modes=1,
    lower_bound=[0.0, -Lz / 2], upper_bound=[Lr, Lz / 2],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["neumann", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    warpx_blocking_factor=8, warpx_max_grid_size=256,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee")
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=eb_implicit, potential=potential_expression,
    cover_multiple_cuts=True)

sim = picmi.Simulation(solver=solver, time_step_size=3.0e-12,
                       warpx_embedded_boundary=embedded_boundary,
                       particle_shape="linear", max_steps=0)
if use_plasma:
    dist = picmi.AnalyticDistribution(
        density_expression=f"{n0}*(x*x>{r_i}**2)*(x*x<{r_o}**2)",
        rms_velocity=[0.0, 0.0, 0.0],
        lower_bound=[r_i, None, -Lz / 2], upper_bound=[r_o, None, Lz / 2])
    ions = picmi.Species(name="ions", particle_type="H", charge_state=1,
                         mass=1.6726e-27, initial_distribution=dist)
    sim.add_species(ions, layout=picmi.PseudoRandomLayout(
        n_macroparticles_per_cell=8, grid=grid))

logger = MultiElectrodePotentialLogger(
    sim=sim,
    electrodes=[
        {"name": "inner", "region": f"(x*x<{r_mid1}**2)", "potential": V1},
        {"name": "shell", "region": f"(x*x>{r_mid1}**2)*(x*x<{r_mid2}**2)", "potential": V2},
    ],
    period=1, out_csv="basis_check.csv",
)


OUT = os.environ.get("BASIS_OUT", "basis_fields")


def _save_fields(tag):
    """Dump Er, Ez and reconstructed phi(z,r) of the live field to <OUT>/<tag>.h5."""
    import h5py  # noqa: PLC0415

    wx = logger._warpx()
    mfr = logger._mfr()
    geom = wx.Geom(lev=0).data()
    dr, dz = geom.CellSize()[0], geom.CellSize()[1]   # RZ: axis 0 = r, 1 = z
    r0, z0 = geom.ProbLo()[0], geom.ProbLo()[1]

    def comp(c):  # -> [z, r] host array
        mf = mfr.get("Efield_fp", dir=logger._Direction(c), level=0)
        dom = geom.Domain().convert(mf.box_array().ix_type())
        lo, hi = dom.small_end, dom.big_end
        arr = mf[lo[0]:hi[0] + 1, :]                  # [r, z]
        a = arr.get() if hasattr(arr, "get") else np.asarray(arr)
        return a.T                                    # [z, r]

    Er, Ez = comp(0), comp(2)
    # phi(r) = int_r^{r_ground} Er dr' from the clean radial integral (Er is [z,r])
    nr = Er.shape[1]
    r_c = r0 + (np.arange(nr) + 0.5) * dr
    ir_g = int(np.clip(np.searchsorted(r_c, r_o + 0.5 * (Lr - r_o)), 0, nr - 1))
    suffix = np.cumsum((Er * dr)[:, ::-1], axis=1)[:, ::-1]
    phi = suffix - suffix[:, [ir_g]]

    os.makedirs(OUT, exist_ok=True)
    with h5py.File(os.path.join(OUT, f"{tag}.h5"), "w") as h:
        h.attrs["gridSpacing"] = [dz, dr]
        h.attrs["gridGlobalOffset"] = [z0, r0]
        h.attrs["plasma"] = int(use_plasma)
        for name, a in (("Er", Er), ("Ez", Ez), ("phi", phi)):
            h.create_dataset(name, data=a, compression="gzip")
    print(f"[basis] wrote {OUT}/{tag}.h5", flush=True)


def _basis_check():
    logger.setup_after_init()                       # build C (corrector's own solves)
    v_raw = logger.measure_now()                    # (a) WarpX's native init field
    _save_fields("raw")
    from pywarpx._libwarpx import libwarpx          # noqa: PLC0415
    libwarpx.libwarpx_so.get_instance().solve_poisson_efield()
    v_resolved = logger.measure_now()               # (b) corrector's own solve basis
    _save_fields("resolved")

    tgt = np.array([e["potential"] for e in logger.electrodes])
    names = logger.names
    print("\n==================  BASIS-CONSISTENCY CHECK  "
          f"(plasma={'on' if use_plasma else 'off'})  ==================")
    print(f"{'electrode':8s} {'target':>10s} {'V_raw':>12s} {'err_raw':>9s} "
          f"{'V_resolved':>12s} {'err_res':>9s}")
    for i, nm in enumerate(names):
        er = abs(v_raw[i] - tgt[i]) / max(abs(tgt[i]), 1.0)
        es = abs(v_resolved[i] - tgt[i]) / max(abs(tgt[i]), 1.0)
        print(f"{nm:8s} {tgt[i]:10.2f} {v_raw[i]:12.3f} {er:8.2%} "
              f"{v_resolved[i]:12.3f} {es:8.2%}")
    print("interpretation: raw off + resolved exact -> representation mismatch "
          "(re-solve is the cure); both off -> capacitance/resolution.\n", flush=True)


installafterInitEsolve(_basis_check)
sim.step(0)

import sys  # noqa: E402
sys.stdout.flush()
os._exit(0)
