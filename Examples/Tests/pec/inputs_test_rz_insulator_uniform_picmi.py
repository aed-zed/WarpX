#!/usr/bin/env python3
"""Uniform tangential current must not generate axial structure at insulating ends.

No EB, voltage clamp, imposed fields, collisions or particle absorption. Neutral
electron/proton rings fill every axial cell identically. Electrons move radially;
gathering is disabled to isolate deposition and Maxwell's response. The old
boundary treatment is an explicit negative control, not an accepted solution.
"""

import argparse
import gc

import numpy as np

from pywarpx import picmi

parser = argparse.ArgumentParser()
parser.add_argument("--normalize", type=int, choices=(0, 1), default=1)
args = parser.parse_args()

nz = 16
dz = 1.0e-3
grid = picmi.CylindricalGrid(
    number_of_cells=[16, nz],
    n_azimuthal_modes=1,
    lower_bound=[0.0, 0.0],
    upper_bound=[16 * dz, nz * dz],
    lower_boundary_conditions=["none", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=8,
)
sim = picmi.Simulation(
    solver=picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.5),
    max_steps=10,
    particle_shape="linear",
    warpx_current_deposition_algo="esirkepov",
    warpx_use_filter=False,
    verbose=0,
)
radial_speed = 1.0e6
proper_speed = radial_speed / np.sqrt(1 - (radial_speed / picmi.constants.c) ** 2)
for name, kind, speed in (
    ("electrons", "electron", proper_speed),
    ("protons", "proton", 0),
):
    distribution = picmi.ParticleListDistribution(
        x=np.full(nz, 8.25 * dz),
        y=np.zeros(nz),
        z=(np.arange(nz) + 0.5) * dz,
        ux=np.full(nz, speed),
        uy=np.zeros(nz),
        uz=np.zeros(nz),
        weight=np.full(nz, 1.0e3),
    )
    sim.add_species(
        picmi.Species(
            name=name,
            particle_type=kind,
            initial_distribution=distribution,
            warpx_do_not_gather=True,
            warpx_random_theta=False,
        ),
        layout=None,
    )

sim.initialize_inputs()
import pywarpx  # noqa: E402

pywarpx.boundary.field_lo = ["none", "pec_insulator"]
pywarpx.boundary.field_hi = ["pec", "pec_insulator"]
insulator = pywarpx.warpx.get_bucket("insulator")
setattr(insulator, "area_z_lo(x,y)", "1")
setattr(insulator, "area_z_hi(x,y)", "1")
insulator.normalize_nodal_sources = bool(args.normalize)
sim.initialize_warpx()

register = sim.extension.warpx.multifab_register()
direction = sim.extension.libwarpx_so.Direction


def maximum(name, component):
    # Collective native reduction over valid fields only; shared nodes do not
    # alter maxima. No rank-zero-only collective or host-only field gather.
    return register.get(name, dir=direction(component), level=0).norm0(
        0, 0, False, False
    )


try:
    sim.step(10)
    er = maximum("Efield_fp", 0)
    jr = maximum("current_fp", 0)
    assert er > 0 and jr > 0, (
        "The test must exercise a nonzero current and electric response"
    )
    magnetic_error = picmi.constants.c * maximum("Bfield_fp", 1) / er
    axial_error = maximum("Efield_fp", 2) / er
    if args.normalize:
        assert magnetic_error < 2.0e-12, magnetic_error
        assert axial_error < 2.0e-12, axial_error
    else:
        assert magnetic_error > 1.0e-5, (
            "Legacy control must expose the boundary mismatch"
        )
    print(
        f"INSULATOR_UNIFORM normalize={args.normalize} cB/Er={magnetic_error} Ez/Er={axial_error}"
    )
finally:
    register = None
    gc.collect()
    sim.finalize()
