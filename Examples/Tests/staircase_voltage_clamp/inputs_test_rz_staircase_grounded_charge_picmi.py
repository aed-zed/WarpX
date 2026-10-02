#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Grounded (plasma-induced) conductor charge: reciprocity vs a real grounded solve.

The staircase clamp computes the charge that the live plasma induces on each grounded
conductor from the unit potentials, Q_g,k = -psi_k^T q (Shockley-Ramo on the discrete
staircase operator). This test places static electron rings between a rod and a
sleeve and compares that value with an independent grounded Poisson solve on the same
operator, read with the same fixed-node observer.
"""

import argparse
import gc

import numpy as np

from pywarpx import picmi
from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

parser = argparse.ArgumentParser()
parser.add_argument("--max-grid-size", type=int, default=1024)
args = parser.parse_args()

NR, NZ = 48, 16
R_ROD, R_SLEEVE_IN, R_SLEEVE_OUT = 6.2e-3, 5.03e-2, 5.53e-2
R_WALL, LENGTH = 6.0e-2, 4.0e-2
TOLERANCE = 1.0e-10  # relative to the total induced charge

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    n_azimuthal_modes=1,
    lower_bound=[0.0, 0.0],
    upper_bound=[R_WALL, LENGTH],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=args.max_grid_size,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
r2 = "(x*x+y*y)"
rod = f"({R_ROD}*{R_ROD}-{r2})"
sleeve = f"(({r2}-{R_SLEEVE_IN}*{R_SLEEVE_IN})*({R_SLEEVE_OUT}*{R_SLEEVE_OUT}-{r2}))"
sim = picmi.Simulation(
    solver=solver,
    max_steps=0,
    particle_shape="linear",
    warpx_embedded_boundary=picmi.EmbeddedBoundary(
        implicit_function=f"-({rod})*({sleeve})", potential=0.0
    ),
    warpx_use_filter=False,
    warpx_current_deposition_algo="esirkepov",
    verbose=0,
)
electrons = picmi.Species(
    name="electrons", particle_type="electron", initial_distribution=None
)
sim.add_species(electrons, layout=None)
sim.initialize_inputs()
sim.initialize_warpx()

corrector = StaircaseBiasCorrector(
    sim,
    correction_interval=1,
    electrodes=[
        {"name": "rod", "region": "(x<0.03)", "potential": -1000.0},
        {"name": "sleeve", "region": "(x>0.03)", "potential": 0.0},
    ],
)
corrector.setup_after_init()

# Static rings at several radii and axial positions (not z-uniform on purpose)
rng = np.random.default_rng(7)
n = 200
radius = rng.uniform(1.0e-2, 4.6e-2, n)
z = rng.uniform(0.0, LENGTH, n)
particles = sim.particles.get("electrons")
particles.add_particles(
    x=radius,
    y=np.zeros(n),
    z=z,
    ux=np.zeros(n),
    uy=np.zeros(n),
    uz=np.zeros(n),
    w=np.full(n, 1.0e6),
    unique_particles=False,
)

result = corrector.compare_grounded_charge()
reciprocity = result["reciprocity"]
solved = result["grounded_solve"]
scale = float(np.sum(np.abs(reciprocity)))
error = float(np.max(np.abs(result["difference"]))) / scale
print(f"reciprocity Q_g = {reciprocity} C; grounded solve = {solved} C")
print(
    f"relative difference = {error:.3e}; solve residual = {result['solve_residual']:.1e}"
)
total = -picmi.constants.q_e * 1.0e6 * n
print(
    f"sum of induced charge / free charge = {np.sum(reciprocity) / total:.6f} (expected ~ -1)"
)

corrector = particles = None
gc.collect()
sim.finalize()

assert error < TOLERANCE, f"grounded charge mismatch {error:.3e}"
assert abs(np.sum(reciprocity) / total + 1.0) < 0.2, (
    "induced charge should nearly balance"
)
