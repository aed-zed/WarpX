#!/usr/bin/env python3
"""
RZ (cylindrical) variant of the Poisson E-field correction test.

Two concentric cylindrical electrodes (inner cathode at r_inner, outer anode
at r_outer) are represented as an embedded boundary. An electromagnetic
(cylindrical Yee) solver advances the fields; the PoissonEfieldCorrector
callback periodically re-solves Poisson with the electrode potential so the
-1 kV cathode potential is maintained instead of drifting.

Only the axisymmetric mode (n_azimuthal_modes = 1) is supported by the RZ
electrostatic Poisson solve.
"""

from scipy.constants import e, m_e

from pywarpx import picmi
from pywarpx.callbacks import installafterEsolve, installafterInitEsolve
from pywarpx.poisson_efield_corrector import PoissonEfieldCorrector

# ---------------------------------------------------------------------------
# Grid parameters
# ---------------------------------------------------------------------------
Lr = 7e-2  # radial extent (a bit beyond the outer electrode)
nr = 32
Lz = 1.5e-2  # short z extent, periodic
nz = 8
dt = 3.5e-12

# Embedded boundary: concentric cylinders (inner cathode, outer anode).
# In RZ the implicit-function variable x is the radius r. WarpX convention:
# negative = simulation (fluid) volume, positive = inside solid. The plain
# product is negative only in the gap r_inner < r < r_outer (do NOT negate).
r_inner = 1.0e-2
r_outer = 6.0e-2
eb_implicit = (
    f"((x*x) - {r_inner}*{r_inner})"
    f"*((x*x) - {r_outer}*{r_outer})"
)
target_potential = -1.0e3  # -1 kV on the inner electrode (cathode)
# -1 kV on the inner electrode (r < midpoint), 0 V on the outer electrode.
potential_expression = f"{target_potential}*(x*x<3.5e-2**2)"

correction_interval = 10
max_steps = 100

# ---------------------------------------------------------------------------
# Grid, solver, EB
# ---------------------------------------------------------------------------
grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    n_azimuthal_modes=1,
    lower_bound=[0.0, -Lz / 2],
    upper_bound=[Lr, Lz / 2],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=1024,
)

solver = picmi.ElectromagneticSolver(grid=grid, method="Yee")

embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=eb_implicit,
    potential=potential_expression,
    cover_multiple_cuts=True,
)

# ---------------------------------------------------------------------------
# Particles: co-located ion/electron plasma in the gap so the net charge
# density is ~0 (minimizes Gauss-law violation, as in the 3D test).
# ---------------------------------------------------------------------------
uniform_plasma = picmi.UniformDistribution(
    density=1.0e14,
    rms_velocity=[0.0, 0.0, 0.0],
    lower_bound=[r_inner, None, -Lz / 2],
    upper_bound=[r_outer, None, Lz / 2],
)
ions = picmi.Species(
    name="ions",
    mass=3.343585651891582e-27,  # deuterium
    charge=e,
    initial_distribution=uniform_plasma,
)
electrons = picmi.Species(
    name="electrons",
    mass=m_e,
    charge=-e,
    initial_distribution=uniform_plasma,
)

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=50,
    data_list=["Er", "Et", "Ez", "Br", "Bt", "Bz", "rho"],
    warpx_format="plotfile",
)
PN_diag = picmi.ReducedDiagnostic(
    name="PN",
    diag_type="ParticleNumber",
    period=10,
)

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    warpx_embedded_boundary=embedded_boundary,
    particle_shape="quadratic",
    max_steps=max_steps,
    warpx_amrex_the_arena_is_managed=1,
)

sim.add_applied_field(picmi.ConstantAppliedField(Bz=-0.1))

sim.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=4),
)
sim.add_species(
    electrons,
    layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=4),
)

sim.add_diagnostic(field_diag)
sim.add_diagnostic(PN_diag)

# ---------------------------------------------------------------------------
# Poisson E-field corrector
# ---------------------------------------------------------------------------
corrector = PoissonEfieldCorrector(
    sim=sim,
    correction_interval=correction_interval,
    potential_expression=potential_expression,
    enable_diagnostics=True,
    diag_name="diag1",
)

installafterInitEsolve(corrector.setup_after_init)
installafterEsolve(corrector.correct_field)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
sim.step(max_steps)

# Final potential check: integrate Er from inner to outer electrode.
delta_phi_final = corrector.compute_potential_difference(
    x_lo_phys=r_inner, x_hi_phys=r_outer
)
print(f"FINAL_DELTA_PHI={delta_phi_final:.6e}")
print(f"TARGET_DELTA_PHI={target_potential:.6e}")
with open("poisson_correction_result.txt", "w") as f:
    f.write(f"FINAL_DELTA_PHI={delta_phi_final:.6e}\n")
    f.write(f"TARGET_DELTA_PHI={target_potential:.6e}\n")
