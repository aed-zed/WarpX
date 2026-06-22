#!/usr/bin/env python3
"""
Phase 0 test for the curl-preserving harmonic-bias E-field correction.

Same simplified Orbitron setup as ``PICMI_inputs_3d.py`` (EM Yee solver,
concentric-cylinder embedded-boundary electrodes), but the electrode potential
is maintained by ``HarmonicBiasCorrector`` -- which *adds* a scaled, precomputed
vacuum electrode field instead of *replacing* E with a full Poisson solve.

The point of Phase 0 is to demonstrate that this additive (harmonic-bias)
correction (a) maintains the electrode potential and (b) is curl-free, so the
self-consistent rotational/inductive field is preserved.  The curl footprint of
the stored vacuum field is written to the result file for the analysis script.
"""

from scipy.constants import c, e, m_e

from pywarpx import picmi
from pywarpx.callbacks import installafterEsolve, installafterInitEsolve
from pywarpx.harmonic_bias_corrector import HarmonicBiasCorrector

# ---------------------------------------------------------------------------
# Physical / grid parameters
# ---------------------------------------------------------------------------
Lx = 12e-2  # box half-size in x, y
nx = 32
Lz = 1.5e-2  # short z extent, periodic
nz = 8
dt = 3.5e-12

# Injection
current = 10.0e17  # particles/s
macro_weight = 1.0e7

# Embedded boundary: concentric cylinders (inner cathode, outer anode)
r_inner = 1.0e-2
r_outer = 6.0e-2
# WarpX EB convention: negative = simulation (fluid) region, positive = inside
# the solid. Product of the two ring factors is negative only in the gap
# r_inner < r < r_outer, which is the plasma region between the electrodes.
eb_implicit = (
    f"((x*x + y*y) - {r_inner}*{r_inner})"
    f"*((x*x + y*y) - {r_outer}*{r_outer})"
)
target_potential = -1.0e3  # -1 kV on the inner electrode (cathode)
potential_expression = f"{target_potential}*(x*x+y*y<3.e-2**2)"

# Correction settings
correction_interval = 10

# ---------------------------------------------------------------------------
# Grid, solver, EB
# ---------------------------------------------------------------------------
grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, nx, nz],
    lower_bound=[-Lx / 2, -Lx / 2, -Lz / 2],
    upper_bound=[Lx / 2, Lx / 2, Lz / 2],
    lower_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "periodic"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "periodic"],
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
# Particles
# ---------------------------------------------------------------------------
flux_distribution = picmi.UniformFluxDistribution(
    flux=current / (1.0e-2 * 1.5e-2),
    flux_normal_axis="y",
    surface_flux_position=0,
    flux_direction=+1,
    directed_velocity=[0, 3.57e-3 * c, 0],
    lower_bound=[4.0e-2, -1, -0.75e-2],
    upper_bound=[5.0e-2, 1, 0.75e-2],
)

ions = picmi.Species(
    name="ions",
    mass=3.343585651891582e-27,  # deuterium
    charge=e,
    initial_distribution=flux_distribution,
)
electrons = picmi.Species(
    name="electrons",
    mass=m_e,
    charge=-e,
    initial_distribution=flux_distribution,
)

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=50,
    data_list=["Ex", "Ey", "Ez", "Bx", "By", "Bz", "rho"],
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
max_steps = 100

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    warpx_embedded_boundary=embedded_boundary,
    particle_shape="quadratic",
    max_steps=max_steps,
    warpx_amrex_the_arena_is_managed=1,
)

sim.add_applied_field(picmi.ConstantAppliedField(Bz=-0.1))

nppcell = (
    current * dt * (Lx / nx) * (Lz / nz) / (1.0e-2 * 1.5e-2) / macro_weight
)
sim.add_species(
    ions,
    layout=picmi.PseudoRandomLayout(
        grid=grid, n_macroparticles_per_cell=nppcell
    ),
)
sim.add_species(
    electrons,
    layout=picmi.PseudoRandomLayout(
        grid=grid, n_macroparticles_per_cell=nppcell
    ),
)

sim.add_diagnostic(field_diag)
sim.add_diagnostic(PN_diag)

# ---------------------------------------------------------------------------
# Curl-preserving harmonic-bias corrector
# ---------------------------------------------------------------------------
# Phase A: drive the feedback with the geometry-agnostic induced-charge flux,
# weighting by the inner-electrode region (same region as the potential
# expression). x_lo_phys/x_hi_phys are kept only for the independent
# line-integral cross-check below.
electrode_region = "(x*x+y*y<3.e-2**2)"
corrector = HarmonicBiasCorrector(
    sim=sim,
    correction_interval=correction_interval,
    potential_expression=potential_expression,
    target_delta_phi=target_potential,
    x_lo_phys=r_inner,
    x_hi_phys=r_outer,
    electrode_weighting=electrode_region,
    verify_curl=True,
    enable_diagnostics=True,
    diag_name="diag1",
)

installafterInitEsolve(corrector.setup_after_init)
installafterEsolve(corrector.correct_field)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
sim.step(max_steps)

# Final potential checks: two *independent* geometry measures of the same
# maintained field --
#  * line integral of Ex from inner to outer electrode (assumes axisymmetry),
#  * geometry-agnostic induced-charge flux (Phase A, used by the feedback).
# Their agreement validates the flux measurement against the established one.
delta_phi_line = corrector.compute_potential_difference()
delta_phi_flux = corrector._measure_delta_phi()
cf = corrector.curl_footprint or {"bulk_rel": float("nan"), "max_rel": float("nan")}

print(f"FINAL_DELTA_PHI={delta_phi_line:.6e}")
print(f"FINAL_DELTA_PHI_FLUX={delta_phi_flux:.6e}")
print(f"TARGET_DELTA_PHI={target_potential:.6e}")
print(f"CURL_BULK_REL={cf['bulk_rel']:.6e}")
print(f"CURL_MAX_REL={cf['max_rel']:.6e}")
with open("poisson_correction_result.txt", "w") as f:
    f.write(f"FINAL_DELTA_PHI={delta_phi_line:.6e}\n")
    f.write(f"FINAL_DELTA_PHI_FLUX={delta_phi_flux:.6e}\n")
    f.write(f"TARGET_DELTA_PHI={target_potential:.6e}\n")
    f.write(f"CURL_BULK_REL={cf['bulk_rel']:.6e}\n")
    f.write(f"CURL_MAX_REL={cf['max_rel']:.6e}\n")
