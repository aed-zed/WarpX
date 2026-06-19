#!/usr/bin/env python3
"""
Combined-scheme test for the curl-preserving E-field correction (Phase 0 + clean).

Identical Orbitron setup to ``PICMI_inputs_3d_harmonic_bias.py``, but the
corrector runs with ``enable_gauss_clean=True``: each correction first applies a
homogeneous Boris/Marder Gauss clean (``warpx.clean_efield_gauss_homogeneous()``)
and then the harmonic bias. This realizes the combined scheme of report Eq. 13.9
-- the clean removes the Gauss-law residual and preserves curl, the bias resets
the electrode potential -- so the corrected field satisfies all three invariants
(Gauss clean, curl preserved, potential maintained).

The analysis (``analysis_harmonic_bias.py``) checks the potential is maintained,
the bias remains curl-free in the bulk, and the field magnitude is physical.
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
# Combined corrector: homogeneous Gauss clean + harmonic bias
# ---------------------------------------------------------------------------
corrector = HarmonicBiasCorrector(
    sim=sim,
    correction_interval=correction_interval,
    potential_expression=potential_expression,
    target_delta_phi=target_potential,
    x_lo_phys=r_inner,
    x_hi_phys=r_outer,
    enable_gauss_clean=True,
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

# Final potential check + curl footprint, written for the analysis script.
delta_phi_final = corrector.compute_potential_difference()
cf = corrector.curl_footprint or {"bulk_rel": float("nan"), "max_rel": float("nan")}

print(f"FINAL_DELTA_PHI={delta_phi_final:.6e}")
print(f"TARGET_DELTA_PHI={target_potential:.6e}")
print(f"CURL_BULK_REL={cf['bulk_rel']:.6e}")
print(f"CURL_MAX_REL={cf['max_rel']:.6e}")
with open("poisson_correction_result.txt", "w") as f:
    f.write(f"FINAL_DELTA_PHI={delta_phi_final:.6e}\n")
    f.write(f"TARGET_DELTA_PHI={target_potential:.6e}\n")
    f.write(f"CURL_BULK_REL={cf['bulk_rel']:.6e}\n")
    f.write(f"CURL_MAX_REL={cf['max_rel']:.6e}\n")
