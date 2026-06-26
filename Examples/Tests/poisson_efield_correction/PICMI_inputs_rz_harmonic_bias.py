#!/usr/bin/env python3
"""
RZ curl-preserving harmonic-bias correction, driven by the geometry-agnostic
induced-charge flux (Phase A flux measurement, now available in RZ).

Concentric cylindrical electrodes (inner cathode at -1 kV, outer grounded).
HarmonicBiasCorrector runs in flux mode -- the effective cathode voltage is
inferred from the induced charge eps0*oint w E.n dA on the inner-electrode
region (no line integral, no r_inner/r_outer assumption in the feedback) --
and combines a homogeneous Gauss clean with the harmonic bias. Both operations
add only a gradient / a harmonic (theta-free) field, so E_theta -- the
device's azimuthal inductive field in RZ -- is left untouched.
"""

import os

from scipy.constants import e, m_e

from pywarpx import picmi
from pywarpx.callbacks import installafterEsolve, installafterInitEsolve
from pywarpx.harmonic_bias_corrector import HarmonicBiasCorrector

# Feedback measurement switch (single knob for A/B runs, e.g. on Perlmutter):
#   POISSON_USE_FLUX=1 (default) -> geometry-agnostic induced-charge flux,
#   POISSON_USE_FLUX=0           -> the 2024 axisymmetric line integral.
# Everything else (geometry, clean, bias) is identical between the two.
use_flux = os.environ.get("POISSON_USE_FLUX", "1") != "0"

# ---------------------------------------------------------------------------
# Grid / geometry (same concentric-cylinder Orbitron-like setup as the RZ test)
# ---------------------------------------------------------------------------
Lr = 7e-2
nr = 32
Lz = 1.5e-2
nz = 8
dt = 3.5e-12
correction_interval = 10
max_steps = 100

r_inner = 1.0e-2
r_outer = 6.0e-2
eb_implicit = (
    f"((x*x) - {r_inner}*{r_inner})"
    f"*((x*x) - {r_outer}*{r_outer})"
)
target_potential = -1.0e3
potential_expression = f"{target_potential}*(x*x<3.5e-2**2)"
# Inner-electrode region (x is the radius r in RZ) selecting the cathode surface.
electrode_region = "(x*x<3.5e-2**2)"

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

uniform_plasma = picmi.UniformDistribution(
    density=1.0e14,
    rms_velocity=[0.0, 0.0, 0.0],
    lower_bound=[r_inner, None, -Lz / 2],
    upper_bound=[r_outer, None, Lz / 2],
)
ions = picmi.Species(
    name="ions", mass=3.343585651891582e-27, charge=e,
    initial_distribution=uniform_plasma,
)
electrons = picmi.Species(
    name="electrons", mass=m_e, charge=-e,
    initial_distribution=uniform_plasma,
)

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=100,
    data_list=["Er", "Et", "Ez", "rho"],
    warpx_format="plotfile",
)
PN_diag = picmi.ReducedDiagnostic(name="PN", diag_type="ParticleNumber", period=10)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    warpx_embedded_boundary=embedded_boundary,
    particle_shape="quadratic",
    max_steps=max_steps,
    warpx_amrex_the_arena_is_managed=1,
)
sim.add_applied_field(picmi.ConstantAppliedField(Bz=-0.1))
for sp in (ions, electrons):
    sim.add_species(
        sp, layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=4)
    )
sim.add_diagnostic(field_diag)
sim.add_diagnostic(PN_diag)

# ---------------------------------------------------------------------------
# Curl-preserving harmonic-bias corrector, flux mode (RZ)
# ---------------------------------------------------------------------------
print(f"[RZ harmonic bias] feedback = {'flux' if use_flux else 'line integral'}")
corrector = HarmonicBiasCorrector(
    sim=sim,
    correction_interval=correction_interval,
    potential_expression=potential_expression,
    target_delta_phi=target_potential,
    x_lo_phys=r_inner,   # line-integral bounds (used as feedback if use_flux=0,
    x_hi_phys=r_outer,   # else only for the independent cross-check)
    # Flux feedback when enabled; None -> line-integral fallback (2024 method).
    electrode_weighting=electrode_region if use_flux else None,
    enable_gauss_clean=True,
    verify_curl=False,   # the 3D curl footprint is not computed in RZ
    enable_diagnostics=True,
    diag_name="diag1",
)
installafterInitEsolve(corrector.setup_after_init)
installafterEsolve(corrector.correct_field)

sim.step(max_steps)

# Two independent measures of the maintained potential: the flux-based effective
# voltage (drives the feedback) and the axisymmetric line integral (cross-check).
delta_phi_line = corrector.compute_potential_difference()
delta_phi_flux = corrector._measure_delta_phi()

print(f"FINAL_DELTA_PHI={delta_phi_line:.6e}")
print(f"FINAL_DELTA_PHI_FLUX={delta_phi_flux:.6e}")
print(f"TARGET_DELTA_PHI={target_potential:.6e}")
with open("poisson_correction_result.txt", "w") as f:
    f.write(f"FINAL_DELTA_PHI={delta_phi_line:.6e}\n")
    f.write(f"FINAL_DELTA_PHI_FLUX={delta_phi_flux:.6e}\n")
    f.write(f"TARGET_DELTA_PHI={target_potential:.6e}\n")
