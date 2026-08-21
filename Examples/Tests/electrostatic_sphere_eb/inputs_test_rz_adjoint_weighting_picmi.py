#!/usr/bin/env python3
"""Validate the discrete Shockley--Ramo observer in RZ.

The test compares the charge induced on a grounded spherical EB in two fully
discrete ways:

1. deposit the off-node CIC particles, solve the metric-aware RZ EB Poisson
   problem, and integrate the resulting EB flux;
2. deposit the same particles and evaluate ``-sum rho_i V_i Psi_i`` with Psi
   obtained from the exact transpose of that operator and flux functional.

Agreement exercises the RZ ``r+/-dr/2`` rows, the ``r=0`` regularity row,
cut-edge gradient stencils, ``2*pi*r_bnd`` surface measure, cylindrical nodal
volumes, sign convention, and the particle deposit.
"""

import numpy as np

from pywarpx import picmi
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector
from pywarpx.multi_electrode_logger import (
    GroundedChargeCrossCheck,
    MultiElectrodeClampTelemetry,
)

nr, nz = 32, 64
grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    n_azimuthal_modes=1,
    lower_bound=[0.0, -0.8],
    upper_bound=[0.8, 0.8],
    lower_boundary_conditions=["none", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=32,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
eb = picmi.EmbeddedBoundary(
    implicit_function="-(x*x+y*y+z*z-radius*radius)",
    potential=0.0,
    radius=0.18,
)

# Deliberately off-node and at several radii, including a point whose CIC
# support touches the small-volume near-axis nodes.
distribution = picmi.ParticleListDistribution(
    x=[0.0123, 0.2871, 0.4317, 0.6213],
    y=[0.0, 0.0311, -0.0247, 0.0189],
    z=[0.2637, -0.3173, 0.1179, -0.4121],
    ux=[0.0] * 4,
    uy=[0.0] * 4,
    uz=[0.0] * 4,
    weight=[1.0e9, 1.7e9, 0.8e9, 1.3e9],
)
electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=distribution,
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=1.0e-12,
    max_steps=0,
    particle_shape="linear",
    warpx_embedded_boundary=eb,
    warpx_use_filter=False,
    verbose=0,
)
sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[0, 0, 0], grid=grid),
)

corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    correction_interval=1,
    electrodes=[{"name": "sphere", "region": "1", "potential": 0.0}],
    qg_mode="reciprocity",
    gather_mode="deposit",
    particle_shape=1,
    filter_passes=0,
    adjoint_tolerance=2.0e-10,
    adjoint_solver="pmlmg",
    adjoint_max_iterations=200,
    verbose=True,
)

sim.initialize_inputs()
sim.initialize_warpx()
corrector.setup_after_init()

warpx = corrector._warpx()
warpx.set_potential_on_eb("0.0")
warpx.solve_poisson_efield()
q_solve = float(warpx.compute_eb_charge(weighting="1", field="Efield_fp"))
q_adjoint = float(corrector._grounded_charge_via_reciprocity(0)[0])

abs_error = abs(q_adjoint - q_solve)
scale = max(abs(q_solve), abs(q_adjoint), 1.0e-30)
rel_error = abs_error / scale
print(
    "RZ adjoint reciprocity: "
    f"grounded_solve={q_solve:+.16e} C, "
    f"adjoint={q_adjoint:+.16e} C, rel_error={rel_error:.3e}"
)

assert np.signbit(q_adjoint) == np.signbit(q_solve), (
    "RZ adjoint and grounded solve disagree in sign: "
    f"{q_adjoint:+.16e} vs {q_solve:+.16e} C"
)
assert rel_error < 2.0e-6, (
    "RZ discrete adjoint identity failed: "
    f"relative error {rel_error:.3e} >= 2e-6"
)

# Exercise the production callback path as well as the identity above. Routine
# telemetry must consume the cached state without another measurement; the
# sparse cross-check intentionally performs one grounded solve and restores E.
telemetry = MultiElectrodeClampTelemetry(
    corrector,
    out_csv="clamp_telemetry_test.csv",
    setup_json="clamp_setup_test.json",
)
cross_check = GroundedChargeCrossCheck(
    corrector,
    period=1,
    out_csv="clamp_grounded_crosscheck_test.csv",
)
telemetry.setup()
cross_check.setup()
corrector.correct_field()
telemetry.log()
cross_check.log()

state = corrector.last_correction_state()
assert state is not None
np.testing.assert_allclose(
    state["voltage_after_predicted"],
    np.asarray(corrector.v_target),
    atol=1.0e-12,
    rtol=0.0,
)
