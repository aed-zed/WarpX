#!/usr/bin/env python3
"""
Phase B test: two independently-biased embedded-boundary electrodes in 3D.

Two separated sphere electrodes (left at +300 V, right at -700 V) sit in a
grounded box. ``MultiElectrodeBiasCorrector`` maintains *both* distinct
potentials at once via the per-electrode harmonic basis and the vacuum
capacitance matrix -- something the single-mode harmonic bias cannot do. A
light co-located plasma provides some dynamics; the net charge is ~0 so the
grounded-plasma charge correction Q_g is small but exercised.
"""

from scipy.constants import e, m_e

from pywarpx import picmi
from pywarpx.callbacks import installafterEsolve, installafterInitEsolve
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector

# ---------------------------------------------------------------------------
# Grid / parameters
# ---------------------------------------------------------------------------
L = 12e-2  # cubic box side
nx = 32
dt = 3.5e-12
max_steps = 100
correction_interval = 10

# Two sphere electrodes, separated along x.
c = 3.0e-2  # center offset
R = 1.5e-2  # radius
V_left = +300.0
V_right = -700.0

# Implicit function: union of two solid spheres (positive = inside solid,
# negative = fluid). max() of the two single-sphere functions R^2 - dist^2.
eb_implicit = (
    f"max({R}*{R}-((x+{c})*(x+{c})+y*y+z*z), "
    f"{R}*{R}-((x-{c})*(x-{c})+y*y+z*z))"
)
# Electrode potential pattern: left sphere (x<0) at V_left, right (x>0) at V_right.
potential_expression = f"({V_left})*(x<0) + ({V_right})*(x>0)"

# ---------------------------------------------------------------------------
# Grid, solver, EB
# ---------------------------------------------------------------------------
grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, nx, nx],
    lower_bound=[-L / 2, -L / 2, -L / 2],
    upper_bound=[L / 2, L / 2, L / 2],
    lower_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
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
# Light co-located plasma (net charge ~0)
# ---------------------------------------------------------------------------
uniform_plasma = picmi.UniformDistribution(
    density=1.0e13,
    rms_velocity=[0.0, 0.0, 0.0],
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
    data_list=["Ex", "Ey", "Ez", "rho"],
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
for sp in (ions, electrons):
    sim.add_species(
        sp, layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=2)
    )
sim.add_diagnostic(field_diag)
sim.add_diagnostic(PN_diag)

# ---------------------------------------------------------------------------
# Multi-electrode corrector
# ---------------------------------------------------------------------------
corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    correction_interval=correction_interval,
    electrodes=[
        {"name": "left", "region": "(x<0)", "potential": V_left},
        {"name": "right", "region": "(x>0)", "potential": V_right},
    ],
    enable_gauss_clean=True,
    verbose=True,
)
installafterInitEsolve(corrector.setup_after_init)
installafterEsolve(corrector.correct_field)

sim.step(max_steps)

# ---------------------------------------------------------------------------
# Final per-electrode voltages (recovered via the capacitance matrix) and the
# capacitance-matrix condition number, written for the analysis script.
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402

v_final = corrector.measure_voltages()
cap = corrector._capacitance
cond = float(np.linalg.cond(cap))

print(f"V_LEFT_FINAL={v_final[0]:.6e}")
print(f"V_RIGHT_FINAL={v_final[1]:.6e}")
print(f"V_LEFT_TARGET={V_left:.6e}")
print(f"V_RIGHT_TARGET={V_right:.6e}")
print(f"CAP_COND={cond:.6e}")
with open("two_electrode_result.txt", "w") as f:
    f.write(f"V_LEFT_FINAL={v_final[0]:.6e}\n")
    f.write(f"V_RIGHT_FINAL={v_final[1]:.6e}\n")
    f.write(f"V_LEFT_TARGET={V_left:.6e}\n")
    f.write(f"V_RIGHT_TARGET={V_right:.6e}\n")
    f.write(f"CAP_COND={cond:.6e}\n")
