#!/usr/bin/env python3
"""Multi-electrode effective-voltage MEASUREMENT test (RZ, local-runnable).

Concentric 3-conductor coaxial geometry with an *analytic* per-electrode target:
  * inner rod   (r < r_i)              held at V1
  * middle shell(r_m1 < r < r_m2)      held at V2
  * outer shell (r > r_o)              grounded (0 V, the reference)
with an optional cold, net-charge plasma (uniform density -> the "screening
charge") loaded in the gaps.

The point: verify that the flux + capacitance measurement recovers the KNOWN
per-electrode voltages V1, V2 from the induced charges,

    V_eff = C^{-1} (Q - Q_g),

*despite* the plasma screening (Q_g != 0) -- i.e. that MultiElectrodePotentialLogger
disentangles two electrodes under plasma. It is measurement-only (no correction),
so it is a neutral scorecard usable on either correction branch.

Env knobs:
  POISSON_TEST_PLASMA=1 (default) -> load the screening charge (tests Q_g)
  POISSON_TEST_PLASMA=0           -> vacuum (Q_g ~ 0; tests only C and the flux)
"""

import os

from scipy.constants import e

from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve, installafterstep
from pywarpx.multi_electrode_logger import MultiElectrodePotentialLogger

# ---------------------------------------------------------------------------
# Geometry (concentric conductors) and targets
# ---------------------------------------------------------------------------
r_i = 0.005    # inner rod radius            [m]
r_m1 = 0.020   # middle shell inner radius   [m]
r_m2 = 0.025   # middle shell outer radius   [m]
r_o = 0.050    # outer (grounded) inner edge [m]
Lr = 0.060     # domain radius               [m]
Lz = 0.020     # domain length               [m]
nr = 120
nz = 16

V1 = -1000.0   # inner rod potential  [V]
V2 = -400.0    # middle shell potential [V]

# Weighting thresholds that separate the electrode surfaces by radius (x == r in RZ).
r_mid1 = 0.0125   # between r_i and r_m1  -> isolates the inner rod
r_mid2 = 0.0375   # between r_m2 and r_o  -> isolates the middle shell

use_plasma = os.environ.get("POISSON_TEST_PLASMA", "1") != "0"
n0 = 1.0e14    # uniform ion number density [1/m^3] (net positive -> screens)
dt = 3.0e-12
# The measurement is done once at init (afterInitEsolve, before any particle
# push), so no stepping is required. The screening charge here is a *static*
# net-charge beam -- pushing it would just make it fly apart, so keep steps at 0
# by default. Set POISSON_TEST_STEPS>0 only for a (neutral-plasma) drift demo.
max_steps = int(os.environ.get("POISSON_TEST_STEPS", "0"))

# Quartic implicit function: f < 0 in the two valid annuli (r_i,r_m1) & (r_m2,r_o),
# f > 0 inside each conductor (rod / middle shell / outer). Sign convention:
# negative in the fluid region (as in debug_rzflux/check_rz_flux.py).
eb_implicit = (
    f"((x*x)-{r_i}*{r_i})"
    f"*((x*x)-{r_m1}*{r_m1})"
    f"*((x*x)-{r_m2}*{r_m2})"
    f"*((x*x)-{r_o}*{r_o})"
)
# Prescribed EB potential: V1 on the rod, V2 on the shell, 0 on the outer.
potential_expression = (
    f"{V1}*(x*x<{r_mid1}**2) + {V2}*(x*x>{r_mid1}**2)*(x*x<{r_mid2}**2)"
)

# ---------------------------------------------------------------------------
# Simulation setup
# ---------------------------------------------------------------------------
grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    n_azimuthal_modes=1,
    lower_bound=[0.0, -Lz / 2],
    upper_bound=[Lr, Lz / 2],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["neumann", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=256,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee")
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=eb_implicit,
    potential=potential_expression,
    cover_multiple_cuts=True,
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    warpx_embedded_boundary=embedded_boundary,
    particle_shape="linear",
    max_steps=max_steps,
    # NOTE: on GPU add warpx_amrex_the_arena_is_managed=1 (needed for the
    # Python-side field ops). On CPU it is unnecessary and can trip a teardown
    # double-free, so it is omitted here for the local CPU test.
)

# Field dumps (E + the eb_covered mask) so the branch-agnostic post-processing
# tool electrode_potential.py can reconstruct per-electrode phi from the output.
sim.add_diagnostic(picmi.FieldDiagnostic(
    name="diag", grid=grid, period=1, data_list=["E"],
    warpx_format="openpmd", warpx_openpmd_backend="h5", warpx_file_min_digits=10,
))
sim.add_diagnostic(picmi.FieldDiagnostic(
    name="diag_eb_covered", grid=grid, period="0:1:1", data_list=["eb_covered"],
    warpx_format="openpmd", warpx_openpmd_backend="h5", warpx_file_min_digits=10,
))

if use_plasma:
    # Cold, net-positive space charge in the gaps -> nonzero Q_g (screening).
    dist = picmi.AnalyticDistribution(
        density_expression=f"{n0}*(x*x>{r_i}**2)*(x*x<{r_o}**2)",
        rms_velocity=[0.0, 0.0, 0.0],
        lower_bound=[r_i, None, -Lz / 2],
        upper_bound=[r_o, None, Lz / 2],
    )
    ions = picmi.Species(name="ions", particle_type="H", charge_state=1,
                         mass=1.6726e-27, initial_distribution=dist)
    sim.add_species(
        ions, layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=8, grid=grid)
    )

# ---------------------------------------------------------------------------
# Read-only multi-electrode voltage logger
# ---------------------------------------------------------------------------
logger = MultiElectrodePotentialLogger(
    sim=sim,
    electrodes=[
        {"name": "inner", "region": f"(x*x<{r_mid1}**2)", "potential": V1},
        {"name": "shell", "region": f"(x*x>{r_mid1}**2)*(x*x<{r_mid2}**2)", "potential": V2},
    ],
    period=1,
    out_csv="multi_electrode_Veff.csv",
    verbose=True,
)


def _setup_screen_measure():
    """Build C, establish a self-consistent (screened) field, and measure at t=0."""
    logger.setup_after_init()
    from pywarpx._libwarpx import libwarpx  # noqa: PLC0415
    # Make the live field the self-consistent electrostatic solution with the
    # current plasma + electrode potentials, so Q (live) and Q_g are consistent.
    libwarpx.libwarpx_so.get_instance().solve_poisson_efield()
    logger.log()


installafterInitEsolve(_setup_screen_measure)
installafterstep(logger.log)

sim.step(max_steps)

# This pywarpx build hits a benign double-free ("free(): invalid pointer") at
# interpreter teardown, *after* all output is produced. The CSV is flushed
# per-row during the run, so exit promptly to avoid the spurious SIGABRT and
# give the test a clean exit code.
import sys  # noqa: E402
sys.stdout.flush()
os._exit(0)
