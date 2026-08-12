#!/usr/bin/env python3
"""T1 -- static image-charge oracle for the electrode-charge diagnostics.

Rung 1 of the absorption test ladder in
`electrode-potential-maintenance/reference/absorption_potential_maintenance.md`.

WHY THIS TEST EXISTS
--------------------
Every charge diagnostic in this project has so far been scored against
*another numerical result* (grounded Poisson re-solve vs. Shockley-Ramo
reciprocity), which can only ever establish that two paths disagree -- never
which one is right. This fixture has a CLOSED-FORM ANSWER.

A point charge q at distance d from the centre of a GROUNDED sphere of radius
R induces a total surface charge

    Q_ind = -q * R / d                                              (exact)

by the classical image construction (image -qR/d at radius R^2/d). The
weighting potential of the isolated sphere is psi(r) = R/r, so
Shockley-Ramo predicts Q_g = -q*psi(d) = -q*R/d -- the SAME expression.
Both are verified symbolically in `reviews/verify_absorption.py` (D1-D3):
the surface integral of -eps0 dphi/dr over the sphere equals -qR/d exactly.

So this run scores WarpX's `ChargeOnEB` against an analytic target at
machine precision. No plasma, no noise, no absorption, no time evolution --
one static macroparticle, one grounded sphere, one number to check, swept
over d/R.

WHAT IT DISCRIMINATES
---------------------
- If `ChargeOnEB` tracks -qR/d, the EB flux integral is sound and the
  section 5 reciprocity discrepancy lives in the psi_k reconstruction.
- If it does NOT, the discrepancy is in the EB flux path, and the
  reconstruction is exonerated.
Either outcome is decisive, which the existing two-numerical-paths
comparison cannot be.

The finite domain is the one systematic: the exact image result assumes an
isolated sphere in free space, while the box walls are grounded Dirichlet at
a finite distance. That correction is O(R/L_wall) and is REPORTED, not
ignored -- see the wall-correction column. Run with a larger `--box_factor`
to push it down and confirm the residual converges toward zero.

USAGE (environment requirements identical to the v5 campaign; the
LD_LIBRARY_PATH/PYTHONPATH lines work around the stale-pyamrex-RPATH local
environment defect documented in inputs_3d_ect_bias_only_clampwork_v4.py):

    export LD_LIBRARY_PATH=/home/mgarten/src/warpx/build/lib:$LD_LIBRARY_PATH
    export UCX_TLS=tcp,self
    export PYTHONPATH=/home/mgarten/src/warpx/build/lib/site-packages
    python inputs_3d_t1_image_charge_oracle.py [--d_over_R 1.5] [--nx 64] \
           [--box_factor 4.0] [--out t1_results.json]
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.constants import c as c_light, epsilon_0, m_e

import preflight  # noqa: E402  (must precede simulation construction)
from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--d_over_R", type=float, default=1.5,
               help="charge distance from sphere CENTRE, in units of R")
p.add_argument("--nx", type=int, default=64, help="cells per side")
p.add_argument("--box_factor", type=float, default=4.0,
               help="half-domain / R; larger = closer to the isolated-sphere limit")
p.add_argument("--R", type=float, default=1.5e-2, help="sphere radius (m)")
p.add_argument("--q_test", type=float, default=1.0e-9, help="test charge (C)")
p.add_argument("--out", type=str, default="t1_results.json")
args, _ = p.parse_known_args()

preflight.require(bindings=("compute_eb_charge",))

R = args.R
d = args.d_over_R * R
q_test = args.q_test
half = args.box_factor * R

bf = 8
nx = ((args.nx + bf - 1) // bf) * bf
ny = nz = nx
dx = 2 * half / nx

# Exact isolated-sphere image-charge prediction
Q_exact = -q_test * R / d
# Rough scale of the grounded-wall correction: the walls are at distance
# ~half, and their leading effect on the induced charge is O(R/half).
wall_scale = R / half

print("=" * 88)
print("T1 -- STATIC IMAGE-CHARGE ORACLE")
print("=" * 88)
print(f"Sphere R          = {R*1e2:.3f} cm, GROUNDED (0 V)")
print(f"Test charge q     = {q_test:.4e} C at d = {d*1e2:.3f} cm "
      f"(d/R = {args.d_over_R:.3f}, gap = {(d-R)*1e3:.3f} mm)")
print(f"Domain            = {2*half*1e2:.2f} cm cube, {nx}^3 cells, "
      f"dx = {dx*1e3:.4f} mm, R/dx = {R/dx:.2f}")
print(f"gap/dx            = {(d-R)/dx:.3f}  (charge must sit OUTSIDE the EB)")
print(f"ANALYTIC TARGET   Q_ind = -qR/d = {Q_exact:.10e} C")
print(f"expected O(R/L) wall correction ~ {wall_scale:.3f} "
      f"(use --box_factor to reduce)")
print("=" * 88)

if d - R < 1.5 * dx:
    print(f"WARNING: gap ({(d-R)/dx:.2f} dx) is under 1.5 cells; the deposited "
          f"charge cloud will overlap the EB and the comparison is not clean.")

# ---------------------------------------------------------------------------
# Grid, solver, EB
# ---------------------------------------------------------------------------
grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[-half, -half, -half],
    upper_bound=[half, half, half],
    lower_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    warpx_blocking_factor=bf,
    warpx_max_grid_size=1024,
)

# Electrostatic solve: this is a STATIC test, so use the ES solver directly.
# That also removes the Yee/ECT time-advance from the comparison entirely --
# we are scoring the Poisson solve + EB flux integral, nothing else.
solver = picmi.ElectrostaticSolver(
    grid=grid, method="Multigrid", required_precision=1e-12,
)

# ONE grounded sphere at the origin.
eb_implicit = f"{R*R}-(x*x+y*y+z*z)"
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=eb_implicit,
    potential="0.0",
    cover_multiple_cuts=True,
)

# ---------------------------------------------------------------------------
# A single static macroparticle on the +x axis at x = d
# ---------------------------------------------------------------------------
# Mass is irrelevant (nothing moves; max_steps=0), charge carries the physics.
test_species = picmi.Species(
    name="testq",
    charge=q_test,
    mass=m_e,
    initial_distribution=picmi.ParticleListDistribution(
        x=[d], y=[0.0], z=[0.0], ux=[0.0], uy=[0.0], uz=[0.0], weight=[1.0],
    ),
)

sim = picmi.Simulation(
    solver=solver,
    warpx_embedded_boundary=embedded_boundary,
    particle_shape="linear",
    max_steps=0,               # static: initialise, solve, measure, stop
    # The electrostatic solver has no CFL condition, so WarpX requires an
    # explicit dt. Nothing is ever advanced here (max_steps=0), so the value
    # only has to exist; it never enters the measurement.
    time_step_size=1.0e-12,
    verbose=1,
)
sim.add_species(test_species, layout=None)

Q_eb_diag = picmi.ReducedDiagnostic(
    diag_type="ChargeOnEB", name="Q_eb_total", period=1,
)
sim.add_diagnostic(Q_eb_diag)

RESULT = {}


def _read_chargeoneb():
    """Locate and read the ChargeOnEB reduced-diagnostic file.

    Written by WarpX when the diagnostic fires; not available at
    afterInitEsolve time, so this is called AFTER sim.step() returns.
    Column layout: [step, time, value].
    """
    import glob
    cands = ["diags/reducedfiles/Q_eb_total.txt", "Q_eb_total.txt"]
    cands += glob.glob("**/Q_eb_total.txt", recursive=True)
    for path in cands:
        if os.path.exists(path):
            data = np.loadtxt(path, skiprows=1, ndmin=2)
            if data.size:
                return float(data[0, 2]), path
    return None, None


def measure():
    """Score ChargeOnEB against the analytic image-charge target."""
    Q_meas, path = _read_chargeoneb()
    if Q_meas is None:
        print("ERROR: Q_eb_total.txt not found or empty; cannot score this run.")
        return

    err_abs = Q_meas - Q_exact
    err_rel = err_abs / abs(Q_exact)

    RESULT.update(
        d_over_R=args.d_over_R, nx=nx, box_factor=args.box_factor,
        R=R, d=d, q_test=q_test, dx=dx, gap_over_dx=(d - R) / dx,
        Q_exact=Q_exact, Q_measured=Q_meas,
        abs_error=err_abs, rel_error=err_rel,
        wall_scale=wall_scale, source_file=path,
    )

    print("\n" + "=" * 88)
    print("T1 RESULT")
    print("=" * 88)
    print(f"  analytic  Q_ind = -qR/d        = {Q_exact:.12e} C")
    print(f"  measured  ChargeOnEB           = {Q_meas:.12e} C")
    print(f"  absolute error                 = {err_abs:.6e} C")
    print(f"  RELATIVE ERROR                 = {err_rel*100:+.4f} %")
    print(f"  (expected O(R/L_wall) offset   ~ {wall_scale*100:.1f} % from the "
          f"finite grounded box; sweep --box_factor to separate this from a "
          f"genuine diagnostic error)")
    print("=" * 88)


sim.step(0)
measure()

if RESULT:
    prev = []
    if os.path.exists(args.out):
        try:
            prev = json.load(open(args.out))
        except Exception:
            prev = []
    prev.append(RESULT)
    json.dump(prev, open(args.out, "w"), indent=1)
    print(f"appended result to {args.out}")
