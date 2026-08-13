#!/usr/bin/env python3
"""T4 -- per-impact absorption bookkeeping with an (electrode x species) ledger.

Rung 4 of the ladder in
`electrode-potential-maintenance/reference/absorption_potential_maintenance.md`,
and the first RUN of ROADMAP item A.

WHAT IS BEING TESTED
--------------------
When a macroparticle is absorbed at x_a, the clamp's charge accounting
implicitly books its full charge q_p onto the struck electrode. The correct
amount is q_p * Psi_k(x_a) for every conductor k. The difference,

    deficit[k, s] = sum_p q_p (1 - Psi_k(x_p))     (struck electrode)
                    - sum_p q_p Psi_k(x_p)          (other conductors)

is a spurious source charge with an exact target of ZERO under correct booking.

Two claims are checked against the real code path:

  1. The deficit is LARGE without the correction. The measured weighting
     potential near the boundary is 0.18-0.26 (T3b), so ~75-85% of each
     absorbed macroparticle's charge is mis-booked -- not the ~11% the
     continuum formula s/(R+s) predicts.

  2. The species-resolved MATRIX retains what an aggregate hides. This fixture
     deliberately runs TWO opposite-sign species into the same electrodes,
     which is the Orbitron situation. The geometric factor 1 - Psi is always
     positive, but the injected charge carries the sign of q_p, so the species
     rows push in opposite directions. A scalar ledger can read near zero while
     hiding two large opposing errors.

WHY CHARGE, NOT ENERGY
----------------------
The acceptance metric is cumulative charge deficit vs impact count, with an
exact target of zero. Energy is the wrong instrument here: it is quadratic,
while this error is linear and one-signed per species.

USAGE
-----
    export LD_LIBRARY_PATH=/home/mgarten/src/warpx/build/lib:$LD_LIBRARY_PATH
    export UCX_TLS=tcp,self
    export PYTHONPATH=/home/mgarten/src/warpx/build/lib/site-packages:\
/home/mgarten/src/warpx/Examples/Tests/multi_electrode_measure
    python inputs_3d_t4_absorption_ledger.py [steps] [--nx 32]
"""

import argparse
import json

import numpy as np
from scipy.constants import c as c_light

import preflight  # noqa: E402  (must precede simulation construction)
from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve, installafterstep

p = argparse.ArgumentParser()
p.add_argument("steps", nargs="?", type=int, default=60)
p.add_argument("--nx", type=int, default=32)
p.add_argument("--report_every", type=int, default=10)
p.add_argument("--out", type=str, default="t4_absorption_ledger.json")
p.add_argument("--psi_table", type=str, default=None,
               help="path to the measured Psi .npz from T5; without it the "
                    "ledger gathers the plain Dirichlet basis and under-books "
                    "the correction by ~10x (see T4's own finding)")
args, _ = p.parse_known_args()

preflight.require(bindings=("compute_eb_charge", "solve_poisson_efield",
                            "set_potential_on_eb"))

# --- fixture: same geometry as T1/T2/T3b -----------------------------------
L = 12e-2
R = 1.5e-2
center_offset = 3e-2
V_left, V_right = +300.0, -700.0

bf = 8
nx = ((args.nx + bf - 1) // bf) * bf
dx = L / nx
half = nx * dx / 2

print("=" * 88)
print("T4 -- PER-IMPACT ABSORPTION LEDGER (electrode x species)")
print("=" * 88)
print(f"{nx}^3, dx = {dx*1e3:.3f} mm, R/dx = {R/dx:.2f}, {args.steps} steps")
print("two opposite-sign species into the same electrodes: the case where an")
print("aggregate ledger hides two large opposing errors.")
print("=" * 88)

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, nx, nx],
    lower_bound=[-half, -half, -half], upper_bound=[half, half, half],
    lower_boundary_conditions=["dirichlet"] * 3,
    upper_boundary_conditions=["dirichlet"] * 3,
    lower_boundary_conditions_particles=["absorbing"] * 3,
    upper_boundary_conditions_particles=["absorbing"] * 3,
    warpx_blocking_factor=bf, warpx_max_grid_size=1024,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="ECT", cfl=0.9)

R2 = R * R
eb_implicit = (
    f"max({R2}-((x+{center_offset})*(x+{center_offset})+y*y+z*z),"
    f"{R2}-((x-{center_offset})*(x-{center_offset})+y*y+z*z))"
)
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=eb_implicit,
    potential=f"({V_left})*(x<0)+({V_right})*(x>0)",
    cover_multiple_cuts=True,
)

# Two opposite-sign species aimed INWARD at the two spheres, so both electrodes
# are struck by both species -- the multi-species case the ledger exists for.
n_seed = 5e13
hw = 0.8e-2
v_in = 0.15 * c_light


def _beam(name, ptype, sign, xc):
    return picmi.Species(
        name=name, particle_type=ptype,
        # Must be set AT CONSTRUCTION: picmi pops this kwarg in __init__ and
        # passes it through at initialisation. Assigning the attribute after
        # the Species exists is silently too late -- the buffer stays empty and
        # the ledger reports zero impacts, which looks like "no absorption"
        # rather than "not recorded".
        warpx_save_particles_at_eb=True,
        initial_distribution=picmi.AnalyticDistribution(
            density_expression=f"{n_seed}",
            directed_velocity=[sign * v_in, 0, 0],
            lower_bound=[xc - hw, -hw, -hw],
            upper_bound=[xc + hw, hw, hw],
        ),
    )


# electrons launched from the left region moving right, ions from the right
# moving left: each crosses the gap and strikes an electrode
electrons = _beam("electrons", "electron", +1.0, -center_offset - 1.9 * R)
ions = _beam("ions", "proton", -1.0, +center_offset + 1.9 * R)

sim = picmi.Simulation(solver=solver, warpx_embedded_boundary=embedded_boundary,
                       particle_shape="linear", max_steps=args.steps, verbose=0)
for sp in (electrons, ions):
    sim.add_species(sp, layout=picmi.PseudoRandomLayout(
        n_macroparticles_per_cell=4, grid=grid))

from pywarpx.multi_electrode_corrector import (  # noqa: E402
    MultiElectrodeBiasCorrector,
)

corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    correction_interval=999999,          # measure only; do not perturb state
    electrodes=[{"name": "left", "region": "(x<0)", "potential": V_left},
                {"name": "right", "region": "(x>0)", "potential": V_right}],
    enable_gauss_clean=False,
    verbose=True,
    book_absorption=True,
    absorption_species=["electrons", "ions"],
    impact_histogram_cap=20000,
    psi_table=args.psi_table,
)
print(f"  weighting potential: "
      f"{'MEASURED table ' + args.psi_table if args.psi_table else 'stored plain psi_unit_k (under-books ~10x)'}")
installafterInitEsolve(corrector.setup_after_init)

REC = []


def tick():
    step = corrector._warpx().getistep(0)
    corrector.accumulate_absorption()
    if step % args.report_every:
        return
    rep = corrector.absorption_report()
    if rep is None or sum(rep["counts"]) == 0:
        return
    d = np.array(rep["deficit"])
    REC.append({"step": int(step), **rep})
    print(f"  step {step:4d}  impacts={rep['counts']}  "
          f"deficit[e-]={d[:,0]} C  deficit[ion]={d[:,1]} C")


installafterstep(tick)
sim.step(args.steps)

rep = corrector.absorption_report()
if rep and sum(rep["counts"]):
    d = np.array(rep["deficit"])
    b = np.array(rep["booked"])
    agg = d.sum(axis=1)
    print("\n" + "=" * 88)
    print("T4 RESULT")
    print("=" * 88)
    print(f"  impacts: {dict(zip(rep['species'], rep['counts']))}")
    for k, nm in enumerate(rep["electrodes"]):
        print(f"  electrode '{nm}':")
        for s, sp in enumerate(rep["species"]):
            print(f"      {sp:<10s} deficit = {d[k,s]:+.6e} C   "
                  f"booked = {b[k,s]:+.6e} C")
        hide = 100 * abs(agg[k]) / max(np.abs(d[k]).sum(), 1e-300)
        print(f"      species SUM = {agg[k]:+.6e} C   "
              f"({hide:.1f}% of the summed magnitude "
              f"{np.abs(d[k]).sum():.3e} C)")
        print(f"      -> an aggregate ledger would report the sum and hide "
              f"{100-hide:.1f}% of the error")
    print("=" * 88)
    json.dump({"records": REC, "final": rep}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
else:
    print("\nNO IMPACTS RECORDED -- check save_particles_at_eb and that the "
          "beams actually reach the spheres within the step count.")
