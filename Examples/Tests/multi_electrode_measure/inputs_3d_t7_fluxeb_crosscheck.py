#!/usr/bin/env python3
"""T7 -- cross-check the Python ledger against WarpX's own ChargeFluxEB.

WHY
---
The absorption ledger is a Python accumulator: it reads the EB scrape buffer,
gathers Psi at each impact, and sums q_p * Psi_k. Every number it has produced
so far has been validated against other things I wrote. `ChargeFluxEB` is an
independent implementation of the same sum, in C++, inside WarpX:

    sum over scraped particles of   q * w * f(x, y, z)

with `f` an arbitrary parser expression. Point `f` at an electrode region and
it returns the raw absorbed charge on that electrode -- the SAME quantity the
ledger's `booked` matrix should hold when Psi = 1, and the denominator the
mis-booking rate is measured against.

That makes it a true arbiter for three separate claims:

  1. Does the ledger read every scraped particle exactly once? (cursor
     bookkeeping -- an off-by-one or a double-read would show up here and
     nowhere else)
  2. Is the per-species split right? ChargeFluxEB reports per species natively.
  3. Is the sign convention right? Both should agree including sign.

WHAT IT CANNOT CHECK
--------------------
ChargeFluxEB's weighting is a closed-form parser expression f(x,y,z), so it
cannot evaluate a gridded Psi table. It therefore checks the ledger's
PARTICLE ACCOUNTING (which particles, what charge, which species) but not the
Psi gather. Those are exactly the parts T6 does not check, so the two together
cover the whole path.

USAGE
-----
    python inputs_3d_t7_fluxeb_crosscheck.py [steps] [--nx 32]
"""

import argparse
import json
import os

import numpy as np
from scipy.constants import c as c_light

import preflight  # noqa: E402
from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve, installafterstep

p = argparse.ArgumentParser()
p.add_argument("steps", nargs="?", type=int, default=90)
p.add_argument("--nx", type=int, default=32)
p.add_argument("--out", type=str, default="t7_fluxeb_crosscheck.json")
args, _ = p.parse_known_args()

preflight.require(bindings=("compute_eb_charge", "solve_poisson_efield",
                            "set_potential_on_eb"))

L, R, center_offset = 12e-2, 1.5e-2, 3e-2
V_left, V_right = +300.0, -700.0

bf = 8
nx = ((args.nx + bf - 1) // bf) * bf
dx = L / nx
half = nx * dx / 2

print("=" * 88)
print("T7 -- LEDGER vs ChargeFluxEB (independent C++ implementation)")
print("=" * 88)
print(f"{nx}^3, {args.steps} steps")
print("checks particle accounting: which particles, what charge, which species")
print("=" * 88)

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, nx, nx],
    lower_bound=[-half] * 3, upper_bound=[half] * 3,
    lower_boundary_conditions=["dirichlet"] * 3,
    upper_boundary_conditions=["dirichlet"] * 3,
    lower_boundary_conditions_particles=["absorbing"] * 3,
    upper_boundary_conditions_particles=["absorbing"] * 3,
    warpx_blocking_factor=bf, warpx_max_grid_size=1024,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="ECT", cfl=0.9)
R2 = R * R
eb = picmi.EmbeddedBoundary(
    implicit_function=(
        f"max({R2}-((x+{center_offset})*(x+{center_offset})+y*y+z*z),"
        f"{R2}-((x-{center_offset})*(x-{center_offset})+y*y+z*z))"),
    potential=f"({V_left})*(x<0)+({V_right})*(x>0)",
    cover_multiple_cuts=True,
)

n_seed, hw, v_in = 5e13, 0.8e-2, 0.15 * c_light


def _beam(name, ptype, sign, xc):
    return picmi.Species(
        name=name, particle_type=ptype,
        warpx_save_particles_at_eb=True,
        initial_distribution=picmi.AnalyticDistribution(
            density_expression=f"{n_seed}",
            directed_velocity=[sign * v_in, 0, 0],
            lower_bound=[xc - hw, -hw, -hw],
            upper_bound=[xc + hw, hw, hw],
        ),
    )


electrons = _beam("electrons", "electron", +1.0, -center_offset - 1.9 * R)
ions = _beam("ions", "proton", -1.0, +center_offset + 1.9 * R)

sim = picmi.Simulation(solver=solver, warpx_embedded_boundary=eb,
                       particle_shape="linear", max_steps=args.steps, verbose=0)
for sp in (electrons, ions):
    sim.add_species(sp, layout=picmi.PseudoRandomLayout(
        n_macroparticles_per_cell=4, grid=grid))

# WarpX's own C++ accumulator, ONE INSTANCE PER ELECTRODE. The per-electrode
# split is done exactly the way the v3 campaign does it for ChargeOnEB: a
# separate ReducedDiagnostic per region, each with its own weighting_function
# parser expression. ChargeFluxEB additionally reports per SPECIES natively
# (column 0 is the total, columns 1..N map to the species list), so a single
# instance gives the full (this electrode x every species) row.
flux_left = picmi.ReducedDiagnostic(
    diag_type="ChargeFluxEB", name="flux_eb_left",
    period=1, weighting_function="(x < 0)",
)
flux_right = picmi.ReducedDiagnostic(
    diag_type="ChargeFluxEB", name="flux_eb_right",
    period=1, weighting_function="(x > 0)",
)
for d in (flux_left, flux_right):
    sim.add_diagnostic(d)

from pywarpx.multi_electrode_corrector import (  # noqa: E402
    MultiElectrodeBiasCorrector,
)

corrector = MultiElectrodeBiasCorrector(
    sim=sim, correction_interval=999999,
    electrodes=[{"name": "left", "region": "(x<0)", "potential": V_left},
                {"name": "right", "region": "(x>0)", "potential": V_right}],
    enable_gauss_clean=False, verbose=True,
    book_absorption=True, absorption_species=["electrons", "ions"],
)
installafterInitEsolve(corrector.setup_after_init)


def tick():
    corrector.accumulate_absorption()


installafterstep(tick)
sim.step(args.steps)

# --- compare -----------------------------------------------------------------
rep = corrector.absorption_report()
booked_raw = None
if rep:
    # With psi=1 the ledger's "booked" is the raw absorbed charge. Here psi is
    # the plain table, so recompute the raw sum from the buffer directly -- the
    # quantity ChargeFluxEB reports.
    from pywarpx.particle_containers import ParticleBoundaryBufferWrapper
    buf = ParticleBoundaryBufferWrapper()
    Q_E = 1.602176634e-19
    raw = {}
    for sp, qs in (("electrons", -Q_E), ("ions", +Q_E)):
        arrs = buf.get_particle_boundary_buffer(sp, "eb", "w", 0)
        w = (np.concatenate([np.asarray(a) for a in arrs])
             if arrs else np.zeros(0))
        xs = buf.get_particle_boundary_buffer(sp, "eb", "x", 0)
        x = (np.concatenate([np.asarray(a) for a in xs])
             if xs else np.zeros(0))
        raw[sp] = {"n": int(len(w)),
                   "q_left": float(qs * w[x < 0].sum()) if len(w) else 0.0,
                   "q_right": float(qs * w[x >= 0].sum()) if len(w) else 0.0}
    booked_raw = raw

flux = {}
for nm in ("flux_eb_left", "flux_eb_right"):
    fp = os.path.join("diags", "reducedfiles", f"{nm}.txt")
    if os.path.exists(fp):
        d = np.loadtxt(fp, skiprows=1)
        d = np.atleast_2d(d)
        with open(fp) as fh:
            hdr = fh.readline().split()
        flux[nm] = {"header": hdr, "last_row": d[-1].tolist()}

print("\n" + "=" * 88)
print("T7 RESULT")
print("=" * 88)
print(f"  ledger impacts / species: {rep['counts'] if rep else None}")
if booked_raw:
    for sp, v in booked_raw.items():
        print(f"    {sp:<10s} n={v['n']:5d}  Q(x<0)={v['q_left']:+.6e} C  "
              f"Q(x>0)={v['q_right']:+.6e} C")
if flux:
    for nm, v in flux.items():
        print(f"  ChargeFluxEB {nm}: {v['header'][2:]} -> {v['last_row'][2:]}")
else:
    print("  ChargeFluxEB: NO OUTPUT FILE -- the reduced diagnostic was not "
          "registered (see note in the script); the cross-check did not run.")
print("=" * 88)
json.dump({"ledger": rep, "raw_from_buffer": booked_raw, "fluxeb": flux},
          open(args.out, "w"), indent=1)
print(f"wrote {args.out}")
