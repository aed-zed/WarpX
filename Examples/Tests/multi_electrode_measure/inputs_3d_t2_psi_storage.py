#!/usr/bin/env python3
"""T2 -- does storing psi_k directly fix the Q_g reciprocity discrepancy?

Rung 2 of the ladder in
`electrode-potential-maintenance/reference/absorption_potential_maintenance.md`,
and the test that settles formal_specification.md section 5's unexplained
~1.5-1.6x factor between the two Q_g paths.

WHAT CHANGED
------------
`MultiElectrodeBiasCorrector` now stores the nodal scalar psi_k
(`psi_unit_k`) at setup, captured from the SAME unit-electrode Laplace solves
that already produce `Efield_unit_k`. `measure_grounded_charge_reciprocity()`
reads it directly instead of reconstructing it by
`psi_k[1:] = -cumsum(Ex0)*dx`. Requires a WarpX that publishes phi_fp from
SolvePoissonEfield; otherwise the corrector falls back to the legacy path and
reports so.

WHAT THIS DISCRIMINATES
-----------------------
Three named hypotheses for the 1.5-1.6x factor, each with a distinct signature
in this run:

  H1 line-integral path-dependence / cut-cell accumulation
     -> stored-psi ratio collapses toward 1. The reconstruction was the bug.
  H2 half-cell psi/rho collocation
     -> ALSO collapses (the stored psi is nodal, co-located with rho by
        construction), so H1 and H2 are NOT separated by this run alone --
        use --emit_legacy to get both numbers from the same state and compare
        their spatial structure.
  H3 operator non-symmetry (the adjoint-Psi finding, section 3b)
     -> ratio does NOT collapse. Both paths use the PLAIN psi, which is the
        wrong weighting potential on a non-symmetric operator regardless of
        how it is obtained. A residual factor surviving this test points at
        H3 and motivates the adjoint solve.

So: ratio -> 1 means the reconstruction was the whole story; a stubborn
residual localises the remainder to the operator, which is exactly the
open question section 3b raises.

The grounded Poisson re-solve (`_measure_grounded_plasma_charge`) is the
reference. It is not an analytic oracle -- see T1
(`inputs_3d_t1_image_charge_oracle.py`) for that -- but it is the quantity
section 5 compares against, so this reproduces that comparison exactly.

USAGE (same environment requirements as the v5 campaign):
    export LD_LIBRARY_PATH=/home/mgarten/src/warpx/build/lib:$LD_LIBRARY_PATH
    export UCX_TLS=tcp,self
    export PYTHONPATH=/home/mgarten/src/warpx/build/lib/site-packages
    python inputs_3d_t2_psi_storage.py [steps] [--nx 32] [--emit_legacy]
"""

import argparse
import json
import sys

import numpy as np
from scipy.constants import c as c_light, m_e

import preflight  # noqa: E402  (must precede simulation construction)
from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve, installafterstep

p = argparse.ArgumentParser()
p.add_argument("steps", nargs="?", type=int, default=60)
p.add_argument("--nx", type=int, default=32)
p.add_argument("--measure_every", type=int, default=10)
p.add_argument("--emit_legacy", action="store_true",
               help="also compute the legacy line-integral Q_g each time, "
                    "from the same state, for a paired comparison")
p.add_argument("--out", type=str, default="t2_psi_storage.json")
args, _ = p.parse_known_args()

preflight.require(bindings=("compute_eb_charge", "solve_poisson_efield", "set_potential_on_eb"))

# --- fixture: the two-sphere geometry of the v5 campaign -------------------
cells_per_R = args.nx / 8.0
L = 12e-2
R = 1.5e-2
center_offset = 3e-2
V_left, V_right = +300.0, -700.0

bf = 8
nx = ((args.nx + bf - 1) // bf) * bf
dx = L / nx
ny = nz = nx
half = nx * dx / 2

n_beam = 3e14
beam_hw = 1.0e-2
vz_drift = 0.1 * c_light
cfl = 0.9

print("=" * 88)
print("T2 -- STORED psi_k vs LINE-INTEGRAL RECONSTRUCTION")
print("=" * 88)
print(f"Domain {2*half*1e2:.2f} cm cube, {nx}^3, dx = {dx*1e3:.3f} mm, "
      f"R/dx = {R/dx:.2f}")
print(f"{args.steps} steps, measuring every {args.measure_every}")
print("=" * 88)

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[-half, -half, -half], upper_bound=[half, half, half],
    lower_boundary_conditions=["dirichlet"] * 3,
    upper_boundary_conditions=["dirichlet"] * 3,
    lower_boundary_conditions_particles=["absorbing"] * 3,
    upper_boundary_conditions_particles=["absorbing"] * 3,
    warpx_blocking_factor=bf, warpx_max_grid_size=1024,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="ECT", cfl=cfl)

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

beam = picmi.Species(
    name="beam", particle_type="electron",
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=f"{n_beam}", directed_velocity=[0, 0, vz_drift],
        lower_bound=[-beam_hw, -beam_hw, -half],
        upper_bound=[beam_hw, beam_hw, -half + 2 * half / 8],
    ),
)

sim = picmi.Simulation(solver=solver, warpx_embedded_boundary=embedded_boundary,
                       particle_shape="linear", max_steps=args.steps, verbose=0)
sim.add_species(beam, layout=picmi.PseudoRandomLayout(n_macroparticles_per_cell=8,
                                                      grid=grid))

sys.path.insert(0, "/home/mgarten/src/warpx/build/lib/site-packages")
from pywarpx.multi_electrode_corrector import (  # noqa: E402
    MultiElectrodeBiasCorrector,
)

corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    # This test MEASURES; it must not also correct, or the two Q_g paths would
    # be compared on states the correction itself has altered between calls.
    correction_interval=999999,
    electrodes=[{"name": "left", "region": "(x<0)", "potential": V_left},
                {"name": "right", "region": "(x>0)", "potential": V_right}],
    enable_gauss_clean=False,
    verbose=True,
)
installafterInitEsolve(corrector.setup_after_init)

REC = []


def _legacy_qg(corr):
    """The pre-fix line-integral Q_g, computed from the CURRENT state so the
    two paths are compared on identical data (paired, not sequential)."""
    warpx = corr._warpx()
    mfr = corr._mfr()
    lev = 0
    mpc = warpx.multi_particle_container()
    rho = np.asarray(mpc.get_charge_density(lev, False)[:, :, :])
    g = warpx.Geom(lev=lev).data()
    ddx = g.CellSize()[0]
    dV = ddx * g.CellSize()[1] * g.CellSize()[2]
    out = np.empty(corr.n)
    for k in range(corr.n):
        Ex0 = np.asarray(
            mfr.get(corr._unit_names[k], dir=corr._Direction(0), level=lev)[:, :, :]
        )
        psi = np.zeros_like(rho)
        psi[1:, :, :] = -np.cumsum(Ex0, axis=0) * ddx
        out[k] = -np.sum(rho * psi) * dV
    return out


def _grounded_qg(corr):
    """The reference Q_g: a fresh all-electrodes-grounded Poisson solve.

    Replicates the sequence inside measure_voltages() exactly (save field ->
    ground the EB -> solve -> read charge -> restore), so this is the same
    quantity section 5 compares the reciprocity path against.
    """
    warpx = corr._warpx()
    lev = 0
    saved = corr._save_efield(lev)
    warpx.set_potential_on_eb("0.0")
    warpx.solve_poisson_efield()
    q_g = np.array([warpx.compute_eb_charge(weighting=r, field="Efield_fp")
                    for r in corr.regions])
    corr._restore_efield(saved, lev)
    warpx.set_potential_on_eb(corr.potential_expression)
    return q_g


def measure():
    step = corrector._warpx().getistep(0)
    if step % args.measure_every or step == 0:
        return
    q_ref = _grounded_qg(corrector)
    q_new = np.asarray(corrector.measure_grounded_charge_reciprocity())
    row = {"step": int(step),
           "psi_stored": bool(corrector._psi_stored),
           "q_grounded": q_ref.tolist(),
           "q_reciprocity": q_new.tolist(),
           "ratio": (q_new / np.where(np.abs(q_ref) > 0, q_ref, np.nan)).tolist()}
    if args.emit_legacy:
        q_old = _legacy_qg(corrector)
        row["q_legacy_lineint"] = q_old.tolist()
        row["ratio_legacy"] = (
            q_old / np.where(np.abs(q_ref) > 0, q_ref, np.nan)).tolist()
    REC.append(row)
    r = row["ratio"]
    extra = (f"   legacy {['%.4f' % v for v in row['ratio_legacy']]}"
             if args.emit_legacy else "")
    print(f"  step {step:5d}  Q_grounded={q_ref}  ratio(stored psi)="
          f"{['%.4f' % v for v in r]}{extra}")


installafterstep(measure)
sim.step(args.steps)

if REC:
    ratios = np.array([r["ratio"] for r in REC], dtype=float).ravel()
    ratios = ratios[np.isfinite(ratios)]
    print("\n" + "=" * 88)
    print("T2 RESULT")
    print("=" * 88)
    print(f"  psi stored directly: {REC[0]['psi_stored']}")
    print(f"  stored-psi ratio (reciprocity / grounded): "
          f"mean {ratios.mean():.4f}, range [{ratios.min():.4f}, "
          f"{ratios.max():.4f}]  over {len(ratios)} samples")
    if args.emit_legacy:
        lr = np.array([r["ratio_legacy"] for r in REC], dtype=float).ravel()
        lr = lr[np.isfinite(lr)]
        print(f"  legacy line-integral ratio: mean {lr.mean():.4f}, "
              f"range [{lr.min():.4f}, {lr.max():.4f}]")
        print(f"  spec section 5 reported 1.632-1.647 for the legacy path")
    print("  interpretation: ratio -> 1 means the reconstruction was the "
          "defect (H1/H2); a stubborn residual points at operator "
          "non-symmetry (H3, section 3b) and motivates the adjoint psi.")
    print("=" * 88)
    json.dump(REC, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
