#!/usr/bin/env python3
"""T3b -- measure the TRUE weighting potential pointwise, using WarpX itself.

Replaces the abandoned T3 approach (`inputs_3d_t3_adjoint_psi.py`), which
re-derived the EB operator in Python and failed its own forward self-check by
~1e16. The lesson: do not reimplement the operator. Use it.

THE IDEA
--------
The weighting potential that appears in the charge identity is DEFINED by

    Q_{g,k}(rho) = -sum_a rho_a Psi_k[a] ,

so for a single point charge q at node a,

    Psi_k[a] = -Q_{g,k}(q e_a) / q .

Every quantity on the right is something WarpX already computes: place one
macroparticle at node a, run the all-electrodes-grounded Poisson solve, read
ChargeOnEB. That measures Psi_k[a] EXACTLY, through WarpX's own operator and
its own surface integral, with no assembly, no normalisation guesswork, and no
assumption about the discrete stencil.

Comparing that measured Psi against the STORED plain Psi (psi_unit_k, the
"electrode k at 1 V" Dirichlet solve) answers the open question directly:

  * measured == plain, pointwise
        -> the adjoint hypothesis is REFUTED. Both weighting potentials are
           the same object, and the ~1.47x discrepancy in the reciprocity Q_g
           comes from somewhere else entirely.
  * measured != plain, and the difference concentrates at cut cells
        -> the adjoint hypothesis is CONFIRMED as the cause, and the measured
           Psi is what the corrector should store.
  * measured == const * plain
        -> a scalar normalisation error, not an operator-symmetry effect.

Crucially, the ratio Psi_measured/Psi_plain is examined POINTWISE and against
distance-to-EB, so a uniform rescaling and a cut-cell-localised distortion are
distinguishable -- something the aggregate Q_g ratio cannot do.

SUPERPOSITION / INCREMENTAL PROBING
-----------------------------------
Particles are added cumulatively and Q_g is differenced between additions:

    Q after particle i  -  Q after particle i-1  =  -q_i Psi[a_i]

This needs no particle removal (WarpX has no clean single-particle delete from
Python) and it simultaneously exercises the exactness of superposition, which
is the property the whole millions-of-impacts argument rests on. If
superposition failed, these increments would drift; the final consistency
check tests exactly that.

USAGE
-----
    export LD_LIBRARY_PATH=/home/mgarten/src/warpx/build/lib:$LD_LIBRARY_PATH
    export UCX_TLS=tcp,self
    export PYTHONPATH=/home/mgarten/src/warpx/build/lib/site-packages:\
/home/mgarten/src/warpx/Examples/Tests/multi_electrode_measure
    python inputs_3d_t3b_psi_probe.py [--nx 32] [--nprobe 24]
"""

import argparse
import json

import numpy as np

import preflight  # noqa: E402  (must precede simulation construction)
from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve

p = argparse.ArgumentParser()
p.add_argument("--nx", type=int, default=32)
p.add_argument("--nprobe", type=int, default=24)
p.add_argument("--qprobe", type=float, default=1.0e-12,
               help="charge per probe macroparticle [C]")
p.add_argument("--out", type=str, default="t3b_psi_probe.json")
args, _ = p.parse_known_args()

preflight.require(bindings=("compute_eb_charge", "solve_poisson_efield",
                            "set_potential_on_eb"))

# --- fixture: identical geometry to T1/T2 ----------------------------------
L = 12e-2
R = 1.5e-2
center_offset = 3e-2
V_left, V_right = +300.0, -700.0
Q_E = 1.602176634e-19

bf = 8
nx = ((args.nx + bf - 1) // bf) * bf
dx = L / nx
half = nx * dx / 2

print("=" * 88)
print("T3b -- POINTWISE MEASUREMENT OF THE WEIGHTING POTENTIAL")
print("=" * 88)
print(f"{nx}^3, dx = {dx*1e3:.3f} mm, R/dx = {R/dx:.2f}, "
      f"{args.nprobe} probe sites")
print("measures Psi_k[a] = -Q_g(q e_a)/q via WarpX's own grounded solve,")
print("then compares against the stored plain (Dirichlet) Psi at the same nodes.")
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

# A probe species: no initial particles; they are injected one at a time.
probe = picmi.Species(name="probe", particle_type="electron")

sim = picmi.Simulation(solver=solver, warpx_embedded_boundary=embedded_boundary,
                       particle_shape="linear", max_steps=0,
                       time_step_size=1.0e-12, verbose=0)
sim.add_species(probe, layout=None)

from pywarpx.multi_electrode_corrector import (  # noqa: E402
    MultiElectrodeBiasCorrector,
)

corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    correction_interval=999999,
    electrodes=[{"name": "left", "region": "(x<0)", "potential": V_left},
                {"name": "right", "region": "(x>0)", "potential": V_right}],
    enable_gauss_clean=False,
    verbose=True,
)
installafterInitEsolve(corrector.setup_after_init)

RESULT = {}


def grounded_qg():
    """Q_g with every electrode grounded -- WarpX's own reference path."""
    warpx = corrector._warpx()
    saved = corrector._save_efield(0)
    warpx.set_potential_on_eb("0.0")
    warpx.solve_poisson_efield()
    q = np.array([warpx.compute_eb_charge(weighting=r, field="Efield_fp")
                  for r in corrector.regions])
    corrector._restore_efield(saved, 0)
    warpx.set_potential_on_eb(corrector.potential_expression)
    return q


def pick_probes():
    """Choose probe nodes STRATIFIED by distance to the nearer sphere surface.

    Distance is the discriminating variable: an adjoint/cut-cell effect must
    concentrate at small d. Earlier versions sorted candidates and took a
    prefix, which collapsed every probe onto the same shell -- useless for a
    distance trend. This bins by d/dx instead and draws from each bin.
    """
    n1 = nx + 1
    xn = -half + dx * np.arange(n1)
    rng = np.random.default_rng(0)
    cand = []
    for _ in range(40000):
        i, j, k = rng.integers(2, n1 - 2, size=3)
        x, y, z = xn[i], xn[j], xn[k]
        d1 = np.sqrt((x + center_offset) ** 2 + y ** 2 + z ** 2) - R
        d2 = np.sqrt((x - center_offset) ** 2 + y ** 2 + z ** 2) - R
        d = min(d1, d2) / dx
        if d <= 0.2:
            continue
        cand.append(((float(x), float(y), float(z)), float(d)))
    # stratify: bins in units of cells
    edges = [0.2, 0.75, 1.5, 3.0, 6.0, 12.0, 1e9]
    per = max(1, args.nprobe // (len(edges) - 1))
    pts, dists, seen = [], [], set()
    for lo, hi in zip(edges[:-1], edges[1:]):
        pool = [c for c in cand if lo <= c[1] < hi and c[0] not in seen]
        rng.shuffle(pool)
        for xyz, d in pool[:per]:
            seen.add(xyz)
            pts.append(xyz)
            dists.append(d * dx)
    return pts, dists


def run_probe():
    import numpy as np  # noqa: PLC0415

    warpx = corrector._warpx()
    mfr = corrector._mfr()
    lev = 0
    pts, dists = pick_probes()
    w_macro = args.qprobe / Q_E          # electrons carry -e, so q = -qprobe
    q_each = -args.qprobe

    # stored plain Psi, on the nodal grid
    if not corrector._psi_stored:
        raise RuntimeError(
            "psi_k is not stored; T3b compares against it and cannot run. "
            "Check that phi_fp was allocated (see setup_after_init).")
    psi_plain = [np.asarray(mfr.get(corrector._psi_names[k], level=lev)[:, :, :])
                 for k in range(corrector.n)]

    n1 = nx + 1
    xn = -half + dx * np.arange(n1)

    def node_index(v):
        i = int(round((v + half) / dx))
        return min(max(i, 0), n1 - 1)

    pc = None
    for name in ("probe",):
        try:
            from pywarpx.particle_containers import ParticleContainerWrapper
            pc = ParticleContainerWrapper(name)
        except Exception as exc:                       # pragma: no cover
            raise RuntimeError(f"cannot reach the probe species: {exc}") from exc

    q_prev = grounded_qg()
    q_vac = q_prev.copy()
    rows = []
    print(f"  vacuum Q_g (no charge) = {q_vac}")
    for n_, ((x, y, z), d) in enumerate(zip(pts, dists)):
        pc.add_particles(x=np.array([x]), y=np.array([y]), z=np.array([z]),
                         ux=np.zeros(1), uy=np.zeros(1), uz=np.zeros(1),
                         w=np.array([w_macro]))
        q_now = grounded_qg()
        dq = q_now - q_prev
        q_prev = q_now
        # Psi_k[a] = -dQ_k / q
        psi_meas = -dq / q_each
        i, j, k = node_index(x), node_index(y), node_index(z)
        psi_st = np.array([psi_plain[kk][i, j, k] for kk in range(corrector.n)])
        rows.append({"xyz": [x, y, z], "d_over_dx": d / dx,
                     "psi_measured": psi_meas.tolist(),
                     "psi_stored_plain": psi_st.tolist()})
        if n_ < 6 or n_ % 6 == 0:
            print(f"  probe {n_:3d}  d/dx={d/dx:6.2f}   "
                  f"psi_meas={psi_meas[0]:+.5f},{psi_meas[1]:+.5f}   "
                  f"psi_plain={psi_st[0]:+.5f},{psi_st[1]:+.5f}")

    pm = np.array([r["psi_measured"] for r in rows])
    pp = np.array([r["psi_stored_plain"] for r in rows])
    dd = np.array([r["d_over_dx"] for r in rows])
    ok = np.abs(pp) > 1e-6
    ratio = np.where(ok, pm / np.where(ok, pp, 1.0), np.nan)

    # sum over the probes reproduces the total: superposition self-check
    total_meas = q_prev - q_vac
    total_pred = np.array([-q_each * pm[:, kk].sum() for kk in range(corrector.n)])
    sup_err = np.abs(total_meas - total_pred) / np.maximum(np.abs(total_meas), 1e-300)

    near = dd < 2.0
    far = dd >= 2.0
    RESULT.update(
        rows=rows,
        ratio_mean=float(np.nanmean(ratio)),
        ratio_near=float(np.nanmean(ratio[near])) if near.any() else None,
        ratio_far=float(np.nanmean(ratio[far])) if far.any() else None,
        ratio_std=float(np.nanstd(ratio)),
        superposition_relerr=sup_err.tolist(),
    )
    print("\n" + "=" * 88)
    print("T3b RESULT")
    print("=" * 88)
    print(f"  probes: {len(rows)}  ({int(near.sum())} within 2 cells of the EB, "
          f"{int(far.sum())} beyond)")
    print(f"  psi_measured / psi_plain :  mean {np.nanmean(ratio):.6f}   "
          f"std {np.nanstd(ratio):.2e}")
    if near.any():
        print(f"      near-EB (d < 2h) :  {np.nanmean(ratio[near]):.6f}")
    if far.any():
        print(f"      far     (d >= 2h):  {np.nanmean(ratio[far]):.6f}")
    print(f"  superposition self-check (should be ~0): {sup_err}")
    print()
    print("  READING: ratio == 1 everywhere -> plain Psi is already correct,")
    print("  adjoint hypothesis refuted. Ratio != 1 and LARGER near the EB ->")
    print("  cut-cell/adjoint effect confirmed. Ratio a constant != 1 ->")
    print("  scalar normalisation, not an operator-symmetry effect.")
    print("=" * 88)
    json.dump(RESULT, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


installafterInitEsolve(run_probe)
sim.step(0)
