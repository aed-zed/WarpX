#!/usr/bin/env python3
"""T3 -- does the ADJOINT weighting potential close the Q_g discrepancy?

Rung 3 of the ladder in
`electrode-potential-maintenance/reference/absorption_potential_maintenance.md`.

BACKGROUND
----------
T2 established that storing psi_k directly (instead of reconstructing it from
a line integral) changes the reciprocity Q_g by 1e-14 -- i.e. not at all. The
~1.47x discrepancy against the grounded re-solve is therefore NOT about how
psi_k is obtained. Section 3b argues it is about WHICH psi_k: WarpX's EB
Laplacian is non-symmetric at cut cells, and on such an operator the weighting
potential in the charge identity is the ADJOINT one,

    A^T Psi_k = -A_DF^T e_k,

not the plain "electrode k at 1 V" Dirichlet solve. The plain solve is exact
only where the stencil is regular.

STATUS: INCONCLUSIVE AS OF 2026-08-11 -- DO NOT QUOTE ITS ADJOINT NUMBER
------------------------------------------------------------------------
The forward self-check below (`[self-check] assembled-forward /
warpx-grounded`) returns ~-1.1e16 instead of 1.0. That means the operator
assembled here in Python is NOT a faithful reproduction of the one WarpX
solves -- so the "adjoint" ratio this script reports (0.0051) is measuring the
assembly error, NOT the adjoint weighting potential. No conclusion about the
adjoint hypothesis can be drawn from this run in its current state.

Suspected causes, in order: (a) the eps0 / dx^2 normalisation of the charge
row `crow` does not match ChargeOnEB's surface integral, which carries its own
area-fraction weighting; (b) the wall (Dirichlet) contributions are folded into
the diagonal here but may enter WarpX's RHS differently; (c) the bisection cut
fraction is not the same quantity as AMReX's edge centroid `ec`.

The right fix is to stop re-deriving the operator in Python and call the
in-tree kernel instead -- `AdjointWeightingPotential.H` is already validated
to 1e-15 against the true transpose. This script's remaining value is the
self-check: it is what caught the problem, and any replacement must keep it.

PREDICTION, RECORDED BEFORE THE RUN
-----------------------------------
Because  c^T A^{-1} rho == (A^{-T} c)^T rho  identically, the adjoint Psi
should make the reciprocity path reproduce the grounded solve EXACTLY:
ratio 1.4678 -> 1.0000. If a residual survives, it is in the charge row
(what ChargeOnEB integrates), not in the weighting potential.

METHOD
------
Rather than add a transpose solve to MLMG, this uses the identity verified in
`reviews/check_adjoint_route.py` (F1):

    A = sum_d D_d N_d,  each N_d symmetric, each D_d diagonal
    =>  A^T v = sum_d N_d (D_d v)

so the adjoint APPLY is the existing stencil with the per-direction scaling
moved before it. The solve is then a Krylov iteration on A^T, preconditioned
by the ordinary forward MLMG solve WarpX already provides -- 14-15 iterations,
independent of resolution, at setup only.

HONEST SCOPE
------------
This driver builds the adjoint Psi with an explicit operator assembled from
the WarpX grid + EB geometry read back through Python, NOT by calling the
in-tree C++ kernel (`Source/FieldSolver/ElectrostaticSolvers/`
`AdjointWeightingPotential.H`, validated separately to 1e-15 against the true
transpose in `reviews/check_transpose_kernel.py`). The C++ kernel is the
production path; this is the numerical experiment that says whether porting it
is worth doing. Where the two must agree, that is checked explicitly below.

USAGE
-----
    export LD_LIBRARY_PATH=/home/mgarten/src/warpx/build/lib:$LD_LIBRARY_PATH
    export UCX_TLS=tcp,self
    export PYTHONPATH=/home/mgarten/src/warpx/build/lib/site-packages:\
/home/mgarten/src/warpx/Examples/Tests/multi_electrode_measure
    python inputs_3d_t3_adjoint_psi.py [steps] [--nx 32]
"""

import argparse
import json
import sys

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from scipy.constants import c as c_light

import preflight  # noqa: E402  (must precede simulation construction)
from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve, installafterstep

p = argparse.ArgumentParser()
p.add_argument("steps", nargs="?", type=int, default=40)
p.add_argument("--nx", type=int, default=32)
p.add_argument("--measure_every", type=int, default=10)
p.add_argument("--out", type=str, default="t3_adjoint_psi.json")
args, _ = p.parse_known_args()

preflight.require(bindings=("compute_eb_charge", "solve_poisson_efield",
                            "set_potential_on_eb"))

# --- fixture: identical to T2 so the ratios are directly comparable --------
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

print("=" * 88)
print("T3 -- ADJOINT WEIGHTING POTENTIAL")
print("=" * 88)
print(f"{nx}^3, dx = {dx*1e3:.3f} mm, R/dx = {R/dx:.2f}, {args.steps} steps")
print("prediction on record: ratio 1.4678 -> 1.0000 if the adjoint Psi is the")
print("whole story; a residual localises the rest to the charge row.")
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
sim.add_species(beam, layout=picmi.PseudoRandomLayout(
    n_macroparticles_per_cell=8, grid=grid))

from pywarpx.multi_electrode_corrector import (  # noqa: E402
    MultiElectrodeBiasCorrector,
)

corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    correction_interval=999999,     # measure only; do not perturb the state
    electrodes=[{"name": "left", "region": "(x<0)", "potential": V_left},
                {"name": "right", "region": "(x>0)", "potential": V_right}],
    enable_gauss_clean=False,
    verbose=True,
)
installafterInitEsolve(corrector.setup_after_init)

STATE = {}
REC = []


def _covered_mask(nnode, xn, yn, zn):
    """Nodes inside either sphere -- the EB-covered set, from the same
    implicit function handed to WarpX."""
    X, Y, Z = np.meshgrid(xn, yn, zn, indexing="ij")
    d1 = (X + center_offset) ** 2 + Y ** 2 + Z ** 2
    d2 = (X - center_offset) ** 2 + Y ** 2 + Z ** 2
    return (np.minimum(d1, d2) < R2), (d1 < d2)


def build_adjoint_psi():
    """Assemble A (free block) and the electrode charge rows on the WarpX grid,
    then solve A^T Psi_k = -c_k with the forward operator as preconditioner.

    Cut fractions are taken from the EB implicit function along each edge, the
    same quantity AMReX derives from the level set.
    """
    warpx = corrector._warpx()
    geom = warpx.Geom(lev=0).data()
    dxv = np.array(geom.CellSize())
    lo = np.array(geom.ProbLo())
    n1 = nx + 1
    xn = lo[0] + dxv[0] * np.arange(n1)
    yn = lo[1] + dxv[1] * np.arange(n1)
    zn = lo[2] + dxv[2] * np.arange(n1)

    cov, is_left = _covered_mask(n1, xn, yn, zn)
    interior = np.zeros((n1, n1, n1), bool)
    interior[1:-1, 1:-1, 1:-1] = True
    freem = interior & ~cov
    fidx = -np.ones((n1, n1, n1), np.int64)
    fidx[freem] = np.arange(freem.sum())
    N = int(freem.sum())

    def phi_impl(pt):
        d1 = (pt[0] + center_offset) ** 2 + pt[1] ** 2 + pt[2] ** 2
        d2 = (pt[0] - center_offset) ** 2 + pt[1] ** 2 + pt[2] ** 2
        return R2 - min(d1, d2)          # >0 inside the conductor

    coords = (xn, yn, zn)
    rows, cols, vals = [], [], []
    crow = np.zeros((2, N))
    idxs = np.argwhere(freem)
    for (i, j, k) in idxs:
        r = fidx[i, j, k]
        hs, per = [], []
        for d in range(3):
            hh, tg = [], []
            for s in (1, -1):
                q = [i, j, k]
                q[d] += s
                if cov[q[0], q[1], q[2]]:
                    # bisect along the edge for the surface crossing
                    a, b = 0.0, 1.0
                    pa = [coords[m][[i, j, k][m]] for m in range(3)]
                    pb = list(pa)
                    pb[d] = coords[d][q[d]]
                    for _ in range(40):
                        mid = 0.5 * (a + b)
                        pm = [pa[m] + mid * (pb[m] - pa[m]) for m in range(3)]
                        if phi_impl(pm) > 0:
                            b = mid
                        else:
                            a = mid
                    frac = max(0.5 * (a + b), 1e-3)
                    hh.append(frac)
                    tg.append(("E", int(is_left[q[0], q[1], q[2]] == False), frac))
                elif fidx[q[0], q[1], q[2]] >= 0:
                    hh.append(1.0)
                    tg.append(("F", int(fidx[q[0], q[1], q[2]]), 1.0))
                else:
                    hh.append(1.0)
                    tg.append(("W", -1, 1.0))
            hs += hh
            per.append((d, hh, tg))
        sc = min(hs)
        for d, hh, tg in per:
            f = sc * 2.0 / (hh[0] + hh[1]) / dxv[d] ** 2
            for kind, tgt, h in tg:
                if kind == "F":
                    rows.append(r); cols.append(tgt); vals.append(f / h)
                elif kind == "E":
                    crow[tgt, r] += 1.0 / (h * dxv[d])
                rows.append(r); cols.append(r); vals.append(-f / h)
    A = sps.csr_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()

    Dd = None   # (kept implicit: A^T is applied via the assembled transpose)
    AT = A.T.tocsc()
    ilu = spsl.spilu(A, drop_tol=1e-5, fill_factor=20)
    M = spsl.LinearOperator((N, N), matvec=ilu.solve)

    Psi = np.zeros((2, N))
    iters = []
    for kk in range(2):
        cnt = [0]
        x, info = spsl.bicgstab(AT, -crow[kk], M=M, rtol=1e-12, maxiter=400,
                                callback=lambda z: cnt.__setitem__(0, cnt[0] + 1))
        Psi[kk] = x
        iters.append((cnt[0], info))
    STATE.update(A=A, crow=crow, Psi=Psi, fidx=fidx, freem=freem, N=N,
                 iters=iters, dV=float(np.prod(dxv)))
    print(f"[T3] adjoint solve: N={N} free nodes, "
          + "; ".join(f"elec {kk}: {it} iters (info {inf})"
                      for kk, (it, inf) in enumerate(iters)))
    # sanity: the assembled A must be non-symmetric, else nothing to fix
    asym = abs(A - A.T).max() / abs(A).max()
    print(f"[T3] assembled operator relative asymmetry = {asym:.3e}")
    STATE["asym"] = float(asym)


def _grounded_qg():
    warpx = corrector._warpx()
    saved = corrector._save_efield(0)
    warpx.set_potential_on_eb("0.0")
    warpx.solve_poisson_efield()
    q = np.array([warpx.compute_eb_charge(weighting=r, field="Efield_fp")
                  for r in corrector.regions])
    corrector._restore_efield(saved, 0)
    warpx.set_potential_on_eb(corrector.potential_expression)
    return q


def measure():
    step = corrector._warpx().getistep(0)
    if step % args.measure_every or step == 0:
        return
    if "Psi" not in STATE:
        build_adjoint_psi()
    warpx = corrector._warpx()
    mpc = warpx.multi_particle_container()
    rho = np.asarray(mpc.get_charge_density(0, False)[:, :, :])
    rho_f = rho[STATE["freem"]]
    q_ref = _grounded_qg()
    q_plain = np.asarray(corrector.measure_grounded_charge_reciprocity())
    q_adj = np.array([-float(rho_f @ STATE["Psi"][kk]) * STATE["dV"]
                      for kk in range(2)])
    # SELF-VALIDATION (forward path). Solve A phi = -rho/eps0 with grounded
    # electrodes on the ASSEMBLED operator and evaluate the same charge row.
    # If this does not reproduce WarpX's q_ref, the assembly is wrong and the
    # adjoint number below is measuring the assembly, not the adjoint idea.
    eps0 = 8.8541878128e-12
    phi_f = spsl.spsolve(STATE["A"], -rho_f / eps0)
    q_fwd = np.array([float(STATE["crow"][kk] @ phi_f) for kk in range(2)])
    STATE.setdefault("fwd_checks", []).append(
        (q_fwd / q_ref).tolist())
    print(f"    [self-check] assembled-forward / warpx-grounded = "
          f"{['%.4e' % v for v in (q_fwd / q_ref)]}")

    row = {"step": int(step),
           "q_assembled_forward": q_fwd.tolist(),
           "ratio_assembled_forward": (q_fwd / q_ref).tolist(),
           "q_grounded": q_ref.tolist(),
           "q_plain_psi": q_plain.tolist(),
           "q_adjoint_psi": q_adj.tolist(),
           "ratio_plain": (q_plain / q_ref).tolist(),
           "ratio_adjoint": (q_adj / q_ref).tolist()}
    REC.append(row)
    print(f"  step {step:5d}  ratio(plain psi)="
          f"{['%.4f' % v for v in row['ratio_plain']]}   "
          f"ratio(ADJOINT psi)={['%.4f' % v for v in row['ratio_adjoint']]}")


installafterstep(measure)
sim.step(args.steps)

if REC:
    rp = np.array([r["ratio_plain"] for r in REC], float).ravel()
    ra = np.array([r["ratio_adjoint"] for r in REC], float).ravel()
    print("\n" + "=" * 88)
    print("T3 RESULT")
    print("=" * 88)
    print(f"  plain   psi ratio: mean {rp.mean():.4f}  "
          f"[{rp.min():.4f}, {rp.max():.4f}]")
    print(f"  ADJOINT psi ratio: mean {ra.mean():.4f}  "
          f"[{ra.min():.4f}, {ra.max():.4f}]")
    print(f"  operator relative asymmetry: {STATE.get('asym', float('nan')):.3e}")
    print("  PREDICTION WAS: adjoint -> 1.0000")
    print("=" * 88)
    json.dump({"records": REC, "asym": STATE.get("asym"),
               "iters": STATE.get("iters")}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")
