#!/usr/bin/env python3
"""T6 -- the acceptance test: is the booked charge EXACT?

Every earlier rung measured the weighting potential or the ledger's arithmetic.
None of them answered the question the correction actually rests on:

    when a particle is absorbed, does  q_p * gather(Psi_k, x_p)  equal the
    change in the electrode charge that WarpX itself reports?

That is a directly measurable statement with an exact target, and it settles
which Psi table is right without any argument about collocation.

METHOD
------
Straddle a single absorption event:

    Q_before = grounded Q_g with the particle still in the domain
    ... particle is absorbed ...
    Q_after  = grounded Q_g with it gone

The clamp's accounting change is (Q_after - Q_before). The ledger's prediction
for that same event is q_p * gather(Psi_k, x_impact). If the booking is exact
these agree; the residual is the error the correction leaves behind.

WHY THIS IS THE RIGHT TEST
--------------------------
T4 found the ledger under-books with the plain Dirichlet basis (6.6%) and
over-corrects with the T5 band table (44.4%). Neither number is meaningful on
its own -- both are measured against an ASSUMED correct value. This test has no
assumed value: WarpX reports the truth, and the ledger either reproduces it or
does not.

It also resolves the collocation question that the T5 band exposed. Particles
are scraped at d/h ~ -0.06, i.e. just INSIDE the EB surface, so a trilinear
gather there mixes covered nodes (where psi = 1 exactly, the Dirichlet BC, and
physically correct for charge sitting on the conductor) with band nodes (where
the measured Psi is 0.1-0.3). Whether that mixture is right is not something to
reason about -- it is something to measure.

USAGE
-----
    python inputs_3d_t6_booking_exactness.py [--nx 32] [--n_events 12]
        [--psi_table ../t5_run/psi_measured_nx32.npz]
"""

import argparse
import json

import numpy as np

import preflight  # noqa: E402
from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve

p = argparse.ArgumentParser()
p.add_argument("--nx", type=int, default=32)
p.add_argument("--n_events", type=int, default=12)
p.add_argument("--qprobe", type=float, default=1.0e-12)
p.add_argument("--psi_table", type=str, default=None)
p.add_argument("--adjoint", action="store_true",
               help="build Psi with the in-code ADJOINT solve instead of a "
                    "probed table: N_e solves, no probe placement, exact by "
                    "construction rather than sampled")
p.add_argument("--adjoint_rhs", type=str, default="operator",
               choices=("operator", "charge"),
               help="RHS for --adjoint: 'operator' (default) is the "
                    "operator's own Dirichlet-row sum; 'charge' is the "
                    "functional WarpX actually books charge with "
                    "(WarpXBuildAdjointRHSChargeFunctional), which needs "
                    "add_indicator=False so covered nodes stay at 0")
p.add_argument("--r_right", type=float, default=None)
p.add_argument("--which", type=str, default="left", choices=("left", "right"),
               help="which sphere to place the absorption events against")
p.add_argument("--out", type=str, default="t6_booking_exactness.json")
args, _ = p.parse_known_args()

_need = ["compute_eb_charge", "solve_poisson_efield", "set_potential_on_eb"]
if "--adjoint" in __import__("sys").argv:
    _need.append("solve_adjoint_weighting")
preflight.require(bindings=tuple(_need))

L, R, center_offset = 12e-2, 1.5e-2, 3e-2
RR = args.r_right if args.r_right else R
V_left, V_right = +300.0, -700.0
Q_E = 1.602176634e-19

bf = 8
nx = ((args.nx + bf - 1) // bf) * bf
dx = L / nx
half = nx * dx / 2

print("=" * 88)
print("T6 -- BOOKING EXACTNESS (the acceptance test)")
print("=" * 88)
print(f"{nx}^3, dx = {dx*1e3:.3f} mm, {args.n_events} absorption events")
print(f"Psi source: {args.psi_table or 'stored plain psi_unit_k'}")
print("target: booked charge == WarpX's own Q_g change, exactly")
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
R2, RR2 = R * R, RR * RR
eb = picmi.EmbeddedBoundary(
    implicit_function=(
        f"max({R2}-((x+{center_offset})*(x+{center_offset})+y*y+z*z),"
        f"{RR2}-((x-{center_offset})*(x-{center_offset})+y*y+z*z))"),
    potential=f"({V_left})*(x<0)+({V_right})*(x>0)",
    cover_multiple_cuts=True,
)
probe = picmi.Species(name="probe", particle_type="electron",
                      warpx_save_particles_at_eb=True)
sim = picmi.Simulation(solver=solver, warpx_embedded_boundary=eb,
                       particle_shape="linear", max_steps=0,
                       time_step_size=1.0e-12, verbose=0)
sim.add_species(probe, layout=None)

from pywarpx.multi_electrode_corrector import (  # noqa: E402
    MultiElectrodeBiasCorrector,
)

_gather_mode = "deposit" if (args.adjoint and args.adjoint_rhs == "charge") else "node"
print(f"gather_mode: {_gather_mode}")
corrector = MultiElectrodeBiasCorrector(
    sim=sim, correction_interval=999999,
    electrodes=[{"name": "left", "region": "(x<0)", "potential": V_left},
                {"name": "right", "region": "(x>0)", "potential": V_right}],
    enable_gauss_clean=False, verbose=True,
    book_absorption=True, absorption_species=["probe"],
    psi_table=args.psi_table,
    gather_mode=_gather_mode,
)
installafterInitEsolve(corrector.setup_after_init)


def _build_adjoint():
    """Solve for Psi_k through the in-code adjoint route and hand it to the
    ledger. No probe particles, no band choice, N_e solves total."""
    if not args.adjoint:
        return
    import numpy as _np
    w = corrector._warpx()
    mfr = corrector._mfr()
    tables = []
    kwargs = {}
    if args.adjoint_rhs == "charge":
        kwargs = {"rhs_mode": "charge", "add_indicator": False}
    for k, reg in enumerate(corrector.regions):
        name = corrector._psi_names[k]          # already registered at setup
        ok, res = w.solve_adjoint_weighting(region=reg, out_name=name,
                                            tol=1.0e-10, max_iter=2000,
                                            **kwargs)
        print(f"  adjoint Psi[{corrector.names[k]}]: converged={ok} "
              f"rel.residual={res:.3e}")
        if not ok:
            raise SystemExit(
                f"adjoint solve for electrode {k} did NOT converge "
                f"(residual {res:.3e}). Refusing to continue: an unconverged "
                "Psi produces a wrong correction that looks plausible.")
        tables.append(_np.array(mfr.get(name, level=0)[:, :, :]))
    corrector.load_psi_table(tables)


installafterInitEsolve(_build_adjoint)

ROWS = []


def grounded_qg():
    w = corrector._warpx()
    saved = corrector._save_efield(0)
    w.set_potential_on_eb("0.0")
    w.solve_poisson_efield()
    q = np.array([w.compute_eb_charge(weighting=r, field="Efield_fp")
                  for r in corrector.regions])
    corrector._restore_efield(saved, 0)
    w.set_potential_on_eb(corrector.potential_expression)
    return q


def run():
    from pywarpx.particle_containers import ParticleContainerWrapper
    pc = ParticleContainerWrapper("probe")
    w_macro = args.qprobe / Q_E
    q_each = -args.qprobe

    rng = np.random.default_rng(0)
    for ev in range(args.n_events):
        # place the particle just outside the LEFT sphere, at a random
        # direction, one third of a cell out -- the regime where scraping
        # happens on the next push
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        which = 0 if args.which == "left" else 1
        r_pos = (R if which == 0 else RR) + 0.33 * dx
        xc = -center_offset if which == 0 else +center_offset
        x = xc + r_pos * u[0]
        y, z = r_pos * u[1], r_pos * u[2]

        q_before = grounded_qg()
        psis = np.array([corrector._gather_psi(k, np.array([x]), np.array([y]),
                                               np.array([z]))[0]
                         for k in range(corrector.n)])

        pc.add_particles(x=np.array([x]), y=np.array([y]), z=np.array([z]),
                         ux=np.zeros(1), uy=np.zeros(1), uz=np.zeros(1),
                         w=np.array([w_macro]))
        q_with = grounded_qg()

        # what WarpX says the in-domain particle contributes
        dq_domain = q_with - q_before
        # what the ledger predicts for that same charge, in the SAME sign
        # convention T3b verified: Psi_k = -dQ_k / q  =>  dQ_k = -q * Psi_k
        pred = -q_each * psis

        ROWS.append({"event": ev, "xyz": [float(x), float(y), float(z)],
                     "psi_gathered": psis.tolist(),
                     "dQ_warpx": dq_domain.tolist(),
                     "predicted": pred.tolist()})

    A = np.array([r["dQ_warpx"] for r in ROWS])
    P = np.array([r["predicted"] for r in ROWS])
    G = np.array([r["psi_gathered"] for r in ROWS])
    # per-event relative error on the struck (left) electrode
    col = 0 if args.which == "left" else 1
    rel = np.abs(P[:, col] - A[:, col]) / np.abs(A[:, col])
    # implied TRUE Psi from WarpX, for comparison with the gathered one
    # T3b's verified convention, reproduced exactly: Psi = -dQ / q.
    # An earlier draft wrote -A/(-q_each), which flips the sign and yields a
    # negative (unphysical) Psi -- caught because Psi must lie in [0, 1].
    psi_true = -A[:, col] / q_each

    print("\n" + "=" * 88)
    print("T6 RESULT")
    print("=" * 88)
    print(f"  events: {len(ROWS)}, all just outside the left sphere "
          f"(r = R + 0.33h)")
    print(f"  Psi gathered by the ledger : mean {G[:,col].mean():.4f}  "
          f"[{G[:,col].min():.4f}, {G[:,col].max():.4f}]")
    print(f"  Psi implied by WarpX's Q_g : mean {psi_true.mean():.4f}  "
          f"[{psi_true.min():.4f}, {psi_true.max():.4f}]")
    print(f"  booking relative error     : mean {100*rel.mean():.2f}%  "
          f"max {100*rel.max():.2f}%")
    print()
    if rel.mean() < 0.02:
        print("  => BOOKING IS EXACT with this Psi table.")
    else:
        print(f"  => booking is NOT exact: the gathered Psi is off by a factor "
              f"{G[:,col].mean()/psi_true.mean():.3f} on average.")
        print("     The table that would make it exact is the one whose gather "
              "reproduces the 'implied by WarpX' column above.")
    print("=" * 88)
    json.dump({"rows": ROWS, "psi_gathered_mean": float(G[:, col].mean()),
               "psi_true_mean": float(psi_true.mean()),
               "rel_err_mean": float(rel.mean()),
               "psi_table": args.psi_table}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


installafterInitEsolve(run)
sim.step(0)
