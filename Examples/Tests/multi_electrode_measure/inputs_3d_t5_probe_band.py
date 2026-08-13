#!/usr/bin/env python3
"""T5 -- probe the TRUE weighting potential on the near-EB band and store it.

Produces the lookup table the absorption ledger needs. T4 showed the ledger
machinery works but gathers `psi_unit_k`, the stored plain (Dirichlet) basis,
which is ~1 at its own electrode BY BOUNDARY CONDITION and therefore reports a
mis-booking rate of 6.6% where the measured weighting potential implies ~80%.

METHOD (same as T3b, applied exhaustively instead of to 18 spot checks)
-----------------------------------------------------------------------
The charge identity DEFINES

    Psi_k[a] = -Q_g,k(q e_a) / q

so for each node a in the near-EB band: inject one macroparticle there, run the
all-electrodes-grounded solve, read ChargeOnEB, difference against the previous
total, divide. Particles are added cumulatively (WarpX has no clean
single-particle delete from Python) and Q_g is differenced between additions --
which also re-exercises superposition on the production path.

The value comes out through WarpX's OWN operator and OWN surface integral. No
assembly, no normalisation guesswork, no assumption about the stencil. That is
the whole point: the earlier attempt to derive Psi by reassembling the operator
in Python failed its own forward self-check by ~1e16.

SCOPE -- THIS IS NOT THE PRODUCTION MECHANISM
---------------------------------------------
One grounded solve per node. At nx=32 the band is ~800 nodes, ~15 s. Surface
nodes grow as (R/h)^2 and each solve gets more expensive, so at nx=512 this
would be thousands of hours. The production route is the adjoint solve
(N_e solves TOTAL, resolution-independent -- AdjointWeightingPotential.H,
validated to 1e-15 against the true transpose but not yet wired into a solve).

This fixture's lasting role is GROUND TRUTH: it is the only thing that can
validate a computed Psi, because it goes through the code rather than around
it. Keep it as the acceptance test for the adjoint path.

OUTPUT
------
`psi_measured_nx{NX}.npz` with one full nodal array per electrode: the plain
psi everywhere, overwritten with measured values on the band. Full arrays (not
a sparse band) so the ledger's trilinear gather works unchanged.

USAGE
-----
    export LD_LIBRARY_PATH=/home/mgarten/src/warpx/build/lib:$LD_LIBRARY_PATH
    export UCX_TLS=tcp,self
    export PYTHONPATH=/home/mgarten/src/warpx/build/lib/site-packages:\
/home/mgarten/src/warpx/Examples/Tests/multi_electrode_measure
    python inputs_3d_t5_probe_band.py [--nx 32] [--band 2.0]
"""

import argparse
import json
import time

import numpy as np

import preflight  # noqa: E402  (must precede simulation construction)
from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve

p = argparse.ArgumentParser()
p.add_argument("--nx", type=int, default=32)
p.add_argument("--band", type=float, default=2.0,
               help="probe nodes out to this many cells OUTSIDE the surface")
p.add_argument("--band_in", type=float, default=0.0,
               help="also probe this many cells INSIDE the surface. T6 found "
                    "every impact at d/h in [-0.088, -0.016], i.e. inside; a "
                    "table probed only outside leaves the gather dominated by "
                    "covered nodes where psi = 1 by Dirichlet BC. Set >= 1.0 "
                    "to cover the real scrape sites.")
p.add_argument("--qprobe", type=float, default=1.0e-12)
p.add_argument("--r_right", type=float, default=None,
               help="radius of the RIGHT sphere; defaults to the left one. "
                    "Unequal radii give the electrodes different cut "
                    "fractions, which is where the discrete capacitance "
                    "matrix was found to be 24%% non-symmetric -- the "
                    "two-sphere symmetric fixture is protected by a mirror "
                    "symmetry a production device does not have.")
p.add_argument("--max_nodes", type=int, default=4000,
               help="safety cap; the run aborts rather than silently truncating")
p.add_argument("--out", type=str, default=None)
args, _ = p.parse_known_args()

preflight.require(bindings=("compute_eb_charge", "solve_poisson_efield",
                            "set_potential_on_eb"))

L = 12e-2
R = 1.5e-2
RR = args.r_right if args.r_right else R      # right-sphere radius
center_offset = 3e-2
V_left, V_right = +300.0, -700.0
Q_E = 1.602176634e-19

bf = 8
nx = ((args.nx + bf - 1) // bf) * bf
dx = L / nx
half = nx * dx / 2
OUT = args.out or f"psi_measured_nx{nx}.npz"

print("=" * 88)
print("T5 -- PROBE THE WEIGHTING POTENTIAL ON THE NEAR-EB BAND")
print("=" * 88)
print(f"{nx}^3, dx = {dx*1e3:.3f} mm, band = {args.band} cells, "
      f"q_probe = {args.qprobe:.1e} C")
print(f"radii: left R = {R*1e2:.2f} cm (R/dx = {R/dx:.2f}), "
      f"right R = {RR*1e2:.2f} cm (R/dx = {RR/dx:.2f})"
      + ("  [UNEQUAL -- mismatched cut fractions]" if RR != R else "  [equal]"))
print("one grounded solve per node; this is ground truth, not the production")
print("mechanism (that is the adjoint solve -- N_e solves, resolution-free).")
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

R2, RR2 = R * R, RR * RR
eb_implicit = (
    f"max({R2}-((x+{center_offset})*(x+{center_offset})+y*y+z*z),"
    f"{RR2}-((x-{center_offset})*(x-{center_offset})+y*y+z*z))"
)
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=eb_implicit,
    potential=f"({V_left})*(x<0)+({V_right})*(x>0)",
    cover_multiple_cuts=True,
)

probe = picmi.Species(name="probe", particle_type="electron")

sim = picmi.Simulation(solver=solver, warpx_embedded_boundary=embedded_boundary,
                       particle_shape="linear", max_steps=0,
                       time_step_size=1.0e-12, verbose=0)
sim.add_species(probe, layout=None)

from pywarpx.multi_electrode_corrector import (  # noqa: E402
    MultiElectrodeBiasCorrector,
)

corrector = MultiElectrodeBiasCorrector(
    sim=sim, correction_interval=999999,
    electrodes=[{"name": "left", "region": "(x<0)", "potential": V_left},
                {"name": "right", "region": "(x>0)", "potential": V_right}],
    enable_gauss_clean=False, verbose=True,
)
installafterInitEsolve(corrector.setup_after_init)


def grounded_qg():
    warpx = corrector._warpx()
    saved = corrector._save_efield(0)
    warpx.set_potential_on_eb("0.0")
    warpx.solve_poisson_efield()
    q = np.array([warpx.compute_eb_charge(weighting=r, field="Efield_fp")
                  for r in corrector.regions])
    corrector._restore_efield(saved, 0)
    warpx.set_potential_on_eb(corrector.potential_expression)
    return q


def band_nodes():
    """Nodes OUTSIDE both spheres but within `band` cells of either surface."""
    n1 = nx + 1
    xn = -half + dx * np.arange(n1)
    X, Y, Z = np.meshgrid(xn, xn, xn, indexing="ij")
    d1 = np.sqrt((X + center_offset) ** 2 + Y ** 2 + Z ** 2) - R
    d2 = np.sqrt((X - center_offset) ** 2 + Y ** 2 + Z ** 2) - RR
    d = np.minimum(d1, d2) / dx
    # Probe BOTH sides of the surface when --band_in is set. Covered nodes
    # (d < 0) hold psi = 1 exactly by Dirichlet BC, which is right for charge
    # sitting ON the conductor but is not the weighting a particle being
    # absorbed should contribute; measuring them is the point.
    sel = (d > -args.band_in) & (d <= args.band)
    if args.band_in <= 0:
        sel &= (d > 0.15)
    idx = np.argwhere(sel)
    return idx, xn, d


def run():
    n1 = nx + 1
    mfr = corrector._mfr()
    if not corrector._psi_stored:
        raise RuntimeError("psi_k not stored; cannot seed the table")
    psi_plain = [np.array(mfr.get(corrector._psi_names[k], level=0)[:, :, :])
                 for k in range(corrector.n)]

    idx, xn, dfield = band_nodes()
    if len(idx) > args.max_nodes:
        raise RuntimeError(
            f"band holds {len(idx)} nodes, above --max_nodes={args.max_nodes}. "
            "Raise the cap deliberately or narrow --band; refusing to "
            "silently probe a subset, which would leave a table that looks "
            "complete but is not.")

    from pywarpx.particle_containers import ParticleContainerWrapper
    pc = ParticleContainerWrapper("probe")

    w_macro = args.qprobe / Q_E
    q_each = -args.qprobe                     # electrons carry -e

    psi_meas = [p_.copy() for p_ in psi_plain]   # seed with plain, overwrite band
    measured = np.zeros((len(idx), corrector.n))

    q_prev = grounded_qg()
    print(f"  vacuum Q_g = {q_prev}  (must be ~0)")
    print(f"  probing {len(idx)} band nodes ...")
    t0 = time.time()
    for n_, (i, j, k) in enumerate(idx):
        x, y, z = xn[i], xn[j], xn[k]
        pc.add_particles(x=np.array([x]), y=np.array([y]), z=np.array([z]),
                         ux=np.zeros(1), uy=np.zeros(1), uz=np.zeros(1),
                         w=np.array([w_macro]))
        q_now = grounded_qg()
        vals = -(q_now - q_prev) / q_each
        q_prev = q_now
        measured[n_] = vals
        for e in range(corrector.n):
            psi_meas[e][i, j, k] = vals[e]
        if n_ and n_ % 200 == 0:
            el = time.time() - t0
            print(f"    {n_}/{len(idx)}  ({el:.0f} s, "
                  f"eta {el/n_*(len(idx)-n_):.0f} s)")

    el = time.time() - t0
    d_band = dfield[idx[:, 0], idx[:, 1], idx[:, 2]]
    own = np.argmax(np.abs(np.stack([psi_plain[e][idx[:, 0], idx[:, 1],
                                                  idx[:, 2]]
                                     for e in range(corrector.n)])), axis=0)
    m_own = measured[np.arange(len(idx)), own]
    p_own = np.stack([psi_plain[e][idx[:, 0], idx[:, 1], idx[:, 2]]
                      for e in range(corrector.n)])[own, np.arange(len(idx))]

    np.savez(OUT,
             **{f"psi_{e}": psi_meas[e] for e in range(corrector.n)},
             band_idx=idx, band_d_over_h=d_band,
             measured=measured, plain=p_own, nx=nx, dx=dx)

    print("\n" + "=" * 88)
    print("T5 RESULT")
    print("=" * 88)
    print(f"  probed {len(idx)} nodes in {el:.0f} s "
          f"({1e3*el/len(idx):.0f} ms/node)")
    print(f"  measured Psi (own electrode): mean {m_own.mean():.4f}, "
          f"range [{m_own.min():.4f}, {m_own.max():.4f}]")
    print(f"  stored plain psi, same nodes: mean {p_own.mean():.4f}, "
          f"range [{p_own.min():.4f}, {p_own.max():.4f}]")
    print(f"  implied mis-booking rate 1-Psi:  measured "
          f"{100*(1-m_own.mean()):.1f}%   plain {100*(1-p_own.mean()):.1f}%")
    for lo, hi in ((0.15, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.01)):
        m = (d_band >= lo) & (d_band < hi)
        if m.any():
            print(f"    d/h in [{lo},{hi}): n={m.sum():4d}  "
                  f"measured {m_own[m].mean():.4f}   plain {p_own[m].mean():.4f}")
    print(f"  wrote {OUT}")
    print("=" * 88)
    json.dump({"nx": int(nx), "n_nodes": int(len(idx)),
               "seconds": float(el),
               "measured_mean": float(m_own.mean()),
               "plain_mean": float(p_own.mean())},
              open(f"t5_summary_nx{nx}.json", "w"), indent=1)


installafterInitEsolve(run)
sim.step(0)
