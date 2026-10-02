#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Checkpoint/restart of the staircase voltage clamp during electron emission.

--mode base:    emit electrons from a clamped rod for 2N steps, write a checkpoint at
                step N and record the clamp observer, the line-integral voltage and E_r
                for steps N+1..2N.
--mode restart: restart from that checkpoint, rebuild the clamp basis, resume (no
                re-initialization of the bias) and require the same trace and final E_r.

Nothing of the clamp is stored in the checkpoint; it is rebuilt from the restored
fields. Particle order may change on restart, so values are compared to 1e-12.
"""

import argparse
import gc
from pathlib import Path

import numpy as np

from pywarpx import callbacks, picmi
from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["base", "restart"], required=True)
args = parser.parse_args()

NR, NZ, N = 48, 8, 30
R_ROD, R_SLEEVE_IN, R_SLEEVE_OUT = 6.2e-3, 5.03e-2, 5.53e-2
R_WALL, LENGTH = 6.0e-2, 4.0e-2
BASE_DIR = Path("../test_rz_staircase_clamp_restart_base_picmi")
TRACE = "restart_trace.npz"

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    n_azimuthal_modes=1,
    lower_bound=[0.0, 0.0],
    upper_bound=[R_WALL, LENGTH],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=16,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
r2 = "(x*x+y*y)"
rod = f"({R_ROD}*{R_ROD}-{r2})"
sleeve = f"(({r2}-{R_SLEEVE_IN}*{R_SLEEVE_IN})*({R_SLEEVE_OUT}*{R_SLEEVE_OUT}-{r2}))"
restart_kwargs = {}
if args.mode == "restart":
    restart_kwargs["warpx_amr_restart"] = str(BASE_DIR / "diags" / f"chk{N:06d}")
sim = picmi.Simulation(
    solver=solver,
    max_steps=2 * N,
    particle_shape="linear",
    warpx_embedded_boundary=picmi.EmbeddedBoundary(
        implicit_function=f"-({rod})*({sleeve})", potential=0.0
    ),
    warpx_use_filter=False,
    warpx_current_deposition_algo="esirkepov",
    verbose=0,
    **restart_kwargs,
)
electrons = picmi.Species(
    name="electrons", particle_type="electron", initial_distribution=None
)
sim.add_species(electrons, layout=None)
if args.mode == "base":
    sim.add_diagnostic(picmi.Checkpoint(period=N, name="chk"))
sim.initialize_inputs()
sim.initialize_warpx()

from pywarpx._libwarpx import libwarpx  # noqa: E402

warpx = libwarpx.libwarpx_so.get_instance()
direction = libwarpx.libwarpx_so.Direction
register = warpx.multifab_register()
particles = sim.particles.get("electrons")
corrector = StaircaseBiasCorrector(
    sim,
    correction_interval=1,
    electrodes=[
        {"name": "cathode", "region": "(x<0.03)", "potential": -1000.0},
        {"name": "anode", "region": "(x>0.03)", "potential": 0.0},
    ],
)
corrector.setup_after_init()
if args.mode == "base":
    corrector.initialize_vacuum_bias()
else:
    assert warpx.getistep(0) == N, f"restart began at step {warpx.getistep(0)}"
    corrector.resume_from_checkpoint()

dr, dz = R_WALL / NR, LENGTH / NZ
# The represented cathode surface on this mesh is the node at 5 dr = 6.25 mm; birth
# between it and the geometric surface keeps the emitted charge on fixed nodes.
r_birth = R_ROD + 0.1 * (5 * dr - R_ROD)
z_birth = (np.arange(NZ) + 0.5) * dz


def owned_valid(mf):
    """Gather valid E_r into one owned array (AMReX-backed views must not outlive finalize)."""
    out = np.zeros((NR, NZ + 1))
    for mfi, arr in zip(mf, mf.to_numpy(copy=True)):
        v, f = mfi.validbox(), mfi.fabbox()
        a = np.squeeze(
            arr[
                v.small_end[0] - f.small_end[0] : v.big_end[0] - f.small_end[0] + 1,
                v.small_end[1] - f.small_end[1] : v.big_end[1] - f.small_end[1] + 1,
            ]
        )
        out[v.small_end[0] : v.big_end[0] + 1, v.small_end[1] : v.big_end[1] + 1] = a
    return out


trace = {"step": [], "observer": [], "er_sum": []}


def inject():
    particles.add_particles(
        x=np.full(NZ, r_birth),
        y=np.zeros(NZ),
        z=z_birth,
        ux=np.full(NZ, 1.0e6),
        uy=np.zeros(NZ),
        uz=np.zeros(NZ),
        w=np.full(NZ, 2.0e6),
        unique_particles=False,
    )


def record():
    step = warpx.getistep(0)
    if step <= N:
        return
    v = np.asarray(corrector.measure_voltage_state()["voltage"], dtype=float)
    er = owned_valid(register.get("Efield_fp", dir=direction(0), level=0))
    from mpi4py import MPI  # noqa: PLC0415

    er = MPI.COMM_WORLD.allreduce(er, op=MPI.SUM)
    trace["step"].append(step)
    trace["observer"].append(v)
    trace["er_sum"].append(float(np.sum(er)))
    if step == 2 * N:
        trace["er_final"] = er


callbacks.installbeforestep(inject)
callbacks.installafterstep(corrector.correct_after_step)
callbacks.installafterstep(record)
sim.step(2 * N - warpx.getistep(0))

result = {k: np.asarray(v) for k, v in trace.items()}
callbacks.uninstallcallback("beforestep", inject)
callbacks.uninstallcallback("afterstep", corrector.correct_after_step)
callbacks.uninstallcallback("afterstep", record)

from mpi4py import MPI  # noqa: E402

# Save or compare before finalize: sim.finalize() also finalizes MPI.
failure = None
if args.mode == "base":
    if MPI.COMM_WORLD.rank == 0:
        np.savez(TRACE, **result)
    MPI.COMM_WORLD.Barrier()
else:
    base = np.load(BASE_DIR / TRACE)
    scale = np.max(np.abs(base["er_final"]))
    checks = {
        "same steps": np.array_equal(base["step"], result["step"]),
        "final E_r": np.max(np.abs(result["er_final"] - base["er_final"]))
        <= 1.0e-12 * scale,
        "observer trace": np.max(np.abs(result["observer"] - base["observer"]))
        <= 1.0e-9,
        "target held": np.max(np.abs(result["observer"][:, 0] + 1000.0)) <= 1.0e-9,
    }
    print(checks)
    failure = [k for k, ok in checks.items() if not ok]

corrector = particles = register = None
gc.collect()
sim.finalize()

assert not failure, f"restart does not reproduce the base run: {failure}"
