#!/usr/bin/env python3
#
# --- Fields registered from Python: written to the diagnostics by their
# --- registered names, and saved to and restored from a checkpoint.
# --- The fields count the steps taken, so after a restart they hold the
# --- right value only if they were restored from the checkpoint.

import sys

from pywarpx import callbacks, libwarpx, picmi

max_steps = 10

grid = picmi.Cartesian3DGrid(
    number_of_cells=[8, 8, 8],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[1.0, 1.0, 1.0],
    lower_boundary_conditions=["periodic", "periodic", "periodic"],
    upper_boundary_conditions=["periodic", "periodic", "periodic"],
    warpx_max_grid_size=4,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.99)

sim = picmi.Simulation(solver=solver, max_steps=max_steps, verbose=1)

# Registered fields are requested in data_list by name. The components of a
# vector field are requested as "<name>x", "<name>y", "<name>z", like "Ex".
sim.add_diagnostic(
    picmi.FieldDiagnostic(
        name="diag1",
        grid=grid,
        period=max_steps,
        data_list=[
            "step_count",
            "step_count_vecx",
            "step_count_vecy",
            "step_count_vecz",
        ],
    )
)
sim.add_diagnostic(picmi.Checkpoint(name="chk", period=5))

for arg in sys.argv:
    if arg.startswith("amr.restart"):
        sim.amr_restart = arg.split("=")[1].strip("'\"")
        sys.argv.remove(arg)


# Registered fields are allocated in the "allocdata" callback, which runs both
# when starting from scratch and when restarting. With checkpoint_restart=True,
# a field is written to every checkpoint and, on restart, read back after this
# callback has allocated it.
@callbacks.callfromallocdata
def allocate_step_count():
    Ex = sim.fields.get("Efield_fp", dir="x", level=0)
    common = dict(
        level=0,
        dm=Ex.dm(),
        ncomp=1,
        ngrow=libwarpx.amr.IntVect.zero_vector(),  # no guard cells needed
        initial_value=0.0,
        redistribute=True,
        redistribute_on_remake=True,
        checkpoint_restart=True,
    )
    # A scalar field, here with the same layout as Ex
    sim.fields.alloc_init(name="step_count", ba=Ex.box_array(), **common)
    # A vector field: one MultiFab per direction, here laid out like E
    for dir in ["x", "y", "z"]:
        E = sim.fields.get("Efield_fp", dir=dir, level=0)
        sim.fields.alloc_init(
            name="step_count_vec", dir=dir, ba=E.box_array(), **common
        )


@callbacks.callfromafterstep
def count_step():
    sim.fields.get("step_count", level=0).plus(1.0)
    # 1, 2 and 3 per step, to tell the components apart
    for i, dir in enumerate(["x", "y", "z"]):
        sim.fields.get("step_count_vec", dir=dir, level=0).plus(i + 1.0)


sim.initialize_inputs()
sim.initialize_warpx()

# A restarted run continues from the checkpoint step
sim.step(max_steps - sim.extension.warpx.getistep(lev=0))

# Exact after max_steps, whether started from scratch or restarted
assert sim.fields.get("step_count", level=0).min(0) == max_steps
assert sim.fields.get("step_count", level=0).max(0) == max_steps
for i, dir in enumerate(["x", "y", "z"]):
    mf = sim.fields.get("step_count_vec", dir=dir, level=0)
    assert mf.min(0) == mf.max(0) == (i + 1) * max_steps
