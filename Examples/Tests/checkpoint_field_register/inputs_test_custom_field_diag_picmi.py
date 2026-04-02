#!/usr/bin/env python3
"""
Test custom field output in FullDiagnostics.

Feature 2 (dynamic add_field_to_diagnostic): fields allocated in Python
callbacks (after C++ init) can be dynamically added to an existing diagnostic
via sim.extension.warpx.add_field_to_diagnostic().
Both scalar and vector fields are tested.

Note on Feature 1 (static data_list):
  Fields that are already in the MultiFabRegister at C++ initialization time
  (e.g. internal auxiliary fields like hybrid_electron_pressure_fp) can be
  requested by name directly in data_list.  User-created fields from Python
  callbacks are NOT available at that time — use add_field_to_diagnostic
  instead.

Also tested here:
  - to_xp() write path: values written to FAB device arrays in a
    callfrombeforediagnostics callback are correctly readable back via
    global indexing (mf[...]) in the following afterstep callback.
"""

import os
import numpy as np
from pywarpx import picmi, callbacks

nx, ny, nz = 8, 8, 8

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[0.0, 0.0, 0.0],
    upper_bound=[1.0, 1.0, 1.0],
    lower_boundary_conditions=['periodic', 'periodic', 'periodic'],
    upper_boundary_conditions=['periodic', 'periodic', 'periodic'],
)

solver = picmi.ElectromagneticSolver(grid=grid, cfl=0.99)

sim = picmi.Simulation(solver=solver, max_steps=1, verbose=0)

diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=1,
    data_list=["Ex"],
)
sim.add_diagnostic(diag)


@callbacks.installafterInitEsolve
def setup_fields():
    Ex = sim.fields.get("Efield_fp", dir="x", level=0)

    # Scalar custom field — initialised to 1.0; the beforediagnostics
    # callback will overwrite it to 7.0 via to_xp() before the first flush.
    sim.fields.alloc_init(
        name="scalar_custom",
        level=0,
        ba=Ex.box_array(),
        dm=Ex.dm(),
        ncomp=1,
        ngrow=Ex.n_grow_vect,
        initial_value=1.0,
        redistribute=True,
        redistribute_on_remake=True,
    )

    # Vector custom field — stays at initial_value=2.0 throughout.
    for dir_str in ["x", "y", "z"]:
        sim.fields.alloc_init(
            name="vector_custom",
            dir=dir_str,
            level=0,
            ba=Ex.box_array(),
            dm=Ex.dm(),
            ncomp=1,
            ngrow=Ex.n_grow_vect,
            initial_value=2.0,
            redistribute=True,
            redistribute_on_remake=True,
        )

    # Feature 2: add both fields dynamically
    sim.extension.warpx.add_field_to_diagnostic("diag1", "scalar_custom", lev=0)
    sim.extension.warpx.add_field_to_diagnostic("diag1", "vector_custom", lev=0)


@callbacks.callfrombeforediagnostics
def update_fields_before_diag():
    """Exercise the to_xp() write path.

    Overwrites scalar_custom (initial_value=1.0) with 7.0 using per-FAB
    device arrays.  On CPU these are NumPy views; on GPU they are CuPy
    views — in both cases the data never leaves the device.
    """
    scalar_custom = sim.fields.get("scalar_custom", level=0)
    for fab in scalar_custom.to_xp(copy=False):
        fab[..., 0] = 7.0


def verify_header():
    """Check field names are present in the diagnostic Header (afterstep)."""
    step = sim.extension.warpx.getistep(lev=0)
    if step != 1:
        return

    # The initial flush writes at step 0 before the PIC loop starts.
    diag_dir = "diags/diag1000000"
    assert os.path.isdir(diag_dir), f"Diagnostic directory {diag_dir} not found!"

    header_path = os.path.join(diag_dir, "Header")
    assert os.path.isfile(header_path), "Header file not found in diagnostic output!"

    with open(header_path) as f:
        header = f.read()

    assert "scalar_custom" in header, "'scalar_custom' not found in diagnostic Header"
    for suffix in ["_x", "_y", "_z"]:
        assert f"vector_custom{suffix}" in header, (
            f"'vector_custom{suffix}' not found in diagnostic Header"
        )
    print("  [OK] scalar_custom and vector_custom_x/y/z found in diagnostic output")


def verify_to_xp():
    """Verify to_xp() write path (afterdiagnostics).

    Execution order within a step is:
      afterstep -> beforediagnostics -> FilterComputePackFlush -> afterdiagnostics

    So by the time this callback runs, update_fields_before_diag() has already
    overwritten scalar_custom via to_xp().  Reading back via global indexing
    ([...]) exercises the full device-write -> host-read path.
    """
    step = sim.extension.warpx.getistep(lev=0)
    if step != 1:
        return

    scalar_custom = sim.fields.get("scalar_custom", level=0)
    scalar_mean = scalar_custom[...].mean()
    assert abs(scalar_mean - 7.0) < 1e-10, (
        f"to_xp() write: expected scalar_custom mean 7.0, got {scalar_mean}"
    )
    print(f"  [OK] to_xp() write path: scalar_custom mean = {scalar_mean} (expected 7.0)")

    # vector_custom was never touched by to_xp() and should still be 2.0
    for dir_str in ["x", "y", "z"]:
        vc = sim.fields.get("vector_custom", dir=dir_str, level=0)
        vc_mean = vc[...].mean()
        assert abs(vc_mean - 2.0) < 1e-10, (
            f"vector_custom_{dir_str}: expected 2.0, got {vc_mean}"
        )
    print("  [OK] vector_custom unchanged at initial_value 2.0")


callbacks.installafterstep(verify_header)
callbacks.callfromafterdiagnostics(verify_to_xp)

sim.step(1)
