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
"""

import os
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

    # Scalar custom field
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

    # Vector custom field (all three components)
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


def verify_output():
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

    assert "scalar_custom" in header, (
        f"'scalar_custom' not found in diagnostic Header"
    )
    # Vector field adds components with _x/_y/_z suffix
    for suffix in ["_x", "_y", "_z"]:
        assert f"vector_custom{suffix}" in header, (
            f"'vector_custom{suffix}' not found in diagnostic Header"
        )
    print("  [OK] scalar_custom and vector_custom_x/y/z found in diagnostic output")


callbacks.installafterstep(verify_output)

sim.step(1)
