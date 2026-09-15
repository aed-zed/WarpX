#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
# License: BSD-3-Clause-LBNL

"""RZ harmonic-trace regression for an EB joined to the grounded radial wall."""

import argparse
import gc

import numpy as np
from mpi4py import MPI

from pywarpx import boundary, picmi
from pywarpx import warpx as warpx_inputs
from pywarpx._libwarpx import libwarpx
from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

NR, NZ = 24, 48
R_WALL, Z_HALF = 0.06, 0.08
R_ROD = 0.0062
RING_IN, RING_OUT, RING_HALF_Z = 0.031, 0.041, 0.007
REFERENCE_IN = 0.050
TARGETS = np.array([-4000.0, 2500.0])
STEPS, SOLVE_TOL = 10, 2.0e-11
COMM = MPI.COMM_WORLD


def valid_blocks(mf):
    """Return local valid data together with global inclusive bounds."""
    result = []
    for mfi, array in zip(mf, mf.to_numpy(copy=True)):
        valid, fab = mfi.validbox(), mfi.fabbox()
        lo = tuple(valid.small_end[d] for d in range(2))
        hi = tuple(valid.big_end[d] for d in range(2))
        start = tuple(lo[d] - fab.small_end[d] for d in range(2))
        stop = tuple(hi[d] - fab.small_end[d] + 1 for d in range(2))
        data = np.squeeze(array[tuple(slice(start[d], stop[d]) for d in range(2))])
        result.append((lo, hi, data))
    return result


def global_array(mf, shape):
    """Assemble replicated valid data and require duplicate owners to agree."""
    values = np.zeros(shape)
    counts = np.zeros(shape, dtype=np.int64)
    lower = np.full(shape, np.inf)
    upper = np.full(shape, -np.inf)
    for lo, hi, data in valid_blocks(mf):
        region = tuple(slice(lo[d], hi[d] + 1) for d in range(2))
        values[region] += data
        counts[region] += 1
        lower[region] = np.minimum(lower[region], data)
        upper[region] = np.maximum(upper[region], data)
    COMM.Allreduce(MPI.IN_PLACE, values, op=MPI.SUM)
    COMM.Allreduce(MPI.IN_PLACE, counts, op=MPI.SUM)
    COMM.Allreduce(MPI.IN_PLACE, lower, op=MPI.MIN)
    COMM.Allreduce(MPI.IN_PLACE, upper, op=MPI.MAX)
    assert np.all(counts > 0)
    result = values / counts
    scale = np.max(np.abs(result), initial=0.0)
    assert np.max(np.abs(upper - lower), initial=0.0) <= (
        16.0 * np.finfo(float).eps * max(scale, 1.0)
    )
    return result


def frozen_graph(masks):
    radial = global_array(masks[0], (NR, NZ + 1))
    axial = global_array(masks[2], (NR + 1, NZ))
    frozen_r = radial == 0
    frozen_z = axial == 0
    fixed = np.zeros((NR + 1, NZ + 1), dtype=bool)
    fixed[:-1, :] |= frozen_r
    fixed[1:, :] |= frozen_r
    fixed[:, :-1] |= frozen_z
    fixed[:, 1:] |= frozen_z

    reference = np.zeros_like(fixed)
    queue = [(NR, j) for j in range(NZ + 1) if fixed[NR, j]]
    for node in queue:
        reference[node] = True
    for i, j in queue:
        neighbors = []
        if i < NR and frozen_r[i, j]:
            neighbors.append((i + 1, j))
        if i > 0 and frozen_r[i - 1, j]:
            neighbors.append((i - 1, j))
        if j < NZ and frozen_z[i, j]:
            neighbors.append((i, j + 1))
        if j > 0 and frozen_z[i, j - 1]:
            neighbors.append((i, j - 1))
        for node in neighbors:
            if not reference[node]:
                reference[node] = True
                queue.append(node)
    return fixed, reference


def frozen_max(fields, masks):
    local = 0.0
    for field, mask in zip(fields, masks):
        for (_, _, values), (_, _, flags) in zip(valid_blocks(field), valid_blocks(mask)):
            local = max(local, np.max(np.abs(values[flags == 0]), initial=0.0))
    return COMM.allreduce(local, op=MPI.MAX)


def free_divergence_max(div_e, fixed):
    """Measure all physical free nodes; exclusion comes from masks, not weights."""
    divergence = global_array(div_e, (NR + 1, NZ + 1))
    radial_interior = np.arange(NR + 1)[:, None] < NR
    free = ~fixed & radial_interior
    end = free & ((np.arange(NZ + 1)[None, :] == 0) |
                  (np.arange(NZ + 1)[None, :] == NZ))
    bulk = free & ~end
    return (
        np.max(np.abs(divergence[end]), initial=0.0),
        np.max(np.abs(divergence[bulk]), initial=0.0),
    )


def implicit_geometry(case):
    rod = f"{R_ROD}*{R_ROD}-(x*x+y*y)"
    ring = (
        f"min(min((x*x+y*y)-{RING_IN}*{RING_IN},"
        f" {RING_OUT}*{RING_OUT}-(x*x+y*y)),"
        f" {RING_HALF_Z}*{RING_HALF_Z}-z*z)"
    )
    wall_reference = f"(x*x+y*y)-{REFERENCE_IN}*{REFERENCE_IN}"
    parts = [rod, ring, wall_reference]
    result = parts[0]
    for part in parts[1:]:
        result = f"max({result},{part})"
    return result


def electrode_list(case):
    rod_region = "(x<0.02)"
    if case == "split":
        rod_region = "(x<0.02)*(z<0)"
    result = [
        {"name": "rod", "region": rod_region, "potential": TARGETS[0]},
        {
            "name": "ring",
            "region": "(x>=0.02)*(x<0.045)",
            "potential": TARGETS[1],
        },
    ]
    if case == "omit-island":
        result.pop()
    if case == "driven-wall":
        result.append(
            {"name": "wall_eb", "region": "x>=0.045", "potential": 0.0}
        )
    return result


def main():
    global NZ, Z_HALF

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-grid-size", type=int, default=16)
    parser.add_argument("--long-coax", action="store_true")
    parser.add_argument(
        "--case",
        choices=("positive", "legacy", "omit-island", "driven-wall", "split"),
        default="positive",
    )
    args = parser.parse_args()
    if args.long_coax:
        NZ, Z_HALF = 400, 0.5
    grid = picmi.CylindricalGrid(
        number_of_cells=[NR, NZ], n_azimuthal_modes=1,
        lower_bound=[0.0, -Z_HALF], upper_bound=[R_WALL, Z_HALF],
        lower_boundary_conditions=["none", "none"],
        upper_boundary_conditions=["dirichlet", "none"],
        lower_boundary_conditions_particles=["none", "absorbing"],
        upper_boundary_conditions_particles=["absorbing", "absorbing"],
        warpx_blocking_factor=8, warpx_max_grid_size=args.max_grid_size,
    )
    solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
    sim = picmi.Simulation(
        solver=solver, max_steps=STEPS, particle_shape="linear",
        warpx_embedded_boundary=picmi.EmbeddedBoundary(
            implicit_function=implicit_geometry(args.case)
        ),
        warpx_use_filter=False, verbose=0,
    )
    initialized = False
    saved = live_e = live_b = masks = units = weights = corrector = None
    try:
        sim.initialize_inputs()
        boundary.field_lo = ["none", "pec_insulator"]
        boundary.field_hi = ["pec", "pec_insulator"]
        insulator = warpx_inputs.get_bucket("insulator")
        setattr(insulator, "area_z_lo(x,y)", "1")
        setattr(insulator, "area_z_hi(x,y)", "1")
        sim.initialize_warpx()
        initialized = True
        wx = libwarpx.libwarpx_so.get_instance()
        direction = libwarpx.libwarpx_so.Direction
        mfr = wx.multifab_register()
        live_e = [mfr.get("Efield_fp", dir=direction(k), level=0) for k in range(3)]
        live_b = [mfr.get("Bfield_fp", dir=direction(k), level=0) for k in range(3)]
        masks = [wx.eb_update_e_flag(0, k) for k in range(3)]
        corrector = StaircaseBiasCorrector(
            sim, correction_interval=1000, electrodes=electrode_list(args.case),
            insulating_endcaps=True, insulating_endcap_model="harmonic_trace",
            grounded_wall_reference=args.case != "legacy",
            tolerance=SOLVE_TOL, max_iterations=300,
        )
        corrector.setup_after_init()
        if args.case != "positive":
            raise AssertionError(f"negative case unexpectedly passed: {args.case}")

        fixed, reference = frozen_graph(masks)
        assert np.any(reference) and np.any(fixed & ~reference)
        weights = [mfr.get(name, level=0) for name in corrector._weight_names]
        weight_sum = sum(global_array(weight, (NR + 1, NZ + 1)) for weight in weights)
        assert np.max(np.abs(weight_sum[reference]), initial=0.0) == 0.0
        assert np.max(np.abs(weight_sum[fixed & ~reference] - 1.0), initial=0.0) == 0.0

        units = [
            [mfr.get(name, dir=direction(k), level=0) for k in range(3)]
            for name in corrector._unit_names
        ]
        div_errors = []
        saved = [field.copy() for field in live_e]
        for unit in units:
            scale = max(field.norm0(0, 0, False, False) for field in unit)
            assert frozen_max(unit, masks) < 1.0e-13 * scale
            for field, source in zip(live_e, unit):
                field.copymf(source, 0, 0, 1, 0)
            wx.staircase_charge_state(
                corrector._psi_names, corrector._weight_names,
                insulating_endcaps=True,
            )
            end_error, bulk_error = free_divergence_max(wx.compute_div_e(0), fixed)
            limit = max(1000.0 * SOLVE_TOL, 1.0e-8) * scale / (R_WALL / NR)
            assert end_error < limit and bulk_error < limit
            div_errors.append((end_error / (scale / (R_WALL / NR)),
                               bulk_error / (scale / (R_WALL / NR))))
            for field, zero in zip(live_e, saved):
                field.copymf(zero, 0, 0, 1, 0)

        setup = corrector.setup_state()
        cap = np.asarray(setup["capacitance_matrix"])
        gauss = np.asarray(setup["raw_gauss_actuator_matrix"])
        flux = np.asarray(setup["boundary_flux_actuator_matrix"])
        assert cap.shape == (2, 2)
        gb_error = np.max(np.abs(gauss - flux - cap)) / np.max(np.abs(cap))
        assert gb_error < 2.0e-12

        corrector.initialize_vacuum_bias()
        initial = [field.copy() for field in live_e]
        scale = max(field.norm0(0, 0, False, False) for field in initial)
        assert frozen_max(initial, masks) < 1.0e-12 * scale
        assert np.allclose(corrector.measure_voltage_state()["voltage"], TARGETS,
                           rtol=0.0, atol=1.0e-8)
        er = global_array(live_e[0], (NR, NZ + 1))
        dr = R_WALL / NR
        rod_weight = global_array(weights[0], (NR + 1, NZ + 1))
        ring_weight = global_array(weights[1], (NR + 1, NZ + 1))
        rod_near = np.flatnonzero(rod_weight[:, 1] > 0.5)[-1]
        rod_mid = np.flatnonzero(rod_weight[:, NZ // 2] > 0.5)[-1]
        ring_mid = np.flatnonzero(ring_weight[:, NZ // 2] > 0.5)[-1]
        path_voltages = np.array([
            dr * np.sum(er[rod_near:, 1]),
            dr * np.sum(er[rod_mid:, NZ // 2]),
            dr * np.sum(er[ring_mid:, NZ // 2]),
        ])
        assert np.allclose(
            path_voltages, [TARGETS[0], TARGETS[0], TARGETS[1]],
            rtol=2.0e-10, atol=1.0e-8,
        )
        sim.step(STEPS)
        change = 0.0
        for field, before in zip(live_e, initial):
            delta = field.copy()
            delta.saxpy(-1.0, before, 0, 0, 1, 0)
            change = max(change, delta.norm0(0, 0, False, False))
        magnetic = picmi.constants.c * max(
            field.norm0(0, 0, False, False) for field in live_b
        ) / scale
        assert change / scale < 1.0e-9 and magnetic < 1.0e-9
        assert frozen_max(live_e, masks) < 1.0e-12 * scale
        if COMM.rank == 0:
            print(
                "RZ grounded-wall reference PASS:",
                f"reference_nodes={np.count_nonzero(reference)},",
                f"G-B-C={gb_error:.3e}, div(end,bulk)={div_errors},",
                f"paths={path_voltages.tolist()}, dE/E={change/scale:.3e},",
                f"cB/E={magnetic:.3e}",
            )
    finally:
        saved = live_e = live_b = masks = units = weights = corrector = None
        initial = delta = field = before = source = zero = unit = None
        gc.collect()
        if initialized:
            sim.finalize()


if __name__ == "__main__":
    main()
