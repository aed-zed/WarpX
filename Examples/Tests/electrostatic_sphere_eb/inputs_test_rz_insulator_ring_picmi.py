#!/usr/bin/env python3
# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
# License: BSD-3-Clause-LBNL

"""Native RZ harmonic-trace insulating-endcap regression."""

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
R_ROD, RING_IN, RING_OUT, RING_HALF_Z = 0.0062, 0.031, 0.041, 0.007
TARGETS = np.array([-4000.0, 2500.0])
STEPS, SOLVE_TOL = 20, 2.0e-11
COMM = MPI.COMM_WORLD


def valid_blocks(mf):
    """Return host copies of local valid regions with their global bounds."""
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


def global_line(mf, j, size):
    values, counts = np.zeros(size), np.zeros(size, dtype=np.int64)
    for lo, hi, data in valid_blocks(mf):
        if lo[1] <= j <= hi[1]:
            segment = np.asarray(data[:, j - lo[1]], dtype=float)
            values[lo[0] : hi[0] + 1] += segment
            counts[lo[0] : hi[0] + 1] += 1
    COMM.Allreduce(MPI.IN_PLACE, values, op=MPI.SUM)
    COMM.Allreduce(MPI.IN_PLACE, counts, op=MPI.SUM)
    assert np.all(counts > 0)
    return values / counts


def frozen_max(fields, masks):
    local = 0.0
    for field, mask in zip(fields, masks):
        for (_, _, values), (_, _, flags) in zip(valid_blocks(field), valid_blocks(mask)):
            local = max(local, np.max(np.abs(values[flags == 0]), initial=0.0))
    return COMM.allreduce(local, op=MPI.MAX)


def free_divergence_max(div_e, weights):
    end_max = bulk_max = 0.0
    blocks = [valid_blocks(field) for field in (div_e, *weights)]
    for (lo, hi, div), (_, _, wr), (_, _, wg) in zip(*blocks):
        ii = np.arange(lo[0], hi[0] + 1)[:, None]
        jj = np.arange(lo[1], hi[1] + 1)[None, :]
        free = (wr + wg < 0.5) & (ii < NR)
        end = free & ((jj == 0) | (jj == NZ))
        bulk = free & (jj > 0) & (jj < NZ)
        end_max = max(end_max, np.max(np.abs(div[end]), initial=0.0))
        bulk_max = max(bulk_max, np.max(np.abs(div[bulk]), initial=0.0))
    return (
        COMM.allreduce(end_max, op=MPI.MAX),
        COMM.allreduce(bulk_max, op=MPI.MAX),
    )


def refreshed_divergence_error(wx, corrector, fields):
    """Compare immediate div(E) with the observer's guard-refreshed value."""
    direct = wx.compute_div_e(0)
    before = [field.copy() for field in fields]
    wx.staircase_charge_state(
        corrector._psi_names, corrector._weight_names,
        insulating_endcaps=True,
    )
    refreshed = wx.compute_div_e(0)
    direct.saxpy(-1.0, refreshed, 0, 0, 1, 0)
    field_change = 0.0
    for field, saved in zip(fields, before):
        saved.saxpy(-1.0, field, 0, 0, 1, 0)
        field_change = max(field_change, saved.norm0(0, 0, False, False))
    return direct.norm0(0, 0, False, False), field_change


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-grid-size", type=int, default=16)
    args = parser.parse_args()
    dr = R_WALL / NR

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
    rod = f"{R_ROD}*{R_ROD}-(x*x+y*y)"
    ring = (
        f"min(min((x*x+y*y)-{RING_IN}*{RING_IN},"
        f" {RING_OUT}*{RING_OUT}-(x*x+y*y)),"
        f" {RING_HALF_Z}*{RING_HALF_Z}-z*z)"
    )
    eb = picmi.EmbeddedBoundary(implicit_function=f"max({rod},{ring})")
    sim = picmi.Simulation(
        solver=solver, max_steps=STEPS, particle_shape="linear",
        warpx_embedded_boundary=eb, warpx_use_filter=False, verbose=0,
    )
    initialized = False
    refs = live_e = live_b = masks = units = weights = div_e = corrector = None
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
        refs = [field.copy() for field in live_e]
        corrector = StaircaseBiasCorrector(
            sim, correction_interval=1000,
            electrodes=[
                {"name": "rod", "region": "x<0.02", "potential": TARGETS[0]},
                {"name": "ring", "region": "x>=0.02", "potential": TARGETS[1]},
            ],
            insulating_endcaps=True, insulating_endcap_model="harmonic_trace",
            tolerance=SOLVE_TOL, max_iterations=300,
        )
        corrector.setup_after_init()
        assert max(field.norm0(0, 0, False, False) for field in live_e) == 0.0

        units = [[mfr.get(name, dir=direction(k), level=0) for k in range(3)]
                 for name in corrector._unit_names]
        weights = [mfr.get(name, level=0) for name in corrector._weight_names]
        div_metrics = []
        for unit in units:
            e_scale = max(field.norm0(0, 0, False, False) for field in unit)
            assert unit[2].norm0(0, 0, False, False) > 1.0e-3 * e_scale
            assert frozen_max(unit, masks) < 1.0e-13 * e_scale
            for live, source in zip(live_e, unit):
                live.copymf(source, 0, 0, 1, 0)
            wx.staircase_charge_state(
                corrector._psi_names, corrector._weight_names,
                insulating_endcaps=True,
            )
            div_e = wx.compute_div_e(0)
            end_max, bulk_max = free_divergence_max(div_e, weights)
            limit = max(1000.0 * SOLVE_TOL, 1.0e-8) * e_scale / dr
            assert end_max < limit and bulk_max < limit
            div_metrics.append((end_max / (e_scale / dr), bulk_max / (e_scale / dr)))
            for live, reference in zip(live_e, refs):
                live.copymf(reference, 0, 0, 1, 0)

        setup = corrector.setup_state()
        cap = np.asarray(setup["capacitance_matrix"])
        gauss = np.asarray(setup["raw_gauss_actuator_matrix"])
        flux = np.asarray(setup["boundary_flux_actuator_matrix"])
        gb_error = np.max(np.abs(gauss - flux - cap)) / np.max(np.abs(cap))
        assert gb_error < 2.0e-12

        corrector.initialize_vacuum_bias()
        e_scale = max(field.norm0(0, 0, False, False) for field in live_e)
        div_guard_error, observer_e_change = refreshed_divergence_error(
            wx, corrector, live_e
        )
        roundoff = 64.0 * np.finfo(float).eps
        div_guard_limit = roundoff * e_scale / dr
        observer_e_limit = roundoff * e_scale
        assert div_guard_error < div_guard_limit, (
            f"stale E guards changed div(E): {div_guard_error} >= {div_guard_limit}"
        )
        assert observer_e_change < observer_e_limit, (
            f"observer refresh changed valid E: {observer_e_change} >= {observer_e_limit}"
        )
        rod_nodes = np.flatnonzero(global_line(weights[0], 1, NR + 1) > 0.5)
        ring_nodes = np.flatnonzero(global_line(weights[1], NZ // 2, NR + 1) > 0.5)
        assert rod_nodes.size and ring_nodes.size
        er_near_end = global_line(live_e[0], 1, NR)
        er_mid = global_line(live_e[0], NZ // 2, NR)
        rod_drop = dr * np.sum(er_near_end[rod_nodes[len(rod_nodes) // 2] :])
        ring_drop = dr * np.sum(er_mid[ring_nodes[len(ring_nodes) // 2] :])
        assert np.isclose(rod_drop, TARGETS[0], rtol=2.0e-10, atol=1.0e-8)
        assert np.isclose(ring_drop, TARGETS[1], rtol=2.0e-10, atol=1.0e-8)

        initial = [field.copy() for field in live_e]
        e_scale = max(field.norm0(0, 0, False, False) for field in initial)
        sim.step(STEPS)  # Deliberately no correction callback.
        changes = []
        for field, before in zip(live_e, initial):
            delta = field.copy()
            delta.saxpy(-1.0, before, 0, 0, 1, 0)
            changes.append(delta.norm0(0, 0, False, False))
        e_change = max(changes) / e_scale
        magnetic = picmi.constants.c * max(
            field.norm0(0, 0, False, False) for field in live_b) / e_scale
        assert e_change < 1.0e-9 and magnetic < 1.0e-9
        assert frozen_max(live_e, masks) < 1.0e-12 * e_scale
        assert corrector.last_correction_state() is None
        if COMM.rank == 0:
            print(
                "RZ insulator ring PASS:", f"G-B-C={gb_error:.3e},",
                f"div(end,bulk)={div_metrics}, guard={div_guard_error:.3e},",
                f"dE/E={e_change:.3e}, cB/E={magnetic:.3e}",
            )
    finally:
        refs = live_e = live_b = masks = units = weights = div_e = corrector = None
        initial = delta = unit = field = before = source = live = reference = None
        gc.collect()
        if initialized:
            sim.finalize()


if __name__ == "__main__":
    main()
