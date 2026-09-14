#!/usr/bin/env python3
"""Native RZ staircase-bias vacuum regression test."""

import argparse
import gc

import numpy as np
from mpi4py import MPI

from pywarpx import callbacks, picmi
from pywarpx._libwarpx import libwarpx
from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

NR = 48
NZ = 8
R_INNER = 6.2e-3
R_OUTER = 6.0e-2
LENGTH = 4.0e-2
VOLTAGE = -1.0e4
STEPS = 20


def build_simulation(max_grid_size):
    grid = picmi.CylindricalGrid(
        number_of_cells=[NR, NZ],
        n_azimuthal_modes=1,
        lower_bound=[0.0, 0.0],
        upper_bound=[R_OUTER, LENGTH],
        lower_boundary_conditions=["none", "periodic"],
        upper_boundary_conditions=["dirichlet", "periodic"],
        lower_boundary_conditions_particles=["none", "periodic"],
        upper_boundary_conditions_particles=["absorbing", "periodic"],
        warpx_blocking_factor=8,
        warpx_max_grid_size=max_grid_size,
    )
    solver = picmi.ElectromagneticSolver(
        grid=grid,
        method="Yee",
        cfl=0.9,
        divE_cleaning=False,
    )
    eb = picmi.EmbeddedBoundary(
        implicit_function=f"{R_INNER}*{R_INNER}-(x*x+y*y)",
        potential=0.0,
    )
    return picmi.Simulation(
        solver=solver,
        max_steps=STEPS,
        particle_shape="linear",
        warpx_embedded_boundary=eb,
        warpx_use_filter=False,
        verbose=0,
    )


def frozen_max(fields, masks):
    local_max = 0.0
    for field, mask in zip(fields, masks):
        for values, flags in zip(field.to_numpy(copy=True), mask.to_numpy(copy=True)):
            values = np.squeeze(values)
            flags = np.squeeze(flags)
            ng_value = tuple(int(n) for n in field.n_grow_vect)
            ng_flag = tuple(int(n) for n in mask.n_grow_vect)
            value_valid = tuple(
                slice(n, values.shape[d] - n) for d, n in enumerate(ng_value)
            )
            flag_valid = tuple(
                slice(n, flags.shape[d] - n) for d, n in enumerate(ng_flag)
            )
            values = values[value_valid]
            flags = flags[flag_valid]
            local_max = max(
                local_max,
                float(np.max(np.abs(values[flags == 0]), initial=0.0)),
            )
    return MPI.COMM_WORLD.allreduce(local_max, op=MPI.MAX)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-grid-size", type=int, default=1024)
    args = parser.parse_args()

    simulation = build_simulation(args.max_grid_size)
    initialized = False
    after_step = []
    correct_callback = None
    audit_callback = None
    try:
        simulation.initialize_inputs()
        simulation.initialize_warpx()
        initialized = True

        warpx = libwarpx.libwarpx_so.get_instance()
        direction = libwarpx.libwarpx_so.Direction
        fields = warpx.multifab_register()
        efield = [fields.get("Efield_fp", dir=direction(k), level=0) for k in range(3)]
        bfield = [fields.get("Bfield_fp", dir=direction(k), level=0) for k in range(3)]
        masks = [warpx.eb_update_e_flag(0, k) for k in range(3)]

        corrector = StaircaseBiasCorrector(
            simulation,
            correction_interval=5,
            electrodes=[
                {
                    "name": "inner",
                    "region": "(x<0.02)",
                    "potential": VOLTAGE,
                }
            ],
        )
        corrector.setup_after_init()

        # Each axial copy of the selected native fixed-node set contributes one.
        # Thus the fixed weight, which is built from the Er/Ez masks, determines
        # the last fixed radial-node index m without a geometric floor/ceil rule.
        weight = fields.get(corrector._weight_names[0], level=0)
        fixed_nodes_per_z = weight.sum_unique(0, False) / (NZ + 1)
        m_real = fixed_nodes_per_z - 1.0
        m = int(round(m_real))
        assert abs(m_real - m) < 1.0e-12 and 0 < m < NR

        dr = R_OUTER / NR
        r_half = (np.arange(m, NR) + 0.5) * dr
        capacitance_exact = (
            2.0 * np.pi * picmi.constants.ep0 * LENGTH / np.sum(dr / r_half)
        )
        capacitance_native = float(corrector._capacitance[0, 0])
        capacitance_error = abs(capacitance_native / capacitance_exact - 1.0)

        assert all(field.norm0(0, 0, False, False) == 0.0 for field in efield)
        corrector.initialize_vacuum_bias()
        initial_e = [field.copy() for field in efield]
        initial_voltage = float(corrector.measure_voltage_state()["voltage"][0])
        initial_frozen = frozen_max(efield, masks)

        dt = float(warpx.getdt(0))

        def audit_after_step():
            after_step.append((int(warpx.getistep(0)), float(warpx.gett_new(0))))

        correct_callback = corrector.correct_after_step
        audit_callback = audit_after_step
        callbacks.installafterstep(correct_callback)
        callbacks.installafterstep(audit_callback)
        simulation.step(STEPS)

        final_voltage = float(corrector.measure_voltage_state()["voltage"][0])
        difference = []
        for field, initial in zip(efield, initial_e):
            delta = field.copy()
            delta.saxpy(-1.0, initial, 0, 0, 1, 0)
            difference.append(delta.norm0(0, 0, False, False))
        e_scale = max(field.norm0(0, 0, False, False) for field in initial_e)
        e_change = max(difference) / e_scale
        b_max = max(field.norm0(0, 0, False, False) for field in bfield)
        magnetic_ratio = picmi.constants.c * b_max / e_scale
        final_frozen = frozen_max(efield, masks)

        callback_steps = np.asarray([entry[0] for entry in after_step])
        callback_times = np.asarray([entry[1] for entry in after_step])
        expected_steps = np.arange(1, STEPS + 1)
        callback_time_error = float(
            np.max(np.abs(callback_times - expected_steps * dt)) / dt
        )

        assert capacitance_error < 1.0e-9
        assert abs(initial_voltage - VOLTAGE) < 1.0e-8
        assert abs(final_voltage - VOLTAGE) < 1.0e-8
        assert e_change < 1.0e-12
        assert initial_frozen == 0.0 and final_frozen == 0.0
        assert magnetic_ratio < 1.0e-12
        assert np.array_equal(callback_steps, expected_steps)
        assert callback_time_error < 1.0e-12
        assert corrector.last_correction_state()["step"] == STEPS

        if MPI.COMM_WORLD.rank == 0:
            print(
                "staircase vacuum PASS:",
                f"m={m}, Crel={capacitance_error:.3e},",
                f"dE/E={e_change:.3e}, cB/E={magnetic_ratio:.3e},",
                f"callback_dt_error={callback_time_error:.3e}",
            )
    finally:
        if correct_callback is not None:
            callbacks.uninstallcallback("afterstep", correct_callback)
        if audit_callback is not None:
            callbacks.uninstallcallback("afterstep", audit_callback)
        gc.collect()
        if initialized:
            simulation.finalize()


if __name__ == "__main__":
    main()
