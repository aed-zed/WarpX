#!/usr/bin/env python3
"""Native RZ staircase-bias vacuum regression test."""

import argparse
import gc

import numpy as np
from mpi4py import MPI

from pywarpx import boundary as boundary_inputs
from pywarpx import callbacks, picmi
from pywarpx import warpx as warpx_inputs
from pywarpx._libwarpx import libwarpx
from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

NR = 48
NZ = 8
R_INNER = 6.2e-3
R_OUTER = 6.0e-2
LENGTH = 4.0e-2
VOLTAGE = -1.0e4
STEPS = 20
MANUFACTURED_EZ = 2.5e3
MANUFACTURED_FLUX_REL_TOL = 2.0e-9
MANUFACTURED_FLUX_ABS_TOL = 1.0e-24
MANUFACTURED_ZERO_FLUX_ABS_TOL = 1.0e-24
DISCRETE_LENGTH_ABS_TOL = 1.0e-15
STATE_PRESERVATION_ABS_TOL = 0.0


def build_simulation(max_grid_size, insulating_endcaps):
    if insulating_endcaps:
        # PICMI does not standardize this WarpX-native mixed boundary yet. The
        # generated placeholder is replaced before initialize_warpx below.
        axial_field_bc = "none"
        axial_particle_bc = "absorbing"
    else:
        axial_field_bc = "periodic"
        axial_particle_bc = "periodic"

    grid = picmi.CylindricalGrid(
        number_of_cells=[NR, NZ],
        n_azimuthal_modes=1,
        lower_bound=[0.0, 0.0],
        upper_bound=[R_OUTER, LENGTH],
        lower_boundary_conditions=["none", axial_field_bc],
        upper_boundary_conditions=["dirichlet", axial_field_bc],
        lower_boundary_conditions_particles=["none", axial_particle_bc],
        upper_boundary_conditions_particles=["absorbing", axial_particle_bc],
        warpx_blocking_factor=8,
        warpx_max_grid_size=max_grid_size,
    )
    solver = picmi.ElectromagneticSolver(
        grid=grid,
        method="Yee",
        cfl=0.9,
        divE_cleaning=False,
    )
    eb_options = {
        "implicit_function": f"{R_INNER}*{R_INNER}-(x*x+y*y)",
    }
    if not insulating_endcaps:
        eb_options["potential"] = 0.0
    eb = picmi.EmbeddedBoundary(**eb_options)
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


def valid_field_change(fields, references):
    difference = []
    for field, reference in zip(fields, references):
        delta = field.copy()
        delta.saxpy(-1.0, reference, 0, 0, 1, 0)
        difference.append(delta.norm0(0, 0, False, False))
    return max(difference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-grid-size", type=int, default=1024)
    parser.add_argument("--insulating-endcaps", action="store_true")
    args = parser.parse_args()

    simulation = build_simulation(args.max_grid_size, args.insulating_endcaps)
    initialized = False
    after_step = []
    correct_callback = None
    audit_callback = None
    try:
        simulation.initialize_inputs()
        if args.insulating_endcaps:
            boundary_inputs.field_lo = ["none", "pec_insulator"]
            boundary_inputs.field_hi = ["pec", "pec_insulator"]
            insulator = warpx_inputs.get_bucket("insulator")
            setattr(insulator, "area_z_lo(x,y)", "1")
            setattr(insulator, "area_z_hi(x,y)", "1")
        simulation.initialize_warpx()
        initialized = True

        warpx = libwarpx.libwarpx_so.get_instance()
        direction = libwarpx.libwarpx_so.Direction
        fields = warpx.multifab_register()
        efield = [fields.get("Efield_fp", dir=direction(k), level=0) for k in range(3)]
        bfield = [fields.get("Bfield_fp", dir=direction(k), level=0) for k in range(3)]
        masks = [warpx.eb_update_e_flag(0, k) for k in range(3)]
        before_setup = (
            [field.copy() for field in efield] if args.insulating_endcaps else None
        )
        corrector_options = (
            {"insulating_endcaps": True} if args.insulating_endcaps else {}
        )

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
            **corrector_options,
        )
        corrector.setup_after_init()

        if args.insulating_endcaps:
            assert (
                valid_field_change(efield, before_setup) <= STATE_PRESERVATION_ABS_TOL
            )

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
        if args.insulating_endcaps:
            dz = LENGTH / NZ
            capacitance_length = dz * (0.5 + (NZ - 1) + 0.5)
            assert abs(capacitance_length - LENGTH) < DISCRETE_LENGTH_ABS_TOL
            capacitance_exact = (
                2.0
                * np.pi
                * picmi.constants.ep0
                * capacitance_length
                / np.sum(dr / r_half)
            )
        else:
            capacitance_exact = (
                2.0 * np.pi * picmi.constants.ep0 * LENGTH / np.sum(dr / r_half)
            )
        capacitance_native = float(corrector._capacitance[0, 0])
        capacitance_error = abs(capacitance_native / capacitance_exact - 1.0)

        assert all(field.norm0(0, 0, False, False) == 0.0 for field in efield)
        manufactured_flux = None
        manufactured_flux_error = None
        reversed_flux = None
        balanced_flux = None
        if args.insulating_endcaps:
            # Exercise the optional fourth native result independently of the
            # voltage calculation.  A single top-face Ez degree of freedom has
            # B = eps0 (2 pi r dr) psi(r) Ez.  The expected psi below comes from
            # the exact radial resistor chain, not from the native psi field.
            saved_zero = [field.copy() for field in efield]
            probe_i = m + 2
            assert probe_i < NR
            efield[2][probe_i, NZ - 1] = MANUFACTURED_EZ
            native_state = np.asarray(
                warpx.staircase_charge_state(
                    corrector._psi_names,
                    corrector._weight_names,
                    insulating_endcaps=True,
                ),
                dtype=float,
            )
            assert native_state.shape == (4, 1)
            manufactured_flux = float(native_state[3, 0])
            probe_path = (np.arange(probe_i, NR) + 0.5) * dr
            psi_probe = np.sum(dr / probe_path) / np.sum(dr / r_half)
            manufactured_flux_exact = (
                picmi.constants.ep0
                * 2.0
                * np.pi
                * (probe_i * dr)
                * dr
                * psi_probe
                * MANUFACTURED_EZ
            )
            manufactured_flux_error = abs(
                manufactured_flux / manufactured_flux_exact - 1.0
            )
            assert manufactured_flux > 0.0
            assert np.isclose(
                manufactured_flux,
                manufactured_flux_exact,
                rtol=MANUFACTURED_FLUX_REL_TOL,
                atol=MANUFACTURED_FLUX_ABS_TOL,
            )

            efield[2][probe_i, NZ - 1] = -MANUFACTURED_EZ
            reversed_flux = float(
                np.asarray(
                    warpx.staircase_charge_state(
                        corrector._psi_names,
                        corrector._weight_names,
                        insulating_endcaps=True,
                    ),
                    dtype=float,
                )[3, 0]
            )
            assert reversed_flux < 0.0
            assert np.isclose(
                reversed_flux,
                -manufactured_flux_exact,
                rtol=MANUFACTURED_FLUX_REL_TOL,
                atol=MANUFACTURED_FLUX_ABS_TOL,
            )

            efield[2][probe_i, 0] = MANUFACTURED_EZ
            efield[2][probe_i, NZ - 1] = MANUFACTURED_EZ
            balanced_flux = float(
                np.asarray(
                    warpx.staircase_charge_state(
                        corrector._psi_names,
                        corrector._weight_names,
                        insulating_endcaps=True,
                    ),
                    dtype=float,
                )[3, 0]
            )
            assert abs(balanced_flux) <= MANUFACTURED_ZERO_FLUX_ABS_TOL

            for field, saved in zip(efield, saved_zero):
                field.copymf(saved, 0, 0, 1, 0)
            assert valid_field_change(efield, saved_zero) <= STATE_PRESERVATION_ABS_TOL

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
            if args.insulating_endcaps:
                print(
                    "staircase insulating-endcap flux:",
                    f"B={manufactured_flux:+.16e} C,",
                    f"rel_error={manufactured_flux_error:.3e},",
                    f"reversed={reversed_flux:+.16e} C,",
                    f"balanced={balanced_flux:+.3e} C",
                )
    finally:
        if correct_callback is not None:
            callbacks.uninstallcallback("afterstep", correct_callback)
        if audit_callback is not None:
            callbacks.uninstallcallback("afterstep", audit_callback)
        # Python-owned MultiFab copies must be destroyed before AMReX tears
        # down its host/device arenas.  Keeping one alive is benign on CPU but
        # invalid on GPU because its destructor otherwise runs after finalize.
        correct_callback = audit_callback = None
        corrector = None
        initial_e = before_setup = saved_zero = difference = delta = None
        weight = efield = bfield = masks = fields = None
        field = saved = initial = None
        gc.collect()
        if initialized:
            simulation.finalize()


if __name__ == "__main__":
    main()
