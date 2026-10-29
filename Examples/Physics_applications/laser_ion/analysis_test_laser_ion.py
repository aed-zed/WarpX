#!/usr/bin/env python3

import os
import sys

import numpy as np
import openpmd_api as io


def load_field_from_iteration(
    series, iteration: int, field: str, coord: str = None
) -> np.ndarray:
    """Load iteration of field data from file."""

    it = series.iterations[iteration]
    field_obj = it.meshes[f"{field}"]

    if field_obj.scalar:
        field_data = field_obj[io.Mesh_Record_Component.SCALAR].load_chunk()
    elif coord in [item[0] for item in list(field_obj.items())]:
        field_data = field_obj[coord].load_chunk()
    else:
        raise Exception(
            f"Specified coordinate: f{coord} is not available for field: f{field}."
        )
    series.flush()

    return field_data


def compare_time_avg_with_instantaneous_diags(dir_inst: str, dir_avg: str):
    """Compare instantaneous data (multiple iterations averaged in post-processing) with in-situ averaged data."""

    field = "E"
    coord = "z"
    avg_period_steps = 5
    avg_output_step = 100

    path_tpl_inst = f"{dir_inst}/openpmd_%T.h5"
    path_tpl_avg = f"{dir_avg}/openpmd_%T.h5"

    si = io.Series(path_tpl_inst, io.Access.read_only)
    sa = io.Series(path_tpl_avg, io.Access.read_only)

    ii0 = si.iterations[0]
    fi0 = ii0.meshes[field][coord]
    shape = fi0.shape

    data_inst = np.zeros(shape)

    for i in np.arange(avg_output_step - avg_period_steps + 1, avg_output_step + 1):
        data_inst += load_field_from_iteration(si, i, field, coord)

    data_inst = data_inst / avg_period_steps

    data_avg = load_field_from_iteration(sa, avg_output_step, field, coord)

    # Compare the data
    if np.allclose(data_inst, data_avg, rtol=1e-12):
        print("Test passed: actual data is close to expected data.")
    else:
        print("Test failed: actual data is not close to expected data.")
        sys.exit(1)


def compare_time_avg_with_original_run(dir_avg: str, dir_orig: str):
    """Compare in-situ averaged data of a restarted run with those of the original run."""

    field = "E"
    coord = "z"
    avg_output_step = 100

    sr = io.Series(f"{dir_avg}/openpmd_%T.h5", io.Access.read_only)
    so = io.Series(f"{dir_orig}/{dir_avg}/openpmd_%T.h5", io.Access.read_only)

    data_restart = load_field_from_iteration(sr, avg_output_step, field, coord)
    data_orig = load_field_from_iteration(so, avg_output_step, field, coord)

    if np.allclose(data_restart, data_orig, rtol=1e-12):
        print("Test passed: restarted data is close to original data.")
    else:
        print("Test failed: restarted data is not close to original data.")
        sys.exit(1)


if __name__ == "__main__":
    # TODO: implement intervals parser for PICMI that allows more complex output periods
    test_name = os.path.split(os.getcwd())[1]
    if test_name.endswith("_restart"):
        # The restart is within the averaging period of the time-averaged diagnostic
        compare_time_avg_with_original_run(
            dir_avg="diags/diagTimeAvg",
            dir_orig=os.getcwd().replace("_restart", ""),
        )
    elif "picmi" not in test_name:
        # Functionality test for TimeAveragedDiagnostics
        compare_time_avg_with_instantaneous_diags(
            dir_inst=sys.argv[1],
            dir_avg="diags/diagTimeAvg/",
        )
