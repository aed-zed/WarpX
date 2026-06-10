#!/usr/bin/env python3
"""
Analysis script for the Poisson E-field correction test.

Verifies that:
1. The simulation completed and produced diagnostic output.
2. The Efield_rot diagnostic fields are present (Poisson correction ran).
3. The electrode potential difference is maintained near the target value
   (the core purpose of the correction).
4. The E-field has the physically expected magnitude in the gap between
   the electrodes (guards against the field being zeroed or replaced by
   a boundary-mask artifact).
"""

import os
import sys

import numpy as np
import yt

# Test parameters (must match PICMI_inputs_3d.py)
target_delta_phi = -1.0e3  # V, potential of inner electrode (outer at 0)
r_inner = 1.0e-2  # m
r_outer = 6.0e-2  # m

# Analytic field at the inner electrode surface of a concentric-cylinder
# capacitor: E(r) = delta_V / (r * ln(r_outer/r_inner))
e_analytic_max = abs(target_delta_phi) / (r_inner * np.log(r_outer / r_inner))


def main():
    # Check diagnostic output exists
    diag_dir = "diags/diag1000100"
    assert os.path.isdir(diag_dir), f"Diagnostic output not found: {diag_dir}"
    print(f"Diagnostic directory found: {diag_dir}")

    # Check that the E_rot fields are written (proves the Poisson
    # correction callback executed and registered diagnostic fields)
    header_file = os.path.join(diag_dir, "Header")
    assert os.path.isfile(header_file), "Header file not found"
    with open(header_file) as f:
        header_text = f.read()
    assert "Efield_rot_x" in header_text, "Efield_rot_x not in diagnostic"
    assert "Efield_rot_y" in header_text, "Efield_rot_y not in diagnostic"
    assert "Efield_rot_z" in header_text, "Efield_rot_z not in diagnostic"
    print("E_rot diagnostic fields are present")

    # Check the reduced diagnostic PN file exists
    pn_file = "diags/reducedfiles/PN.txt"
    assert os.path.isfile(pn_file), f"Reduced diagnostic not found: {pn_file}"
    pn_data = np.loadtxt(pn_file)
    assert pn_data.shape[0] >= 10, "Expected at least 10 diagnostic outputs"
    print(f"ParticleNumber diagnostic has {pn_data.shape[0]} entries")

    # Check the maintained potential difference written by the run script.
    # The line integral through the staircased EB on this coarse grid
    # carries a discretization error of a few percent; 15% tolerance
    # still cleanly detects the failure mode of a lost/zeroed field.
    result_file = "poisson_correction_result.txt"
    assert os.path.isfile(result_file), f"Result file not found: {result_file}"
    results = {}
    with open(result_file) as f:
        for line in f:
            key, _, value = line.partition("=")
            results[key.strip()] = float(value)
    delta_phi = results["FINAL_DELTA_PHI"]
    rel_err = abs(delta_phi - target_delta_phi) / abs(target_delta_phi)
    print(
        f"Final delta_phi = {delta_phi:.1f} V (target {target_delta_phi:.1f} V, "
        f"relative error {rel_err:.1%})"
    )
    assert rel_err < 0.15, (
        f"Electrode potential not maintained: measured {delta_phi:.1f} V, "
        f"target {target_delta_phi:.1f} V"
    )

    # Check the final E-field magnitude in the electrode gap. The analytic
    # peak (at the inner electrode) is ~5.6e4 V/m; the staircased solution
    # on this grid resolves ~5e4 V/m. A field below half the analytic value
    # indicates the correction zeroed the field; a field far above it
    # indicates a mask-step artifact (e.g. inverted EB geometry produced
    # |E| = delta_V/dx ~ 2.7e5 V/m).
    ds = yt.load(diag_dir)
    cg = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    ex = np.array(cg["boxlib", "Ex"])
    ey = np.array(cg["boxlib", "Ey"])
    e_max = float(np.sqrt(ex**2 + ey**2).max())
    print(f"Final max |E_xy| = {e_max:.3e} V/m (analytic peak {e_analytic_max:.3e} V/m)")
    assert e_max > 0.5 * e_analytic_max, (
        f"E-field too small ({e_max:.3e} V/m): correction may have zeroed the field"
    )
    assert e_max < 2.0 * e_analytic_max, (
        f"E-field too large ({e_max:.3e} V/m): possible EB mask-step artifact"
    )

    print("TEST PASSED")


if __name__ == "__main__":
    main()
