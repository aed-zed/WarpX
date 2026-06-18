#!/usr/bin/env python3
"""
Analysis script for the RZ Poisson E-field correction test.

Verifies that:
1. The simulation completed and produced diagnostic output.
2. The Efield_correction diagnostic fields are present (Poisson correction ran).
3. The electrode potential difference is maintained near the target value
   (the core purpose of the correction).
4. The radial E-field has the physically expected magnitude in the gap.
"""

import os
import sys

import numpy as np
import yt

# Test parameters (must match PICMI_inputs_rz.py)
target_delta_phi = -1.0e3  # V, potential of inner electrode (outer at 0)
r_inner = 1.0e-2  # m
r_outer = 6.0e-2  # m

# Analytic field at the inner electrode surface of a concentric-cylinder
# capacitor: E(r) = delta_V / (r * ln(r_outer/r_inner))
e_analytic_max = abs(target_delta_phi) / (r_inner * np.log(r_outer / r_inner))


def main():
    diag_dir = "diags/diag1000100"
    assert os.path.isdir(diag_dir), f"Diagnostic output not found: {diag_dir}"
    print(f"Diagnostic directory found: {diag_dir}")

    # Efield_correction fields prove the Poisson correction callback executed.
    # In RZ the component suffixes are r/theta/z.
    header_file = os.path.join(diag_dir, "Header")
    assert os.path.isfile(header_file), "Header file not found"
    with open(header_file) as f:
        header_text = f.read()
    corr_fields = [ln for ln in header_text.splitlines() if ln.startswith("Efield_correction")]
    assert len(corr_fields) == 3, f"Expected 3 Efield_correction fields, found {corr_fields}"
    print(f"Efield_correction diagnostic fields present: {corr_fields}")

    pn_file = "diags/reducedfiles/PN.txt"
    assert os.path.isfile(pn_file), f"Reduced diagnostic not found: {pn_file}"
    pn_data = np.loadtxt(pn_file)
    assert pn_data.shape[0] >= 10, "Expected at least 10 diagnostic outputs"
    print(f"ParticleNumber diagnostic has {pn_data.shape[0]} entries")

    # Maintained potential difference (dimension-independent physics check).
    # The radial line integral through the staircased EB on this coarse grid
    # carries a discretization error of a few percent; 20% still cleanly
    # detects a lost/zeroed field.
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
    assert rel_err < 0.20, (
        f"Electrode potential not maintained: measured {delta_phi:.1f} V, "
        f"target {target_delta_phi:.1f} V"
    )

    # Final radial E-field magnitude. Guards against a zeroed field (lower
    # bound) and against an EB mask-step artifact delta_V/dx (upper bound).
    ds = yt.load(diag_dir)
    cg = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    er = np.array(cg["boxlib", "Er"])
    e_max = float(np.abs(er).max())
    print(f"Final max |Er| = {e_max:.3e} V/m (analytic peak {e_analytic_max:.3e} V/m)")
    assert e_max > 0.5 * e_analytic_max, (
        f"E-field too small ({e_max:.3e} V/m): correction may have zeroed the field"
    )
    assert e_max < 2.0 * e_analytic_max, (
        f"E-field too large ({e_max:.3e} V/m): possible EB mask-step artifact"
    )

    print("TEST PASSED")


if __name__ == "__main__":
    main()
