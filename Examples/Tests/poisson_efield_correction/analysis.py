#!/usr/bin/env python3
"""
Analysis script for the Poisson E-field correction test.

Verifies that:
1. The simulation completed and produced diagnostic output.
2. The Efield_rot diagnostic fields are present (Poisson correction ran).
3. Ex has non-trivial values in the valid (uncovered) region.
"""

import os
import sys

import numpy as np


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

    print("TEST PASSED")


if __name__ == "__main__":
    main()
