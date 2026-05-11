#!/usr/bin/env python3
"""
Analysis script for the Poisson E-field correction test.

Reads the simulation stdout to extract the final potential difference
and verifies it stays within tolerance of the target.
"""

import re
import sys


def main():
    # Read stdout that was piped to a file by CTest
    # The PICMI script prints FINAL_DELTA_PHI and TARGET_DELTA_PHI at the end
    stdout_text = sys.stdin.read() if not sys.argv[1:] else open(sys.argv[1]).read()

    # Extract values from output
    match_final = re.search(r"FINAL_DELTA_PHI=([-+\d.eE]+)", stdout_text)
    match_target = re.search(r"TARGET_DELTA_PHI=([-+\d.eE]+)", stdout_text)

    if match_final is None or match_target is None:
        # Try reading from the default output file
        import os
        out_files = [f for f in os.listdir(".") if f.endswith(".out") or f == "stdout"]
        for fname in out_files:
            with open(fname) as f:
                text = f.read()
            if match_final is None:
                match_final = re.search(r"FINAL_DELTA_PHI=([-+\d.eE]+)", text)
            if match_target is None:
                match_target = re.search(r"TARGET_DELTA_PHI=([-+\d.eE]+)", text)

    assert match_final is not None, "Could not find FINAL_DELTA_PHI in output"
    assert match_target is not None, "Could not find TARGET_DELTA_PHI in output"

    delta_phi_final = float(match_final.group(1))
    delta_phi_target = float(match_target.group(1))

    # Allow 20% relative tolerance — the correction is applied every 10 steps
    # so some drift occurs between corrections
    rel_error = abs(delta_phi_final - delta_phi_target) / abs(delta_phi_target)

    print(f"Target Delta-V: {delta_phi_target:.4e}")
    print(f"Final  Delta-V: {delta_phi_final:.4e}")
    print(f"Relative error: {rel_error:.4f}")

    tolerance = 0.20
    assert rel_error < tolerance, (
        f"Potential difference drifted too far from target: "
        f"rel_error={rel_error:.4f} > {tolerance}"
    )

    print("TEST PASSED")


if __name__ == "__main__":
    main()
