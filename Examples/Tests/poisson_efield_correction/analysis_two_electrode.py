#!/usr/bin/env python3
"""
Analysis for the Phase B two-electrode test.

Verifies the multi-electrode capability that the single bias mode cannot
provide: two embedded-boundary electrodes held at *distinct* potentials at
once. Checks (from two_electrode_result.txt):

1. The vacuum capacitance matrix is well-conditioned (the per-electrode unit
   fields are independent -- the regions actually separate the electrodes).
2. Each electrode's effective voltage, recovered via V = C^-1 (Q - Q_g), tracks
   its own distinct target.
3. The two targets are genuinely different and both maintained (so a single
   global scale could not satisfy both).
"""

import os

V_left_target = 300.0
V_right_target = -700.0
VOLTAGE_TOL = 0.15  # relative; staircased-EB flux + coarse grid


def _read(path="two_electrode_result.txt"):
    assert os.path.isfile(path), f"Result file not found: {path}"
    out = {}
    with open(path) as f:
        for line in f:
            key, _, value = line.partition("=")
            out[key.strip()] = float(value)
    return out


def main():
    r = _read()

    # 1. Capacitance matrix well-conditioned.
    cond = r["CAP_COND"]
    print(f"Capacitance matrix condition number: {cond:.3e}")
    assert cond < 1.0e6, (
        f"Capacitance matrix ill-conditioned (cond={cond:.3e}): electrode "
        "regions may overlap or not separate the electrodes."
    )

    # 2. Each electrode tracks its own target.
    vl, vr = r["V_LEFT_FINAL"], r["V_RIGHT_FINAL"]
    el = abs(vl - V_left_target) / abs(V_left_target)
    er = abs(vr - V_right_target) / abs(V_right_target)
    print(
        f"Left  electrode: {vl:8.1f} V (target {V_left_target:.1f} V, err {el:.1%})"
    )
    print(
        f"Right electrode: {vr:8.1f} V (target {V_right_target:.1f} V, err {er:.1%})"
    )
    assert el < VOLTAGE_TOL, f"Left electrode not maintained ({vl:.1f} V)"
    assert er < VOLTAGE_TOL, f"Right electrode not maintained ({vr:.1f} V)"

    # 3. The two maintained potentials are genuinely distinct (a single global
    # bias scale could not hold both). Separation >> the per-electrode error.
    separation = abs(vl - vr)
    target_sep = abs(V_left_target - V_right_target)
    print(f"Maintained separation: {separation:.1f} V (target {target_sep:.1f} V)")
    assert separation > 0.5 * target_sep, (
        "The two electrodes collapsed toward a single potential -- the "
        "multi-electrode control is not separating them."
    )

    print("TEST PASSED")


if __name__ == "__main__":
    main()
