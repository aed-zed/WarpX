#!/usr/bin/env python3
"""
Analysis for the RZ curl-preserving harmonic-bias correction (flux mode).

Verifies:
1. The corrector ran (Efield_correction fields present).
2. The cathode potential is maintained -- the geometry-agnostic flux-based
   effective voltage tracks the target and agrees with the independent
   axisymmetric line integral.
3. E_theta is preserved: the applied correction has a negligible azimuthal
   component (Efield_correction_theta ~ 0), i.e. neither the homogeneous clean
   nor the harmonic bias injects an inductive field -- the RZ curl-preservation
   check (the 3D bulk-curl footprint is not computed in RZ).
"""

import os

import numpy as np
import yt

target_delta_phi = -1.0e3
VOLTAGE_TOL = 0.20      # staircase-limited on a coarse RZ grid
AGREE_TOL = 0.15        # flux vs line integral
ETHETA_REL_TOL = 1.0e-6  # bias is theta-free by construction; ~round-off


def _read(path="poisson_correction_result.txt"):
    assert os.path.isfile(path), f"Result file not found: {path}"
    out = {}
    with open(path) as f:
        for line in f:
            key, _, value = line.partition("=")
            out[key.strip()] = float(value)
    return out


def main():
    diag_dir = "diags/diag1000100"
    assert os.path.isdir(diag_dir), f"Diagnostic output not found: {diag_dir}"
    print(f"Diagnostic directory found: {diag_dir}")

    # 1. Efield_correction fields written (corrector ran).
    header = os.path.join(diag_dir, "Header")
    with open(header) as f:
        htext = f.read()
    corr_fields = [ln for ln in htext.splitlines() if ln.startswith("Efield_correction")]
    assert len(corr_fields) == 3, f"Expected 3 Efield_correction fields, found {corr_fields}"
    print(f"Efield_correction fields present: {corr_fields}")

    # 2. Maintained potential: flux-based effective voltage tracks target and
    # agrees with the independent line integral.
    r = _read()
    line = r["FINAL_DELTA_PHI"]
    flux = r["FINAL_DELTA_PHI_FLUX"]
    flux_err = abs(flux - target_delta_phi) / abs(target_delta_phi)
    agree = abs(flux - line) / abs(target_delta_phi)
    print(
        f"delta_phi: flux = {flux:.1f} V (target err {flux_err:.1%}), "
        f"line = {line:.1f} V (flux-vs-line {agree:.1%})"
    )
    assert flux_err < VOLTAGE_TOL, (
        f"Flux-based potential not maintained: {flux:.1f} V vs {target_delta_phi:.1f} V"
    )
    assert agree < AGREE_TOL, (
        f"Flux and line-integral measurements disagree by {agree:.1%} of target"
    )

    # 3. E_theta preservation: the applied correction is azimuthally free.
    ds = yt.load(diag_dir)
    cg = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    corr_r = np.abs(np.array(cg["boxlib", "Efield_correction_r"])).max()
    corr_theta = np.abs(np.array(cg["boxlib", "Efield_correction_theta"])).max()
    rel = corr_theta / corr_r if corr_r > 0 else corr_theta
    print(
        f"Applied correction: max|corr_r| = {corr_r:.3e}, "
        f"max|corr_theta| = {corr_theta:.3e} (rel {rel:.2e})"
    )
    assert rel < ETHETA_REL_TOL, (
        f"Correction injects an azimuthal field (corr_theta/corr_r = {rel:.2e}): "
        "E_theta would not be preserved"
    )

    print("TEST PASSED")


if __name__ == "__main__":
    main()
