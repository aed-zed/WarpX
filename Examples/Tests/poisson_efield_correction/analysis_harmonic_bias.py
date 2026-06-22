#!/usr/bin/env python3
"""
Analysis for the curl-preserving harmonic-bias E-field correction (Phase 0).

Verifies that:
1. The simulation completed and produced diagnostic output.
2. The Efield_correction diagnostic fields are present (the corrector ran).
3. The electrode potential difference is maintained near the target value.
4. The applied correction is curl-free in the bulk -- i.e. the harmonic bias
   preserves the rotational/inductive field (the central Phase 0 claim). This
   is checked via the discrete curl footprint of the stored vacuum field,
   ``CURL_BULK_REL = max|curl(E_vac)|*dx / max|E_vac|``, away from the EB.
5. The final E-field magnitude in the gap is physical (not zeroed, not a
   boundary mask-step artifact).
"""

import os

import numpy as np
import yt

# Test parameters (must match PICMI_inputs_3d_harmonic_bias.py)
target_delta_phi = -1.0e3  # V, potential of inner electrode (outer at 0)
r_inner = 1.0e-2  # m
r_outer = 6.0e-2  # m

# Analytic field at the inner electrode surface of a concentric-cylinder
# capacitor: E(r) = delta_V / (r * ln(r_outer/r_inner)).
e_analytic_max = abs(target_delta_phi) / (r_inner * np.log(r_outer / r_inner))

# Curl-preservation tolerance. curl of a discrete gradient is ~machine epsilon
# in regular cells (independent of the MLMG solve tolerance); 1e-6 is a safe
# bound that still cleanly separates the curl-preserving bias (~1e-12) from a
# curl-destroying full replacement (~O(1) relative change).
CURL_BULK_TOL = 1.0e-6


def _read_results(path="poisson_correction_result.txt"):
    assert os.path.isfile(path), f"Result file not found: {path}"
    results = {}
    with open(path) as f:
        for line in f:
            key, _, value = line.partition("=")
            results[key.strip()] = float(value)
    return results


def main():
    # 1. Diagnostic output exists.
    diag_dir = "diags/diag1000100"
    assert os.path.isdir(diag_dir), f"Diagnostic output not found: {diag_dir}"
    print(f"Diagnostic directory found: {diag_dir}")

    # 2. Efield_correction fields written (proves the corrector executed).
    header_file = os.path.join(diag_dir, "Header")
    assert os.path.isfile(header_file), "Header file not found"
    with open(header_file) as f:
        header_text = f.read()
    for comp in ("x", "y", "z"):
        assert f"Efield_correction_{comp}" in header_text, (
            f"Efield_correction_{comp} not in diagnostic"
        )
    print("Efield_correction diagnostic fields are present")

    # 3. Reduced diagnostic PN exists with enough entries.
    pn_file = "diags/reducedfiles/PN.txt"
    assert os.path.isfile(pn_file), f"Reduced diagnostic not found: {pn_file}"
    pn_data = np.loadtxt(pn_file)
    assert pn_data.shape[0] >= 10, "Expected at least 10 diagnostic outputs"
    print(f"ParticleNumber diagnostic has {pn_data.shape[0]} entries")

    results = _read_results()

    # 4a. Potential maintained near target (line integral through the
    # staircased EB carries a few-percent discretization error; 15% cleanly
    # detects a lost/zeroed field).
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

    # 4a'. Phase A: when the run used the geometry-agnostic induced-charge flux
    # to drive the feedback, cross-validate it against the independent
    # axisymmetric line integral (both should track the target). Only checked
    # when the run wrote FINAL_DELTA_PHI_FLUX (i.e. ran in flux mode).
    if "FINAL_DELTA_PHI_FLUX" in results:
        delta_phi_flux = results["FINAL_DELTA_PHI_FLUX"]
        flux_err = abs(delta_phi_flux - target_delta_phi) / abs(target_delta_phi)
        agree = abs(delta_phi_flux - delta_phi) / abs(target_delta_phi)
        print(
            f"Flux-based delta_phi = {delta_phi_flux:.1f} V (target rel err {flux_err:.1%}; "
            f"vs line integral {agree:.1%})"
        )
        assert flux_err < 0.15, (
            f"Flux measurement does not track the target: {delta_phi_flux:.1f} V "
            f"vs {target_delta_phi:.1f} V"
        )
        assert agree < 0.10, (
            f"Flux and line-integral measurements disagree by {agree:.1%} of target "
            f"(flux {delta_phi_flux:.1f} V, line {delta_phi:.1f} V)"
        )

    # 4b. The harmonic bias is curl-free in the bulk (rotational field preserved).
    curl_bulk_rel = results["CURL_BULK_REL"]
    curl_max_rel = results.get("CURL_MAX_REL", float("nan"))
    print(
        f"Curl footprint of vacuum bias: bulk_rel = {curl_bulk_rel:.3e} "
        f"(tol {CURL_BULK_TOL:.0e}), max_rel = {curl_max_rel:.3e} (incl. EB cut cells)"
    )
    assert curl_bulk_rel < CURL_BULK_TOL, (
        f"Bias is not curl-free in the bulk (bulk_rel={curl_bulk_rel:.3e} >= "
        f"{CURL_BULK_TOL:.0e}): the correction would perturb the rotational field"
    )

    # 5. Final E-field magnitude in the electrode gap is physical.
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
