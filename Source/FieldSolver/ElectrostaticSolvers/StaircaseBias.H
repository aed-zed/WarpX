/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_STAIRCASE_BIAS_H_
#define WARPX_STAIRCASE_BIAS_H_

#include <AMReX_REAL.H>
#include <AMReX_Vector.H>

#include <string>
#include <vector>

/** Solve one charge-free unit-potential problem on the frozen-edge staircase graph.
 *
 * All output fields must already be registered.  The scalar fields are nodal;
 * ``out_efield`` has the native staggered E layouts.  The returned value is
 * MLMG's final absolute residual.
 */
amrex::Real WarpXSolveStaircaseUnitBias (
    std::string const& selector,
    std::string const& out_phi,
    std::string const& out_weight,
    std::string const& out_efield,
    amrex::Real rtol,
    int max_iter);

/** Return the maximum fixed-node error in the sum of all unit weights. */
amrex::Real WarpXValidateStaircaseWeights (
    std::vector<std::string> const& weight_fields);

/** Research staircase observer: total Gauss charge, deposited live charge on
 * fixed electrode nodes, and the all-node grounded pairing, respectively.
 * Requires matching registered nodal unit potentials and fixed-node weights.
 * Rejects deposited charge on an unselected RZ axis node when deposition and
 * geometric axis measures differ. Collective; no Poisson solve.
 */
amrex::Vector<amrex::Vector<amrex::Real>>
WarpXStaircaseChargeState (std::vector<std::string> const& psi_fields,
                          std::vector<std::string> const& weight_fields);

#endif // WARPX_STAIRCASE_BIAS_H_
