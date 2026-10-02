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

/* Staircase voltage clamp (RZ, explicit Yee, embedded boundaries).
 * Method and supported configurations: Docs/source/usage/workflows/electrode_voltage_clamp.rst
 * All output fields are registered by the caller; scalars are nodal, E outputs use
 * the native staggered E layouts. Collective over MPI.
 */

/** \brief Unit-potential (harmonic) solve on the frozen-edge staircase.
 *
 * Unit potential on the conductor selected by `selector`, zero on all other fixed
 * nodes. Fills the potential, the fixed-node weight and E = -grad(phi).
 * \return final MLMG residual
 */
amrex::Real WarpXSolveStaircaseUnitBias (
    std::string const& selector,
    std::string const& out_phi,
    std::string const& out_weight,
    std::string const& out_efield,
    amrex::Real rtol,
    int max_iter);

/** \brief Grounded staircase Poisson solve with the live charge as source.
 *
 * All conductors and the PEC outer radius at 0 V. Reading the resulting E with the
 * staircase observer gives the plasma-induced conductor charge, an independent
 * check of the reciprocity value -psi^T q.
 * \return final MLMG residual
 */
amrex::Real WarpXSolveStaircaseGrounded (
    std::string const& out_phi, std::string const& out_weight,
    std::string const& out_efield, amrex::Real rtol, int max_iter);

/** \brief Setup check: maximum error of the summed unit weights on fixed nodes.
 *
 * Each conductor must be selected exactly once.
 */
amrex::Real WarpXValidateStaircaseWeights (
    std::vector<std::string> const& weight_fields);

/** \brief Staircase observer, one entry per conductor.
 *
 * Returns {total Gauss charge eps0 s^T V div(E), live charge s^T q on the fixed
 * nodes, grounded charge -psi^T q}. Deposits charge once; no Poisson solve.
 */
amrex::Vector<amrex::Vector<amrex::Real>>
WarpXStaircaseChargeState (std::vector<std::string> const& psi_fields,
                          std::vector<std::string> const& weight_fields);

#endif // WARPX_STAIRCASE_BIAS_H_
