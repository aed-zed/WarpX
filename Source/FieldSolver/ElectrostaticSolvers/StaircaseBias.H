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
    int max_iter,
    bool insulating_endcaps = false,
    bool grounded_wall_reference = false);

/** Build an insulating-endcap observer and a distinct fringing-field actuator.
 *
 * ``out_psi`` is the homogeneous-Neumann harmonic weighting potential used by
 * the charge observer; it is not the actuator potential.  ``out_efield`` is
 * computed from a second harmonic potential whose two endpoint traces solve the
 * exact one-dimensional RZ nodal resistance network between fixed metal nodes.
 * All output fields must already be registered.  Returns the larger absolute
 * residual of the observer and actuator solves.
 */
amrex::Real WarpXSolveStaircaseInsulatorBias (
    std::string const& selector,
    std::string const& out_psi,
    std::string const& out_weight,
    std::string const& out_efield,
    amrex::Real rtol,
    int max_iter,
    bool grounded_wall_reference = false);

/** Return the maximum fixed-node error in the sum of all unit weights.
 * With grounded_wall_reference, components joined by frozen edges to the
 * outer PEC radius require zero weight; all other components require one.
 * This setup-only RZ audit replicates the graph on each host rank.
 */
amrex::Real WarpXValidateStaircaseWeights (
    std::vector<std::string> const& weight_fields,
    bool grounded_wall_reference = false);

/** Research staircase observer: total Gauss charge, deposited live charge on
 * fixed electrode nodes, and the all-node grounded pairing, respectively.
 * Requires matching registered nodal unit potentials and fixed-node weights.
 * Rejects deposited charge on an unselected RZ axis node when deposition and
 * geometric axis measures differ. Collective; no Poisson solve.
 */
amrex::Vector<amrex::Vector<amrex::Real>>
WarpXStaircaseChargeState (std::vector<std::string> const& psi_fields,
                          std::vector<std::string> const& weight_fields,
                          bool insulating_endcaps = false);

#endif // WARPX_STAIRCASE_BIAS_H_
