/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_EB_VOLTAGE_CLAMP_H_
#define WARPX_EB_VOLTAGE_CLAMP_H_

#include <AMReX_REAL.H>

#include <string>
#include <utility>
#include <vector>

/* Voltage clamp for embedded-boundary conductors in electromagnetic runs
 * (RZ m=0, explicit staggered Yee, one level; periodic, PEC or PMC z faces).
 *
 * The conductors are the components of WarpX's frozen-edge staircase. The fixed
 * nodes are the endpoints of frozen E edges. Conductor k has a unit potential
 * psi_k (1 on its fixed nodes, 0 on the others, harmonic elsewhere) and a unit
 * field E_k = -grad psi_k. The Python driver (pywarpx.staircase_bias_corrector)
 * measures the conductor voltages and adds sum_k dV_k E_k to E.
 * See Docs/source/usage/workflows/electrode_voltage_clamp.rst.
 *
 * The functions register their output fields as voltage_clamp_{phi,weight,E}_<k>,
 * or with suffix "grounded", on level 0. All functions are collective over MPI.
 */
namespace warpx::embedded_boundary::voltage_clamp
{
    /** Name of the unit field E_k (k >= 0) or of the grounded field (k < 0).
     *
     * \param[in] k conductor index, or negative for the grounded field
     */
    std::string EFieldName (int k);

    /** Solve for the unit potential and field of conductor k.
     *
     * \param[in] k              conductor index, starting at 0
     * \param[in] selector       parser expression in (x, y, z), 1 on the fixed nodes
     *                           of conductor k and 0 on all other fixed nodes
     * \param[in] rtol           relative tolerance of the MLMG solve
     * \param[in] max_iter       maximum number of MLMG iterations
     * \param[in] wall_reference allow unselected conductors to touch the PEC wall:
     *                           they belong to the ground reference
     * \return final MLMG residual
     */
    amrex::Real SolveUnitPotential (int k, std::string const& selector,
                                    amrex::Real rtol, int max_iter,
                                    bool wall_reference = false);

    /** Solve with every conductor grounded and the live charge as source.
     *
     * Reading the result with GaussCharge gives the charge the plasma induces on
     * the conductors, an independent check of the reciprocity value -psi_k^T q.
     *
     * \param[in] rtol           relative tolerance of the MLMG solve
     * \param[in] max_iter       maximum number of MLMG iterations
     * \param[in] wall_reference treat conductors joined to the outer PEC radius as
     *                           ground, as in the unit solves
     * \return final MLMG residual
     */
    amrex::Real SolveGrounded (amrex::Real rtol, int max_iter, bool wall_reference = false);

    /** Abort unless every fixed node belongs to exactly one of the n conductors, or,
     *  with wall_reference, is frozen-edge connected to the outer PEC radius (the
     *  ground reference, two PMC z faces required). Call once after the unit solves.
     *
     * \param[in] n              number of conductors
     * \param[in] wall_reference treat conductors joined to the outer PEC radius as
     *                           ground (requires two PMC z faces)
     */
    void CheckSelection (int n, bool wall_reference = false);

    /** Gauss charge eps0 sum_{fixed nodes of k} V div E of each conductor k.
     *
     * \param[in] n           number of conductors
     * \param[in] efield_name registered name of the E field to read
     * \return charge of each conductor, in coulombs
     */
    std::vector<amrex::Real> GaussCharge (int n, std::string const& efield_name);

    /** Capacitance matrix C[j][k]: charge that the observer reads on conductor j
     *  for the unit field E_k. Call after CheckSelection.
     *
     * \param[in] n number of conductors
     * \return C[j][k], in farads
     */
    std::vector<std::vector<amrex::Real>> CapacitanceMatrix (int n);

    /** Charge state of the live fields, from one deposition and one divergence:
     *  the field charge eps0 s_k^T V div E - s_k^T V q of each conductor, and the
     *  grounded (plasma-induced) charge -psi_k^T V q. The voltages are
     *  C^-1 (field charge - grounded charge).
     *
     * \param[in] n number of conductors
     * \return field charge and grounded charge of each conductor, in coulombs
     */
    std::pair<std::vector<amrex::Real>, std::vector<amrex::Real>>
    ChargeState (int n);

    /** E_fp += sum_k dv[k] E_k, then refresh the guard cells of E_fp.
     *
     * \param[in] dv voltage change of each conductor, in volts
     */
    void AddUnitFields (std::vector<amrex::Real> const& dv);
}

#endif // WARPX_EB_VOLTAGE_CLAMP_H_
