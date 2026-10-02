/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_NODAL_CHARGE_MEASUREMENT_H_
#define WARPX_NODAL_CHARGE_MEASUREMENT_H_

#include <AMReX_MultiFab.H>
#include <AMReX_REAL.H>

#include <memory>

namespace warpx::electrostatic
{
    /** The axis measure that makes sum rho V equal the deposited charge. */
    amrex::Real DepositionAxisFactor ();

    /** The axis node's geometric control-volume factor, for a Gauss-law integral. */
    constexpr amrex::Real GaussAxisFactor () { return amrex::Real(0.25); }

    /** Nodal inner product sum_a f[a] g[a] V_a with unique MPI ownership. */
    amrex::Real IntegrateRhoPsi (
        amrex::MultiFab const& rho,
        amrex::MultiFab const& psi,
        int lev,
        amrex::Real axis_factor);

    /** Nodal divergence of Efield_fp, on the level's nodal BoxArray. */
    std::unique_ptr<amrex::MultiFab> NodalDivEFromFp (int lev);
}

#endif // WARPX_NODAL_CHARGE_MEASUREMENT_H_
