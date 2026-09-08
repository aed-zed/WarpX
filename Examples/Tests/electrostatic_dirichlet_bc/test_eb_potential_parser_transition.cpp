/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ElectrostaticSolvers/PoissonBoundaryHandler.H"

#include <AMReX.H>
#include <AMReX_REAL.H>

#include <string>

namespace {

void checkTransition (PoissonBoundaryHandler& handler, std::string const& expression,
                      bool const expected_time_only, amrex::Real const expected_value)
{
    handler.setPotentialEB(expression);
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        handler.phi_EB_only_t == expected_time_only,
        "Incorrect embedded-boundary potential parser mode after setting " + expression);

#ifndef AMREX_USE_GPU
    constexpr amrex::Real x = 0.5;
    constexpr amrex::Real z = -0.75;
    constexpr amrex::Real time = 0.25;
    amrex::Real const value = expected_time_only
        ? handler.potential_eb_t(time)
#ifdef WARPX_DIM_RZ
        : handler.getPhiEB(time)(x, z);
#else
        : handler.getPhiEB(time)(x, 0.0, z);
#endif
    AMREX_ALWAYS_ASSERT(value == expected_value);
#else
    amrex::ignore_unused(expected_value);
#endif
}

} // namespace

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);
    {
        PoissonBoundaryHandler handler;
        checkTransition(handler, "4.0", true, 4.0);
        checkTransition(handler, "x+2.0", false, 2.5);
        checkTransition(handler, "0.0", true, 0.0);
        checkTransition(handler, "1.0+t", true, 1.25);
        checkTransition(handler, "z+3.0", false, 2.25);
    }
    amrex::Finalize();
    return 0;
}
