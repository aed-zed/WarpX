/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Remi Lehe, Roelof Groenewald, Arianna Formenti, Revathi Jambunathan
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.H"

#include "Fields.H"
#include "Fluids/MultiFluidContainer.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"

#include <ablastr/profiler/ProfilerWrapper.H>

void WarpX::ComputeSpaceChargeField (bool const reset_E_field, bool const reset_B_field,
                                     bool const verbose_step)
{
    ABLASTR_PROFILE("WarpX::ComputeSpaceChargeField");
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    // Reset E and B fields to 0, before calculating space-charge fields if requested
    for (int lev = 0; lev <= max_level; lev++) {
        for (int comp=0; comp<3; comp++) {
            if (reset_E_field) {
                m_fields.get(FieldType::Efield_fp, Direction{comp}, lev)->setVal(0);
            }
            if (reset_B_field) {
                m_fields.get(FieldType::Bfield_fp, Direction{comp}, lev)->setVal(0);
            }
        }
    }

    m_electrostatic_solver->ComputeSpaceChargeField(
        m_fields, *mypc, myfl.get(), max_level, verbose_step);
}

std::unique_ptr<amrex::MultiFab> WarpX::DepositScratchRho (int const lev)
{
    ABLASTR_PROFILE("WarpX::DepositScratchRho");
    using namespace amrex::literals;

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(finest_level == 0,
        "DepositScratchRho is only implemented for a single level");
    // RZ deposition writes 2*nmodes-1 components; this scratch density has one.
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::ncomps == 1,
        "DepositScratchRho supports a single RZ azimuthal mode only");

    amrex::BoxArray nodal_ba = boxArray(lev);
    nodal_ba.surroundingNodes();
    auto rho = std::make_unique<amrex::MultiFab>(
        nodal_ba, DistributionMap(lev), 1, get_ng_depos_rho());

    // Same order as LabFrameExplicitES: particles (zeroing rho, with the RZ
    // inverse-volume scaling), fluids, then filter and guard-cell sum.
    amrex::Vector<amrex::MultiFab*> const rho_lev{rho.get()};
    mypc->DepositCharge(rho_lev, 0._rt);
    if (do_fluid_species) {
        myfl->DepositCharge(m_fields, *rho, lev);
    }

    amrex::Vector<std::unique_ptr<amrex::MultiFab>> const no_coarse_patch(1);
    SyncRho(rho_lev, amrex::GetVecOfPtrs(no_coarse_patch),
            amrex::GetVecOfPtrs(no_coarse_patch));

#ifndef WARPX_DIM_RZ
    // Reflect the density over PEC boundaries, if needed.
    ApplyRhofieldBoundary(lev, rho.get(), PatchType::fine);
#endif

    return rho;
}
