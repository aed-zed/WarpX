/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Remi Lehe, Roelof Groenewald, Arianna Formenti, Revathi Jambunathan
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "WarpX.H"

#include <ablastr/profiler/ProfilerWrapper.H>

void WarpX::ComputeSpaceChargeField (bool const reset_E_field, bool const reset_B_field)
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
        m_fields, *mypc, myfl.get(), max_level );
}

void WarpX::SolvePoissonEfield ()
{
    WARPX_PROFILE("WarpX::SolvePoissonEfield");

    using ablastr::fields::Direction;
    using ablastr::fields::MultiLevelScalarField;
    using ablastr::fields::MultiLevelVectorField;
    using warpx::fields::FieldType;

    auto& es = GetElectrostaticSolver();
    const int nlevs = max_level + 1;

    // Allocate temporary rho and phi MultiFabs
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho(nlevs);
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> phi(nlevs);
    const amrex::IntVect ng = get_ng_depos_rho();
    for (int lev = 0; lev < nlevs; lev++) {
        amrex::BoxArray nba = boxArray(lev);
        nba.surroundingNodes();
        rho[lev] = std::make_unique<amrex::MultiFab>(
            nba, DistributionMap(lev), 1, ng);
        rho[lev]->setVal(0.);
        phi[lev] = std::make_unique<amrex::MultiFab>(
            nba, DistributionMap(lev), 1, 1);
        phi[lev]->setVal(0.);
    }

    // Deposit charge from all particle species
    mypc->DepositCharge(amrex::GetVecOfPtrs(rho), 0.0_rt);

    // Sync rho: apply filter, MPI exchange, interpolate across MR levels
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho_buf(nlevs);
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho_cp(nlevs);
    SyncRho(amrex::GetVecOfPtrs(rho),
            amrex::GetVecOfPtrs(rho_cp),
            amrex::GetVecOfPtrs(rho_buf));

#ifndef WARPX_DIM_RZ
    for (int lev = 0; lev < nlevs; lev++) {
        ApplyRhofieldBoundary(lev, rho[lev].get(), PatchType::fine);
    }
#endif

    // Set boundary potentials
    es.setPhiBC(amrex::GetVecOfPtrs(phi), gett_new(0));

    // Zero Efield_fp before solving (computePhi/computeE ADD to E)
    MultiLevelVectorField Efield_fp =
        m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, max_level);
    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            Efield_fp[lev][comp]->setVal(0.);
        }
    }

    // Solve Poisson and compute E
    const std::array<amrex::Real, 3> beta = {0._rt, 0._rt, 0._rt};
    if (EB::enabled()) {
        // With EB: pass Efield to computePhi for EB-aware E computation
        es.computePhi(amrex::GetVecOfPtrs(rho), amrex::GetVecOfPtrs(phi),
                      beta, es.self_fields_required_precision,
                      es.self_fields_absolute_tolerance,
                      es.self_fields_max_iters, es.self_fields_verbosity,
                      es.is_igf_2d_slices, Efield_fp);
    } else {
        es.computePhi(amrex::GetVecOfPtrs(rho), amrex::GetVecOfPtrs(phi),
                      beta, es.self_fields_required_precision,
                      es.self_fields_absolute_tolerance,
                      es.self_fields_max_iters, es.self_fields_verbosity,
                      es.is_igf_2d_slices);
        es.computeE(Efield_fp, amrex::GetVecOfPtrs(phi), beta);
    }
}
