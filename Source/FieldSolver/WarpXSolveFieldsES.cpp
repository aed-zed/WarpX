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

using namespace amrex::literals;

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
    ABLASTR_PROFILE("WarpX::SolvePoissonEfield");

    using ablastr::fields::MultiLevelVectorField;
    using warpx::fields::FieldType;

    auto& es = GetElectrostaticSolver();
    int const nlevs = max_level + 1;

    amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho(nlevs);
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> phi(nlevs);
    amrex::IntVect const ng = get_ng_depos_rho();
    for (int lev = 0; lev < nlevs; ++lev) {
        amrex::BoxArray nodal_ba = boxArray(lev);
        nodal_ba.surroundingNodes();
        rho[lev] = std::make_unique<amrex::MultiFab>(
            nodal_ba, DistributionMap(lev), 1, ng);
        phi[lev] = std::make_unique<amrex::MultiFab>(
            nodal_ba, DistributionMap(lev), 1, 1);
        rho[lev]->setVal(0.0_rt);
        phi[lev]->setVal(0.0_rt);
    }

    mypc->DepositCharge(amrex::GetVecOfPtrs(rho), 0.0_rt);
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho_buf(nlevs);
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho_cp(nlevs);
    SyncRho(
        amrex::GetVecOfPtrs(rho), amrex::GetVecOfPtrs(rho_cp),
        amrex::GetVecOfPtrs(rho_buf));

#ifndef WARPX_DIM_RZ
    for (int lev = 0; lev < nlevs; ++lev) {
        ApplyRhofieldBoundary(lev, rho[lev].get(), PatchType::fine);
    }
#endif

    es.setPhiBC(amrex::GetVecOfPtrs(phi), gett_new(0));

    MultiLevelVectorField efield =
        m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, max_level);
    for (int lev = 0; lev < nlevs; ++lev) {
        for (int component = 0; component < 3; ++component) {
#ifdef WARPX_DIM_RZ
            if (component == 1) { continue; }
#endif
            efield[lev][component]->setVal(0.0_rt);
        }
    }

    std::array<amrex::Real, 3> const beta = {0.0_rt, 0.0_rt, 0.0_rt};
    if (EB::enabled()) {
        es.computePhi(
            amrex::GetVecOfPtrs(rho), amrex::GetVecOfPtrs(phi), beta,
            es.self_fields_required_precision, es.self_fields_absolute_tolerance,
            es.self_fields_max_iters, es.self_fields_verbosity,
            es.is_igf_2d_slices, efield);
    } else {
        es.computePhi(
            amrex::GetVecOfPtrs(rho), amrex::GetVecOfPtrs(phi), beta,
            es.self_fields_required_precision, es.self_fields_absolute_tolerance,
            es.self_fields_max_iters, es.self_fields_verbosity,
            es.is_igf_2d_slices);
        es.computeE(efield, amrex::GetVecOfPtrs(phi), beta);
    }

    for (int lev = 0; lev < nlevs; ++lev) {
        if (!m_fields.has(FieldType::phi_fp, lev)) { continue; }
        amrex::MultiFab* registered_phi = m_fields.get(FieldType::phi_fp, lev);
        if (registered_phi->boxArray() != phi[lev]->boxArray() ||
            registered_phi->DistributionMap() != phi[lev]->DistributionMap()) {
            continue;
        }
        amrex::IntVect const ng_copy =
            amrex::min(registered_phi->nGrowVect(), phi[lev]->nGrowVect());
        amrex::MultiFab::Copy(*registered_phi, *phi[lev], 0, 0, 1, ng_copy);
    }
}
