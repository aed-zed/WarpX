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
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXConst.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <ablastr/warn_manager/WarnManager.H>


void WarpX::ComputeSpaceChargeField (bool const reset_fields)
{
    WARPX_PROFILE("WarpX::ComputeSpaceChargeField");
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    if (reset_fields) {
        // Reset all E and B fields to 0, before calculating space-charge fields
        WARPX_PROFILE("WarpX::ComputeSpaceChargeField::reset_fields");
        for (int lev = 0; lev <= max_level; lev++) {
            for (int comp=0; comp<3; comp++) {
                m_fields.get(FieldType::Efield_fp, Direction{comp}, lev)->setVal(0);
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

    // The correction subtracts a pure gradient from E (see implementation
    // report, Section 10). This is only consistent on a STAGGERED (Yee) grid:
    // the EB-aware gradient it subtracts is edge-centered and matches Yee
    // Efield_fp. On a collocated/nodal grid the centering is inconsistent and
    // the correction is not supported.
    if (WarpX::grid_type == ablastr::utils::enums::GridType::Collocated) {
        ablastr::warn_manager::WMRecordWarning(
            "Poisson E-field correction",
            "SolvePoissonEfield assumes a staggered (Yee) grid: the EB-aware "
            "gradient it subtracts is edge-centered and matches Yee Efield_fp. "
            "On a collocated/nodal grid the centering is inconsistent and the "
            "correction is not supported. Use warpx.grid_type = staggered.",
            ablastr::warn_manager::WarnPriority::high);
    }
    // The Helmholtz-consistent correction removes exactly the Gauss-law residual
    // div(E) - rho/eps0, which is also the source of the hyperbolic div(E)
    // cleaning field F. Enabling both double-corrects Gauss's law.
    if (WarpX::do_dive_cleaning) {
        ablastr::warn_manager::WMRecordWarning(
            "Poisson E-field correction",
            "SolvePoissonEfield and div(E) cleaning (warpx.do_dive_cleaning) both "
            "act on the Gauss-law residual div(E) - rho/eps0; enabling both "
            "double-corrects Gauss's law. Disable one.",
            ablastr::warn_manager::WarnPriority::low);
    }

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

    // Set boundary potentials (electrode values V_k).
    es.setPhiBC(amrex::GetVecOfPtrs(phi), gett_new(0));

    // Replace the irrotational (poloidal) field with the Poisson solution, but
    // PRESERVE the azimuthal field E_theta in RZ.
    //
    // In RZ axisymmetric (m=0) the electrostatic solve produces no azimuthal
    // component ((grad phi)_theta = 0), so E_theta is purely solenoidal --
    // it carries the device's E×B-driven inductive field. Overwriting it would
    // zero that field every correction and force a slow multi-step rebuild
    // (and radiate a transient). We therefore leave Direction 1 (theta)
    // untouched: only Er and Ez are re-solved.
    //
    // This is the boundary-condition-robust, leading-order special case of the
    // gradient-only correction in report Section 10. The fully general version
    // (preserving the poloidal inductive part as well) requires a Helmholtz/
    // Leray projection with a careful boundary gauge -- a naive single Poisson
    // solve over-corrects by a harmonic gradient -- and is deferred (Section 10).
    MultiLevelVectorField Efield_fp =
        m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, max_level);
    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
#ifdef WARPX_DIM_RZ
            // Preserve E_theta (inductive); the EB solve writes only Er, Ez.
            if (comp == 1) { continue; }
#endif
            Efield_fp[lev][comp]->setVal(0.);
        }
    }

    // Solve Poisson and compute E (adds -grad(phi) into the zeroed components).
    const std::array<amrex::Real, 3> beta = {0._rt, 0._rt, 0._rt};
    if (EB::enabled()) {
        // With EB: pass Efield to computePhi for EB-aware E computation.
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

void WarpX::SolvePoissonEfieldHomogeneousClean ()
{
    WARPX_PROFILE("WarpX::SolvePoissonEfieldHomogeneousClean");

    using ablastr::fields::Direction;
    using ablastr::fields::MultiLevelVectorField;
    using warpx::fields::FieldType;

    auto& es = GetElectrostaticSolver();
    const int nlevs = max_level + 1;

    // This homogeneous clean removes exactly the Gauss-law residual
    // div(E) - rho/eps0, which is also the source of the hyperbolic div(E)
    // cleaning field F; enabling both double-corrects Gauss's law.
    if (WarpX::do_dive_cleaning) {
        ablastr::warn_manager::WMRecordWarning(
            "Poisson E-field correction",
            "SolvePoissonEfieldHomogeneousClean and div(E) cleaning "
            "(warpx.do_dive_cleaning) both act on the Gauss-law residual "
            "div(E) - rho/eps0; enabling both double-corrects. Disable one.",
            ablastr::warn_manager::WarnPriority::low);
    }

    // Allocate temporary rho, phi and divE MultiFabs (all nodal).
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho(nlevs);
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> phi(nlevs);
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> divE(nlevs);
    const amrex::IntVect ng = get_ng_depos_rho();
    for (int lev = 0; lev < nlevs; lev++) {
        amrex::BoxArray nba = boxArray(lev);
        nba.surroundingNodes();
        rho[lev]  = std::make_unique<amrex::MultiFab>(nba, DistributionMap(lev), 1, ng);
        rho[lev]->setVal(0._rt);
        phi[lev]  = std::make_unique<amrex::MultiFab>(nba, DistributionMap(lev), 1, 1);
        phi[lev]->setVal(0._rt);
        divE[lev] = std::make_unique<amrex::MultiFab>(nba, DistributionMap(lev), 1, 0);
        divE[lev]->setVal(0._rt);
    }

    // Deposit charge from all species and synchronize (same path as SolvePoissonEfield).
    mypc->DepositCharge(amrex::GetVecOfPtrs(rho), 0.0_rt);
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

    // Build the homogeneous-clean source rho_eff = rho - eps0 * div(E_fp).
    // computePhi then solves nabla^2 psi = -rho_eff/eps0 = div(E_fp) - rho/eps0,
    // i.e. psi is the potential of the present field's Gauss-law residual.
    // div(E) is taken of Efield_fp directly (WarpX::ComputeDivE uses Efield_aux).
    for (int lev = 0; lev < nlevs; lev++) {
        const ablastr::fields::VectorField Efield_fp_lev =
            m_fields.get_alldirs(FieldType::Efield_fp, lev);
        m_fdtd_solver_fp[lev]->ComputeDivE(Efield_fp_lev, *divE[lev]);
        amrex::MultiFab::Saxpy(*rho[lev], -PhysConst::epsilon_0, *divE[lev], 0, 0, 1, 0);
    }

    // Save E_n, then zero the components we will re-solve. In RZ the axisymmetric
    // gradient has no theta component, so E_theta (comp 1) is left untouched.
    MultiLevelVectorField Efield_fp =
        m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, max_level);
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> Esaved(nlevs);
    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
#ifdef WARPX_DIM_RZ
            if (comp == 1) { continue; }
#endif
            amrex::MultiFab& mf = *Efield_fp[lev][comp];
            Esaved[lev][comp] = std::make_unique<amrex::MultiFab>(
                mf.boxArray(), mf.DistributionMap(), 1, mf.nGrowVect());
            amrex::MultiFab::Copy(*Esaved[lev][comp], mf, 0, 0, 1, mf.nGrowVect());
            mf.setVal(0._rt);
        }
    }

    // Use homogeneous boundary potentials (EB and domain) for the clean,
    // saving and restoring the user's potential strings around the solve.
    auto& bh = es.m_poisson_boundary_handler;
    const std::string s_eb  = bh->potential_eb_str;
    const std::string s_xlo = bh->potential_xlo_str;
    const std::string s_xhi = bh->potential_xhi_str;
    const std::string s_ylo = bh->potential_ylo_str;
    const std::string s_yhi = bh->potential_yhi_str;
    const std::string s_zlo = bh->potential_zlo_str;
    const std::string s_zhi = bh->potential_zhi_str;
    bh->potential_xlo_str = "0"; bh->potential_xhi_str = "0";
    bh->potential_ylo_str = "0"; bh->potential_yhi_str = "0";
    bh->potential_zlo_str = "0"; bh->potential_zhi_str = "0";
    bh->BuildParsers();
    bh->setPotentialEB("0");

    es.setPhiBC(amrex::GetVecOfPtrs(phi), gett_new(0));

    // Solve for psi and write -grad(psi) into the (zeroed) Efield_fp.
    const std::array<amrex::Real, 3> beta = {0._rt, 0._rt, 0._rt};
    if (EB::enabled()) {
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

    // Restore the user's boundary potentials.
    bh->potential_xlo_str = s_xlo; bh->potential_xhi_str = s_xhi;
    bh->potential_ylo_str = s_ylo; bh->potential_yhi_str = s_yhi;
    bh->potential_zlo_str = s_zlo; bh->potential_zhi_str = s_zhi;
    bh->BuildParsers();
    bh->setPotentialEB(s_eb);

    // Add E_n back: Efield_fp = E_n - grad(psi). Cleans Gauss, preserves curl.
    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
#ifdef WARPX_DIM_RZ
            if (comp == 1) { continue; }
#endif
            amrex::MultiFab::Add(*Efield_fp[lev][comp], *Esaved[lev][comp],
                                 0, 0, 1, Esaved[lev][comp]->nGrowVect());
        }
    }
}
