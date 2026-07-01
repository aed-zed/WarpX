/* Copyright 2024 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Remi Lehe, Roelof Groenewald, Arianna Formenti, Revathi Jambunathan
 *
 * License: BSD-3-Clause-LBNL
 */
#include "FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "FieldSolver/MagnetostaticSolver/MagnetostaticSolver.H"

#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Particles/MultiParticleContainer.H"
#include "Utils/WarpXProfilerWrapper.H"
#include "WarpX.H"

#include <ablastr/warn_manager/WarnManager.H>
#include <ablastr/fields/VectorPoissonSolver.H>
#include <ablastr/constant.H>

#ifdef AMREX_USE_EB
#   include <AMReX_EBFArrayBox.H>
#endif




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

void WarpX::ComputeVacuumEfield ()
{
    WARPX_PROFILE("WarpX::ComputeVacuumEfield");

    using ablastr::fields::MultiLevelScalarField;
    using ablastr::fields::MultiLevelVectorField;

    auto& es = GetElectrostaticSolver();
    const int nlevs = max_level + 1;
    const amrex::IntVect ng = get_ng_depos_rho();

    // Allocate temporary rho=0 and phi fields.
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho(nlevs);
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> phi(nlevs);
    for (int lev = 0; lev < nlevs; lev++) {
        amrex::BoxArray nba = boxArray(lev);
        nba.surroundingNodes();
        rho[lev] = std::make_unique<amrex::MultiFab>(
            nba, DistributionMap(lev), WarpX::ncomps, ng);
        rho[lev]->setVal(0.);
        phi[lev] = std::make_unique<amrex::MultiFab>(
            nba, DistributionMap(lev), WarpX::ncomps, 1);
        phi[lev]->setVal(0.);
    }

    // Apply the real electrode/domain potential BCs.
    es.setPhiBC(amrex::GetVecOfPtrs(phi), gett_new(0));

    // E_vac must have been allocated from Python.
    MultiLevelVectorField E_vac =
        m_fields.get_mr_levels_alldirs("E_vac", max_level);

    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            E_vac[lev][comp]->setVal(0.);
        }
    }

    const std::array<amrex::Real, 3> beta = {0._rt, 0._rt, 0._rt};
    if (EB::enabled()) {
        es.computePhi(amrex::GetVecOfPtrs(rho), amrex::GetVecOfPtrs(phi),
                      beta, es.self_fields_required_precision,
                      es.self_fields_absolute_tolerance,
                      es.self_fields_max_iters, es.self_fields_verbosity,
                      es.is_igf_2d_slices, E_vac);
    } else {
        es.computePhi(amrex::GetVecOfPtrs(rho), amrex::GetVecOfPtrs(phi),
                      beta, es.self_fields_required_precision,
                      es.self_fields_absolute_tolerance,
                      es.self_fields_max_iters, es.self_fields_verbosity,
                      es.is_igf_2d_slices);
        es.computeE(E_vac, amrex::GetVecOfPtrs(phi), beta);
    }

    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            E_vac[lev][comp]->FillBoundaryAndSync(Geom(lev).periodicity());
        }
    }
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
    constexpr int correction_verbosity = 1;

    amrex::IntVect const no_grow = amrex::IntVect(AMREX_D_DECL(0, 0, 0));
    auto sync_vector_field = [&] (
        ablastr::fields::MultiLevelVectorField const& field
    )
    {
        for (int lev = 0; lev < nlevs; lev++) {
            for (int comp = 0; comp < 3; comp++) {
                field[lev][comp]->FillBoundaryAndSync(Geom(lev).periodicity());
            }
        }
    };

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
            nba, DistributionMap(lev), WarpX::ncomps, ng);
        rho[lev]->setVal(0.);
        phi[lev] = std::make_unique<amrex::MultiFab>(
            nba, DistributionMap(lev), WarpX::ncomps, 1);
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

    MultiLevelVectorField Efield_fp =
        m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, max_level);

    MultiLevelVectorField E_diff_diag =
        m_fields.get_mr_levels_alldirs("E_diff_diag", max_level);

    MultiLevelVectorField E_irrot_drift_diag =
        m_fields.get_mr_levels_alldirs("E_irrot_drift_diag", max_level);

    MultiLevelVectorField E_rot_n_diag =
        m_fields.get_mr_levels_alldirs("E_rot_n_diag", max_level);

    // Save the original grid electric field as E_n.
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> E_n_storage(nlevs);
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> E_irrot_n_storage(nlevs);
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> E_diff_storage(nlevs);
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> E_irrot_drift_storage(nlevs);
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> E_rot_n_storage(nlevs);

    MultiLevelVectorField E_n(nlevs);
    MultiLevelVectorField E_irrot_n(nlevs);
    MultiLevelVectorField E_diff(nlevs);
    MultiLevelVectorField E_irrot_drift(nlevs);
    MultiLevelVectorField E_rot_n(nlevs);

    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            E_n_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Efield_fp[lev][comp]->boxArray(),
                Efield_fp[lev][comp]->DistributionMap(),
                Efield_fp[lev][comp]->nComp(),
                Efield_fp[lev][comp]->nGrowVect());

            E_irrot_n_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Efield_fp[lev][comp]->boxArray(),
                Efield_fp[lev][comp]->DistributionMap(),
                Efield_fp[lev][comp]->nComp(),
                Efield_fp[lev][comp]->nGrowVect());

            E_diff_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Efield_fp[lev][comp]->boxArray(),
                Efield_fp[lev][comp]->DistributionMap(),
                Efield_fp[lev][comp]->nComp(),
                Efield_fp[lev][comp]->nGrowVect());

            E_irrot_drift_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Efield_fp[lev][comp]->boxArray(),
                Efield_fp[lev][comp]->DistributionMap(),
                Efield_fp[lev][comp]->nComp(),
                Efield_fp[lev][comp]->nGrowVect());

            E_rot_n_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Efield_fp[lev][comp]->boxArray(),
                Efield_fp[lev][comp]->DistributionMap(),
                Efield_fp[lev][comp]->nComp(),
                Efield_fp[lev][comp]->nGrowVect());

            amrex::MultiFab::Copy(*E_n_storage[lev][comp],
                                  *Efield_fp[lev][comp],
                                  0, 0, Efield_fp[lev][comp]->nComp(),
                                  no_grow);

            E_irrot_n_storage[lev][comp]->setVal(0.);
            E_diff_storage[lev][comp]->setVal(0.);
            E_irrot_drift_storage[lev][comp]->setVal(0.);
            E_rot_n_storage[lev][comp]->setVal(0.);

            E_n[lev][comp] = E_n_storage[lev][comp].get();
            E_irrot_n[lev][comp] = E_irrot_n_storage[lev][comp].get();
            E_diff[lev][comp] = E_diff_storage[lev][comp].get();
            E_irrot_drift[lev][comp] = E_irrot_drift_storage[lev][comp].get();
            E_rot_n[lev][comp] = E_rot_n_storage[lev][comp].get();
        }
    }

    // Solve Poisson and compute E_irrot_n.
    const std::array<amrex::Real, 3> beta = {0._rt, 0._rt, 0._rt};
    if (EB::enabled()) {
        es.computePhi(amrex::GetVecOfPtrs(rho), amrex::GetVecOfPtrs(phi),
                      beta, es.self_fields_required_precision,
                      es.self_fields_absolute_tolerance,
                      es.self_fields_max_iters, correction_verbosity,
                      es.is_igf_2d_slices, E_irrot_n);
    } else {
        es.computePhi(amrex::GetVecOfPtrs(rho), amrex::GetVecOfPtrs(phi),
                      beta, es.self_fields_required_precision,
                      es.self_fields_absolute_tolerance,
                      es.self_fields_max_iters, correction_verbosity,
                      es.is_igf_2d_slices);
        es.computeE(E_irrot_n, amrex::GetVecOfPtrs(phi), beta);
    }

    // Compute E_diff = E_n - E_irrot_n.
    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            amrex::MultiFab::LinComb(*E_diff[lev][comp],
                                     1._rt, *E_n[lev][comp], 0,
                                    -1._rt, *E_irrot_n[lev][comp], 0,
                                     0, Efield_fp[lev][comp]->nComp(),
                                     no_grow);
        }
    }

    sync_vector_field(E_diff);


    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            amrex::MultiFab::Copy(*E_diff_diag[lev][comp],
                                  *E_diff[lev][comp],
                                  0, 0, E_diff[lev][comp]->nComp(),
                                  no_grow);
        }
    }

    // Allocate temporary rho_correction and phi_correction_tmp MultiFabs.
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> rho_correction(nlevs);
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> phi_correction_tmp(nlevs);
    for (int lev = 0; lev < nlevs; lev++) {
        amrex::BoxArray nba = boxArray(lev);
        nba.surroundingNodes();
        rho_correction[lev] = std::make_unique<amrex::MultiFab>(
            nba, DistributionMap(lev), WarpX::ncomps, ng);
        rho_correction[lev]->setVal(0.);
        phi_correction_tmp[lev] = std::make_unique<amrex::MultiFab>(
            nba, DistributionMap(lev), WarpX::ncomps, 1);
        phi_correction_tmp[lev]->setVal(0.);
    }

    // Compute rho_correction = epsilon_0 * div(E_n - E_irrot_n).
    for (int lev = 0; lev < nlevs; lev++) {
        get_pointer_fdtd_solver_fp(lev)->ComputeDivE(E_diff[lev],
                                                     *rho_correction[lev]);
        rho_correction[lev]->mult(ablastr::constant::SI::epsilon_0);
    }
    
    // Make shared nodal values consistent.
    for (int lev = 0; lev < nlevs; lev++) {
        rho_correction[lev]->OverrideSync(Geom(lev).periodicity());
        rho_correction[lev]->FillBoundary(Geom(lev).periodicity());
    }

#ifndef WARPX_DIM_RZ
    for (int lev = 0; lev < nlevs; lev++) {
        ApplyRhofieldBoundary(lev, rho_correction[lev].get(), PatchType::fine);
    }
#endif

    if (EB::enabled()) {
    // Solve for phi_correction_tmp with EB geometry, but with homogeneous EB
    // Dirichlet data. This keeps the correction solve EB-aware without applying
    // the electrode potential a second time.
        es.computePhi_EBhomogeneous(amrex::GetVecOfPtrs(rho_correction),
                                    amrex::GetVecOfPtrs(phi_correction_tmp),
                                    beta, es.self_fields_required_precision,
                                    es.self_fields_absolute_tolerance,
                                    es.self_fields_max_iters, correction_verbosity,
                                    es.is_igf_2d_slices,
                                    E_irrot_drift);
    } else {
        es.computePhi(amrex::GetVecOfPtrs(rho_correction), amrex::GetVecOfPtrs(phi_correction_tmp),
                      beta, es.self_fields_required_precision,
                      es.self_fields_absolute_tolerance,
                      es.self_fields_max_iters, correction_verbosity,
                      es.is_igf_2d_slices);
        // Compute E_irrot_drift = -grad(phi_correction_tmp) into a temporary field.
        es.computeE(E_irrot_drift, amrex::GetVecOfPtrs(phi_correction_tmp), beta);
    }

    // Compute E_rot_n = (E_n - E_irrot_n) - E_irrot_drift.
    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            amrex::MultiFab::LinComb(*E_rot_n[lev][comp],
                                     1._rt, *E_diff[lev][comp], 0,
                                    -1._rt, *E_irrot_drift[lev][comp], 0,
                                     0, Efield_fp[lev][comp]->nComp(),
                                     no_grow);
        }
    }

    bool has_E_vac = true;
    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            has_E_vac = has_E_vac &&
                m_fields.has("E_vac", Direction{comp}, lev);
        }
    }

    if (has_E_vac) {
        MultiLevelVectorField E_vac =
            m_fields.get_mr_levels_alldirs("E_vac", max_level);

        amrex::Real alpha_num = 0._rt;
        amrex::Real alpha_den = 0._rt;

        for (int lev = 0; lev < nlevs; lev++) {
            for (int comp = 0; comp < 3; comp++) {
                alpha_num += amrex::MultiFab::Dot(*E_rot_n[lev][comp], 0, *E_vac[lev][comp], 0, Efield_fp[lev][comp]->nComp(), 0);
                alpha_den += amrex::MultiFab::Dot(*E_vac[lev][comp], 0, *E_vac[lev][comp], 0, Efield_fp[lev][comp]->nComp(), 0);
            }
        }

        amrex::Real alpha = 0._rt;
        if (alpha_den > 0._rt) {
            alpha = alpha_num / alpha_den;
        }

        amrex::Print() << "[PoissonCorrector] harmonic alpha = "
                    << alpha << "\n";
                    
        for (int lev = 0; lev < nlevs; lev++) {
            for (int comp = 0; comp < 3; comp++) {
                amrex::MultiFab::Saxpy(*E_irrot_drift[lev][comp],
                                    alpha, *E_vac[lev][comp],
                                    0, 0, Efield_fp[lev][comp]->nComp(),
                                    no_grow);

                amrex::MultiFab::Saxpy(*E_rot_n[lev][comp],
                                    -alpha, *E_vac[lev][comp],
                                    0, 0, Efield_fp[lev][comp]->nComp(),
                                    no_grow);
            }
        }
    } else {
        amrex::Print() << "[PoissonCorrector] E_vac is not registered; "
                    << "skipping harmonic projection.\n";
    }


    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            amrex::MultiFab::Copy(*E_irrot_drift_diag[lev][comp],
                                  *E_irrot_drift[lev][comp],
                                  0, 0, E_irrot_drift[lev][comp]->nComp(),
                                  no_grow);
        }
    }


    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            amrex::MultiFab::Copy(*E_rot_n_diag[lev][comp],
                                  *E_rot_n[lev][comp],
                                  0, 0, E_rot_n[lev][comp]->nComp(),
                                  no_grow);
        }
    }

    // Replace the grid electric field with E_irrot_n + E_rot_n.
    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            amrex::MultiFab::LinComb(*Efield_fp[lev][comp],
                                     1._rt, *E_irrot_n[lev][comp], 0,
                                     1._rt, *E_rot_n[lev][comp], 0,
                                     0, Efield_fp[lev][comp]->nComp(),
                                     no_grow);
        }
    }
    
    sync_vector_field(Efield_fp);
}

void WarpX::SolvePoissonEfield_w_A ()
{
    WARPX_PROFILE("WarpX::SolvePoissonEfield_w_A");

    using ablastr::fields::Direction;
    using ablastr::fields::MultiLevelScalarField;
    using ablastr::fields::MultiLevelVectorField;
    using warpx::fields::FieldType;

    auto& es = GetElectrostaticSolver();
    const int nlevs = max_level + 1;
    constexpr int correction_verbosity = 1;

    amrex::IntVect const no_grow = amrex::IntVect(AMREX_D_DECL(0, 0, 0));
    auto sync_vector_field = [&] (
        ablastr::fields::MultiLevelVectorField const& field
    )
    {
        for (int lev = 0; lev < nlevs; lev++) {
            for (int comp = 0; comp < 3; comp++) {
                field[lev][comp]->FillBoundaryAndSync(Geom(lev).periodicity());
            }
        }
    };

    if (WarpX::grid_type == ablastr::utils::enums::GridType::Collocated) {
        ablastr::warn_manager::WMRecordWarning(
            "Poisson E-field correction",
            "SolvePoissonEfield_w_A assumes a staggered (Yee) grid. "
            "The vector-potential projection is not validated on collocated grids.",
            ablastr::warn_manager::WarnPriority::high);
    }
    if (WarpX::do_dive_cleaning) {
        ablastr::warn_manager::WMRecordWarning(
            "Poisson E-field correction",
            "SolvePoissonEfield_w_A and div(E) cleaning both act on field "
            "constraint errors; enabling both can double-correct.",
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
            nba, DistributionMap(lev), WarpX::ncomps, ng);
        rho[lev]->setVal(0.);
        phi[lev] = std::make_unique<amrex::MultiFab>(
            nba, DistributionMap(lev), WarpX::ncomps, 1);
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

    // Set real applied electrode/domain potentials.
    es.setPhiBC(amrex::GetVecOfPtrs(phi), gett_new(0));

    MultiLevelVectorField Efield_fp =
        m_fields.get_mr_levels_alldirs(FieldType::Efield_fp, max_level);

    MultiLevelVectorField E_diff_diag =
        m_fields.get_mr_levels_alldirs("E_diff_diag", max_level);

    MultiLevelVectorField E_irrot_drift_diag =
        m_fields.get_mr_levels_alldirs("E_irrot_drift_diag", max_level);

    MultiLevelVectorField E_rot_n_diag =
        m_fields.get_mr_levels_alldirs("E_rot_n_diag", max_level);

    MultiLevelVectorField Bfield_fp =
        m_fields.get_mr_levels_alldirs(FieldType::Bfield_fp, max_level);

    // Save E_n and allocate temporary E-like fields.
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> E_n_storage(nlevs);
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> E_irrot_n_storage(nlevs);
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> E_diff_storage(nlevs);
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> E_irrot_drift_storage(nlevs);
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> E_rot_n_storage(nlevs);

    MultiLevelVectorField E_n(nlevs);
    MultiLevelVectorField E_irrot_n(nlevs);
    MultiLevelVectorField E_diff(nlevs);
    MultiLevelVectorField E_irrot_drift(nlevs);
    MultiLevelVectorField E_rot_n(nlevs);

    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            E_n_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Efield_fp[lev][comp]->boxArray(),
                Efield_fp[lev][comp]->DistributionMap(),
                Efield_fp[lev][comp]->nComp(),
                Efield_fp[lev][comp]->nGrowVect());

            E_irrot_n_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Efield_fp[lev][comp]->boxArray(),
                Efield_fp[lev][comp]->DistributionMap(),
                Efield_fp[lev][comp]->nComp(),
                Efield_fp[lev][comp]->nGrowVect());

            E_diff_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Efield_fp[lev][comp]->boxArray(),
                Efield_fp[lev][comp]->DistributionMap(),
                Efield_fp[lev][comp]->nComp(),
                Efield_fp[lev][comp]->nGrowVect());

            E_irrot_drift_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Efield_fp[lev][comp]->boxArray(),
                Efield_fp[lev][comp]->DistributionMap(),
                Efield_fp[lev][comp]->nComp(),
                Efield_fp[lev][comp]->nGrowVect());

            E_rot_n_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Efield_fp[lev][comp]->boxArray(),
                Efield_fp[lev][comp]->DistributionMap(),
                Efield_fp[lev][comp]->nComp(),
                Efield_fp[lev][comp]->nGrowVect());

            amrex::MultiFab::Copy(*E_n_storage[lev][comp],
                                  *Efield_fp[lev][comp],
                                  0, 0, Efield_fp[lev][comp]->nComp(),
                                  no_grow);

            E_irrot_n_storage[lev][comp]->setVal(0.);
            E_diff_storage[lev][comp]->setVal(0.);
            E_irrot_drift_storage[lev][comp]->setVal(0.);
            E_rot_n_storage[lev][comp]->setVal(0.);

            E_n[lev][comp] = E_n_storage[lev][comp].get();
            E_irrot_n[lev][comp] = E_irrot_n_storage[lev][comp].get();
            E_diff[lev][comp] = E_diff_storage[lev][comp].get();
            E_irrot_drift[lev][comp] = E_irrot_drift_storage[lev][comp].get();
            E_rot_n[lev][comp] = E_rot_n_storage[lev][comp].get();
        }
    }

    // Solve Poisson and compute E_irrot_n.
    const std::array<amrex::Real, 3> beta = {0._rt, 0._rt, 0._rt};
    if (EB::enabled()) {
        es.computePhi(amrex::GetVecOfPtrs(rho), amrex::GetVecOfPtrs(phi),
                      beta, es.self_fields_required_precision,
                      es.self_fields_absolute_tolerance,
                      es.self_fields_max_iters, correction_verbosity,
                      es.is_igf_2d_slices, E_irrot_n);
    } else {
        es.computePhi(amrex::GetVecOfPtrs(rho), amrex::GetVecOfPtrs(phi),
                      beta, es.self_fields_required_precision,
                      es.self_fields_absolute_tolerance,
                      es.self_fields_max_iters, correction_verbosity,
                      es.is_igf_2d_slices);
        es.computeE(E_irrot_n, amrex::GetVecOfPtrs(phi), beta);
    }

    // Compute E_diff = E_n - E_irrot_n.
    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            amrex::MultiFab::LinComb(*E_diff[lev][comp],
                                     1._rt, *E_n[lev][comp], 0,
                                    -1._rt, *E_irrot_n[lev][comp], 0,
                                     0, Efield_fp[lev][comp]->nComp(),
                                     no_grow);
        }
    }

    sync_vector_field(E_diff);

    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            amrex::MultiFab::Copy(*E_diff_diag[lev][comp],
                                  *E_diff[lev][comp],
                                  0, 0, E_diff[lev][comp]->nComp(),
                                  no_grow);
        }
    }

    // Allocate temporary curl_Ediff and A MultiFabs.
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> curl_Ediff_storage(nlevs);
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> A_storage(nlevs);

    MultiLevelVectorField curl_Ediff(nlevs);
    MultiLevelVectorField A_vec(nlevs);

    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            // curl(E_diff) lives on B-like staggering, so allocate from Bfield_fp.
            curl_Ediff_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Bfield_fp[lev][comp]->boxArray(),
                Bfield_fp[lev][comp]->DistributionMap(),
                Bfield_fp[lev][comp]->nComp(),
                Bfield_fp[lev][comp]->nGrowVect());
            A_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Bfield_fp[lev][comp]->boxArray(),
                Bfield_fp[lev][comp]->DistributionMap(),
                Bfield_fp[lev][comp]->nComp(),
                Bfield_fp[lev][comp]->nGrowVect());
            curl_Ediff_storage[lev][comp]->setVal(0.);
            A_storage[lev][comp]->setVal(0.);
            curl_Ediff[lev][comp] = curl_Ediff_storage[lev][comp].get();
            A_vec[lev][comp] = A_storage[lev][comp].get();
        }
    }

    // These are only passed to finite-difference curl helpers. The helpers
    // write into the explicit output fields above. They should not touch
    // registered Bfield/current fields.
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> dummy_eb_update_B;
    std::array<std::unique_ptr<amrex::iMultiFab>, 3> dummy_eb_update_E;

    for (int lev = 0; lev < nlevs; lev++) {
        get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(curl_Ediff[lev], E_diff[lev], dummy_eb_update_B, lev);
        for (int comp = 0; comp < 3; comp++) {
            // Pseudo-current source for computeVectorPotential.
            // computeVectorPotential will multiply this by -mu0 internally.
            curl_Ediff[lev][comp]->mult(1._rt / ablastr::constant::SI::mu0);
        }
    }

    sync_vector_field(curl_Ediff);

    MagnetostaticSolver::VectorPoissonBoundaryHandler vector_bc;
    vector_bc.defineVectorPotentialBCs();

#ifdef AMREX_USE_EB
    std::optional<amrex::Vector<amrex::EBFArrayBoxFactory const*>> eb_farray_box_factory;
    if (EB::enabled()) {
        amrex::Vector<amrex::EBFArrayBoxFactory const*> factories;
        factories.reserve(nlevs);
        for (int lev = 0; lev < nlevs; lev++) {
            factories.push_back(&fieldEBFactory(lev));
        }
        eb_farray_box_factory = std::move(factories);
    }

    ablastr::fields::computeVectorPotential<
        MagnetostaticSolver::VectorPoissonBoundaryHandler,
        std::nullopt_t,
        amrex::EBFArrayBoxFactory>(
        curl_Ediff,
        A_vec,
        es.self_fields_required_precision,
        es.self_fields_absolute_tolerance,
        es.self_fields_max_iters,
        correction_verbosity,
        Geom(),
        DistributionMap(),
        boxArray(),
        vector_bc,
        EB::enabled(),
        WarpX::do_single_precision_comms,
        refRatio(),
        std::nullopt,
        gett_new(0),
        eb_farray_box_factory
    );
#else
    ablastr::fields::computeVectorPotential(
        curl_Ediff,
        A_vec,
        es.self_fields_required_precision,
        es.self_fields_absolute_tolerance,
        es.self_fields_max_iters,
        correction_verbosity,
        Geom(),
        DistributionMap(),
        boxArray(),
        vector_bc,
        false,
        WarpX::do_single_precision_comms,
        refRatio()
    );
#endif

    sync_vector_field(A_vec);

    // Recover E_rot_n = curl(A).
    //
    // CalculateCurrentAmpere gives J = curl(B) / mu0. Treat A_vec as the
    // B-like input, then multiply the result by mu0 to get curl(A).
    for (int lev = 0; lev < nlevs; lev++) {
        get_pointer_fdtd_solver_fp(lev)->CalculateCurrentAmpere(E_rot_n[lev], A_vec[lev], dummy_eb_update_E, lev);
        for (int comp = 0; comp < 3; comp++) {
            E_rot_n[lev][comp]->mult(ablastr::constant::SI::mu0);
        }
    }

    sync_vector_field(E_rot_n);

    // E_irrot_drift is now whatever remains after removing the direct
    // rotational projection.
    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            amrex::MultiFab::LinComb(*E_irrot_drift[lev][comp],
                                     1._rt, *E_diff[lev][comp], 0,
                                    -1._rt, *E_rot_n[lev][comp], 0,
                                     0, Efield_fp[lev][comp]->nComp(),
                                     no_grow);
        }
    }

    sync_vector_field(E_irrot_drift);

    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            amrex::MultiFab::Copy(*E_irrot_drift_diag[lev][comp],
                                  *E_irrot_drift[lev][comp],
                                  0, 0, E_irrot_drift[lev][comp]->nComp(),
                                  no_grow);

            amrex::MultiFab::Copy(*E_rot_n_diag[lev][comp],
                                  *E_rot_n[lev][comp],
                                  0, 0, E_rot_n[lev][comp]->nComp(),
                                  no_grow);
        }
    }

    // Replace Efield_fp with E_irrot_n + E_rot_n.
    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            amrex::MultiFab::LinComb(*Efield_fp[lev][comp],
                                     1._rt, *E_irrot_n[lev][comp], 0,
                                     1._rt, *E_rot_n[lev][comp], 0,
                                     0, Efield_fp[lev][comp]->nComp(),
                                     no_grow);
        }
    }

    sync_vector_field(Efield_fp);
}