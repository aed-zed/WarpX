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

#include "Parallelization/WarpXComm_K.H"

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
    constexpr int vector_poisson_verbosity = 2;
    constexpr int vector_poisson_max_iters = 2000;
    const amrex::Real vector_poisson_required_precision = es.self_fields_required_precision;
    // const amrex::Real vector_poisson_required_precision = 1.e-12_rt;
    const amrex::Real vector_poisson_absolute_tolerance = 1.e-30_rt;


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

    auto apply_eb_update_mask = [&] (
        ablastr::fields::VectorField const& field,
        std::array<std::unique_ptr<amrex::iMultiFab>, 3> const& eb_update,
        int const lev
    )
    {
        if (!EB::enabled()) { return; }

        for (int comp = 0; comp < 3; comp++) {
    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
    #endif
            for (amrex::MFIter mfi(*field[comp], amrex::TilingIfNotGPU());
                mfi.isValid(); ++mfi)
            {
                amrex::Array4<amrex::Real> const& field_arr =
                    field[comp]->array(mfi);

                amrex::Array4<int const> const& update_arr =
                    eb_update[comp]->const_array(mfi);

                amrex::Box const bx =
                    mfi.growntilebox(field[comp]->ixType().toIntVect());

                int const ncomp = field[comp]->nComp();

                amrex::ParallelFor(bx, ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        if (update_arr(i, j, k, n) == 0) {
                            field_arr(i, j, k, n) = 0._rt;
                        }
                    });
            }

            field[comp]->FillBoundaryAndSync(Geom(lev).periodicity());
        }
    };

    auto print_vec_norms = [&] (
        char const* label,
        ablastr::fields::MultiLevelVectorField const& field
    )
    {
        for (int lev = 0; lev < nlevs; lev++) {
            for (int comp = 0; comp < 3; comp++) {
                amrex::Print() << label
                            << " lev " << lev
                            << " comp " << comp
                            << " ixType = "
                            << field[lev][comp]->ixType().toIntVect()
                            << " norm0 = "
                            << field[lev][comp]->norm0()
                            << "\n";
            }
        }
    };

    auto interpolate_vector_field = [&] (
        ablastr::fields::MultiLevelVectorField const& src,
        ablastr::fields::MultiLevelVectorField const& dst
    )
    {
        for (int lev = 0; lev < nlevs; lev++) {
            for (int comp = 0; comp < 3; comp++) {
                dst[lev][comp]->setVal(0.);

                amrex::IntVect const dst_stag =
                    dst[lev][comp]->ixType().toIntVect();
                amrex::IntVect const src_stag =
                    src[lev][comp]->ixType().toIntVect();

                int const fg_nox = WarpX::field_centering_nox;
                int const fg_noy = WarpX::field_centering_noy;
                int const fg_noz = WarpX::field_centering_noz;

                amrex::Real const* stencil_coeffs_x =
                    WarpX::device_field_centering_stencil_coeffs_x.data();
                amrex::Real const* stencil_coeffs_y =
                    WarpX::device_field_centering_stencil_coeffs_y.data();
                amrex::Real const* stencil_coeffs_z =
                    WarpX::device_field_centering_stencil_coeffs_z.data();

    #ifdef AMREX_USE_OMP
    #pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
    #endif
                for (amrex::MFIter mfi(*dst[lev][comp], amrex::TilingIfNotGPU());
                    mfi.isValid(); ++mfi)
                {
                    amrex::Array4<amrex::Real> const& dst_arr =
                        dst[lev][comp]->array(mfi);
                    amrex::Array4<amrex::Real const> const& src_arr =
                        src[lev][comp]->const_array(mfi);

                    amrex::Box const bx = mfi.growntilebox(dst_stag);

                    amrex::ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            warpx_interp(i, j, k,
                                        dst_arr, src_arr,
                                        dst_stag, src_stag,
                                        fg_nox, fg_noy, fg_noz,
                                        stencil_coeffs_x,
                                        stencil_coeffs_y,
                                        stencil_coeffs_z);
                        });
                }

                dst[lev][comp]->FillBoundaryAndSync(Geom(lev).periodicity());
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
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> J_pseudo_nodal_storage(nlevs);
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> A_storage(nlevs);
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> curl_A_storage(nlevs);

    MultiLevelVectorField curl_Ediff(nlevs);
    MultiLevelVectorField J_pseudo_nodal(nlevs);
    MultiLevelVectorField A_vec(nlevs);
    MultiLevelVectorField curl_A(nlevs);
    
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> grad_buf_e_stag_storage(nlevs);
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> grad_buf_b_stag_storage(nlevs);
    MultiLevelVectorField grad_buf_e_stag(nlevs);
    MultiLevelVectorField grad_buf_b_stag(nlevs);

    for (int lev = 0; lev < nlevs; lev++) {
        amrex::BoxArray nba = boxArray(lev);
        nba.surroundingNodes();
        for (int comp = 0; comp < 3; comp++) {
            // curl(E_diff) is B-like.
            curl_Ediff_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Bfield_fp[lev][comp]->boxArray(),
                Bfield_fp[lev][comp]->DistributionMap(),
                Bfield_fp[lev][comp]->nComp(),
                Bfield_fp[lev][comp]->nGrowVect());
            // J_pseudo_nodal is the nodal source for MLEBNodeFDLaplacian.
            J_pseudo_nodal_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                nba, DistributionMap(lev), WarpX::ncomps, 1);
            // A_vec must also be nodal.
            A_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                nba, DistributionMap(lev), WarpX::ncomps, 1);
            // curl(A) is first computed on B-like staggering.
            curl_A_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Bfield_fp[lev][comp]->boxArray(),
                Bfield_fp[lev][comp]->DistributionMap(),
                Bfield_fp[lev][comp]->nComp(),
                Bfield_fp[lev][comp]->nGrowVect());
            curl_Ediff_storage[lev][comp]->setVal(0.);
            J_pseudo_nodal_storage[lev][comp]->setVal(0.);
            A_storage[lev][comp]->setVal(0.);
            curl_A_storage[lev][comp]->setVal(0.);
            curl_Ediff[lev][comp] = curl_Ediff_storage[lev][comp].get();
            J_pseudo_nodal[lev][comp] = J_pseudo_nodal_storage[lev][comp].get();
            A_vec[lev][comp] = A_storage[lev][comp].get();
            curl_A[lev][comp] = curl_A_storage[lev][comp].get();

            grad_buf_e_stag_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Efield_fp[lev][comp]->boxArray(),
                Efield_fp[lev][comp]->DistributionMap(),
                Efield_fp[lev][comp]->nComp(),
                Efield_fp[lev][comp]->nGrowVect());
            grad_buf_b_stag_storage[lev][comp] = std::make_unique<amrex::MultiFab>(
                Bfield_fp[lev][comp]->boxArray(),
                Bfield_fp[lev][comp]->DistributionMap(),
                Bfield_fp[lev][comp]->nComp(),
                Bfield_fp[lev][comp]->nGrowVect());
            grad_buf_e_stag_storage[lev][comp]->setVal(0.);
            grad_buf_b_stag_storage[lev][comp]->setVal(0.);
            grad_buf_e_stag[lev][comp] = grad_buf_e_stag_storage[lev][comp].get();
            grad_buf_b_stag[lev][comp] = grad_buf_b_stag_storage[lev][comp].get();            
        }
    }

    print_vec_norms("E_diff before curl", E_diff);

    for (int lev = 0; lev < nlevs; lev++) {
        // auto eb_update_B = make_unit_eb_update(curl_Ediff[lev]);
        get_pointer_fdtd_solver_fp(lev)->ComputeCurlA(curl_Ediff[lev], E_diff[lev], m_eb_update_B[lev], lev);
        print_vec_norms("curl_Ediff before /mu0", curl_Ediff);
        apply_eb_update_mask(curl_Ediff[lev], m_eb_update_B[lev], lev);
    }

    sync_vector_field(curl_Ediff);
    print_vec_norms("curl_Ediff B-like before interp", curl_Ediff);

    // Interpolate B-staggered curl(E_diff) to nodal pseudo-current.
    interpolate_vector_field(curl_Ediff, J_pseudo_nodal);

    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            J_pseudo_nodal[lev][comp]->mult(1._rt / ablastr::constant::SI::mu0);
        }
    }

    sync_vector_field(J_pseudo_nodal);
    print_vec_norms("J_pseudo_nodal after interp and /mu0", J_pseudo_nodal);

    MagnetostaticSolver::VectorPoissonBoundaryHandler vector_bc;
    vector_bc.defineVectorPotentialBCs();

    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            curl_A[lev][comp]->setVal(0.);
            grad_buf_e_stag[lev][comp]->setVal(0.);
            grad_buf_b_stag[lev][comp]->setVal(0.);
        }
    }

    using VectorPotentialPostCalc =
        std::optional<MagnetostaticSolver::EBCalcBfromVectorPotentialPerLevel>;
    VectorPotentialPostCalc post_A_calculation =
        MagnetostaticSolver::EBCalcBfromVectorPotentialPerLevel(
            curl_A, grad_buf_e_stag, grad_buf_b_stag);

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
        VectorPotentialPostCalc,
        amrex::EBFArrayBoxFactory>(
        J_pseudo_nodal,
        A_vec,
        vector_poisson_required_precision,
        vector_poisson_absolute_tolerance,
        vector_poisson_max_iters,
        vector_poisson_verbosity,
        Geom(),
        DistributionMap(),
        boxArray(),
        vector_bc,
        EB::enabled(),
        WarpX::do_single_precision_comms,
        refRatio(),
        post_A_calculation,
        gett_new(0),
        eb_farray_box_factory
    );
#else
    ablastr::fields::computeVectorPotential<
        MagnetostaticSolver::VectorPoissonBoundaryHandler,
        VectorPotentialPostCalc>(
        J_pseudo_nodal,
        A_vec,
        vector_poisson_required_precision,
        vector_poisson_absolute_tolerance,
        vector_poisson_max_iters,
        vector_poisson_verbosity,
        Geom(),
        DistributionMap(),
        boxArray(),
        vector_bc,
        false,
        WarpX::do_single_precision_comms,
        refRatio(),
        post_A_calculation,
        gett_new(0)
    );
#endif

    sync_vector_field(A_vec);
    print_vec_norms("A_vec after vector Poisson", A_vec);

    for (int lev = 0; lev < nlevs; lev++) {
        apply_eb_update_mask(curl_A[lev], m_eb_update_B[lev], lev);
    }
    sync_vector_field(curl_A);
    print_vec_norms("curl_A B-like before E interpolation", curl_A);

    // Center curl(A) from B-like staggering onto the Efield_fp staggering.
    interpolate_vector_field(curl_A, E_rot_n);

    for (int lev = 0; lev < nlevs; lev++) {
        apply_eb_update_mask(E_rot_n[lev], m_eb_update_E[lev], lev);
    }
    sync_vector_field(E_rot_n);
    print_vec_norms("E_rot_n after curl_A interpolation", E_rot_n);

    bool has_E_vac = true;
    for (int lev = 0; lev < nlevs; lev++) {
        for (int comp = 0; comp < 3; comp++) {
            has_E_vac = has_E_vac && m_fields.has("E_vac", Direction{comp}, lev);
        }
    }

    if (has_E_vac) {
        MultiLevelVectorField E_vac = m_fields.get_mr_levels_alldirs("E_vac", max_level);
        amrex::Real alpha_num = 0._rt;
        amrex::Real alpha_den = 0._rt;
        for (int lev = 0; lev < nlevs; lev++) {
            for (int comp = 0; comp < 3; comp++) {
                alpha_num += amrex::MultiFab::Dot(*E_rot_n[lev][comp], 0, *E_vac[lev][comp], 0, E_rot_n[lev][comp]->nComp(), 0);
                alpha_den += amrex::MultiFab::Dot(*E_vac[lev][comp], 0, *E_vac[lev][comp], 0, E_vac[lev][comp]->nComp(), 0);
            }
        }

        amrex::ParallelDescriptor::ReduceRealSum(alpha_num);
        amrex::ParallelDescriptor::ReduceRealSum(alpha_den);
        
        amrex::Real alpha = 0._rt;
        if (alpha_den > 0._rt) {
            alpha = alpha_num / alpha_den;
        }
        amrex::Print() << "[PoissonCorrector] harmonic alpha = " << alpha << "\n";
                    
        for (int lev = 0; lev < nlevs; lev++) {
            for (int comp = 0; comp < 3; comp++) {
                amrex::MultiFab::Saxpy(*E_rot_n[lev][comp],
                                    -alpha, *E_vac[lev][comp],
                                    0, 0, E_rot_n[lev][comp]->nComp(),
                                    no_grow);
            }
        }
    } else {
        amrex::Print() << "[PoissonCorrector] E_vac is not registered; " << "skipping harmonic projection.\n";
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