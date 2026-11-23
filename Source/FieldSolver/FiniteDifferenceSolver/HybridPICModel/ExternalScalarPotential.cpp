/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Marco Garten
 *
 * License: BSD-3-Clause-LBNL
 */

#include "ExternalScalarPotential.H"
#include "Fields.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/warn_manager/WarnManager.H>
#include "EmbeddedBoundary/Enabled.H"

using namespace amrex;
using namespace warpx::fields;

ExternalScalarPotential::ExternalScalarPotential ()
{
    ReadParameters();
}

void
ExternalScalarPotential::ReadParameters ()
{
    const ParmParse pp_ext_Phi("external_scalar_potential");

    // Check if any external scalar potential fields are defined
    pp_ext_Phi.queryarr("fields", m_field_names);

    if (m_field_names.empty()) {
        // No explicit Phi_ext fields specified
        // Check if user wants to use boundary potentials for external fields
        const ParmParse pp_hybrid("hybrid_pic_model");
        bool use_boundary_potentials = false;
        pp_hybrid.query("use_boundary_potentials_for_external_field", use_boundary_potentials);

        if (use_boundary_potentials) {
            // Create a single field entry to represent boundary potentials
            m_field_names.push_back("boundary_potentials");
            amrex::Print() << Utils::TextMsg::Info(
                "Hybrid solver will use boundary.potential_* and warpx.eb_potential "
                "as external fields via Poisson solve.\n"
            );
        } else {
            // No external scalar potentials specified at all
            return;
        }
    }

    m_nFields = static_cast<int>(m_field_names.size());

    // Resize vectors and set defaults
    m_Phi_ext_grid_function.resize(m_nFields);
    for (std::string & field : m_Phi_ext_grid_function) { field = "0.0"; }

    m_Phi_external_parser.resize(m_nFields);
    m_Phi_external.resize(m_nFields);

    m_Phi_ext_time_function.resize(m_nFields);
    for (std::string & field_time : m_Phi_ext_time_function) { field_time = "1.0"; }

    m_Phi_external_time_parser.resize(m_nFields);
    m_Phi_time_scale.resize(m_nFields);

    m_E_unit.resize(m_nFields);
    m_E_unit_valid.resize(m_nFields, false);

    m_read_Phi_from_file.resize(m_nFields);
    m_external_file_path.resize(m_nFields);
    for (std::string & file_name : m_external_file_path) { file_name = ""; }

    for (int i = 0; i < m_nFields; ++i) {
        // Skip parameter reading for the special "boundary_potentials" field
        if (m_field_names[i] == "boundary_potentials") {
            m_read_Phi_from_file[i] = false;
            continue;
        }

        bool read_from_file = false;
        utils::parser::queryWithParser(pp_ext_Phi,
            (m_field_names[i]+".read_from_file").c_str(), read_from_file);
        m_read_Phi_from_file[i] = read_from_file;

        if (m_read_Phi_from_file[i]) {
            pp_ext_Phi.query((m_field_names[i]+".path").c_str(), m_external_file_path[i]);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!m_external_file_path[i].empty(),
                "external_scalar_potential: read_from_file=true but no path specified for field " + m_field_names[i]);
        } else {
            pp_ext_Phi.query((m_field_names[i]+".Phi_external_grid_function(x,y,z)").c_str(),
                m_Phi_ext_grid_function[i]);
        }

        pp_ext_Phi.query((m_field_names[i]+".Phi_time_external_function(t)").c_str(),
            m_Phi_ext_time_function[i]);
    }
}

void
ExternalScalarPotential::InitData ()
{
    // Early return if no external scalar potential fields are configured
    if (m_nFields == 0) { return; }

    auto& warpx = WarpX::GetInstance();

    for (int i = 0; i < m_nFields; ++i) {

        if (m_field_names[i] != "boundary_potentials" && !m_read_Phi_from_file[i]) {
            // Initialize the Phi parser
            m_Phi_external_parser[i] = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(m_Phi_ext_grid_function[i],{"x","y","z","t"}));
            m_Phi_external[i] = m_Phi_external_parser[i]->compile<4>();

            // Check if the external potential parser depends on time
            const std::set<std::string> Phi_ext_symbols = m_Phi_external_parser[i]->symbols();
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(Phi_ext_symbols.count("t") == 0,
                "Externally Applied Scalar potential time variation must be set with Phi_time_external_function(t)");
        }

        // Generate parser for time function
        m_Phi_external_time_parser[i] = std::make_unique<amrex::Parser>(
            utils::parser::makeParser(m_Phi_ext_time_function[i],{"t",}));
        m_Phi_time_scale[i] = m_Phi_external_time_parser[i]->compile<1>();
    }

    // Fill hybrid_E_fp_external_phi at the initial time
    UpdateExternalElectricField(warpx.gett_new(0), warpx.getdt(0));
}

void
ExternalScalarPotential::UpdateExternalElectricField (const amrex::Real t, [[maybe_unused]] const amrex::Real dt)
{
    ABLASTR_PROFILE("ExternalScalarPotential::UpdateExternalElectricField");

    // Early return if no external scalar potential fields are configured
    if (m_nFields == 0) { return; }

    auto& warpx = WarpX::GetInstance();
    using ablastr::fields::Direction;

    ablastr::fields::MultiLevelVectorField E_phi =
        warpx.m_fields.get_mr_levels_alldirs(FieldType::hybrid_E_fp_external_phi, warpx.finestLevel());

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        for (int idim = 0; idim < 3; ++idim) {
            E_phi[lev][Direction{idim}]->setVal(0.0_rt);
        }
    }

    // E_phi = sum_i Phi_time_i(t) * (-grad Phi_i)
    for (int i = 0; i < m_nFields; ++i) {
        if (!m_E_unit_valid[i]) { SolveUnitField(i); }

        const amrex::Real time_scale_factor = m_Phi_time_scale[i](t);
        for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
            for (int idim = 0; idim < 3; ++idim) {
                amrex::MultiFab::Saxpy(
                    *E_phi[lev][Direction{idim}], time_scale_factor,
                    *m_E_unit[i][lev][idim], 0, 0, 1,
                    E_phi[lev][Direction{idim}]->nGrowVect());
            }
        }
    }

    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        for (int idim = 0; idim < 3; ++idim) {
            E_phi[lev][Direction{idim}]->FillBoundary(warpx.Geom(lev).periodicity());
        }
    }
}

void
ExternalScalarPotential::SolveUnitField (int field_index)
{
    ABLASTR_PROFILE("ExternalScalarPotential::SolveUnitField");

    auto& warpx = WarpX::GetInstance();
    using ablastr::fields::Direction;

    // Get the electrostatic solver to compute E from Phi
    auto& es_solver = warpx.GetElectrostaticSolver();

    // Get EB update flags for masking (if EB is enabled)
    const bool eb_on = EB::enabled();
    auto& ebE = warpx.GetEBUpdateEFlag_fp();

    ablastr::fields::MultiLevelVectorField E_phi =
        warpx.m_fields.get_mr_levels_alldirs(FieldType::hybrid_E_fp_external_phi, warpx.finestLevel());

    // Temporary storage for the E field of this potential
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> E_temp_storage(warpx.finestLevel() + 1);
    ablastr::fields::MultiLevelVectorField E_temp(warpx.finestLevel() + 1);
    m_E_unit[field_index].resize(warpx.finestLevel() + 1);
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        for (int idim = 0; idim < 3; ++idim) {
            amrex::MultiFab const& Eref = *E_phi[lev][Direction{idim}];
            E_temp_storage[lev][idim] = std::make_unique<amrex::MultiFab>(
                Eref.boxArray(), Eref.DistributionMap(), 1, Eref.nGrowVect());
            E_temp_storage[lev][idim]->setVal(0.0_rt);
            m_E_unit[field_index][lev][idim] = std::make_unique<amrex::MultiFab>(
                Eref.boxArray(), Eref.DistributionMap(), 1, Eref.nGrowVect());
            m_E_unit[field_index][lev][idim]->setVal(0.0_rt);
        }
        E_temp[lev] = {
            E_temp_storage[lev][0].get(),
            E_temp_storage[lev][1].get(),
            E_temp_storage[lev][2].get()
        };
    }

    if (m_field_names[field_index] == "boundary_potentials") {
        // Vacuum Laplace solve with boundary.potential_* and warpx.eb_potential
        es_solver.AddBoundaryField(E_temp);
    } else {
        // Custom Phi_external_grid_function: set it as EB potential and use Poisson solve
        es_solver.m_poisson_boundary_handler->setPotentialEB(m_Phi_ext_grid_function[field_index]);
        es_solver.AddBoundaryField(E_temp);
    }

    // Copy into the per-field cache, leaving edges covered by the EB at zero
    for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
        for (int idim = 0; idim < 3; ++idim) {
            amrex::MultiFab& E_dst = *m_E_unit[field_index][lev][idim];
            amrex::MultiFab const& E_src = *E_temp[lev][idim];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
            for (amrex::MFIter mfi(E_dst, amrex::TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                auto const dst = E_dst.array(mfi);
                auto const src = E_src.const_array(mfi);

                // EB mask: 0 means the edge is covered by the EB
                amrex::Array4<int const> mask;
                if (eb_on && ebE[lev][idim]) {
                    mask = ebE[lev][idim]->const_array(mfi);
                }

                const amrex::Box& tb = mfi.tilebox(E_dst.ixType().toIntVect());
                amrex::ParallelFor(tb, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    if (mask && mask(i,j,k) == 0) { return; }
                    dst(i,j,k) = src(i,j,k);
                });
            }
        }
    }
    m_E_unit_valid[field_index] = true;
}
