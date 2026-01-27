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
        const ParmParse pp_hybrid("hybridpicmodel");
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
    m_prev_time_scale.resize(m_nFields, -std::numeric_limits<Real>::max());

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
    using ablastr::fields::Direction;
    auto& warpx = WarpX::GetInstance();

    for (int i = 0; i < m_nFields; ++i) {

        // Skip parser initialization for the special "boundary_potentials" field
        if (m_field_names[i] == "boundary_potentials") {
            // For boundary potentials, time function still applies
            m_Phi_external_time_parser[i] = std::make_unique<amrex::Parser>(
                utils::parser::makeParser(m_Phi_ext_time_function[i],{"t",}));
            m_Phi_time_scale[i] = m_Phi_external_time_parser[i]->compile<1>();

            const std::set<std::string> time_symbols = m_Phi_external_time_parser[i]->symbols();
            if (time_symbols.count("t") > 0) {
                m_has_time_dependence = true;
            }
            continue;
        }

        if (!m_read_Phi_from_file[i]) {
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

        // Check if time function is non-constant
        const std::set<std::string> time_symbols = m_Phi_external_time_parser[i]->symbols();
        if (time_symbols.count("t") > 0) {
            m_has_time_dependence = true;
        }
    }

    // Perform initial field update at t=0
    UpdateExternalElectricField(warpx.gett_new(0), warpx.getdt(0));
}

void
ExternalScalarPotential::UpdateExternalElectricField (const amrex::Real t, const amrex::Real dt)
{
    WARPX_PROFILE("ExternalScalarPotential::UpdateExternalElectricField");

    // Iterate over external fields and add contributions with individual time functions
    for (int i = 0; i < m_nFields; ++i) {
        // Get time scaling factor
        const amrex::Real time_scale_factor = m_Phi_time_scale[i](t);

        // For time-varying potentials, only update if the time scale has changed
        // significantly (or if this is the first call)
        // \TODO Expose control to user. Also add option to update potential every N steps.
        const bool needs_update = !m_has_time_dependence || 
                                  std::abs(time_scale_factor - m_prev_time_scale[i]) > 1e-14;

        if (needs_update) {
            AddToExternalElectricField(i, time_scale_factor);
            m_prev_time_scale[i] = time_scale_factor;
        }
    }
}

void
ExternalScalarPotential::AddToExternalElectricField (
    int field_index,
    amrex::Real time_scale_factor)
{
    WARPX_PROFILE("ExternalScalarPotential::AddToExternalElectricField");

    auto& warpx = WarpX::GetInstance();

    // Get the electrostatic solver to compute E from Phi
    auto& es_solver = warpx.GetElectrostaticSolver();

    // Get reference to the external E field that we'll add to
    ablastr::fields::MultiLevelVectorField Efield_external =
        warpx.m_fields.get_mr_levels_alldirs(FieldType::hybrid_E_fp_external, warpx.finestLevel());

    // Handle the special case of boundary potentials
    if (m_field_names[field_index] == "boundary_potentials") {
        // Use the existing boundary potential system via AddBoundaryField
        // This uses boundary.potential_* and warpx.eb_potential from the input
        
        // \TODO replace black-box hardcode with user control
        if (std::abs(time_scale_factor - 1.0_rt) < 1e-14) {
            // No time scaling needed, just add directly
            es_solver.AddBoundaryField(Efield_external);
        } else {
            // Need to apply time scaling
            // Strategy: compute E into temporary field, scale it, then add
            
            // Create temporary storage for the E field from this potential
            amrex::Vector<std::unique_ptr<amrex::MultiFab>> E_temp(warpx.finestLevel() + 1);
            for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
                const auto& ba = Efield_external[lev][ablastr::fields::Direction{0}]->boxArray();
                const auto& dm = Efield_external[lev][ablastr::fields::Direction{0}]->DistributionMap();
                E_temp[lev] = std::make_unique<amrex::MultiFab>(ba, dm, 3, 
                    Efield_external[lev][ablastr::fields::Direction{0}]->nGrowVect());
                E_temp[lev]->setVal(0.0_rt);
            }

            // Compute E from boundary potential into temporary field
            // Note: We need to create a temporary MultiLevelVectorField view
            ablastr::fields::MultiLevelVectorField E_temp_view(warpx.finestLevel() + 1);
            for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
                E_temp_view[lev] = {
                    E_temp[lev].get(),
                    E_temp[lev].get(), 
                    E_temp[lev].get()
                };
            }
            es_solver.AddBoundaryField(E_temp_view);

            // Scale and add to external field
            for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
                for (int idim = 0; idim < 3; ++idim) {
                    Efield_external[lev][ablastr::fields::Direction{idim}]->saxpy(
                        time_scale_factor, *E_temp[lev], idim, 0, 1, 
                        Efield_external[lev][ablastr::fields::Direction{idim}]->nGrowVect()
                    );
                }
            }
        }
    } else {
        // Custom Phi_external_grid_function: set it as EB potential and use Poisson solve
        // This approach reuses the existing Poisson solver infrastructure
        
        auto& es_solver = warpx.GetElectrostaticSolver();
        auto& boundary_handler = es_solver.m_poisson_boundary_handler;
        
        // Set the custom Phi as the EB potential
        // The function is Phi(x,y,z,t) where time scaling is applied
        std::string phi_with_time = m_Phi_ext_grid_function[field_index];
        if (std::abs(time_scale_factor - 1.0_rt) > 1e-14) {
            // If time scaling is non-unity, we need to handle it
            // For now, evaluate at specific time and scale
            // TODO: Ideally scale the expression itself
            amrex::Print() << Utils::TextMsg::Warn(
                "ExternalScalarPotential: Time-varying custom Phi with scaling factor " 
                + std::to_string(time_scale_factor) + " not fully supported yet.\n"
                "        Time scaling will be approximate.\n"
            );
        }
        
        boundary_handler->setPotentialEB(phi_with_time);
        
        // Now compute E field from this potential via Poisson solve
        es_solver.AddBoundaryField(Efield_external);
        
        // Apply time scaling if needed
        if (std::abs(time_scale_factor - 1.0_rt) > 1e-14) {
            for (int lev = 0; lev <= warpx.finestLevel(); ++lev) {
                for (int idim = 0; idim < 3; ++idim) {
                    // Scale the E field that was just computed
                    Efield_external[lev][ablastr::fields::Direction{idim}]->mult(time_scale_factor);
                }
            }
        }
    }
}
