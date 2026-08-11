/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * Authors: Marco Garten
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_EXTERNAL_SCALAR_POTENTIAL_H_
#define WARPX_EXTERNAL_SCALAR_POTENTIAL_H_

#include "Fields.H"

#include "Utils/WarpXAlgorithmSelection.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/WarpXConst.H"
#include <ablastr/profiler/ProfilerWrapper.H>

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_BoxArray.H>
#include <AMReX_IntVect.H>
#include <AMReX_DistributionMapping.H>

#include <optional>

/**
 * \brief This class contains the parameters needed to evaluate a
 * time-varying external scalar potential (Phi_ext), leading to external
 * electric fields to be applied in Hybrid Solver. This class handles
 * spatial and time-dependent potentials.
 *
 * The electric field is computed via E = -grad(Phi) using a Poisson solve
 * with specified boundary conditions.
 *
 * This can be used independently or in combination with ExternalVectorPotential
 * to provide complete external electromagnetic field specifications.
 */
class ExternalScalarPotential
{
protected:
    int m_nFields = 0;

    std::vector<std::string> m_field_names;

    std::vector<std::string> m_Phi_ext_grid_function;
    std::vector<std::unique_ptr<amrex::Parser>> m_Phi_external_parser;
    std::vector<amrex::ParserExecutor<4>> m_Phi_external;

    std::vector<std::string> m_Phi_ext_time_function;
    std::vector<std::unique_ptr<amrex::Parser>> m_Phi_external_time_parser;
    std::vector<amrex::ParserExecutor<1>> m_Phi_time_scale;

    std::vector<bool> m_read_Phi_from_file;
    std::vector<std::string> m_external_file_path;

    // Track if time dependence exists
    bool m_has_time_dependence = false;

    // Store previous time scale values to detect changes
    std::vector<amrex::Real> m_prev_time_scale;

    // Cache for static E field (computed once, reused every timestep)
    // Indexed as m_E_cached[lev][dim]
    amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>> m_E_cached;
    bool m_cache_valid = false;

public:

    // Default Constructor
    ExternalScalarPotential ();

    void ReadParameters ();

    /** Check if this class has external fields that need hybrid_E_fp_external */
    [[nodiscard]] bool HasExternalFields () const { return m_nFields > 0; }

    /**
     * \brief Initialize external scalar potential data.
     * 
     * This performs the initial Poisson solve to compute E = -grad(Phi)
     * and adds the result to hybrid_E_fp_external.
     */
    void InitData ();

    /**
     * \brief Update the external electric field from scalar potential at time t.
     * 
     * For time-varying potentials, this re-evaluates Phi(t) and if it has changed,
     * re-solves Poisson equation to update E_ext.
     * 
     * \param[in] t current simulation time
     * \param[in] dt timestep (unused for scalar potential, kept for interface consistency)
     */
    void UpdateExternalElectricField (
        amrex::Real t,
        amrex::Real dt
    );

    /**
     * \brief Add external E field from scalar potential to hybrid_E_fp_external.
     * 
     * This solves the Poisson equation with the specified Phi boundary conditions
     * and zero charge density, computing E = -grad(Phi), then adds the result
     * to the existing hybrid_E_fp_external field.
     * 
     * \param[in] field_index Index of the field in m_field_names to process
     * \param[in] time_scale_factor Multiplicative factor for time dependence
     */
    void AddToExternalElectricField (
        int field_index,
        amrex::Real time_scale_factor
    );

};

// Note: Custom Phi_external_grid_function is implemented by setting the
// Phi function as an embedded boundary potential in the PoissonBoundaryHandler,
// then using the existing Poisson solver via AddBoundaryField to compute E = -grad(Phi).
// This approach reuses existing infrastructure and properly handles embedded boundaries.

#endif // WARPX_EXTERNAL_SCALAR_POTENTIAL_H_
