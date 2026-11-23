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

#include <ablastr/fields/MultiFabRegister.H>
#include <ablastr/profiler/ProfilerWrapper.H>

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

    /** Per-field electric field -grad(Phi_i) of the spatial part of each
     *  potential, before the time factor Phi_time_external_function(t) is
     *  applied. Indexed as m_E_unit[field][lev][dim]. The total external field
     *  is hybrid_E_fp_external_phi = sum_i Phi_time_i(t) * m_E_unit[i]. */
    amrex::Vector<amrex::Vector<std::array<std::unique_ptr<amrex::MultiFab>, 3>>> m_E_unit;
    /** Whether m_E_unit[i] holds a valid solution */
    std::vector<bool> m_E_unit_valid;

    /** Solve the vacuum Laplace problem of field i and store -grad(Phi_i)
     *  (zero on edges covered by embedded boundaries) in m_E_unit[i]. */
    void SolveUnitField (int field_index);

public:

    // Default Constructor
    ExternalScalarPotential ();

    void ReadParameters ();

    /** Check if this class has external fields that need hybrid_E_fp_external_phi */
    [[nodiscard]] bool HasExternalFields () const { return m_nFields > 0; }

    /**
     * \brief Initialize external scalar potential data.
     *
     * This builds the time parsers and fills hybrid_E_fp_external_phi at the
     * initial time.
     */
    void InitData ();

    /**
     * \brief Set hybrid_E_fp_external_phi to the field of the external scalar potential at time t.
     *
     * The spatial solution of each potential is computed once and cached;
     * the time factor Phi_time_external_function(t) is applied on every call.
     *
     * \param[in] t current simulation time
     * \param[in] dt timestep (unused for scalar potential, kept for interface consistency)
     */
    void UpdateExternalElectricField (
        amrex::Real t,
        amrex::Real dt
    );

};

// Note: Custom Phi_external_grid_function is implemented by setting the
// Phi function as an embedded boundary potential in the PoissonBoundaryHandler,
// then using the existing Poisson solver via AddBoundaryField to compute E = -grad(Phi).
// This approach reuses existing infrastructure and properly handles embedded boundaries.

#endif // WARPX_EXTERNAL_SCALAR_POTENTIAL_H_
