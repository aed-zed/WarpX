/* Copyright 2021-2022 The WarpX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyWarpX.H"

#include <WarpX.H>
// see WarpX.cpp - full includes for _fwd.H headers
#include <BoundaryConditions/PEC_Insulator.H>
#include <BoundaryConditions/PML.H>
#include <Diagnostics/MultiDiagnostics.H>
#include <Diagnostics/ReducedDiags/MultiReducedDiags.H>
#include <EmbeddedBoundary/WarpXFaceInfoBox.H>
#include <FieldSolver/ElectrostaticSolvers/StaircaseBias.H>
#include <FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H>
#include <FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H>
#include <FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H>
#include <FieldSolver/ImplicitSolvers/ImplicitSolver.H>
#ifdef WARPX_USE_FFT
#   include <FieldSolver/SpectralSolver/SpectralKSpace.H>
#   ifdef WARPX_DIM_RZ
#       include <FieldSolver/SpectralSolver/SpectralSolverRZ.H>
#       include <BoundaryConditions/PML_RZ.H>
#   else
#       include <FieldSolver/SpectralSolver/SpectralSolver.H>
#   endif // RZ ifdef
#endif // use PSATD ifdef
#include <FieldSolver/WarpX_FDTD.H>
#include <Filter/NCIGodfreyFilter.H>
#include <Initialization/ExternalField.H>
#include <Particles/MultiParticleContainer.H>
#include <Fluids/MultiFluidContainer.H>
#include <Fluids/WarpXFluidContainer.H>
#include <Particles/ParticleBoundaryBuffer.H>
#include <AcceleratorLattice/AcceleratorLattice.H>
#include <Utils/TextMsg.H>
#include <Utils/WarpXAlgorithmSelection.H>
#include <Utils/WarpXConst.H>
#include <Utils/WarpXUtil.H>
#include "FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.H"

#include <ablastr/profiler/ProfilerWrapper.H>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_OpenMP.H>

#if defined(AMREX_DEBUG) || defined(DEBUG)
#   include <cstdio>
#endif
#include <memory>
#include <string>


//using namespace warpx;

namespace detail
{
    /** Helper Function for Property Getters
     *
     * This queries an amrex::ParmParse entry. This throws a
     * std::runtime_error if the entry is not found.
     *
     * This handles the most common throw exception logic in WarpX instead of
     * going over library boundaries via amrex::Abort().
     *
     * @tparam T type of the amrex::ParmParse entry
     * @param prefix the prefix, e.g., "warpx" or "amr"
     * @param name the actual key of the entry, e.g., "particle_shape"
     * @return the queried value (or throws if not found)
     */
    template< typename T>
    auto get_or_throw (std::string const & prefix, std::string const & name)
    {
        using V = std::decay_t<T>;
        V value;

        bool has_name = false;
        // TODO: if array do queryarr
        // has_name = amrex::ParmParse(prefix).queryarr(name.c_str(), value);
        if constexpr (std::is_same_v<V, bool> || std::is_same_v<V, std::string>) {
            has_name = amrex::ParmParse(prefix).query(name.c_str(), value);
        }
        else {
            has_name = amrex::ParmParse(prefix).queryWithParser(name.c_str(), value);
        }

        if (!has_name) {
            throw std::runtime_error(prefix + "." + name + " is not set yet");
        }
        return value;
    }
}

void init_WarpX (py::module& m)
{
    using ablastr::fields::Direction;

    // Expose the WarpX instance
    m.def("get_instance",
        [] () { return &WarpX::GetInstance(); },
        "Return a reference to the WarpX object.");

    m.def("finalize", &WarpX::Finalize,
        "Close out the WarpX related data");

    // WarpX is a singleton owned by the C++ side: its lifetime ends in
    // WarpX::Finalize (i.e. WarpX::ResetInstance), never when the last Python
    // reference goes away. Without py::nodelete, pybind11's default
    // return_value_policy for the raw pointer returned by get_instance below is
    // take_ownership, and destroying the Python object would leave
    // WarpX::m_instance dangling and WarpX::Finalize double-freeing it.
    py::class_<WarpX, std::unique_ptr<WarpX, py::nodelete>> warpx(m, "WarpX");
    warpx
        // WarpX is a Singleton Class with a private constructor
        //   https://github.com/BLAST-WarpX/warpx/pull/4104
        //   https://pybind11.readthedocs.io/en/stable/advanced/classes.html?highlight=singleton#custom-constructors
        .def(py::init([]() {
            return &WarpX::GetInstance();
        }))
        .def_static("get_instance",
            [] () { return &WarpX::GetInstance(); },
            "Return a reference to the WarpX object."
        )
        .def_static("finalize", &WarpX::Finalize,
            "Close out the WarpX related data"
        )

        .def("initialize_data", &WarpX::InitData,
            "Initializes the WarpX simulation"
        )
        .def("evolve", &WarpX::Evolve,
            "Evolve the simulation the specified number of steps"
        )

        .def_property("omp_threads",
            [](WarpX & /* wx */){
                return detail::get_or_throw<std::string>("amrex", "omp_threads");
            },
            [](WarpX & /* wx */, std::variant<int, std::string> omp_threads_var) {
                std::visit([&]( auto && omp_threads) {
                    amrex::ParmParse pp_amrex("amrex");
                    pp_amrex.add("omp_threads", omp_threads);

                    // set the value if not "system" or "nosmt"
                    if constexpr(std::is_same_v<std::decay_t<decltype(omp_threads)>, int>) {
                        amrex::Print() << "Changing WarpX threads to N=" << omp_threads << "\n";
                        amrex::OpenMP::set_num_threads(omp_threads);
                    }
                }, omp_threads_var);
            },
            "Controls the number of OpenMP threads to use (WarpX default: \"nosmt\").\n"
            "https://amrex-codes.github.io/amrex/docs_html/InputsComputeBackends.html."
        )

        // from amrex::AmrCore / amrex::AmrMesh
        .def_property_readonly("max_level",
            [](WarpX const & wx){ return wx.maxLevel(); },
            "The maximum mesh-refinement level for the simulation."
        )
        .def_property_readonly("finest_level",
            [](WarpX const & wx){ return wx.finestLevel(); },
            "The currently finest level of mesh-refinement used. This is always less or equal to max_level."
        )
        .def("Geom",
            //[](WarpX const & wx, int const lev) { return wx.Geom(lev); },
            py::overload_cast< int >(&WarpX::Geom, py::const_),
            py::arg("lev")
        )
        .def("DistributionMap",
            [](WarpX const & wx, int const lev) { return wx.DistributionMap(lev); },
            //py::overload_cast< int >(&WarpX::DistributionMap, py::const_),
            py::arg("lev")
        )
        .def("boxArray",
            [](WarpX const & wx, int const lev) { return wx.boxArray(lev); },
            //py::overload_cast< int >(&WarpX::boxArray, py::const_),
            py::arg("lev")
        )
        .def("multifab_register",&WarpX::GetMultiFabRegister,
            py::return_value_policy::reference_internal)

        .def("multi_particle_container",
            [](WarpX& wx){ return &wx.GetPartContainer(); },
            py::return_value_policy::reference_internal
        )
        .def("get_particle_boundary_buffer",
            [](WarpX& wx){ return &wx.GetParticleBoundaryBuffer(); },
            py::return_value_policy::reference_internal
        )

        // Expose the implicit solver and the mass matrices deposition
        .def("implicit_solver",
            [](WarpX& wx){ return wx.get_pointer_ImplicitSolver(); },
            py::return_value_policy::reference_internal,
            R"pbdoc(Return the implicit solver, or None when the evolve scheme is explicit)pbdoc"
        )
        .def("save_particles_at_implicit_step_start",
            [](WarpX& wx){ wx.SaveParticlesAtImplicitStepStart(); },
            R"pbdoc(Save the particle positions and velocities at the start of the step)pbdoc"
        )
        .def("deposit_mass_matrices",
            [](WarpX& wx){ wx.DepositMassMatrices(); },
            R"pbdoc(Zero and deposit the mass matrices from all species)pbdoc"
        )
        .def("sync_mass_matrices",
            [](WarpX& wx){ wx.SyncMassMatrices(); },
            R"pbdoc(Sum the guard cells of the mass matrices into the valid cells)pbdoc"
        )

        // Expose functions used to sync the current and charge density multifabs
        // accross tiles and apply appropriate boundary conditions
        .def("sync_current",
            [](WarpX& wx, const std::string& current_fp_string){ wx.SyncCurrent(current_fp_string); },
            py::arg("current_fp_string"),
            R"pbdoc(Sum the guard cells of a current-like vector field into the valid cells)pbdoc"
        )
        .def("sync_rho",
            [](WarpX& wx){ wx.SyncRho(); }
        )
#if defined(WARPX_DIM_RZ) || defined(WARPX_DIM_RCYLINDER) || defined(WARPX_DIM_RSPHERE)
        .def("apply_inverse_volume_scaling_to_charge_density",
            [](WarpX& wx, amrex::MultiFab* rho, int const lev) {
                wx.ApplyInverseVolumeScalingToChargeDensity(rho, lev);
            },
            py::arg("rho"), py::arg("lev")
        )
#endif

        // Expose functions to get the current simulation step and time
        .def("getistep",
            [](WarpX const & wx, int lev){ return wx.getistep(lev); },
            py::arg("lev"),
            "Get the current step on mesh-refinement level ``lev``."
        )
        .def("gett_new",
            [](WarpX const & wx, int lev){ return wx.gett_new(lev); },
            py::arg("lev"),
            "Get the current physical time on mesh-refinement level ``lev``."
        )
        .def("getdt",
            [](WarpX const & wx, int lev){ return wx.getdt(lev); },
            py::arg("lev"),
            "Get the current physical time step size on mesh-refinement level ``lev``."
        )

        .def("set_potential_on_domain_boundary",
            [](WarpX& wx,
               std::string potential_lo_x, std::string potential_hi_x,
               std::string potential_lo_y, std::string potential_hi_y,
               std::string potential_lo_z, std::string potential_hi_z)
            {
                if (potential_lo_x != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_xlo_str = potential_lo_x;
                if (potential_hi_x != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_xhi_str = potential_hi_x;
                if (potential_lo_y != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_ylo_str = potential_lo_y;
                if (potential_hi_y != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_yhi_str = potential_hi_y;
                if (potential_lo_z != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_zlo_str = potential_lo_z;
                if (potential_hi_z != "") wx.GetElectrostaticSolver().m_poisson_boundary_handler->potential_zhi_str = potential_hi_z;
                wx.GetElectrostaticSolver().m_poisson_boundary_handler->BuildParsers();
            },
            py::arg("potential_lo_x") = "",
            py::arg("potential_hi_x") = "",
            py::arg("potential_lo_y") = "",
            py::arg("potential_hi_y") = "",
            py::arg("potential_lo_z") = "",
            py::arg("potential_hi_z") = "",
            "Sets the domain boundary potential string(s) and updates the function parser."
        )
        .def("set_potential_on_eb",
            [](WarpX& wx, std::string potential) {
                wx.GetElectrostaticSolver().m_poisson_boundary_handler->setPotentialEB(potential);
            },
            py::arg("potential"),
            "Sets the EB potential string and updates the function parser."
        )
        .def("compute_div_e",
            [] (WarpX& wx, int const lev, std::string const& field) {
                if (field != "Efield_fp" && field != "Efield_aux") {
                    throw py::value_error("compute_div_e field must be Efield_fp or Efield_aux");
                }
                // WarpX computes divE on the nodes, matching nodal rho and the
                // operator the nodal Poisson solver inverts. ComputeDivE
                // dispatches per geometry, so the cylindrical
                // (1/r) d(r E_r)/dr form is used in RZ without a second kernel
                // here. Additive binding to an existing native operator: no new
                // discretisation is introduced.
                amrex::BoxArray nodal_ba = wx.boxArray(lev);
                nodal_ba.surroundingNodes();
                amrex::MultiFab div_e(
                    nodal_ba, wx.DistributionMap(lev), WarpX::ncomps, 0);
                wx.ComputeDivE(div_e, lev,
                    field == "Efield_fp" ? warpx::fields::FieldType::Efield_fp
                                         : warpx::fields::FieldType::Efield_aux);
                return div_e;
            },
            py::arg("lev") = 0, py::arg("field") = "Efield_fp",
            py::return_value_policy::move,
            "Nodal divergence of Efield_fp (or Efield_aux) as a new MultiFab, "
            "using WarpX::ComputeDivE (cylindrical form in RZ)."
        )
#if defined(WARPX_DIM_RZ)
        .def("rz_axis_volume_factor",
            [] (WarpX const & wx) { return wx.RZAxisVolumeFactor(); },
            "Axis-node radial volume factor of charge deposition: 1/3 with the "
            "Verboncoeur correction, 1/4 without."
        )
#endif
        .def("eb_update_e_flag",
            [] (WarpX& wx, int const lev, int const dir) {
                auto& flags = wx.GetEBUpdateEFlag();
                if (lev < 0 || lev >= static_cast<int>(flags.size())
                    || dir < 0 || dir > 2 || !flags[lev][dir]) {
                    throw py::value_error(
                        "eb_update_e_flag: no EB update mask for this level/direction "
                        "(is an embedded boundary defined?)");
                }
                return flags[lev][dir].get();
            },
            py::arg("lev") = 0, py::arg("dir") = 0,
            py::return_value_policy::reference_internal,
            "Read-only EB update mask of one E component: 1 where the field is "
            "advanced, 0 where it is frozen (staircase)."
        )
        .def("deposit_scratch_rho",
            [] (WarpX& wx, int const lev) {
                // Fresh nodal charge density from all live species, filtered and
                // guard-cell summed exactly as the electrostatic solvers consume
                // it, on a scratch allocation that leaves the registered rho_fp
                // untouched. A volume charge observer needs this to subtract the
                // plasma charge inside its region with the SAME measure the
                // field was built from; reading rho_fp is not an option because
                // it is not allocated in a plain electromagnetic run.
                auto rho = wx.DepositScratchRho(lev);
                return amrex::MultiFab(std::move(*rho));
            },
            py::arg("lev") = 0,
            py::return_value_policy::move,
            "Freshly deposited nodal charge density of all species as a new MultiFab "
            "(rho_fp is not touched)."
        )
        .def("solve_staircase_unit_bias",
            [] (WarpX&, std::string const& selector, std::string const& out_phi,
                std::string const& out_weight, std::string const& out_efield,
                amrex::Real rtol, int max_iter, bool insulating_endcaps,
                bool grounded_wall_reference) {
                return WarpXSolveStaircaseUnitBias(
                    selector, out_phi, out_weight, out_efield, rtol, max_iter,
                    insulating_endcaps, grounded_wall_reference);
            },
            py::arg("selector"), py::arg("out_phi"), py::arg("out_weight"),
            py::arg("out_efield"), py::arg("rtol") = 1.e-12,
            py::arg("max_iter") = 200,
            py::arg("insulating_endcaps") = false,
            py::arg("grounded_wall_reference") = false,
            "Staircase unit-potential solve for the conductor selected by `selector` "
            "(binary, constant on each frozen-edge component). Fills registered "
            "outputs; live E is unchanged. Returns the solve residual."
        )
        .def("solve_staircase_insulator_bias",
            [] (WarpX&, std::string const& selector, std::string const& out_psi,
                std::string const& out_weight, std::string const& out_efield,
                amrex::Real rtol, int max_iter, bool grounded_wall_reference) {
                return WarpXSolveStaircaseInsulatorBias(
                    selector, out_psi, out_weight, out_efield, rtol, max_iter,
                    grounded_wall_reference);
            },
            py::arg("selector"), py::arg("out_psi"), py::arg("out_weight"),
            py::arg("out_efield"), py::arg("rtol") = 1.e-12,
            py::arg("max_iter") = 200,
            py::arg("grounded_wall_reference") = false,
            "Staircase unit solve with insulating z faces: Neumann observer potential "
            "(out_psi) and a fringing actuator field (out_efield). Returns the residual."
        )
        .def("solve_staircase_grounded",
            [] (WarpX& /*wx*/, std::string const& out_phi, std::string const& out_weight,
                std::string const& out_efield, amrex::Real const rtol, int const max_iter) {
                return WarpXSolveStaircaseGrounded(out_phi, out_weight, out_efield,
                                                   rtol, max_iter);
            },
            py::arg("out_phi"), py::arg("out_weight"), py::arg("out_efield"),
            py::arg("rtol") = 1.e-12, py::arg("max_iter") = 200,
            "Grounded staircase Poisson solve with the live charge as source (all "
            "conductors at 0 V). Fills registered output fields; returns the residual."
        )
        .def("staircase_grounded_pairing",
            [] (WarpX&, std::string const& requested, bool insulating_endcaps) {
                return WarpXStaircaseGroundedPairing(requested, insulating_endcaps);
            },
            py::arg("requested") = "auto",
            py::arg("insulating_endcaps") = false,
            "Effective staircase grounded-charge pairing ('field' or 'rho') for the "
            "requested mode ('auto', 'field' or 'rho') and the z boundaries."
        )
        .def("staircase_charge_state",
            [] (WarpX&, std::vector<std::string> const& psi_fields,
                std::vector<std::string> const& weight_fields, bool insulating_endcaps,
                std::string const& grounded_pairing) {
                auto const q = WarpXStaircaseChargeState(
                    psi_fields, weight_fields, insulating_endcaps, grounded_pairing);
                std::vector<std::vector<amrex::Real>> result;
                for (auto const& row : q) {
                    result.emplace_back(row.begin(), row.end());
                }
                return result;
            },
            py::arg("psi_fields"), py::arg("weight_fields"),
            py::arg("insulating_endcaps") = false,
            py::arg("grounded_pairing") = "auto",
            "Staircase observer per conductor, in coulombs: (Gauss charge, live charge "
            "on fixed nodes, grounded charge); with insulating_endcaps a fourth "
            "row holds the axial face flux. Collective; no Poisson solve."
        )
        .def("validate_staircase_weights",
            [] (WarpX&, std::vector<std::string> const& weight_fields,
                bool grounded_wall_reference) {
                return WarpXValidateStaircaseWeights(weight_fields, grounded_wall_reference);
            },
            py::arg("weight_fields"),
            py::arg("grounded_wall_reference") = false,
            "Setup check: maximum error of the summed unit weights on fixed nodes "
            "(0 = every conductor selected exactly once)."
        )
        .def("refresh_staircase_efield_guards",
            [] (WarpX& wx) {
                // A staircase correction is added to the valid E region after
                // the field push.  Refresh physical guards first, then exchange
                // inter-box guards, in the same order as an explicit field push.
                wx.ApplyEfieldBoundary(0, PatchType::fine, wx.gett_new(0));
                wx.FillBoundaryE(0, wx.getngEB(), true);
            },
            "Refresh Efield_fp guard cells (level 0) after a direct field update."
        )
        .def("run_div_cleaner",
            [] (WarpX& wx) { wx.ProjectionCleanDivB(); },
            "Executes projection based divergence cleaner on loaded Bfield_fp_external."
        )
        .def_static("calculate_hybrid_external_curlA",
            [] (WarpX& wx) { wx.CalculateExternalCurlA(); },
            "Executes calculation of the curl of the external A in the hybrid solver."
        )
        .def("synchronize_velocity_with_position",
            [] (WarpX& wx) { wx.SynchronizeVelocityWithPosition(); },
            "Synchronize particle velocities and positions."
        )
        // Add some accessor bindings for the Hybrid Ohm's Law Solver
        .def("set_hybrid_pic_substeps",
            [](WarpX& wx, int substeps) {
                wx.get_pointer_HybridPICModel()->m_substeps = substeps;
            },
            py::arg("substeps"),
            "Sets the number of substeps to take in the hybrid solver."
        )
        .def("get_hybrid_pic_substeps",
            [](WarpX& wx) {
                return wx.get_pointer_HybridPICModel()->m_substeps;
            },
            "Gets the number of substeps taken in the hybrid solver."
        )
        .def("set_hybrid_pic_density_floor",
            [](WarpX& wx, amrex::Real n_floor) {
                wx.get_pointer_HybridPICModel()->m_n_floor = n_floor;
            },
            py::arg("n_floor"),
            "Sets the density floor to use in the hybrid solver."
        )
        .def("get_hybrid_pic_density_floor",
            [](WarpX& wx) {
                return wx.get_pointer_HybridPICModel()->m_n_floor;
            },
            "Gets the number of substeps to take in the hybrid solver."
        )
    ;
}
