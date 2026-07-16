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
#include <Diagnostics/FullDiagnostics.H>
#include <Diagnostics/MultiDiagnostics.H>
#include <Diagnostics/ReducedDiags/MultiReducedDiags.H>
#include <EmbeddedBoundary/WarpXFaceInfoBox.H>
#include <FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H>
#include <FieldSolver/FiniteDifferenceSolver/MacroscopicProperties/MacroscopicProperties.H>
#include <FieldSolver/FiniteDifferenceSolver/HybridPICModel/HybridPICModel.H>
#ifdef WARPX_USE_FFT
#   include <FieldSolver/SpectralSolver/SpectralKSpace.H>
#   ifdef WARPX_DIM_RZ
#       include <FieldSolver/SpectralSolver/SpectralSolverRZ.H>
#       include <BoundaryConditions/PML_RZ.H>
#   else
#       include <FieldSolver/SpectralSolver/SpectralSolver.H>
#   endif // RZ ifdef
#endif // use PSATD ifdef
#include <FieldSolver/ElectrostaticSolvers/RelativisticExplicitES.H>
#include <FieldSolver/WarpX_FDTD.H>
#include <Filter/NCIGodfreyFilter.H>
#include <Initialization/ExternalField.H>
#include <Particles/MultiParticleContainer.H>
#include <Fluids/MultiFluidContainer.H>
#include <Fluids/WarpXFluidContainer.H>
#include <Particles/ParticleBoundaryBuffer.H>
#include <AcceleratorLattice/AcceleratorLattice.H>
#include <Utils/TextMsg.H>
#include <Utils/Parser/ParserUtils.H>
#include <Utils/WarpXAlgorithmSelection.H>
#include <Utils/WarpXConst.H>
#include <Utils/WarpXProfilerWrapper.H>
#include <Utils/WarpXUtil.H>
#include "FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.H"
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_SIMD.H>
#include <AMReX_OpenMP.H>

#if defined(AMREX_DEBUG) || defined(DEBUG)
#   include <cstdio>
#endif
#include <string>


//using namespace warpx;

namespace warpx {
    struct Config {};
}

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

    py::class_<WarpX> warpx(m, "WarpX");
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

        // Expose functions used to sync the charge density multifab
        // accross tiles and apply appropriate boundary conditions
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
        .def("add_boundary_electrostatic_field",
            [] (WarpX& wx) {
                auto Efield_fp = wx.m_fields.get_mr_levels_alldirs("Efield_fp", wx.maxLevel());
                wx.GetElectrostaticSolver().AddBoundaryField( Efield_fp );
            },
            "Compute the electric field due to the potential specified on the domain boundaries and embedded boundaries."
        )
        .def("solve_poisson_efield",
            [] (WarpX& wx, bool force_plain_gradient) {
                wx.SolvePoissonEfield(force_plain_gradient);
            },
            py::arg("force_plain_gradient") = false,
            "Deposit charge from all species, solve Poisson with current EB/domain BCs, "
            "and replace Efield_fp with the result. force_plain_gradient=True bypasses "
            "the EB-aware E computation and uses the plain computePhi+computeE path "
            "(diagnostic for cut-edge sign attribution)."
        )
        .def("clean_efield_gauss_homogeneous",
            [] (WarpX& wx) { wx.SolvePoissonEfieldHomogeneousClean(); },
            "Homogeneous Boris/Marder Gauss clean of Efield_fp: subtract grad(psi) with "
            "nabla^2 psi = div(E) - rho/eps0 and psi = 0 on all boundaries. Cleans Gauss's "
            "law and preserves curl(E); does not reset the electrode potential."
        )
        .def("compute_eb_charge",
            [] (WarpX& wx, const std::string& weighting, const std::string& field) {
                int const lev = 0;
                ablastr::fields::VectorField E = {
                    wx.m_fields.get(field, ablastr::fields::Direction{0}, lev),
                    wx.m_fields.get(field, ablastr::fields::Direction{1}, lev),
                    wx.m_fields.get(field, ablastr::fields::Direction{2}, lev)
                };
                if (weighting.empty() || weighting == "1") {
                    return wx.ComputeEBChargeWeighted(E, lev, nullptr);
                }
                amrex::Parser parser = utils::parser::makeParser(weighting, {"x", "y", "z"});
                return wx.ComputeEBChargeWeighted(E, lev, &parser);
            },
            py::arg("weighting") = "1",
            py::arg("field") = "Efield_fp",
            "Induced charge eps0 * oint w(x,y,z) E.n dS over the embedded boundary for the "
            "named field (default Efield_fp), with an optional spatial weighting w(x,y,z) "
            "that selects a region/electrode (default w=1, the whole EB). 3D + EB only."
        )
        .def("saxpy_field_masked",
            [] (WarpX& wx, const std::string& target, const std::string& source,
                amrex::Real alpha, int lev) {
                wx.SaxpyFieldMasked(target, source, alpha, lev);
            },
            py::arg("target"),
            py::arg("source"),
            py::arg("alpha"),
            py::arg("lev") = 0,
            "Masked saxpy: target += alpha * source, skipping cells where "
            "m_eb_update_E == 0 (cut + covered EB cells). Used by the harmonic "
            "bias correctors to avoid writing sign-flipped cut-edge values."
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
        .def("set_hybrid_pic_shield_external_E_field_in_dense_plasma",
            [](WarpX& wx, bool shield) {
                wx.get_pointer_HybridPICModel()->m_shield_external_E_field_in_dense_plasma = shield;
            },
            py::arg("shield"),
            "Sets whether to shield the external E-field in dense plasma regions."
        )
        .def("get_hybrid_pic_shield_external_E_field_in_dense_plasma",
            [](WarpX& wx) {
                return wx.get_pointer_HybridPICModel()->m_shield_external_E_field_in_dense_plasma;
            },
            "Gets whether the external E-field is shielded in dense plasma regions."
        )
        .def("add_field_to_diagnostic",
            [](WarpX& wx, const std::string& diag_name, const std::string& field_name, int lev) {
                auto& multi_diags = wx.GetMultiDiags();
                int ndiags = multi_diags.GetTotalDiags();
                
                // Find the diagnostic by name
                for (int idiag = 0; idiag < ndiags; ++idiag) {
                    auto& diag = multi_diags.GetDiag(idiag);
                    
                    // Check if this diagnostic matches the requested name
                    if (diag.GetDiagName() == diag_name) {
                        // Check if it's a FullDiagnostics (only FullDiagnostics supports AddFieldToOutput)
                        auto* full_diag = dynamic_cast<FullDiagnostics*>(&diag);
                        if (full_diag) {
                            full_diag->AddFieldToOutput(field_name, lev);
                            return;
                        } else {
                            WARPX_ABORT_WITH_MESSAGE(
                                "add_field_to_diagnostic: Diagnostic '" + diag_name + 
                                "' is not a FullDiagnostics (only field/field diagnostics support adding fields)");
                        }
                    }
                }
                WARPX_ABORT_WITH_MESSAGE(
                    "add_field_to_diagnostic: Diagnostic '" + diag_name + "' not found");
            },
            py::arg("diag_name"),
            py::arg("field_name"),
            py::arg("lev") = 0,
            "Dynamically add a field from MultiFabRegister to diagnostic output\n"
            "Parameters:\n"
            "  diag_name: name of the diagnostic\n"
            "  field_name: name of the field in MultiFabRegister\n"
            "  lev: refinement level (default: 0)"
        )
    ;

    py::class_<warpx::Config>(m, "Config")
//        .def_property_readonly_static(
//            "warpx_version",
//            [](py::object) { return Version(); },
//            "WarpX version")
        .def_property_readonly_static(
            "have_mpi",
            [](py::object){
#ifdef AMREX_USE_MPI
                return true;
#else
                return false;
#endif
            })
        .def_property_readonly_static(
            "have_gpu",
            [](py::object){
#ifdef AMREX_USE_GPU
                return true;
#else
                return false;
#endif
            })
        .def_property_readonly_static(
            "have_omp",
            [](py::object){
#ifdef AMREX_USE_OMP
                return true;
#else
                return false;
#endif
        })
        .def_property_readonly_static(
            "have_simd",
            [](py::object const &){
#ifdef AMREX_USE_SIMD
                return true;
#else
                return false;
#endif
        })
        .def_property_readonly_static(
            "simd_size",
            [](py::object const &){
                return amrex::simd::native_simd_size_particlereal;
        })
        .def_property_readonly_static(
            "gpu_backend",
            [](py::object){
#ifdef AMREX_USE_CUDA
                return "CUDA";
#elif defined(AMREX_USE_HIP)
                return "HIP";
#elif defined(AMREX_USE_DPCPP)
                return "SYCL";
#else
                return py::none();
#endif
        })
        .def_property_readonly_static(
            "precision",
            [](py::object){
#ifdef AMREX_USE_FLOAT
                return "SINGLE";
#else
                return "DOUBLE";
#endif
        })
        .def_property_readonly_static(
            "precision_particles",
            [](py::object){
#ifdef AMREX_SINGLE_PRECISION_PARTICLES
                return "SINGLE";
#else
                return "DOUBLE";
#endif
        })
        ;
}
