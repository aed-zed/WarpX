/* Copyright 2021-2022 The WarpX Community
 *
 * Authors: Axel Huebl
 * License: BSD-3-Clause-LBNL
 */
#include "pyWarpX.H"

#include <WarpX.H>

// Adjoint weighting-potential solve, defined in
// Source/FieldSolver/ElectrostaticSolvers/AdjointWeightingSolve.cpp.
bool WarpXSolveAdjointWeighting (amrex::MultiFab& psi, amrex::MultiFab const& rhs,
                                 amrex::iMultiFab const& dmsk, int lev,
                                 amrex::Real tol, int max_iter,
                                 amrex::Real* final_res);
bool WarpXSolveAdjointWeightingPrecond (amrex::MultiFab& psi, amrex::MultiFab const& rhs,
                                        amrex::iMultiFab const& dmsk, int lev,
                                        amrex::Real tol, int max_iter,
                                        amrex::Real* final_res,
                                        int* iters_out = nullptr,
                                        amrex::Real prec_rtol = 1.0e-4);
void WarpXBuildAdjointRHS (amrex::MultiFab& rhs, amrex::MultiFab const& indicator,
                           amrex::iMultiFab const& dmsk, int lev);
void WarpXBuildAdjointRHSChargeFunctional (amrex::MultiFab& rhs,
                                           std::string const& region,
                                           amrex::iMultiFab const& dmsk, int lev);
void WarpXFinalizeChargeFunctionalPsi (amrex::MultiFab& psi, int lev);

// TEMPORARY DEBUG bindings -- topic-ect-gauss-rebase-2026-08 diagnostic task.
// See AdjointWeightingSolve.cpp for what they do and why; remove before merging.
void WarpXDebugForwardApply (amrex::MultiFab& y, amrex::MultiFab const& x,
                             amrex::iMultiFab const& dmsk, int lev);
void WarpXDebugComputeScale (amrex::MultiFab& scale_mf, int lev);
void WarpXDebugDumpLevelSet (amrex::MultiFab& out_mf, int lev);
void WarpXDebugPrecondApply (amrex::MultiFab& z_out, amrex::MultiFab const& v_in,
                             amrex::iMultiFab const& dmsk, int lev,
                             int variant, amrex::Real prec_rtol);

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
#include <algorithm>
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
        .def("solve_adjoint_weighting",
            [] (WarpX& wx, const std::string& region, const std::string& out_name,
                amrex::Real tol, int max_iter,
                const std::string& rhs_mode, bool add_indicator,
                const std::string& solver) {
                int const lev = 0;
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rhs_mode == "operator" || rhs_mode == "charge",
                    "solve_adjoint_weighting: rhs_mode must be \"operator\" or \"charge\"");
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    solver == "auto" || solver == "cgnr" || solver == "pmlmg",
                    "solve_adjoint_weighting: solver must be \"auto\", \"cgnr\", or \"pmlmg\"");
                auto const& eb_fact = wx.fieldEBFactory(lev);
                auto const& levset = eb_fact.getLevelSet();

                amrex::MultiFab* psi = wx.m_fields.get(out_name, lev);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(psi != nullptr,
                    "solve_adjoint_weighting: output field not registered");

                const amrex::BoxArray& ba = psi->boxArray();
                const amrex::DistributionMapping& dm = psi->DistributionMap();

                // Dirichlet mask: nodes covered by the EB are the constrained
                // rows; everything else is free. The electrode indicator is 1
                // on THIS electrode's constrained nodes only.
                amrex::iMultiFab dmsk(ba, dm, 1, 1);
                dmsk.setVal(0);
                amrex::MultiFab ind(ba, dm, 1, 1);
                ind.setVal(0.0);

                amrex::Parser rparser = utils::parser::makeParser(region, {"x","y","z"});
                auto rexe = rparser.compile<3>();
                const auto plo = wx.Geom(lev).ProbLoArray();
                const auto dx  = wx.Geom(lev).CellSizeArray();
                const amrex::Box ndom = amrex::surroundingNodes(wx.Geom(lev).Domain());
                const auto dlo = ndom.smallEnd();
                const auto dhi = ndom.bigEnd();

                for (amrex::MFIter mfi(dmsk); mfi.isValid(); ++mfi) {
                    const amrex::Box& bx = mfi.growntilebox();
                    auto const& dma = dmsk.array(mfi);
                    auto const& ina = ind.array(mfi);
                    auto const& ls  = levset.const_array(mfi);
                    amrex::ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            // Constrained rows are (a) nodes covered by the
                            // EB and (b) nodes on the grounded outer walls.
                            // Omitting (b) leaves those rows free, so the
                            // free-node operator is posed on a space that
                            // includes unconstrained boundary values.
                            const bool covered = (ls(i,j,k) >= amrex::Real(0.0));
                            const bool wall =
                                (i <= dlo[0] || i >= dhi[0] ||
                                 j <= dlo[1] || j >= dhi[1] ||
                                 k <= dlo[2] || k >= dhi[2]);
                            dma(i,j,k) = (covered || wall) ? 1 : 0;
                            if (covered) {
                                const amrex::Real x = plo[0] + i*dx[0];
                                const amrex::Real y = plo[1] + j*dx[1];
                                const amrex::Real z = plo[2] + k*dx[2];
                                ina(i,j,k) = rexe(x,y,z);
                            }
                        });
                }

                amrex::MultiFab rhs(ba, dm, 1, 1);
                if (rhs_mode == "charge") {
                    // The functional WarpX actually books charge with
                    // (WarpX::ComputeEBChargeWeighted), not the operator's own
                    // Dirichlet-row sum -- see AdjointWeightingSolve.cpp.
                    WarpXBuildAdjointRHSChargeFunctional(rhs, region, dmsk, lev);
                } else {
                    WarpXBuildAdjointRHS(rhs, ind, dmsk, lev);
                }

                amrex::Real res = -1.0;
                bool ok = false;
                std::string solver_used;
                int pmlmg_iters = -1;
                if (solver == "cgnr") {
                    ok = WarpXSolveAdjointWeighting(*psi, rhs, dmsk, lev, tol, max_iter, &res);
                    solver_used = "cgnr";
                } else {
                    // "pmlmg" or "auto": try the preconditioned BiCGSTAB route
                    // first (F4 in check_adjoint_route.py: ~14-15 outer
                    // iterations independent of resolution, vs the normal
                    // equations' squared condition number). "auto" falls back
                    // to cgnr if pmlmg fails to converge; "pmlmg" reports
                    // whatever it got, matching cgnr's existing contract of
                    // handing the caller (ok, res) and letting it decide.
                    //
                    // pmlmg's outer-iteration unit is a whole MLMG Poisson
                    // solve, not a cheap matrix-free apply -- max_iter values
                    // sized for cgnr (callers pass up to 30000-40000) would
                    // let a stagnating pmlmg burn tens of thousands of
                    // Poisson solves before ever falling back, orders of
                    // magnitude worse than the cgnr cost it exists to avoid.
                    // Measured outer iterations were 11-12, flat across
                    // nx=32..80 (F4 predicts ~14-15), so 200 is a >10x margin
                    // while still bounding worst-case cost to "one CG
                    // iteration's-worth" of Poisson solves, not thousands.
                    // The caller's own max_iter is still honoured for cgnr
                    // (either the direct "cgnr" path above, or the fallback
                    // below), so a caller who explicitly wants cgnr behaviour
                    // is unaffected.
                    constexpr int pmlmg_max_outer_iter = 200;
                    ok = WarpXSolveAdjointWeightingPrecond(
                        *psi, rhs, dmsk, lev, tol, std::min(max_iter, pmlmg_max_outer_iter),
                        &res, &pmlmg_iters);
                    solver_used = "pmlmg";
                    if (!ok && solver == "auto") {
                        amrex::Print() << "solve_adjoint_weighting: pmlmg did not "
                                       << "converge (rel.residual " << res
                                       << " after " << pmlmg_iters
                                       << " outer iterations); falling back to cgnr.\n";
                        ok = WarpXSolveAdjointWeighting(*psi, rhs, dmsk, lev, tol, max_iter, &res);
                        solver_used = "cgnr";
                    }
                }
                amrex::Print() << "solve_adjoint_weighting: solver=" << solver_used;
                if (pmlmg_iters >= 0 && solver_used == "pmlmg") {
                    amrex::Print() << " (" << pmlmg_iters << " outer iterations)";
                }
                amrex::Print() << ", converged=" << (ok ? "true" : "false")
                               << ", rel.residual=" << res << "\n";

                if (rhs_mode == "charge") {
                    // Apply MLMG's scaleRHS diagonal and the 1/(eps0*dV) unit
                    // -charge normalisation -- see WarpXFinalizeChargeFunctionalPsi.
                    WarpXFinalizeChargeFunctionalPsi(*psi, lev);
                }

                // The plain operator-mode basis sets psi = 1 on its own
                // electrode from the constrained (Dirichlet) rows -- the
                // solve above only ever determines the free nodes. The
                // charge functional has no such Dirichlet meaning (the
                // covered nodes are simply left at 0, which is the correct
                // booking value: charge deposited there is inert in the
                // grounded solve), so add_indicator defaults to off for it
                // and callers may opt out of it for "operator" mode too.
                if (add_indicator) {
                    amrex::MultiFab::Add(*psi, ind, 0, 0, 1, 0);
                }
                psi->FillBoundary(wx.Geom(lev).periodicity());
                return py::make_tuple(ok, res);
            },
            py::arg("region"), py::arg("out_name"),
            py::arg("tol") = 1.0e-10, py::arg("max_iter") = 2000,
            py::arg("rhs_mode") = "operator", py::arg("add_indicator") = true,
            py::arg("solver") = "auto",
            "Solve for the ADJOINT weighting potential Psi_k of the electrode "
            "selected by region(x,y,z), writing it into the registered nodal "
            "field out_name. rhs_mode=\"operator\" (default) reproduces the "
            "prior behaviour exactly; rhs_mode=\"charge\" builds the RHS from "
            "the charge functional WarpX actually books with "
            "(ComputeEBChargeWeighted) instead of the operator's own "
            "Dirichlet-row sum. add_indicator=True (default) sets psi=1 on "
            "this electrode's own Dirichlet nodes, the correct convention for "
            "rhs_mode=\"operator\"; pass False (required for rhs_mode=\"charge\") "
            "to leave covered nodes at 0, the correct booking value there. "
            "solver selects the linear solve for A^T psi = rhs: \"cgnr\" is "
            "CG on the normal equations (A A^T), whose squared condition "
            "number costs ~30000 iterations at nx=80; \"pmlmg\" is BiCGSTAB "
            "on A^T directly, right-preconditioned by a forward MLMG solve "
            "of A (WarpX's own EB Laplacian) -- ~11-15 outer iterations "
            "independent of resolution, and the same wall-clock cost as "
            "roughly one ordinary Poisson solve per outer iteration instead "
            "of ~30000 matrix-free applies; \"auto\" (the default) tries "
            "pmlmg and falls back to cgnr if it fails to converge. Prints "
            "which solver was actually used and the outer iteration count "
            "(pmlmg/auto only). "
            "Returns (converged, relative_residual) -- CHECK "
            "converged: an unconverged Psi yields a wrong absorption "
            "correction that looks entirely plausible. Unlike the plain "
            "unit-voltage basis, this Psi makes the grounded-charge identity "
            "Q_k = -sum_a rho_a Psi_k[a] exact on WarpX's non-symmetric EB "
            "Laplacian. 3D + EB only."
        )
        // ---------------------------------------------------------------
        // TEMPORARY DEBUG bindings -- topic-ect-gauss-rebase-2026-08.
        // Isolate whether warpx_mlebndfdlap_adotx_forward_eb (the
        // transliterated forward EB Laplacian in AdjointWeightingPotential.H)
        // reproduces AMReX's real MLEBNodeFDLaplacian::Fapply. Not for
        // production use; remove before merging.
        // ---------------------------------------------------------------
        .def("debug_adjoint_forward_apply",
            [] (WarpX& wx, const std::string& in_name, const std::string& out_name) {
                int const lev = 0;
                auto const& eb_fact = wx.fieldEBFactory(lev);
                auto const& levset = eb_fact.getLevelSet();

                amrex::MultiFab* in_mf  = wx.m_fields.get(in_name, lev);
                amrex::MultiFab* out_mf = wx.m_fields.get(out_name, lev);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(in_mf != nullptr && out_mf != nullptr,
                    "debug_adjoint_forward_apply: in/out field not registered");
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(in_mf != out_mf,
                    "debug_adjoint_forward_apply: in_name and out_name must differ");

                const amrex::BoxArray& ba = out_mf->boxArray();
                const amrex::DistributionMapping& dm = out_mf->DistributionMap();

                // Same Dirichlet mask as solve_adjoint_weighting: EB-covered
                // nodes (levset>=0) and the grounded outer walls are the
                // constrained rows; everything else is free.
                amrex::iMultiFab dmsk(ba, dm, 1, 1);
                dmsk.setVal(0);

                const amrex::Box ndom = amrex::surroundingNodes(wx.Geom(lev).Domain());
                const auto dlo = ndom.smallEnd();
                const auto dhi = ndom.bigEnd();

                for (amrex::MFIter mfi(dmsk); mfi.isValid(); ++mfi) {
                    const amrex::Box& bx = mfi.growntilebox();
                    auto const& dma = dmsk.array(mfi);
                    auto const& ls  = levset.const_array(mfi);
                    amrex::ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            const bool covered = (ls(i,j,k) >= amrex::Real(0.0));
                            const bool wall =
                                (i <= dlo[0] || i >= dhi[0] ||
                                 j <= dlo[1] || j >= dhi[1] ||
                                 k <= dlo[2] || k >= dhi[2]);
                            dma(i,j,k) = (covered || wall) ? 1 : 0;
                        });
                }

                WarpXDebugForwardApply(*out_mf, *in_mf, dmsk, lev);
            },
            py::arg("in_name"), py::arg("out_name"),
            "DEBUG/TEMPORARY: apply the transliterated forward EB Laplacian "
            "(warpx_mlebndfdlap_adotx_forward_eb, via AdjointWeightingSolve.cpp's "
            "file-local ApplyOp) to the registered field in_name, writing the "
            "result into out_name. Builds the same Dirichlet mask as "
            "solve_adjoint_weighting (levset>=0 EB-covered nodes plus the "
            "grounded outer walls). For diagnosing whether the transliteration "
            "matches AMReX's real Fapply near the embedded boundary."
        )
        .def("debug_adjoint_precond_apply",
            [] (WarpX& wx, const std::string& in_name, const std::string& out_name,
                int variant, amrex::Real prec_rtol) {
                int const lev = 0;
                auto const& eb_fact = wx.fieldEBFactory(lev);
                auto const& levset = eb_fact.getLevelSet();

                amrex::MultiFab* in_mf  = wx.m_fields.get(in_name, lev);
                amrex::MultiFab* out_mf = wx.m_fields.get(out_name, lev);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(in_mf != nullptr && out_mf != nullptr,
                    "debug_adjoint_precond_apply: in/out field not registered");
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(in_mf != out_mf,
                    "debug_adjoint_precond_apply: in_name and out_name must differ");

                const amrex::BoxArray& ba = out_mf->boxArray();
                const amrex::DistributionMapping& dm = out_mf->DistributionMap();

                // Same Dirichlet mask as solve_adjoint_weighting /
                // debug_adjoint_forward_apply.
                amrex::iMultiFab dmsk(ba, dm, 1, 1);
                dmsk.setVal(0);

                const amrex::Box ndom = amrex::surroundingNodes(wx.Geom(lev).Domain());
                const auto dlo = ndom.smallEnd();
                const auto dhi = ndom.bigEnd();

                for (amrex::MFIter mfi(dmsk); mfi.isValid(); ++mfi) {
                    const amrex::Box& bx = mfi.growntilebox();
                    auto const& dma = dmsk.array(mfi);
                    auto const& ls  = levset.const_array(mfi);
                    amrex::ParallelFor(bx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            const bool covered = (ls(i,j,k) >= amrex::Real(0.0));
                            const bool wall =
                                (i <= dlo[0] || i >= dhi[0] ||
                                 j <= dlo[1] || j >= dhi[1] ||
                                 k <= dlo[2] || k >= dhi[2]);
                            dma(i,j,k) = (covered || wall) ? 1 : 0;
                        });
                }

                WarpXDebugPrecondApply(*out_mf, *in_mf, dmsk, lev, variant, prec_rtol);
            },
            py::arg("in_name"), py::arg("out_name"),
            py::arg("variant") = 0, py::arg("prec_rtol") = 1.0e-4,
            "DEBUG/TEMPORARY: apply the MLMG-based preconditioner (an "
            "approximate A^{-1}) to in_name, writing the result into "
            "out_name. variant=0: plain MLMG::solve (ignores the internal "
            "scaleRHS S diagonal); variant=1: pre-divide by S (floored) "
            "before solving. For round-trip validation against "
            "debug_adjoint_forward_apply before wiring the preconditioner "
            "into the outer BiCGSTAB solve."
        )
        .def("debug_adjoint_scale",
            [] (WarpX& wx, const std::string& out_name) {
                int const lev = 0;
                amrex::MultiFab* out_mf = wx.m_fields.get(out_name, lev);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(out_mf != nullptr,
                    "debug_adjoint_scale: output field not registered");
                WarpXDebugComputeScale(*out_mf, lev);
            },
            py::arg("out_name"),
            "DEBUG/TEMPORARY: write the per-node MLMG rhs-scale factor "
            "S(node) = min(hp,hm) over the node's six edges "
            "(MLEBNodeFDLaplacian::scaleRHS / mlebndfdlap_scale_rhs) into the "
            "registered nodal field out_name."
        )
        .def("debug_adjoint_dump_levset",
            [] (WarpX& wx, const std::string& out_name) {
                int const lev = 0;
                amrex::MultiFab* out_mf = wx.m_fields.get(out_name, lev);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(out_mf != nullptr,
                    "debug_adjoint_dump_levset: output field not registered");
                WarpXDebugDumpLevelSet(*out_mf, lev);
            },
            py::arg("out_name"),
            "DEBUG/TEMPORARY: copy the exact EB levelset MultiFab "
            "(EBFArrayBoxFactory::getLevelSet(), the same array the forward "
            "kernel and MLMG's mask both read) into the registered nodal "
            "field out_name, so free (levset<0) / covered (levset>=0) nodes "
            "can be classified in Python without an analytic proxy."
        )
        .def("debug_adjoint_query_node",
            [] (WarpX& wx, int i, int j, int k) {
                int const lev = 0;
                auto const& eb_fact = wx.fieldEBFactory(lev);
                auto const& levset = eb_fact.getLevelSet();
                auto const& edge_cent = eb_fact.getEdgeCent();

                py::dict result;
                bool found = false;
                for (amrex::MFIter mfi(levset); mfi.isValid(); ++mfi) {
                    const amrex::Box& gbx = mfi.fabbox();
                    if (!gbx.contains(amrex::IntVect(i-1,j-1,k-1)) ||
                        !gbx.contains(amrex::IntVect(i+1,j+1,k+1))) { continue; }
                    auto const& ls = levset.const_array(mfi);
                    auto const& ecx = edge_cent[0]->const_array(mfi);
                    auto const& ecy = edge_cent[1]->const_array(mfi);
                    auto const& ecz = edge_cent[2]->const_array(mfi);

                    result["levset_c"]  = ls(i,j,k);
                    result["levset_xp"] = ls(i+1,j,k);
                    result["levset_xm"] = ls(i-1,j,k);
                    result["levset_yp"] = ls(i,j+1,k);
                    result["levset_ym"] = ls(i,j-1,k);
                    result["levset_zp"] = ls(i,j,k+1);
                    result["levset_zm"] = ls(i,j,k-1);
                    result["ecx_p"] = ecx(i,j,k);
                    result["ecx_m"] = ecx(i-1,j,k);
                    result["ecy_p"] = ecy(i,j,k);
                    result["ecy_m"] = ecy(i,j-1,k);
                    result["ecz_p"] = ecz(i,j,k);
                    result["ecz_m"] = ecz(i,j,k-1);
                    found = true;
                    break;
                }
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(found,
                    "debug_adjoint_query_node: (i,j,k) plus its neighbours are "
                    "not all inside one fab's valid+ghost region");
                return result;
            },
            py::arg("i"), py::arg("j"), py::arg("k"),
            "DEBUG/TEMPORARY: dump levset (self + 6 neighbours) and ecx/ecy/ecz "
            "(this node's 6 edges) at nodal index (i,j,k), for inspecting "
            "individual nodes flagged by the forward-apply vs scale*rhs "
            "residual comparison."
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
}
