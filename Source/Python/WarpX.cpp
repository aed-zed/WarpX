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
#include <Diagnostics/ReducedDiags/ChargeOnEB.H>
#include <FieldSolver/ElectrostaticSolvers/AdjointWeightingSolve.H>
#include <FieldSolver/ElectrostaticSolvers/StaircaseBias.H>
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
#include <memory>
#include <string>
#include <vector>


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
        .def("solve_poisson_efield",
            [] (WarpX& wx, bool const eb_aware_gradient) {
                wx.SolvePoissonEfield(eb_aware_gradient);
            },
            py::arg("eb_aware_gradient") = true,
            "Deposit charge from all species, solve Poisson with the current EB and "
            "domain boundary conditions, replace Efield_fp with the result, and "
            "publish phi_fp when it is registered.\n\n"
            "With an embedded boundary, eb_aware_gradient=True (the default) takes E "
            "from the solve itself, which uses the shortened fluid length on cut "
            "edges. Pass False to extract E with the ordinary full-grid gradient "
            "instead: that is the representation the Yee Faraday update differences, "
            "so the resulting field has no discrete curl, at the cost of a less "
            "accurate local field next to the boundary. Without an embedded boundary "
            "the flag has no effect."
        )
        .def("compute_eb_charge",
            [] (WarpX& wx, const std::string& weighting, const std::string& field) {
                int const lev = 0;
                ablastr::fields::VectorField const E = wx.m_fields.get_alldirs(field, lev);
                if (weighting.empty() || weighting == "1") {
                    return WeightedChargeOnEB(E, lev, nullptr);
                }
                amrex::Parser parser = utils::parser::makeParser(weighting, {"x", "y", "z"});
                return WeightedChargeOnEB(E, lev, &parser);
            },
            py::arg("weighting") = "1",
            py::arg("field") = "Efield_fp",
            "Induced charge eps0 * oint w(x,y,z) E.n dS over the embedded boundary for "
            "the named field, with an optional weighting w(x,y,z) selecting one "
            "electrode. 3D and RZ with EB only."
        )
        .def("solve_staircase_unit_bias",
            [] (WarpX&, std::string const& selector, std::string const& out_phi,
                std::string const& out_weight, std::string const& out_efield,
                amrex::Real rtol, int max_iter, bool insulating_endcaps) {
                return WarpXSolveStaircaseUnitBias(
                    selector, out_phi, out_weight, out_efield, rtol, max_iter,
                    insulating_endcaps);
            },
            py::arg("selector"), py::arg("out_phi"), py::arg("out_weight"),
            py::arg("out_efield"), py::arg("rtol") = 1.e-12,
            py::arg("max_iter") = 200,
            py::arg("insulating_endcaps") = false,
            "Research RZ Yee staircase unit bias. Fill already registered nodal "
            "potential/fixed-node weight and staggered E fields using native frozen "
            "edges and AMReX's regular FD Laplacian. Does not change live E or EB "
            "potential parsers. Selector must be binary and constant on every "
            "connected frozen-edge component. Returns the absolute solve residual. "
            "insulating_endcaps enables a separate, z-independent whole-face "
            "pec_insulator research mode; it does not model dielectric charging."
        )
        .def("solve_staircase_insulator_bias",
            [] (WarpX&, std::string const& selector, std::string const& out_psi,
                std::string const& out_weight, std::string const& out_efield,
                amrex::Real rtol, int max_iter) {
                return WarpXSolveStaircaseInsulatorBias(
                    selector, out_psi, out_weight, out_efield, rtol, max_iter);
            },
            py::arg("selector"), py::arg("out_psi"), py::arg("out_weight"),
            py::arg("out_efield"), py::arg("rtol") = 1.e-12,
            py::arg("max_iter") = 200,
            "Research RZ Yee insulating-endcap staircase bias. out_psi is the "
            "homogeneous-Neumann charge-observer potential, not the actuator "
            "potential. out_efield comes from a second harmonic solve with exact "
            "one-dimensional RZ endpoint traces between fixed metal nodes. Returns "
            "the larger absolute residual of the two solves. Does not change live E."
        )
        .def("staircase_charge_state",
            [] (WarpX&, std::vector<std::string> const& psi_fields,
                std::vector<std::string> const& weight_fields, bool insulating_endcaps) {
                auto const q = WarpXStaircaseChargeState(
                    psi_fields, weight_fields, insulating_endcaps);
                std::vector<std::vector<amrex::Real>> result;
                for (auto const& row : q) {
                    result.emplace_back(row.begin(), row.end());
                }
                return result;
            },
            py::arg("psi_fields"), py::arg("weight_fields"),
            py::arg("insulating_endcaps") = false,
            "Research staircase charge observer: returns (total Gauss charge, live "
            "charge on fixed nodes, grounded all-node pairing), in coulombs. "
            "Collective, with unique nodal ownership and separate Gauss/deposition "
            "axis measures. No Poisson solve. With insulating_endcaps, the third "
            "row includes the outward weighted axial electric flux, also returned "
            "separately as a fourth row; field endpoint volumes are halved. "
            "This observer does not repair native endpoint particle-current handling."
        )
        .def("validate_staircase_weights",
            [] (WarpX&, std::vector<std::string> const& weight_fields) {
                return WarpXValidateStaircaseWeights(weight_fields);
            },
            py::arg("weight_fields"),
            "Research setup check: global maximum error from assigning exactly "
            "one electrode weight to each native frozen-edge endpoint node. "
            "Zero means complete, non-overlapping coverage."
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
            "Freshly deposited nodal charge density of all live species, as a new "
            "MultiFab. Does not touch rho_fp, does not solve and does not move "
            "particles."
        )
        .def("compute_div_e",
            [] (WarpX& wx, int const lev) {
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
                wx.ComputeDivE(div_e, lev);
                return div_e;
            },
            py::arg("lev") = 0,
            py::return_value_policy::move,
            "Native discrete divergence of Efield_fp on the nodes, as a new "
            "MultiFab. Uses WarpX::ComputeDivE, so the cylindrical form is "
            "applied in RZ. The caller supplies the control-volume measure; in "
            "RZ that is the cylindrical nodal volume, not dr*dz."
        )
        .def("refresh_staircase_efield_guards",
            [] (WarpX& wx) {
                // A staircase correction is added to the valid E region after
                // the field push.  Refresh physical guards first, then exchange
                // inter-box guards, in the same order as an explicit field push.
                wx.ApplyEfieldBoundary(0, PatchType::fine, wx.gett_new(0));
                wx.FillBoundaryE(0, wx.getngEB(), true);
            },
            "Refresh level-zero Efield_fp physical and inter-box guard cells after "
            "a direct staircase-bias field update."
        )
#if defined(WARPX_DIM_RZ)
        .def("rz_axis_volume_factor",
            [] (WarpX const & wx) { return wx.RZAxisVolumeFactor(); },
            "The axis-node radial volume factor charge deposition uses: 1/3 with "
            "the Verboncoeur correction, 1/4 without. Exposed so a diagnostic "
            "integrating rho or divE over nodal control volumes can use the same "
            "axis measure as the deposition instead of guessing it."
        )
#endif
        .def("grounded_charge_from_adjoint",
            [] (WarpX& /*wx*/, const std::vector<std::string>& psi_fields, int lev) {
                auto const q = WarpXGroundedChargeFromAdjoint(psi_fields, lev);
                return std::vector<amrex::Real>(q.begin(), q.end());
            },
            py::arg("psi_fields"), py::arg("lev") = 0,
            "Plasma-induced charge on each electrode with all electrodes grounded, "
            "evaluated as -sum rho*Psi*dV against a freshly deposited charge density. "
            "No Poisson solve. Shared nodes on box boundaries are counted once and the "
            "RZ measure is WarpX's cylindrical nodal volume."
        )
        .def("div_e_charge_in_regions",
            [] (WarpX& /*wx*/, const std::vector<std::string>& regions, int lev) {
                auto const q = WarpXDivEChargeInRegions(regions, lev);
                return std::vector<amrex::Real>(q.begin(), q.end());
            },
            py::arg("regions"), py::arg("lev") = 0,
            "Charge enclosed by each region, eps0 * sum w(x,y,z) divE dV on the nodes, "
            "using WarpX's own ComputeDivE on Efield_fp. The volume counterpart of "
            "compute_eb_charge, and the preferred electrode-charge observer: a region "
            "bounded in live fluid does not read the cut cells whose E the staircase "
            "FDTD update freezes. Needs about two cells of vacuum clearance between the "
            "metal and the region boundary. Shared nodes on box boundaries are counted "
            "once, so the result is independent of the domain decomposition and the "
            "number of ranks. This is the TOTAL enclosed charge; subtract "
            "live_charge_in_regions over the same weights to isolate the electrode's."
        )
        .def("live_charge_in_regions",
            [] (WarpX& /*wx*/, const std::vector<std::string>& regions, int lev) {
                auto const q = WarpXLiveChargeInRegions(regions, lev);
                return std::vector<amrex::Real>(q.begin(), q.end());
            },
            py::arg("regions"), py::arg("lev") = 0,
            "Deposited live charge inside each region, sum w(x,y,z) rho dV, against a "
            "freshly deposited scratch density on the same nodes and measure as "
            "div_e_charge_in_regions."
        )
        .def("eb_update_e_flag",
            [] (WarpX& wx, int const lev, int const dir) {
                return wx.GetEBUpdateEFlag()[lev][dir].get();
            },
            py::arg("lev") = 0, py::arg("dir") = 0,
            py::return_value_policy::reference_internal,
            "The staircase update mask for one E component: 1 where the FDTD update "
            "advances the field, 0 where it is frozen because a cell in the component's "
            "stencil is cut or covered. Read-only view of WarpX's own mask, so a "
            "corrector can keep its correction out of the frozen cells without "
            "reconstructing the topology."
        )
        .def("solve_adjoint_weighting",
            [] (WarpX& wx, const std::string& region, const std::string& out_name,
                amrex::Real tol, int max_iter, const std::string& functional) {
                int const lev = 0;
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    functional == "surface" || functional == "volume",
                    "solve_adjoint_weighting: functional must be 'surface' or 'volume'");
                auto const& levset = wx.fieldEBFactory(lev).getLevelSet();

                amrex::MultiFab* psi = wx.m_fields.get(out_name, lev);
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(psi != nullptr,
                    "solve_adjoint_weighting: output field not registered");

                const amrex::BoxArray& ba = psi->boxArray();
                const amrex::DistributionMapping& dm = psi->DistributionMap();

                // Constrained rows are the nodes covered by the EB and the nodes on
                // the grounded outer walls. Leaving the walls free would pose the
                // operator on a space that includes unconstrained boundary values.
                amrex::iMultiFab dmsk(ba, dm, 1, 1);
                dmsk.setVal(0);

                const amrex::Box ndom = amrex::surroundingNodes(wx.Geom(lev).Domain());
                const auto dlo = ndom.smallEnd();
                const auto dhi = ndom.bigEnd();

                // A periodic direction has no wall: its domain-edge nodes are
                // ordinary unknowns wrapped onto their images, not constrained
                // rows. Marking them would pin the solution at the seam and
                // silently pose a different operator, so drop those faces from
                // the wall test rather than the whole node.
                amrex::GpuArray<int, AMREX_SPACEDIM> is_periodic{};
                for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                    is_periodic[idim] = wx.Geom(lev).isPeriodic(idim);
                }

                for (amrex::MFIter mfi(dmsk); mfi.isValid(); ++mfi) {
                    auto const& dma = dmsk.array(mfi);
                    auto const& ls  = levset.const_array(mfi);
                    amrex::ParallelFor(mfi.growntilebox(),
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            const bool covered = (ls(i,j,k) >= amrex::Real(0.0));
#ifdef WARPX_DIM_RZ
                            // r=0 is a regularity axis, not a grounded wall
                            const bool wall =
                                (!is_periodic[0] && i >= dhi[0]) ||
                                (!is_periodic[1] && (j <= dlo[1] || j >= dhi[1]));
#else
                            const bool wall =
                                (!is_periodic[0] && (i <= dlo[0] || i >= dhi[0])) ||
                                (!is_periodic[1] && (j <= dlo[1] || j >= dhi[1])) ||
                                (!is_periodic[2] && (k <= dlo[2] || k >= dhi[2]));
#endif
                            dma(i,j,k) = (covered || wall) ? 1 : 0;
                        });
                }

                amrex::MultiFab rhs(ba, dm, 1, 1);
                if (functional == "volume") {
                    WarpXBuildAdjointRHSVolumeFunctional(rhs, region, dmsk, lev);
                } else {
                    WarpXBuildAdjointRHSChargeFunctional(rhs, region, dmsk, lev);
                }

                amrex::Real res = -1.0;
                int iters = -1;
                const bool ok = WarpXSolveAdjointWeighting(
                    *psi, rhs, dmsk, lev, tol, max_iter, &res, &iters);
                amrex::Print() << "solve_adjoint_weighting: " << iters
                               << " outer iterations, converged="
                               << (ok ? "true" : "false")
                               << ", rel.residual=" << res << "\n";

                WarpXFinalizeChargeFunctionalPsi(*psi, lev);
                psi->FillBoundary(wx.Geom(lev).periodicity());
                return py::make_tuple(ok, res);
            },
            py::arg("region"), py::arg("out_name"),
            py::arg("tol") = 1.0e-10, py::arg("max_iter") = 200,
            py::arg("functional") = "surface",
            "Solve for the adjoint weighting potential Psi_k of the electrode selected "
            "by region(x,y,z) and write it into the registered nodal field out_name. "
            "Unlike the plain unit-voltage basis, this Psi makes the grounded-charge "
            "identity Q_k = -sum_a rho_a Psi_k[a] exact on WarpX's non-symmetric EB "
            "Laplacian. Requires grounded (PEC) outer boundaries. "
            "functional selects WHICH charge the identity is exact for: 'surface' for "
            "compute_eb_charge, in which case region selects the EB surface patch, and "
            "'volume' for div_e_charge_in_regions, in which case region is the "
            "enclosing volume. Psi is the adjoint of one specific measurement, so "
            "pairing it with the other observer leaves a residual of a few percent of "
            "the bias. "
            "Returns (converged, relative_residual); the caller must check converged, "
            "because an unconverged Psi yields a plausible but wrong correction. "
            "3D and RZ with EB only."
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
