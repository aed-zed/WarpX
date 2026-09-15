/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "StaircaseBias.H"

#include "ElectrostaticSolver.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "NodalChargeMeasurement.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_EB2.H>
#include <AMReX_EB2_IF_AllRegular.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_MLEBNodeFDLaplacian.H>
#include <AMReX_MLMG.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_iMultiFab.H>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace
{
#ifdef WARPX_DIM_RZ
    bool IsReservedFieldName (std::string const& name)
    {
        auto const names = amrex::getEnumNameStrings<warpx::fields::FieldType>();
        return std::find(names.begin(), names.end(), name) != names.end();
    }

    void ValidateStaircaseMode (WarpX const& warpx)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(EB::enabled(),
            "The staircase unit-bias path requires an embedded boundary");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.maxLevel() == 0,
            "The staircase unit-bias path currently supports one AMR level");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::n_rz_azimuthal_modes == 1,
            "The staircase unit-bias path currently supports only RZ mode m=0");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            warpx.Geom(0).ProbLo(0) == amrex::Real(0.0),
            "The staircase unit-bias path requires a physical RZ axis at r=0");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee &&
            WarpX::grid_type == GridType::Staggered,
            "The staircase unit-bias path requires the staggered Yee solver");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.evolve_scheme == EvolveScheme::Explicit,
            "The staircase unit-bias path requires the explicit evolve scheme");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::None,
            "The staircase unit-bias path cannot run with a global electrostatic solver");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            warpx.GetEMSolverMedium() == MediumForEM::Vacuum,
            "The staircase unit-bias path requires the vacuum electromagnetic medium");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::do_moving_window == 0,
            "The staircase unit-bias path requires a fixed simulation window");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!warpx.DoFluidSpecies(),
            "The staircase unit-bias path currently supports particle species only");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!warpx.HasMirrors(),
            "The staircase unit-bias path does not support internal field mirrors");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::gamma_boost == amrex::Real(1.0),
            "The staircase unit-bias path currently supports the laboratory frame only");
    }

    void ValidateInsulatingEndcaps ()
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC_Insulator &&
            WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC_Insulator,
            "insulating_endcaps requires two native pec_insulator z boundaries");
        amrex::ParmParse const pp("insulator");
        for (auto const* side : {"lo", "hi"}) {
            std::string area;
            utils::parser::Query_parserString(pp, std::string("area_z_")+side+"(x,y)", area);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(area == "1",
                "The research insulating-endcap clamp requires area_z_lo(x,y) = 1 "
                "and area_z_hi(x,y) = 1; metal footprints are supplied by the EB");
            for (auto const* component : {"Ex", "Ey", "Bx", "By"}) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                    !pp.contains(std::string(component)+"_z_"+side+"(x,y,t)"),
                    "The insulating-endcap clamp cannot prescribe tangential boundary fields");
            }
        }
    }
#endif
}

namespace
{
amrex::Real SolveStaircaseBias (
    std::string const& selector, std::string const& out_phi,
    std::string const& out_weight, std::string const& out_efield,
    amrex::Real const rtol, int const max_iter, bool const insulating_endcaps,
    bool const harmonic_trace)
{
#ifndef WARPX_DIM_RZ
    amrex::ignore_unused(selector, out_phi, out_weight, out_efield, rtol, max_iter,
                        insulating_endcaps, harmonic_trace);
    WARPX_ABORT_WITH_MESSAGE("The staircase unit-bias solve is implemented only in RZ");
    return amrex::Real(0.0);
#else
    using ablastr::fields::MultiLevelScalarField;
    using ablastr::fields::MultiLevelVectorField;
    using ablastr::fields::VectorField;
    using namespace amrex::literals;

    auto& warpx = WarpX::GetInstance();
    ValidateStaircaseMode(warpx);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!harmonic_trace || insulating_endcaps,
        "A harmonic endpoint trace requires native insulating endcaps");
    if (insulating_endcaps) { ValidateInsulatingEndcaps(); }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rtol > 0._rt && rtol < 1._rt && max_iter > 0,
        "The staircase unit-bias solve needs 0 < rtol < 1 and max_iter > 0");

    const bool z_periodic =
        WarpX::field_boundary_lo[1] == FieldBoundaryType::Periodic &&
        WarpX::field_boundary_hi[1] == FieldBoundaryType::Periodic;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        WarpX::field_boundary_lo[0] == FieldBoundaryType::None &&
        WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
        (z_periodic || insulating_endcaps ||
         (WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC &&
          WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC)),
        "Supported staircase boundaries are the RZ axis, a PEC outer radius, and "
        "either two PEC, two periodic, or explicitly enabled insulating z boundaries");

    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        out_phi != out_weight && out_phi != out_efield && out_weight != out_efield,
        "Staircase output field names must be distinct");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !IsReservedFieldName(out_phi) && !IsReservedFieldName(out_weight) &&
        !IsReservedFieldName(out_efield),
        "Staircase outputs must not use reserved live-field names");

    constexpr int lev = 0;
    auto& fields = warpx.m_fields;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        fields.has(out_phi, lev) && fields.has(out_weight, lev) &&
        fields.has_vector(out_efield, lev) && fields.has_vector("Efield_fp", lev),
        "Staircase output and native E fields must be registered before the solve");
    amrex::MultiFab* phi = fields.get(out_phi, lev);
    amrex::MultiFab* weight = fields.get(out_weight, lev);
    VectorField efield = fields.get_alldirs(out_efield, lev);
    VectorField const live_e = fields.get_alldirs("Efield_fp", lev);
    amrex::BoxArray nodal_ba = warpx.boxArray(lev);
    nodal_ba.surroundingNodes();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        phi != weight && phi->boxArray() == nodal_ba && weight->boxArray() == nodal_ba &&
        phi->DistributionMap() == warpx.DistributionMap(lev) &&
        weight->DistributionMap() == warpx.DistributionMap(lev) &&
        phi->nComp() == 1 && weight->nComp() == 1 && phi->nGrowVect().allGE(1),
        "Staircase scalar outputs must be distinct one-component nodal fields; phi "
        "needs at least one guard cell");
    for (int component = 0; component < 3; ++component) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            efield[component] != nullptr && efield[component] != live_e[component] &&
            efield[component] != phi && efield[component] != weight &&
            efield[component]->boxArray() == live_e[component]->boxArray() &&
            efield[component]->DistributionMap() == live_e[component]->DistributionMap() &&
            efield[component]->nGrowVect() == live_e[component]->nGrowVect() &&
            efield[component]->nComp() == 1,
            "Staircase E output must be a distinct one-component vector with native layouts");
        for (int other = 0; other < 3; ++other) {
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                efield[component] != efield[other] || component == other,
                "Staircase E output components must not alias one another");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(efield[component] != live_e[other],
                "Staircase E output must not alias a live E component");
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
                phi != live_e[other] && weight != live_e[other],
                "Staircase scalar outputs must not alias a live E component");
        }
    }

    auto& updates = warpx.GetEBUpdateEFlag();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(updates.size() > lev,
        "Native staircase E update masks are not allocated at level zero");
    auto& update = updates[lev];
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(update[0] && update[2],
        "Native staircase Er/Ez update masks are not allocated");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        update[0]->nGrowVect().allGE(1) && update[2]->nGrowVect().allGE(1),
        "Native staircase Er/Ez update masks need one guard cell");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        update[0]->boxArray() == live_e[0]->boxArray() &&
        update[2]->boxArray() == live_e[2]->boxArray() &&
        update[0]->DistributionMap() == live_e[0]->DistributionMap() &&
        update[2]->DistributionMap() == live_e[2]->DistributionMap() &&
        update[0]->nComp() >= 1 && update[2]->nComp() >= 1,
        "Native staircase masks must have the native Er/Ez field layouts");
    update[0]->FillBoundary(warpx.Geom(lev).periodicity());
    update[2]->FillBoundary(warpx.Geom(lev).periodicity());

    amrex::iMultiFab overset(nodal_ba, warpx.DistributionMap(lev), 1, 0);
    amrex::MultiFab checks(nodal_ba, warpx.DistributionMap(lev), 5, 0);
    phi->setVal(0._rt);
    weight->setVal(0._rt);
    checks.setVal(0._rt);

    amrex::Parser parser = utils::parser::makeParser(selector, {"x", "y", "z"});
    auto const select = parser.compile<3>();
    auto const dx = warpx.Geom(lev).CellSizeArray();
    auto const lo = warpx.Geom(lev).ProbLoArray();
    amrex::Box const ndomain = amrex::surroundingNodes(warpx.Geom(lev).Domain());
    auto const dlo = ndomain.smallEnd();
    auto const dhi = ndomain.bigEnd();

    for (amrex::MFIter mfi(*phi); mfi.isValid(); ++mfi) {
        auto const& pa = phi->array(mfi);
        auto const& wa = weight->array(mfi);
        auto const& ma = overset.array(mfi);
        auto const& ca = checks.array(mfi);
        auto const& ur = update[0]->const_array(mfi);
        auto const& uz = update[2]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const bool incident =
                    (i > dlo[0] && ur(i-1,j,k) == 0) ||
                    (i < dhi[0] && ur(i,j,k) == 0) ||
                    ((z_periodic || j > dlo[1]) && uz(i,j-1,k) == 0) ||
                    ((z_periodic || j < dhi[1]) && uz(i,j,k) == 0);
                const amrex::Real r = lo[0] + amrex::Real(i)*dx[0];
                const amrex::Real z = lo[1] + amrex::Real(j)*dx[1];
                const amrex::Real s = select(r, 0._rt, z);
                const bool binary = s == 0._rt || s == 1._rt;
                const bool wall = i == dhi[0] ||
                    (!z_periodic && !insulating_endcaps && (j == dlo[1] || j == dhi[1]));
                pa(i,j,k) = incident && s == 1._rt ? 1._rt : 0._rt;
                wa(i,j,k) = pa(i,j,k);
                ma(i,j,k) = incident || wall ? 0 : 1; // overset: 1 unknown, 0 fixed
                ca(i,j,k,0) = incident && !binary;
                ca(i,j,k,1) = incident && wall;
                ca(i,j,k,2) =
                    i < dhi[0] && ur(i,j,k) == 0 &&
                    s != select(r+dx[0], 0._rt, z);
                ca(i,j,k,3) =
                    j < dhi[1] && uz(i,j,k) == 0 &&
                    s != select(r, 0._rt, z+dx[1]);
                ca(i,j,k,4) = z_periodic && incident && j == dlo[1] &&
                    s != select(r, 0._rt,
                        lo[1]+amrex::Real(dhi[1]-dlo[1])*dx[1]);
            });
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(checks.max(0) == 0._rt,
        "The staircase selector must be binary on every frozen-edge endpoint");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(checks.max(1) == 0._rt,
        "A frozen staircase component touches a nonperiodic PEC wall");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(checks.max(2) == 0._rt && checks.max(3) == 0._rt,
        "The staircase selector changes value across a frozen Er or Ez edge");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(checks.max(4) == 0._rt,
        "The staircase selector differs on duplicate periodic z nodes");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        weight->sum_unique(0, false, warpx.Geom(lev).periodicity()) > 0._rt,
        "The staircase selector does not select a frozen component");
    phi->OverrideSync(warpx.Geom(lev).periodicity());
    phi->FillBoundary(warpx.Geom(lev).periodicity());
    weight->OverrideSync(warpx.Geom(lev).periodicity());
    weight->FillBoundary(warpx.Geom(lev).periodicity());
    overset.OverrideSync(warpx.Geom(lev).periodicity());

    amrex::MultiFab rhs(nodal_ba, warpx.DistributionMap(lev), 1, 0);
    rhs.setVal(0._rt);
    amrex::LPInfo info;
    amrex::EB2::AllRegularIF all_regular;
    using RegularShop = amrex::EB2::GeometryShop<amrex::EB2::AllRegularIF>;
    RegularShop regular_shop(all_regular);
    auto regular_index = std::make_unique<amrex::EB2::IndexSpaceImp<RegularShop>>(
        regular_shop, warpx.Geom(lev), 0, 30, 1, true, false, 0);
    auto regular_factory = amrex::makeEBFabFactory(
        regular_index.get(), warpx.Geom(lev), warpx.boxArray(lev),
        warpx.DistributionMap(lev), {1, 1, 1}, amrex::EBSupport::full);
    auto solve_potential = [&] (amrex::MultiFab* potential, amrex::iMultiFab const& mask)
    {
        // Construct a fresh operator because nodal Dirichlet masks are cached by MLMG.
        amrex::MLEBNodeFDLaplacian linop(
            {warpx.Geom(lev)}, {warpx.boxArray(lev)}, {warpx.DistributionMap(lev)}, info,
            {regular_factory.get()});
        linop.setRZ(true);
        linop.setSigma({1._rt, 1._rt});
        linop.setAlpha(0._rt);
        if (insulating_endcaps) {
            // The observer uses homogeneous Neumann endpoint faces. The trace actuator
            // preloads every endpoint node and removes those equations with the mask.
            linop.setDomainBC({amrex::LinOpBCType::Neumann, amrex::LinOpBCType::Neumann},
                              {amrex::LinOpBCType::Dirichlet, amrex::LinOpBCType::Neumann});
        } else {
            auto& handler = *warpx.GetElectrostaticSolver().m_poisson_boundary_handler;
            handler.DefinePhiBCs(warpx.Geom(lev));
            linop.setDomainBC(handler.lobc, handler.hibc);
        }
        linop.setOversetMask(lev, mask);
        amrex::MLMG mlmg(linop);
        mlmg.setMaxIter(max_iter);
        return mlmg.solve({potential}, {&rhs}, rtol, 0._rt);
    };

    amrex::Real residual = solve_potential(phi, overset);
    phi->FillBoundary(warpx.Geom(lev).periodicity());

    std::unique_ptr<amrex::MultiFab> trace_phi;
    amrex::MultiFab* actuator_phi = phi;
    if (harmonic_trace) {
        int const nr_nodes = dhi[0] - dlo[0] + 1;
        int const face_size = 2 * nr_nodes;
        amrex::Gpu::DeviceVector<int> device_flags(2 * face_size, 0);
        int* const flags = device_flags.data();
        for (amrex::MFIter mfi(*phi); mfi.isValid(); ++mfi) {
            auto const ma = overset.const_array(mfi);
            auto const wa = weight->const_array(mfi);
            amrex::For(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (j == dlo[1] || j == dhi[1]) {
                        int const side = (j == dhi[1]);
                        int const index = side * nr_nodes + i - dlo[0];
                        amrex::Gpu::Atomic::Max(flags + index, int(ma(i,j,k) == 0));
                        amrex::Gpu::Atomic::Max(flags + face_size + index,
                                               int(wa(i,j,k) != 0._rt));
                    }
                });
        }
        std::vector<int> host_flags(device_flags.size());
        amrex::Gpu::copy(amrex::Gpu::deviceToHost, device_flags.begin(),
                         device_flags.end(), host_flags.begin());
        amrex::ParallelDescriptor::ReduceIntMax(
            host_flags.data(), static_cast<int>(host_flags.size()));

        std::vector<amrex::Real> host_trace(face_size, 0._rt);
        for (int side = 0; side < 2; ++side) {
            int const base = side * nr_nodes;
            auto fixed = [&] (int i) { return host_flags[base+i] != 0; };
            auto selected = [&] (int i) { return host_flags[face_size+base+i] != 0; };
            for (int i = 0; i < nr_nodes; ++i) {
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!selected(i) || fixed(i),
                    "An insulating-face electrode weight is not on a fixed staircase node");
            }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fixed(nr_nodes-1) && !selected(nr_nodes-1),
                "The PEC outer radius must ground both insulating-face traces");

            int left = 0;
            while (left < nr_nodes && !fixed(left)) { ++left; }
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(left < nr_nodes,
                "Each insulating-face trace needs at least one fixed-potential anchor");
            amrex::Real const first_value = selected(left) ? 1._rt : 0._rt;
            for (int i = 0; i <= left; ++i) { host_trace[base+i] = first_value; }
            while (left < nr_nodes-1) {
                int right = left + 1;
                while (right < nr_nodes && !fixed(right)) { ++right; }
                WARPX_ALWAYS_ASSERT_WITH_MESSAGE(right < nr_nodes,
                    "An insulating-face interval is not bounded by fixed-potential nodes");
                amrex::Real resistance = 0._rt;
                for (int edge = left; edge < right; ++edge) {
                    resistance += 1._rt / (amrex::Real(edge) + 0.5_rt);
                }
                amrex::Real accumulated = 0._rt;
                amrex::Real const left_value = selected(left) ? 1._rt : 0._rt;
                amrex::Real const right_value = selected(right) ? 1._rt : 0._rt;
                for (int i = left + 1; i < right; ++i) {
                    accumulated += 1._rt / (amrex::Real(i-1) + 0.5_rt);
                    host_trace[base+i] = left_value
                        + (right_value-left_value) * accumulated/resistance;
                }
                host_trace[base+right] = right_value;
                left = right;
            }
        }

        amrex::Gpu::DeviceVector<amrex::Real> device_trace(face_size);
        amrex::Gpu::copy(amrex::Gpu::hostToDevice, host_trace.begin(), host_trace.end(),
                         device_trace.begin());
        amrex::Real const* const trace = device_trace.data();
        amrex::iMultiFab trace_overset(nodal_ba, warpx.DistributionMap(lev), 1, 0);
        amrex::iMultiFab::Copy(trace_overset, overset, 0, 0, 1, 0);
        trace_phi = std::make_unique<amrex::MultiFab>(
            phi->boxArray(), phi->DistributionMap(), 1, phi->nGrowVect());
        amrex::MultiFab::Copy(*trace_phi, *phi, 0, 0, 1, phi->nGrowVect());
        for (amrex::MFIter mfi(*trace_phi); mfi.isValid(); ++mfi) {
            auto const pa = trace_phi->array(mfi);
            auto const ma = trace_overset.array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (j == dlo[1] || j == dhi[1]) {
                        int const side = (j == dhi[1]);
                        pa(i,j,k) = trace[side*nr_nodes+i-dlo[0]];
                        ma(i,j,k) = 0;
                    }
                });
        }
        trace_phi->OverrideSync(warpx.Geom(lev).periodicity());
        trace_phi->FillBoundary(warpx.Geom(lev).periodicity());
        trace_overset.OverrideSync(warpx.Geom(lev).periodicity());
        residual = std::max(amrex::Math::abs(residual),
                            amrex::Math::abs(solve_potential(trace_phi.get(), trace_overset)));
        trace_phi->FillBoundary(warpx.Geom(lev).periodicity());
        actuator_phi = trace_phi.get();
    }

    for (auto* field : efield) { field->setVal(0._rt); }
    MultiLevelVectorField efield_levels{efield};
    MultiLevelScalarField phi_levels{actuator_phi};
    warpx.GetElectrostaticSolver().computeE(
        efield_levels, phi_levels, {0._rt, 0._rt, 0._rt});
    if (insulating_endcaps && !harmonic_trace) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            efield[2]->norm0() <= std::max(100._rt*rtol, 1.e-12_rt)*efield[0]->norm0(),
            "The insulating-endcap clamp currently requires z-independent unit potentials; "
            "axially varying/fringing geometries need a matched boundary construction");
    }
    for (auto* field : efield) {
        field->FillBoundary(warpx.Geom(lev).periodicity());
    }
    if (insulating_endcaps && !harmonic_trace) {
        // Also check the transverse field itself, including across MPI boxes.
        // Do not post-mask or flatten a field that fails this compatibility gate.
        for (amrex::MFIter mfi(checks); mfi.isValid(); ++mfi) {
            auto const error = checks.array(mfi);
            auto const er = efield[0]->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    error(i,j,k,0) = i < dhi[0] && j < dhi[1]
                        ? amrex::Math::abs(er(i,j+1,k)-er(i,j,k)) : 0._rt;
                });
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            checks.norm0(0) <= std::max(100._rt*rtol, 1.e-12_rt)*efield[0]->norm0() &&
            efield[1]->norm0() == 0._rt,
            "Insulating-endcap unit fields must have z-independent Er and zero Etheta");
    }
    return residual;
#endif
}
}

amrex::Real WarpXSolveStaircaseUnitBias (
    std::string const& selector, std::string const& out_phi,
    std::string const& out_weight, std::string const& out_efield,
    amrex::Real const rtol, int const max_iter, bool const insulating_endcaps)
{
    return SolveStaircaseBias(selector, out_phi, out_weight, out_efield, rtol, max_iter,
                              insulating_endcaps, false);
}

amrex::Real WarpXSolveStaircaseInsulatorBias (
    std::string const& selector, std::string const& out_psi,
    std::string const& out_weight, std::string const& out_efield,
    amrex::Real const rtol, int const max_iter)
{
    return SolveStaircaseBias(selector, out_psi, out_weight, out_efield, rtol, max_iter,
                              true, true);
}

amrex::Real WarpXValidateStaircaseWeights (
    std::vector<std::string> const& weight_fields)
{
#ifndef WARPX_DIM_RZ
    amrex::ignore_unused(weight_fields);
    WARPX_ABORT_WITH_MESSAGE("Staircase weights are implemented only in RZ");
    return amrex::Real(0.0);
#else
    using namespace amrex::literals;

    auto& warpx = WarpX::GetInstance();
    ValidateStaircaseMode(warpx);
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!weight_fields.empty(),
        "At least one staircase weight field is required");
    constexpr int lev = 0;
    auto& fields = warpx.m_fields;
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fields.has_vector("Efield_fp", lev),
        "The native E field must be registered before validating staircase weights");
    auto const live_e = fields.get_alldirs("Efield_fp", lev);
    amrex::BoxArray nodal_ba = warpx.boxArray(lev);
    nodal_ba.surroundingNodes();
    amrex::MultiFab sum(nodal_ba, warpx.DistributionMap(lev), 1, 0);
    sum.setVal(0._rt);
    for (auto const& name : weight_fields) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(fields.has(name, lev),
            "A staircase weight field is not registered");
        auto const* weight = fields.get(name, lev);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            weight->boxArray() == nodal_ba &&
            weight->DistributionMap() == warpx.DistributionMap(lev) &&
            weight->nComp() == 1,
            "Staircase weights must be one-component native nodal fields");
        amrex::MultiFab::Add(sum, *weight, 0, 0, 1, 0);
    }

    auto& updates = warpx.GetEBUpdateEFlag();
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(updates.size() > lev,
        "Native staircase E update masks are not allocated at level zero");
    auto& update = updates[lev];
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        update[0] && update[2] && update[0]->nGrowVect().allGE(1) &&
        update[2]->nGrowVect().allGE(1),
        "Native staircase Er/Ez update masks need one guard cell");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        update[0]->boxArray() == live_e[0]->boxArray() &&
        update[2]->boxArray() == live_e[2]->boxArray() &&
        update[0]->DistributionMap() == live_e[0]->DistributionMap() &&
        update[2]->DistributionMap() == live_e[2]->DistributionMap() &&
        update[0]->nComp() >= 1 && update[2]->nComp() >= 1,
        "Native staircase masks must have the native Er/Ez field layouts");
    update[0]->FillBoundary(warpx.Geom(lev).periodicity());
    update[2]->FillBoundary(warpx.Geom(lev).periodicity());
    amrex::Box const ndomain = amrex::surroundingNodes(warpx.Geom(lev).Domain());
    auto const dlo = ndomain.smallEnd();
    auto const dhi = ndomain.bigEnd();
    bool const z_periodic = warpx.Geom(lev).isPeriodic(1);
    for (amrex::MFIter mfi(sum); mfi.isValid(); ++mfi) {
        auto const s = sum.array(mfi);
        auto const ur = update[0]->const_array(mfi);
        auto const uz = update[2]->const_array(mfi);
        amrex::ParallelFor(mfi.validbox(),
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                const bool fixed =
                    (i > dlo[0] && ur(i-1,j,k) == 0) ||
                    (i < dhi[0] && ur(i,j,k) == 0) ||
                    ((z_periodic || j > dlo[1]) && uz(i,j-1,k) == 0) ||
                    ((z_periodic || j < dhi[1]) && uz(i,j,k) == 0);
                s(i,j,k) = fixed ? amrex::Math::abs(s(i,j,k)-1._rt) : 0._rt;
            });
    }
    return sum.norm0();
#endif
}

amrex::Vector<amrex::Vector<amrex::Real>>
WarpXStaircaseChargeState (std::vector<std::string> const& psi_fields,
                          std::vector<std::string> const& weight_fields,
                          bool const insulating_endcaps)
{
#ifndef WARPX_DIM_RZ
    amrex::ignore_unused(insulating_endcaps);
    WARPX_ABORT_WITH_MESSAGE("The research staircase observer is implemented only in RZ");
#endif
    using namespace amrex::literals;
    using warpx::electrostatic::DepositionAxisFactor;
    using warpx::electrostatic::GaussAxisFactor;
    using warpx::electrostatic::IntegrateRhoPsi;
    using warpx::electrostatic::NodalDivEFromFp;

    auto& warpx = WarpX::GetInstance();
#ifdef WARPX_DIM_RZ
    if (insulating_endcaps) {
        ValidateInsulatingEndcaps();
    } else {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::field_boundary_lo[1] != FieldBoundaryType::PEC_Insulator &&
            WarpX::field_boundary_hi[1] != FieldBoundaryType::PEC_Insulator,
            "Native insulator faces require the insulating_endcaps charge convention");
    }
#endif
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(EB::enabled() && warpx.maxLevel() == 0
        && WarpX::ncomps == 1 && warpx.evolve_scheme == EvolveScheme::Explicit
        && WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee
        && WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::None
        && warpx.GetEMSolverMedium() == MediumForEM::Vacuum
        && WarpX::grid_type == GridType::Staggered,
        "The research staircase observer requires single-level explicit vacuum RZ Yee EM");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::do_moving_window == 0,
        "The research staircase observer requires a fixed simulation window");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!warpx.DoFluidSpecies(),
        "The research staircase observer currently supports particle species only");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!warpx.HasMirrors(),
        "The research staircase observer does not support internal field mirrors");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::gamma_boost == 1._rt,
        "The research staircase observer currently supports the laboratory frame only");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(!WarpX::use_filter && WarpX::nox <= 1,
        "The research staircase observer requires CIC particles and no rho filter");
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
        !psi_fields.empty() && psi_fields.size() == weight_fields.size(),
        "Staircase charge state needs one potential and fixed-node weight per electrode");
    for (std::size_t k = 0; k < psi_fields.size(); ++k) {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.m_fields.has(psi_fields[k], 0)
            && warpx.m_fields.has(weight_fields[k], 0),
            "Staircase charge state: a requested field is not registered");
    }
    auto const rho = warpx.DepositScratchRho(0);
    for (std::size_t k = 0; k < psi_fields.size(); ++k) {
        for (auto const& name : {psi_fields[k], weight_fields[k]}) {
            auto const* field = warpx.m_fields.get(name, 0);
            WARPX_ALWAYS_ASSERT_WITH_MESSAGE(field->nComp() == 1
                && field->boxArray() == rho->boxArray()
                && field->DistributionMap() == rho->DistributionMap(),
                "Staircase charge fields must be scalar native nodal fields on level zero");
        }
    }
    std::unique_ptr<amrex::MultiFab> axial_flux_density;
#ifdef WARPX_DIM_RZ
    if (insulating_endcaps) {
        // Reuse the native even-normal/linear-tangential guard extension.
        // For a boundary-compatible field in this whole-face, no-parser mode,
        // reapplying the axis, PEC and insulator rules leaves valid E unchanged.
        auto const e = warpx.m_fields.get_alldirs("Efield_fp", 0);
        warpx.ApplyEfieldBoundary(0, PatchType::fine, warpx.gett_new(0));
        axial_flux_density = std::make_unique<amrex::MultiFab>(
            rho->boxArray(), rho->DistributionMap(), 1, 0);
        auto const ndomain = amrex::surroundingNodes(warpx.Geom(0).Domain());
        int const zlo = ndomain.smallEnd(1);
        int const zhi = ndomain.bigEnd(1);
        auto const dz = warpx.Geom(0).CellSize(1);
        for (amrex::MFIter mfi(*axial_flux_density); mfi.isValid(); ++mfi) {
            auto const f = axial_flux_density->array(mfi);
            auto const ez = e[2]->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    // The native even Ez guard makes its face value equal to
                    // the adjacent cell value. This is outward axial flux / dz,
                    // not a new material-charge density or a particle ledger.
                    f(i,j,k) = j == zlo ? -ez(i,j,k)/dz :
                        (j == zhi ? ez(i,j-1,k)/dz : 0._rt);
                });
        }
    }
#endif
    auto const div_e = NodalDivEFromFp(0);
    amrex::Vector<amrex::Vector<amrex::Real>> result(insulating_endcaps ? 4 : 3);

#ifdef WARPX_DIM_RZ
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.Geom(0).ProbLo(0) == 0._rt,
        "The research staircase observer requires an RZ domain starting at the axis");
    if (DepositionAxisFactor() != GaussAxisFactor()) {
        amrex::MultiFab selected(rho->boxArray(), rho->DistributionMap(), 1, 0);
        selected.setVal(0._rt);
        for (auto const& name : weight_fields) {
            amrex::MultiFab::Add(selected, *warpx.m_fields.get(name, 0), 0, 0, 1, 0);
        }
        for (amrex::MFIter mfi(selected); mfi.isValid(); ++mfi) {
            auto const s = selected.array(mfi);
            auto const q = rho->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    s(i,j,k) = (i == 0 && s(i,j,k) == 0._rt) ? q(i,j,k) : 0._rt;
                });
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(selected.norm0() == 0._rt,
            "Staircase observer: free-axis plasma with the Verboncoeur correction "
            "is outside the verified source-measure envelope");
    }
#endif

    for (std::size_t k = 0; k < psi_fields.size(); ++k) {
        auto const* psi = warpx.m_fields.get(psi_fields[k], 0);
        auto const* weight = warpx.m_fields.get(weight_fields[k], 0);
        result[0].push_back(PhysConst::epsilon_0
            * IntegrateRhoPsi(*div_e, *weight, 0, GaussAxisFactor(), insulating_endcaps));
        result[1].push_back(IntegrateRhoPsi(*rho, *weight, 0, DepositionAxisFactor()));
        // Include fixed-node source support: the caller subtracts result[1]
        // from result[0], so Q_g must include the corresponding -q_fixed term.
        result[2].push_back(-IntegrateRhoPsi(*rho, *psi, 0, DepositionAxisFactor()));
        if (insulating_endcaps) {
            auto const boundary_charge = PhysConst::epsilon_0
                * IntegrateRhoPsi(*axial_flux_density, *psi, 0, GaussAxisFactor());
            result[2].back() += boundary_charge;
            result[3].push_back(boundary_charge);
        }
    }
    return result;
}
