/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "VoltageClamp.H"

#include "EmbeddedBoundary/Enabled.H"
#include "FieldSolver/ElectrostaticSolvers/ElectrostaticSolver.H"
#include "FieldSolver/FiniteDifferenceSolver/FiniteDifferenceSolver.H"
#include "Fields.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXAlgorithmSelection.H"
#include "Utils/WarpXConst.H"
#include "WarpX.H"

#include <ablastr/fields/MultiFabRegister.H>

#include <AMReX_GpuLaunch.H>
#include <AMReX_MFIter.H>
#include <AMReX_Math.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_Reduce.H>
#include <AMReX_iMultiFab.H>
#ifdef AMREX_USE_EB
#   include <AMReX_EB2.H>
#   include <AMReX_EB2_IF_AllRegular.H>
#   include <AMReX_EBFabFactory.H>
#   include <AMReX_MLEBNodeFDLaplacian.H>
#endif
#include <AMReX_MLMG.H>

#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace amrex::literals;

namespace warpx::embedded_boundary::voltage_clamp
{

std::string EFieldName (int k)
{
    return k < 0 ? "voltage_clamp_E_grounded" : "voltage_clamp_E_" + std::to_string(k);
}

#if !(defined(WARPX_DIM_RZ) && defined(AMREX_USE_EB))

namespace
{
    void NotImplemented ()
    {
        WARPX_ABORT_WITH_MESSAGE(
            "The embedded-boundary voltage clamp is implemented for RZ builds with "
            "embedded boundaries only");
    }
}

amrex::Real SolveUnitPotential (int, std::string const&, amrex::Real, int)
{ NotImplemented(); return 0._rt; }
amrex::Real SolveGrounded (amrex::Real, int) { NotImplemented(); return 0._rt; }
void CheckSelection (int) { NotImplemented(); }
std::vector<amrex::Real> GaussCharge (int, std::string const&) { NotImplemented(); return {}; }
std::vector<std::vector<amrex::Real>> CapacitanceMatrix (int) { NotImplemented(); return {}; }
std::pair<std::vector<amrex::Real>, std::vector<amrex::Real>>
ChargeState (int) { NotImplemented(); return {}; }
void AddUnitFields (std::vector<amrex::Real> const&) { NotImplemented(); }

#else

namespace
{
    constexpr int lev = 0;

    std::string PhiName (int k)
    {
        return k < 0 ? "voltage_clamp_phi_grounded" : "voltage_clamp_phi_" + std::to_string(k);
    }
    std::string WeightName (int k) { return "voltage_clamp_weight_" + std::to_string(k); }
    // Node measures (volume x owner mask) for Gauss and deposition integrals
    char const* const gauss_measure = "voltage_clamp_measure_gauss";
    char const* const rho_measure = "voltage_clamp_measure_rho";
    // Union of the fixed nodes of all conductors
    char const* const fixed_nodes = "voltage_clamp_fixed";

    amrex::BoxArray NodalBoxArray (WarpX const& warpx)
    {
        return amrex::convert(warpx.boxArray(lev), amrex::IntVect::TheNodeVector());
    }

    /* Abort outside the supported envelope. The observer is exact when rho and J
     * satisfy discrete continuity on the staircase: one level, explicit staggered
     * Yee, charge-conserving deposition, no filter. Particle shapes above 1 also
     * need rho deposited with the EB reduced shape that Esirkepov J uses. */
    void CheckSupported (WarpX const& warpx)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(EB::enabled(),
            "The voltage clamp requires an embedded boundary");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.maxLevel() == 0,
            "The voltage clamp supports one AMR level");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(WarpX::n_rz_azimuthal_modes == 1,
            "The voltage clamp supports RZ mode m=0 only");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(warpx.Geom(lev).ProbLo(0) == 0._rt,
            "The voltage clamp requires an RZ domain that includes the axis");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::electromagnetic_solver_id == ElectromagneticSolverAlgo::Yee &&
            WarpX::grid_type == GridType::Staggered &&
            warpx.evolve_scheme == EvolveScheme::Explicit &&
            WarpX::electrostatic_solver_id == ElectrostaticSolverAlgo::None,
            "The voltage clamp requires the explicit staggered Yee solver");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            warpx.GetEMSolverMedium() == MediumForEM::Vacuum && !warpx.HasMirrors() &&
            !warpx.DoFluidSpecies() && WarpX::do_moving_window == 0 &&
            WarpX::gamma_boost == 1._rt,
            "The voltage clamp requires a vacuum medium, no mirrors, no fluids, a fixed "
            "window and the laboratory frame");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !WarpX::use_filter &&
            WarpX::current_deposition_algo == CurrentDepositionAlgo::Esirkepov &&
            WarpX::nox == 1,
            "The voltage clamp requires Esirkepov deposition, particle_shape = 1 and "
            "no current/charge filter");
        bool const z_periodic = warpx.Geom(lev).isPeriodic(1);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::field_boundary_lo[0] == FieldBoundaryType::None &&
            WarpX::field_boundary_hi[0] == FieldBoundaryType::PEC &&
            (z_periodic || (WarpX::field_boundary_lo[1] == FieldBoundaryType::PEC &&
                            WarpX::field_boundary_hi[1] == FieldBoundaryType::PEC)),
            "The voltage clamp supports a PEC outer radius and two PEC or two "
            "periodic z boundaries");
    }

    // --- Geometry: everything specific to RZ is in the helpers below ---------

    /* Node classes on the staircase. A node is fixed if it ends a frozen E edge (an
     * Er or Ez edge that the EB update mask excludes from the FDTD update). Wall
     * nodes lie on a grounded PEC domain boundary. */
    struct Staircase
    {
        amrex::Array4<int const> ur, uz;
        amrex::IntVect dlo, dhi;
        bool z_periodic;

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        bool fixed (int i, int j, int k) const noexcept
        {
            return (i > dlo[0] && ur(i-1,j,k) == 0) || (i < dhi[0] && ur(i,j,k) == 0) ||
                ((z_periodic || j > dlo[1]) && uz(i,j-1,k) == 0) ||
                ((z_periodic || j < dhi[1]) && uz(i,j,k) == 0);
        }

        AMREX_GPU_DEVICE AMREX_FORCE_INLINE
        bool wall (int i, int j) const noexcept
        {
            return i == dhi[0] || (!z_periodic && (j == dlo[1] || j == dhi[1]));
        }
    };

    Staircase MakeStaircase (WarpX& warpx, amrex::MFIter const& mfi)
    {
        auto& update = warpx.GetEBUpdateEFlag()[lev];
        amrex::Box const domain = amrex::surroundingNodes(warpx.Geom(lev).Domain());
        return {update[0]->const_array(mfi), update[2]->const_array(mfi),
                domain.smallEnd(), domain.bigEnd(), warpx.Geom(lev).isPeriodic(1)};
    }

    /* Node volumes: 2 pi r dr dz off axis. On the axis, deposition uses
     * pi dr^2 dz * 4 * RZAxisVolumeFactor (1/3 with the Verboncoeur correction),
     * the divergence control volume uses 1/4. */
    void FillMeasure (WarpX const& warpx, amrex::MultiFab& measure, amrex::Real axis_factor)
    {
        auto const dx = warpx.Geom(lev).CellSizeArray();
        auto const owner = measure.OwnerMask(warpx.Geom(lev).periodicity());
        for (amrex::MFIter mfi(measure); mfi.isValid(); ++mfi) {
            auto const m = measure.array(mfi);
            auto const own = owner->const_array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                amrex::Real const r = i * dx[0];
                amrex::Real const radial = (i == 0) ? MathConst::pi * dx[0] * axis_factor
                                                    : 2._rt * MathConst::pi * r;
                m(i,j,k) = own(i,j,k) ? radial * dx[0] * dx[1] : 0._rt;
            });
        }
    }

    // sum_nodes a b measure (each node once)
    amrex::Real Integrate (amrex::MultiFab const& a, amrex::MultiFab const& b,
                           amrex::MultiFab const& measure)
    {
        amrex::ReduceOps<amrex::ReduceOpSum> op;
        amrex::ReduceData<amrex::Real> data(op);
        for (amrex::MFIter mfi(measure); mfi.isValid(); ++mfi) {
            auto const aa = a.const_array(mfi);
            auto const ba = b.const_array(mfi);
            auto const ma = measure.const_array(mfi);
            op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                -> amrex::GpuTuple<amrex::Real>
            {
                return {aa(i,j,k) * ba(i,j,k) * ma(i,j,k)};
            });
        }
        amrex::Real sum = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealSum(sum);
        return sum;
    }

    void AllocNodal (WarpX& warpx, std::string const& name)
    {
        if (!warpx.m_fields.has(name, lev)) {
            warpx.m_fields.alloc_init(name, lev, NodalBoxArray(warpx),
                warpx.DistributionMap(lev), 1, amrex::IntVect(1), 0._rt);
        }
    }

    void AllocLikeE (WarpX& warpx, std::string const& name)
    {
        using ablastr::fields::Direction;
        for (int d = 0; d < 3; ++d) {
            if (warpx.m_fields.has(name, Direction{d}, lev)) { continue; }
            auto const* e = warpx.m_fields.get(warpx::fields::FieldType::Efield_fp,
                                               Direction{d}, lev);
            warpx.m_fields.alloc_init(name, Direction{d}, lev, e->boxArray(),
                e->DistributionMap(), 1, e->nGrowVect(), 0._rt);
        }
    }

    /* Potential on the frozen-edge staircase:
     *   phi = selector (0 or 1) on the fixed nodes, 0 on the PEC walls,
     *   div grad phi = -source/eps0 elsewhere (MLEBNodeFDLaplacian on an all-regular
     *   geometry, fixed nodes as overset mask), then E = -grad phi with the Yee
     *   gradient, so curl E = 0 and the frozen edges stay zero. */
    amrex::Real SolveStaircase (int k, std::string const& selector,
                                amrex::MultiFab const* source,
                                amrex::Real rtol, int max_iter)
    {
        auto& warpx = WarpX::GetInstance();
        CheckSupported(warpx);
        auto const& geom = warpx.Geom(lev);
        auto const period = geom.periodicity();
        auto& update = warpx.GetEBUpdateEFlag()[lev];
        update[0]->FillBoundary(period);
        update[2]->FillBoundary(period);

        AllocNodal(warpx, PhiName(k));
        AllocLikeE(warpx, EFieldName(k));
        amrex::MultiFab& phi = *warpx.m_fields.get(PhiName(k), lev);
        amrex::MultiFab weight(phi.boxArray(), phi.DistributionMap(), 1, 0);
        amrex::iMultiFab overset(phi.boxArray(), phi.DistributionMap(), 1, 0);
        amrex::iMultiFab invalid(phi.boxArray(), phi.DistributionMap(), 1, 0);

        amrex::Parser parser = utils::parser::makeParser(selector, {"x", "y", "z"});
        auto const select = parser.compile<3>();
        auto const dx = geom.CellSizeArray();
        amrex::Real const zlo = geom.ProbLo(1);
        amrex::Box const domain = amrex::surroundingNodes(geom.Domain());
        auto const dlo = domain.smallEnd();
        auto const dhi = domain.bigEnd();
        for (amrex::MFIter mfi(phi); mfi.isValid(); ++mfi) {
            auto const sc = MakeStaircase(warpx, mfi);
            auto const p = phi.array(mfi);
            auto const w = weight.array(mfi);
            auto const m = overset.array(mfi);
            auto const bad = invalid.array(mfi);
            amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int kk)
            {
                // node (r, z) -> parser (x, y, z) = (r, 0, z)
                auto const s_at = [=] (int ii, int jj) {
                    return select(ii*dx[0], 0._rt, zlo + jj*dx[1]);
                };
                amrex::Real const s = s_at(i, j);
                bool const fixed = sc.fixed(i,j,kk);
                bool const wall = sc.wall(i,j);
                p(i,j,kk) = (fixed && s == 1._rt) ? 1._rt : 0._rt;
                w(i,j,kk) = p(i,j,kk);
                m(i,j,kk) = (fixed || wall) ? 0 : 1; // overset: 1 = unknown
                // The selector must be 0 or 1 on fixed nodes, constant across frozen
                // edges (one value per conductor) and equal on periodic copies, and
                // no conductor may touch a PEC wall.
                bad(i,j,kk) = fixed && (
                    (s != 0._rt && s != 1._rt) || wall ||
                    (i < dhi[0] && sc.ur(i,j,kk) == 0 && s != s_at(i+1, j)) ||
                    (j < dhi[1] && sc.uz(i,j,kk) == 0 && s != s_at(i, j+1)) ||
                    (sc.z_periodic && j == dlo[1] && s != s_at(i, dhi[1])));
            });
        }
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(invalid.max(0) == 0,
            "The voltage-clamp selector must be 0 or 1 on every fixed node and constant "
            "on each conductor, and no conductor may touch a PEC domain wall");
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(source != nullptr ||
            weight.sum_unique(0, false, period) > 0._rt,
            "The voltage-clamp selector does not select any fixed node");
        phi.OverrideSync(period);
        weight.OverrideSync(period);
        overset.OverrideSync(period);

        if (k >= 0) {
            AllocNodal(warpx, WeightName(k));
            amrex::MultiFab::Copy(*warpx.m_fields.get(WeightName(k), lev), weight,
                                  0, 0, 1, 0);
        }

        amrex::MultiFab rhs(phi.boxArray(), phi.DistributionMap(), 1, 0);
        rhs.setVal(0._rt);
        if (source != nullptr) {
            for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
                auto const r = rhs.array(mfi);
                auto const q = source->const_array(mfi);
                auto const m = overset.const_array(mfi);
                amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int kk)
                {
                    r(i,j,kk) = m(i,j,kk) ? -q(i,j,kk) / PhysConst::epsilon_0 : 0._rt;
                });
            }
        }

        // The fixed nodes are the conductors, so the operator itself has no EB
        amrex::EB2::AllRegularIF all_regular;
        amrex::EB2::GeometryShop<amrex::EB2::AllRegularIF> shop(all_regular);
        amrex::EB2::IndexSpaceImp<decltype(shop)> index(shop, geom, 0, 30, 1, true, false, 0);
        auto const factory = amrex::makeEBFabFactory(&index, geom, warpx.boxArray(lev),
            warpx.DistributionMap(lev), {1, 1, 1}, amrex::EBSupport::full);
        amrex::MLEBNodeFDLaplacian linop({geom}, {warpx.boxArray(lev)},
            {warpx.DistributionMap(lev)}, amrex::LPInfo{}, {factory.get()});
        linop.setRZ(true);
        linop.setSigma({1._rt, 1._rt});
        linop.setAlpha(0._rt);
        auto& bc = *warpx.GetElectrostaticSolver().m_poisson_boundary_handler;
        bc.DefinePhiBCs(geom);
        linop.setDomainBC(bc.lobc, bc.hibc);
        linop.setOversetMask(lev, overset);
        amrex::MLMG mlmg(linop);
        mlmg.setMaxIter(max_iter);
        amrex::Real const residual = mlmg.solve({&phi}, {&rhs}, rtol, 0._rt);
        phi.FillBoundary(period);

        ablastr::fields::VectorField efield = warpx.m_fields.get_alldirs(EFieldName(k), lev);
        for (auto* e : efield) { e->setVal(0._rt); }
        ablastr::fields::MultiLevelVectorField const e_levels{efield};
        ablastr::fields::MultiLevelScalarField const phi_levels{&phi};
        warpx.GetElectrostaticSolver().computeE(e_levels, phi_levels, {0._rt, 0._rt, 0._rt});
        for (auto* e : efield) { e->FillBoundary(period); }
        return residual;
    }

    std::unique_ptr<amrex::MultiFab> NodalDivE (WarpX& warpx, std::string const& name)
    {
        // The divergence stencil reads guard cells; syncing them from valid data
        // keeps the result independent of the domain decomposition.
        ablastr::fields::VectorField efield = warpx.m_fields.get_alldirs(name, lev);
        for (auto* e : efield) { e->FillBoundary(warpx.Geom(lev).periodicity()); }
        auto div_e = std::make_unique<amrex::MultiFab>(
            NodalBoxArray(warpx), warpx.DistributionMap(lev), 1, 0);
        warpx.get_pointer_fdtd_solver_fp(lev)->ComputeDivE(efield, *div_e);
        return div_e;
    }
}

amrex::Real SolveUnitPotential (int k, std::string const& selector,
                                amrex::Real rtol, int max_iter)
{
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(k >= 0, "Conductor indices start at 0");
    auto& warpx = WarpX::GetInstance();
    if (!warpx.m_fields.has(gauss_measure, lev)) {
        AllocNodal(warpx, gauss_measure);
        AllocNodal(warpx, rho_measure);
        FillMeasure(warpx, *warpx.m_fields.get(gauss_measure, lev), 0.25_rt);
        FillMeasure(warpx, *warpx.m_fields.get(rho_measure, lev),
                    warpx.RZAxisVolumeFactor());
    }
    return SolveStaircase(k, selector, nullptr, rtol, max_iter);
}

amrex::Real SolveGrounded (amrex::Real rtol, int max_iter)
{
    auto const rho = WarpX::GetInstance().DepositScratchRho(lev);
    return SolveStaircase(-1, "0", rho.get(), rtol, max_iter);
}

void CheckSelection (int n)
{
    auto& warpx = WarpX::GetInstance();
    amrex::MultiFab sum(NodalBoxArray(warpx), warpx.DistributionMap(lev), 1, 0);
    sum.setVal(0._rt);
    for (int k = 0; k < n; ++k) {
        amrex::MultiFab::Add(sum, *warpx.m_fields.get(WeightName(k), lev), 0, 0, 1, 0);
    }
    // Also record all fixed and wall nodes, for the observer
    AllocNodal(warpx, fixed_nodes);
    amrex::MultiFab& fixed_mask = *warpx.m_fields.get(fixed_nodes, lev);
    amrex::iMultiFab invalid(sum.boxArray(), sum.DistributionMap(), 1, 0);
    for (amrex::MFIter mfi(sum); mfi.isValid(); ++mfi) {
        auto const sc = MakeStaircase(warpx, mfi);
        auto const s = sum.const_array(mfi);
        auto const f = fixed_mask.array(mfi);
        auto const bad = invalid.array(mfi);
        amrex::ParallelFor(mfi.validbox(), [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            bool const fixed = sc.fixed(i,j,k);
            f(i,j,k) = (fixed || sc.wall(i,j)) ? 1._rt : 0._rt;
            bad(i,j,k) = fixed && s(i,j,k) != 1._rt;
        });
    }
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(invalid.max(0) == 0,
        "Every fixed node must belong to exactly one conductor; list grounded "
        "conductors too");
}

std::vector<amrex::Real> GaussCharge (int n, std::string const& efield_name)
{
    auto& warpx = WarpX::GetInstance();
    auto const div_e = NodalDivE(warpx, efield_name);
    auto const& measure = *warpx.m_fields.get(gauss_measure, lev);
    std::vector<amrex::Real> charge(n);
    for (int k = 0; k < n; ++k) {
        charge[k] = PhysConst::epsilon_0 *
            Integrate(*div_e, *warpx.m_fields.get(WeightName(k), lev), measure);
    }
    return charge;
}

std::vector<std::vector<amrex::Real>> CapacitanceMatrix (int n)
{
    std::vector<std::vector<amrex::Real>> cap(n, std::vector<amrex::Real>(n));
    for (int k = 0; k < n; ++k) {
        auto const column = GaussCharge(n, EFieldName(k));
        for (int j = 0; j < n; ++j) { cap[j][k] = column[j]; }
    }
    return cap;
}

std::pair<std::vector<amrex::Real>, std::vector<amrex::Real>>
ChargeState (int n)
{
    auto& warpx = WarpX::GetInstance();
    auto charge = GaussCharge(n, "Efield_fp");
    auto const rho = warpx.DepositScratchRho(lev);
    auto const& measure = *warpx.m_fields.get(rho_measure, lev);

    // With the Verboncoeur correction the axis node volumes of rho and div E
    // differ. That is exact on fixed nodes but not verified for charge on free
    // axis nodes, so such charge is rejected.
    if (warpx.RZAxisVolumeFactor() != 0.25_rt) {
        auto const& fixed = *warpx.m_fields.get(fixed_nodes, lev);
        amrex::ReduceOps<amrex::ReduceOpMax> op;
        amrex::ReduceData<amrex::Real> data(op);
        for (amrex::MFIter mfi(fixed); mfi.isValid(); ++mfi) {
            auto const q = rho->const_array(mfi);
            auto const f = fixed.const_array(mfi);
            op.eval(mfi.validbox(), data, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                -> amrex::GpuTuple<amrex::Real>
            {
                return {(i == 0 && f(i,j,k) == 0._rt) ? amrex::Math::abs(q(i,j,k)) : 0._rt};
            });
        }
        amrex::Real free_axis_rho = amrex::get<0>(data.value());
        amrex::ParallelDescriptor::ReduceRealMax(free_axis_rho);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(free_axis_rho == 0._rt,
            "The voltage clamp does not support charge on free axis nodes with the "
            "Verboncoeur axis correction");
    }

    std::vector<amrex::Real> grounded(n);
    for (int k = 0; k < n; ++k) {
        charge[k] -= Integrate(*rho, *warpx.m_fields.get(WeightName(k), lev), measure);
        grounded[k] = -Integrate(*rho, *warpx.m_fields.get(PhiName(k), lev), measure);
    }
    return {charge, grounded};
}

void AddUnitFields (std::vector<amrex::Real> const& dv)
{
    auto& warpx = WarpX::GetInstance();
    ablastr::fields::VectorField efield =
        warpx.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp, lev);
    for (int k = 0; k < static_cast<int>(dv.size()); ++k) {
        ablastr::fields::VectorField unit = warpx.m_fields.get_alldirs(EFieldName(k), lev);
        for (int d = 0; d < 3; ++d) {
            amrex::MultiFab::Saxpy(*efield[d], dv[k], *unit[d], 0, 0, 1, 0);
        }
    }
    // Same guard-cell refresh as after a field push
    warpx.ApplyEfieldBoundary(lev, PatchType::fine, warpx.gett_new(lev));
    warpx.FillBoundaryE(lev, warpx.getngEB(), true);
}

#endif

}
