/* Copyright 2026 The WarpX Community
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "NodalChargeMeasurement.H"

#include "Fields.H"
#include "WarpX.H"

#include <AMReX_MFIter.H>
#include <AMReX_Math.H>

using namespace amrex::literals;

namespace warpx::electrostatic
{
    amrex::Real DepositionAxisFactor ()
    {
#ifdef WARPX_DIM_RZ
        return WarpX::GetInstance().RZAxisVolumeFactor();
#else
        return amrex::Real(0.25);
#endif
    }

    /* Nodal inner product sum_a f[a] g[a] V_a. In RZ the axis-node volume differs
     * between deposited charge (DepositionAxisFactor, 1/3 with the Verboncoeur
     * correction) and the divergence control volume (GaussAxisFactor, 1/4), so the
     * caller supplies it. See the electrode voltage clamp documentation.
     */
    amrex::Real IntegrateRhoPsi (
        amrex::MultiFab const& rho,
        amrex::MultiFab const& psi,
        int lev,
        amrex::Real axis_factor)
    {
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            rho.boxArray() == psi.boxArray() &&
            rho.DistributionMap() == psi.DistributionMap() &&
            rho.ixType() == psi.ixType(),
            "IntegrateRhoPsi: rho and psi must use the same nodal layout");

        auto& warpx = WarpX::GetInstance();
        amrex::MultiFab product(rho.boxArray(), rho.DistributionMap(), 1, 0);
        const auto dx = warpx.Geom(lev).CellSizeArray();
#ifdef WARPX_DIM_RZ
        const amrex::Real dr = dx[0];
        const amrex::Real dz = dx[1];
        const amrex::Real rlo = warpx.Geom(lev).ProbLo(0);
#else
        amrex::ignore_unused(axis_factor);
        const amrex::Real node_volume = dx[0] * dx[1] * dx[2];
#endif

        for (amrex::MFIter mfi(product); mfi.isValid(); ++mfi) {
            auto const& qa = rho.const_array(mfi);
            auto const& pa = psi.const_array(mfi);
            auto const& wa = product.array(mfi);
            amrex::ParallelFor(mfi.validbox(),
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
#ifdef WARPX_DIM_RZ
                    const amrex::Real r = rlo + amrex::Real(i)*dr;
                    const amrex::Real radial_measure = (r == 0._rt)
                        ? MathConst::pi*dr*axis_factor : 2._rt*MathConst::pi*r;
                    const amrex::Real node_volume = dr*dz*radial_measure;
#endif
                    wa(i,j,k) = qa(i,j,k,0) * pa(i,j,k,0) * node_volume;
                });
        }

        return product.sum_unique(0, false, warpx.Geom(lev).periodicity());
    }

    std::unique_ptr<amrex::MultiFab> NodalDivEFromFp (int lev)
    {
        auto& warpx = WarpX::GetInstance();
        amrex::BoxArray nodal_ba = warpx.boxArray(lev);
        nodal_ba.surroundingNodes();
        // Only m = 0 is integrated (ComputeDivE writes 2*nmodes-1 components).
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::ncomps == 1,
            "The staircase charge observer supports n_rz_azimuthal_modes = 1 only");
        auto div_e = std::make_unique<amrex::MultiFab>(
            nodal_ba, warpx.DistributionMap(lev), WarpX::ncomps, 0);

        // The divergence stencil reads guard cells, which may be stale after a push
        // or solve; refresh them (valid data unchanged) so the result does not
        // depend on the domain decomposition.
        ablastr::fields::VectorField const E =
            warpx.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp, lev);
        for (int idim = 0; idim < 3; ++idim) {
            E[idim]->FillBoundary(warpx.Geom(lev).periodicity());
        }

        // Efield_fp is the field the Maxwell solver advances (aux may differ).
        warpx.ComputeDivE(*div_e, lev, warpx::fields::FieldType::Efield_fp);
        return div_e;
    }
}
