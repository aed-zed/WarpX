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

    /* Nodal inner product sum_a f[a] g[a] V_a. The axis measure is supplied by the
     * caller because the two uses need DIFFERENT volumes:
     *
     *   deposition measure   V_axis = pi dr^2 dz * RZAxisVolumeFactor()
     *       what makes sum rho V equal the deposited charge: 1/3 with the
     *       Verboncoeur correction, which is on by default, otherwise 1/4.
     *
     *   Gauss measure        V_axis = pi dr^2 dz / 4
     *       the axis node's geometric control volume, a cylinder of radius dr/2 and
     *       height dz. That is the volume the on-axis stencil 4 Er/dr is the
     *       divergence over: (4 Er/dr)(pi dr^2 dz/4) = pi dr dz Er, which is exactly
     *       the flux Er(dr/2) * 2 pi (dr/2) dz through its curved surface.
     *
     * Off the axis the two coincide. Using the deposition measure for a divergence
     * integral breaks Gauss's law at the axis node by 4/3 at the default setting:
     * the flux through an enclosing surface can be zero while the reported enclosed
     * charge is not. An adjoint can match that wrong integral perfectly, so the
     * pairing check cannot detect it.
     */
    amrex::Real IntegrateRhoPsi (
        amrex::MultiFab const& rho,
        amrex::MultiFab const& psi,
        int lev,
        amrex::Real axis_factor,
        bool half_axial_endpoints)
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
        auto const ndomain = amrex::surroundingNodes(warpx.Geom(lev).Domain());
        int const zlo = ndomain.smallEnd(1);
        int const zhi = ndomain.bigEnd(1);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            !half_axial_endpoints || !warpx.Geom(lev).isPeriodic(1),
            "Half axial endpoint volumes require nonperiodic z");
#else
        amrex::ignore_unused(axis_factor, half_axial_endpoints);
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
                    const amrex::Real axial_factor = half_axial_endpoints &&
                        (j == zlo || j == zhi) ? 0.5_rt : 1._rt;
                    const amrex::Real node_volume = dr*dz*radial_measure*axial_factor;
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
        // ComputeDivECylindrical writes 2*nmodes-1 components, so allocating
        // one would be an out-of-bounds write for nmodes > 1. Allocate what the
        // solver writes, and refuse the modes this integral does not handle:
        // only the m = 0 component is integrated below, and a region weight
        // w(x,y,z) carries no azimuthal dependence to pair with m > 0.
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            WarpX::ncomps == 1,
            "The volume charge observer supports a single azimuthal mode only "
            "(n_rz_azimuthal_modes = 1); the m > 0 contributions to the enclosed "
            "charge are not integrated.");
        auto div_e = std::make_unique<amrex::MultiFab>(
            nodal_ba, warpx.DistributionMap(lev), WarpX::ncomps, 0);

        // The divergence stencil at a node on a box boundary reaches into the
        // guard cells, and after a field push or a Poisson solve WarpX leaves
        // those outdated. Refresh them, or the integral stops being independent
        // of the domain decomposition: measured on a 96^3 sphere fixture, a
        // region whose boundary crossed a box edge came out 2.8% low with 27
        // boxes and every region was wrong with 216, while the single-box
        // answer was exact. Only ghost cells are written; the valid region is
        // untouched, so this is a refresh and not a change of state.
        ablastr::fields::VectorField const E =
            warpx.m_fields.get_alldirs(warpx::fields::FieldType::Efield_fp, lev);
        for (int idim = 0; idim < 3; ++idim) {
            E[idim]->FillBoundary(warpx.Geom(lev).periodicity());
        }

        // Efield_fp, not Efield_aux: aux is only an alias of fp at level 0
        // without time averaging, read-from-file external fields, or a
        // collocated grid.
        warpx.ComputeDivE(*div_e, lev, warpx::fields::FieldType::Efield_fp);
        return div_e;
    }
}
