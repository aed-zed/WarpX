/* Copyright 2025 Marco Garten
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#include "TangentialEOnEB.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "EmbeddedBoundary/Enabled.H"
#include "Fields.H"
#include "Utils/TextMsg.H"
#include "WarpX.H"

#include <AMReX_Config.H>
#include <AMReX_Geometry.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

using namespace amrex;

TangentialEOnEB::TangentialEOnEB (const std::string& rd_name)
: ReducedDiags{rd_name}
{
#if !(defined WARPX_DIM_3D)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(false,
        "TangentialEOnEB reduced diagnostic only works in 3D");
#endif
#if !(defined AMREX_USE_EB)
    WARPX_ALWAYS_ASSERT_WITH_MESSAGE(false,
        "TangentialEOnEB reduced diagnostic requires EB support");
#endif
    if (!EB::enabled()) {
        throw std::runtime_error(
            "TangentialEOnEB reduced diagnostic requires EBs enabled at runtime");
    }

    // Two output columns: max|E_tan|, area-weighted RMS|E_tan|
    m_data.resize(2, 0.0_rt);

    if (ParallelDescriptor::IOProcessor())
    {
        if (m_write_header)
        {
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension,
                              std::ofstream::out};
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";
            ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]max|E_tan|(V/m)";
            ofs << m_sep;
            ofs << "[" << c++ << "]rms|E_tan|(V/m)";
            ofs << "\n";
            ofs.close();
        }
    }
}

void TangentialEOnEB::ComputeDiags (const int step)
{
    if (!m_intervals.contains(step+1)) { return; }

    if (!EB::enabled()) {
        throw std::runtime_error(
            "TangentialEOnEB::ComputeDiags requires EBs enabled at runtime");
    }

#if ((defined WARPX_DIM_3D) && (defined AMREX_USE_EB))
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    auto & warpx = WarpX::GetInstance();
    int const lev = 0;

    const amrex::MultiFab & Ex = *warpx.m_fields.get(FieldType::Efield_fp, Direction{0}, lev);
    const amrex::MultiFab & Ey = *warpx.m_fields.get(FieldType::Efield_fp, Direction{1}, lev);
    const amrex::MultiFab & Ez = *warpx.m_fields.get(FieldType::Efield_fp, Direction{2}, lev);

    amrex::EBFArrayBoxFactory const& eb_box_factory = warpx.fieldEBFactory(lev);
    amrex::FabArray<amrex::EBCellFlagFab> const& eb_flag =
        eb_box_factory.getMultiEBCellFlagFab();
    amrex::MultiCutFab const& eb_bnd_normal = eb_box_factory.getBndryNormal();
    amrex::Array<const amrex::MultiCutFab*,AMREX_SPACEDIM> eb_area_fraction =
        eb_box_factory.getAreaFrac();

    const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> dx =
        warpx.Geom(lev).CellSizeArray();

    // GPU buffers: [0] = max|E_tan|^2, [1] = sum(|E_tan|^2 * dA), [2] = sum(dA)
    amrex::Gpu::Buffer<amrex::Real> buf({0.0_rt, 0.0_rt, 0.0_rt});
    amrex::Real* buf_ptr = buf.data();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(Ex, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const amrex::Box & box = mfi.tilebox(amrex::IntVect::TheCellVector());
        const amrex::FabType fab_type = eb_flag[mfi].getType(box);
        if (fab_type == amrex::FabType::regular) { continue; }
        if (fab_type == amrex::FabType::covered) { continue; }

        auto const& Ex_arr = Ex.const_array(mfi);
        auto const& Ey_arr = Ey.const_array(mfi);
        auto const& Ez_arr = Ez.const_array(mfi);
        auto const& eb_flag_arr = eb_flag.const_array(mfi);
        auto const& n_arr = eb_bnd_normal.const_array(mfi);
        auto const& ax_arr = eb_area_fraction[0]->const_array(mfi);
        auto const& ay_arr = eb_area_fraction[1]->const_array(mfi);
        auto const& az_arr = eb_area_fraction[2]->const_array(mfi);

        amrex::ParallelFor(box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) {

                if (eb_flag_arr(i,j,k).isRegular() ||
                    eb_flag_arr(i,j,k).isCovered()) { return; }

                // Interpolate E to cell center from surrounding Yee edges.
                // Ex lives at (i+1/2, j, k), so cell-center average is
                // 0.5*(Ex(i,j,k) + Ex(i+1,j,k)), etc.
                amrex::Real ex_cc = 0.5_rt * (Ex_arr(i,j,k) + Ex_arr(i+1,j,k));
                amrex::Real ey_cc = 0.5_rt * (Ey_arr(i,j,k) + Ey_arr(i,j+1,k));
                amrex::Real ez_cc = 0.5_rt * (Ez_arr(i,j,k) + Ez_arr(i,j,k+1));

                // EB surface normal (points into the conductor)
                amrex::Real nx = n_arr(i,j,k,0);
                amrex::Real ny = n_arr(i,j,k,1);
                amrex::Real nz = n_arr(i,j,k,2);

                // E_normal = (E . n)
                amrex::Real e_dot_n = ex_cc*nx + ey_cc*ny + ez_cc*nz;

                // E_tan = E - (E.n) n
                amrex::Real etx = ex_cc - e_dot_n * nx;
                amrex::Real ety = ey_cc - e_dot_n * ny;
                amrex::Real etz = ez_cc - e_dot_n * nz;
                amrex::Real etan_sq = etx*etx + ety*ety + etz*etz;

                // EB surface area element for this cut cell (magnitude of the
                // outward EB area vector, reconstructed from face area fractions)
                amrex::Real dAx = dx[1]*dx[2] *
                    (ax_arr(i+1,j,k) - ax_arr(i,j,k));
                amrex::Real dAy = dx[2]*dx[0] *
                    (ay_arr(i,j+1,k) - ay_arr(i,j,k));
                amrex::Real dAz = dx[0]*dx[1] *
                    (az_arr(i,j,k+1) - az_arr(i,j,k));
                amrex::Real dA = std::sqrt(dAx*dAx + dAy*dAy + dAz*dAz);

                // max|E_tan|^2
                amrex::Gpu::Atomic::Max(&buf_ptr[0], etan_sq);
                // sum(|E_tan|^2 * dA)
                amrex::Gpu::Atomic::Add(&buf_ptr[1], etan_sq * dA);
                // sum(dA)
                amrex::Gpu::Atomic::Add(&buf_ptr[2], dA);
        });
    }

    buf.copyToHost();
    amrex::Real max_etan_sq = buf.hostData()[0];
    amrex::Real sum_etan_sq_dA = buf.hostData()[1];
    amrex::Real sum_dA = buf.hostData()[2];

    amrex::ParallelDescriptor::ReduceRealMax(max_etan_sq);
    amrex::ParallelDescriptor::ReduceRealSum(sum_etan_sq_dA);
    amrex::ParallelDescriptor::ReduceRealSum(sum_dA);

    m_data[0] = std::sqrt(max_etan_sq);
    m_data[1] = (sum_dA > 0.0_rt)
              ? std::sqrt(sum_etan_sq_dA / sum_dA)
              : 0.0_rt;
#endif
}
