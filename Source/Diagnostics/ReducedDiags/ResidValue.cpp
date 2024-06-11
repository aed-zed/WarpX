/* Copyright 2019-2020 Neil Zaim, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#include "ResidValue.H"

#include "Diagnostics/ReducedDiags/ReducedDiags.H"
#include "WarpX.H"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_PODVector.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParticleReduce.H>
#include <AMReX_Particles.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <map>
#include <ostream>
#include <vector>

using namespace amrex::literals;
// constructor
ResidValue::ResidValue (const std::string& rd_name)
: ReducedDiags{rd_name}
{

    m_data.resize(1, 0.0_rt);
    if (amrex::ParallelDescriptor::IOProcessor())
    {
        if ( m_write_header )
        {
            // open file
            std::ofstream ofs{m_path + m_rd_name + "." + m_extension, std::ofstream::out};
            // write header row
            int c = 0;
            ofs << "#";
            ofs << "[" << c++ << "]step()";
            ofs << m_sep;
            ofs << "[" << c++ << "]time(s)";
            ofs << m_sep;
            ofs << "[" << c++ << "]residual";
            ofs << std::endl;
            // close file
            ofs.close();
        }
    }
}
// end constructor
// function that computes residual value after poisson equation
void ResidValue::ComputeDiags (int step)
{
    // Judge if the diags should be done
    // if (!m_intervals.contains(step+1)) { return; }

    auto & warpx = WarpX::GetInstance();
    // do the diag anytime the poisson equation is calculated
    if (warpx.getPoissonSkipped()) {
        return;
    }
    m_data[0] = warpx.getPoissonResidual();
    ReducedDiags::WriteToFile(step);
}

void ResidValue::WriteToFile (int /*step*/) const
{}