/* Copyright 2023 RemiLehe
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_SPECIES_UTILS_H_
#define WARPX_SPECIES_UTILS_H_


#ifdef AMREX_USE_EB
#include <AMReX_Array.H>
#include <AMReX_Array4.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_BoxList.H>
#include <AMReX_EB2.H>
#include <AMReX_EB_utils.H>
#include <AMReX_FabArray.H>
#include <AMReX_FabFactory.H>
#include <AMReX_GpuControl.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_Loop.H>
#include <AMReX_MFIter.H>
#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_SPACE.H>
#include <AMReX_Vector.H>
#endif


#include <AMReX_REAL.H>
#include "Initialization/InjectorDensity.H"
#include "Initialization/InjectorMomentum.H"
#include "Particles/SpeciesPhysicalProperties.H"

namespace SpeciesUtils {

    void StringParseAbortMessage(const std::string& var,
                                 const std::string& name);

    void extractSpeciesProperties ( std::string const& species_name,
        std::string const& injection_style, amrex::Real& charge, amrex::Real& mass,
        PhysicalSpecies& physical_species);

    void parseDensity (std::string const& species_name, std::string const& source_name,
        std::unique_ptr<InjectorDensity,InjectorDensityDeleter>& h_inj_rho,
        std::unique_ptr<amrex::Parser>& density_parser);

    void parseMomentum (std::string const& species_name, std::string const& source_name, const std::string& style,
        std::unique_ptr<InjectorMomentum,InjectorMomentumDeleter>& h_inj_mom,
        std::unique_ptr<amrex::Parser>& ux_parser,
        std::unique_ptr<amrex::Parser>& uy_parser,
        std::unique_ptr<amrex::Parser>& uz_parser,
        std::unique_ptr<amrex::Parser>& ux_th_parser,
        std::unique_ptr<amrex::Parser>& uy_th_parser,
        std::unique_ptr<amrex::Parser>& uz_th_parser,
        std::unique_ptr<TemperatureProperties>& h_mom_temp,
        std::unique_ptr<VelocityProperties>& h_mom_vel,
        int flux_normal_axis=0, int flux_direction=0);

#ifdef AMREX_USE_EB
    void parseMomentum (std::string const& species_name, std::string const& source_name, const std::string& style,
        std::unique_ptr<InjectorMomentum,InjectorMomentumDeleter>& h_inj_mom,
        amrex::EBFArrayBoxFactory* field_factory_ptr,
        const amrex::BoxArray* ba,
        const amrex::DistributionMapping* dm);
#endif


}

#endif // WARPX_SPECIES_UTILS_H_
