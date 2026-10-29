#include "MultiDiagnostics.H"

#include "Diagnostics/BTDiagnostics.H"
#include "Diagnostics/FullDiagnostics.H"
#include "Diagnostics/BoundaryScrapingDiagnostics.H"
#include "Utils/TextMsg.H"
#include <ablastr/warn_manager/WarnManager.H>
#include <AMReX_ParmParse.H>
#include <AMReX.H>
#include <AMReX_REAL.H>

using namespace amrex;

MultiDiagnostics::MultiDiagnostics ()
{
    ReadParameters();
    /** Resize alldiags and initialize each element to a pointer to a
     * diagnostics. Calls the corresponding diagnostics constructor.
     */
    alldiags.resize( ndiags );
    for (int i=0; i<ndiags; i++){
        if ( diags_types[i] == DiagTypes::Full || diags_types[i] == DiagTypes::TimeAveraged ){
            alldiags[i] = std::make_unique<FullDiagnostics>(i, diags_names[i], diags_types[i]);
        } else if ( diags_types[i] == DiagTypes::BackTransformed ){
            alldiags[i] = std::make_unique<BTDiagnostics>(i, diags_names[i], diags_types[i]);
        } else if ( diags_types[i] == DiagTypes::BoundaryScraping ){
            alldiags[i] = std::make_unique<BoundaryScrapingDiagnostics>(i, diags_names[i], diags_types[i]);
        } else {
            WARPX_ABORT_WITH_MESSAGE("Unknown diagnostic type");
        }
    }
}

void
MultiDiagnostics::InitData (const MultiParticleContainer& mpc)
{
    for( auto& diag : alldiags ){
        diag->InitData(mpc);
    }
}

void
MultiDiagnostics::InitializeFieldFunctors ( int lev )
{
    for( auto& diag : alldiags ){
        // Initialize functors to store pointers to fields.
        diag->InitializeFieldFunctors( lev );
    }
}

void
MultiDiagnostics::ReadParameters ()
{
    const ParmParse pp_diagnostics("diagnostics");

    int enable_diags = 1;
    pp_diagnostics.query("enable", enable_diags);
    if (enable_diags == 1) {
        pp_diagnostics.queryarr("diags_names", diags_names);
        ndiags = static_cast<int>(diags_names.size());
    }

    diags_types.resize( ndiags );
    for (int i=0; i<ndiags; i++){
        const ParmParse pp_diag_name(diags_names[i]);
        std::string diag_type_str;
        pp_diag_name.get("diag_type", diag_type_str);
        WARPX_ALWAYS_ASSERT_WITH_MESSAGE(
            diag_type_str == "Full" || diag_type_str == "TimeAveraged" || diag_type_str == "BackTransformed" || diag_type_str == "BoundaryScraping",
            "<diag>.diag_type must be Full, TimeAveraged, BackTransformed or BoundaryScraping");
        if (diag_type_str == "Full") { diags_types[i] = DiagTypes::Full; }
        if (diag_type_str == "TimeAveraged") { diags_types[i] = DiagTypes::TimeAveraged; }
        if (diag_type_str == "BackTransformed") { diags_types[i] = DiagTypes::BackTransformed; }
        if (diag_type_str == "BoundaryScraping") { diags_types[i] = DiagTypes::BoundaryScraping; }
    }
}

bool
MultiDiagnostics::DoComputeAndPack (int step, bool force_flush)
{
    bool result = false;
    for( auto& diag : alldiags ){
        result = result || diag->DoComputeAndPack(step, force_flush);
    }
    return result;
}

void
MultiDiagnostics::FilterComputePackFlush (int step, bool force_flush, bool BackTransform)
{
    // Checkpoints are written last, so that they contain the data of this step
    // that other diagnostics need to continue after a restart (e.g., time averages)
    for (const bool checkpoints : {false, true}) {
        for (int i = 0; i < ndiags; ++i) {
            const bool is_btd = (diags_types[i] == DiagTypes::BackTransformed);
            if (alldiags[i]->IsCheckpoint() == checkpoints && is_btd == BackTransform) {
                alldiags[i]->FilterComputePackFlush (step, force_flush);
            }
        }
    }
}

void
MultiDiagnostics::FilterComputePackFlushLastTimestep (int step)
{
    // Checkpoints are written last (see FilterComputePackFlush)
    for (const bool checkpoints : {false, true}) {
        for (auto& diag : alldiags){
            if (diag->IsCheckpoint() == checkpoints && diag->DoDumpLastTimestep()){
                constexpr bool force_flush = true;
                diag->FilterComputePackFlush (step, force_flush);
            }
        }
    }
}

void
MultiDiagnostics::NewIteration ()
{
    for( auto& diag : alldiags ){
        diag->NewIteration();
    }
}
