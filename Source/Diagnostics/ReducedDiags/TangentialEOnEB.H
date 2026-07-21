/* Copyright 2025 Marco Garten
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_TANGENTIALEONEB_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_TANGENTIALEONEB_H_

#include "ReducedDiags.H"

#include <string>

/**
 *  Measures the tangential electric field on the embedded boundary surface.
 *  For a perfect electric conductor E_tan should be zero.
 *
 *  At each cut cell the EB surface normal n is obtained from the EB factory.
 *  E is interpolated to the cell center from the surrounding Yee edges, then
 *  decomposed into normal and tangential parts:
 *    E_n   = (E . n) n
 *    E_tan = E - E_n
 *  The diagnostic reports max|E_tan| and area-weighted RMS|E_tan| over all
 *  cut cells on level 0.
 */
class TangentialEOnEB : public ReducedDiags
{
public:
    TangentialEOnEB (const std::string& rd_name);
    void ComputeDiags (int step) final;
};

#endif
