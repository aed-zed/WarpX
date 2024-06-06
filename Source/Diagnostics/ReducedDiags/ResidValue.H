/* Copyright 2019-2020 Neil Zaim, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_RESIDVALUE_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_RESIDVALUE_H_

#include "ReducedDiags.H"

#include <string>

/**
 *  This class mainly contains a function that get the residual value of the poisson equation calculation.
 */
class ResidValue : public ReducedDiags
{
public:

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    ResidValue(const std::string& rd_name);

    /**
     * This function adds the residual value after the Poisson equation is calcualted.
     *
     * @param[in] step current time step
     */
    void ComputeDiags(int step) final;

    /**
     * write to file function for costs;  this differs from the base class
     * `ReducedDiags` in that it will write to the file based on whether the
     *  Poisson equation was calculated or not, not on the value of intervals
     *
     * @param[in] step current time step
     */
    void WriteToFile(int step) const final;

private:
    bool skipWrite = false;

};

#endif // WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLENUMBER_H_