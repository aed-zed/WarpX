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
     *  Write to file function for residual;  this differs from the base class
     * `ReducedDiags` in that it will not write to the file siince ComputeDiags
     *  calls the write to file function instead.
     *
     * @param[in] step current time step
     */
    void WriteToFile(int step) const final;

};
#endif // WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLENUMBER_H_