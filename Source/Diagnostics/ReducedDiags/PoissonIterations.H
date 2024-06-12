/* Copyright 2019-2020 Neil Zaim, Yinjian Zhao
 *
 * This file is part of WarpX.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_DIAGNOSTICS_REDUCEDDIAGS_POISSONITER_H_
#define WARPX_DIAGNOSTICS_REDUCEDDIAGS_POISSONITER_H_
#include "ReducedDiags.H"

#include <string>
/**
 *  This class mainly contains a function that gets the number of iterations
 *  acheived by the poisson equation before convergence was acheived for
 *  every timestep that the poisson equation was calculated.
 */
class PoissonIterations : public ReducedDiags
{
public:

    /**
     * The type of data this will collect is ints, not reals.
     */
    std::vector<int> m_data;

    /**
     * constructor
     * @param[in] rd_name reduced diags names
     */
    PoissonIterations(const std::string& rd_name);

    /**
     * This function adds the number of iterations of the poisson equation before convergence.
     *
     * @param[in] step current time step
     */
    void ComputeDiags(int step) final;

    /**
     *  Write to file function for poisson iterations;  this differs from the base class
     * `ReducedDiags` in that it will not write to the file siince ComputeDiags
     *  calls the write to file function instead.
     *
     * @param[in] step current time step
     */
    void WriteToFile(int step) const final;

private:
    /**
     * Should this step be written to the file?
     */
    bool write = true;

};
#endif // WARPX_DIAGNOSTICS_REDUCEDDIAGS_PARTICLENUMBER_H_