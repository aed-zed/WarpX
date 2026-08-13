/* Adjoint weighting-potential solve.
 *
 * Computes Psi_k = A^{-T} A_DF^T e_k, the weighting potential that makes the
 * grounded-electrode charge identity
 *
 *     Q_k(rho) = - sum_a  rho_a  Psi_k[a]
 *
 * EXACT rather than approximate. See
 * electrode-potential-maintenance/reviews/check_adjoint_identity.py (5/5):
 * the plain unit-voltage basis solves A psi = -A_FD (the Dirichlet coupling
 * column) while the identity needs A^T psi = A_DF^T (the charge-functional
 * row). Those differ unless A_FD = -A_DF^T AND A is symmetric, and WarpX's EB
 * Laplacian is non-symmetric at cut cells.
 *
 * WHY CG AND NOT MLMG
 * -------------------
 * MLMG solves A x = b through amrex::MLLinOp, whose apply() is the FORWARD
 * operator; there is no transposed-apply entry point and no way to hand it
 * one. Conjugate gradient needs only matrix-vector products, so the transpose
 * apply in AdjointWeightingPotential.H (validated to 1e-15 against a true
 * transpose, check_transpose_kernel.py 4/4) is enough.
 *
 * A^T is not symmetric, so plain CG on it is not guaranteed to converge (a
 * numpy prototype of exactly that loop stalled at 99% error). The normal
 * equations fix it: left-multiply A^T x = b by A to get
 *
 *     (A A^T) x = A b
 *
 * which is symmetric positive definite for nonsingular A, and whose solution
 * is x DIRECTLY -- no back-substitution. (Solving A A^T y = b and returning
 * A^T y is a different, wrong system; that mistake was caught in the same
 * prototype.) Cost is two applies per iteration -- one A^T, one A -- and a
 * squared condition number. Prototype on a non-symmetric operator: 58
 * iterations to a true residual of 1.1e-12.
 *
 * Convergence is judged on the TRUE residual |A^T x - b| / |b|, not on the
 * normal-equation residual, because the latter can be small while the former
 * is not. The caller MUST check the return value: a silently unconverged Psi
 * produces a wrong correction that looks entirely plausible.
 */

#include "AdjointWeightingPotential.H"

#include "WarpX.H"
#include "Utils/TextMsg.H"

#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_ParallelDescriptor.H>

#include <cmath>

namespace {

/** Apply the forward or transposed EB Laplacian.
 *
 * `transpose=false` reproduces WarpX's own MLEBNodeFDLaplacian row; `true`
 * applies its exact transpose (AdjointWeightingPotential.H, validated to
 * 1e-15 by check_transpose_kernel.py).
 */
void ApplyOp (amrex::MultiFab& y, amrex::MultiFab const& x,
              amrex::iMultiFab const& dmsk, int lev, bool transpose,
              int scatter_from = 0)
{
    auto& warpx = WarpX::GetInstance();
    auto const& eb_fact = warpx.fieldEBFactory(lev);
    auto const& edge_cent = eb_fact.getEdgeCent();
    auto const& levset = eb_fact.getLevelSet();

    const auto dx = warpx.Geom(lev).CellSizeArray();
    const amrex::Real bx = amrex::Real(1.0) / (dx[0] * dx[0]);
    const amrex::Real by = amrex::Real(1.0) / (dx[1] * dx[1]);
    const amrex::Real bz = amrex::Real(1.0) / (dx[2] * dx[2]);

    y.setVal(0.0);
    // Ghost handling is load-bearing here. FillBoundary only fills ghosts that
    // have a neighbour or a periodic image; on a non-periodic domain the
    // ghosts OUTSIDE the physical boundary are left untouched. Those walls are
    // grounded Dirichlet, so their value is zero -- but "untouched" is
    // uninitialised memory, not zero. Zeroing xg before the copy makes the
    // homogeneous outer boundary condition explicit. Skipping this produced a
    // Psi of order 1e13 with the CG residual still reporting convergence: the
    // iteration was faithfully solving a system whose right-hand side included
    // garbage.
    amrex::MultiFab xg(x.boxArray(), x.DistributionMap(), 1, 1);
    xg.setVal(0.0);
    amrex::MultiFab::Copy(xg, x, 0, 0, 1, 0);
    xg.FillBoundary(warpx.Geom(lev).periodicity());

    for (amrex::MFIter mfi(y); mfi.isValid(); ++mfi) {
        const amrex::Box& vbx = mfi.validbox();
        auto const& ya = y.array(mfi);
        auto const& xa = xg.const_array(mfi);
        auto const& ls = levset.const_array(mfi);
        auto const& dm = dmsk.const_array(mfi);
        auto const& ecx = edge_cent[0]->const_array(mfi);
        auto const& ecy = edge_cent[1]->const_array(mfi);
        auto const& ecz = edge_cent[2]->const_array(mfi);

        if (transpose) {
            amrex::ParallelFor(vbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    warpx_mlebndfdlap_adotx_transpose_eb(
                        i, j, k, ya, xa, ls, dm, ecx, ecy, ecz, bx, by, bz,
                        scatter_from);
                });
        } else {
            amrex::ParallelFor(vbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    warpx_mlebndfdlap_adotx_forward_eb(
                        i, j, k, ya, xa, ls, dm, ecx, ecy, ecz, bx, by, bz);
                });
        }
    }
    y.FillBoundary(warpx.Geom(lev).periodicity());
}

amrex::Real Dot (amrex::MultiFab const& a, amrex::MultiFab const& b)
{
    amrex::Real s = amrex::MultiFab::Dot(a, 0, b, 0, 1, 0);
    return s;
}

} // namespace

/** Build the adjoint RHS for electrode k: (A_DF)^T e_k on the free nodes.
 *
 * `indicator` must be 1 on electrode k's Dirichlet nodes and 0 elsewhere.
 * Applying A^T with scatter_from=1 lets exactly those rows scatter into the
 * free nodes, which is the column sum that the identity requires.
 */
void WarpXBuildAdjointRHS (amrex::MultiFab& rhs,
                           amrex::MultiFab const& indicator,
                           amrex::iMultiFab const& dmsk, int lev)
{
    ApplyOp(rhs, indicator, dmsk, lev, /*transpose=*/true, /*scatter_from=*/1);
}

/** Solve A^T Psi = rhs by CG on the normal equations (A A^T) Psi = A rhs.
 *
 * Validated by check_adjoint_identity.py Q6, which transliterates THIS loop
 * line by line and runs it: 13 iterations on a well-conditioned SPD system,
 * 59 on a non-symmetric one, true residual < 1e-12 in both.
 *
 * \return true if the relative residual reached `tol` within `max_iter`.
 *         The caller MUST check this: an unconverged Psi yields a wrong
 *         correction that looks entirely reasonable.
 */
bool WarpXSolveAdjointWeighting (amrex::MultiFab& psi,
                                 amrex::MultiFab const& rhs,
                                 amrex::iMultiFab const& dmsk,
                                 int lev, amrex::Real tol, int max_iter,
                                 amrex::Real* final_res)
{
    const amrex::BoxArray& ba = psi.boxArray();
    const amrex::DistributionMapping& dm = psi.DistributionMap();
    const int ng = psi.nGrow();

    amrex::MultiFab r(ba, dm, 1, ng), p(ba, dm, 1, ng);
    amrex::MultiFab t(ba, dm, 1, ng), Mp(ba, dm, 1, ng), nrhs(ba, dm, 1, ng);

    // Normal equations: (A A^T) psi = A rhs.
    ApplyOp(nrhs, rhs, dmsk, lev, /*transpose=*/false);

    psi.setVal(0.0);
    amrex::MultiFab::Copy(r, nrhs, 0, 0, 1, 0);      // r = A rhs - (A A^T) 0
    amrex::MultiFab::Copy(p, r, 0, 0, 1, 0);
    amrex::Real rr = Dot(r, r);

    const amrex::Real b_norm = std::sqrt(Dot(rhs, rhs));
    if (b_norm == amrex::Real(0.0)) {
        if (final_res) { *final_res = 0.0; }
        return true;
    }

    bool converged = false;
    for (int it = 0; it < max_iter; ++it) {
        ApplyOp(t, p, dmsk, lev, /*transpose=*/true);     // t  = A^T p
        ApplyOp(Mp, t, dmsk, lev, /*transpose=*/false);   // Mp = A A^T p
        const amrex::Real pMp = Dot(p, Mp);
        if (pMp == amrex::Real(0.0)) { break; }
        const amrex::Real alpha = rr / pMp;

        amrex::MultiFab::Saxpy(psi, alpha, p, 0, 0, 1, 0);
        amrex::MultiFab::Saxpy(r, -alpha, Mp, 0, 0, 1, 0);

        // Convergence on the TRUE residual |A^T psi - rhs| / |rhs|.
        ApplyOp(t, psi, dmsk, lev, /*transpose=*/true);
        amrex::MultiFab::Subtract(t, rhs, 0, 0, 1, 0);
        const amrex::Real rel = std::sqrt(Dot(t, t)) / b_norm;
        if (final_res) { *final_res = rel; }
        if (rel < tol) { converged = true; break; }

        const amrex::Real rr_new = Dot(r, r);
        const amrex::Real beta = rr_new / rr;
        rr = rr_new;
        amrex::MultiFab::Xpay(p, beta, r, 0, 0, 1, 0);
    }
    return converged;
}
