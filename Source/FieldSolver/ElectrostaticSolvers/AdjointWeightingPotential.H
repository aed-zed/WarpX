/* Copyright 2026 The WarpX Community
 *
 * Adjoint of the MLEBNodeFDLaplacian apply, for electrode weighting potentials.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ADJOINT_WEIGHTING_POTENTIAL_H_
#define WARPX_ADJOINT_WEIGHTING_POTENTIAL_H_

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

/** \brief Apply the TRANSPOSE of the MLEBNodeFDLaplacian operator.
 *
 * WHY THIS EXISTS
 * ---------------
 * The electrode weighting potential psi_k satisfies the discrete charge
 * identity  Q_k = (C V)_k - sum_nodes rho * psi_k  only when it is built from
 * the ADJOINT of the operator the solver uses:
 *
 *     A^T psi_k = -A_DF^T e_k
 *
 * On a symmetric operator A^T = A and the familiar "solve the same Laplace
 * problem with electrode k at 1 V" recipe is correct. WarpX's EB Laplacian is
 * NOT symmetric at partially cut cells: mlebndfdlap_adotx_eb_doit applies a
 * per-direction factor 2/(hp+hm) INSIDE the row sum, and those factors differ
 * between directions on the same row. Using the plain solve there gives a
 * weighting potential wrong by tens of percent per cut cell.
 *
 * HOW IT WORKS
 * ------------
 * The operator factors as
 *
 *     A = sum_d D_d N_d,
 *     D_d = diag(scale_r * 2/(hm_d + hp_d))     (diagonal)
 *     N_d = the two-point flux stencil in direction d   (SYMMETRIC)
 *
 * so, since each N_d is symmetric and each D_d is diagonal,
 *
 *     A^T v = sum_d N_d (D_d v).
 *
 * That is: the same stencil, with the per-direction scaling applied BEFORE
 * the stencil instead of after it. This routine is therefore a transcription
 * of mlebndfdlap_adotx_eb_doit with the ordering changed -- no new
 * discretisation and no matrix assembly.
 *
 * The scatter form below is the natural way to write "apply the scaling, then
 * the symmetric stencil": each node distributes its scaled value to its
 * neighbours with the same coefficients the forward apply would gather with.
 *
 * Verified against a dense transpose on the AMReX stencil to 5e-15
 * (electrode-potential-maintenance/reviews/check_adjoint_route.py, F1).
 *
 * \param i,j,k     node index
 * \param y         output (accumulated into; caller must zero first)
 * \param x         input vector
 * \param levset    level set (negative = regular/uncovered)
 * \param dmsk      Dirichlet mask (nonzero = constrained node, skipped)
 * \param ecx,ecy,ecz  edge centroids, as used by the forward apply
 * \param bx,by,bz  1/dx^2 per direction
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void warpx_mlebndfdlap_adotx_transpose_eb (
    int i, int j, int k,
    amrex::Array4<amrex::Real> const& y,
    amrex::Array4<amrex::Real const> const& x,
    amrex::Array4<amrex::Real const> const& levset,
    amrex::Array4<int const> const& dmsk,
    amrex::Array4<amrex::Real const> const& ecx,
    amrex::Array4<amrex::Real const> const& ecy,
    amrex::Array4<amrex::Real const> const& ecz,
    amrex::Real bx, amrex::Real by, amrex::Real bz) noexcept
{
    using amrex::Real;

    if (dmsk(i,j,k)) { return; }

    // ---- pass 1: recover the same hp/hm/scale this row would use ----------
    const Real hpx = (ecx(i  ,j,k) == Real(1.0)) ? Real(1.0)
                   : (Real(1.0) + Real(2.0)*ecx(i  ,j,k));
    const Real hmx = (ecx(i-1,j,k) == Real(1.0)) ? Real(1.0)
                   : (Real(1.0) - Real(2.0)*ecx(i-1,j,k));
    const Real hpy = (ecy(i,j  ,k) == Real(1.0)) ? Real(1.0)
                   : (Real(1.0) + Real(2.0)*ecy(i,j  ,k));
    const Real hmy = (ecy(i,j-1,k) == Real(1.0)) ? Real(1.0)
                   : (Real(1.0) - Real(2.0)*ecy(i,j-1,k));
    const Real hpz = (ecz(i,j,k  ) == Real(1.0)) ? Real(1.0)
                   : (Real(1.0) + Real(2.0)*ecz(i,j,k  ));
    const Real hmz = (ecz(i,j,k-1) == Real(1.0)) ? Real(1.0)
                   : (Real(1.0) - Real(2.0)*ecz(i,j,k-1));

    Real scale = amrex::min(hmx, hpx);
    scale = amrex::min(scale, hmy, hpy);
    scale = amrex::min(scale, hmz, hpz);

    // ---- D_d v : the per-direction scaling, applied to THIS node's value --
    const Real vx = x(i,j,k) * scale * bx * Real(2.0) / (hpx + hmx);
    const Real vy = x(i,j,k) * scale * by * Real(2.0) / (hpy + hmy);
    const Real vz = x(i,j,k) * scale * bz * Real(2.0) / (hpz + hmz);

    // ---- N_d (D_d v) : scatter with the SYMMETRIC two-point stencil -------
    // A covered neighbour contributes only to the diagonal here: its value is
    // Dirichlet data, which belongs in the right-hand side, not the operator.
    // This mirrors the forward apply, where xeb(...) plays that role.
    Real diag = Real(0.0);

    if (levset(i+1,j,k) < Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i+1,j,k), vx);
        diag -= vx;
    } else {
        diag -= vx / hpx;
    }
    if (levset(i-1,j,k) < Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i-1,j,k), vx);
        diag -= vx;
    } else {
        diag -= vx / hmx;
    }

    if (levset(i,j+1,k) < Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i,j+1,k), vy);
        diag -= vy;
    } else {
        diag -= vy / hpy;
    }
    if (levset(i,j-1,k) < Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i,j-1,k), vy);
        diag -= vy;
    } else {
        diag -= vy / hmy;
    }

    if (levset(i,j,k+1) < Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i,j,k+1), vz);
        diag -= vz;
    } else {
        diag -= vz / hpz;
    }
    if (levset(i,j,k-1) < Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i,j,k-1), vz);
        diag -= vz;
    } else {
        diag -= vz / hmz;
    }

    amrex::Gpu::Atomic::AddNoRet(&y(i,j,k), diag);
}

#endif // WARPX_ADJOINT_WEIGHTING_POTENTIAL_H_
