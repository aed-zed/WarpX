/* Copyright 2026 The WarpX Community
 *
 * Adjoint of the MLEBNodeFDLaplacian apply, for electrode weighting potentials.
 *
 * License: BSD-3-Clause-LBNL
 */
#ifndef WARPX_ADJOINT_WEIGHTING_POTENTIAL_H_
#define WARPX_ADJOINT_WEIGHTING_POTENTIAL_H_

#include <AMReX_Array4.H>
#include <AMReX_GpuAtomic.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Algorithm.H>
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
    amrex::Real bx, amrex::Real by, amrex::Real bz,
    int scatter_from = 0) noexcept
{
    using amrex::Real;

    // scatter_from = 0 : only FREE rows scatter -- this is A^T restricted to
    //                    the free-free block, the operator the CG solve needs.
    // scatter_from = 1 : only DIRICHLET rows scatter -- this is A^T applied to
    //                    an electrode indicator, i.e. the RHS  (A_DF)^T e_k
    //                    that the adjoint weighting potential solves against.
    //                    Verified against a dense assembly: the free-node
    //                    restriction of A^T 1_k equals the column sum of A
    //                    over electrode k's Dirichlet rows, exactly
    //                    (check_adjoint_identity.py Q7).
    const bool is_d = (dmsk(i,j,k) != 0);
    if (scatter_from == 0 && is_d)  { return; }
    if (scatter_from == 1 && !is_d) { return; }

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

/** Forward apply: y = A x, reproducing AMReX's mlebndfdlap_adotx_eb row.
 *
 * Needed because the normal-equation solve for the adjoint weighting potential
 * applies BOTH A^T and A, and AMReX's own forward apply is not reachable
 * outside its MLLinOp. Keeping the two in one header makes it obvious that
 * they must recover the same hp/hm/scale factors -- if the forward apply here
 * ever diverges from AMReX's, the transpose validation silently becomes a
 * self-consistency check rather than a check against the real operator.
 * check_transpose_kernel.py compares this pair against a dense transpose.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void warpx_mlebndfdlap_adotx_forward_eb (
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

    if (dmsk(i,j,k)) { y(i,j,k) = Real(0.0); return; }

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
    scale = amrex::min(scale, Real(1.0));

    // NOTE the scale placement: AMReX applies min(h) ONCE to the summed
    // result (AMReX_MLEBNodeFDLap_3D_K.H: `y(i,j,k) = out*scale`), not
    // per-direction inside the sum. In 1-D the two coincide; in 3-D they do
    // not, and getting it wrong reintroduces exactly the row-scaling error
    // this whole adjoint construction exists to handle.
    // A covered neighbour's value is Dirichlet data (xeb in AMReX's own
    // kernel), homogeneous (0) for the adjoint problem -- it must NOT be
    // read from x. The first version of this kernel read x at covered
    // neighbours unconditionally, diverging from the numpy pair that
    // check_transpose_kernel.py H1 validated (that pair has this branch):
    // the compiled forward was then not the exact transpose of the compiled
    // scatter. Harmless only while every input is identically zero on
    // covered nodes -- which the unmasked RHS was not.
    const Real x0 = x(i,j,k);
    const Real xpx = (levset(i+1,j,k) < Real(0.0)) ? x(i+1,j,k) : Real(0.0);
    const Real xmx = (levset(i-1,j,k) < Real(0.0)) ? x(i-1,j,k) : Real(0.0);
    const Real xpy = (levset(i,j+1,k) < Real(0.0)) ? x(i,j+1,k) : Real(0.0);
    const Real xmy = (levset(i,j-1,k) < Real(0.0)) ? x(i,j-1,k) : Real(0.0);
    const Real xpz = (levset(i,j,k+1) < Real(0.0)) ? x(i,j,k+1) : Real(0.0);
    const Real xmz = (levset(i,j,k-1) < Real(0.0)) ? x(i,j,k-1) : Real(0.0);
    Real out = Real(0.0);
    out += bx * Real(2.0) / (hpx + hmx)
         * ((xpx - x0) / hpx + (xmx - x0) / hmx);
    out += by * Real(2.0) / (hpy + hmy)
         * ((xpy - x0) / hpy + (xmy - x0) / hmy);
    out += bz * Real(2.0) / (hpz + hmz)
         * ((xpz - x0) / hpz + (xmz - x0) / hmz);
    y(i,j,k) = out * scale;
}

#if (AMREX_SPACEDIM == 2)
/** Exact row-wise transpose of AMReX's metric-aware RZ EB nodal Laplacian.
 *
 * Unlike the Cartesian stencil above, the radial coefficients contain the
 * cylindrical metric r+/-dr/2 and the axis row uses the regularity limit
 * 4*(phi_1-phi_0)/dr^2.  Writing the transpose as a scatter of each forward
 * row keeps those factors on the row where AMReX evaluates them.
 */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void warpx_mlebndfdlap_adotx_rz_transpose_eb (
    int i, int j, int k,
    amrex::Array4<amrex::Real> const& y,
    amrex::Array4<amrex::Real const> const& x,
    amrex::Array4<amrex::Real const> const& levset,
    amrex::Array4<int const> const& dmsk,
    amrex::Array4<amrex::Real const> const& ecr,
    amrex::Array4<amrex::Real const> const& ecz,
    amrex::Real dr, amrex::Real dz, amrex::Real rlo,
    int scatter_from = 0) noexcept
{
    using amrex::Real;

    const bool is_d = (dmsk(i,j,k) != 0);
    if (scatter_from == 0 && is_d) { return; }
    if (scatter_from == 1 && !is_d) { return; }

    const Real r = rlo + Real(i)*dr;
    const Real hpr = (ecr(i  ,j,k) == Real(1.0)) ? Real(1.0)
                   : Real(1.0) + Real(2.0)*ecr(i  ,j,k);
    const Real hmr = (ecr(i-1,j,k) == Real(1.0)) ? Real(1.0)
                   : Real(1.0) - Real(2.0)*ecr(i-1,j,k);
    const Real hpz = (ecz(i,j  ,k) == Real(1.0)) ? Real(1.0)
                   : Real(1.0) + Real(2.0)*ecz(i,j  ,k);
    const Real hmz = (ecz(i,j-1,k) == Real(1.0)) ? Real(1.0)
                   : Real(1.0) - Real(2.0)*ecz(i,j-1,k);

    Real scale;
    Real diag = Real(0.0);
    const Real x0 = x(i,j,k);
    Real crp = Real(0.0);
    Real crm = Real(0.0);

    if (r == Real(0.0)) {
        if (levset(i+1,j,k) < Real(0.0)) {
            const Real c = Real(4.0)/(dr*dr);
            crp = c;
            diag -= c;
            scale = Real(1.0);
        } else {
            diag -= Real(4.0)/(dr*dr*hpr*hpr);
            scale = hpr;
        }
    } else {
        const Real fac = Real(2.0)/((hpr+hmr)*r*dr*dr);
        if (levset(i+1,j,k) < Real(0.0)) {
            const Real c = fac*(r + Real(0.5)*dr);
            crp = c;
            diag -= c;
        } else {
            diag -= fac*(r + Real(0.5)*hpr*dr)/hpr;
        }
        if (levset(i-1,j,k) < Real(0.0)) {
            const Real c = fac*(r - Real(0.5)*dr);
            crm = c;
            diag -= c;
        } else {
            diag -= fac*(r - Real(0.5)*hmr*dr)/hmr;
        }
        scale = amrex::min(hmr, hpr);
    }

    const Real facz = Real(2.0)/((hpz+hmz)*dz*dz);
    const bool zp_free = (levset(i,j+1,k) < Real(0.0));
    const bool zm_free = (levset(i,j-1,k) < Real(0.0));
    diag -= facz * ((levset(i,j+1,k) < Real(0.0)) ? Real(1.0) : Real(1.0)/hpz);
    diag -= facz * ((levset(i,j-1,k) < Real(0.0)) ? Real(1.0) : Real(1.0)/hmz);

    scale = amrex::min(scale, hmz, hpz);
    if (crp != Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i+1,j,k), x0*crp*scale);
    }
    if (crm != Real(0.0)) {
        amrex::Gpu::Atomic::AddNoRet(&y(i-1,j,k), x0*crm*scale);
    }
    if (zp_free) {
        amrex::Gpu::Atomic::AddNoRet(&y(i,j+1,k), x0*facz*scale);
    }
    if (zm_free) {
        amrex::Gpu::Atomic::AddNoRet(&y(i,j-1,k), x0*facz*scale);
    }
    amrex::Gpu::Atomic::AddNoRet(&y(i,j,k), x0*diag*scale);
}

/** Forward RZ row, matching mlebndfdlap_adotx_rz_eb_doit (alpha=0). */
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void warpx_mlebndfdlap_adotx_rz_forward_eb (
    int i, int j, int k,
    amrex::Array4<amrex::Real> const& y,
    amrex::Array4<amrex::Real const> const& x,
    amrex::Array4<amrex::Real const> const& levset,
    amrex::Array4<int const> const& dmsk,
    amrex::Array4<amrex::Real const> const& ecr,
    amrex::Array4<amrex::Real const> const& ecz,
    amrex::Real dr, amrex::Real dz, amrex::Real rlo) noexcept
{
    using amrex::Real;
    if (dmsk(i,j,k)) { y(i,j,k) = Real(0.0); return; }

    const Real r = rlo + Real(i)*dr;
    const Real hpr = (ecr(i  ,j,k) == Real(1.0)) ? Real(1.0)
                   : Real(1.0) + Real(2.0)*ecr(i  ,j,k);
    const Real hmr = (ecr(i-1,j,k) == Real(1.0)) ? Real(1.0)
                   : Real(1.0) - Real(2.0)*ecr(i-1,j,k);
    const Real hpz = (ecz(i,j  ,k) == Real(1.0)) ? Real(1.0)
                   : Real(1.0) + Real(2.0)*ecz(i,j  ,k);
    const Real hmz = (ecz(i,j-1,k) == Real(1.0)) ? Real(1.0)
                   : Real(1.0) - Real(2.0)*ecz(i,j-1,k);
    const Real x0 = x(i,j,k);
    Real out;
    Real scale;

    if (r == Real(0.0)) {
        if (levset(i+1,j,k) < Real(0.0)) {
            out = Real(4.0)*(x(i+1,j,k)-x0)/(dr*dr);
            scale = Real(1.0);
        } else {
            out = -Real(4.0)*x0/(dr*dr*hpr*hpr);
            scale = hpr;
        }
    } else {
        Real tmp;
        if (levset(i+1,j,k) < Real(0.0)) {
            tmp = (x(i+1,j,k)-x0)*(r + Real(0.5)*dr);
        } else {
            tmp = -x0/hpr*(r + Real(0.5)*hpr*dr);
        }
        if (levset(i-1,j,k) < Real(0.0)) {
            tmp += (x(i-1,j,k)-x0)*(r - Real(0.5)*dr);
        } else {
            tmp += -x0/hmr*(r - Real(0.5)*hmr*dr);
        }
        out = tmp*Real(2.0)/((hpr+hmr)*r*dr*dr);
        scale = amrex::min(hmr,hpr);
    }

    Real tmpz = (levset(i,j+1,k) < Real(0.0))
        ? x(i,j+1,k)-x0 : -x0/hpz;
    tmpz += (levset(i,j-1,k) < Real(0.0))
        ? x(i,j-1,k)-x0 : -x0/hmz;
    out += tmpz*Real(2.0)/((hpz+hmz)*dz*dz);
    scale = amrex::min(scale,hmz,hpz);
    y(i,j,k) = out*scale;
}
#endif

#endif // WARPX_ADJOINT_WEIGHTING_POTENTIAL_H_
