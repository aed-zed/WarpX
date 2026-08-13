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

#include "Fields.H"
#include "WarpX.H"
#include "Utils/Parser/ParserUtils.H"
#include "Utils/TextMsg.H"
#include "Utils/WarpXConst.H"

#include <AMReX_GpuAtomic.H>
#include <AMReX_MultiFab.H>
#include <AMReX_MFIter.H>
#include <AMReX_EBFabFactory.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_Parser.H>

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
            // Constrained rows carry no equation: restrict the scatter output
            // to FREE rows, exactly as the validated numpy transliteration
            // does (check_transpose_kernel.py, "constrained nodes carry no
            // equation"). Without this the scatter leaves values on Dirichlet
            // rows -- the diag term each scatter_from=1 source node writes
            // onto ITSELF, and spill into wall/covered neighbours -- and the
            // system A^T psi = rhs acquires equations no psi can satisfy:
            // rows where the scatter_from=0 operator is identically zero but
            // the RHS is not. That inconsistency is precisely the 2.004e-02
            // least-squares plateau the first version stalled at (identical
            // at max_iter 2000/20000). Q7 (check_adjoint_wiring_status.py)
            // always validated the FREE-NODE RESTRICTION of the RHS; this
            // makes the code compute the object Q7 validated.
            amrex::ParallelFor(vbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (dm(i,j,k) != 0) { ya(i,j,k) = amrex::Real(0.0); }
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

/** Build the adjoint RHS for the CHARGE functional
 * Q(phi) = eps0 * oint w(x,y,z) E.n dS, i.e. the linear functional WarpX
 * actually books charge with (WarpX::ComputeEBChargeWeighted / the 3D branch
 * of ChargeOnEB.cpp) -- as opposed to WarpXBuildAdjointRHS above, which is
 * the operator's own Dirichlet-row sum and couples to 1/h-scaled near-zero
 * rows that blow up on commensurate geometries (see file header).
 *
 * TWO PASSES, because the functional is stated on EDGES (Ex, Ey, Ez at the
 * cut-cell surface) but the row space of A^T is NODAL:
 *
 *   A. Cut-cell surface loop -> edge coefficients. Transcribed EXACTLY from
 *      ComputeEBChargeWeighted's 3D branch (same i_c/j_n/k_n selection, same
 *      dS*_fraction differences, same weighting parser evaluated at the
 *      boundary centroid): every cut cell adds eps0*w*dSx*(area-frac diff)
 *      onto the coefficient of Ex(i_c,j_n,k_n), and likewise for Ey, Ez.
 *      Different cut cells can hit the same edge (their "outside" cell/node
 *      coincide), hence the atomic add.
 *   B. Edge coefficients -> nodal rhs, through the EXACT transpose of
 *      AMReX's compGrad stencil (mlebndfdlap_grad_*_doit,
 *      AMReX_MLEBNodeFDLap_K.H) with E = -grad(phi) and a grounded EB
 *      (phieb = 0): each edge's coefficient f contributes f*E = -f*(the
 *      finite-difference gradient AMReX would compute on that edge), which
 *      is linear in the two endpoint nodes (or one, if the edge is cut) --
 *      scatter that linear form onto the nodal rhs.
 *
 * SumBoundary is required in BOTH passes, unlike the plain node->node
 * transpose in ApplyOp above. There, the SOURCE is a node, and nodes are
 * duplicated across box boundaries (AMReX extends each box's nodal valid
 * region by one at its own hi face -- Box::convert shifts bigend, so two
 * neighbouring boxes both carry the shared face node as valid); every box
 * that owns a target also independently reprocesses that shared source, so
 * a plain FillBoundary at the end is enough. Here the SOURCE of pass A is a
 * CELL (i_c) or the cut cell's own (i,j,k) -- unique to one box, not
 * duplicated -- so a cut cell near a box face can target an edge/node that
 * belongs to the neighbouring box, and that contribution is only ever
 * computed once, in the box that owns the source. Without folding it back
 * in, the box that owns the target would simply be missing it. This is
 * exactly the reason WarpXSumGuardCells exists for rho after particle
 * deposition, and the same idiom (SumBoundary(period)) is used here.
 *
 * `region` is evaluated as w(x,y,z) at the boundary centroid, exactly as the
 * binding already evaluates it for the operator-mode indicator -- it need
 * not be 0/1; ComputeEBChargeWeighted's weighting parser isn't either.
 */
void WarpXBuildAdjointRHSChargeFunctional (amrex::MultiFab& rhs,
                                           std::string const& region,
                                           amrex::iMultiFab const& dmsk,
                                           int lev)
{
    using ablastr::fields::Direction;
    using warpx::fields::FieldType;

    auto& warpx = WarpX::GetInstance();
    auto const& eb_fact = warpx.fieldEBFactory(lev);
    auto const& levset = eb_fact.getLevelSet();
    auto const& edge_cent = eb_fact.getEdgeCent();

    amrex::FabArray<amrex::EBCellFlagFab> const& eb_flag = eb_fact.getMultiEBCellFlagFab();
    amrex::MultiCutFab const& eb_bnd_cent = eb_fact.getBndryCent();
    amrex::MultiCutFab const& eb_bnd_normal = eb_fact.getBndryNormal();
    amrex::Array<const amrex::MultiCutFab*,AMREX_SPACEDIM> eb_area_fraction = eb_fact.getAreaFrac();

    const amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> dx = warpx.Geom(lev).CellSizeArray();
    amrex::Real const dSx = dx[1]*dx[2];
    amrex::Real const dSy = dx[2]*dx[0];
    amrex::Real const dSz = dx[0]*dx[1];
    const amrex::RealBox& real_box = warpx.Geom(lev).ProbDomain();
    const amrex::Periodicity& period = warpx.Geom(lev).periodicity();

    amrex::Parser rparser = utils::parser::makeParser(region, {"x","y","z"});
    auto fun_w = utils::parser::compileParser<3>(&rparser);

    // fx/fy/fz need only Efield_fp's geometry (BoxArray / DistributionMapping
    // / IndexType) to iterate cut cells with the same MFIter ChargeOnEB uses
    // and to hold one coefficient per edge; their VALUES are never read. The
    // RHS built here is a pure geometric functional, independent of whatever
    // field state happens to be live in Efield_fp when this is called.
    const amrex::MultiFab& Ex = *warpx.m_fields.get(FieldType::Efield_fp, Direction{0}, lev);
    const amrex::MultiFab& Ey = *warpx.m_fields.get(FieldType::Efield_fp, Direction{1}, lev);
    const amrex::MultiFab& Ez = *warpx.m_fields.get(FieldType::Efield_fp, Direction{2}, lev);

    amrex::MultiFab fx(Ex.boxArray(), Ex.DistributionMap(), 1, 1);
    amrex::MultiFab fy(Ey.boxArray(), Ey.DistributionMap(), 1, 1);
    amrex::MultiFab fz(Ez.boxArray(), Ez.DistributionMap(), 1, 1);
    fx.setVal(0.0);
    fy.setVal(0.0);
    fz.setVal(0.0);

    // A running (host/device-safe) count of cut edges skipped because their
    // transpose denominator (1 -+ 2*ec) was too close to zero to divide by --
    // an extreme cut fraction on that particular edge. Reported, not hidden.
    amrex::Gpu::Buffer<amrex::Long> skip_buf({amrex::Long(0)});
    amrex::Long* skip_ptr = skip_buf.data();

    // ---- Pass A: cut-cell surface loop -> edge coefficients ---------------
    // Transcribed from WarpX::ComputeEBChargeWeighted's 3D branch
    // (ChargeOnEB.cpp), with the "read E, reduce to a scalar" step replaced
    // by "scatter the coefficient of E onto fx/fy/fz". No tiling/OMP: this
    // loop only ever touches the cut-cell fraction of the domain, and unlike
    // ComputeDiags' plain scalar reduction, atomics land in a shared
    // MultiFab, which does not need (and should not fight) an OMP tiling
    // split for what is already a tiny amount of work.
    for (amrex::MFIter mfi(Ex); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.tilebox(amrex::IntVect::TheCellVector());

        // Skip boxes that do not intersect with the embedded boundary.
        const amrex::FabType fab_type = eb_flag[mfi].getType(box);
        if (fab_type == amrex::FabType::regular) { continue; }
        if (fab_type == amrex::FabType::covered) { continue; }

        auto const& eb_flag_arr = eb_flag.array(mfi);
        auto const& eb_bnd_normal_arr = eb_bnd_normal.array(mfi);
        auto const& eb_bnd_cent_arr = eb_bnd_cent.array(mfi);
        auto const& dSx_frac = eb_area_fraction[0]->array(mfi);
        auto const& dSy_frac = eb_area_fraction[1]->array(mfi);
        auto const& dSz_frac = eb_area_fraction[2]->array(mfi);
        auto const& fxa = fx.array(mfi);
        auto const& fya = fy.array(mfi);
        auto const& fza = fz.array(mfi);

        amrex::For(box,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                // Only cells that are partially covered contribute.
                if (eb_flag_arr(i,j,k).isRegular() || eb_flag_arr(i,j,k).isCovered()) { return; }

                // Nodal point outside the EB (eb_normal points to the EB interior).
                int const i_n = (eb_bnd_normal_arr(i,j,k,0) > 0) ? i : i+1;
                int const j_n = (eb_bnd_normal_arr(i,j,k,1) > 0) ? j : j+1;
                int const k_n = (eb_bnd_normal_arr(i,j,k,2) > 0) ? k : k+1;

                // Cell-centered point outside the EB.
                int i_c = i;
                if ((eb_bnd_normal_arr(i,j,k,0)>0) && (eb_bnd_cent_arr(i,j,k,0)<=0)) { i_c -= 1; }
                if ((eb_bnd_normal_arr(i,j,k,0)<0) && (eb_bnd_cent_arr(i,j,k,0)>=0)) { i_c += 1; }
                int j_c = j;
                if ((eb_bnd_normal_arr(i,j,k,1)>0) && (eb_bnd_cent_arr(i,j,k,1)<=0)) { j_c -= 1; }
                if ((eb_bnd_normal_arr(i,j,k,1)<0) && (eb_bnd_cent_arr(i,j,k,1)>=0)) { j_c += 1; }
                int k_c = k;
                if ((eb_bnd_normal_arr(i,j,k,2)>0) && (eb_bnd_cent_arr(i,j,k,2)<=0)) { k_c -= 1; }
                if ((eb_bnd_normal_arr(i,j,k,2)<0) && (eb_bnd_cent_arr(i,j,k,2)>=0)) { k_c += 1; }

                // Boundary-element centroid, same formula ComputeEBChargeWeighted
                // uses for its weighting-parser evaluation.
                const amrex::Real x = (i + amrex::Real(0.5) + eb_bnd_cent_arr(i,j,k,0))*dx[0] + real_box.lo(0);
                const amrex::Real y = (j + amrex::Real(0.5) + eb_bnd_cent_arr(i,j,k,1))*dx[1] + real_box.lo(1);
                const amrex::Real z = (k + amrex::Real(0.5) + eb_bnd_cent_arr(i,j,k,2))*dx[2] + real_box.lo(2);
                const amrex::Real w = fun_w(x, y, z);

                const amrex::Real cx = PhysConst::epsilon_0 * w * dSx
                    * (dSx_frac(i+1,j,k) - dSx_frac(i,j,k));
                const amrex::Real cy = PhysConst::epsilon_0 * w * dSy
                    * (dSy_frac(i,j+1,k) - dSy_frac(i,j,k));
                const amrex::Real cz = PhysConst::epsilon_0 * w * dSz
                    * (dSz_frac(i,j,k+1) - dSz_frac(i,j,k));

                amrex::Gpu::Atomic::AddNoRet(&fxa(i_c,j_n,k_n), cx);
                amrex::Gpu::Atomic::AddNoRet(&fya(i_n,j_c,k_n), cy);
                amrex::Gpu::Atomic::AddNoRet(&fza(i_n,j_n,k_c), cz);
            });
    }
    // Fold ghost-region contributions (a cut cell near a box face targeting
    // the neighbour's cell/node) back into the owning box's valid data.
    fx.SumBoundary(period);
    fy.SumBoundary(period);
    fz.SumBoundary(period);

    // ---- Pass B: edge coefficients -> nodal rhs (transpose of compGrad) ---
    // Per-edge rule (x shown; y, z analogous with the obvious index shifts).
    // Edge (I,J,K) joins node lo=(I,J,K) to node hi=(I+1,J,K); f = fx(I,J,K);
    // this edge's contribution to Q is f*Ex(I,J,K), and Ex = -px, so:
    //   both uncovered : px = dxi*(phi_hi-phi_lo)
    //                    -> rhs(hi) += -f*dxi ; rhs(lo) += +f*dxi
    //   lo covered     : px = dxi*(phi_hi-0)/(1-2*ecx(I,J,K))
    //                    -> rhs(hi) += -f*dxi/(1-2*ecx(I,J,K))
    //   hi covered     : px = dxi*(0-phi_lo)/(1+2*ecx(I,J,K))
    //                    -> rhs(lo) += +f*dxi/(1+2*ecx(I,J,K))
    //   both covered   : px undefined by the forward stencil -- no equation.
    // "covered" here means the EB level set, exactly as mlebndfdlap_grad_x_doit
    // tests it (dmsk>=0 free / dmsk<0 covered there; ls>=0 covered here) --
    // NOT the binding's dmsk, which also marks the outer walls. An edge that
    // straddles a grounded wall is "both uncovered" by the EB level set and
    // deposits normally; the wall-node contribution is discarded afterwards
    // by the dmsk!=0 zeroing below, same as every other constrained row.
    rhs.setVal(0.0);
    const amrex::Real dxi = amrex::Real(1.0) / dx[0];
    const amrex::Real dyi = amrex::Real(1.0) / dx[1];
    const amrex::Real dzi = amrex::Real(1.0) / dx[2];
    constexpr amrex::Real singular_tol = amrex::Real(1.0e-14);

    for (amrex::MFIter mfi(fx); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& fxa = fx.const_array(mfi);
        auto const& ls = levset.const_array(mfi);
        auto const& ecx = edge_cent[0]->const_array(mfi);
        auto const& rhsa = rhs.array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const amrex::Real f = fxa(i,j,k);
                if (f == amrex::Real(0.0)) { return; }
                const bool lo_cov = (ls(i,  j,k) >= amrex::Real(0.0));
                const bool hi_cov = (ls(i+1,j,k) >= amrex::Real(0.0));
                if (!lo_cov && !hi_cov) {
                    amrex::Gpu::Atomic::AddNoRet(&rhsa(i+1,j,k), -f*dxi);
                    amrex::Gpu::Atomic::AddNoRet(&rhsa(i,  j,k),  f*dxi);
                } else if (lo_cov && !hi_cov) {
                    const amrex::Real denom = amrex::Real(1.0) - amrex::Real(2.0)*ecx(i,j,k);
                    if (amrex::Math::abs(denom) < singular_tol) {
                        amrex::HostDevice::Atomic::Add(skip_ptr, amrex::Long(1));
                    } else {
                        amrex::Gpu::Atomic::AddNoRet(&rhsa(i+1,j,k), -f*dxi/denom);
                    }
                } else if (hi_cov && !lo_cov) {
                    const amrex::Real denom = amrex::Real(1.0) + amrex::Real(2.0)*ecx(i,j,k);
                    if (amrex::Math::abs(denom) < singular_tol) {
                        amrex::HostDevice::Atomic::Add(skip_ptr, amrex::Long(1));
                    } else {
                        amrex::Gpu::Atomic::AddNoRet(&rhsa(i,j,k), f*dxi/denom);
                    }
                }
                // both covered: no equation, nothing to add.
            });
    }

    for (amrex::MFIter mfi(fy); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& fya = fy.const_array(mfi);
        auto const& ls = levset.const_array(mfi);
        auto const& ecy = edge_cent[1]->const_array(mfi);
        auto const& rhsa = rhs.array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const amrex::Real f = fya(i,j,k);
                if (f == amrex::Real(0.0)) { return; }
                const bool lo_cov = (ls(i,j,  k) >= amrex::Real(0.0));
                const bool hi_cov = (ls(i,j+1,k) >= amrex::Real(0.0));
                if (!lo_cov && !hi_cov) {
                    amrex::Gpu::Atomic::AddNoRet(&rhsa(i,j+1,k), -f*dyi);
                    amrex::Gpu::Atomic::AddNoRet(&rhsa(i,j,  k),  f*dyi);
                } else if (lo_cov && !hi_cov) {
                    const amrex::Real denom = amrex::Real(1.0) - amrex::Real(2.0)*ecy(i,j,k);
                    if (amrex::Math::abs(denom) < singular_tol) {
                        amrex::HostDevice::Atomic::Add(skip_ptr, amrex::Long(1));
                    } else {
                        amrex::Gpu::Atomic::AddNoRet(&rhsa(i,j+1,k), -f*dyi/denom);
                    }
                } else if (hi_cov && !lo_cov) {
                    const amrex::Real denom = amrex::Real(1.0) + amrex::Real(2.0)*ecy(i,j,k);
                    if (amrex::Math::abs(denom) < singular_tol) {
                        amrex::HostDevice::Atomic::Add(skip_ptr, amrex::Long(1));
                    } else {
                        amrex::Gpu::Atomic::AddNoRet(&rhsa(i,j,k), f*dyi/denom);
                    }
                }
            });
    }

    for (amrex::MFIter mfi(fz); mfi.isValid(); ++mfi)
    {
        const amrex::Box& bx = mfi.validbox();
        auto const& fza = fz.const_array(mfi);
        auto const& ls = levset.const_array(mfi);
        auto const& ecz = edge_cent[2]->const_array(mfi);
        auto const& rhsa = rhs.array(mfi);
        amrex::ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                const amrex::Real f = fza(i,j,k);
                if (f == amrex::Real(0.0)) { return; }
                const bool lo_cov = (ls(i,j,k  ) >= amrex::Real(0.0));
                const bool hi_cov = (ls(i,j,k+1) >= amrex::Real(0.0));
                if (!lo_cov && !hi_cov) {
                    amrex::Gpu::Atomic::AddNoRet(&rhsa(i,j,k+1), -f*dzi);
                    amrex::Gpu::Atomic::AddNoRet(&rhsa(i,j,k  ),  f*dzi);
                } else if (lo_cov && !hi_cov) {
                    const amrex::Real denom = amrex::Real(1.0) - amrex::Real(2.0)*ecz(i,j,k);
                    if (amrex::Math::abs(denom) < singular_tol) {
                        amrex::HostDevice::Atomic::Add(skip_ptr, amrex::Long(1));
                    } else {
                        amrex::Gpu::Atomic::AddNoRet(&rhsa(i,j,k+1), -f*dzi/denom);
                    }
                } else if (hi_cov && !lo_cov) {
                    const amrex::Real denom = amrex::Real(1.0) + amrex::Real(2.0)*ecz(i,j,k);
                    if (amrex::Math::abs(denom) < singular_tol) {
                        amrex::HostDevice::Atomic::Add(skip_ptr, amrex::Long(1));
                    } else {
                        amrex::Gpu::Atomic::AddNoRet(&rhsa(i,j,k), f*dzi/denom);
                    }
                }
            });
    }

    // Same reasoning as pass A: a cut edge near a box face can scatter into
    // a node owned by the neighbouring box.
    rhs.SumBoundary(period);

    // Free-row restriction: constrained rows (EB-covered nodes AND the
    // grounded outer walls, per the binding's dmsk) carry no equation, same
    // rationale as the masked zeroing in ApplyOp.
    for (amrex::MFIter mfi(rhs); mfi.isValid(); ++mfi) {
        const amrex::Box& vbx = mfi.validbox();
        auto const& rhsa = rhs.array(mfi);
        auto const& dm = dmsk.const_array(mfi);
        amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                if (dm(i,j,k) != 0) { rhsa(i,j,k) = amrex::Real(0.0); }
            });
    }
    rhs.FillBoundary(period);

    skip_buf.copyToHost();
    amrex::Long skipped = *(skip_buf.hostData());
    amrex::ParallelDescriptor::ReduceLongSum(skipped);
    if (skipped > 0) {
        amrex::Print() << "WarpXBuildAdjointRHSChargeFunctional: skipped "
                       << skipped << " cut edge(s) with a near-singular "
                       << "transpose denominator (|1 -+ 2*ec| < "
                       << singular_tol << ").\n";
    }
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
        // Guard the beta division: on an inconsistent system the
        // normal-equation residual can underflow to exactly 0.0 while the
        // true residual is still large (observed in the numpy reproduction of
        // this loop). Without this break, beta = 0/0 = NaN on the following
        // iteration NaN-poisons psi, and `pMp == 0` never fires again because
        // NaN != 0. CG is fully converged on (A A^T) at this point; whatever
        // `rel` remains is inconsistency, and `converged` stays false.
        if (rr_new == amrex::Real(0.0) || rr == amrex::Real(0.0)) { break; }
        const amrex::Real beta = rr_new / rr;
        rr = rr_new;
        amrex::MultiFab::Xpay(p, beta, r, 0, 0, 1, 0);
    }
    return converged;
}

/** Rescale a solved charge-functional Psi (WarpXBuildAdjointRHSChargeFunctional
 * + WarpXSolveAdjointWeighting) into the weighting potential the booking
 * ledger needs.
 *
 * MLMG (AMReX_MLEBNodeFDLaplacian::scaleRHS, called once from MLMG::prepareForSolve
 * before the V-cycles) multiplies the user's physical RHS by a diagonal
 * S(i,j,k) = min(hmx,hpx,hmy,hpy,hmz,hpz) (the SAME per-node min(h) already
 * computed as `scale` in warpx_mlebndfdlap_adotx_{forward,transpose}_eb) at
 * every free node before solving; the operator A itself is untouched. So
 * WarpX's own solve is
 *
 *     phi = A^{-1} S rhs_phys,   rhs_phys(a) = -rho_a / eps0,
 *
 * and for the booking identity Q(phi) = c^T phi = -sum_a rho_a Psi[a] to hold
 * for EVERY rho, Psi = S A^{-T} c / eps0. A single node a carrying a probe
 * charge q over the nodal deposit volume dV = dx*dy*dz has rho_a = q/dV, so
 * the per-unit-charge weighting potential this correction ledger gathers is
 *
 *     Psi[a] = S(a) * (A^{-T} c)[a] / (eps0 * dV).
 *
 * `psi` on input is A^{-T} c restricted to the free nodes (WarpXSolveAdjoint-
 * Weighting's output); covered/wall nodes are already exactly 0.0 there (an
 * invariant of that CG loop: r, p are identically zero on dmsk!=0 rows from
 * the first iteration, so psi is never Saxpy'd away from its dmsk!=0 initial
 * value of 0). Multiplying those zero rows by S is harmless, so this applies
 * S unconditionally rather than re-testing dmsk.
 */
void WarpXFinalizeChargeFunctionalPsi (amrex::MultiFab& psi, int lev)
{
    auto& warpx = WarpX::GetInstance();
    auto const& eb_fact = warpx.fieldEBFactory(lev);
    auto const& edge_cent = eb_fact.getEdgeCent();

    const auto dx = warpx.Geom(lev).CellSizeArray();
    const amrex::Real dV = dx[0] * dx[1] * dx[2];
    const amrex::Real global_const = amrex::Real(1.0) / (PhysConst::epsilon_0 * dV);

    for (amrex::MFIter mfi(psi); mfi.isValid(); ++mfi) {
        const amrex::Box& vbx = mfi.validbox();
        auto const& pa = psi.array(mfi);
        auto const& ecx = edge_cent[0]->const_array(mfi);
        auto const& ecy = edge_cent[1]->const_array(mfi);
        auto const& ecz = edge_cent[2]->const_array(mfi);
        amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                using amrex::Real;
                // Same hp/hm recovery as mlebndfdlap_scale_rhs (AMReX_MLEBNodeFDLap_3D_K.H)
                // and the `scale` in warpx_mlebndfdlap_adotx_transpose_eb.
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
                pa(i,j,k) *= scale * global_const;
            });
    }
    psi.FillBoundary(warpx.Geom(lev).periodicity());
}

/* ===========================================================================
 * TEMPORARY DEBUG BINDINGS -- topic-ect-gauss-rebase-2026-08 diagnostic task.
 *
 * These two functions exist ONLY to let a Python driver test the hypothesis
 * that warpx_mlebndfdlap_adotx_forward_eb (AdjointWeightingPotential.H) does
 * not reproduce AMReX's real Fapply (MLEBNodeFDLaplacian::Fapply ->
 * mlebndfdlap_adotx_eb_doit), and that any mismatch localizes near the EB.
 *
 * The test compares, node by node, the transliterated forward apply against
 * MLMG's own real action reconstructed WITHOUT touching MLMG at all: MLMG
 * scales the user's rhs once, in MLMGT::prepareForSolve, via
 * MLEBNodeFDLaplacian::scaleRHS (AMReX_MLEBNodeFDLaplacian.cpp:~329), by
 * S(node) = min over the node's six edge heights -- the SAME per-node min(h)
 * already computed as `scale` above. So for a converged grounded solve,
 * A_real(phi) ~= S * rhs_phys = S * (-rho/eps0), up to the solver tolerance.
 *
 * Not for production use; do not call outside the diagnostic driver. Remove
 * before merging.
 * ===========================================================================
 */

/** Apply the transliterated forward EB Laplacian (ApplyOp, transpose=false)
 * to `x`, writing into `y`. Thin export of the file-local ApplyOp above so a
 * Python binding can drive it directly on arbitrary registered fields.
 */
void WarpXDebugForwardApply (amrex::MultiFab& y, amrex::MultiFab const& x,
                             amrex::iMultiFab const& dmsk, int lev)
{
    ApplyOp(y, x, dmsk, lev, /*transpose=*/false);
}

/** Write the per-node MLMG rhs-scale S(node) = min(hp,hm) over the node's six
 * edges into `scale_mf`, using the SAME hp/hm recovery as
 * mlebndfdlap_scale_rhs (AMReX_MLEBNodeFDLap_3D_K.H) and
 * WarpXFinalizeChargeFunctionalPsi above -- so S is computed by the same code
 * path being tested, not re-derived.
 */
void WarpXDebugComputeScale (amrex::MultiFab& scale_mf, int lev)
{
    auto& warpx = WarpX::GetInstance();
    auto const& eb_fact = warpx.fieldEBFactory(lev);
    auto const& edge_cent = eb_fact.getEdgeCent();

    scale_mf.setVal(0.0);
    for (amrex::MFIter mfi(scale_mf); mfi.isValid(); ++mfi) {
        const amrex::Box& vbx = mfi.validbox();
        auto const& sa = scale_mf.array(mfi);
        auto const& ecx = edge_cent[0]->const_array(mfi);
        auto const& ecy = edge_cent[1]->const_array(mfi);
        auto const& ecz = edge_cent[2]->const_array(mfi);
        amrex::ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                using amrex::Real;
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
                sa(i,j,k) = scale;
            });
    }
    scale_mf.FillBoundary(warpx.Geom(lev).periodicity());
}

/** Copy the exact levelset MultiFab MLMG/the transliteration both read
 * (EBFArrayBoxFactory::getLevelSet()) into a registered field, so a Python
 * driver can classify nodes as covered (levset>=0) / free (levset<0) using
 * the SAME array the kernels see, without a Python-side analytic proxy.
 */
void WarpXDebugDumpLevelSet (amrex::MultiFab& out_mf, int lev)
{
    auto& warpx = WarpX::GetInstance();
    auto const& eb_fact = warpx.fieldEBFactory(lev);
    auto const& levset_mf = eb_fact.getLevelSet();

    out_mf.setVal(0.0);
    const amrex::IntVect ng = amrex::elemwiseMin(out_mf.nGrowVect(), levset_mf.nGrowVect());
    out_mf.ParallelCopy(levset_mf, 0, 0, 1, ng, ng, warpx.Geom(lev).periodicity());
}
