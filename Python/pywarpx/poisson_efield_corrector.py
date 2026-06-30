"""
Poisson E-field Corrector for EM Solver with Embedded Boundaries

Periodically replaces the electric field with the result of a Poisson solve
that includes the plasma charge density and enforces correct boundary conditions
on embedded boundaries and domain boundaries.

This corrects the electrode potential drift that occurs in electromagnetic
(FDTD) simulations because the Maxwell solver has no mechanism to enforce
a fixed potential difference between electrodes.

Usage::

    from pywarpx.poisson_efield_corrector import PoissonEfieldCorrector
    from pywarpx.callbacks import installafterEsolve, installafterInitEsolve

    corrector = PoissonEfieldCorrector(
        sim=sim,
        correction_interval=10,
        potential_expression="-1.e5*(x*x+y*y<3.e-2**2)",
        enable_diagnostics=True,
        diag_name="diag1",
    )
    installafterInitEsolve(corrector.setup_after_init)
    installafterEsolve(corrector.correct_field)

The correction is installed on ``afterEsolve`` (immediately after the Maxwell
field solve, inside the step) rather than ``afterstep``. Both produce the same
field for the next particle gather, but ``afterEsolve`` keeps the corrected
field as the unambiguous final E^{n+1} of the step, before any diagnostics or
guard-cell/aux bookkeeping.
"""


def _get_libwarpx():
    from pywarpx._libwarpx import libwarpx  # noqa: PLC0415
    return libwarpx


class PoissonEfieldCorrector:
    """Correct E-field drift in EM+EB simulations via periodic Poisson solves.

    Parameters
    ----------
    sim : picmi.Simulation
        The PICMI simulation object (must be initialized).
    correction_interval : int
        Apply Poisson correction every this many steps.
    potential_expression : str
        EB potential expression (same syntax as picmi.EmbeddedBoundary potential).
    enable_diagnostics : bool, optional
        If True, register the Efield_correction diagnostic MultiFabs and print Delta-V.
    diag_name : str, optional
        Name of the FullDiagnostics to add Efield_correction fields to. Required if
        enable_diagnostics is True.
    """

    def __init__(
        self,
        sim,
        correction_interval,
        potential_expression,
        enable_diagnostics=False,
        diag_name=None,
    ):
        self.sim = sim
        self.correction_interval = correction_interval
        self.potential_expression = potential_expression
        self.enable_diagnostics = enable_diagnostics
        self.diag_name = diag_name
        self._diagnostics_initialized = False

    def _warpx(self):
        return _get_libwarpx().libwarpx_so.get_instance()

    def _mfr(self):
        return self._warpx().multifab_register()

    def _Direction(self, comp):
        return _get_libwarpx().libwarpx_so.Direction(comp)

    def _amr(self):
        return _get_libwarpx().amr

    def setup_after_init(self):
        """Called via installafterInitEsolve after the initial field solve.

        Sets up the Efield_correction diagnostic MultiFabs if requested.
        """

        warpx = self._warpx()
        mfr = self._mfr()
        lev = 0

        for comp in (0, 1, 2):
            direction = self._Direction(comp)
            ref_mf = mfr.get("Efield_fp", dir=direction, level=lev)
            mfr.alloc_init(
                "E_vac",
                direction,
                lev,
                ref_mf.box_array(),
                ref_mf.dm(),
                ref_mf.n_comp,
                ref_mf.n_grow_vect,
                0.0,
                True,
                True,
            )
            
        warpx.compute_vacuum_efield()

        if not self.enable_diagnostics:
            return

        for comp in (0, 1, 2):
            direction = self._Direction(comp)
            ref_mf = mfr.get("Efield_fp", dir=direction, level=lev)
            mfr.alloc_init(
                "Efield_correction",
                direction,
                lev,
                ref_mf.box_array(),
                ref_mf.dm(),
                ref_mf.n_comp,
                ref_mf.n_grow_vect,
                0.0,
                True,
                True,
            )

        if self.diag_name is not None:
            warpx.add_field_to_diagnostic(self.diag_name, "Efield_correction", lev)

        self._diagnostics_initialized = True

    def correct_field(self):
        """Called via installafterEsolve. Applies the Poisson correction every N steps."""
        warpx = self._warpx()
        # afterEsolve fires inside the step, before istep is incremented, so
        # getistep returns the 0-based index of the step that just solved its
        # fields. Use (step + 1) so corrections land at the end of steps
        # correction_interval, 2*correction_interval, ... (matching afterstep).
        step = warpx.getistep(lev=0)

        if (step + 1) % self.correction_interval != 0:
            return

        if self.enable_diagnostics:
            self._save_current_efield()

        # warpx.set_potential_on_eb(self.potential_expression)
        warpx.solve_poisson_efield()

        if self.enable_diagnostics and self._diagnostics_initialized:
            self._compute_correction_field()

        if self.enable_diagnostics:
            delta_phi = self.compute_potential_difference()
            print(f"[PoissonCorrector] Step {step}: Delta_phi = {delta_phi:.6e}")

    def compute_potential_difference(self, x_lo_phys=None, x_hi_phys=None):
        """Compute potential difference by integrating Ex along x at y=0.

        Parameters
        ----------
        x_lo_phys, x_hi_phys : float, optional
            Physical x coordinates for integration bounds.
            Defaults to inner and outer electrode radii.

        Returns
        -------
        float
            Approximate potential difference phi(x_lo) - phi(x_hi).
        """
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        mfr = self._mfr()

        from pywarpx import geometry  # noqa: PLC0415

        is_rz = geometry.dims == "RZ"

        geom_data = warpx.Geom(lev=0).data()
        # Radial (RZ) or x (Cartesian) is direction 0 in both layouts.
        Ex_mf = mfr.get("Efield_fp", dir=self._Direction(0), level=0)
        Ex_index_type = Ex_mf.box_array().ix_type()

        domain = geom_data.Domain().convert(Ex_index_type)
        lo = domain.small_end
        hi = domain.big_end

        prob_lo = geom_data.ProbLo()
        dx = geom_data.CellSize()[0]

        if x_lo_phys is None:
            x_lo_phys = prob_lo[0] + 0.60 * (geom_data.ProbHi()[0] - prob_lo[0])
        if x_hi_phys is None:
            x_hi_phys = prob_lo[0] + 0.95 * (geom_data.ProbHi()[0] - prob_lo[0])

        i_lo = int(round((x_lo_phys - prob_lo[0]) / dx))
        i_hi = int(round((x_hi_phys - prob_lo[0]) / dx))
        i_lo = max(i_lo, lo[0])
        i_hi = min(i_hi, hi[0])

        # Use global numpy indexing to read E along the radial integration
        # path; this performs an MPI allgather internally. Average over the
        # z direction (axisymmetric in RZ; nominally uniform in Cartesian).
        if is_rz:
            # RZ MultiFabs are 2D: [ir, iz]
            E_slice = Ex_mf[i_lo:i_hi + 1, :]
            nz = hi[1] - lo[1] + 1
        else:
            # 3D MultiFabs are [ix, iy, iz]; sample the radial line at y = 0
            mid_y = (hi[1] + lo[1]) // 2
            E_slice = Ex_mf[i_lo:i_hi + 1, mid_y, :]
            nz = hi[2] - lo[2] + 1
        integral = float(np.sum(E_slice))
        return (dx / nz) * integral

    def _save_current_efield(self):
        """Save copies of current E field components for the correction-field diagnostic."""
        mfr = self._mfr()
        self._saved_E = {}
        for comp in (0, 1, 2):
            mf = mfr.get("Efield_fp", dir=self._Direction(comp), level=0)
            self._saved_E[comp] = mf.copy()

    def _compute_correction_field(self):
        """Store E_saved - E_after in the ``Efield_correction`` diagnostic MultiFabs.

        The C++ correction re-solves the irrotational (poloidal) field and, in
        RZ, preserves E_theta (see report Section 10). This quantity is therefore
        the field change applied this step -- the poloidal drift that was
        *removed*, and ~0 in the azimuthal component in RZ -- i.e. how much
        correction was applied, not the rotational field (which is preserved in
        E itself). Hence the name ``Efield_correction`` rather than the
        historical ``Efield_rot``.
        """
        mfr = self._mfr()

        for comp in (0, 1, 2):
            direction = self._Direction(comp)
            E_after = mfr.get("Efield_fp", dir=direction, level=0)
            corr_mf = mfr.get("Efield_correction", dir=direction, level=0)

            corr_mf.copymf(self._saved_E[comp], 0, 0, 1, 0)
            corr_mf.saxpy(-1.0, E_after, 0, 0, 1, 0)

        del self._saved_E
