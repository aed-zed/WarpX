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
        calculate_vector_potential=False,
    ):
        self.sim = sim
        self.correction_interval = correction_interval
        self.potential_expression = potential_expression
        self.enable_diagnostics = enable_diagnostics
        self.diag_name = diag_name
        self.calculate_vector_potential = calculate_vector_potential
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

        if not self.calculate_vector_potential:
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
        # warpx.solve_poisson_efield()
        if self.calculate_vector_potential:
            warpx.solve_poisson_efield_w_A()
        else:
            warpx.solve_poisson_efield()

        if self.enable_diagnostics and self._diagnostics_initialized:
            self._compute_correction_field()

        if self.enable_diagnostics:
            delta_phi = self.compute_potential_difference()
            print(f"[PoissonCorrector] Step {step + 1}: Delta_phi = {delta_phi:.8f}")

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
        from pywarpx import geometry  # noqa: PLC0415
        warpx = self._warpx()
        amr = self._amr()
        mfr = self._mfr()

        is_rz = geometry.dims == "RZ"

        geom_data = warpx.Geom(lev=0).data()
        # Radial (RZ) or x (Cartesian) is direction 0 in both layouts.
        Ex_mf = mfr.get("Efield_fp", dir=self._Direction(0), level=0)
        Ex_index_type = Ex_mf.box_array().ix_type()
        slice_region = geom_data.Domain().convert(Ex_index_type)

        if is_rz:
            integral = Ex_mf.sum_unique( slice_region )
            nz = slice_region.big_end[1] - slice_region.small_end[1] + 1
            dx = geom_data.CellSize()[0]
        else:
            midpoint_x = (slice_region.big_end[0] + slice_region.small_end[0])//2
            midpoint_y = (slice_region.big_end[1] + slice_region.small_end[1])//2
            slice_region.big_end = amr.IntVect([slice_region.big_end[0], midpoint_y, slice_region.big_end[2]])
            slice_region.small_end = amr.IntVect([midpoint_x, midpoint_y, slice_region.small_end[2]])
            integral = Ex_mf.sum_unique( slice_region )
            nz = slice_region.big_end[2] - slice_region.small_end[2] + 1
            dx = geom_data.CellSize()[0]
        
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
