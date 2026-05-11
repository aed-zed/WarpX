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
    from pywarpx.callbacks import installafterstep, installafterInitEsolve

    corrector = PoissonEfieldCorrector(
        sim=sim,
        correction_interval=10,
        potential_expression="-1.e5*(x*x+y*y<3.e-2**2)",
        enable_diagnostics=True,
        diag_name="diag1",
    )
    installafterInitEsolve(corrector.setup_after_init)
    installafterstep(corrector.correct_field)
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
        If True, register E_rot diagnostic MultiFabs and print Delta-V.
    diag_name : str, optional
        Name of the FullDiagnostics to add E_rot fields to. Required if
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

        Sets up diagnostic MultiFabs for E_rot if requested.
        """
        if not self.enable_diagnostics:
            return

        warpx = self._warpx()
        mfr = self._mfr()
        lev = 0

        for comp_name in ("x", "y", "z"):
            direction = self._Direction(comp_name)
            ref_mf = mfr.get("Efield_fp", dir=direction, level=lev)
            mfr.alloc_init(
                "Efield_rot",
                direction,
                lev,
                ref_mf.box_array(),
                ref_mf.dm(),
                1,
                ref_mf.n_grow_vect,
                0.0,
                True,
                True,
            )

        if self.diag_name is not None:
            warpx.add_field_to_diagnostic(self.diag_name, "Efield_rot", lev)

        self._diagnostics_initialized = True

    def correct_field(self):
        """Called via installafterstep. Applies the Poisson correction every N steps."""
        warpx = self._warpx()
        step = warpx.getistep(lev=0)

        if step % self.correction_interval != 0:
            return

        if self.enable_diagnostics:
            self._save_current_efield()

        warpx.set_potential_on_eb(self.potential_expression)
        warpx.solve_poisson_efield()

        if self.enable_diagnostics and self._diagnostics_initialized:
            self._compute_e_rot()

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

        geom_data = warpx.Geom(lev=0).data()
        Ex_mf = mfr.get("Efield_fp", dir=self._Direction("x"), level=0)
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

        mid_y = (hi[1] + lo[1]) // 2

        # Use global numpy indexing to read Ex along the integration path
        # This performs an MPI allgather internally
        Ex_slice = Ex_mf[i_lo:i_hi + 1, mid_y, :]
        integral = float(np.sum(Ex_slice))
        nz = hi[2] - lo[2] + 1
        return (dx / nz) * integral

    def _save_current_efield(self):
        """Save copies of current E field components for E_rot computation."""
        mfr = self._mfr()
        self._saved_E = {}
        for comp_name in ("x", "y", "z"):
            mf = mfr.get("Efield_fp", dir=self._Direction(comp_name), level=0)
            self._saved_E[comp_name] = mf.copy()

    def _compute_e_rot(self):
        """Compute E_rot = E_saved - E_poisson and store in diagnostic MultiFabs."""
        mfr = self._mfr()

        for comp_name in ("x", "y", "z"):
            direction = self._Direction(comp_name)
            E_poisson = mfr.get("Efield_fp", dir=direction, level=0)
            E_rot = mfr.get("Efield_rot", dir=direction, level=0)

            E_rot.copymf(self._saved_E[comp_name], 0, 0, 1, 0)
            E_rot.saxpy(-1.0, E_poisson, 0, 0, 1, 0)

        del self._saved_E
