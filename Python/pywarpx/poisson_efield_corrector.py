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

import numpy as np


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

    def setup_after_init(self):
        """Called via installafterInitEsolve after the initial field solve.

        Sets up diagnostic MultiFabs for E_rot if requested.
        """
        if not self.enable_diagnostics:
            return

        warpx = self.sim.extension.warpx
        mfr = warpx.multifab_register()
        lev = 0

        component_names = {0: "x", 1: "y", 2: "z"}
        for comp in range(3):
            direction = getattr(
                self.sim.extension.warpx_Direction,
                component_names[comp],
            )
            ref_mf = warpx.multifab(f"Efield_fp[{component_names[comp]}][level={lev}]")
            mfr.alloc_init(
                "Efield_rot",
                direction,
                lev,
                ref_mf.box_array(),
                ref_mf.dm(),
                1,
                ref_mf.n_grow_vect(),
                0.0,
                True,
                True,
            )

        if self.diag_name is not None:
            warpx.add_field_to_diagnostic(self.diag_name, "Efield_rot", lev)

        self._diagnostics_initialized = True

    def correct_field(self):
        """Called via installafterstep. Applies the Poisson correction every N steps."""
        warpx = self.sim.extension.warpx
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

    def compute_potential_difference(self):
        """Compute potential difference by integrating Ex along x at y=0.

        Averages over all z positions for robustness.

        Returns
        -------
        float
            The potential difference (integral of Ex * dx, averaged over z).
        """
        warpx = self.sim.extension.warpx
        Ex_mf = warpx.multifab("Efield_fp[x][level=0]")
        Ex_index_type = Ex_mf.box_array().ix_type()

        slice_region = (
            warpx.Geom(lev=0).data().Domain().convert(Ex_index_type)
        )
        midpoint_x = (
            slice_region.big_end[0] + slice_region.small_end[0]
        ) // 2
        midpoint_y = (
            slice_region.big_end[1] + slice_region.small_end[1]
        ) // 2

        from pywarpx._libwarpx import amr  # noqa: PLC0415

        slice_region.big_end = amr.IntVect(
            [slice_region.big_end[0], midpoint_y, slice_region.big_end[2]]
        )
        slice_region.small_end = amr.IntVect(
            [midpoint_x, midpoint_y, slice_region.small_end[2]]
        )

        integral = Ex_mf.sum_unique(slice_region)
        nz = slice_region.big_end[2] - slice_region.small_end[2] + 1
        dx, _, _ = warpx.Geom(lev=0).data().CellSize()
        return -(dx / nz) * integral

    def _save_current_efield(self):
        """Save copies of current E field components for E_rot computation."""
        warpx = self.sim.extension.warpx
        self._saved_E = {}
        for comp_name in ("x", "y", "z"):
            mf = warpx.multifab(f"Efield_fp[{comp_name}][level=0]")
            self._saved_E[comp_name] = mf.copy()

    def _compute_e_rot(self):
        """Compute E_rot = E_saved - E_poisson and store in diagnostic MultiFabs."""
        warpx = self.sim.extension.warpx

        for comp_name in ("x", "y", "z"):
            E_poisson = warpx.multifab(f"Efield_fp[{comp_name}][level=0]")
            E_rot = warpx.multifab(f"Efield_rot[{comp_name}][level=0]")

            # E_rot = E_saved - E_poisson
            # First copy E_saved into E_rot
            E_rot.copy_from(self._saved_E[comp_name], 0, 0, 1, E_rot.n_grow_vect())
            # Then subtract E_poisson: E_rot = E_rot + (-1) * E_poisson
            E_rot.saxpy(E_rot, -1.0, E_poisson, 0, 0, 1, 0)

        del self._saved_E
