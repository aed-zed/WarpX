"""
Curl-preserving harmonic-bias E-field corrector (Phase 0).

Maintains a fixed conductor potential in an electromagnetic (FDTD) PIC
simulation with embedded-boundary electrodes by *adding* a scaled, precomputed
vacuum (charge-free) electrode field to E every ``correction_interval`` steps,
instead of *replacing* E with a full Poisson solve.

Why add instead of replace
--------------------------
The stored vacuum field ``E_vac = -grad(phi_vac)`` is a pure (discrete)
gradient, so adding a scalar multiple of it leaves ``curl E`` unchanged.  The
self-consistent rotational/inductive field of the plasma (E x B rotation,
drift-instability and fluctuation structure) therefore survives every
correction -- unlike ``PoissonEfieldCorrector`` (full replacement), which
returns a pure gradient and discards the rotational field.  This is the
multi-electrode-ready, curl-preserving generalization of the 2024 scalar
correction; see ``warpx-implementation-reports/poisson_efield_correction.md``
Sections 10-12 and the Vahedi-DiPeso / Verboncoeur superposition (Section 12.5).

Algorithm
---------
* ``setup_after_init`` (once): compute the EB-aware, charge-free vacuum field
  ``E_vac`` for the configured electrode potential and store it in a
  persistent, checkpointed ``Efield_vacuum`` MultiFab.  ``E_vac`` is obtained
  as the difference of two EB-aware solves at the same instant,
  ``E_vac = E_full - E_grounded``, where ``E_full`` uses the electrode
  potential and ``E_grounded`` grounds all electrodes; the (identical) plasma
  charge cancels, leaving the pure boundary (vacuum) field.  Because both
  solves are EB-aware (``setEBDirichlet`` + EB gradient) this avoids the
  naive-finite-difference mask artifact of ``add_boundary_electrostatic_field``
  (report Section 8.1).
* ``correct_field`` (every interval): measure the present potential deficit
  ``dV = V_target - V_now`` and add ``alpha * E_vac`` with
  ``alpha = relax * dV / phi_vac`` (``phi_vac`` is the line integral of
  ``E_vac``, so the added field restores the measured potential to target).

Phase 0 scope / limitations (by design)
----------------------------------------
* Single driven electrode pair; the deficit is measured by a line integral.
  With ``curl E != 0`` a line integral mixes in the rotational field; in 3D
  this is approximate (Phase 2 replaces it with a charge/flux + capacitance
  measure).  In RZ m=0 a radial integral is blind to E_theta and is clean.
* ``E_vac`` is computed on the simulation grid (producer (a)).  Its near-EB
  staircase accuracy is the known limitation that Phase 1 (BEM/FEM from STL)
  addresses.  Curl-preservation itself is independent of this accuracy.
* The optional homogeneous Gauss clean is left as an off-by-default stub.

Usage::

    from pywarpx.harmonic_bias_corrector import HarmonicBiasCorrector
    from pywarpx.callbacks import installafterEsolve, installafterInitEsolve

    corrector = HarmonicBiasCorrector(
        sim=sim,
        correction_interval=10,
        potential_expression="-1.e3*(x*x+y*y<3.e-2**2)",
        target_delta_phi=-1.0e3,
        x_lo_phys=1.0e-2,
        x_hi_phys=6.0e-2,
        enable_diagnostics=True,
        diag_name="diag1",
    )
    installafterInitEsolve(corrector.setup_after_init)
    installafterEsolve(corrector.correct_field)
"""


def _get_libwarpx():
    from pywarpx._libwarpx import libwarpx  # noqa: PLC0415

    return libwarpx


class HarmonicBiasCorrector:
    """Maintain a fixed electrode potential without destroying the rotational E-field.

    Parameters
    ----------
    sim : picmi.Simulation
        The PICMI simulation object (must be initialized).
    correction_interval : int
        Apply the harmonic-bias correction every this many steps.
    potential_expression : str
        EB potential expression (same syntax as picmi.EmbeddedBoundary potential)
        used to compute the vacuum electrode field.
    target_delta_phi : float
        Target potential difference (V) between the integration bounds; the
        correction drives the measured difference toward this value.
    x_lo_phys, x_hi_phys : float
        Physical bounds (inner/outer electrode radius) for the line-integral
        potential measurement.
    relaxation : float, optional
        Under-relaxation factor in (0, 1].  1.0 (default) resets the potential
        exactly each correction; < 1.0 applies a gentler restoring drift.
    enable_gauss_clean : bool, optional
        If True, apply a homogeneous Boris/Marder Gauss clean
        (``warpx.clean_efield_gauss_homogeneous()``) before the harmonic bias
        each correction.  This realizes the combined scheme (report Eq. 13.9):
        the clean removes the Gauss-law residual and preserves curl; the bias
        resets the potential.  Default False (bias only).
    verify_curl : bool, optional
        If True (default), measure the discrete curl footprint of the stored
        ``E_vac`` (3D only) at setup and expose it as ``self.curl_footprint``.
    enable_diagnostics : bool, optional
        If True, register the ``Efield_correction`` diagnostic (the applied
        per-step bias) and print the measured potential difference.
    diag_name : str, optional
        Name of the FullDiagnostics to add ``Efield_correction`` to.
    """

    def __init__(
        self,
        sim,
        correction_interval,
        potential_expression,
        target_delta_phi,
        x_lo_phys=None,
        x_hi_phys=None,
        electrode_weighting=None,
        relaxation=1.0,
        enable_gauss_clean=False,
        verify_curl=True,
        enable_diagnostics=False,
        diag_name=None,
    ):
        if not (0.0 < relaxation <= 1.0):
            raise ValueError("relaxation must be in (0, 1].")

        self.sim = sim
        self.correction_interval = correction_interval
        self.potential_expression = potential_expression
        self.target_delta_phi = target_delta_phi
        # Drift measurement:
        #  * Flux (geometry-agnostic, 3D): set ``electrode_weighting`` to a
        #    region expression w(x,y,z) that selects one reference electrode
        #    (e.g. "(x*x+y*y<3.5e-2**2)"). The effective applied voltage is
        #    inferred from the induced charge Q = eps0*oint w*E.n over the EB,
        #    measured against the known vacuum field -- no radii, no symmetry.
        #  * Line integral (RZ / fallback): provide ``x_lo_phys``/``x_hi_phys``
        #    and integrate the radial component along x (assumes axisymmetry).
        # See implementation report Sections 14.5 and 15.
        self.x_lo_phys = x_lo_phys
        self.x_hi_phys = x_hi_phys
        self.electrode_weighting = electrode_weighting
        self.relaxation = relaxation
        self.enable_gauss_clean = enable_gauss_clean
        self.verify_curl = verify_curl
        self.enable_diagnostics = enable_diagnostics
        self.diag_name = diag_name

        self._vacuum_ready = False
        self._phi_vac = None
        self._q_vac = None  # reference induced charge of E_vac (flux mode)
        self._use_flux = False  # decided at setup (3D + electrode_weighting)
        self._diagnostics_initialized = False
        # Populated at setup if verify_curl: {"bulk_rel", "max_rel", ...}.
        self.curl_footprint = None

    # -- small libwarpx accessors (mirror PoissonEfieldCorrector) ------------
    def _warpx(self):
        return _get_libwarpx().libwarpx_so.get_instance()

    def _mfr(self):
        return self._warpx().multifab_register()

    def _Direction(self, comp):
        return _get_libwarpx().libwarpx_so.Direction(comp)

    def _is_rz(self):
        from pywarpx import geometry  # noqa: PLC0415

        return geometry.dims == "RZ"

    # -- setup ----------------------------------------------------------------
    def setup_after_init(self):
        """Allocate the vacuum/diagnostic fields and precompute ``E_vac`` (once)."""
        if self._vacuum_ready:
            return  # idempotent: only allocate and precompute once
        mfr = self._mfr()
        warpx = self._warpx()
        lev = 0

        # Allocate the persistent, checkpointed vacuum-field MultiFabs and
        # (optionally) the applied-correction diagnostic, matching Efield_fp.
        self._alloc_vector_like_efield("Efield_vacuum", lev, checkpoint=True)
        if self.enable_diagnostics:
            self._alloc_vector_like_efield("Efield_correction", lev, checkpoint=True)
            if self.diag_name is not None:
                warpx.add_field_to_diagnostic(self.diag_name, "Efield_correction", lev)
            self._diagnostics_initialized = True

        self._compute_vacuum_field(lev)

        # Choose the drift measurement: the geometry-agnostic induced-charge
        # flux when an electrode weighting is given (now available in RZ as well
        # as 3D -- ComputeEBChargeWeighted has an RZ branch), else fall back to
        # the axisymmetric line integral (x_lo_phys/x_hi_phys).
        self._use_flux = self.electrode_weighting is not None
        if self._use_flux:
            # E_vac is built at the configured electrode potentials, so it
            # represents an effective voltage equal to target_delta_phi. With
            # phi_vac = target_delta_phi the per-step feedback reduces to
            # alpha = relax * (1 - Q[E]/Q[E_vac]) -- no line, no radii.
            self._phi_vac = self.target_delta_phi
            self._q_vac = self._warpx().compute_eb_charge(
                weighting=self.electrode_weighting, field="Efield_vacuum"
            )
            if abs(self._q_vac) == 0.0:
                raise RuntimeError(
                    "Vacuum-field induced charge is zero; cannot normalize the bias. "
                    "Check the EB potential expression and the electrode_weighting region."
                )
        else:
            if self.x_lo_phys is None or self.x_hi_phys is None:
                raise ValueError(
                    "Line-integral measurement requires x_lo_phys and x_hi_phys; "
                    "or provide electrode_weighting for the geometry-agnostic flux "
                    "measurement (3D)."
                )
            self._phi_vac = self.compute_potential_difference(field_name="Efield_vacuum")
            if abs(self._phi_vac) == 0.0:
                raise RuntimeError(
                    "Vacuum field line integral is zero; cannot normalize the bias. "
                    "Check the EB potential expression and integration bounds."
                )
        self._vacuum_ready = True

        if self.verify_curl and not self._is_rz():
            self.curl_footprint = self._measure_curl_footprint("Efield_vacuum", lev)

    def _alloc_vector_like_efield(self, name, lev, checkpoint):
        mfr = self._mfr()
        for comp in (0, 1, 2):
            direction = self._Direction(comp)
            ref_mf = mfr.get("Efield_fp", dir=direction, level=lev)
            mfr.alloc_init(
                name,
                direction,
                lev,
                ref_mf.box_array(),
                ref_mf.dm(),
                1,
                ref_mf.n_grow_vect,
                0.0,
                True,
                checkpoint,
            )

    def _compute_vacuum_field(self, lev):
        """Store the EB-aware, charge-free vacuum field in ``Efield_vacuum``.

        Computed as ``E_full - E_grounded`` from two EB-aware Poisson solves at
        the same instant, so the (identical) plasma charge cancels and only the
        electrode boundary contribution -- the vacuum field -- remains.  E is
        saved and restored so the live field is untouched; B is never modified
        by the solves.
        """
        mfr = self._mfr()
        warpx = self._warpx()

        saved = {}
        full = {}
        for comp in (0, 1, 2):
            saved[comp] = mfr.get("Efield_fp", dir=self._Direction(comp), level=lev).copy()

        # Solve with the electrode potential (E_full = E_plasma + E_boundary).
        warpx.set_potential_on_eb(self.potential_expression)
        warpx.solve_poisson_efield()
        for comp in (0, 1, 2):
            full[comp] = mfr.get("Efield_fp", dir=self._Direction(comp), level=lev).copy()

        # Solve with grounded electrodes (E_grounded = E_plasma only).
        warpx.set_potential_on_eb("0.0")
        warpx.solve_poisson_efield()

        # E_vac = E_full - E_grounded; the plasma contribution cancels.
        for comp in (0, 1, 2):
            direction = self._Direction(comp)
            grounded = mfr.get("Efield_fp", dir=direction, level=lev)
            vac = mfr.get("Efield_vacuum", dir=direction, level=lev)
            vac.copymf(full[comp], 0, 0, 1, 0)
            vac.saxpy(-1.0, grounded, 0, 0, 1, 0)

        # Restore the EB potential string and the live E-field.
        warpx.set_potential_on_eb(self.potential_expression)
        for comp in (0, 1, 2):
            mfr.get("Efield_fp", dir=self._Direction(comp), level=lev).copymf(
                saved[comp], 0, 0, 1, 0
            )

    # -- per-step correction --------------------------------------------------
    def correct_field(self):
        """Add the harmonic bias every ``correction_interval`` steps (curl-safe)."""
        warpx = self._warpx()
        # afterEsolve fires before istep is incremented; use (step + 1) so
        # corrections land at the end of steps interval, 2*interval, ...
        step = warpx.getistep(lev=0)
        if (step + 1) % self.correction_interval != 0:
            return
        if not self._vacuum_ready:
            return

        # Combined scheme (report Eq. 13.9): homogeneous Gauss clean first
        # (removes the Gauss-law residual, preserves curl), then the harmonic
        # bias (resets the potential). The clean restores the EB/domain
        # potential strings itself, so the bias still sees the configured BCs.
        if self.enable_gauss_clean:
            warpx.clean_efield_gauss_homogeneous()

        delta_phi_now = self._measure_delta_phi()
        alpha = self.relaxation * (self.target_delta_phi - delta_phi_now) / self._phi_vac
        self._apply_bias(alpha)

        if self.enable_diagnostics and self._diagnostics_initialized:
            self._store_applied_correction(alpha)
        if self.enable_diagnostics:
            print(
                f"[HarmonicBias] Step {step}: Delta_phi(before) = "
                f"{delta_phi_now:.6e}, alpha = {alpha:.6e}"
            )

    def _apply_bias(self, alpha):
        """Efield_fp += alpha * Efield_vacuum (a scaled discrete gradient)."""
        mfr = self._mfr()
        for comp in (0, 1, 2):
            direction = self._Direction(comp)
            E = mfr.get("Efield_fp", dir=direction, level=0)
            vac = mfr.get("Efield_vacuum", dir=direction, level=0)
            E.saxpy(alpha, vac, 0, 0, 1, 0)

    def _store_applied_correction(self, alpha):
        """Store alpha * E_vac (the field added this step) in ``Efield_correction``."""
        mfr = self._mfr()
        for comp in (0, 1, 2):
            direction = self._Direction(comp)
            corr = mfr.get("Efield_correction", dir=direction, level=0)
            vac = mfr.get("Efield_vacuum", dir=direction, level=0)
            # corr = alpha * vac, using only copymf + saxpy (as elsewhere):
            # corr = vac; corr += (alpha - 1) * vac  ->  corr = alpha * vac.
            corr.copymf(vac, 0, 0, 1, 0)
            corr.saxpy(alpha - 1.0, vac, 0, 0, 1, 0)

    # -- measurement ----------------------------------------------------------
    def _measure_delta_phi(self):
        """Present effective potential difference of the live ``Efield_fp``.

        Flux mode (geometry-agnostic, 3D): the induced charge is proportional to
        the effective electrode voltage, so the live voltage is the vacuum-field
        voltage scaled by the charge ratio, ``phi_vac * Q[E_fp] / Q[E_vac]``.
        Line-integral mode: the axisymmetric radial integral.
        """
        if self._use_flux:
            q_now = self._warpx().compute_eb_charge(
                weighting=self.electrode_weighting, field="Efield_fp"
            )
            return self._phi_vac * q_now / self._q_vac
        return self.compute_potential_difference()

    def compute_potential_difference(self, field_name="Efield_fp"):
        """Line-integral potential difference of ``field_name`` between the bounds.

        Integrates the radial (RZ) / x (Cartesian) component along x at y=0,
        averaged over z.  Using the same operator to normalize ``E_vac``
        (``phi_vac``) and to measure the live field makes the bias self-
        consistent: the staircase error of the line integral cancels.
        """
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        mfr = self._mfr()
        is_rz = self._is_rz()

        geom_data = warpx.Geom(lev=0).data()
        Ex_mf = mfr.get(field_name, dir=self._Direction(0), level=0)
        Ex_index_type = Ex_mf.box_array().ix_type()

        domain = geom_data.Domain().convert(Ex_index_type)
        lo = domain.small_end
        hi = domain.big_end

        prob_lo = geom_data.ProbLo()
        dx = geom_data.CellSize()[0]

        i_lo = int(round((self.x_lo_phys - prob_lo[0]) / dx))
        i_hi = int(round((self.x_hi_phys - prob_lo[0]) / dx))
        i_lo = max(i_lo, lo[0])
        i_hi = min(i_hi, hi[0])

        if is_rz:
            E_slice = Ex_mf[i_lo : i_hi + 1, :]
            nz = hi[1] - lo[1] + 1
        else:
            mid_y = (hi[1] + lo[1]) // 2
            E_slice = Ex_mf[i_lo : i_hi + 1, mid_y, :]
            nz = hi[2] - lo[2] + 1
        integral = float(np.sum(E_slice))
        return (dx / nz) * integral

    # -- curl-preservation check (3D) ----------------------------------------
    def _measure_curl_footprint(self, field_name, lev):
        """Discrete Yee curl of a stored vector field, as a curl-preservation check.

        Returns a dict with the max |curl|*dx / max|E| ratio over the whole
        valid region (``max_rel``) and over the "bulk" -- cells more than
        ``margin`` away from either electrode radius (``bulk_rel``).  For a pure
        discrete gradient the bulk ratio is ~machine epsilon; any non-trivial
        value is localized at the staircased EB cut cells (report Section 10.6).
        Cartesian 3D only.
        """
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        mfr = self._mfr()
        geom_data = warpx.Geom(lev=lev).data()
        dx, dy, dz = (geom_data.CellSize()[i] for i in range(3))
        prob_lo = geom_data.ProbLo()

        # Global (allgathered) staggered component arrays.
        Ex = np.asarray(mfr.get(field_name, dir=self._Direction(0), level=lev)[:, :, :])
        Ey = np.asarray(mfr.get(field_name, dir=self._Direction(1), level=lev)[:, :, :])
        Ez = np.asarray(mfr.get(field_name, dir=self._Direction(2), level=lev)[:, :, :])

        # Yee curl: E on edges -> curl on faces. Generic slicing keeps the
        # components aligned without hardcoding the grid size.
        curl_x = (Ez[:, 1:, :] - Ez[:, :-1, :]) / dy - (Ey[:, :, 1:] - Ey[:, :, :-1]) / dz
        curl_y = (Ex[:, :, 1:] - Ex[:, :, :-1]) / dz - (Ez[1:, :, :] - Ez[:-1, :, :]) / dx
        curl_z = (Ey[1:, :, :] - Ey[:-1, :, :]) / dx - (Ex[:, 1:, :] - Ex[:, :-1, :]) / dy

        e_scale = max(
            float(np.abs(Ex).max()), float(np.abs(Ey).max()), float(np.abs(Ez).max())
        )
        if e_scale == 0.0:
            return {"max_rel": 0.0, "bulk_rel": 0.0, "e_scale": 0.0}

        def _rel(curl, comp_dx):
            # Non-dimensionalize: |curl| has units E/length; compare to e_scale/dx.
            return float(np.abs(curl).max()) * comp_dx / e_scale

        max_rel = max(_rel(curl_x, dx), _rel(curl_y, dy), _rel(curl_z, dz))

        # Bulk: exclude cells within `margin` of either electrode radius. Build a
        # radial mask at the Bz face centers (the curl_z location) as a
        # representative interior measure.
        margin = 2.0 * max(dx, dy)
        nx, ny = curl_z.shape[0], curl_z.shape[1]
        xc = prob_lo[0] + (np.arange(nx) + 0.5) * dx
        yc = prob_lo[1] + (np.arange(ny) + 0.5) * dy
        r = np.sqrt(xc[:, None] ** 2 + yc[None, :] ** 2)
        r_inner, r_outer = self.x_lo_phys, self.x_hi_phys
        bulk = (
            (np.abs(r - r_inner) > margin)
            & (np.abs(r - r_outer) > margin)
            & (r > r_inner)
            & (r < r_outer)
        )
        bulk_mask = bulk[:, :, None] & np.ones_like(curl_z, dtype=bool)
        if bulk_mask.any():
            bulk_rel = float(np.abs(curl_z)[bulk_mask].max()) * dx / e_scale
        else:
            bulk_rel = max_rel

        return {"max_rel": max_rel, "bulk_rel": bulk_rel, "e_scale": e_scale}
