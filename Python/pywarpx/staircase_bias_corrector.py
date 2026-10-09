# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Staircase voltage clamp for embedded-boundary electrodes (RZ, explicit Yee).

Holds separate EB conductors at prescribed potentials during electromagnetic runs.
Method and supported configurations: ``Docs/source/usage/workflows/electrode_voltage_clamp.rst``.
"""


def _get_libwarpx():
    from pywarpx._libwarpx import libwarpx  # noqa: PLC0415

    return libwarpx


class StaircaseBiasCorrector:
    """Ideal voltage source for EB conductors on WarpX's frozen-edge staircase.

    Parameters
    ----------
    sim : picmi.Simulation
    correction_interval : int
        Correct every this many steps.
    electrodes : list of dict
        ``{"name", "region", "potential"}`` per conductor. ``region`` is a parser
        expression in (x, z) that selects one whole staircase component (evaluated
        on frozen-edge endpoint nodes); regions must not overlap.
    relaxation : float, optional
        Fraction of the voltage error removed per correction, in (0, 1].
    insulating_endcaps : bool, optional
        Use whole-face ``pec_insulator`` z boundaries. ``insulating_endcap_model``
        selects ``"z_uniform"`` (default) or the experimental fringing
        ``"harmonic_trace"``.
    grounded_wall_reference : bool, optional
        Treat staircase components connected to the outer PEC radius as part of the
        ground reference (omit them from ``electrodes``). Requires two PMC
        (``neumann``) z faces, or insulating endcaps with ``harmonic_trace``.
    grounded_pairing : {"auto", "field", "rho"}, optional
        How the observer weights non-electrode charge. ``"auto"`` (default) uses
        ``"field"`` when a z face is PMC and ``"rho"`` otherwise. ``"field"`` pairs
        the unit potentials with the Gauss charge eps0 div(E) on free nodes, so it
        also counts the surface charge that particles absorbed at a PMC face leave
        behind; where Gauss's law holds it equals ``"rho"``. Select ``"rho"`` with
        PMC faces only for regression tests.

    Supported z boundaries: two periodic, PEC or PMC (``neumann``) faces in any
    PEC/PMC combination (electrodes may touch PMC faces but not PEC faces), or
    ``insulating_endcaps``.

    Call order
    ----------
    After ``sim.initialize_warpx()``: ``setup_after_init()``, then
    ``initialize_vacuum_bias()`` for a new run or ``resume_from_checkpoint()`` on a
    restart, then ``callbacks.installafterstep(clamp.correct_after_step)``
    (after particle scraping; never on ``afterEsolve``).

    ``measure_voltage_state()`` returns the clamp's own voltage coordinate; an
    independent check is the line integral of E between the conductors.
    ``compare_grounded_charge()`` checks the reciprocity grounded charge against a
    grounded solve on the same operator.
    """

    def __init__(
        self,
        sim,
        correction_interval,
        electrodes,
        relaxation=1.0,
        *,
        tolerance=1.0e-12,
        max_iterations=200,
        insulating_endcaps=False,
        insulating_endcap_model="z_uniform",
        grounded_wall_reference=False,
        grounded_pairing="auto",
        prefix="staircase",
        verbose=False,
    ):
        import numpy as np  # noqa: PLC0415

        if not isinstance(correction_interval, int) or correction_interval < 1:
            raise ValueError("correction_interval must be a positive integer")
        if not np.isfinite(tolerance) or not 0.0 < tolerance < 1.0:
            raise ValueError("tolerance must be finite and in (0, 1)")
        if not isinstance(max_iterations, int) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not isinstance(insulating_endcaps, bool):
            raise ValueError("insulating_endcaps must be a boolean")
        if insulating_endcap_model not in ("z_uniform", "harmonic_trace"):
            raise ValueError(
                "insulating_endcap_model must be 'z_uniform' or 'harmonic_trace'"
            )
        if insulating_endcap_model != "z_uniform" and not insulating_endcaps:
            raise ValueError("harmonic_trace requires insulating_endcaps=True")
        if not isinstance(grounded_wall_reference, bool):
            raise ValueError("grounded_wall_reference must be a boolean")
        if (
            grounded_wall_reference
            and insulating_endcaps
            and insulating_endcap_model != "harmonic_trace"
        ):
            # Without insulating endcaps the native setup requires two PMC z faces.
            raise ValueError(
                "grounded_wall_reference requires PMC z faces or insulating harmonic_trace"
            )
        if grounded_pairing not in ("auto", "field", "rho"):
            raise ValueError("grounded_pairing must be 'auto', 'field' or 'rho'")
        if not prefix.isidentifier():
            raise ValueError("prefix must be a Python-style identifier")
        if not 0.0 < relaxation <= 1.0:
            raise ValueError("relaxation must be in (0, 1].")
        if len(electrodes) < 1:
            raise ValueError("Provide at least one electrode.")

        self.sim = sim
        self.correction_interval = correction_interval
        self.relaxation = relaxation
        self.qg_mode = "reciprocity"
        self.adjoint_tolerance = float(tolerance)
        self.adjoint_max_iterations = int(max_iterations)
        self.verbose = verbose
        self.observer = "staircase"
        self.actuator_gradient = "staircase"
        self.insulating_endcaps = insulating_endcaps
        self.insulating_endcap_model = insulating_endcap_model
        self.grounded_wall_reference = grounded_wall_reference
        self._requested_grounded_pairing = grounded_pairing
        self.grounded_pairing = None

        self.n = len(electrodes)
        self.regions = [entry["region"] for entry in electrodes]
        self.v_target = [float(entry["potential"]) for entry in electrodes]
        self.names = [
            entry.get("name", f"electrode_{k}") for k, entry in enumerate(electrodes)
        ]
        if len(set(self.names)) != self.n or not np.all(np.isfinite(self.v_target)):
            raise ValueError("Electrode names must be unique and potentials finite")

        self._ready = False
        self._capacitance = None
        self._capacitance_condition = None
        self._capacitance_asymmetry = None
        self._raw_gauss_actuator_matrix = None
        self._boundary_flux_actuator_matrix = None
        self._adjoint_residuals = None
        self._last_correction_state = None
        self._prefix = prefix
        self._unit_names = [f"{prefix}_E_{k}" for k in range(self.n)]
        self._psi_names = [f"{prefix}_phi_{k}" for k in range(self.n)]
        self._weight_names = [f"{prefix}_fixed_{k}" for k in range(self.n)]
        self._setup_absolute_residuals = None
        self._initialization_confirmed = False
        self._last_corrected_step = None
        self._source_charge = np.zeros(self.n)
        self._raw_gauss_actuation_charge = np.zeros(self.n)
        self._boundary_flux_actuation_charge = np.zeros(self.n)
        self._axial_boundary_flux_charge = np.zeros(self.n)

    # -- native field plumbing ----------------------------------------------
    def _warpx(self):
        return _get_libwarpx().libwarpx_so.get_instance()

    def _mfr(self):
        return self._warpx().multifab_register()

    def _Direction(self, comp):
        return _get_libwarpx().libwarpx_so.Direction(comp)

    def _alloc_psi_fields(self, lev):
        """Allocate one nodal scalar MultiFab per observation weight potential."""
        libwarpx = _get_libwarpx()
        mfr = self._mfr()
        ref = mfr.get("Efield_fp", dir=self._Direction(0), level=lev)
        nodal_ba = ref.box_array().surroundingNodes()
        ngrow = libwarpx.amr.IntVect(1)
        for name in self._psi_names:
            mfr.alloc_init(name, lev, nodal_ba, ref.dm(), 1, ngrow, 0.0, True, True)

    def _alloc_vector_like_efield(self, name, lev):
        mfr = self._mfr()
        for comp in (0, 1, 2):
            direction = self._Direction(comp)
            ref = mfr.get("Efield_fp", dir=direction, level=lev)
            mfr.alloc_init(
                name,
                direction,
                lev,
                ref.box_array(),
                ref.dm(),
                1,
                ref.n_grow_vect,
                0.0,
                True,
                True,
            )

    def _save_efield(self, lev):
        mfr = self._mfr()
        return {
            comp: mfr.get("Efield_fp", dir=self._Direction(comp), level=lev).copy()
            for comp in (0, 1, 2)
        }

    def _restore_efield(self, saved, lev):
        """Restore valid E; WarpX regenerates guard cells before each step."""
        mfr = self._mfr()
        for comp in (0, 1, 2):
            mfr.get("Efield_fp", dir=self._Direction(comp), level=lev).copymf(
                saved[comp], 0, 0, 1, 0
            )

    def _apply_bias(self, dv):
        """Add a dense combination of the native staircase unit fields."""
        mfr = self._mfr()
        for k in range(self.n):
            for comp in (0, 1, 2):
                direction = self._Direction(comp)
                field = mfr.get("Efield_fp", dir=direction, level=0)
                unit = mfr.get(self._unit_names[k], dir=direction, level=0)
                field.saxpy(float(dv[k]), unit, 0, 0, 1, 0)
        self._warpx().refresh_staircase_efield_guards()

    def setup_after_init(self):
        """Build the harmonic basis; collective, idempotent, preserves live E."""
        import weakref  # noqa: PLC0415

        owner = getattr(self.sim, "_staircase_bias_owner", None)
        if owner is not None and owner() is not None and owner() is not self:
            raise RuntimeError("Only one staircase corrector may own a simulation")
        if self._ready:
            return
        mfr = self._mfr()
        scalars = self._psi_names + self._weight_names
        old_scalars = {name for name in scalars if mfr.has(name, 0)}
        old_vectors = {
            (name, k)
            for name in self._unit_names
            for k in range(3)
            if mfr.has(name, self._Direction(k), 0)
        }
        try:
            self._setup_basis()
            self.sim._staircase_bias_owner = weakref.ref(self)
        except Exception:
            # Never remove fields belonging to another object. The inner setup
            # restores live E before this rollback of our scratch allocations.
            for name in scalars:
                if name not in old_scalars and mfr.has(name, 0):
                    mfr.erase(name, 0)
            for name in self._unit_names:
                for k in range(3):
                    d = self._Direction(k)
                    if (name, k) not in old_vectors and mfr.has(name, d, 0):
                        mfr.erase(name, d, 0)
            self._ready = False
            raise

    def _setup_basis(self):
        import numpy as np  # noqa: PLC0415

        if self._ready:
            return
        wx, mfr = self._warpx(), self._mfr()
        for name in self._psi_names + self._weight_names:
            if mfr.has(name, 0):
                raise RuntimeError(f"Setup field already exists: {name}")
        for name in self._unit_names:
            if any(mfr.has(name, self._Direction(k), 0) for k in range(3)):
                raise RuntimeError(f"Setup field already exists: {name}")
            self._alloc_vector_like_efield(name, 0)
        self._alloc_psi_fields(0)
        ref = mfr.get(self._psi_names[0], level=0)
        for name in self._weight_names:
            mfr.alloc_init(
                name,
                0,
                ref.box_array(),
                ref.dm(),
                1,
                _get_libwarpx().amr.IntVect(1),
                0.0,
                True,
                True,
            )

        self.grounded_pairing = wx.staircase_grounded_pairing(
            self._requested_grounded_pairing, self.insulating_endcaps
        )
        residuals = []
        boundary_options = (
            {"insulating_endcaps": True} if self.insulating_endcaps else {}
        )
        unit_solver = wx.solve_staircase_unit_bias
        if self.insulating_endcap_model == "harmonic_trace":
            # This setup solves separately for the Neumann observer and the
            # fringing actuator. No extra solve occurs during correction.
            unit_solver = wx.solve_staircase_insulator_bias
            boundary_options = {}
        reference_options = (
            {"grounded_wall_reference": True} if self.grounded_wall_reference else {}
        )
        boundary_options.update(reference_options)
        for k, region in enumerate(self.regions):
            residuals.append(
                float(
                    unit_solver(
                        region,
                        self._psi_names[k],
                        self._weight_names[k],
                        self._unit_names[k],
                        self.adjoint_tolerance,
                        self.adjoint_max_iterations,
                        **boundary_options,
                    )
                )
            )
        if (
            wx.validate_staircase_weights(self._weight_names, **reference_options)
            != 0.0
        ):
            raise ValueError(
                "Every non-reference staircase component must belong to exactly one "
                "electrode; list disconnected zero-volt electrodes too. Components "
                "connected to the enabled grounded-wall reference must be unselected."
            )

        saved = self._save_efield(0)
        try:
            columns = []
            raw_columns = []
            flux_columns = []
            for name in self._unit_names:
                for k in range(3):
                    d = self._Direction(k)
                    mfr.get("Efield_fp", dir=d, level=0).copymf(
                        mfr.get(name, dir=d, level=0), 0, 0, 1, 0
                    )
                # Raw Gauss charge of a vacuum UNIT FIELD. Live particles are
                # not part of the calibration even when setup runs on restart.
                raw, live_fixed, grounded = self._native_charge_state()
                raw_columns.append(raw.copy())
                flux_columns.append(self._axial_boundary_flux_charge.copy())
                # Calibrate the actual observer, including its measured end flux;
                # never include the unrelated live-particle terms on restart.
                if self.insulating_endcaps:
                    raw = raw - self._axial_boundary_flux_charge
                if self.grounded_pairing == "field":
                    # The field pairing also reads the unit field's free-node
                    # divergence (solve residual). The live terms cancel exactly
                    # because psi equals the fixed-node weight on electrode nodes.
                    raw = raw - live_fixed - grounded
                columns.append(raw)
        finally:
            self._restore_efield(saved, 0)
        cap = np.column_stack(columns)
        condition = float(np.linalg.cond(cap))
        asymmetry = float(np.max(abs(cap - cap.T)) / np.max(abs(cap)))
        if (
            not np.all(np.isfinite(cap))
            or not np.isfinite(condition)
            or condition > 1e10
        ):
            raise RuntimeError("Staircase capacitance is singular or ill-conditioned")
        if asymmetry > 1e-9 or np.any(np.diag(cap) <= 0.0):
            raise RuntimeError(
                "Staircase unit fields failed charge reciprocity/sign checks"
            )
        self._capacitance = cap
        self._raw_gauss_actuator_matrix = np.column_stack(raw_columns)
        self._boundary_flux_actuator_matrix = np.column_stack(flux_columns)
        self._capacitance_condition = condition
        self._capacitance_asymmetry = asymmetry
        self._setup_absolute_residuals = residuals
        # These are harmonic solves, not the old nonsymmetric adjoint iterations.
        self._adjoint_residuals = None
        self._ready = True

    def _native_charge_state(self):
        import numpy as np  # noqa: PLC0415

        options = {"insulating_endcaps": True} if self.insulating_endcaps else {}
        if self.grounded_pairing == "field":
            options["grounded_pairing"] = "field"
        elif self._requested_grounded_pairing == "rho":
            options["grounded_pairing"] = "rho"
        state = np.asarray(
            self._warpx().staircase_charge_state(
                self._psi_names,
                self._weight_names,
                **options,
            ),
            dtype=float,
        )
        if self.insulating_endcaps:
            self._axial_boundary_flux_charge = state[3].copy()
        return state[:3]

    def initialize_vacuum_bias(self, *, rho_tolerance=0.0):
        """Initialize a fresh zero-E run; never replace an existing field.

        Intended for vacuum or an initially neutral plasma. A nonneutral
        space-charge initialization requires its own matched Poisson reference
        and is not supplied by this experimental class. ``rho_tolerance`` is
        an explicit absolute density tolerance in C/m^3 for cancellation
        roundoff in an initially neutral loading; its default is strict zero.
        """
        import numpy as np  # noqa: PLC0415

        if not self._ready or self._warpx().getistep(0) != 0:
            raise RuntimeError(
                "Initialize the staircase bias after setup, before step one"
            )
        if not np.isfinite(rho_tolerance) or rho_tolerance < 0.0:
            raise ValueError("rho_tolerance must be finite and nonnegative")
        for k in range(3):
            field = self._mfr().get("Efield_fp", dir=self._Direction(k), level=0)
            if field.norm0(0, 0, False, False) != 0.0:
                raise RuntimeError(
                    "Fresh staircase initialization requires zero incoming E"
                )
        rho = self._warpx().deposit_scratch_rho(0)
        if rho.norm0(0, 0, False, False) > rho_tolerance:
            raise RuntimeError(
                "Vacuum-bias initialization requires zero deposited rho; a charged "
                "initial state needs a matched space-charge field, not a harmonic correction"
            )
        self._apply_bias(np.asarray(self.v_target))
        self._initialization_confirmed = True

    def resume_from_checkpoint(self):
        """Keep fields from a checkpoint produced with this same staircase model.

        This does not validate an old geometric-EB checkpoint or restore a
        cumulative source ledger. Source telemetry starts a new segment.
        """
        if not self._ready or self._warpx().getistep(0) <= 0:
            raise RuntimeError("Resume requires setup and a nonzero checkpoint step")
        self._initialization_confirmed = True

    def correct_field(self):
        raise RuntimeError(
            "Use installafterstep(corrector.correct_after_step), not afterEsolve: "
            "staircase charge measurement must follow particle scraping"
        )

    def correct_after_step(self):
        """Correct after scraping; all ranks call at the updated step and time."""
        import numpy as np  # noqa: PLC0415

        if not self._initialization_confirmed:
            raise RuntimeError(
                "Initialize the staircase bias or confirm a compatible restart"
            )
        wx = self._warpx()
        step = int(wx.getistep(0))
        if (
            step == 0
            or step % self.correction_interval
            or step == self._last_corrected_step
        ):
            return
        raw, live_fixed, grounded = self._native_charge_state()
        field = raw - live_fixed
        voltage = np.linalg.solve(self._capacitance, field - grounded)
        target = np.asarray(self.v_target)
        dv = self.relaxation * (target - voltage)
        self._apply_bias(dv)
        # Legacy telemetry name: this is the OBSERVER coordinate increment.
        # With fringing end flux it is not the raw electrode Gauss increment,
        # and neither quantity alone proves a physical wire-current budget.
        self._source_charge += self._capacitance @ dv
        self._raw_gauss_actuation_charge += self._raw_gauss_actuator_matrix @ dv
        self._boundary_flux_actuation_charge += self._boundary_flux_actuator_matrix @ dv
        self._last_corrected_step = step
        self._last_correction_state = {
            "step": step,
            "time": float(wx.gett_new(0)),
            "voltage_before": voltage,
            "voltage_after_predicted": voltage + dv,
            "target_voltage": target,
            "voltage_error_before": target - voltage,
            "voltage_error_after_predicted": target - (voltage + dv),
            "delta_voltage": dv,
            "field_charge": field,
            "grounded_charge": grounded,
            "raw_gauss_charge": raw,
            "live_fixed_charge": live_fixed,
            "source_charge_since_initialization": self._source_charge.copy(),
            "raw_gauss_actuation_charge": self._raw_gauss_actuation_charge.copy(),
            "boundary_flux_actuation_charge": self._boundary_flux_actuation_charge.copy(),
            "axial_boundary_flux_charge": self._axial_boundary_flux_charge.copy(),
        }

    def measure_voltage_state(self):
        """One native divergence/deposition pair; no geometric-EB solve."""
        import numpy as np  # noqa: PLC0415

        if not self._ready:
            raise RuntimeError("Staircase corrector has not been initialized")
        raw, live_fixed, grounded = self._native_charge_state()
        field = raw - live_fixed
        return {
            "voltage": np.linalg.solve(self._capacitance, field - grounded),
            "field_charge": field,
            "grounded_charge": grounded,
            "axial_boundary_flux_charge": self._axial_boundary_flux_charge.copy(),
        }

    def _charge_from_live_field(self):
        raw, live_fixed, _ = self._native_charge_state()
        return raw - live_fixed

    def _grounded_charge_via_reciprocity(self, lev):
        if lev != 0:
            raise ValueError("The staircase observer supports level zero only")
        return self._native_charge_state()[2]

    def _grounded_charge_via_solve(self, lev, rtol=1.0e-13, max_iter=200):
        """Plasma-induced charge from a real grounded solve on the staircase operator.

        Solve the native staircase Poisson problem with every conductor at 0 V and the
        live charge density as source, then read that field with the same fixed-node
        observer. This is independent of the reciprocity route (-psi^T q).
        """
        if lev != 0:
            raise ValueError("The staircase observer supports level zero only")
        if self.insulating_endcaps:
            raise NotImplementedError(
                "The grounded staircase cross-check is not implemented for insulating endcaps"
            )
        mfr = self._mfr()
        phi_name = f"{self._prefix}_grounded_phi"
        weight_name = f"{self._prefix}_grounded_weight"
        e_name = f"{self._prefix}_grounded_E"
        ref = mfr.get("Efield_fp", dir=self._Direction(0), level=lev)
        for name in (phi_name, weight_name):
            if not mfr.has(name, lev):
                mfr.alloc_init(
                    name,
                    lev,
                    ref.box_array().surroundingNodes(),
                    ref.dm(),
                    1,
                    _get_libwarpx().amr.IntVect(1),
                    0.0,
                    True,
                    True,
                )
        if not mfr.has(e_name, self._Direction(0), lev):
            self._alloc_vector_like_efield(e_name, lev)
        residual = self._warpx().solve_staircase_grounded(
            phi_name, weight_name, e_name, float(rtol), int(max_iter)
        )
        # Read the grounded field with the native observer: swap it into Efield_fp.
        saved = self._save_efield(lev)
        try:
            for comp in (0, 1, 2):
                direction = self._Direction(comp)
                mfr.get("Efield_fp", dir=direction, level=lev).copymf(
                    mfr.get(e_name, dir=direction, level=lev), 0, 0, 1, 0
                )
            raw, live_fixed, _ = self._native_charge_state()
        finally:
            self._restore_efield(saved, lev)
            saved = None
            self._warpx().refresh_staircase_efield_guards()
        return raw - live_fixed, residual

    def compare_grounded_charge(self, reciprocity_charge=None, rtol=1.0e-13):
        """Compare the reciprocity grounded charge with a real grounded staircase solve.

        Returns a dict with both values per electrode, their difference, and the solve
        residual. Collective; does not change the live field.
        """
        import numpy as np  # noqa: PLC0415

        if reciprocity_charge is None:
            reciprocity_charge = self._grounded_charge_via_reciprocity(0)
        solved, residual = self._grounded_charge_via_solve(0, rtol=rtol)
        reciprocity_charge = np.asarray(reciprocity_charge, dtype=float)
        solved = np.asarray(solved, dtype=float)
        return {
            "reciprocity": reciprocity_charge,
            "grounded_solve": solved,
            "difference": solved - reciprocity_charge,
            "solve_residual": float(residual),
        }

    def setup_state(self):
        """Return a serializable copy of the completed setup state."""
        if not self._ready:
            raise RuntimeError("The multi-electrode corrector is not initialized yet.")
        return {
            "electrode_names": list(self.names),
            "electrode_regions": list(self.regions),
            "target_voltages": list(self.v_target),
            "capacitance_matrix": self._capacitance.copy(),
            "raw_gauss_actuator_matrix": self._raw_gauss_actuator_matrix.copy(),
            "boundary_flux_actuator_matrix": self._boundary_flux_actuator_matrix.copy(),
            "source_charge_definition": (
                "observer-coordinate actuation charge C*dV; not a measured wire current"
            ),
            "capacitance_condition": float(self._capacitance_condition),
            "adjoint_residuals": (
                None
                if self._adjoint_residuals is None
                else list(self._adjoint_residuals)
            ),
            "qg_mode": self.qg_mode,
            "observer": self.observer,
            "capacitance_asymmetry": self._capacitance_asymmetry,
            "actuator_gradient": self.actuator_gradient,
            "current_step": int(self._warpx().getistep(lev=0)),
            "harmonic_absolute_residuals": list(self._setup_absolute_residuals),
            "fixed_node_weight_fields": list(self._weight_names),
            "observer_weight_potential_fields": list(self._psi_names),
            "actuator_field_names": list(self._unit_names),
            "insulating_endcaps": self.insulating_endcaps,
            "insulating_endcap_model": self.insulating_endcap_model,
            "grounded_wall_reference": self.grounded_wall_reference,
            "grounded_pairing": self.grounded_pairing,
        }

    def last_correction_state(self):
        """Return a copy of the most recent correction state, or ``None``."""
        if self._last_correction_state is None:
            return None
        return {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in self._last_correction_state.items()
        }
