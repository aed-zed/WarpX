# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Experimental, solver-specific RZ Yee voltage clamp.

This opt-in class changes the represented conductor to WarpX's frozen-edge
staircase. It does not change the existing corrector or any simulation default.
There is no SciPy dependency and no Poisson solve during correction. Unit fields
and fixed-node observation weights are constructed by the native setup solve.

The research envelope is one level, RZ m=0, CIC without filtering, grounded
outer-r and grounded or periodic z boundaries. Free-axis plasma with
the default Verboncoeur measure is rejected. ECT uses a different operator and
must not use this class. A same-model serial collection checkpoint has passed;
restart from other models is not supported. Field recovery at absorbing outer
domain walls and higher shapes remain separate validation gates. CPU/MPI and
single-GPU fixtures are distinct from multi-GPU production validation.

``insulating_endcaps=True`` uses whole-face native pec_insulator boundaries with
no field parsers. The default ``insulating_endcap_model="z_uniform"`` retains
the restricted coax construction. The separate experimental ``"harmonic_trace"``
model allows fringing fields: radial harmonic end-face potentials determine the
actuator, while a separate Neumann harmonic potential remains the observer.
Both include weighted normal-E end flux in observation and calibration. Particle
source normalization is a separate opt-in; particle exit accounting still needs
validation. Neither model represents dielectric polarization or trapped charge.
"""


def _get_libwarpx():
    from pywarpx._libwarpx import libwarpx  # noqa: PLC0415

    return libwarpx


class StaircaseBiasCorrector:
    """Native staircase basis with its matching fixed-node charge observer.

    ``electrodes`` has the same name/region/potential entries as the original
    corrector. Here a region is a binary selector evaluated at *frozen-edge
    endpoint nodes*, not EB centroids. It must contain the whole staircase
    component; a selector cutting a frozen edge is rejected. Distinct electrode
    selectors must not overlap. No manually placed integration surface is used.

    Setup preserves live E and does not impose the initial bias. For a new run,
    finish ``sim.initialize_warpx()`` with zero E, then call ``setup_after_init``
    followed by ``initialize_vacuum_bias`` before evolving. Do not register
    these initialization calls on ``afterInitEsolve``: external initial fields
    can still be added after that hook. Do not initialize the bias with
    the geometric-EB Poisson solver: its residual is outside this actuator span.
    Register ``correct_after_step`` with ``installafterstep``, after particle
    scraping, NOT with ``installafterEsolve``. On a compatible checkpoint restart,
    build the basis and call ``resume_from_checkpoint`` instead of initializing.
    The reported voltage is a discrete charge coordinate, not an independent
    physical-voltage diagnostic. A geometric-EB grounded solve is NOT its
    cross-check and is deliberately unavailable on this class.

    With experimental insulating endcaps, measured state additionally exposes
    ``axial_boundary_flux_charge`` in coulombs. The grounded row includes that
    outward electric flux; it is not just a live-particle charge pairing and
    is not a collected-particle ledger. General fringing unit fields require the
    explicit ``insulating_endcap_model="harmonic_trace"`` research option.
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

        self.n = len(electrodes)
        self.regions = [entry["region"] for entry in electrodes]
        self.v_target = [float(entry["potential"]) for entry in electrodes]
        self.names = [
            entry.get("name", f"electrode_{k}")
            for k, entry in enumerate(electrodes)
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

        residuals = []
        boundary_options = {"insulating_endcaps": True} if self.insulating_endcaps else {}
        unit_solver = wx.solve_staircase_unit_bias
        if self.insulating_endcap_model == "harmonic_trace":
            # This setup solves separately for the Neumann observer and the
            # fringing actuator. No extra solve occurs during correction.
            unit_solver = wx.solve_staircase_insulator_bias
            boundary_options = {}
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
        if wx.validate_staircase_weights(self._weight_names) != 0.0:
            raise ValueError(
                "Every staircase component must belong to exactly one electrode; "
                "list grounded embedded electrodes too"
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
                raw = self._native_charge_state()[0]
                raw_columns.append(raw.copy())
                flux_columns.append(self._axial_boundary_flux_charge.copy())
                # Calibrate the actual observer, including its measured end flux;
                # never include the unrelated live-particle terms on restart.
                if self.insulating_endcaps:
                    raw = raw - self._axial_boundary_flux_charge
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

        state = np.asarray(
            self._warpx().staircase_charge_state(
                self._psi_names, self._weight_names,
                **({"insulating_endcaps": True} if self.insulating_endcaps else {}),
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

    def _grounded_charge_via_solve(self, lev):
        raise NotImplementedError(
            "A geometric-EB Poisson solve is not the staircase reference operator. "
            "Use the independent staircase research checks; no runtime grounded "
            "cross-check is implemented for this class yet."
        )

    def compare_grounded_charge(self, reciprocity_charge=None):
        """Reject the old geometric-EB grounded cross-check for this model."""
        del reciprocity_charge
        raise NotImplementedError(
            "A geometric-EB Poisson solve is not the staircase reference operator. "
            "Use the independent staircase research checks; no runtime grounded "
            "cross-check is implemented for this class yet."
        )

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
            "research_only": True,
            "insulating_endcaps": self.insulating_endcaps,
            "insulating_endcap_model": self.insulating_endcap_model,
        }

    def last_correction_state(self):
        """Return a copy of the most recent correction state, or ``None``."""
        if self._last_correction_state is None:
            return None
        return {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in self._last_correction_state.items()
        }
