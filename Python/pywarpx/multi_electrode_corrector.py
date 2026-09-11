# Copyright 2026 The WarpX Community
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Quasi-static harmonic voltage clamp for driven embedded-boundary electrodes.

Controls quasi-static electrode-voltage estimates during an electromagnetic run
without a Poisson solve per step.

Per electrode ``k`` a charge-free unit field ``E_0k = -grad(phi_0k)`` is
precomputed once, with electrode ``k`` at 1 V and all others grounded. The
correction is a linear combination of these fields. The EB Poisson gradient and
native Maxwell divergence/curl need not form a compatible discrete complex, so
this does not imply that native ``div(E)`` or ``curl(E)`` is unchanged everywhere.
In particular the default ``actuator_gradient="eb_aware"`` field is not annihilated
by the Yee Faraday stencil and drives a spurious boundary ``B``;
``actuator_gradient="ordinary"`` selects the full-grid gradient, whose discrete
curl vanishes, and trades local accuracy at the conductor for that compatibility.

Per correction the effective voltages follow from the capacitance relation

    Q_j = Q_g,j + sum_k C_jk V_k   =>   V = C^-1 (Q - Q_g),

where ``C_jk = eps0 * oint_j E_0k . n dS`` is the precomputed vacuum capacitance
matrix, ``Q_j`` is the live EB-flux charge measurement for electrode ``j`` and
``Q_g,j`` is the plasma-induced charge with every electrode grounded. The feedback
is ``delta_V = relaxation * (V_target - V)``, applied as ``E += sum_k delta_V_k E_0k``.

``Q_g`` is observed either with a grounded Poisson solve (``qg_mode="grounded"``)
or, by default, with the discrete Shockley-Ramo pairing
``Q_g,k = -sum_a rho_a Psi_k[a]`` against the adjoint weighting potentials, which
needs no solve. ``Psi_k`` is built from the transpose of WarpX's own discrete EB
operator and charge functional; the plain unit-voltage basis is not the adjoint of
that functional on a non-symmetric cut-cell operator.

Usage::

    from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector
    from pywarpx.callbacks import (
        installafterEsolve,
        installafterInitatRestart,
        installafterInitEsolve,
    )

    corrector = MultiElectrodeBiasCorrector(
        sim=sim,
        correction_interval=10,
        electrodes=[
            {"name": "left", "region": "(x<0)", "potential": +300.0},
            {"name": "right", "region": "(x>0)", "potential": -700.0},
        ],
    )

With ``observer="volume"`` the electrode charge is read as a divergence over a
volume enclosing each electrode instead of a flux over its cut cells, which
needs a second expression per electrode::

    corrector = MultiElectrodeBiasCorrector(
        sim=sim,
        correction_interval=10,
        observer="volume",
        electrodes=[
            {
                "name": "cathode",
                "region": "(x*x+y*y<1.4e-4)",          # surface: sets the potential
                "volume": "(x*x+y*y<3.24e-4)*(z*z<2.7e-3)",   # encloses it
                "potential": -1.0e5,
            },
            ...
        ],
    )

Each ``volume`` must enclose its electrode with the region boundary in live
vacuum -- about two cells of clearance from the metal, a figure that is constant
in cells and so shrinks as the grid is refined -- and the volumes must not
overlap or touch. Setup checks the measured capacitance matrix for reciprocity
and refuses to run if it is violated, which is what a region boundary cutting
through metal produces.
    # afterInitEsolve does not run on a restart, so register both init hooks
    installafterInitEsolve(corrector.setup_after_init)
    installafterInitatRestart(corrector.setup_after_init)
    installafterEsolve(corrector.correct_field)

Supported envelope: driven (prescribed-potential) electrodes, grounded PEC outer
boundaries, and in RZ nodal CIC deposition without a charge filter. Floating
electrodes, circuit coupling and dielectric boundaries are not modelled.
Grounded-adjoint agreement validates the selected discrete charge measurement;
it does not by itself establish physical voltage control during particle collection.
"""


def _get_libwarpx():
    from pywarpx._libwarpx import libwarpx  # noqa: PLC0415

    return libwarpx


class MultiElectrodeBiasCorrector:
    """Maintain several driven EB electrode potentials in EM mode.

    .. important::
       **No conductor surface may coincide with a Dirichlet or PEC domain
       boundary at any point.** Such a conductor must have a layer of vacuum
       between it and that boundary. The rule is about *conducting* walls: the
       RZ regularity axis at r = 0 (boundary type "none") is a symmetry
       condition rather than a charge sink, so a body may sit on it, and a
       periodic seam is not a wall. A conductor touching a Dirichlet or PEC
       boundary is electrically the same conductor as that boundary: the
       capacitance matrix then loses its reference, becomes non-symmetric and
       ill-conditioned (measured condition number 1.6e12 against 8.1 when
       separated), and no closed surface around the electrode can be drawn in
       vacuum. The vacuum layer is required by the electrostatics, not by the
       particles; none need ever enter it. Transverse domain boundaries should
       be Dirichlet, since only a boundary that can sink charge supplies the
       reference the matrix needs.

    Parameters
    ----------
    sim : picmi.Simulation
        The initialized PICMI simulation.
    correction_interval : int
        Apply the correction every this many steps.
    electrodes : list of dict
        One entry per electrode, with keys ``"name"`` (identifier),
        ``"region"`` (a parser expression ``w_k(x,y,z)``, nonzero on that
        electrode's surface) and ``"potential"`` (the target voltage in V).
        The regions should be close to disjoint indicators of the surfaces.
    relaxation : float, optional
        Feedback under-relaxation in (0, 1]. 1.0 reaches the target in one
        correction; smaller values soften the per-step jump.
    qg_mode : {"reciprocity", "grounded"}, optional
        How the grounded plasma charge is observed. ``"reciprocity"`` (default)
        pairs the deposited charge density with the adjoint weighting potentials.
        ``"grounded"`` performs a real grounded Poisson solve per call, which is
        exact but costs one MLMG solve.
    adjoint_tolerance : float, optional
        Relative residual tolerance for the adjoint weighting solves.
    adjoint_max_iterations : int, optional
        Maximum outer iteration count for the adjoint solves.
    verbose : bool, optional
        Print the capacitance matrix and the per-correction voltages.
    actuator_gradient : {"eb_aware", "ordinary"}, keyword-only, optional
        Which discrete gradient of the setup potentials becomes the unit actuator
        field. ``"eb_aware"`` (default) uses the shortened fluid length on cut
        edges, which is the locally more accurate field but is not annihilated by
        the Yee Faraday stencil: adding it drives a spurious boundary ``B``.
        ``"ordinary"`` uses WarpX's full-grid gradient, whose discrete curl
        vanishes, at the cost of a less accurate field next to the conductor. The
        choice applies only to the actuator setup solves. The charge observer, its
        capacitance measurement, the adjoint and the grounded cross-check are
        unchanged; the capacitance is always measured from the unit fields that
        are actually applied, so the feedback stays self-consistent either way.
        Keyword-only and last so that existing positional calls are unaffected.

    State contract for ``setup_after_init``
    ---------------------------------------
    Setup runs several one-off Poisson solves, each of which overwrites
    ``Efield_fp`` and the embedded-boundary potential parser. On return, the
    **valid** cells of ``Efield_fp`` and the parser expression are restored
    exactly. Guard cells are deliberately *not* restored: WarpX treats them as
    derived rather than as state -- ``OneStep_nosub`` refills them at the start of
    every step ("E and B are up-to-date inside the domain only",
    ``Source/Evolve/WarpXEvolve.cpp``), and both setup hooks run before the first
    step. The restoration is protected by ``try``/``finally``, so a failed solve
    cannot leave the live field or the parser holding a setup value.
    """

    def __init__(
        self,
        sim,
        correction_interval,
        electrodes,
        relaxation=1.0,
        qg_mode="reciprocity",
        adjoint_tolerance=1.0e-10,
        adjoint_max_iterations=200,
        verbose=False,
        # keyword-only and last, so existing positional calls keep their meaning
        *,
        actuator_gradient="eb_aware",
        observer="surface",
        reciprocity_tolerance=1.0e-2,
    ):
        if not (0.0 < relaxation <= 1.0):
            raise ValueError("relaxation must be in (0, 1].")
        if len(electrodes) < 1:
            raise ValueError("Provide at least one electrode.")
        if qg_mode not in ("grounded", "reciprocity"):
            raise ValueError(
                f"qg_mode must be 'grounded' or 'reciprocity', got {qg_mode!r}"
            )
        if actuator_gradient not in ("eb_aware", "ordinary"):
            raise ValueError(
                "actuator_gradient must be 'eb_aware' or 'ordinary', "
                f"got {actuator_gradient!r}"
            )

        if observer not in ("surface", "volume"):
            raise ValueError(
                f"observer must be 'surface' or 'volume', got {observer!r}"
            )

        self.sim = sim
        self.observer = observer
        self.reciprocity_tolerance = float(reciprocity_tolerance)
        self.correction_interval = correction_interval
        self.electrodes = electrodes
        self.relaxation = relaxation
        self.qg_mode = qg_mode
        self.actuator_gradient = actuator_gradient
        self.adjoint_tolerance = float(adjoint_tolerance)
        self.adjoint_max_iterations = int(adjoint_max_iterations)
        self.verbose = verbose

        self.n = len(electrodes)
        self.regions = [e["region"] for e in electrodes]
        self.v_target = [float(e["potential"]) for e in electrodes]
        # The volume observer reads the charge enclosed by a region whose boundary
        # lies in live vacuum, so it needs an enclosing volume per electrode in
        # addition to the surface selector that sets the potential. About two
        # cells of clearance between the metal and the region boundary is the
        # measured requirement; that figure is constant in cells, so it shrinks
        # physically as the grid is refined.
        self.names = [
            electrodes[k].get("name", f"electrode_{k}") for k in range(self.n)
        ]
        if observer == "volume":
            missing = [
                self.names[k] for k in range(self.n) if "volume" not in electrodes[k]
            ]
            if missing:
                raise ValueError(
                    "observer='volume' needs a 'volume' expression for every "
                    f"electrode; missing for {missing}. It is the enclosing region "
                    "for the charge integral, not the surface selector that sets "
                    "the potential."
                )
            self.volumes = [e["volume"] for e in electrodes]
        else:
            self.volumes = None
        # combined EB potential of the configured electrode pattern
        self.potential_expression = " + ".join(
            f"({v})*({r})" for v, r in zip(self.v_target, self.regions)
        )

        self._ready = False
        self._capacitance = None
        self._capacitance_condition = None
        self._capacitance_asymmetry = None
        self._adjoint_residuals = None
        self._last_correction_state = None
        self._unit_names = [f"Efield_unit_{k}" for k in range(self.n)]
        self._psi_names = [f"psi_unit_{k}" for k in range(self.n)]

    # -- libwarpx accessors --------------------------------------------------
    def _warpx(self):
        return _get_libwarpx().libwarpx_so.get_instance()

    def _mfr(self):
        return self._warpx().multifab_register()

    def _Direction(self, comp):
        return _get_libwarpx().libwarpx_so.Direction(comp)

    # -- the observer --------------------------------------------------------
    def _charge_from_live_field(self):
        """Charge on each electrode, read from ``Efield_fp``.

        ``surface`` integrates ``E.n`` over the embedded boundary's cut cells.
        ``volume`` integrates the divergence over a region enclosing the
        electrode and takes off the live charge inside it, so that the boundary
        of the integration surface lies in ordinary fluid rather than on the cut
        cells whose ``E`` the staircase FDTD update freezes.
        """
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        if self.observer == "surface":
            return np.array(
                [
                    warpx.compute_eb_charge(weighting=r, field="Efield_fp")
                    for r in self.regions
                ]
            )
        q_div = np.asarray(warpx.div_e_charge_in_regions(self.volumes, 0))
        q_live = np.asarray(warpx.live_charge_in_regions(self.volumes, 0))
        return q_div - q_live

    # -- setup ---------------------------------------------------------------
    def setup_after_init(self):
        """Precompute the unit fields, the capacitance matrix and the adjoints.

        Idempotent, so it can be registered on both ``afterInitEsolve`` and
        ``afterInitatRestart``.
        """
        import numpy as np  # noqa: PLC0415

        if self._ready:
            return
        warpx = self._warpx()
        lev = 0

        for name in self._unit_names:
            self._alloc_vector_like_efield(name, lev)
        if self.qg_mode == "reciprocity":
            self._alloc_psi_fields(lev)

        # One grounded solve shared by all electrodes: E_grounded carries the
        # plasma field with every electrode at 0 V, so differencing it out of
        # each "electrode k at 1 V" solve leaves the charge-free unit field.
        # Both the baseline and the biased solves must use the same extraction, or
        # their difference would mix the two representations.
        eb_aware = self.actuator_gradient == "eb_aware"

        # The volume observer reads Efield_fp, so its columns have to be taken
        # while each unit solution is still the live field. The surface observer
        # can be applied to the stored unit fields afterwards.
        volume_columns = []

        saved = self._save_efield(lev)
        try:
            warpx.set_potential_on_eb("0.0")
            warpx.solve_poisson_efield(eb_aware_gradient=eb_aware)
            grounded = self._save_efield(lev)
            volume_base = (
                self._charge_from_live_field() if self.observer == "volume" else None
            )

            for k in range(self.n):
                warpx.set_potential_on_eb(f"1.0*({self.regions[k]})")
                warpx.solve_poisson_efield(eb_aware_gradient=eb_aware)
                for comp in (0, 1, 2):
                    direction = self._Direction(comp)
                    unit = self._mfr().get(
                        self._unit_names[k], dir=direction, level=lev
                    )
                    unit.copymf(
                        self._mfr().get("Efield_fp", dir=direction, level=lev),
                        0,
                        0,
                        1,
                        0,
                    )
                    unit.saxpy(-1.0, grounded[comp], 0, 0, 1, 0)
                if self.observer == "volume":
                    volume_columns.append(self._charge_from_live_field() - volume_base)
        finally:
            # A failed solve must not leave the live field, or the EB parser,
            # holding a setup value. See the state contract in the class docstring.
            self._restore_efield(saved, lev)
            warpx.set_potential_on_eb(self.potential_expression)

        # vacuum capacitance matrix: column k is the charge each electrode
        # carries when electrode k alone is raised to 1 V
        if self.observer == "volume":
            self._capacitance = np.column_stack(volume_columns)
        else:
            self._capacitance = np.empty((self.n, self.n))
            for j in range(self.n):
                for k in range(self.n):
                    self._capacitance[j, k] = warpx.compute_eb_charge(
                        weighting=self.regions[j], field=self._unit_names[k]
                    )
        cond = float(np.linalg.cond(self._capacitance))
        self._capacitance_condition = cond
        if not np.isfinite(cond) or cond > 1.0e12:
            raise RuntimeError(
                f"Capacitance matrix is singular/ill-conditioned (cond={cond:.3e}). "
                "Check that the electrode regions are distinct and non-overlapping."
            )

        # Reciprocity, C_jk = C_kj, is a theorem for any set of conductors, so a
        # measured violation means the measurement is wrong -- most often because
        # a region boundary passes through metal, or because two electrodes are
        # too close to enclose separately. It is checked only for the volume
        # observer: the surface integral violates it by tens of percent even on
        # well separated bodies, so asserting it there would fail every run.
        self._capacitance_asymmetry = None
        if self.n > 1:
            c = self._capacitance
            off = np.abs(c - np.diag(np.diag(c)))
            scale = off.max()
            if scale > 0.0:
                asym = float(np.max(np.abs(c - c.T)) / scale)
                self._capacitance_asymmetry = asym
                if self.observer == "volume" and asym > self.reciprocity_tolerance:
                    raise RuntimeError(
                        f"Capacitance matrix is not reciprocal (relative "
                        f"asymmetry {asym:.3e} > {self.reciprocity_tolerance:.1e}). "
                        "C_jk = C_kj is a theorem for a set of conductors, so this "
                        "is a measurement failure: check that each electrode's "
                        "'volume' encloses it with its boundary in vacuum, that the "
                        "volumes do not overlap or touch, and that no electrode "
                        "surface coincides with a domain boundary."
                    )

        if self.qg_mode == "reciprocity":
            self._build_adjoint_weighting()

        self._ready = True
        if self.verbose:
            print(f"[MultiElectrode] capacitance matrix (cond={cond:.3e}):")
            print(self._capacitance)

    def _build_adjoint_weighting(self):
        """Solve for the discrete Shockley-Ramo weighting potential per electrode."""
        warpx = self._warpx()
        self._adjoint_residuals = []
        # Psi must be the adjoint of the SAME functional the observer uses; the
        # two rows differ, and pairing one with the other leaves a residual of a
        # few percent of the bias.
        psi_regions = self.volumes if self.observer == "volume" else self.regions
        for k, region in enumerate(psi_regions):
            ok, residual = warpx.solve_adjoint_weighting(
                region=region,
                out_name=self._psi_names[k],
                tol=self.adjoint_tolerance,
                max_iter=self.adjoint_max_iterations,
                functional=self.observer,
            )
            if not ok:
                raise RuntimeError(
                    f"Adjoint weighting solve for electrode {self.names[k]!r} "
                    f"did not converge (relative residual {residual:.3e})."
                )
            self._adjoint_residuals.append(float(residual))
            if self.verbose:
                print(
                    f"[MultiElectrode] adjoint Psi[{self.names[k]}]: "
                    f"relative residual={residual:.3e}"
                )

    def _alloc_psi_fields(self, lev):
        """Allocate one nodal scalar MultiFab per electrode for Psi_k."""
        libwarpx = _get_libwarpx()
        mfr = self._mfr()
        ref = mfr.get("Efield_fp", dir=self._Direction(0), level=lev)
        nodal_ba = ref.box_array().surroundingNodes()
        # the adjoint solve scatters onto neighbouring nodes, so one ghost cell
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
        """Restore the valid cells of Efield_fp. See the class state contract.

        The trailing 0 is the ghost count: guard cells are deliberately left
        alone, because WarpX refills them at the start of each step rather than
        carrying them as state.
        """
        mfr = self._mfr()
        for comp in (0, 1, 2):
            mfr.get("Efield_fp", dir=self._Direction(comp), level=lev).copymf(
                saved[comp], 0, 0, 1, 0
            )

    # -- measurement ---------------------------------------------------------
    def measure_voltage_state(self):
        """Measure the charge/voltage state used by the feedback.

        Returns a dict with ``voltage``, ``field_charge`` and ``grounded_charge``.
        Collective: every rank must call it.
        """
        import numpy as np  # noqa: PLC0415

        lev = 0

        field_charge = self._charge_from_live_field()
        if self.qg_mode == "grounded":
            grounded_charge = self._grounded_charge_via_solve(lev)
        else:
            grounded_charge = self._grounded_charge_via_reciprocity(lev)

        voltage = np.linalg.solve(self._capacitance, field_charge - grounded_charge)
        return {
            "voltage": voltage,
            "field_charge": field_charge,
            "grounded_charge": grounded_charge,
        }

    def _grounded_charge_via_solve(self, lev):
        """Grounded plasma charge from a real save/solve/restore Poisson solve."""
        warpx = self._warpx()
        saved = self._save_efield(lev)
        try:
            warpx.set_potential_on_eb("0.0")
            warpx.solve_poisson_efield()
            q_g = self._charge_from_live_field()
        finally:
            self._restore_efield(saved, lev)
            warpx.set_potential_on_eb(self.potential_expression)
        return q_g

    def _grounded_charge_via_reciprocity(self, lev):
        """Grounded plasma charge from the adjoint pairing, without a solve."""
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        q_g = np.asarray(
            warpx.grounded_charge_from_adjoint(psi_fields=self._psi_names, lev=lev)
        )
        if self.observer == "volume":
            # The volume Psi is the adjoint of the TOTAL enclosed charge, while
            # the observable is Q_div - Q_live, so the live charge comes off the
            # reference as well. Leaving this out costs a few percent of the bias.
            q_g = q_g - np.asarray(warpx.live_charge_in_regions(self.volumes, lev))
        return q_g

    def compare_grounded_charge(self, reciprocity_charge=None):
        """Compare the adjoint observer with a real grounded Poisson solve.

        Collective and deliberately expensive. Pass the ``grounded_charge`` from
        :meth:`measure_voltage_state` to reuse the correction's own result.
        """
        import numpy as np  # noqa: PLC0415

        if reciprocity_charge is None:
            reciprocity_charge = self._grounded_charge_via_reciprocity(0)
        reciprocity_charge = np.asarray(reciprocity_charge, dtype=float)
        solved_charge = self._grounded_charge_via_solve(0)
        difference = reciprocity_charge - solved_charge
        scale = np.maximum(
            np.maximum(np.abs(reciprocity_charge), np.abs(solved_charge)),
            np.finfo(float).tiny,
        )
        return {
            "reciprocity_charge": reciprocity_charge,
            "solved_charge": solved_charge,
            "difference": difference,
            "relative_difference": np.abs(difference) / scale,
        }

    # -- state for diagnostics ----------------------------------------------
    def setup_state(self):
        """Return serializable setup data, after initialization."""
        if not self._ready:
            raise RuntimeError("The multi-electrode corrector is not initialized yet.")
        return {
            "electrode_names": list(self.names),
            "electrode_regions": list(self.regions),
            "target_voltages": list(self.v_target),
            "capacitance_matrix": self._capacitance.copy(),
            "capacitance_condition": float(self._capacitance_condition),
            "adjoint_residuals": (
                None
                if self._adjoint_residuals is None
                else list(self._adjoint_residuals)
            ),
            "qg_mode": self.qg_mode,
            "observer": self.observer,
            "capacitance_asymmetry": self._capacitance_asymmetry,
            # the capacitance above is only comparable across runs that used the
            # same actuator representation, so record which one produced it
            "actuator_gradient": self.actuator_gradient,
            "current_step": int(self._warpx().getistep(lev=0)),
        }

    def last_correction_state(self):
        """Return a copy of the most recent correction state, or ``None``."""
        if self._last_correction_state is None:
            return None
        return {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in self._last_correction_state.items()
        }

    # -- correction ----------------------------------------------------------
    def correct_field(self):
        """Drive every electrode to its target potential, preserving div and curl."""
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        # afterEsolve fires before istep is incremented; use (step + 1)
        step = warpx.getistep(lev=0)
        if (step + 1) % self.correction_interval != 0:
            return
        if not self._ready:
            return

        state = self.measure_voltage_state()
        v_now = state["voltage"]
        dv = self.relaxation * (np.array(self.v_target) - v_now)
        self._apply_bias(dv)

        try:
            time = float(warpx.gett_new(0))
        except Exception:  # noqa: BLE001
            time = float("nan")
        self._last_correction_state = {
            "step": int(step + 1),
            "time": time,
            "voltage_before": np.array(v_now, copy=True),
            # the unit fields are normalized by the same capacitance matrix used
            # above, so this is the exact linear prediction after the saxpy
            "voltage_after_predicted": np.array(v_now + dv, copy=True),
            "target_voltage": np.array(self.v_target, copy=True),
            "voltage_error_before": np.array(self.v_target - v_now, copy=True),
            "voltage_error_after_predicted": np.array(
                self.v_target - (v_now + dv), copy=True
            ),
            "delta_voltage": np.array(dv, copy=True),
            "field_charge": np.array(state["field_charge"], copy=True),
            "grounded_charge": np.array(state["grounded_charge"], copy=True),
        }

        if self.verbose:
            with np.printoptions(precision=2):
                print(
                    f"[MultiElectrode] step {step + 1}: V_now={v_now}, "
                    f"target={np.array(self.v_target)}, dV={dv}"
                )

    def _apply_bias(self, dv):
        """Efield_fp += sum_k dv_k * Efield_unit_k, a sum of discrete gradients."""
        mfr = self._mfr()
        for k in range(self.n):
            for comp in (0, 1, 2):
                direction = self._Direction(comp)
                efield = mfr.get("Efield_fp", dir=direction, level=0)
                unit = mfr.get(self._unit_names[k], dir=direction, level=0)
                efield.saxpy(float(dv[k]), unit, 0, 0, 1, 0)
