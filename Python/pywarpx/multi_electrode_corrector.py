"""
Multi-electrode harmonic-bias E-field corrector (Phase B).

Generalizes :class:`HarmonicBiasCorrector` from a single bias mode to an
arbitrary number of independently driven embedded-boundary electrodes, using a
precomputed harmonic basis and the vacuum capacitance matrix (the
Vahedi-DiPeso / Verboncoeur superposition). It is geometry- and dimension-
agnostic: each electrode is identified only by a region weighting expression
``w_k(x,y,z)``; no cylindrical symmetry, no ``r_inner``/``r_outer``.

Per electrode ``k`` a charge-free *unit field* ``E_0k = -grad(phi_0k)`` is
precomputed once (electrode ``k`` at 1 V, all others grounded). Adding any
combination ``sum_k V_k E_0k`` is a sum of discrete gradients, so it changes
neither ``div(E)`` nor ``curl(E)`` -- it maintains the electrode bias while
preserving Gauss's law and the inductive field.

Per correction the present per-electrode effective voltages are recovered from
the induced charges through the capacitance relation

    Q_j = Q_{g,j} + sum_k C_{jk} V_k   =>   V = C^{-1} (Q - Q_g),

where ``C_{jk} = eps0 * oint_j E_0k . n dS`` is the (precomputed) vacuum
capacitance matrix, ``Q_j`` is the induced charge on electrode ``j`` of the
live field, and ``Q_{g,j}`` is the plasma-induced charge with all electrodes
grounded (measured each correction via a grounded solve, so the screening is
handled exactly -- not assumed small). The feedback is

    delta_V = relaxation * (V_target - V),   E += sum_k delta_V_k E_0k.

See the implementation report, Sections 12.5 and 15 (Phase B).

Usage::

    from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector
    from pywarpx.callbacks import installafterEsolve, installafterInitEsolve

    corrector = MultiElectrodeBiasCorrector(
        sim=sim,
        correction_interval=10,
        electrodes=[
            {"name": "left",  "region": "(x<0)", "potential": +300.0},
            {"name": "right", "region": "(x>0)", "potential": -700.0},
        ],
        enable_gauss_clean=True,
    )
    installafterInitEsolve(corrector.setup_after_init)
    installafterEsolve(corrector.correct_field)

Only driven (prescribed-potential) electrodes are handled here; floating
electrodes (potential set by collected charge / a circuit) are Phase C.
"""


def _get_libwarpx():
    from pywarpx._libwarpx import libwarpx  # noqa: PLC0415

    return libwarpx


class MultiElectrodeBiasCorrector:
    """Maintain several driven EB electrode potentials in EM mode (curl-safe).

    Parameters
    ----------
    sim : picmi.Simulation
        The initialized PICMI simulation.
    correction_interval : int
        Apply the correction every this many steps.
    electrodes : list of dict
        One entry per independent electrode, each with keys:
          * ``"name"``      -- a short identifier (for diagnostics),
          * ``"region"``    -- a parser expression ``w_k(x,y,z)`` (nonzero on
                               that electrode's surface region, e.g. ``"(x<0)"``),
          * ``"potential"`` -- the target potential ``V_k`` in volts.
        The regions should be (close to) disjoint indicators of the electrode
        surfaces. The combined EB potential expression is built as
        ``sum_k V_k*(region_k)``.
    relaxation : float, optional
        Feedback under-relaxation in (0, 1]. 1.0 reaches the target in one
        correction (the fixed point); <1 softens the per-step jump.
    enable_gauss_clean : bool, optional
        If True, run the homogeneous Boris/Marder Gauss clean before the bias.
    verbose : bool, optional
        Print per-electrode voltages each correction.
    """

    def __init__(
        self,
        sim,
        correction_interval,
        electrodes,
        relaxation=1.0,
        enable_gauss_clean=False,
        verbose=False,
    ):
        if not (0.0 < relaxation <= 1.0):
            raise ValueError("relaxation must be in (0, 1].")
        if len(electrodes) < 1:
            raise ValueError("Provide at least one electrode.")

        self.sim = sim
        self.correction_interval = correction_interval
        self.electrodes = electrodes
        self.relaxation = relaxation
        self.enable_gauss_clean = enable_gauss_clean
        self.verbose = verbose

        self.n = len(electrodes)
        self.regions = [e["region"] for e in electrodes]
        self.v_target = [float(e["potential"]) for e in electrodes]
        self.names = [
            electrodes[k].get("name", f"electrode_{k}") for k in range(self.n)
        ]
        # Combined EB potential of the configured electrode pattern.
        self.potential_expression = " + ".join(
            f"({v})*({r})" for v, r in zip(self.v_target, self.regions)
        )

        self._ready = False
        self._capacitance = None  # numpy (n, n) matrix C_jk
        self._unit_names = [f"Efield_unit_{k}" for k in range(self.n)]

    # -- libwarpx accessors --------------------------------------------------
    def _warpx(self):
        return _get_libwarpx().libwarpx_so.get_instance()

    def _mfr(self):
        return self._warpx().multifab_register()

    def _Direction(self, comp):
        return _get_libwarpx().libwarpx_so.Direction(comp)

    # -- setup ---------------------------------------------------------------
    def setup_after_init(self):
        """Precompute the per-electrode unit fields and the capacitance matrix."""
        import numpy as np  # noqa: PLC0415

        if self._ready:
            return
        warpx = self._warpx()
        lev = 0

        for name in self._unit_names:
            self._alloc_vector_like_efield(name, lev)

        # One grounded solve (shared by all electrodes): E_grounded carries the
        # plasma field with every electrode at 0 V, so differencing it out of
        # each "electrode k at 1 V" solve leaves the charge-free unit field.
        saved = self._save_efield(lev)
        warpx.set_potential_on_eb("0.0")
        warpx.solve_poisson_efield()
        grounded = self._save_efield(lev)

        for k in range(self.n):
            warpx.set_potential_on_eb(f"1.0*({self.regions[k]})")
            warpx.solve_poisson_efield()
            for comp in (0, 1, 2):
                d = self._Direction(comp)
                unit = self._mfr().get(self._unit_names[k], dir=d, level=lev)
                unit.copymf(
                    self._mfr().get("Efield_fp", dir=d, level=lev), 0, 0, 1, 0
                )
                unit.saxpy(-1.0, grounded[comp], 0, 0, 1, 0)

        # Restore the live field and the configured EB potential.
        self._restore_efield(saved, lev)
        warpx.set_potential_on_eb(self.potential_expression)

        # Vacuum capacitance matrix C_jk = eps0 * oint_j E_0k . n dS.
        self._capacitance = np.empty((self.n, self.n))
        for j in range(self.n):
            for k in range(self.n):
                self._capacitance[j, k] = warpx.compute_eb_charge(
                    weighting=self.regions[j], field=self._unit_names[k]
                )
        cond = float(np.linalg.cond(self._capacitance))
        if not np.isfinite(cond) or cond > 1.0e12:
            raise RuntimeError(
                f"Capacitance matrix is singular/ill-conditioned (cond={cond:.3e}). "
                "Check that the electrode regions are distinct and non-overlapping."
            )
        self._ready = True
        if self.verbose:
            print(f"[MultiElectrode] capacitance matrix (cond={cond:.3e}):")
            print(self._capacitance)

    def _alloc_vector_like_efield(self, name, lev):
        mfr = self._mfr()
        for comp in (0, 1, 2):
            direction = self._Direction(comp)
            ref = mfr.get("Efield_fp", dir=direction, level=lev)
            mfr.alloc_init(
                name, direction, lev, ref.box_array(), ref.dm(), 1,
                ref.n_grow_vect, 0.0, True, True,
            )

    def _save_efield(self, lev):
        mfr = self._mfr()
        return {
            comp: mfr.get("Efield_fp", dir=self._Direction(comp), level=lev).copy()
            for comp in (0, 1, 2)
        }

    def _restore_efield(self, saved, lev):
        mfr = self._mfr()
        for comp in (0, 1, 2):
            mfr.get("Efield_fp", dir=self._Direction(comp), level=lev).copymf(
                saved[comp], 0, 0, 1, 0
            )

    # -- per-step correction -------------------------------------------------
    def measure_voltages(self):
        """Return the present per-electrode effective voltages V = C^-1 (Q - Q_g)."""
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        lev = 0

        # Live-field induced charge per electrode.
        q_now = np.array(
            [warpx.compute_eb_charge(weighting=r, field="Efield_fp") for r in self.regions]
        )

        # Plasma-induced charge with all electrodes grounded (handles screening
        # exactly). Save/restore the live field around the grounded solve.
        saved = self._save_efield(lev)
        warpx.set_potential_on_eb("0.0")
        warpx.solve_poisson_efield()
        q_g = np.array(
            [warpx.compute_eb_charge(weighting=r, field="Efield_fp") for r in self.regions]
        )
        self._restore_efield(saved, lev)
        warpx.set_potential_on_eb(self.potential_expression)

        return np.linalg.solve(self._capacitance, q_now - q_g)

    def correct_field(self):
        """Drive every electrode to its target potential (curl-preserving)."""
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        # afterEsolve fires before istep is incremented; use (step + 1).
        step = warpx.getistep(lev=0)
        if (step + 1) % self.correction_interval != 0:
            return
        if not self._ready:
            return

        if self.enable_gauss_clean:
            warpx.clean_efield_gauss_homogeneous()

        v_now = self.measure_voltages()
        dv = self.relaxation * (np.array(self.v_target) - v_now)
        self._apply_bias(dv)

        if self.verbose:
            with np.printoptions(precision=2):
                print(
                    f"[MultiElectrode] step {step}: V_now={v_now}, "
                    f"target={np.array(self.v_target)}, dV={dv}"
                )

    def _apply_bias(self, dv):
        """Efield_fp += sum_k dv_k * Efield_unit_k (a sum of discrete gradients)."""
        mfr = self._mfr()
        for k in range(self.n):
            for comp in (0, 1, 2):
                direction = self._Direction(comp)
                E = mfr.get("Efield_fp", dir=direction, level=0)
                unit = mfr.get(self._unit_names[k], dir=direction, level=0)
                E.saxpy(float(dv[k]), unit, 0, 0, 1, 0)
