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

Shockley-Ramo reciprocity cross-check (opt-in, not on the correction path)
---------------------------------------------------------------------------
The same unit fields ``E_0k`` are the Shockley-Ramo weighting fields for this
geometry, so ``Q_g``/``I_g`` can also be obtained by reciprocity from ``rho``/
``current_fp`` directly, without a grounded Poisson re-solve (rigor review,
``chatgpt_rigor_review.md`` Section 5). ``measure_grounded_charge_reciprocity()``
and ``measure_grounded_current_reciprocity()`` implement this as an
independent, opt-in cross-check -- see their docstrings for the formulas,
the reconstruction caveat for ``Q_g``, and measured agreement with the
existing grounded-solve path. Neither is called by ``correct_field()``/
``measure_voltages()``; the default behavior of this class is unchanged.

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

    # -- Shockley-Ramo weighting-potential reciprocity (cross-check) --------
    # rigor review (chatgpt_rigor_review.md, item 5 / Section 5) points out that
    # Q_g need not be re-measured with a fresh grounded Poisson solve every
    # correction: the per-electrode unit fields E_0k = -grad(psi_k) already
    # stored in Efield_unit_k *are* the Shockley-Ramo weighting fields for this
    # geometry (electrode k at 1 V, all others grounded, same boundary
    # conditions used to build the vacuum capacitance matrix). By reciprocity,
    #
    #     Q_{g,k} = -int_Omega rho(x) psi_k(x) dV        (review Eq. in Sec. 5)
    #     I_{g,k} =  int_Omega J(x) . grad(psi_k) dV = -int_Omega J . E_0k dV
    #
    # up to the sign/normal convention noted in the review. These two methods
    # implement that identity directly from the already-computed rho_fp /
    # current_fp and Efield_unit_k -- no extra Poisson solve -- as an
    # INDEPENDENT CROSS-CHECK path alongside (not a replacement for) the
    # existing grounded-resolve measure_voltages()/Q_g machinery. They are
    # never called by correct_field()/measure_voltages(); call them explicitly
    # if you want the comparison.
    def measure_grounded_charge_reciprocity(self):
        """Q_g via Shockley-Ramo weighting-potential reciprocity (cross-check).

        Returns an ``(n,)`` numpy array, the per-electrode plasma-induced
        charge ``Q_{g,k} = -int rho(x) psi_k(x) dV``, computed directly from
        the live charge density and the stored unit fields -- no Poisson
        solve.

        Implementation note (psi_k, not just grad(psi_k))
        ---------------------------------------------------
        ``Q_g`` needs the *potential* ``psi_k``, not merely its gradient
        ``E_0k``. WarpX's electromagnetic (Yee) path does not register a
        ``phi_fp`` MultiFab -- ``SolvePoissonEfield`` computes phi as a local
        C++ temporary and never stores it (confirmed: ``multifab_register()``
        has no ``phi_fp`` entry after any solve in this EM test fixture) -- so
        there is no stored potential to read back from Python. ``psi_k`` is
        therefore reconstructed here by a 1D line integral of the stored
        gradient, ``psi_k(x,y,z) = -int_{x_lo}^{x} E_0k,x(x',y,z) dx'``,
        anchored at ``psi_k = 0`` on the low-x domain face. This anchor is
        exact for this fixture: the domain boundary potential defaults to 0 V
        on every face when not overridden (verified in
        ``PoissonBoundaryHandler``: ``potential_xlo_str`` etc. default to
        ``"0"``, and the two-electrode fixture never sets
        ``warpx_potential_lo_x`` and friends), and each ``psi_k`` is itself a
        Laplace solution with ``psi_k = 0`` on all *grounded* conductors and
        domain walls, so the reconstruction anchor is a true zero of the field
        it integrates.

        Caveat -- corrected per the Opus-5 rigor review (SHOULD-FIX-1,
        CONSIDER-1); the primary evidence below supersedes the original
        quasi-neutral-only characterization.

        **Primary evidence: two clean controls, both showing a stable
        systematic factor, not noise.** On a deliberate charge-imbalance
        positive control (``recip_results_imbalanced.json``: ion density
        2.0e13 vs. electron density 1.0e13, net charge 2.4543e-9 C, ~670x
        above the quasi-neutral fixture's largest |Q_g|), the two Q_g paths
        do **not** converge as the signal rises above any solver-residual
        floor -- they disagree by a *stable* ratio (reciprocity/grounded)
        of 1.632-1.647 (mean 1.639) across all 5 logged steps and both
        electrodes (relative disagreement 63.2%-64.7%, a 1.5-percentage-point
        spread). An independent analytic single-point-charge control
        (``point_charge_test.log``: one 1e-6 C macroparticle, deposited
        charge verified to 1.0000000000000002e-6 C) gives the same picture
        with no plasma noise at all: relative difference 47.4%, identical to
        13 significant figures on both electrodes -- a systematic ratio of
        1.474. **The finding is therefore not "noise near a floor" but a
        stable multiplicative factor of ~1.5-1.6 between the two Q_g paths,
        of unknown origin, present on fixtures where noise is excluded.**
        The originally-reported quasi-neutral-fixture range (26% to
        >100,000% relative disagreement, ``reciprocity_comparison.json``,
        10 points) is consistent with this but should be read as
        noise-dominated, not as the primary evidence: the *true* Q_g there
        is itself near the ~1e-13-1e-12 C floor set by MLMG solver residual
        and floating-point cancellation in ``rho*psi_k``, so the reported
        26%-100,000%+ spread reflects that floor, not the underlying
        systematic factor visible in the two controls above.

        **Outer-boundary flux: measured, does not close (was: "none found").**
        ``point_charge_test.log`` records the Gauss-closure check for the
        point-charge control: sum(Q_g_recip) = -4.758e-7 C, sum(Q_g_existing)
        = -3.228e-7 C, vs. -Q_test = -1e-6 C. **Neither path's electrode sum
        equals -Q_test** -- a substantial fraction of the flux (over half, by
        either path) is landing on the outer domain box rather than the two
        electrodes. This retracts the earlier claim that "no missing
        outer-boundary contribution was found": grounded-ness of the domain
        walls in the ``psi_k`` construction and the reconstruction anchor
        (still true, and still needed for the anchor to be exact) is a
        different property from flux closure, and the point-charge log is
        direct evidence that the outer-boundary term is not negligible.

        **The curl-leakage explanation for the Q_g gap is a hypothesis,
        contradicted by this session's own localization data -- not the
        finding it was previously presented as.** The original diagnosis
        (line-integral reconstruction of ``psi_k`` accumulating cut-cell
        curl-leakage from the EB-aware unit-field solve, per
        ``harmonic_bias_corrector.py``'s ``_measure_curl_footprint``) is
        contradicted by two measurements from this same session:
        (1) ``locate_contrib.log`` shows that for the imbalanced fixture,
        81% of the reciprocity Q_g (-1.697e-10 of -2.100e-10 C, 34201 far
        nodes) comes from nodes *far* from the EB (|d| >= 2h), with only 19%
        (-4.03e-11 C, 1736 nodes) from the near-EB band -- the opposite of
        what a cut-cell-curl-driven error should show, since it should be
        concentrated where the curl defect actually lives; (2) the bulk
        curl residual sits at machine epsilon away from cut cells (an exact
        discrete identity, not a measurement of leakage capable of
        corrupting a line integral by 37%-64%), and ``debug_psi.log`` shows
        the reconstructed ``psi_k`` is machine-exact at its Dirichlet
        boundary values (0.0 / -3.66e-17 / 0.9999999999999998 / -4.00e-17 at
        the four checked anchor/electrode points) while overshooting to
        1.3673 elsewhere in the domain -- a pattern a path-accumulated curl
        error would not produce (it would not leave the endpoints exact).
        A better candidate root cause, not yet fixed, is a half-cell
        psi/rho collocation mismatch: immediately below, ``psi_k =
        np.zeros_like(rho)`` allocates ``psi_k`` on ``rho``'s nodal
        ``(n+1)**3`` grid, but ``Ex0`` (``Efield_unit_k``'s x-component) has
        shape ``(n, n+1, n+1)``, so ``psi_k[1:, :, :] = -np.cumsum(Ex0,
        axis=0) * dx`` consumes all ``n`` planes of ``Ex0`` to fill ``psi_k``'s
        ``n`` interior nodal planes -- an indexing choice that has not been
        checked for whether it puts ``psi_k`` and ``rho`` at the same
        physical location before they are multiplied in ``rho * psi_k``.
        This is flagged here as a candidate root cause for a future fix,
        not fixed in this pass.

        Because of all of the above, this reciprocity path should continue
        to be read as a same-order-of-magnitude / sign-sanity cross-check,
        not as a precise numerical replacement for ``measure_voltages()``'s
        own ``Q_g``. ``measure_grounded_current_reciprocity`` below, which
        only needs the gradient field ``E_0k`` (no reconstruction), does not
        have the reconstruction-path problem, though its own current-sign
        and outer-boundary completeness remain untested (see its
        docstring).

        Cost: no Poisson solve -- one ``get_charge_density`` reduction (shared
        across all electrodes) plus one ``O(N)`` cumulative sum per electrode
        along the line-integral axis. Measured 3.1-5.5 ms per call (mean 3.8
        ms, 10 calls) on the two-sphere 32^3 fixture, vs. 15.5-23.2 ms (mean
        18.4 ms) for the existing grounded-solve ``Q_g`` on the same calls --
        about 4.9x cheaper here; the gap should widen on larger grids since
        the existing path's cost is dominated by an MLMG Poisson solve while
        this path is a fixed number of MultiFab reductions.
        """
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        mfr = self._mfr()
        lev = 0

        # Live charge density (all species, MPI-reduced), on the same nodal
        # grid as Efield_unit_k's x-component cell-centering.
        mpc = warpx.multi_particle_container()
        rho = np.asarray(mpc.get_charge_density(lev, False)[:, :, :])

        geom_data = warpx.Geom(lev=lev).data()
        dx = geom_data.CellSize()[0]
        dV = dx * geom_data.CellSize()[1] * geom_data.CellSize()[2]

        q_g = np.empty(self.n)
        for k in range(self.n):
            Ex0 = np.asarray(
                mfr.get(self._unit_names[k], dir=self._Direction(0), level=lev)[
                    :, :, :
                ]
            )
            # psi_k(x_lo) = 0 (grounded domain boundary); psi_k(x_i) =
            # psi_k(x_{i-1}) - Ex0(cell i-1) * dx (forward line integral).
            psi_k = np.zeros_like(rho)
            psi_k[1:, :, :] = -np.cumsum(Ex0, axis=0) * dx
            q_g[k] = -np.sum(rho * psi_k) * dV
        return q_g

    def measure_grounded_current_reciprocity(self):
        """I_g via Shockley-Ramo weighting-field reciprocity (cross-check).

        Returns an ``(n,)`` numpy array, the per-electrode external current
        ``I_{g,k} = int J(x) . grad(psi_k) dV = -int J . E_0k dV``, computed
        directly from ``current_fp`` and the stored unit fields ``E_0k`` --
        no Poisson solve, and (unlike ``measure_grounded_charge_reciprocity``)
        no potential reconstruction: only the already-stored gradient fields
        are needed, so this diagnostic is exact given ``current_fp`` and
        ``Efield_unit_k`` (no line-integral path-dependence).

        Requires ``current_fp`` to be allocated and populated, which is only
        the case once particles have deposited current (i.e. after the first
        few PIC steps of an electromagnetic run with moving charged
        particles); with ``J = 0`` (no current yet, or an electrostatic-only
        setup where ``current_fp`` is never allocated) this returns all
        zeros -- a trivial but useful sanity check that the sign/units are
        wired correctly. On the two-sphere fixture (an EM Yee run, so
        ``current_fp`` *is* allocated and populated once the co-located
        plasma responds to the electrode fields) this method does return
        nonzero values -- e.g. ``I_g = [-1.608e-3, +3.265e-3]`` A at step 20
        (``ig_test.log``, which logs steps 10/20/30/40 only; the observed
        range across those four logged steps is ``I_g`` growing
        monotonically from ``[-7.890e-4, +1.602e-3]`` A at step 10 to
        ``[-3.217e-3, +6.519e-3]`` A at step 40), with ``current_fp`` itself
        of order 0.1-0.6 A/m^2 -- so the
        zero-current trivial case does not apply to this particular fixture.
        However, this diagnostic-implementation pass did not validate those
        nonzero values against an independent ground truth: there is no
        equivalent "existing" ``I_g`` measurement in this codebase to
        cross-check against (only ``Q_g`` has one, via the grounded-Poisson
        resolve), so the numbers above should be read as "the formula
        executes and returns a plausible nonzero, sign-consistent-looking
        result," not as a validated current. A genuine non-trivial validation
        of ``I_g`` (nonzero collected/drift current, checked against an
        independent measure such as the time derivative of collected
        electrode charge, dQ/dt) needs a dedicated test with real particle
        transport/collection and is flagged as a follow-on; it is out of
        scope for this diagnostic-implementation pass.

        Cost: no Poisson solve -- a handful of already-allocated MultiFab
        reductions. Measured ~2-3 ms per call (3 vector components x n
        electrodes) on the two-sphere 32^3 fixture.
        """
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        mfr = self._mfr()
        lev = 0

        if not mfr.has("current_fp", dir=self._Direction(0), level=lev):
            return np.zeros(self.n)

        geom_data = warpx.Geom(lev=lev).data()
        dV = (
            geom_data.CellSize()[0]
            * geom_data.CellSize()[1]
            * geom_data.CellSize()[2]
        )

        i_g = np.empty(self.n)
        for k in range(self.n):
            dot = 0.0
            for comp in (0, 1, 2):
                J = np.asarray(
                    mfr.get("current_fp", dir=self._Direction(comp), level=lev)[
                        :, :, :
                    ]
                )
                E0 = np.asarray(
                    mfr.get(self._unit_names[k], dir=self._Direction(comp), level=lev)[
                        :, :, :
                    ]
                )
                dot += np.sum(J * E0)
            i_g[k] = -dot * dV
        return i_g

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
