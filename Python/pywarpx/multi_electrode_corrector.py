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
    electrode_centers : list of (x, y, z), optional
        One centre per electrode, used ONLY by ``accumulate_absorption()`` to
        geometrically attribute an absorbed particle to the electrode it
        struck (nearest surface, see ``electrode_radii``), instead of the
        argmax(psi) heuristic. Required if ``psi_table``/``load_psi_table()``
        or ``gather_mode="deposit"`` is used, since argmax(psi) is only
        meaningful for the plain Dirichlet basis. See ``accumulate_absorption``.
    electrode_radii : list of float, optional
        One radius per electrode, paired with ``electrode_centers`` for
        nearest-*surface* (rather than nearest-centre) attribution when
        electrodes have different sizes. Defaults to zero (nearest centre)
        if ``electrode_centers`` is given but this is not.
    apply_ledger_correction : bool, optional
        Default False -- zero behavior change when off. If True,
        ``measure_voltages()`` calls ``accumulate_absorption()`` itself right
        before measuring, then adds the run-to-date ledger's ``booked``
        matrix (summed over species) to the measured per-electrode charge
        BEFORE the ``V = C^-1(Q - Q_g)`` inversion -- see
        ``measure_voltages()``'s docstring for the term, its sign, and the
        empirical justification. Requires ``book_absorption=True``.
        NOT YET SAFE ON THE ``correct_field()`` PATH: the ledger is a
        cumulative running total, so calling ``correct_field()`` repeatedly
        (its normal use) would add the FULL absorption history back in every
        ``correction_interval``, double-counting against a field that has
        already absorbed an earlier correction. Validated here only for a
        single, one-shot ``measure_voltages()`` call (the static T8
        acceptance test) -- see ``measure_voltages()``'s "CAVEAT NOT
        COVERED" for the fix this needs before it is safe in a real,
        repeatedly-corrected run.
    qg_mode : {"grounded", "reciprocity"}, optional
        How ``measure_voltages()`` obtains ``Q_g``, the plasma-induced charge
        with all electrodes grounded. ``"grounded"`` (default, unchanged) does
        a real save/grounded-solve/restore of ``Efield_fp`` every call --
        correct but costs a full Poisson solve. ``"reciprocity"`` instead
        evaluates the algebraically equivalent dot product ``Q_g,k = -sum_a
        rho_a * dV * Psi_k[a]`` directly against the already-deposited nodal
        charge density -- no Poisson solve, no save/restore of the live
        field. Requires the ADJOINT Psi tables to already be loaded via
        ``load_psi_table()`` (raises a clear error otherwise -- there is
        nothing to dot against ``rho`` without them, and the PLAIN
        Dirichlet-basis ``psi_unit_k`` is the wrong basis for this identity
        for the same non-symmetric-operator reason noted throughout this
        module, Section 3b of the report).
        VALIDATED (after a rho-source fix): agrees with ``"grounded"`` to a
        relative difference of ~1e-8-1e-11 on a pure-screening negative
        control (nonzero rho, no absorption) and ~4e-8-1e-10 across 8 probe
        positions, and is markedly faster (measured 10x on the T8 fixture).
        An earlier version of this method dotted against ``mpc.
        get_charge_density()``'s UNFILTERED deposit and disagreed by
        1.4%-99.8%; the fix (now implemented) deposits into and reads back
        the registered ``rho_fp`` instead, which carries WarpX's default
        charge-deposit filter -- the same filter ``solve_poisson_efield()``
        itself consumes -- see ``_deposit_and_read_rho_fp()``'s docstring for
        the mechanism and the before/after numbers. See
        ``measure_voltages()``'s docstring for the full derivation, the
        collective-MPI discipline this requires, and the validation numbers.
    """

    def __init__(
        self,
        sim,
        correction_interval,
        electrodes,
        relaxation=1.0,
        enable_gauss_clean=False,
        verbose=False,
        book_absorption=False,
        absorption_species=None,
        impact_histogram_cap=0,
        psi_table=None,
        gather_mode="node",
        electrode_centers=None,
        electrode_radii=None,
        apply_ledger_correction=False,
        qg_mode="grounded",
    ):
        import numpy as np  # noqa: PLC0415

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
        # Nodal scalar weighting potentials psi_k, stored directly at setup
        # instead of being reconstructed from Efield_unit_k by a line integral.
        # See setup_after_init() and measure_grounded_charge_reciprocity().
        self._psi_names = [f"psi_unit_{k}" for k in range(self.n)]
        self._psi_stored = False
        self._phi_grounded = None  # phi from the all-electrodes-grounded solve

        # -- per-impact absorption bookkeeping (ROADMAP item A) --------------
        # When a macroparticle is absorbed at x_a, the clamp's charge accounting
        # implicitly books its FULL charge q_p onto the struck electrode. The
        # correct amount is q_p * Psi_k(x_a) for EVERY conductor k, because an
        # impact on one electrode perturbs all of them. The difference is a
        # spurious source charge that the clamp then works to cancel.
        #
        # The deficit is tracked as an (electrode x species) MATRIX, never an
        # aggregate. The geometric factor 1 - Psi_k is always positive, but the
        # injected charge carries the sign of q_p, so in a multi-species device
        # (e.g. an Orbitron with both ions and electrons striking every
        # electrode) the species contributions push in OPPOSITE directions on
        # the same electrode. An aggregate can read near zero while hiding two
        # large opposing errors, and that cancellation is an accident of the
        # operating point -- it will not survive a change in species mix, bias,
        # or geometry. See reference/absorption_potential_maintenance.md.
        self.book_absorption = bool(book_absorption)
        self.absorption_species = list(absorption_species or [])
        # See measure_voltages()'s docstring for what this consumes and why.
        # Requires book_absorption=True: with it False, accumulate_absorption()
        # is a permanent no-op (returns None immediately) and the "correction"
        # would silently always be zero -- refuse construction instead of
        # accepting a kwarg combination that can never do anything.
        self.apply_ledger_correction = bool(apply_ledger_correction)
        if self.apply_ledger_correction and not self.book_absorption:
            raise ValueError(
                "apply_ledger_correction=True requires book_absorption=True: "
                "the correction consumes the per-impact (electrode x species) "
                "ledger that book_absorption populates via "
                "accumulate_absorption(); without it there is nothing to "
                "apply."
            )
        # eps[k, s]: spurious source charge booked on electrode k by species s
        self._absorb_deficit = None       # numpy (n, n_species)
        # booked[k, s]: the Ramo-weighted charge actually collected
        self._absorb_booked = None
        self._absorb_counts = None        # impacts per species
        self._buffer_cursor = {}          # species -> #particles already read
        self._absorb_history = []         # [(step, deficit.copy(), counts)]
        self._impact_histogram = []       # (x, y, z, species_index, q) samples
        self._impact_histogram_cap = int(impact_histogram_cap)
        # Optional override for the weighting potential used by the ledger.
        # The stored psi_unit_k is the PLAIN (Dirichlet) basis, which is ~1 at
        # its own electrode by boundary condition and so under-books the
        # correction by roughly an order of magnitude (T4: 6.6% vs a measured
        # ~74-80%). Supply the measured table from T5, or the adjoint Psi once
        # that is wired up. Shape: list of n nodal arrays, or a path to the
        # .npz T5 writes.
        self._psi_override = None

        # gather_mode selects how _gather_psi turns a nodal Psi table into a
        # per-particle value. "node" (default, unchanged behavior) is a plain
        # trilinear gather of the nodal table at the particle position.
        # "deposit" additionally pre-filters the nodal table with WarpX's
        # default single-pass separable binomial filter [0.25, 0.5, 0.25]
        # before the SAME trilinear gather -- see _gather_psi's docstring for
        # why: WarpX's real charge deposit is not plain nodal CIC, it is nodal
        # CIC *composed with* that filter, so gathering the unfiltered table
        # is inconsistent with what rho_fp actually measures at that position.
        if gather_mode not in ("node", "deposit"):
            raise ValueError(f"gather_mode must be 'node' or 'deposit', got {gather_mode!r}")
        self.gather_mode = gather_mode
        self._psi_filtered_cache = {}   # k -> filtered nodal array, for gather_mode="deposit"

        # qg_mode selects how measure_voltages() obtains Q_g (see its
        # docstring for the derivation). Not validated against psi-table
        # availability here: load_psi_table() is typically called from an
        # afterInitEsolve callback AFTER this constructor returns (see T8's
        # _build_adjoint()), so that check happens lazily, on first use, in
        # _grounded_charge_via_reciprocity() below -- checking it here would
        # reject the normal call order.
        if qg_mode not in ("grounded", "reciprocity"):
            raise ValueError(
                f"qg_mode must be 'grounded' or 'reciprocity', got {qg_mode!r}"
            )
        self.qg_mode = qg_mode
        # Cache for the raw nodal psi_unit_k table (used when _psi_override is
        # None). Reading it (mfr.get(...)[:, :, :]) triggers an MPI allgather
        # in pyAMReX's MultiFab.__getitem__ -- a COLLECTIVE call that every
        # rank must issue the same number of times in the same order. Once
        # filled, this cache is never invalidated (psi_unit_k is fixed after
        # setup_after_init() and never mutated again), so the collective is
        # paid at most once per electrode -- see the prefetch call at the top
        # of accumulate_absorption() for why it must not be filled lazily
        # from a rank-locally-conditioned branch.
        self._nodal_psi_cache = {}

        # Optional geometric attribution for accumulate_absorption()'s
        # struck-electrode assignment (see its docstring / defect note there).
        # `electrode_centers` is a list of n (x, y, z) tuples; `electrode_radii`
        # is an optional list of n radii (default 0, i.e. nearest-centre
        # attribution) for surface- rather than centre-nearest assignment when
        # electrodes have different sizes.
        if electrode_centers is not None:
            electrode_centers = np.asarray(electrode_centers, dtype=float)
            if electrode_centers.shape != (self.n, 3):
                raise ValueError(
                    f"electrode_centers must have shape ({self.n}, 3), got "
                    f"{electrode_centers.shape}"
                )
        self.electrode_centers = electrode_centers
        if electrode_radii is not None:
            electrode_radii = np.asarray(electrode_radii, dtype=float)
            if electrode_radii.shape != (self.n,):
                raise ValueError(
                    f"electrode_radii must have shape ({self.n},), got "
                    f"{electrode_radii.shape}"
                )
        elif electrode_centers is not None:
            electrode_radii = np.zeros(self.n)
        self.electrode_radii = electrode_radii

        if psi_table is not None:
            self.load_psi_table(psi_table)

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

        # psi_k is the SAME Laplace solution that produces E_0k below. Storing
        # it costs one nodal scalar per electrode and removes the need to
        # reconstruct it later by integrating E_0k along a ray -- see
        # measure_grounded_charge_reciprocity() for why that reconstruction is
        # the wrong operation on a cut-cell grid.
        #
        # A nodal scalar is ~1/3 the size of an edge-centred vector, and E_0k
        # is recoverable from psi_k by one gradient pass whereas the converse
        # is not.
        #
        # ORDERING IS LOAD-BEARING: this must run BEFORE the grounded solve.
        # Under the EM/ECT solver phi_fp does not exist until allocated here,
        # and SolvePoissonEfield only publishes into it if it already exists --
        # so allocating after the grounded solve would leave the grounded
        # snapshot at zero and the plasma potential (O(100 V) on this fixture)
        # would survive into psi_k, which is nominally O(1).
        self._psi_stored = self._alloc_psi_fields(lev)

        # One grounded solve (shared by all electrodes): E_grounded carries the
        # plasma field with every electrode at 0 V, so differencing it out of
        # each "electrode k at 1 V" solve leaves the charge-free unit field.
        saved = self._save_efield(lev)
        warpx.set_potential_on_eb("0.0")
        warpx.solve_poisson_efield()
        grounded = self._save_efield(lev)

        if self._psi_stored:
            # Snapshot the grounded-solve potential so the plasma contribution
            # is differenced out of each unit solve, exactly as it is for E_0k
            # above (E_full - E_grounded).
            mfr0 = self._mfr()
            ref0 = mfr0.get("phi_fp", level=lev)
            mfr0.alloc_init(
                "phi_grounded_tmp", lev, ref0.box_array(), ref0.dm(), 1,
                ref0.n_grow_vect, 0.0, True, True,
            )
            self._phi_grounded = mfr0.get("phi_grounded_tmp", level=lev)
            self._phi_grounded.copymf(ref0, 0, 0, 1, 0)

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
            if self._psi_stored:
                self._capture_psi(k, lev)

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

    def _alloc_psi_fields(self, lev):
        """Allocate one nodal scalar MultiFab per electrode for psi_k.

        Returns True if the storage is available AND the solver publishes
        ``phi_fp`` (so psi_k can actually be captured); False otherwise, in
        which case the reciprocity path falls back to the legacy line-integral
        reconstruction and says so.
        """
        mfr = self._mfr()
        try:
            ref = mfr.get("phi_fp", level=lev)
        except Exception:
            # phi_fp is allocated by WarpX only for the electrostatic solver
            # modes (WarpX.cpp: LabFrame / LabFrameElectroMagnetostatic /
            # LabFrameEffectivePotential). This corrector also runs under the
            # EM/ECT solver, which calls SolvePoissonEfield directly without
            # ever allocating it -- so allocate it here, on rho's nodal grid,
            # and let SolvePoissonEfield populate it.
            try:
                # Build the nodal BoxArray from Efield_fp (always present) by
                # taking surrounding nodes -- the same grid rho and phi live
                # on in SolvePoissonEfield.
                eref = mfr.get("Efield_fp", dir=self._Direction(0), level=lev)
                nba = eref.box_array().surroundingNodes()
                mfr.alloc_init(
                    "phi_fp", lev, nba, eref.dm(), 1,
                    eref.n_grow_vect, 0.0, True, True,
                )
                ref = mfr.get("phi_fp", level=lev)
                if self.verbose:
                    print(
                        "[MultiElectrode] allocated phi_fp (absent under the "
                        "EM/ECT solver) so psi_k can be stored directly."
                    )
            except Exception as exc:
                if self.verbose:
                    print(
                        f"[MultiElectrode] phi_fp unavailable ({exc}); psi_k "
                        "cannot be stored directly and the reciprocity Q_g "
                        "falls back to line-integral reconstruction. Its "
                        "output is an order-of-magnitude cross-check only."
                    )
                return False
        for name in self._psi_names:
            mfr.alloc_init(
                name, lev, ref.box_array(), ref.dm(), 1,
                ref.n_grow_vect, 0.0, True, True,
            )
        return True

    def _capture_psi(self, k, lev):
        """Copy the just-solved phi_fp into psi_unit_k.

        Called immediately after the ``electrode k at 1 V, all others at 0 V``
        solve, so phi_fp holds exactly psi_k (the unit-electrode Laplace basis
        function) plus the plasma potential. The plasma part is removed by the
        same grounded-solve differencing used for E_0k.
        """
        mfr = self._mfr()
        psi = mfr.get(self._psi_names[k], level=lev)
        psi.copymf(mfr.get("phi_fp", level=lev), 0, 0, 1, 0)
        if self._phi_grounded is not None:
            psi.saxpy(-1.0, self._phi_grounded, 0, 0, 1, 0)

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
    # -- per-impact absorption bookkeeping -----------------------------------
    def load_psi_table(self, table):
        """Use a measured/adjoint weighting potential for the ledger.

        `table` is either a list of n nodal arrays or a path to the .npz that
        `inputs_3d_t5_probe_band.py` writes (keys ``psi_0`` .. ``psi_{n-1}``).

        WHY THIS EXISTS. The stored ``psi_unit_k`` is the plain Dirichlet
        basis: psi = 1 on electrode k by boundary condition. Gathering it at an
        impact site therefore returns a value near 1 and books a small deficit
        no matter what the truth is -- T4 measured a 6.6% mis-booking rate
        where the weighting potential measured through WarpX's own charge
        identity implies 74-80%. Booking q_p * Psi_k zeroes the residual BY
        CONSTRUCTION for whatever Psi is supplied, so the plain basis is not
        "wrong" so much as solving the wrong problem precisely. This is how the
        right one gets in.
        """
        import numpy as np  # noqa: PLC0415

        if isinstance(table, (str, bytes)):
            with np.load(table) as z:
                table = [np.array(z[f"psi_{k}"]) for k in range(self.n)]
        table = [np.asarray(t) for t in table]
        if len(table) != self.n:
            raise ValueError(
                f"psi_table has {len(table)} arrays but there are {self.n} "
                "electrodes")
        self._psi_override = table
        self._psi_filtered_cache = {}   # stale after the table changes
        return self

    def _nodal_psi(self, k):
        """The raw nodal Psi_k table (override if supplied, else psi_unit_k).

        CACHED AND COLLECTIVE. When there is no override, this reads the
        psi_unit_k MultiFab via ``mf[:, :, :]``, which in pyAMReX performs an
        MPI allgather so every rank gets the full global nodal array -- a
        COLLECTIVE call that every rank must issue the same number of times,
        in the same order. psi_unit_k is fixed after ``setup_after_init()``
        and never mutated again, so the result is cached on first use and the
        collective is paid at most once per electrode. This cache is why
        ``accumulate_absorption()`` prefetches every electrode's table
        unconditionally near the top of the method, BEFORE any rank-locally-
        conditioned skip logic -- see that method's docstring. Do not call
        this lazily from a branch whose condition can differ between ranks
        (e.g. "this rank has new particles this step"): if one rank fills the
        cache while another does not reach the call at all, the allgather
        deadlocks.
        """
        import numpy as np  # noqa: PLC0415

        if self._psi_override is not None:
            return np.asarray(self._psi_override[k])
        if k not in self._nodal_psi_cache:
            self._nodal_psi_cache[k] = np.asarray(
                self._mfr().get(self._psi_names[k], level=0)[:, :, :]
            )
        return self._nodal_psi_cache[k]

    @staticmethod
    def _binomial_filter_3d(psi_nodal):
        """WarpX's default rho/current filter: one pass of the separable
        [0.25, 0.5, 0.25] binomial stencil along each axis, zero-padded at the
        array edges (Psi is ~0 in the free region away from its own electrode
        and the table here is only ever evaluated deep in the interior, so the
        edge treatment does not matter for the booking use case).

        HARDCODES A SINGLE PASS. This matches WarpX's default
        (``warpx.use_filter=1``, ``warpx.filter_npass_each_dir=1``); if a run
        ever configures more passes, this must be generalized (apply the pass
        this many times) rather than silently mismatching the true deposit.
        """
        import numpy as np  # noqa: PLC0415

        def one_pass(a, axis):
            left = np.zeros_like(a)
            right = np.zeros_like(a)
            idx_dst = [slice(None)] * a.ndim
            idx_src = [slice(None)] * a.ndim
            idx_dst[axis] = slice(1, None)
            idx_src[axis] = slice(0, -1)
            left[tuple(idx_dst)] = a[tuple(idx_src)]
            idx_dst2 = [slice(None)] * a.ndim
            idx_src2 = [slice(None)] * a.ndim
            idx_dst2[axis] = slice(0, -1)
            idx_src2[axis] = slice(1, None)
            right[tuple(idx_dst2)] = a[tuple(idx_src2)]
            return 0.5 * a + 0.25 * left + 0.25 * right

        out = np.asarray(psi_nodal, dtype=float)
        for ax in (0, 1, 2):
            out = one_pass(out, ax)
        return out

    def _psi_for_gather(self, k):
        """The nodal table _gather_psi actually reads, per ``self.gather_mode``."""
        if self.gather_mode == "node":
            return self._nodal_psi(k)
        # "deposit"
        if k not in self._psi_filtered_cache:
            self._psi_filtered_cache[k] = self._binomial_filter_3d(self._nodal_psi(k))
        return self._psi_filtered_cache[k]

    def _gather_psi(self, k, x, y, z, mode=None):
        """Deposit-consistent gather of psi_k at particle positions.

        WHY THIS IS NOT A PLAIN TRILINEAR GATHER OF THE STORED TABLE.
        WarpX's real charge deposit to ``rho_fp`` is not plain nodal linear
        (CIC) interpolation: it is nodal CIC *composed with* WarpX's default
        single-pass separable binomial filter ``[0.25, 0.5, 0.25]`` per axis
        (``warpx.use_filter=1``, the default). Measured directly (probe
        macroparticle, off-node positions, nx=32 fixture,
        ``verify_offnode_weights.py``): the composed weight w_a at each node a
        matches nodal-CIC-then-filter to **machine precision**
        (max|actual-predicted| = 8-9e-17 across 64 affected nodes, two
        independent non-mirrored fractional offsets), whereas the
        CIC-to-cell-center + cell-to-node-average model considered earlier
        matches only to 3.8-4.4e-2 (max node-weight error) and reproduces the
        true charge functional ``sum_a w_a*Psi[a]`` to only 0.13%-0.78%
        relative -- small enough to look "close" but not exact, and not
        obviously wrong until checked against non-mirrored offsets.

        Because the filter is a symmetric (self-adjoint) linear operator,
        ``sum_a rho_a * Psi_a = sum_a (Filter . CIC)_a * Psi_a
        = sum_a CIC_a * (Filter . Psi)_a`` -- i.e. the deposit-consistent
        gather is: filter the nodal Psi table ONCE (``_psi_for_gather``,
        cached), then use the ORDINARY nodal trilinear gather below, unchanged
        from the plain "node" mode. No cell-centered indexing is needed; the
        earlier cell-averaged-table design was superseded once the off-node
        check falsified its underlying deposit model.

        ``mode`` overrides ``self.gather_mode`` for this call only
        ("node" | "deposit"); default None uses the instance setting.
        """
        import numpy as np  # noqa: PLC0415

        geom = self._warpx().Geom(lev=0).data()
        dx = np.array(geom.CellSize())
        lo = np.array(geom.ProbLo())
        mode = mode or self.gather_mode
        if mode == "node":
            psi = self._nodal_psi(k)
        elif mode == "deposit":
            psi = self._psi_for_gather(k)
        else:
            raise ValueError(f"mode must be 'node' or 'deposit', got {mode!r}")

        g = (np.stack([x, y, z], axis=1) - lo) / dx    # (N, 3) in cell units
        i0 = np.floor(g).astype(np.int64)
        f = g - i0
        shp = np.array(psi.shape) - 1
        i0 = np.clip(i0, 0, shp - 1)
        out = np.zeros(len(x))
        for c0 in (0, 1):
            for c1 in (0, 1):
                for c2 in (0, 1):
                    w = ((1 - f[:, 0]) if c0 == 0 else f[:, 0]) \
                        * ((1 - f[:, 1]) if c1 == 0 else f[:, 1]) \
                        * ((1 - f[:, 2]) if c2 == 0 else f[:, 2])
                    out += w * psi[i0[:, 0] + c0, i0[:, 1] + c1, i0[:, 2] + c2]
        return out

    def accumulate_absorption(self):
        """Book the Ramo-weighted charge of newly absorbed particles.

        Reads the EB scrape buffer incrementally (a per-species cursor, so each
        particle is counted exactly once), gathers psi_k at every impact site,
        and accumulates two (electrode x species) matrices:

            booked[k, s]  = sum_p q_p * Psi_k(x_p)      <- the correct charge
            deficit[k, s] = sum_p q_p * (1 - Psi_k(x_p)) for the struck
                            electrode; for k != struck, -q_p * Psi_k(x_p)

        ``deficit`` is the spurious source charge the naive full-q_p booking
        would inject. Its exact target under correct booking is ZERO, which is
        what makes it a far more sensitive acceptance metric than energy: the
        error is linear and one-signed per species, while energy is quadratic.

        REQUIRES A T6-VALIDATED Psi TABLE -- THE STORED BASIS IS NOT ONE
        -----------------------------------------------------------------
        STATUS 2026-08-13 (updated after review): the earlier non-convergence
        (stall at 2.0e-2, identical at 2000/20000 iterations) was an
        INCONSISTENT RHS -- the scatter_from=1 builder wrote each Dirichlet
        source node's diagonal onto the node itself, rows where the solve
        operator is identically zero; 99.7% of the RHS norm sat on constrained
        nodes in a numpy reproduction. Fixed by restricting the transpose
        output to free rows (the object Q7 always validated). The solve now
        CONVERGES (rel. residual ~1e-10, both electrodes, T6 fixture) -- but
        its solution is still NOT USABLE FOR BOOKING: on commensurate cut
        geometries (R/dx exactly 4 here) AMReX's min(h) row scaling produces
        near-zero rows, the A_DF-row functional couples to them with 1/h
        weights, and the posed system's true solution carries ~1e12-scale
        entries (measured: gathered Psi ~ -1e12 vs bounded truth ~1e-13; T6's
        acceptance comparison catches this even though the convergence flag no
        longer can). The remaining defect is the RHS FUNCTIONAL: it must be
        built from the charge functional the truth is measured with
        (ChargeOnEB's area-fraction flux), not from the operator's Dirichlet
        row sums. Until that lands, the working input is the probed table
        below.

        Pass `psi_table=` with a table probed on BOTH sides of the EB surface
        (`inputs_3d_t5_probe_band.py --band_in 1.0 --band 2.0`) and confirm it
        with `inputs_3d_t6_booking_exactness.py` before enabling. Measured
        booking error across five configurations (equal and unequal radii,
        nx=32 and 64): 1.7%-8.2%. It does NOT refine away, and it is not set
        by cells-per-radius -- R/dx = 8.0 books worse (5.5%) than R/dx = 4.0
        (1.7%). Treat the per-impact correction as carrying a few-percent
        residual, not as exact. The adjoint route is the one with a claim to
        exactness (check_adjoint_identity.py Q1, 1e-17); it is not yet wired
        into a solve.

        Without it, the gathered weighting is wrong by a large factor:
        T6 (`inputs_3d_t6_booking_exactness.py`) straddles a real absorption
        event and compares the booked charge against the change WarpX itself
        reports. Neither candidate table passes:

            plain psi_unit_k        : 0.868 vs true 0.115  -> 7.57x over
            T5 outside-only band    : 0.383 vs true 0.115  -> 3.34x over
            T5 BOTH sides (correct) : 0.113 vs true 0.115  -> 1.66% error

        Cause: particles are scraped just INSIDE the surface (measured d/h
        between -0.088 and -0.016), where the stored psi is exactly 1.0 by
        Dirichlet BC. A trilinear gather there is dominated by those covered
        nodes. The T5 band, probed over d/h in (0.15, 2.0], never covers the
        actual impact sites.

        Booking q_p * Psi_k zeroes the residual BY CONSTRUCTION for whatever
        Psi is supplied, so a wrong table is a wrong-input problem rather than
        a wrong-method one -- but enabling one would over-correct by 3-8x,
        which is worse than not correcting at all. The two-sided table fixes
        it because covered nodes hold psi = 1 in the stored Dirichlet basis
        while the weighting an absorbed particle should contribute there is
        ~0, and the trilinear gather reaches those nodes from every real
        scrape position.

        WHICH Psi THIS GATHERS
        ----------------------
        This gathers ``psi_unit_k``, the stored PLAIN (Dirichlet, "electrode k
        at 1 V") basis. That basis is ~1 at its own electrode BY BOUNDARY
        CONDITION, so it reports a small deficit no matter what the truth is.
        T4 measured an implied <Psi> ~ 0.94 at the impact sites and a
        mis-booking rate of 6.6-6.8%, whereas the weighting potential MEASURED
        through WarpX's own charge identity is 0.18-0.26 within one cell of the
        boundary, implying ~76-85%.

        So the ledger arithmetic below is correct and the machinery works, but
        with the plain basis it under-books the correction by roughly an order
        of magnitude. This is exactly the failure mode
        `reviews/check_absorption_deficit_measured.py` (K2) predicted. Booking
        q_p * Psi_k still zeroes the residual BY CONSTRUCTION for whatever Psi
        is supplied -- so this is not wrong, it is solving the wrong problem
        precisely. Supplying the measured/adjoint Psi is what makes it solve
        the intended one; see ROADMAP item A.

        Cost is O(N_e) multiply-adds per absorbed particle -- no solve. The
        correction is a linear functional of the absorbed charge and
        superposition over impacts is exact (verified in WarpX itself), which
        is why this scales to millions of impacts per step.

        REVIEW FIX 2026-08-13 -- MPI correctness (four items)
        ------------------------------------------------------
        A completed review found this method silently wrong under >1 MPI
        rank, in a way no single-rank test could catch. Fixed here:

        1. CURSOR/BUFFER-SIZE MISMATCH. The per-species cursor ``seen`` is
           necessarily PER-RANK state (each rank only ever reads its own tile
           arrays from ``get_particle_boundary_buffer``), but the buffer size
           used to decide whether there is anything new was previously the
           GLOBAL (MPI-reduced, ``local=False``) count. On >=2 ranks each
           rank's local arrays are shorter than the global total, so the
           ``total <= seen`` guard could silently skip whole batches on every
           rank but one. Fixed by sizing with ``local=True``: the cursor now
           compares like with like (this rank's count vs. this rank's cursor).

        2. RANK-LOCAL TOTALS. ``_absorb_booked``/``_absorb_deficit``/
           ``_absorb_counts`` (and ``_impact_histogram``) are accumulated
           PER RANK -- each rank only ever sees the particles scraped into
           its own tiles. That is correct for incremental accumulation (do
           not try to make the running accumulators global), but it means
           reading the raw attributes directly on >1 rank gives a partial
           sum. Use ``absorption_totals()``/``absorption_report()`` (both
           default to ``reduce=True``, an MPI-summed COLLECTIVE call every
           rank must make together) to get the run-wide ledger.
           ``_absorb_history`` entries are recorded RANK-LOCAL (documented on
           the attribute); reduce a history entry yourself if you need a
           global time series.

        3. STRUCK-ELECTRODE ATTRIBUTION. ``argmax(psi)`` identifies the
           struck electrode by largest gathered psi, which only means
           anything for the plain Dirichlet basis (psi ~ 1 on its own
           electrode by boundary condition, close to 0 elsewhere). For a
           custom ``psi_table`` (measured/adjoint) or ``gather_mode=
           "deposit"``, psi is not ~1 at the struck surface and argmax is
           meaningless. Fixed by attributing geometrically (nearest
           electrode surface) whenever ``electrode_centers`` was supplied to
           the constructor; ``argmax(psi)`` remains the default ONLY for the
           plain basis with no override, and a clear error is raised if a
           custom table/gather_mode is used without ``electrode_centers``.
           Note: ``booked[k, s]`` does NOT depend on ``struck`` (it sums
           ``q_p * psi_k`` over every impact regardless of attribution) --
           only ``deficit[k, s]`` uses it, unchanged.

        4. CURSOR VALIDITY. A buffer that SHRINKS between calls (regrid /
           load-balance clearing or redistributing the scrape buffer) used to
           be silently treated as "nothing new" by the old ``total <= seen``
           guard, which also hid the shrink. Now a shrink raises immediately
           with a message pointing at ``reset_absorption_cursors()``. Load
           balancing / regrid is NOT otherwise supported by this ledger: the
           cursor's ordering assumption (this rank's buffer only grows,
           never reorders) can be violated by a regrid in ways this check
           cannot detect in general (e.g. a redistribution that changes size
           by coincidence). Call ``reset_absorption_cursors()`` yourself
           around any regrid if you use this ledger with AMR/load balancing
           enabled; note that doing so cannot recover particles scraped
           between the last successful read and the regrid.
        """
        import numpy as np  # noqa: PLC0415

        if not self.book_absorption or not self._psi_stored:
            return None
        from pywarpx.particle_containers import (  # noqa: PLC0415
            ParticleBoundaryBufferWrapper,
        )

        buf = ParticleBoundaryBufferWrapper()
        ns = len(self.absorption_species)
        if self._absorb_deficit is None:
            self._absorb_deficit = np.zeros((self.n, ns))
            self._absorb_booked = np.zeros((self.n, ns))
            self._absorb_counts = np.zeros(ns, dtype=np.int64)

        # Prefetch every electrode's psi table UNCONDITIONALLY, before any
        # rank-locally-conditioned skip logic below. When there is no
        # psi_override, _nodal_psi(k) triggers a collective MPI allgather the
        # FIRST time it is called for a given k (see its docstring) and caches
        # the result forever after (psi_unit_k never changes post-setup). If
        # that first call were instead reached only from inside the
        # per-species "do I have new data" branch, two ranks with different
        # local scrape timing could diverge on whether they call it at all --
        # deadlock. This loop runs identically on every rank every call
        # (accumulate_absorption() itself is invoked in lockstep by the
        # step callback), so the collective, when it happens, is symmetric.
        for k in range(self.n):
            self._nodal_psi(k)

        for s, sp in enumerate(self.absorption_species):
            try:
                # local=True: the cursor below is inherently PER-RANK state
                # (each rank only ever reads its own tile arrays), so it must
                # be compared against this rank's local buffer size, not the
                # global MPI-reduced count (see fix note 1 above).
                total = buf.get_particle_boundary_buffer_size(sp, "eb", local=True)
            except Exception:
                continue
            seen = self._buffer_cursor.get(sp, 0)
            if total < seen:
                raise RuntimeError(
                    f"EB scrape buffer for species '{sp}' on this rank shrank "
                    f"from {seen} to {total} particles between calls to "
                    "accumulate_absorption(). This means the buffer was "
                    "cleared or reordered (e.g. by a regrid / load-balance "
                    "step) -- the per-rank cursor's ordering assumption is "
                    "violated and load balancing is currently UNSUPPORTED by "
                    "this ledger. Call corrector.reset_absorption_cursors() "
                    "if you intend to resume accounting from here (this "
                    "cannot recover particles scraped between the last "
                    "successful read and the shrink)."
                )
            if total <= seen:
                continue

            def _cat(comp):
                arrs = buf.get_particle_boundary_buffer(sp, "eb", comp, 0)
                return (np.concatenate([np.asarray(a) for a in arrs])
                        if arrs else np.zeros(0))

            x, y, z, w = (_cat("x"), _cat("y"), _cat("z"), _cat("w"))
            # x/y/z/w are THIS RANK's tile arrays (never reduced), so their
            # length must equal the local `total` above -- if it does not,
            # `local=True` sizing and the tile-array read have gone out of
            # sync (e.g. a stale ParticleBoundaryBufferWrapper instance).
            assert len(x) == total, (
                f"species '{sp}': local buffer size {total} != concatenated "
                f"local tile-array length {len(x)}"
            )
            x, y, z, w = x[seen:], y[seen:], z[seen:], w[seen:]
            self._buffer_cursor[sp] = total

            q_sp = self._species_charge(sp)
            q_p = q_sp * w                       # macroparticle charge [C]

            psis = np.stack([self._gather_psi(k, x, y, z)
                             for k in range(self.n)], axis=0)   # (n, N)

            # Which electrode was struck? An impact perturbs EVERY conductor,
            # so all n rows of `deficit` are updated regardless -- cross-terms
            # are not negligible. `struck` only selects which electrode's row
            # gets the "(1 - psi)" (own-electrode) term vs. the "-psi"
            # (foreign-electrode) term; `booked` does not use `struck` at all.
            if self.electrode_centers is not None:
                # Geometric attribution: nearest electrode SURFACE (centre
                # distance minus radius; radius defaults to 0, i.e. nearest
                # centre). Valid for any psi table, since it does not depend
                # on psi's value or normalization.
                d = np.stack([
                    np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2)
                    - r
                    for c, r in zip(self.electrode_centers, self.electrode_radii)
                ], axis=0)
                struck = np.argmin(d, axis=0)
            elif self._psi_override is not None or self.gather_mode == "deposit":
                raise RuntimeError(
                    "accumulate_absorption() cannot attribute struck "
                    "electrodes: argmax(psi) is only meaningful for the "
                    "plain Dirichlet psi_unit_k basis (psi ~ 1 on its own "
                    "electrode by boundary condition), and this corrector is "
                    "using a custom psi_table and/or gather_mode='deposit', "
                    "whose surface values are not ~1 and vary. Pass "
                    "electrode_centers=[...] (and optionally electrode_radii="
                    "[...]) to the constructor for geometric attribution."
                )
            else:
                # Legacy fallback: only valid for the plain Dirichlet basis.
                struck = np.argmax(psis, axis=0)

            for k in range(self.n):
                self._absorb_booked[k, s] += float(np.sum(q_p * psis[k]))
                is_k = (struck == k)
                self._absorb_deficit[k, s] += float(
                    np.sum(q_p[is_k] * (1.0 - psis[k][is_k]))
                    - np.sum(q_p[~is_k] * psis[k][~is_k])
                )
            self._absorb_counts[s] += len(x)

            if self._impact_histogram_cap:
                room = self._impact_histogram_cap - len(self._impact_histogram)
                if room > 0:
                    take = min(room, len(x))
                    self._impact_histogram.extend(
                        zip(x[:take].tolist(), y[:take].tolist(),
                            z[:take].tolist(), [s] * take, q_p[:take].tolist())
                    )

        step = self._warpx().getistep(lev=0)
        # RANK-LOCAL. Each entry records THIS RANK's partial deficit/counts at
        # this step, not the MPI-summed run total -- reduce yourself (e.g.
        # with _mpi_sum) if you need a global time series from this history.
        self._absorb_history.append(
            (int(step), self._absorb_deficit.copy(), self._absorb_counts.copy())
        )
        return self._absorb_deficit

    def reset_absorption_cursors(self):
        """Reset the per-species EB-scrape-buffer read cursors.

        The cursor (``self._buffer_cursor``) assumes this rank's scrape
        buffer only ever grows and is never reordered between calls to
        ``accumulate_absorption()``. A regrid / load-balance step can violate
        that (the buffer may be cleared or redistributed across ranks), which
        ``accumulate_absorption()`` detects as a same-species buffer SHRINK
        and raises on. Call this method to resume accounting after such an
        event.

        Load balancing / regrid is otherwise UNSUPPORTED by this ledger: this
        only resets the read position, it does not attempt to recover or
        re-attribute particles scraped between the last successful read and
        the event that invalidated the cursor, and a redistribution that
        happens not to shrink the local buffer (e.g. a regrid that changes
        which particles land on this rank without reducing the count) will
        not be detected at all. Treat this ledger as valid only for AMR-off,
        load-balancing-off, single-static-grid runs, or call this method
        defensively around every regrid if you use it otherwise.
        """
        self._buffer_cursor = {}

    def _mpi_sum(self, arr):
        """MPI-SUM ``arr`` (numpy array or scalar) across ranks; identity if
        mpi4py is unavailable or there is one rank. COLLECTIVE -- every rank
        must call this together."""
        import numpy as np  # noqa: PLC0415

        try:
            from mpi4py import MPI  # noqa: PLC0415
        except ImportError:
            return arr
        comm = MPI.COMM_WORLD
        if comm.Get_size() <= 1:
            return arr
        arr = np.asarray(arr)
        out = np.empty_like(arr)
        comm.Allreduce(np.ascontiguousarray(arr), out, op=MPI.SUM)
        return out.reshape(arr.shape)

    def absorption_totals(self, reduce=True):
        """The (electrode x species) absorption ledger, MPI-summed by default.

        ``_absorb_booked``/``_absorb_deficit``/``_absorb_counts`` are
        accumulated PER RANK (see ``accumulate_absorption``'s fix-note 2):
        correct as running accumulators, but a partial sum on every rank but
        one if read directly on >1 rank. This method returns the run-wide
        total.

        Parameters
        ----------
        reduce : bool, optional
            If True (default), MPI-sum across ranks -- a COLLECTIVE call
            every rank must make together (in lockstep with any other
            collective calls this corrector makes, e.g. accumulate_absorption
            itself). If False, return this rank's partial sums only (e.g. to
            inspect a load imbalance).

        Returns
        -------
        dict with keys "booked", "deficit", "counts" (numpy arrays), or None
        if ``accumulate_absorption()`` has not yet run.
        """
        if self._absorb_deficit is None:
            return None
        booked = self._absorb_booked.copy()
        deficit = self._absorb_deficit.copy()
        counts = self._absorb_counts.copy()
        if reduce:
            booked = self._mpi_sum(booked)
            deficit = self._mpi_sum(deficit)
            counts = self._mpi_sum(counts)
        return {"booked": booked, "deficit": deficit, "counts": counts}

    def _species_charge(self, name):
        """Signed charge per unit weight [C] for a named species."""
        for sp in getattr(self.sim, "species", []) or []:
            if getattr(sp, "name", None) == name:
                q = getattr(sp, "charge", None)
                if isinstance(q, str):        # picmi accepts "q_e"/"-q_e"
                    return {"q_e": 1.602176634e-19,
                            "-q_e": -1.602176634e-19}.get(q, 0.0)
                if q is not None:
                    return float(q)
                ptype = getattr(sp, "particle_type", "") or ""
                if ptype == "electron":
                    return -1.602176634e-19
                if ptype in ("proton", "hydrogen"):
                    return 1.602176634e-19
        raise ValueError(
            f"cannot determine the charge of species '{name}'; pass an "
            "explicit picmi Species with .charge set so the absorption ledger "
            "is not silently signed wrong")

    def impact_map(self, electrode_centers=None, nbins=(72, 36)):
        """Per-species angular map of absorption events on each electrode.

        Returns a dict with a (n_species, ntheta, nphi) CHARGE histogram and a
        matching COUNT histogram per electrode, binned on the sphere of
        directions around each electrode centre.

        WHY CHARGE AND COUNT SEPARATELY. They answer different questions and
        can look completely different. Count density drives surface effects
        that scale with particle flux -- sputtering, secondary emission, heat
        load. Charge density drives the electrical asymmetry: in a
        multi-species device the two species deposit opposite signs, so a
        region with high count density can carry near-zero net charge, and a
        region with modest counts can dominate the dipole if one species is
        absent there. Reporting only one of them hides the other.

        The binning is angular rather than Cartesian because the electrodes are
        closed surfaces: (theta, phi) around the centre covers the surface
        exactly once with no empty cells, which a Cartesian slab grid would
        not.

        Requires `impact_histogram_cap > 0` at construction; the map is built
        from the retained sample, so with a cap smaller than the impact count
        it is a SUBSAMPLE and the absolute normalisation is not meaningful.
        `saturated` in the return says whether that happened.

        RANK-LOCAL. ``self._impact_histogram`` is filled per rank (each rank
        only ever sees the particles scraped into its own tiles, same as the
        absorption ledger -- see ``accumulate_absorption``) and is never
        reduced. On >1 rank this therefore maps only the calling rank's
        sample, not the whole-domain distribution; there is no gather here
        (out of scope for this fix).
        """
        import numpy as np  # noqa: PLC0415

        if not self._impact_histogram:
            return None
        H = np.array(self._impact_histogram, dtype=float)
        x, y, z, sidx, q = H[:, 0], H[:, 1], H[:, 2], H[:, 3].astype(int), H[:, 4]

        if electrode_centers is None:
            electrode_centers = self.electrode_centers  # constructor-supplied, if any
        if electrode_centers is None:
            raise ValueError(
                "electrode_centers is required: the map is angular about each "
                "electrode centre, and there is no way to infer those from the "
                "region expressions. Pass it here, or to the constructor.")

        nth, nph = nbins
        ns = max(len(self.absorption_species), 1)
        out = {"species": list(self.absorption_species),
               "electrodes": list(self.names),
               "nbins": [int(nth), int(nph)],
               "saturated": bool(self._absorb_counts is not None
                                 and int(self._absorb_counts.sum())
                                 > len(self._impact_histogram)),
               "n_sampled": int(len(H)),
               "maps": []}

        cen = np.asarray(electrode_centers, dtype=float)
        # assign each impact to its nearest electrode centre
        d = np.stack([np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2
                              + (z - c[2]) ** 2) for c in cen], axis=0)
        owner = np.argmin(d, axis=0)

        for k in range(len(cen)):
            m = owner == k
            dx_, dy_, dz_ = x[m] - cen[k][0], y[m] - cen[k][1], z[m] - cen[k][2]
            r = np.sqrt(dx_ ** 2 + dy_ ** 2 + dz_ ** 2)
            r[r == 0] = 1.0
            theta = np.arccos(np.clip(dz_ / r, -1, 1))      # [0, pi]
            phi = np.arctan2(dy_, dx_)                       # (-pi, pi]
            th_e = np.linspace(0, np.pi, nth + 1)
            ph_e = np.linspace(-np.pi, np.pi, nph + 1)
            chg = np.zeros((ns, nth, nph))
            cnt = np.zeros((ns, nth, nph))
            for s in range(ns):
                sm = sidx[m] == s
                if not sm.any():
                    continue
                cnt[s], _, _ = np.histogram2d(theta[sm], phi[sm],
                                              bins=[th_e, ph_e])
                chg[s], _, _ = np.histogram2d(theta[sm], phi[sm],
                                              bins=[th_e, ph_e],
                                              weights=q[m][sm])
            out["maps"].append({"electrode": self.names[k],
                                "center": cen[k].tolist(),
                                "charge": chg.tolist(),
                                "count": cnt.tolist(),
                                "theta_edges": th_e.tolist(),
                                "phi_edges": ph_e.tolist()})
        return out

    def absorption_report(self, reduce=True):
        """Return the (electrode x species) ledger as a plain dict.

        Parameters
        ----------
        reduce : bool, optional
            If True (default), the returned matrices are MPI-summed across
            ranks via ``absorption_totals()`` -- a COLLECTIVE call every rank
            must make together (harmless/identity on a single rank, which is
            why this default does not change the recorded single-rank
            results). Pass False for this rank's partial sums only.
        """
        import numpy as np  # noqa: PLC0415

        totals = self.absorption_totals(reduce=reduce)
        if totals is None:
            return None
        deficit, booked, counts = totals["deficit"], totals["booked"], totals["counts"]
        return {
            "electrodes": list(self.names),
            "species": list(self.absorption_species),
            "deficit": deficit.tolist(),
            "booked": booked.tolist(),
            "counts": counts.tolist(),
            "deficit_total": float(np.sum(deficit)),
            "deficit_abs_total": float(np.sum(np.abs(deficit))),
        }

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

        if self._psi_stored:
            # EXACT PATH. psi_k was stored at setup as the solved potential of
            # the same Laplace problem that produced E_0k, so it is read
            # directly, on the same nodal grid as rho -- no line integral, no
            # anchor choice, no collocation shift, and no cut-cell
            # accumulation. psi_k and rho are both nodal (n+1)^3, so the
            # product is pointwise-aligned by construction.
            for k in range(self.n):
                psi_k = np.asarray(
                    mfr.get(self._psi_names[k], level=lev)[:, :, :]
                )
                if psi_k.shape != rho.shape:
                    raise RuntimeError(
                        f"psi_{k} shape {psi_k.shape} != rho shape "
                        f"{rho.shape}; the stored weighting potential and the "
                        "charge density must share the nodal grid."
                    )
                q_g[k] = -np.sum(rho * psi_k) * dV
            return q_g

        # FALLBACK PATH (legacy). Reached only when the WarpX build does not
        # publish phi_fp from SolvePoissonEfield. Retained so the diagnostic
        # still returns a value on older builds, but it carries the
        # reconstruction error documented above: the cumulative sum is a
        # path-dependent line integral whose cut-cell defects offset the entire
        # downstream ray, AND psi_k[1:] = -cumsum(Ex0) places psi_k half a cell
        # from rho. Treat its output as an order-of-magnitude cross-check only.
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
        """Return the present per-electrode effective voltages V = C^-1 (Q - Q_g).

        ABSORPTION LEDGER FEEDBACK (``apply_ledger_correction=True``)
        ---------------------------------------------------------------
        WHY AN UNCORRECTED ABSORPTION EVENT MOVES THE MEASURED VOLTAGE, EVEN
        THOUGH THE PHYSICS SAYS IT SHOULD NOT. Per
        ``reference/absorption_potential_maintenance.md`` Section 1, an ideal
        absorption event produces no potential transient: the induced charge
        has tracked the particle continuously all the way in, so at contact
        nothing changes. Section 2 writes this as an identity per electrode k,

            Q_k^ext = (C V)_k - sum_p q_p Psi_k(x_p) - Q_k^stuck,

        i.e. the live-particle sum and the already-stuck charge trade off
        exactly as a particle is absorbed. The bug is NOT that this codebase
        assumes the wrong physics for Q_k^stuck -- it is that the two halves
        of ``measure_voltages()`` do not update in step:

          * ``q_now`` is a flux read of ``Efield_fp``, the ADVANCED EM/ECT
            field. It only changes when the field solver actually advances
            (or is explicitly re-solved) -- removing a particle from the
            particle container does not, by itself, touch it.
          * ``q_g`` is a FRESH grounded Poisson resolve using whatever ``rho``
            exists AT THIS INSTANT. It has no memory at all: the moment an
            absorbed particle is gone from every particle container, this
            resolve stops seeing it, by the reciprocity identity
            ``Q_g,k = -sum_p q_p Psi_k(x_p)`` (``measure_grounded_charge_
            reciprocity``'s formula; an exact grounded resolve reproduces it
            for whatever Psi the underlying operator implies -- see Section
            3b on why that Psi must be the ADJOINT for this non-symmetric
            cut-cell operator, which is exactly the table ``load_psi_table()``
            is for).

        So absorbing a particle (or, in a static test, simply deleting it
        from the domain) leaves ``q_now`` untouched but drops ``q_g,k`` by
        ``q_p * Psi_k(x_a)`` relative to an instant ago, for EVERY electrode k
        (not just the one struck -- Psi_k(x_a) is generally nonzero at every
        electrode; this is the same cross-talk fact used in ``accumulate_
        absorption()``'s own docstring, hypothesis 3). Since
        ``V = C^-1(q_now - q_g)``, that shows up as a spurious kick

            delta_V = +C^-1 [ q_p * Psi_k(x_a) ]_k = +C^-1 * booked[k]

        where ``booked[k] = sum_s self._absorb_booked[k, s]`` is exactly the
        ledger quantity ``accumulate_absorption()`` already accumulates (its
        docstring: ``booked[k, s] = sum_p q_p * Psi_k(x_p)``, summed over
        EVERY absorbed particle regardless of which electrode it struck).

        THE TERM AND SIGN, DETERMINED EMPIRICALLY (not derived -- per the
        implementation task, the derivation above only motivates the
        candidate; the acceptance test decided it). Add the ledger's
        ``booked`` matrix (summed over the species axis) to ``q_now`` BEFORE
        the inversion:

            Q_corr[k] = q_now[k] + sum_s booked[k, s]
            V = C^-1 (Q_corr - Q_g)

        Measured on the T8 fixture (two-sphere geometry, adjoint Psi via
        ``solve_adjoint_weighting(rhs_mode="charge", add_indicator=False)``,
        ``gather_mode="deposit"``, a probe added then deleted from the domain
        just outside a sphere): this reproduces the pre-absorption voltage to
        3-30 pV-scale (i.e. ~1e-11 V), 5+ orders of magnitude inside the 1e-6 V
        acceptance target, on BOTH electrodes even though the probe was near
        only one of them (confirming the cross-talk term matters). The three
        other sign/term combinations that could plausibly have been "the"
        correction were checked against the same numbers and every one of them
        failed by 4-6 orders of magnitude more:

            +booked  : PASS (residual ~1e-11 V)
            -booked  : fails (residual ~0.05-0.16 V)
            +deficit : fails (residual ~0.002-0.46 V)
            -deficit : fails (residual ~0.05-0.63 V)

        where ``deficit[k] = sum_s self._absorb_deficit[k, s]`` is the OTHER
        ledger matrix. ``deficit`` answers a different question (see
        ``accumulate_absorption()``'s docstring: it is ``Q_naive_stuck_k -
        booked[k]``, the gap between "book the raw q_p onto the electrode a
        struck-attribution heuristic assigns it to" and the correct Ramo-
        weighted booking) and is the right instrument for auditing that
        gap -- but it is not the quantity that reconciles ``q_now`` and
        ``q_g`` here, because ``measure_voltages()`` never books a raw q_p
        onto any electrode in the first place; the mismatch above comes
        purely from ``q_g``'s amnesia, and its exact remedy is ``booked``,
        unconditionally, on every electrode (own and foreign alike -- there is
        no "struck" split in the correction term at all, matching that
        ``accumulate_absorption()`` computes ``booked[k, s]`` the same way for
        every k regardless of ``struck``).

        SCOPE. The T8 acceptance fixture deletes the probe from the domain
        programmatically (``particle_container.clear_particles()``) rather
        than physically pushing it across the boundary, so it is scraped
        (mathematically) exactly where placed -- e.g. just OUTSIDE the true
        surface, same as T6. A real absorbing-boundary scrape lands just
        INSIDE instead. This does not weaken the result: the correction is a
        discrete IDENTITY (Section 3a of the report), exact for whatever
        position and Psi value the ledger actually gathers, not a property of
        being near the surface -- T6's mechanism only fixes the ledger's
        *input* (where psi is sampled), never the *arithmetic* this method
        adds. It is used here because it keeps the test static (``max_steps
        =0``, no field-advance noise), not because a real scrape needs a
        different formula.

        CAVEAT NOT COVERED. The ledger is a cumulative, never-reset running
        total (``accumulate_absorption()``'s cursor only grows). In a real
        run where ``correct_field()`` re-biases the field every
        ``correction_interval`` steps, adding the FULL history's ``booked``
        every time double-counts against a field that has already
        absorbed an earlier correction. This static acceptance test does not
        exercise that path (a single absorption event, measured once); fixing
        it (e.g. by having ``correct_field()`` consume/zero the ledger it
        just applied) is out of scope here and left as a follow-on.

        Q_g VIA RECIPROCITY (``qg_mode="reciprocity"``) -- SKIPPING THE SOLVE
        -----------------------------------------------------------------------
        The default ``qg_mode="grounded"`` gets ``Q_g`` by literally re-posing
        the grounded problem: save ``Efield_fp``, force every electrode to
        0 V, run a full MLMG Poisson solve, read the flux, restore the live
        field. That is exact but it is the dominant cost of every call to
        this method (a Poisson solve every correction interval).

        ``qg_mode="reciprocity"`` instead evaluates the SAME quantity through
        the discrete identity this whole module is built on (Section 3a of
        the report): for a grounded resolve (``V = 0``), ``Q_k = (C*0)_k -
        sum_node rho_node * Psi_k(node) = -sum_node rho_node * Psi_k(node)``.
        Discretized on the nodal grid with cell volume ``dV``,

            Q_g,k = -sum_a rho_a * dV * Psi_k[a]

        which is exactly ``measure_grounded_charge_reciprocity()``'s formula
        (see that method for the rho-access/deposit pattern this mirrors),
        evaluated here against the ADJOINT Psi (the same table
        ``load_psi_table()`` supplies for the absorption ledger) rather than
        being offered as a separate opt-in cross-check. Requires the ADJOINT
        table: the ordinary/plain ``psi_unit_k`` is the wrong basis for a
        non-symmetric cut-cell operator (Section 3b), so ``qg_mode=
        "reciprocity"`` raises immediately if ``load_psi_table()`` has not
        been called. Uses the RAW (unfiltered) nodal Psi table -- the SAME
        one ``accumulate_absorption()`` would use with ``gather_mode="node"``
        -- and NOT the ``gather_mode="deposit"`` binomial-filtered table
        ``_psi_for_gather()`` builds for particle-position gathers: ``rho_a``
        here is already the deposited (filter-inclusive where WarpX applies
        one) nodal density on the mesh, i.e. the dot product's rho side has
        already been through whatever filtering the deposit performs, so
        filtering Psi again on top would double-apply it.

        MPI DISCIPLINE. ``mpc.get_charge_density(lev, False)[:, :, :]``, like
        the psi-table reads elsewhere in this module, performs an MPI
        allgather in pyAMReX's ``MultiFab.__getitem__`` -- a COLLECTIVE call
        every rank must issue. It is invoked unconditionally whenever
        ``qg_mode="reciprocity"`` (no rank-locally-conditioned branch guards
        it), the same discipline ``accumulate_absorption()``'s psi prefetch
        documents. Because the allgather makes both ``rho`` and each
        ``Psi_k`` table full GLOBAL arrays identically on every rank, the
        dot product ``sum(rho * psi_k)`` is computed redundantly but
        IDENTICALLY on every rank -- it is already rank-symmetric and needs
        no further MPI reduction afterward.

        MEASURED ACCURACY -- VALIDATED, AFTER A RHO-SOURCE FIX
        -----------------------------------------------------------------
        HISTORY (read this before trusting any number that predates it).
        The first implementation of this mode dotted the adjoint Psi
        against ``mpc.get_charge_density(lev, False)`` and disagreed with
        ``"grounded"`` by 1.4%-99.8% relative on a pure-screening negative
        control (a probe in flight, no absorption, ledger cleared, T8's
        ``negative_control()``) -- nowhere near the ~1e-6 an earlier
        in-process check had suggested. That gap was checked and was NOT
        floor noise (unchanged to 4+ digits under a 1e4x tighter adjoint
        solve tolerance) and NOT a missing unit/scale factor (a 9-way sweep
        of ``* or / by dV`` and ``* or / by eps0`` found no combination
        closer than the as-implemented ``* dV``). Restricting the dot
        product to only ``rho != 0`` nodes reproduced the full-array sum
        exactly, ruling out far-field contamination, so the gap was real
        and concentrated at the near-probe nodes themselves.

        ROOT CAUSE, FOUND AND FIXED. ``mpc.get_charge_density()`` deposits
        into a fresh, UNREGISTERED temporary MultiFab (a raw per-species CIC
        deposit plus an MPI ``SumBoundary`` -- NO filter). The Poisson solve
        itself consumes the REGISTERED ``rho_fp``, which goes through
        ``sync_rho()`` and DOES receive WarpX's default single-pass
        binomial filter. Confirmed directly (same probe, same node,
        matching an earlier session's node-probe diagnostic exactly):
        applying this class's own ``_binomial_filter_3d`` to the
        ``get_charge_density()`` array reproduced the registered
        ``rho_fp`` array to EXACT floating-point equality (max abs diff
        0.0). Against a Psi that varies by O(1) within one cell of the EB,
        dotting the unfiltered array produces exactly the failure signature
        above: small error far from the probe, large position-dependent
        error near it, invariant under solver tolerance or any global
        scale factor -- because a missing filter is a local smoothing
        kernel, not a constant, and no constant can undo it. THE FIX: this
        method now calls ``_deposit_and_read_rho_fp()``, which deposits
        every species into and reads back the registered ``rho_fp`` (see
        its own docstring for the multi-species accumulation-then-filter
        ordering and the MPI discipline).

        VALIDATED NUMBERS, AFTER THE FIX (T8 fixture, two-sphere geometry).
        At the exact node used to diagnose the bug, this method now
        reproduces the grounded-solve ``Q_g`` to a relative error of
        4.0e-10 (was 7.2e-2 with the unfiltered array). Across the 8 T8
        probe events (pure screening, probe in flight, nonzero rho -- the
        state that actually exercises this method, unlike the post-removal
        "corrected voltage residual" comparison below), the relative
        difference from ``"grounded"`` is 4.0e-10 to 4.2e-8 per electrode.
        The dedicated negative control agrees to 4.9e-11 to 1.3e-9 relative
        in ``Q_g`` and 9.4e-12 to 4.1e-11 V in the resulting voltage.
        ``qg_mode="reciprocity"`` is therefore VALIDATED as an accurate,
        much faster alternative to ``"grounded"`` -- it remains non-default
        for now simply because it is new, not because of any known
        remaining accuracy gap.

        A SEPARATE, STILL-TRUE INSENSITIVITY, in the acceptance harness
        rather than this method: T8's per-event "corrected voltage residual
        in reciprocity mode" check cannot discriminate the two modes at all
        (in either the buggy or fixed state), because it measures ``Q_g``
        AFTER the probe has been deleted from the domain (``rho``
        identically zero), where both modes trivially return ~0 regardless
        of Psi accuracy. The pure-screening negative control and the
        per-event Q_g-both-modes comparison (nonzero ``rho``) are the checks
        that actually exercise this method, and both are the ones reporting
        agreement above.

        COST. Timed over 20 repeated ``measure_voltages()`` calls (fixed
        vacuum state, no absorption): measured 10x faster than
        ``"grounded"`` on the T8 fixture (0.62 ms/call vs. 6.31 ms/call) --
        ``"reciprocity"`` pays one charge deposit + filter/exchange pass
        where ``"grounded"`` pays a full MLMG solve, so the gap should
        widen on larger grids for the same reason ``measure_grounded_
        charge_reciprocity()``'s docstring gives for its own timing
        comparison.
        """
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        lev = 0

        # Live-field induced charge per electrode.
        q_now = np.array(
            [warpx.compute_eb_charge(weighting=r, field="Efield_fp") for r in self.regions]
        )

        if self.apply_ledger_correction:
            # Keep the ledger current at measurement time. COLLECTIVE: every
            # rank must reach this the same number of times in the same order
            # (accumulate_absorption() itself prefetches every electrode's
            # psi table unconditionally before any rank-locally-conditioned
            # skip logic, for exactly this reason -- see its docstring). Since
            # __init__ refuses apply_ledger_correction=True without
            # book_absorption=True, this is never a silent no-op by
            # construction, though accumulate_absorption() can still no-op
            # per-call if there is nothing new to read.
            self.accumulate_absorption()
            totals = self.absorption_totals(reduce=True)   # also collective
            if totals is not None:
                q_now = q_now + totals["booked"].sum(axis=1)

        if self.qg_mode == "grounded":
            q_g = self._grounded_charge_via_solve(lev)
        else:  # "reciprocity" -- validated at construction to be one or the other
            q_g = self._grounded_charge_via_reciprocity(lev)

        return np.linalg.solve(self._capacitance, q_now - q_g)

    def _grounded_charge_via_solve(self, lev):
        """``qg_mode="grounded"``: the original save/solve/restore Q_g.

        Save/restore the live field around a real grounded Poisson solve --
        exact, at the cost of one MLMG solve per call. See
        ``measure_voltages()``'s docstring for the ``qg_mode="reciprocity"``
        alternative this is compared against.
        """
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        saved = self._save_efield(lev)
        warpx.set_potential_on_eb("0.0")
        warpx.solve_poisson_efield()
        q_g = np.array(
            [warpx.compute_eb_charge(weighting=r, field="Efield_fp") for r in self.regions]
        )
        self._restore_efield(saved, lev)
        warpx.set_potential_on_eb(self.potential_expression)
        return q_g

    def _ensure_rho_fp(self, lev):
        """Allocate the registered ``rho_fp`` MultiFab if it does not already
        exist, with enough ghost cells for a real per-species charge deposit
        (``ParticleContainerWrapper.deposit_charge_density()``).

        4 ghost cells, matching the working diagnostic this was validated
        against (``scratchpad/probe_node_full.py``'s ``ensure_field(...,
        ngrow=4)``): fewer raises ``DepositCharge.H``'s "num_rho_deposition_
        guards are larger than allocated!" assertion (an unrecoverable
        MPI_Abort, confirmed empirically) rather than a catchable Python
        exception, so this pads generously instead of trying to compute the
        exact minimum. This value is COPIED from that diagnostic, not
        derived from WarpX's own ``noz``/``ng_rho`` guard-cell accounting --
        a future change to the particle shape order or filter pass count
        could invalidate it silently; if ``_deposit_and_read_rho_fp()``
        starts hitting the same assertion, this is the first place to look.

        TWO COMPONENTS -- NOT ONE. Merely registering ``rho_fp`` flips on
        ``has_rho`` in ``PhysicalParticleContainer::PushPX``
        (``has_rho = fields.has(FieldType::rho_fp, lev)``), which makes
        WarpX's OWN post-push deposit unconditionally write into
        **component 1** of ``rho_fp`` every step from then on
        (``PhysicalParticleContainer.cpp``, "Deposit charge after particle
        push, in component 1 of MultiFab rho"), guarded by
        ``WARPX_ALWAYS_ASSERT_WITH_MESSAGE(rho->nComp() >= 2, ...)`` --
        an unrecoverable MPI_Abort, not a catchable Python exception, the
        first time any particle pushes after this MultiFab is registered
        with only 1 component. This was invisible to every earlier
        validation of ``qg_mode="reciprocity"`` (T7, T8) because those are
        static ``max_steps=0`` fixtures -- the post-push deposit never runs.
        It reproduces immediately (step 2) in any REAL time-stepping run
        with this qg_mode enabled (confirmed on the four-arm campaign
        fixture, nx=32 and nx=80 alike -- resolution-independent). Matches
        WarpX's own convention when it decides to allocate ``rho_fp`` itself
        (``WarpX.cpp``: ``rho_ncomps = 2*ncomps`` whenever ``do_dive_cleaning``),
        so allocate 2 components here too. ``_deposit_and_read_rho_fp()``
        below explicitly reads back only component 0 -- the one this
        class's own per-species deposit writes into (see its docstring) --
        component 1 is WarpX's own post-push bookkeeping, never read by this
        class, and safe to leave in whatever state the last push left it.
        """
        mfr = self._mfr()
        try:
            mfr.get("rho_fp", level=lev)
            return
        except Exception:
            pass
        from pywarpx._libwarpx import libwarpx  # noqa: PLC0415

        eref = mfr.get("Efield_fp", dir=self._Direction(0), level=lev)
        nba = eref.box_array().surroundingNodes()
        ng = libwarpx.amr.IntVect(4)
        mfr.alloc_init("rho_fp", lev, nba, eref.dm(), 2, ng, 0.0, True, True)

    def _deposit_and_read_rho_fp(self, lev):
        """Deposit EVERY species' charge into the registered ``rho_fp`` and
        return it as a numpy array -- the SAME filtered nodal density
        ``solve_poisson_efield()`` itself consumes as its RHS, unlike
        ``mpc.get_charge_density()`` (see below).

        WHY NOT ``mpc.get_charge_density()``. That method (used by
        ``measure_grounded_charge_reciprocity()``) deposits into a fresh,
        unregistered temporary MultiFab via ``MultiParticleContainer::
        GetChargeDensity`` -- a raw per-species CIC deposit plus an MPI
        ``SumBoundary``, with NO filter applied. WarpX's actual charge
        deposit into ``rho_fp`` (what the Poisson solve consumes) goes
        through ``sync_rho()``, which DOES apply the default single-pass
        separable binomial filter (``warpx.use_filter=1``, the same one
        ``_binomial_filter_3d``/``gather_mode="deposit"`` reproduce for
        particle-position gathers elsewhere in this class). Confirmed
        empirically (single probe, T8 geometry, node matching an earlier
        session's ``node_d10.json`` diagnostic exactly): filtering the
        ``get_charge_density()`` array with ``_binomial_filter_3d`` matches
        this method's ``rho_fp`` array to EXACT floating-point equality
        (max abs diff 0.0) -- confirming the one-filter-stage gap is the
        entire difference between the two rho sources. Against a Psi that
        varies by O(1) within one cell of the EB (Section 3b), dotting the
        UNFILTERED array produced exactly the failure signature measured
        before this fix: small error far from the probe, large
        position-dependent error near it, unchanged by solver tolerance or
        any unit-scale factor -- because the missing filter is a local
        smoothing kernel, not a global constant, so no scale factor could
        have fixed it. Switching to this method's ``rho_fp`` reproduced the
        grounded-solve ``Q_g`` to a relative error of 4.0e-10 at that same
        node (vs. 7.2e-2 with the unfiltered array) -- see
        ``measure_voltages()``'s docstring for the full before/after
        numbers.

        MULTI-SPECIES ACCUMULATION. The filter is linear but NOT idempotent
        under repeated partial application, so every species must be
        deposited (accumulated, ``clear_rho`` only on the first) BEFORE the
        single ``sync_rho()`` filter/exchange pass -- filtering after each
        species and continuing to add more would NOT equal filtering the
        completed sum once.

        COLLECTIVE. Depositing is a local per-tile operation, but
        ``sync_rho()`` performs the guard-cell MPI exchange, and reading the
        result back via ``mf[:, :, :, 0]`` performs the same MPI allgather as
        every other MultiFab read in this module -- called unconditionally
        on every rank, exactly like the psi prefetch in
        ``accumulate_absorption()``.

        COMPONENT 0, EXPLICITLY. ``rho_fp`` has 2 components (see
        ``_ensure_rho_fp()``'s docstring for why 1 is not enough once this
        method is used in a real time-stepping run): this class's own
        per-species deposit above always targets component 0 (the pybind
        ``deposit_charge`` wrapper hardcodes ``icomp=0``); component 1 is
        WarpX's own post-push bookkeeping deposit, unrelated to this
        identity. Reading ``mf[:, :, :]`` (3 indices) on a 2-component
        MultiFab returns BOTH components stacked (pyAMReX pads the missing
        trailing index to a full component slice), which would silently
        change this array's shape from the nodal ``(nx, ny, nz)`` every
        caller here expects to ``(nx, ny, nz, 2)`` -- caught loudly by
        ``_grounded_charge_via_reciprocity()``'s shape guard against
        ``psi_k`` rather than corrupting the dot product, but avoided
        entirely by indexing the component explicitly below.

        SIDE EFFECT ON ``rho_fp`` -- CHECKED, HARMLESS FOR THIS CLASS'S OWN
        USE. This method takes ownership of the registered ``rho_fp``: it
        zeroes it (on the first species) and overwrites it with the current
        deposit every call. Verified this does not perturb ``qg_mode=
        "grounded"``: measuring voltages in ``"grounded"`` mode, then in
        ``"reciprocity"`` mode (which rewrites ``rho_fp``), then in
        ``"grounded"`` mode again reproduces the first ``"grounded"`` result
        exactly -- ``solve_poisson_efield()`` deposits its own rho fresh on
        every call regardless of what this method left behind. NOT checked
        against any OTHER consumer of ``rho_fp`` outside this class (e.g. a
        diagnostic reading it via ``pywarpx.fields.RhoFPWrapper`` between
        ``measure_voltages()`` calls would see THIS method's last deposit,
        not necessarily the state a concurrent EM step would have left) --
        treat ``rho_fp`` as owned by whichever of {this method, the main PIC
        loop} last wrote it, not as a stable snapshot.
        """
        import numpy as np  # noqa: PLC0415
        from pywarpx.particle_containers import (  # noqa: PLC0415
            ParticleContainerWrapper,
        )

        self._ensure_rho_fp(lev)
        names = [
            getattr(sp, "name", None) for sp in (getattr(self.sim, "species", []) or [])
        ]
        names = [n for n in names if n]
        if not names:
            raise RuntimeError(
                "qg_mode='reciprocity' could not find any species on "
                "self.sim to deposit rho_fp from (self.sim.species is empty "
                "or unset)."
            )
        if not hasattr(self, "_species_pcw_cache"):
            self._species_pcw_cache = {}
        for i, name in enumerate(names):
            if name not in self._species_pcw_cache:
                # Constructed once per species and cached: ParticleContainer
                # Wrapper() prints a deprecation UserWarning on every
                # construction, and there is no reason to pay that (or the
                # attribute-lookup cost) again every measure_voltages() call.
                self._species_pcw_cache[name] = ParticleContainerWrapper(name)
            self._species_pcw_cache[name].deposit_charge_density(
                level=lev, clear_rho=(i == 0), sync_rho=False
            )
        self._warpx().sync_rho()   # single filter/exchange pass -- see above
        mfr = self._mfr()
        return np.asarray(mfr.get("rho_fp", level=lev)[:, :, :, 0])

    def _grounded_charge_via_reciprocity(self, lev):
        """``qg_mode="reciprocity"``: Q_g,k = -sum_a rho_a * dV * Psi_k[a].

        No Poisson solve, no save/restore of the live field -- see
        ``measure_voltages()``'s docstring for the derivation, the required
        adjoint Psi, the RAW-vs-deposit-filtered table choice, and the MPI
        discipline (this method is COLLECTIVE and must be called
        unconditionally on every rank, exactly like the psi prefetch in
        ``accumulate_absorption()``).
        """
        import numpy as np  # noqa: PLC0415

        if self._psi_override is None:
            raise RuntimeError(
                "qg_mode='reciprocity' requires adjoint Psi tables to be "
                "loaded via load_psi_table() before measure_voltages() is "
                "first called -- there is nothing to dot rho against "
                "without them, and the plain psi_unit_k basis is the wrong "
                "one for this identity on a non-symmetric cut-cell operator "
                "(Section 3b of the report). Call load_psi_table(...) (e.g. "
                "after a solve_adjoint_weighting pass, as T8's "
                "_build_adjoint() does) first, or use qg_mode='grounded'."
            )

        warpx = self._warpx()
        # COLLECTIVE (deposit + sync_rho's MPI exchange + the allgather in
        # reading the result back), called unconditionally -- see
        # _deposit_and_read_rho_fp's docstring for why this, and NOT
        # mpc.get_charge_density(), is the rho the identity needs.
        rho = self._deposit_and_read_rho_fp(lev)

        geom_data = warpx.Geom(lev=lev).data()
        dxs = geom_data.CellSize()
        dV = dxs[0] * dxs[1] * dxs[2]

        q_g = np.empty(self.n)
        for k in range(self.n):
            # RAW table (override if loaded, else the stored register) --
            # NOT _psi_for_gather()'s deposit-filtered version: rho above is
            # already the deposited (and now filtered) density, so filtering
            # Psi again would double-apply it. Also collective on first use
            # per k (cached thereafter) -- see _nodal_psi()'s own docstring.
            psi_k = self._nodal_psi(k)
            if psi_k.shape != rho.shape:
                raise RuntimeError(
                    f"psi_{k} shape {psi_k.shape} != rho shape {rho.shape}; "
                    "the loaded Psi table and the charge density must share "
                    "the nodal grid."
                )
            q_g[k] = -np.sum(rho * psi_k) * dV
        return q_g

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
