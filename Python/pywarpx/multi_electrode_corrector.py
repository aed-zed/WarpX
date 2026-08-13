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
        book_absorption=False,
        absorption_species=None,
        impact_histogram_cap=0,
        psi_table=None,
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
        return self

    def _gather_psi(self, k, x, y, z):
        """Trilinear gather of psi_k at particle positions.

        Uses the SAME shape function as the field gather (``particle_shape =
        "linear"``). That is not a detail: the bookkeeping identity holds
        exactly only when gather and deposit are adjoint operations, which for
        a shared shape function they are. Using a different (e.g. nearest-node)
        interpolation here would reintroduce an error of the same order as the
        one being removed.
        """
        import numpy as np  # noqa: PLC0415

        mfr = self._mfr()
        geom = self._warpx().Geom(lev=0).data()
        dx = np.array(geom.CellSize())
        lo = np.array(geom.ProbLo())
        if self._psi_override is not None:
            psi = self._psi_override[k]
        else:
            psi = np.asarray(mfr.get(self._psi_names[k], level=0)[:, :, :])

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
        STATUS 2026-08-13: the in-code adjoint route (`solve_adjoint_weighting`)
        is BOUND AND CALLABLE BUT DOES NOT CONVERGE -- residual stalls at
        2.0e-2, identical at 2000 and 20000 iterations, so the system is
        inconsistent as posed rather than slow. See
        reviews/check_adjoint_wiring_status.py. Do not use it. The working
        input is the probed table below.

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

        for s, sp in enumerate(self.absorption_species):
            try:
                total = buf.get_particle_boundary_buffer_size(sp, "eb")
            except Exception:
                continue
            seen = self._buffer_cursor.get(sp, 0)
            if total <= seen:
                continue

            def _cat(comp):
                arrs = buf.get_particle_boundary_buffer(sp, "eb", comp, 0)
                return (np.concatenate([np.asarray(a) for a in arrs])
                        if arrs else np.zeros(0))

            x, y, z, w = (_cat("x"), _cat("y"), _cat("z"), _cat("w"))
            if len(x) <= seen:
                continue
            x, y, z, w = x[seen:], y[seen:], z[seen:], w[seen:]
            self._buffer_cursor[sp] = total

            q_sp = self._species_charge(sp)
            q_p = q_sp * w                       # macroparticle charge [C]

            # Which electrode was struck? The one whose psi is largest at the
            # impact site. An impact perturbs EVERY conductor, so all n rows
            # are updated -- cross-terms are not negligible.
            psis = np.stack([self._gather_psi(k, x, y, z)
                             for k in range(self.n)], axis=0)   # (n, N)
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
        self._absorb_history.append(
            (int(step), self._absorb_deficit.copy(), self._absorb_counts.copy())
        )
        return self._absorb_deficit

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
        """
        import numpy as np  # noqa: PLC0415

        if not self._impact_histogram:
            return None
        H = np.array(self._impact_histogram, dtype=float)
        x, y, z, sidx, q = H[:, 0], H[:, 1], H[:, 2], H[:, 3].astype(int), H[:, 4]

        if electrode_centers is None:
            raise ValueError(
                "electrode_centers is required: the map is angular about each "
                "electrode centre, and there is no way to infer those from the "
                "region expressions")

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

    def absorption_report(self):
        """Return the (electrode x species) ledger as a plain dict."""
        import numpy as np  # noqa: PLC0415

        if self._absorb_deficit is None:
            return None
        return {
            "electrodes": list(self.names),
            "species": list(self.absorption_species),
            "deficit": self._absorb_deficit.tolist(),
            "booked": self._absorb_booked.tolist(),
            "counts": self._absorb_counts.tolist(),
            "deficit_total": float(np.sum(self._absorb_deficit)),
            "deficit_abs_total": float(np.sum(np.abs(self._absorb_deficit))),
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
