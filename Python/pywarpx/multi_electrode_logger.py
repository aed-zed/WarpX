"""Read-only per-electrode effective-voltage logger (measurement half of
:class:`MultiElectrodeBiasCorrector`).

Builds the vacuum capacitance matrix ``C`` once, then every ``period`` steps
measures the per-electrode induced charge ``Q_k`` (live field) and the
grounded-plasma image charge ``Q_{g,k}`` and reports the effective voltages

    V_eff = C^{-1} (Q - Q_g)

to a CSV. It applies **no** correction and restores the live E-field after each
measurement, so it is a neutral, corrector-agnostic diagnostic: install it
alongside *any* correction scheme (or none) to score per-electrode voltage drift
under plasma on a multi-electrode setup -- the shared multi-electrode scorecard a
single line integral cannot provide (report Sections 16.5-16.6).

Why measurement is safe where correction is hard: reading ``V_eff`` is a one-shot
linear solve with no feedback loop, so the stability/regularization constraints
of multi-electrode *correction* do not apply. It is exact by linear superposition
for conductors + plasma (constant epsilon), the same assumption as ``Q_g``.

Cost: one grounded Poisson solve per logged step (plus ``n+1`` solves once at
setup to build ``C``); keep ``period`` sparse. Single-box/single-rank slicing and
rank-0 write, as in the sibling diagnostics.

Usage::

    from pywarpx.multi_electrode_logger import MultiElectrodePotentialLogger
    from pywarpx.callbacks import installafterInitEsolve, installafterstep

    logger = MultiElectrodePotentialLogger(
        sim=sim,
        electrodes=[
            {"name": "inner", "region": "(x*x<0.0125**2)",               "potential": -1000.0},
            {"name": "shell", "region": "(x*x>0.0125**2)*(x*x<0.0375**2)", "potential":  -400.0},
        ],
        period=1,
        out_csv="multi_electrode_Veff.csv",
    )
    installafterInitEsolve(logger.setup_after_init)  # build C
    installafterstep(logger.log)                     # measure each step
"""

from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector


class MultiElectrodePotentialLogger(MultiElectrodeBiasCorrector):
    """Measure (never correct) per-electrode effective voltages under plasma.

    Parameters
    ----------
    sim : picmi.Simulation
        The initialized PICMI simulation.
    electrodes : list of dict
        Same schema as :class:`MultiElectrodeBiasCorrector` -- each entry has
        ``name``, ``region`` (weighting expression ``w_k``), and ``potential``
        (the *known/target* voltage, used only for the CSV/print comparison).
    period : int, optional
        Log every this many steps (default 1).
    out_csv : str, optional
        CSV output path.
    label : str, optional
        Tag for the printed line.
    verbose : bool, optional
        Passed through to the setup (prints the capacitance matrix).
    """

    def __init__(self, sim, electrodes, period=1,
                 out_csv="multi_electrode_Veff.csv", label="Veff", verbose=False):
        # correction_interval is unused (we never call correct_field); pass period.
        super().__init__(
            sim=sim, correction_interval=period, electrodes=electrodes,
            relaxation=1.0, verbose=verbose,
        )
        self.period = int(period)
        self.out_csv = out_csv
        self.label = label
        self._header_written = False

    def _measure_all(self):
        """Return (V_eff, Q_now, Q_g) with a single grounded solve (read-only)."""
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        lev = 0
        q_now = np.array(
            [warpx.compute_eb_charge(weighting=r, field="Efield_fp") for r in self.regions]
        )
        # Plasma image charge with all electrodes grounded (exact screening).
        saved = self._save_efield(lev)
        warpx.set_potential_on_eb("0.0")
        warpx.solve_poisson_efield()
        q_g = np.array(
            [warpx.compute_eb_charge(weighting=r, field="Efield_fp") for r in self.regions]
        )
        self._restore_efield(saved, lev)
        warpx.set_potential_on_eb(self.potential_expression)
        v = np.linalg.solve(self._capacitance, q_now - q_g)
        return v, q_now, q_g

    def log(self):
        """afterstep/afterInitEsolve callback: measure V_eff and append to CSV."""
        import numpy as np  # noqa: PLC0415

        if not self._ready:
            return
        warpx = self._warpx()
        step = warpx.getistep(lev=0)
        if self.period > 1 and step % self.period != 0:
            return

        try:
            t = float(warpx.gett_new(0))
        except Exception:  # noqa: BLE001
            t = float("nan")

        v, q_now, q_g = self._measure_all()

        if self._rank() == 0:
            self._write_row(step, t, v, q_now, q_g)
        with np.printoptions(precision=4):
            print(
                f"[{self.label}] step {step}: V_eff={v}  "
                f"target={np.array(self.v_target)}",
                flush=True,
            )

    def measure_now(self):
        """Convenience: return the current V_eff (e.g. for an init-time check)."""
        return self._measure_all()[0] if self._ready else None

    def _rank(self):
        try:
            from mpi4py import MPI  # noqa: PLC0415

            return MPI.COMM_WORLD.Get_rank()
        except Exception:  # noqa: BLE001
            return 0

    def _write_row(self, step, t, v, q_now, q_g):
        import csv  # noqa: PLC0415

        mode = "w" if not self._header_written else "a"
        with open(self.out_csv, mode, newline="") as f:
            w = csv.writer(f)
            if not self._header_written:
                w.writerow(
                    ["step", "time"]
                    + [f"Veff_{n}" for n in self.names]
                    + [f"Vtarget_{n}" for n in self.names]
                    + [f"Q_{n}" for n in self.names]
                    + [f"Qg_{n}" for n in self.names]
                )
                self._header_written = True
            w.writerow(
                [step, t]
                + [float(x) for x in v]
                + [float(x) for x in self.v_target]
                + [float(x) for x in q_now]
                + [float(x) for x in q_g]
            )


def _mpi_rank():
    try:
        from mpi4py import MPI  # noqa: PLC0415

        return MPI.COMM_WORLD.Get_rank()
    except Exception:  # noqa: BLE001
        return 0


def _ensure_parent(path):
    import os  # noqa: PLC0415

    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _initialize_csv(path, header, preserve_existing):
    import csv  # noqa: PLC0415
    import os  # noqa: PLC0415

    _ensure_parent(path)
    if preserve_existing and os.path.isfile(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", newline="") as stream:
        csv.writer(stream).writerow(header)


class MultiElectrodeClampTelemetry:
    """Write the state already measured by a harmonic voltage correction.

    This logger takes an existing :class:`MultiElectrodeBiasCorrector`; it
    does not build another capacitance basis, deposit charge, or solve a field
    equation. Install ``setup`` after the corrector's setup callback and
    ``log`` after its correction callback.
    """

    def __init__(
        self,
        corrector,
        period=1,
        out_csv="clamp_telemetry.csv",
        setup_json="clamp_setup.json",
        include_first=True,
    ):
        self.corrector = corrector
        self.period = int(period)
        if self.period < 1:
            raise ValueError("telemetry period must be positive")
        self.out_csv = out_csv
        self.setup_json = setup_json
        self.include_first = bool(include_first)
        self._last_written_step = None

    def setup(self):
        """Write setup metadata and initialize the correction CSV."""
        import json  # noqa: PLC0415

        state = self.corrector.setup_state()
        if _mpi_rank() != 0:
            return

        _ensure_parent(self.setup_json)
        serializable = dict(state)
        serializable["capacitance_matrix"] = state["capacitance_matrix"].tolist()
        with open(self.setup_json, "w", encoding="utf-8") as stream:
            json.dump(serializable, stream, indent=2)
            stream.write("\n")

        # A checkpoint restart has a positive current step. Preserve its
        # existing time series; a fresh step-zero run intentionally replaces
        # stale output from an older run in the same directory.
        _initialize_csv(
            self.out_csv,
            self._header(),
            preserve_existing=state["current_step"] > 0,
        )

    def _header(self):
        names = self.corrector.names
        qg_mode = self.corrector.qg_mode
        return (
            ["step", "time_s"]
            + [f"V_before_{name}_V" for name in names]
            + [f"V_after_predicted_{name}_V" for name in names]
            + [f"V_target_{name}_V" for name in names]
            + [f"V_error_before_{name}_V" for name in names]
            + [f"V_error_after_predicted_{name}_V" for name in names]
            + [f"delta_V_{name}_V" for name in names]
            + [f"Q_field_{name}_C" for name in names]
            + [f"Q_ledger_{name}_C" for name in names]
            + [f"Q_grounded_{qg_mode}_{name}_C" for name in names]
        )

    def _selected(self, step):
        return (self.include_first and step == 1) or step % self.period == 0

    def log(self):
        """Append the latest correction state without re-measuring it."""
        import csv  # noqa: PLC0415

        state = self.corrector.last_correction_state()
        if state is None:
            return
        step = int(state["step"])
        if step == self._last_written_step or not self._selected(step):
            return
        self._last_written_step = step
        if _mpi_rank() != 0:
            return

        row = [step, float(state["time"])]
        for key in (
            "voltage_before",
            "voltage_after_predicted",
            "target_voltage",
            "voltage_error_before",
            "voltage_error_after_predicted",
            "delta_voltage",
            "field_charge",
            "ledger_charge",
            "grounded_charge",
        ):
            row.extend(float(value) for value in state[key])
        with open(self.out_csv, "a", newline="") as stream:
            csv.writer(stream).writerow(row)


class GroundedChargeCrossCheck:
    """Sparsely compare adjoint Qg with a real grounded Poisson solve.

    Unlike :class:`MultiElectrodeClampTelemetry`, every selected call is an
    intentionally expensive collective diagnostic. It saves and restores the
    corrected live electric field around one grounded solve.
    """

    def __init__(
        self,
        corrector,
        period,
        out_csv="clamp_grounded_crosscheck.csv",
        include_first=True,
    ):
        if corrector.qg_mode != "reciprocity":
            raise ValueError(
                "GroundedChargeCrossCheck requires qg_mode='reciprocity' "
                "to compare the adjoint observer with a grounded solve"
            )
        self.corrector = corrector
        self.period = int(period)
        if self.period < 1:
            raise ValueError("grounded cross-check period must be positive")
        self.out_csv = out_csv
        self.include_first = bool(include_first)
        self._last_written_step = None

    def setup(self):
        """Initialize the comparison CSV after corrector setup."""
        state = self.corrector.setup_state()
        if _mpi_rank() != 0:
            return
        names = self.corrector.names
        header = (
            ["step", "time_s", "max_relative_difference"]
            + [f"Q_grounded_adjoint_{name}_C" for name in names]
            + [f"Q_grounded_solve_{name}_C" for name in names]
            + [f"difference_{name}_C" for name in names]
            + [f"relative_difference_{name}" for name in names]
        )
        _initialize_csv(
            self.out_csv,
            header,
            preserve_existing=state["current_step"] > 0,
        )

    def _selected(self, step):
        return (self.include_first and step == 1) or step % self.period == 0

    def log(self):
        """Run and record a selected grounded-solve comparison."""
        import csv  # noqa: PLC0415

        import numpy as np  # noqa: PLC0415

        state = self.corrector.last_correction_state()
        if state is None:
            return
        step = int(state["step"])
        if step == self._last_written_step or not self._selected(step):
            return
        self._last_written_step = step

        reciprocity_charge = None
        if self.corrector.qg_mode == "reciprocity":
            reciprocity_charge = state["grounded_charge"]
        comparison = self.corrector.compare_grounded_charge(reciprocity_charge)

        if _mpi_rank() != 0:
            return
        relative = comparison["relative_difference"]
        row = [step, float(state["time"]), float(np.max(relative))]
        for key in (
            "reciprocity_charge",
            "solved_charge",
            "difference",
            "relative_difference",
        ):
            row.extend(float(value) for value in comparison[key])
        with open(self.out_csv, "a", newline="") as stream:
            csv.writer(stream).writerow(row)
