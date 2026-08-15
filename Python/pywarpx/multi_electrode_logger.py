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
