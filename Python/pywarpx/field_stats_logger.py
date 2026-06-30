"""Corrector-agnostic, read-only field-magnitude / potential diagnostic.

Install via ``installafterstep`` in *any* run -- the 2024 line-integral
corrector, the new ``HarmonicBiasCorrector``, or no corrector at all -- to print
directly comparable numbers across runs:

* ``Delta_phi``   : radial line integral of E_r between two radii (the bias the
                    correctors maintain). Configurable bounds; see the note
                    below on how this differs from the 2024 corrector's own
                    full-domain print.
* ``sum|E|^2``    : an unweighted sum of squares over the valid domain -- a
                    *relative* field-energy indicator (no eps0/2, and in RZ no
                    2*pi*r volume metric). On the same grid it is directly
                    comparable between runs, and a rising value while Delta_phi
                    stays put is the signature of field accumulating in the
                    frozen embedded-boundary (conductor) cells.
* ``max|E_r|``, ``max|E_theta|``, ``max|E_z|`` : per-component magnitudes;
                    ``max|E_theta|`` tracks the inductive field.

It is **read-only**: it never touches the fields, so it does not change the run
it measures (safe to add to the faithful 2024 baseline).

Usage::

    from pywarpx.field_stats_logger import FieldStatsLogger
    from pywarpx.callbacks import installafterstep

    logger = FieldStatsLogger(sim, x_lo_phys=cathode_radius, x_hi_phys=anode_radius)
    installafterstep(logger.log)

Note on ``Delta_phi``: the 2024 ``ElectrostaticFieldCorrector`` integrates E_r
over the *whole* radial domain (r=0 .. R_max), whereas this logger integrates
between ``x_lo_phys`` and ``x_hi_phys`` (matching ``HarmonicBiasCorrector``).
Pass the electrode radii to compare the new-run prints; the old corrector also
prints its own full-domain value, so the old run gives both.

Caveats: like ``HarmonicBiasCorrector``'s diagnostics this uses global-index
slicing and assumes a single-box / single-rank layout (true for the production
GPU runs). numpy ufuncs dispatch to cupy on device arrays. The full-domain
reduction runs every ``interval`` steps; at ``interval=1`` over a long run that
is a noticeable (but read-only) cost.
"""


def _get_libwarpx():
    from pywarpx._libwarpx import libwarpx  # noqa: PLC0415

    return libwarpx


class FieldStatsLogger:
    """Read-only per-step field-magnitude / potential logger (corrector-agnostic).

    Parameters
    ----------
    sim : picmi.Simulation
        The initialized PICMI simulation object.
    x_lo_phys, x_hi_phys : float
        Radial bounds for the Delta_phi line integral (inner/outer electrode
        radius). Use the same values you pass to the corrector so the printed
        Delta_phi is comparable.
    interval : int, optional
        Log every this many steps (default 1, i.e. every step -- the most
        informative for spotting field accumulation).
    field : str, optional
        Registered vector field to measure (default "Efield_fp").
    label : str, optional
        Tag for the printed line (default "FieldStats").
    """

    def __init__(self, sim, x_lo_phys, x_hi_phys, interval=1, field="Efield_fp", label="FieldStats"):
        self.sim = sim
        self.x_lo_phys = x_lo_phys
        self.x_hi_phys = x_hi_phys
        self.interval = interval
        self.field = field
        self.label = label

    # -- libwarpx accessors (mirror HarmonicBiasCorrector) -------------------
    def _warpx(self):
        return _get_libwarpx().libwarpx_so.get_instance()

    def _mfr(self):
        return self._warpx().multifab_register()

    def _Direction(self, comp):
        return _get_libwarpx().libwarpx_so.Direction(comp)

    def _is_rz(self):
        from pywarpx import geometry  # noqa: PLC0415

        return geometry.dims == "RZ"

    # -- callback -------------------------------------------------------------
    def log(self):
        """afterstep callback: print Delta_phi + field-magnitude diagnostics."""
        # installafterstep fires after istep is incremented, so getistep is the
        # number of completed steps; log at steps interval, 2*interval, ...
        step = self._warpx().getistep(lev=0)
        if self.interval > 1 and step % self.interval != 0:
            return
        dphi = self._compute_potential_difference()
        stats = self._measure_field_stats()
        print(
            f"[{self.label}] Step {step}: "
            f"Delta_phi = {dphi:.6e}  "
            f"sum|E|^2 = {stats['energy']:.6e}  "
            f"max|E_r| = {stats['max'][0]:.6e}  "
            f"max|E_theta| = {stats['max'][1]:.6e}  "
            f"max|E_z| = {stats['max'][2]:.6e}",
            flush=True,
        )

    # -- measurement (same idioms as HarmonicBiasCorrector) ------------------
    def _compute_potential_difference(self):
        """Radial line integral of E_r between x_lo_phys and x_hi_phys, averaged over z."""
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        mfr = self._mfr()
        is_rz = self._is_rz()

        geom_data = warpx.Geom(lev=0).data()
        Ex_mf = mfr.get(self.field, dir=self._Direction(0), level=0)
        domain = geom_data.Domain().convert(Ex_mf.box_array().ix_type())
        lo = domain.small_end
        hi = domain.big_end

        prob_lo = geom_data.ProbLo()
        dx = geom_data.CellSize()[0]
        i_lo = max(int(round((self.x_lo_phys - prob_lo[0]) / dx)), lo[0])
        i_hi = min(int(round((self.x_hi_phys - prob_lo[0]) / dx)), hi[0])

        if is_rz:
            E_slice = Ex_mf[i_lo : i_hi + 1, :]
            nz = hi[1] - lo[1] + 1
        else:
            mid_y = (hi[1] + lo[1]) // 2
            E_slice = Ex_mf[i_lo : i_hi + 1, mid_y, :]
            nz = hi[2] - lo[2] + 1
        return (dx / nz) * float(np.sum(E_slice))

    def _measure_field_stats(self):
        """sum|E|^2 and per-component max|E| over the valid domain."""
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        mfr = self._mfr()
        is_rz = self._is_rz()
        geom_data = warpx.Geom(lev=0).data()

        energy_comp = [0.0, 0.0, 0.0]
        max_comp = [0.0, 0.0, 0.0]
        for comp in (0, 1, 2):
            mf = mfr.get(self.field, dir=self._Direction(comp), level=0)
            domain = geom_data.Domain().convert(mf.box_array().ix_type())
            lo = domain.small_end
            hi = domain.big_end
            if is_rz:
                arr = mf[lo[0] : hi[0] + 1, :]
            else:
                arr = mf[lo[0] : hi[0] + 1, lo[1] : hi[1] + 1, :]
            energy_comp[comp] = float(np.sum(arr * arr))
            max_comp[comp] = float(np.max(np.abs(arr)))
        return {
            "energy": energy_comp[0] + energy_comp[1] + energy_comp[2],
            "energy_comp": energy_comp,
            "max": max_comp,
        }
