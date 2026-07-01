"""Reference-potential diagnostic: the self-consistent electrostatic ground truth.

At a chosen stride this performs a full EB-Dirichlet Poisson solve with the
*current* charge density and the electrodes at their prescribed potentials
(``warpx.solve_poisson_efield()``), reconstructs the potential phi from the
freshly solved field, and writes it to its own HDF5 file -- then restores the
live E-field so the running EM simulation is untouched.

Why this is the right "ground truth"
------------------------------------
Neither the reconstructed-phi (line integral) nor the induced-charge (flux)
metric is a neutral arbiter: each is essentially what one corrector *controls*
(report Section 16.1). The self-consistent electrostatic solve is independent of
both -- it answers "given this plasma, what *should* the potential be if the
electrodes were held at their prescribed voltages (Dirichlet)?" Comparing each
run's actual phi (reconstructed from its live E) against its own reference phi
tells you which corrector maintains the correct per-electrode field topology:
if the crisp cathode/EC/insulator separation appears in the reference, the
scheme whose live field reproduces it is the faithful one (report Section 16.4).

The reference solve is EXACT to reconstruct from: ``solve_poisson_efield``
replaces Er, Ez with a pure gradient (-grad phi), so the radial integral of Er
is path-independent -- unlike the live EM field, whose small inductive Er makes
the actual-phi reconstruction only approximate.

Usage::

    from pywarpx.reference_potential import ReferencePotentialDiagnostic
    from pywarpx.callbacks import installafterEsolve

    refphi = ReferencePotentialDiagnostic(
        sim=sim,
        potential_expression=potential_expression,   # same string as the EB potential
        out_dir="reference_phi",
        period=1000,          # solve+dump every 1000 steps
        anode_radius=0.15,    # phi=0 reference (grounded anode)
    )
    installafterEsolve(refphi.dump)

Then compare in post (the reference phi is on the same grid as the main diag):
overlay reference phi against the actual phi from electrode_potential.py.

Cost/caveats: one EB Poisson solve per dump (keep ``period`` sparse -- match it
to your full-field dump cadence). Uses global-index slicing (single-box /
single-rank layout, as in FieldStatsLogger); the HDF5 is written on rank 0.
"""

import os


def _get_libwarpx():
    from pywarpx._libwarpx import libwarpx  # noqa: PLC0415

    return libwarpx


class ReferencePotentialDiagnostic:
    """Periodic self-consistent electrostatic phi, written to HDF5 (read-only to the run)."""

    def __init__(self, sim, potential_expression, out_dir, period,
                 anode_radius=0.15, label="ref"):
        self.sim = sim
        self.potential_expression = potential_expression
        self.out_dir = out_dir
        self.period = int(period)
        self.anode_radius = anode_radius
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

    def _rank(self):
        try:
            from mpi4py import MPI  # noqa: PLC0415

            return MPI.COMM_WORLD.Get_rank()
        except Exception:
            return 0

    # -- callback -------------------------------------------------------------
    def dump(self):
        """afterEsolve callback: reference solve -> reconstruct phi -> write -> restore."""
        warpx = self._warpx()
        # afterEsolve fires before istep is incremented; use (step + 1) so dumps
        # land on steps period, 2*period, ... (aligns with the corrector gating).
        step = warpx.getistep(lev=0)
        if (step + 1) % self.period != 0:
            return

        mfr = self._mfr()
        lev = 0
        # 1) save the live E-field (all three components).
        saved = {}
        for comp in (0, 1, 2):
            saved[comp] = mfr.get("Efield_fp", dir=self._Direction(comp), level=lev).copy()

        # 2) reference solve: electrodes at prescribed V, current rho (deposited
        #    inside solve_poisson_efield). Replaces Er, Ez with -grad(phi_ref).
        warpx.set_potential_on_eb(self.potential_expression)
        warpx.solve_poisson_efield()

        # 3) reconstruct phi from the clean (gradient) solved Er, then write.
        phi, meta = self._reconstruct_phi(lev)
        if self._rank() == 0:
            self._write(step + 1, phi, meta)

        # 4) restore the live E-field so the EM run is untouched.
        for comp in (0, 1, 2):
            mfr.get("Efield_fp", dir=self._Direction(comp), level=lev).copymf(
                saved[comp], 0, 0, 1, 0
            )

    # -- reconstruction -------------------------------------------------------
    def _reconstruct_phi(self, lev):
        """phi(z,r) = int_r^{r_anode} Er dr' from the freshly solved (gradient) Er."""
        import numpy as np  # noqa: PLC0415

        warpx = self._warpx()
        mfr = self._mfr()
        geom = warpx.Geom(lev=lev).data()
        dz, dr = (geom.CellSize()[i] for i in range(2))
        z0, r0 = (geom.ProbLo()[i] for i in range(2))

        Er_mf = mfr.get("Efield_fp", dir=self._Direction(0), level=lev)
        domain = geom.Domain().convert(Er_mf.box_array().ix_type())
        lo, hi = domain.small_end, domain.big_end
        if self._is_rz():
            arr = Er_mf[lo[0] : hi[0] + 1, :]
        else:
            mid_y = (hi[1] + lo[1]) // 2
            arr = Er_mf[lo[0] : hi[0] + 1, mid_y, :]
        # to host numpy (arr may be a cupy device array on GPU)
        Er = arr.get() if hasattr(arr, "get") else np.asarray(arr)
        # Er is [z, r]; suffix-cumsum along r, referenced to the anode radius.
        nr = Er.shape[-1]
        r_centers = r0 + (np.arange(nr) + 0.5) * dr
        ir_anode = int(np.clip(np.searchsorted(r_centers, self.anode_radius), 0, nr - 1))
        suffix = np.cumsum((Er * dr)[..., ::-1], axis=-1)[..., ::-1]
        phi = suffix - suffix[..., [ir_anode]]
        meta = {"dz": float(dz), "dr": float(dr), "z0": float(z0), "r0": float(r0)}
        return phi, meta

    def _write(self, step, phi, meta):
        import h5py  # noqa: PLC0415

        os.makedirs(self.out_dir, exist_ok=True)
        path = os.path.join(self.out_dir, f"reference_phi_{step:010d}.h5")
        with h5py.File(path, "w") as h:
            h.attrs["label"] = self.label
            h.attrs["step"] = step
            h.attrs["gridSpacing"] = [meta["dz"], meta["dr"]]
            h.attrs["gridGlobalOffset"] = [meta["z0"], meta["r0"]]
            h.attrs["anode_radius"] = self.anode_radius
            h.create_dataset("phi", data=phi, compression="gzip")
        print(f"[{self.label}] wrote {path}", flush=True)
