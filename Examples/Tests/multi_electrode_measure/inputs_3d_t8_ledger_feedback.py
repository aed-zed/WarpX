#!/usr/bin/env python3
"""T8 -- wire the absorption ledger into the clamp's voltage FEEDBACK.

T4/T6/T7 validated the ledger itself: it measures the per-impact mis-booked
charge (``accumulate_absorption()``'s ``booked``/``deficit`` matrices). None
of them fed that measurement back into ``measure_voltages()``'s ``V = C^-1
(Q - Q_g)`` inversion -- the clamp still saw the raw, uncorrected charge.
This is the acceptance test for the wiring added in
``multi_electrode_corrector.py`` (``apply_ledger_correction=True``).

THE PHYSICS BEING CHECKED
-------------------------
Per ``reference/absorption_potential_maintenance.md`` Section 1, an ideal
absorption event produces NO potential transient: the induced charge already
tracked the particle in continuously, so nothing changes at contact. Section 2
writes the identity ``Q_k^ext = (C V)_k - sum_p q_p Psi_k(x_p) - Q_k^stuck``.
The two halves of ``measure_voltages()`` do not update in step when a particle
vanishes from the mesh charge density: ``q_now`` is a flux read of the
ADVANCED field (unaffected by the removal until the field is next solved/
advanced), while ``q_g`` is a FRESH grounded resolve of the CURRENT ``rho``
(zero memory of anything just removed). The result is a spurious voltage kick
on every electrode -- see ``measure_voltages()``'s docstring for the full
derivation.

METHOD (static, exact target -- same spirit as T6)
---------------------------------------------------
For each probe position:

    V_before              = measure_voltages() with the probe in flight
                             near a sphere surface
    ... probe removed from the domain PROGRAMMATICALLY (clear_particles()) ...
    V_after_uncorrected   = measure_voltages() with apply_ledger_correction
                             OFF (today's behavior)
    V_after_corrected     = measure_voltages() with apply_ledger_correction
                             ON, ledger seeded with this exact impact

PASS: |V_after_corrected - V_before| < 1e-6 V per electrode. The uncorrected
gap is reported too (finite, and matches C^-1 * q_p * Psi_k -- see
``measure_voltages()``'s docstring, the "+booked" derivation).

WHY THE LEDGER IS SEEDED DIRECTLY RATHER THAN READ FROM A REAL EB SCRAPE
-------------------------------------------------------------------------
The corrector's production code (``measure_voltages()`` with
``apply_ledger_correction=True``) calls the REAL ``accumulate_absorption()``,
which reads WarpX's C++ EB scrape buffer -- but that buffer is only populated
by an actual physical push across the boundary, which requires advancing the
simulation (``max_steps >= 1``). This fixture is explicitly ``max_steps=0``
(same as T6), so there is no physical push, and the buffer container for a
species is not even C++-side "defined" until its first real scrape -- confirmed
empirically: querying it before any real event raises WARPX_ALWAYS_ASSERT
("Tried to get a buffer that is not defined!"), which is an unrecoverable
MPI_Abort, not a Python exception. So instead of a real scrape, this fixture
does exactly what T6 does for the same reason (see its docstring): it
constructs the absorption event MATHEMATICALLY. Concretely, it seeds
``corrector._absorb_booked``/``_absorb_deficit``/``_absorb_counts`` with
EXACTLY the per-impact formula ``accumulate_absorption()`` itself uses
(same ``_gather_psi`` call, same geometric struck-electrode attribution via
``electrode_centers``/``electrode_radii``) for the one probe just removed.
``measure_voltages()`` still calls the REAL ``accumulate_absorption()``
unconditionally (per its contract) -- it finds an empty/undefined buffer,
safely no-ops (``get_particle_boundary_buffer_size`` returns 0 for an
undefined per-species buffer; the cursor guard ``total <= seen`` then skips
without ever touching the seeded arrays), and the seeded ledger survives to
be read by ``absorption_totals()``. This exercises the NEW wiring in
``measure_voltages()`` end to end; the ledger's own accounting arithmetic
(the ``booked``/``deficit`` formulas, the buffer cursor, MPI reduction) is
already covered by T4/T6/T7 and is not re-litigated here.

CAVEAT (real scrapes land just inside, not just outside)
---------------------------------------------------------
T6-style probes here are removed OUTSIDE the true surface (same placement
convention as T6: ``r = R + frac * dx``). A real absorbing-boundary scrape
lands just INSIDE instead. This is fine for a static acceptance test: the
correction ``measure_voltages()`` adds is a discrete IDENTITY, exact for
whatever position and Psi value ``_gather_psi`` actually returns -- it does
not depend on being near the surface in any particular direction, only on
using the SAME Psi the ledger's booking and the grounded resolve both imply
(the ADJOINT table, Section 3b of the report -- built here via
``solve_adjoint_weighting(rhs_mode="charge", add_indicator=False)`` +
``gather_mode="deposit"``, exactly as T6's ``--adjoint --adjoint_rhs charge``
path). The point of this fixture is the exact target that placement affords,
not a claim about real-scrape geometry.

QG_MODE EXTENSION (``qg_mode="reciprocity"``)
-----------------------------------------------
Also exercises ``measure_voltages(qg_mode=...)``: the default ``"grounded"``
mode gets ``Q_g`` from a real save/grounded-solve/restore of ``Efield_fp``
every call (a full Poisson solve); ``"reciprocity"`` instead evaluates the
algebraically equivalent ``Q_g,k = -sum_a rho_a * dV * Psi_k[a]`` directly
against the deposited nodal charge density and the already-loaded ADJOINT
Psi tables -- no solve, no save/restore.

QG_MODE ACCURACY -- A BUG WAS FOUND, DIAGNOSED, AND FIXED HERE
-----------------------------------------------------------------------
An EARLIER version of ``_grounded_charge_via_reciprocity()`` dotted the
adjoint Psi against ``mpc.get_charge_density()``'s output and measured
``Q_g`` DISAGREEING with ``"grounded"`` by 1.4%-99.8% relative across these
8 probe positions -- not the ~1e-6 an earlier in-process check had
suggested. That gap was checked and was NOT floor noise (unchanged to 4+
digits under a 1e4x tighter adjoint solve tolerance) and NOT a missing
unit/scale factor (a 9-way sweep of ``* or / dV``, ``* or / eps0`` found
none closer than the as-implemented ``* dV``). ROOT CAUSE: ``mpc.
get_charge_density()`` returns a raw, UNFILTERED per-species deposit, but
the Poisson solve itself consumes the registered ``rho_fp``, which DOES
receive WarpX's default charge-deposit filter via ``sync_rho()`` --
confirmed by applying the corrector's own ``_binomial_filter_3d`` to the
unfiltered array and matching the registered ``rho_fp`` array to EXACT
floating-point equality. THE FIX (now in ``multi_electrode_corrector.py``):
``_grounded_charge_via_reciprocity()`` now deposits every species into and
reads back the registered ``rho_fp`` via the new
``_deposit_and_read_rho_fp()`` helper, instead of ``mpc.
get_charge_density()``. See ``measure_voltages()``'s docstring in
``multi_electrode_corrector.py`` for the full writeup and the validated
numbers.

POST-FIX, THIS FIXTURE MEASURES: the per-event Q_g relative difference
(probe in flight, nonzero rho -- see the table below) at 4.0e-10 to
4.2e-8, and the dedicated ``negative_control()`` (ledger explicitly
cleared, pure screening) agreeing to 4.9e-11-1.3e-9 relative in ``Q_g`` and
9.4e-12-4.1e-11 V in the resulting voltage -- ``qg_mode="reciprocity"`` is
now VALIDATED. The one caveat that predates and survives the fix: this
fixture's per-event "corrected voltage residual in reciprocity mode"
number is still NOT a meaningful accuracy check on its own -- it is
measured AFTER the probe is removed (rho=0), where both qg_modes trivially
agree regardless of Psi accuracy, in both the buggy and fixed states. The
negative control and the per-event Q_g comparison (nonzero rho) are the
checks that actually discriminate the modes, and both now agree.

USAGE
-----
    python inputs_3d_t8_ledger_feedback.py [--nx 32] [--out t8.json]
"""

import argparse
import json
import time

import numpy as np

import preflight  # noqa: E402
from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve

p = argparse.ArgumentParser()
p.add_argument("--nx", type=int, default=32)
p.add_argument("--qprobe", type=float, default=1.0e-12)
p.add_argument("--out", type=str, default="t8_ledger_feedback.json")
p.add_argument("--tol", type=float, default=1.0e-6,
               help="PASS threshold on |V_after_corrected - V_before| [V]")
p.add_argument("--n_timing", type=int, default=20,
               help="repeated measure_voltages() calls per qg_mode for timing")
args, _ = p.parse_known_args()

preflight.require(bindings=("compute_eb_charge", "solve_poisson_efield",
                            "set_potential_on_eb", "solve_adjoint_weighting"))

L, R, center_offset = 12e-2, 1.5e-2, 3e-2
V_left, V_right = +300.0, -700.0
Q_E = 1.602176634e-19

bf = 8
nx = ((args.nx + bf - 1) // bf) * bf
dx = L / nx
half = nx * dx / 2

print("=" * 88)
print("T8 -- LEDGER FEEDBACK INTO THE CLAMP'S MEASURED VOLTAGES")
print("=" * 88)
print(f"{nx}^3, dx = {dx*1e3:.3f} mm, tol = {args.tol:.1e} V")
print("target: an ideal absorption changes NO electrode's measured voltage")
print("=" * 88)

grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, nx, nx],
    lower_bound=[-half] * 3, upper_bound=[half] * 3,
    lower_boundary_conditions=["dirichlet"] * 3,
    upper_boundary_conditions=["dirichlet"] * 3,
    lower_boundary_conditions_particles=["absorbing"] * 3,
    upper_boundary_conditions_particles=["absorbing"] * 3,
    warpx_blocking_factor=bf, warpx_max_grid_size=1024,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="ECT", cfl=0.9)
R2 = R * R
eb = picmi.EmbeddedBoundary(
    implicit_function=(
        f"max({R2}-((x+{center_offset})*(x+{center_offset})+y*y+z*z),"
        f"{R2}-((x-{center_offset})*(x-{center_offset})+y*y+z*z))"),
    potential=f"({V_left})*(x<0)+({V_right})*(x>0)",
    cover_multiple_cuts=True,
)
probe = picmi.Species(name="probe", particle_type="electron",
                      warpx_save_particles_at_eb=True)
sim = picmi.Simulation(solver=solver, warpx_embedded_boundary=eb,
                       particle_shape="linear", max_steps=0,
                       time_step_size=1.0e-12, verbose=0)
sim.add_species(probe, layout=None)

from pywarpx.multi_electrode_corrector import (  # noqa: E402
    MultiElectrodeBiasCorrector,
)

corrector = MultiElectrodeBiasCorrector(
    sim=sim, correction_interval=999999,
    electrodes=[{"name": "left", "region": "(x<0)", "potential": V_left},
                {"name": "right", "region": "(x>0)", "potential": V_right}],
    enable_gauss_clean=False, verbose=False,
    book_absorption=True, absorption_species=["probe"],
    electrode_centers=[(-center_offset, 0.0, 0.0), (center_offset, 0.0, 0.0)],
    electrode_radii=[R, R],
    gather_mode="deposit",
    apply_ledger_correction=False,   # flipped per-measurement below
)
installafterInitEsolve(corrector.setup_after_init)


def _build_adjoint():
    """Same recipe as T6's ``_build_adjoint()``: solve for the adjoint Psi_k
    via the in-code adjoint route (no probe placement, N_e solves) and hand
    it to the ledger. Required for both the correction (measure_voltages())
    and the grounded resolve inside it to imply the SAME Psi -- see Section
    3b of the report: the cut-cell operator is not symmetric, so the ordinary
    (plain Dirichlet) psi_unit_k is the wrong basis for either.
    """
    w = corrector._warpx()
    mfr = corrector._mfr()
    tables = []
    for k, reg in enumerate(corrector.regions):
        name = corrector._psi_names[k]          # already registered at setup
        ok, res = w.solve_adjoint_weighting(region=reg, out_name=name,
                                            rhs_mode="charge", add_indicator=False,
                                            tol=1.0e-10, max_iter=2000)
        print(f"  adjoint Psi[{corrector.names[k]}]: converged={ok} "
              f"rel.residual={res:.3e}")
        if not ok:
            raise SystemExit(
                f"adjoint solve for electrode {k} did NOT converge "
                f"(residual {res:.3e}). Refusing to continue.")
        tables.append(np.array(mfr.get(name, level=0)[:, :, :]))
    corrector.load_psi_table(tables)


installafterInitEsolve(_build_adjoint)

ROWS = []


def refresh_live():
    """Solve Poisson with the corrector's current bias, updating Efield_fp to
    reflect whatever particles presently exist. This is the ONLY thing that
    ever gives Efield_fp memory of a particle in this max_steps=0 fixture --
    exactly the property measure_voltages()'s docstring exploits/explains."""
    w = corrector._warpx()
    w.set_potential_on_eb(corrector.potential_expression)
    w.solve_poisson_efield()


def seed_ledger(x, y, z, q_p):
    """Directly populate the (electrode x species) ledger with EXACTLY the
    per-impact formula accumulate_absorption() itself uses (same _gather_psi
    call, same geometric struck-electrode attribution) for the single probe
    just removed -- see the module docstring for why a real EB-buffer read
    is unavailable in a max_steps=0 fixture."""
    ns = len(corrector.absorption_species)
    corrector._absorb_booked = np.zeros((corrector.n, ns))
    corrector._absorb_deficit = np.zeros((corrector.n, ns))
    corrector._absorb_counts = np.zeros(ns, dtype=np.int64)

    psis = np.array([corrector._gather_psi(k, np.array([x]), np.array([y]),
                                           np.array([z]))[0]
                     for k in range(corrector.n)])

    centers = corrector.electrode_centers
    radii = corrector.electrode_radii
    d = np.array([np.sqrt((x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2) - r
                  for c, r in zip(centers, radii)])
    struck = int(np.argmin(d))

    for k in range(corrector.n):
        corrector._absorb_booked[k, 0] = q_p * psis[k]
        if k == struck:
            corrector._absorb_deficit[k, 0] = q_p * (1.0 - psis[k])
        else:
            corrector._absorb_deficit[k, 0] = -q_p * psis[k]
    corrector._absorb_counts[0] = 1
    return psis, struck


def zero_ledger():
    """Reset the ledger to its never-accumulated state (None, same as right
    after construction) -- used for the negative control below so the
    qg_mode comparison there is unconfounded by apply_ledger_correction."""
    corrector._absorb_booked = None
    corrector._absorb_deficit = None
    corrector._absorb_counts = None
    corrector._buffer_cursor = {}


def qg_both_modes():
    """Q_g per electrode from both qg_mode paths, at whatever state Efield_fp
    /rho currently hold. Calls the private helpers directly (not through
    measure_voltages()) so this is a pure Q_g comparison, independent of
    apply_ledger_correction/q_now."""
    q_g_grounded = corrector._grounded_charge_via_solve(0)
    q_g_recip = corrector._grounded_charge_via_reciprocity(0)
    rel_diff = np.abs(q_g_recip - q_g_grounded) / np.maximum(
        np.abs(q_g_grounded), 1.0e-300)
    return q_g_grounded, q_g_recip, rel_diff


def negative_control(pc, w_macro):
    """PURE SCREENING, NO ABSORPTION: probe in flight, ledger explicitly
    zeroed, compare Q_g and V from both qg_modes. They must still agree --
    this isolates the qg_mode change from the ledger-feedback change."""
    print("\n" + "=" * 88)
    print("NEGATIVE CONTROL -- probe in flight, ledger cleared, no absorption")
    print("=" * 88)
    zero_ledger()
    x, y, z = -center_offset + (R + 0.7 * dx), 0.3 * dx, -0.2 * dx
    pc.add_particles(x=np.array([x]), y=np.array([y]), z=np.array([z]),
                     ux=np.zeros(1), uy=np.zeros(1), uz=np.zeros(1),
                     w=np.array([w_macro]))
    refresh_live()

    q_g_grounded, q_g_recip, rel_diff = qg_both_modes()
    print(f"  Q_g grounded   = {q_g_grounded}")
    print(f"  Q_g reciprocity= {q_g_recip}")
    print(f"  relative diff  = {rel_diff}")

    corrector.apply_ledger_correction = False
    corrector.qg_mode = "grounded"
    v_grounded = corrector.measure_voltages()
    corrector.qg_mode = "reciprocity"
    v_recip = corrector.measure_voltages()
    corrector.qg_mode = "grounded"
    v_diff = np.abs(v_recip - v_grounded)
    print(f"  V (qg=grounded)    = {v_grounded}")
    print(f"  V (qg=reciprocity) = {v_recip}")
    print(f"  |V diff|           = {v_diff}")
    agree = bool(np.all(rel_diff < 1.0e-4) and np.all(v_diff < 1.0e-6))
    print(f"  => {'AGREE' if agree else 'DISAGREE'}")
    print("=" * 88)

    pc.particle_container.clear_particles()
    zero_ledger()
    return {"Q_g_grounded": q_g_grounded.tolist(), "Q_g_reciprocity": q_g_recip.tolist(),
            "rel_diff": rel_diff.tolist(), "V_grounded": v_grounded.tolist(),
            "V_reciprocity": v_recip.tolist(), "V_diff": v_diff.tolist(),
            "agree": agree}


def time_qg_modes(n):
    """Time n repeated measure_voltages() calls in each qg_mode, fixed vacuum
    state (no probe, apply_ledger_correction off) so only the Q_g path's own
    cost is measured."""
    print("\n" + "=" * 88)
    print(f"TIMING -- {n} repeated measure_voltages() calls per qg_mode")
    print("=" * 88)
    corrector.apply_ledger_correction = False
    refresh_live()

    corrector.qg_mode = "grounded"
    t0 = time.perf_counter()
    for _ in range(n):
        corrector.measure_voltages()
    t_grounded = (time.perf_counter() - t0) / n

    corrector.qg_mode = "reciprocity"
    t0 = time.perf_counter()
    for _ in range(n):
        corrector.measure_voltages()
    t_recip = (time.perf_counter() - t0) / n
    corrector.qg_mode = "grounded"

    speedup = t_grounded / t_recip if t_recip > 0 else float("inf")
    print(f"  grounded   : {t_grounded*1e3:.4f} ms/call")
    print(f"  reciprocity: {t_recip*1e3:.4f} ms/call")
    print(f"  speedup    : {speedup:.2f}x")
    print("=" * 88)
    return {"t_grounded_ms": t_grounded * 1e3, "t_reciprocity_ms": t_recip * 1e3,
            "speedup": speedup, "n": n}


def run():
    from pywarpx.particle_containers import ParticleContainerWrapper
    pc = ParticleContainerWrapper("probe")

    q_p = -args.qprobe                 # electron: negative charge
    w_macro = args.qprobe / Q_E

    neg_control = negative_control(pc, w_macro)
    timing = time_qg_modes(args.n_timing)

    rng = np.random.default_rng(1234)
    # >= 4 positions, different distances/angles, both electrodes.
    fracs = [0.2, 0.5, 1.0, 1.5]        # distance beyond the surface, in dx
    events = []
    for which, xc in ((0, -center_offset), (1, center_offset)):
        for frac in fracs:
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            r_pos = R + frac * dx
            events.append({
                "electrode": corrector.names[which],
                "which": which,
                "frac": frac,
                "x": xc + r_pos * u[0],
                "y": r_pos * u[1],
                "z": r_pos * u[2],
            })

    for ev in events:
        x, y, z = ev["x"], ev["y"], ev["z"]

        pc.add_particles(x=np.array([x]), y=np.array([y]), z=np.array([z]),
                         ux=np.zeros(1), uy=np.zeros(1), uz=np.zeros(1),
                         w=np.array([w_macro]))
        refresh_live()

        # Q_g from both modes, at this pure-screening (probe in flight, no
        # absorption yet) state -- the acceptance table's Q_g comparison.
        q_g_grounded, q_g_recip, qg_rel_diff = qg_both_modes()

        corrector.apply_ledger_correction = False
        corrector.qg_mode = "grounded"
        V_before = corrector.measure_voltages()

        # Remove the probe PROGRAMMATICALLY -- see module docstring. Efield_fp
        # is left exactly as the refresh_live() above set it (with the probe's
        # contribution baked in); nothing re-solves it until measure_voltages()
        # does its own internal (save/restore-wrapped) grounded resolve.
        pc.particle_container.clear_particles()

        psis, struck = seed_ledger(x, y, z, q_p)

        corrector.apply_ledger_correction = False
        V_after_uncorrected = corrector.measure_voltages()

        corrector.apply_ledger_correction = True
        V_after_corrected = corrector.measure_voltages()

        corrector.qg_mode = "reciprocity"
        V_after_corrected_recip = corrector.measure_voltages()
        corrector.qg_mode = "grounded"
        corrector.apply_ledger_correction = False

        resid_unc = np.abs(V_after_uncorrected - V_before)
        resid_cor = np.abs(V_after_corrected - V_before)
        resid_cor_recip = np.abs(V_after_corrected_recip - V_before)
        # PASS/FAIL is decided by the "grounded" qg_mode residual only -- the
        # original ledger-feedback acceptance target from earlier in this
        # module's docstring. resid_cor_recip is reported alongside it but is
        # NOT part of the pass criterion: it is measured AFTER clear_particles()
        # zeroes rho, where q_g is trivially ~0 in BOTH qg_modes regardless of
        # Psi accuracy -- see the module docstring's "QG_MODE ACCURACY" section
        # for why this specific check cannot discriminate the modes, and the
        # negative_control() result (probe in flight, nonzero rho) for the
        # check that actually can.
        passed = bool(np.all(resid_cor < args.tol))

        ROWS.append({
            "electrode": ev["electrode"], "frac_dx": ev["frac"],
            "xyz": [float(x), float(y), float(z)],
            "psi": psis.tolist(), "struck": struck,
            "Q_g_grounded": q_g_grounded.tolist(), "Q_g_reciprocity": q_g_recip.tolist(),
            "Q_g_rel_diff": qg_rel_diff.tolist(),
            "V_before": V_before.tolist(),
            "V_after_uncorrected": V_after_uncorrected.tolist(),
            "V_after_corrected": V_after_corrected.tolist(),
            "V_after_corrected_reciprocity": V_after_corrected_recip.tolist(),
            "resid_uncorrected": resid_unc.tolist(),
            "resid_corrected": resid_cor.tolist(),
            "resid_corrected_reciprocity": resid_cor_recip.tolist(),
            "pass": passed,
        })

    print("\n" + "=" * 88)
    print("T8 RESULT")
    print("=" * 88)
    hdr = (f"{'electrode':<9s} {'r-R [dx]':>9s} "
           f"{'Q_g grounded':>26s} {'Q_g recip':>26s} {'Qg reldiff':>12s} "
           f"{'resid_unc':>12s} {'resid_cor(g)':>12s} {'resid_cor(r)':>12s} "
           f"{'pass':>5s}")
    print(hdr)
    for r in ROWS:
        qgg = np.array(r["Q_g_grounded"])
        qgr = np.array(r["Q_g_reciprocity"])
        qgd = np.array(r["Q_g_rel_diff"])
        ru = np.array(r["resid_uncorrected"])
        rc = np.array(r["resid_corrected"])
        rcr = np.array(r["resid_corrected_reciprocity"])
        print(f"{r['electrode']:<9s} {r['frac_dx']:>9.2f} "
              f"{np.array2string(qgg, precision=6):>26s} "
              f"{np.array2string(qgr, precision=6):>26s} "
              f"{np.max(qgd):>12.3e} "
              f"{np.max(ru):>12.3e} {np.max(rc):>12.3e} {np.max(rcr):>12.3e} "
              f"{'PASS' if r['pass'] else 'FAIL':>5s}")

    ledger_pass = all(r["pass"] for r in ROWS)
    max_resid_unc = max(max(r["resid_uncorrected"]) for r in ROWS)
    max_resid_cor = max(max(r["resid_corrected"]) for r in ROWS)
    max_resid_cor_recip = max(max(r["resid_corrected_reciprocity"]) for r in ROWS)
    max_qg_reldiff = max(max(r["Q_g_rel_diff"]) for r in ROWS)
    print()
    print("  -- LEDGER FEEDBACK (the T8 acceptance target) --")
    print(f"  max |V_after_uncorrected - V_before|            = {max_resid_unc:.3e} V "
          "(the uncorrected drift; finite, non-zero -- this is the bug)")
    print(f"  max |V_after_corrected(grounded)    - V_before| = {max_resid_cor:.3e} V "
          f"(tol {args.tol:.1e} V) -- THE LEDGER-FEEDBACK ACCEPTANCE TARGET")
    print(f"  => LEDGER FEEDBACK (qg_mode='grounded'): "
          f"{'ALL EVENTS PASS' if ledger_pass else 'SOME EVENTS FAIL'}")
    print()
    print("  -- QG_MODE='reciprocity' ACCURACY: the checks that DISCRIMINATE "
          "the two modes (nonzero rho) --")
    print(f"  max Q_g relative difference (grounded vs reciprocity), PROBE IN "
          f"FLIGHT (per-event, nonzero rho) = {max_qg_reldiff:.3e} "
          "-- see module docstring's QG_MODE ACCURACY section for the fix "
          "history (an earlier rho source gave 1.4%-99.8% here).")
    print(f"  dedicated negative control (probe in flight, ledger cleared, "
          f"no absorption): {'AGREE' if neg_control['agree'] else 'DISAGREE'} "
          f"(Q_g rel diff {neg_control['rel_diff']}, "
          f"|V diff| {neg_control['V_diff']})")
    print(f"  timing: grounded {timing['t_grounded_ms']:.4f} ms/call, "
          f"reciprocity {timing['t_reciprocity_ms']:.4f} ms/call, "
          f"speedup {timing['speedup']:.2f}x")
    print(f"  => QG_MODE='reciprocity' ACCURACY: "
          f"{'VALIDATED (agrees with grounded)' if neg_control['agree'] else 'NOT VALIDATED -- disagrees with grounded, see docstring'}")
    print()
    print(f"  (NOT a discriminating check -- reported for completeness only: "
          f"max |V_after_corrected(reciprocity) - V_before| = "
          f"{max_resid_cor_recip:.3e} V. Measured AFTER the probe is removed, "
          f"rho=0, where both qg_modes trivially agree regardless of Psi "
          f"accuracy; see the negative control above instead.)")
    print("=" * 88)

    json.dump({"rows": ROWS, "tol": args.tol, "ledger_pass": ledger_pass,
               "max_resid_uncorrected": max_resid_unc,
               "max_resid_corrected": max_resid_cor,
               "max_resid_corrected_reciprocity": max_resid_cor_recip,
               "max_qg_reldiff": max_qg_reldiff,
               "negative_control": neg_control,
               "timing": timing},
              open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")

    # Module-level flag checked AFTER sim.step(0) returns, below -- NOT raised
    # from inside this afterInitEsolve callback. A SystemExit raised here
    # propagates through the pybind11 callback dispatch as an uncaught C++
    # exception (observed: "pure virtual method called" / SIGABRT), which
    # also discards any buffered (non -u) stdout -- the opposite of a useful
    # failure report.
    global LEDGER_PASS
    LEDGER_PASS = ledger_pass


LEDGER_PASS = None
installafterInitEsolve(run)
sim.step(0)

if LEDGER_PASS is False:
    raise SystemExit(1)
