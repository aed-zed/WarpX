#!/usr/bin/env python3
"""Four-arm electrode-potential-maintenance comparison campaign (v6).

Same clamp-work fixture as ``inputs_3d_ect_bias_only_clampwork_v5.py`` (two
sphere electrodes at +300/-700 V, ECT solver, asymmetric absorbing beams,
2000 steps) but run four ways so a human can see, in directly comparable
plots, whether the new correction stack (adjoint weighting-potential ledger +
reciprocity Q_g) improves electrode-potential maintenance over the historical
v3-v5 recipe. Geometry, beams, seed, dt, ECT solver, and the v5 energy/charge
accounting are UNCHANGED across arms -- the only thing that differs is which
correction machinery is wired up:

  arm a : no correction at all. Voltages are measured (read-only,
          corrector.measure_voltages()) but never applied.
  arm b : bias-only, correction_interval=10 -- the historical recipe (the
          v5 inline _do_bias_correction closure, unchanged). ADDITIONALLY
          books absorption with the adjoint Psi table (book_absorption=True)
          but never applies the ledger (apply_ledger_correction=False), so
          the ledger MEASURES what the old recipe silently mis-books without
          changing its behavior -- the "implied mis-booked charge" curve for
          the old method.
  arm c : bias-only + ledger, correction_interval=10: book_absorption=True,
          apply_ledger_correction=True, qg_mode="grounded". Isolates what the
          ledger buys at the familiar cadence.
  arm d : the full modern stack: correction_interval=1 (every step),
          book_absorption=True, apply_ledger_correction=True,
          qg_mode="reciprocity".

All four arms go through the SAME MultiElectrodeBiasCorrector class (never an
inline closure of its own) so the new kwargs (book_absorption, gather_mode,
electrode_centers/radii, apply_ledger_correction, qg_mode) are always
available; arm a simply never calls the bias-apply path.

For arms b/c/d the adjoint weighting-potential Psi_k is built at init exactly
as inputs_3d_t6_booking_exactness.py's/_t8_ledger_feedback.py's
``_build_adjoint()`` does: one ``solve_adjoint_weighting(region=..., out_name
=..., rhs_mode="charge", add_indicator=False)`` solve per electrode, then
``corrector.load_psi_table(tables)``. ``gather_mode="deposit"`` is used
throughout (deposit-consistent psi gather at absorbed-particle positions), and
``electrode_centers``/``electrode_radii`` are supplied so the ledger's
struck-electrode attribution is geometric (required whenever gather_mode=
"deposit" or a custom psi_table is loaded -- argmax(psi) is meaningless there).

Voltage-measurement cadence. ``measure_voltages()`` in qg_mode="grounded"
(arms a, b, c) performs a full grounded MLMG solve every call -- every-5-steps
is the deliberate compromise (measuring every step would roughly double the
solver cost of those three arms for little diagnostic benefit). Whenever a
correction already measured this step (steps that are corrector-correction
steps), the trace reuses that measurement's PRE-correction V rather than
re-measuring -- this is exact (measure_voltages() is idempotent / read-only)
and avoids paying for the grounded solve twice at the same instant. Arm d
(qg_mode="reciprocity", correction every step) measures every step essentially
for free (no Poisson solve), so its trace is dense with no separate cost.

Usage (env exactly as v5 requires -- see its module docstring for the
pyamrex-RPATH background):
    export LD_LIBRARY_PATH=/home/mgarten/src/warpx/build/lib:$LD_LIBRARY_PATH
    export UCX_TLS=tcp,self
    export PYTHONPATH=/home/mgarten/src/warpx/build/lib/site-packages:/home/mgarten/src/warpx/Examples/Tests/multi_electrode_measure
    python inputs_3d_ect_fourarm_v6.py --arm {a,b,c,d} [--total_steps 2000] [--out PATH.npz]

ALWAYS single-rank (no mpirun): the four arms must share an identical
particle realization, which requires an identical domain decomposition, so
none of them may be run with more than one MPI rank.
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy.constants import c as c_light, epsilon_0, e, m_e

import preflight  # noqa: E402
from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve, installafterstep

T_SCRIPT_START = time.time()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--arm", required=True, choices=("a", "b", "c", "d"))
p.add_argument("--total_steps", type=int, default=2000)
p.add_argument("--out", type=str, default=None,
               help="output .npz path (default: fourarm_v6/arm_<arm>.npz "
                    "relative to the repo's multi_electrode_measure dir)")
args = p.parse_args()

ARM = args.arm
total_steps = args.total_steps

_DEFAULT_OUTDIR = "/home/mgarten/src/warpx/Examples/Tests/multi_electrode_measure/fourarm_v6"
OUT_NPZ = args.out or os.path.join(_DEFAULT_OUTDIR, f"arm_{ARM}.npz")
os.makedirs(os.path.dirname(os.path.abspath(OUT_NPZ)), exist_ok=True)

# ---------------------------------------------------------------------------
# Per-arm configuration
# ---------------------------------------------------------------------------
ARM_CONFIG = {
    "a": dict(do_correction=False, correction_interval=10, measure_stride=5,
              book_absorption=False, apply_ledger_correction=False,
              qg_mode="grounded", use_adjoint=False,
              label="no correction"),
    "b": dict(do_correction=True, correction_interval=10, measure_stride=5,
              book_absorption=True, apply_ledger_correction=False,
              qg_mode="grounded", use_adjoint=True,
              label="bias-only every 10 (historical)"),
    "c": dict(do_correction=True, correction_interval=10, measure_stride=5,
              book_absorption=True, apply_ledger_correction=True,
              qg_mode="grounded", use_adjoint=True,
              label="bias-only + ledger every 10"),
    "d": dict(do_correction=True, correction_interval=1, measure_stride=1,
              book_absorption=True, apply_ledger_correction=True,
              qg_mode="reciprocity", use_adjoint=True,
              label="ledger + every-step (reciprocity)"),
}
CFG = ARM_CONFIG[ARM]
DO_CORRECTION = CFG["do_correction"]
correction_interval = CFG["correction_interval"]
measure_stride = CFG["measure_stride"]
LEDGER_STRIDE = 5

preflight.require(bindings=(
    "compute_eb_charge", "solve_poisson_efield", "set_potential_on_eb",
) + (("solve_adjoint_weighting",) if CFG["use_adjoint"] else ()))

# ---------------------------------------------------------------------------
# Parameters -- IDENTICAL to inputs_3d_ect_bias_only_clampwork_v5.py across
# all four arms (geometry, beams, seed, dt, ECT solver, energy accounting).
# ---------------------------------------------------------------------------
cells_per_R = 10
L = 12e-2
R = 1.5e-2
center_offset = 3e-2
V_left = +300.0
V_right = -700.0

bf = 8
dx = R / cells_per_R
nx = ((int(round(L / dx)) + bf - 1) // bf) * bf
dy = dz = dx
ny = nz = nx
L_actual = nx * dx
half = L_actual / 2

n_beam = 3e14
m_species = m_e
beam_hw = 1.0e-2
z_beam_lo = -half
z_beam_hi = -half + L_actual / 8
vz_drift = 0.1 * c_light
vx_impact = -vz_drift * center_offset / half

h_boxes = [R + 2 * dx, R + 5 * dx, R + 8 * dx]
box_labels = ["close", "mid", "far"]
cfl = 0.9
frame_interval = 50

SNAPSHOT_STEPS = sorted({0, 300, 450, 600, 750, 1000, 1500, 2000})

MODE = f"ect_fourarm_v6_arm{ARM}"

cfl_dt = cfl * dx / (c_light * np.sqrt(3))
omega_correction = 2.0 * np.pi / (cfl_dt * correction_interval)

# ---------------------------------------------------------------------------
print("=" * 90)
print(f"FOUR-ARM ELECTRODE-POTENTIAL-MAINTENANCE CAMPAIGN -- arm {ARM} "
      f"({CFG['label']})")
print("=" * 90)
print(f"Domain: {L_actual*1e2:.2f} cm cube, {nx}^3 cells, dx = {dx*1e3:.3f} mm")
print(f"Total steps: {total_steps}, correction every {correction_interval} steps, "
      f"measure_stride={measure_stride}")
print(f"do_correction={DO_CORRECTION} book_absorption={CFG['book_absorption']} "
      f"apply_ledger_correction={CFG['apply_ledger_correction']} qg_mode={CFG['qg_mode']}")
print(f"dt (estimate) = {cfl_dt:.6e} s")
print(f"Output npz: {OUT_NPZ}")

# ---------------------------------------------------------------------------
# Grid, solver, EB -- identical to v5
# ---------------------------------------------------------------------------
grid = picmi.Cartesian3DGrid(
    number_of_cells=[nx, ny, nz],
    lower_bound=[-half, -half, -half],
    upper_bound=[half, half, half],
    lower_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing", "absorbing"],
    warpx_blocking_factor=bf,
    warpx_max_grid_size=1024,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="ECT", cfl=cfl)

R2 = R * R
eb_implicit = (
    f"max({R2}-((x+{center_offset})*(x+{center_offset})+y*y+z*z),"
    f"{R2}-((x-{center_offset})*(x-{center_offset})+y*y+z*z))"
)
potential_expression = f"({V_left})*(x<0)+({V_right})*(x>0)"
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=eb_implicit,
    potential=potential_expression,
    cover_multiple_cuts=True,
)

# ---------------------------------------------------------------------------
# Asymmetric beams -- identical to v5 (same seed/layout: no --arm dependence
# anywhere in species/grid/layout construction, so the particle realization
# is bit-identical across arms).
# ---------------------------------------------------------------------------
beam_dist_pos = picmi.AnalyticDistribution(
    density_expression=f"{n_beam}",
    directed_velocity=[0, 0, vz_drift],
    lower_bound=[-beam_hw, -beam_hw, z_beam_lo],
    upper_bound=[beam_hw, beam_hw, z_beam_hi],
)
beam_dist_neg = picmi.AnalyticDistribution(
    density_expression=f"{n_beam}",
    directed_velocity=[vx_impact, 0, vz_drift],
    lower_bound=[-beam_hw, -beam_hw, z_beam_lo],
    upper_bound=[beam_hw, beam_hw, z_beam_hi],
)
_save_particles_kwargs = {
    f"warpx_save_particles_at_{b}": 1
    for b in ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi", "eb")
}
pos_species = picmi.Species(
    name="pos", charge=e, mass=m_species,
    initial_distribution=beam_dist_pos,
    **_save_particles_kwargs,
)
neg_species = picmi.Species(
    name="neg", charge=-e, mass=m_species,
    initial_distribution=beam_dist_neg,
    **_save_particles_kwargs,
)

sim = picmi.Simulation(
    solver=solver,
    warpx_embedded_boundary=embedded_boundary,
    particle_shape="linear",
    max_steps=total_steps,
)
for sp in (pos_species, neg_species):
    sim.add_species(
        sp,
        layout=picmi.PseudoRandomLayout(grid=grid, n_macroparticles_per_cell=2),
    )

# Reduced diagnostics -- identical to v5
Q_eb_left_diag = picmi.ReducedDiagnostic(
    diag_type="ChargeOnEB", name="Q_eb_left",
    period=frame_interval, weighting_function="(x < 0)",
)
Q_eb_right_diag = picmi.ReducedDiagnostic(
    diag_type="ChargeOnEB", name="Q_eb_right",
    period=frame_interval, weighting_function="(x > 0)",
)
field_energy_diag = picmi.ReducedDiagnostic(
    diag_type="FieldEnergy", name="field_energy",
    period=frame_interval,
)
etan_diag = picmi.ReducedDiagnostic(
    diag_type="TangentialEOnEB", name="etan_eb",
    period=frame_interval,
)
particle_energy_diag = picmi.ReducedDiagnostic(
    diag_type="ParticleEnergy", name="particle_energy",
    period=frame_interval,
)
poynting_diag = picmi.ReducedDiagnostic(
    diag_type="FieldPoyntingFlux", name="poynting_flux",
    period=1,
)
sim.add_diagnostic(Q_eb_left_diag)
sim.add_diagnostic(Q_eb_right_diag)
sim.add_diagnostic(field_energy_diag)
sim.add_diagnostic(etan_diag)
sim.add_diagnostic(particle_energy_diag)
sim.add_diagnostic(poynting_diag)

# ---------------------------------------------------------------------------
# Corrector -- ALWAYS MultiElectrodeBiasCorrector; arm a simply never applies
# a bias. Arms b/c/d additionally book absorption with the adjoint Psi table.
# ---------------------------------------------------------------------------
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector  # noqa: E402
from pywarpx.field_stats_logger import ClampWorkTracker, field_energy  # noqa: E402

_electrodes = [
    {"name": "left", "region": "(x<0)", "potential": V_left},
    {"name": "right", "region": "(x>0)", "potential": V_right},
]

_corrector_kwargs = dict(
    sim=sim,
    correction_interval=999999,   # cadence managed by hand in after_step below
    electrodes=_electrodes,
    enable_gauss_clean=False,
    verbose=True,
    qg_mode=CFG["qg_mode"],
)
if CFG["book_absorption"]:
    _corrector_kwargs.update(
        book_absorption=True,
        absorption_species=["pos", "neg"],
        gather_mode="deposit",
        electrode_centers=[(-center_offset, 0.0, 0.0), (center_offset, 0.0, 0.0)],
        electrode_radii=[R, R],
        apply_ledger_correction=CFG["apply_ledger_correction"],
    )

corrector = MultiElectrodeBiasCorrector(**_corrector_kwargs)

# -- timing instrumentation ---------------------------------------------
# Wrap measure_voltages()/_apply_bias() so EVERY call (whether from a
# diagnostic voltage-trace read or from an actual correction) is timed,
# regardless of call site.
_timing = {"measure_voltages_s": 0.0, "apply_bias_s": 0.0, "n_measure": 0,
           "n_apply": 0}
_orig_measure_voltages = corrector.measure_voltages
_orig_apply_bias = corrector._apply_bias


def _timed_measure_voltages():
    t0 = time.perf_counter()
    v = _orig_measure_voltages()
    _timing["measure_voltages_s"] += time.perf_counter() - t0
    _timing["n_measure"] += 1
    return v


def _timed_apply_bias(dv):
    t0 = time.perf_counter()
    r = _orig_apply_bias(dv)
    _timing["apply_bias_s"] += time.perf_counter() - t0
    _timing["n_apply"] += 1
    return r


corrector.measure_voltages = _timed_measure_voltages
corrector._apply_bias = _timed_apply_bias


def _build_adjoint():
    """Adjoint Psi_k, exactly as t6/t8's _build_adjoint(): one
    solve_adjoint_weighting(rhs_mode="charge", add_indicator=False) solve per
    electrode, then load_psi_table(). Required for book_absorption to book a
    meaningful (not order-of-magnitude-wrong) correction, and for
    qg_mode="reciprocity"."""
    if not CFG["use_adjoint"]:
        return
    w = corrector._warpx()
    mfr = corrector._mfr()
    tables = []
    for k, reg in enumerate(corrector.regions):
        name = corrector._psi_names[k]
        # max_iter=2000 (T6/T8's default) converges at nx=32 but stalls well
        # short of tol=1e-10 at this fixture's production nx=80 (measured
        # rel.residual ~2e-3 after 2000 iters, unchanged whether tried at
        # 2000 or 8000 -- CG on the squared-condition-number normal
        # equations simply needs more iterations at higher resolution, not a
        # stall: nx=64 reaches 4.2e-5 in 2000 iters, nx=80 reaches ~1e-10 by
        # 30000). Verified via inputs_3d_t6_booking_exactness.py --nx 80
        # --adjoint --adjoint_rhs charge with max_iter=30000: converged=True,
        # rel.residual~9.95e-11, T6 acceptance booking relative error 0.00%.
        ok, res = w.solve_adjoint_weighting(
            region=reg, out_name=name, rhs_mode="charge", add_indicator=False,
            tol=1.0e-10, max_iter=30000,
        )
        print(f"  adjoint Psi[{corrector.names[k]}]: converged={ok} "
              f"rel.residual={res:.3e}", flush=True)
        if not ok:
            raise SystemExit(
                f"adjoint solve for electrode {k} did NOT converge "
                f"(residual {res:.3e}). Refusing to continue.")
        tables.append(np.array(mfr.get(name, level=0)[:, :, :]))
    corrector.load_psi_table(tables)


# ---------------------------------------------------------------------------
# Data collection helpers -- gauss-box charge bookkeeping, identical to v5
# ---------------------------------------------------------------------------
sphere_centers = [
    ("left", -center_offset, 0.0, 0.0, V_left),
    ("right", +center_offset, 0.0, 0.0, V_right),
]
q_data = {}
field_frames = []
correction_steps = []
voltage_history = []          # (step, V_before, V_after) -- correction events only
voltage_trace = []            # (step, V, is_correction) -- the full measured trace
ledger_records = []           # (step, booked (n,ns), deficit (n,ns), counts (ns,))
field_snapshots = {}          # step -> {"Ez": 2D, "Emag": 2D}


def gauss_box_charge(Ex, Ey, Ez, node_box, dxg, dyg, dzg):
    i0, i1, j0, j1, k0, k1 = node_box
    flux = 0.0
    flux += np.sum(Ex[i1, j0:j1 + 1, k0:k1 + 1]) * dyg * dzg
    flux -= np.sum(Ex[i0 - 1, j0:j1 + 1, k0:k1 + 1]) * dyg * dzg
    flux += np.sum(Ey[i0:i1 + 1, j1, k0:k1 + 1]) * dxg * dzg
    flux -= np.sum(Ey[i0:i1 + 1, j0 - 1, k0:k1 + 1]) * dxg * dzg
    flux += np.sum(Ez[i0:i1 + 1, j0:j1 + 1, k1]) * dxg * dyg
    flux -= np.sum(Ez[i0:i1 + 1, j0:j1 + 1, k0 - 1]) * dxg * dyg
    return epsilon_0 * flux


def _warpx():
    from pywarpx._libwarpx import libwarpx  # noqa: PLC0415
    return libwarpx.libwarpx_so.get_instance(), libwarpx


def _get_fields():
    wx, lib = _warpx()
    mfr = wx.multifab_register()
    D = lib.libwarpx_so.Direction
    Ex = np.asarray(mfr.get("Efield_fp", dir=D(0), level=0)[:, :, :])
    Ey = np.asarray(mfr.get("Efield_fp", dir=D(1), level=0)[:, :, :])
    Ez = np.asarray(mfr.get("Efield_fp", dir=D(2), level=0)[:, :, :])
    return Ex, Ey, Ez


def collect(step):
    Ex, Ey, Ez = _get_fields()
    wx, _ = _warpx()
    geom = wx.Geom(lev=0).data()
    dxg, dyg, dzg = geom.CellSize()
    x_lo, y_lo, z_lo = geom.ProbLo()

    jmid_Ex = Ex.shape[1] // 2
    ex_slice = Ex[:, jmid_Ex, :].copy()
    field_frames.append((step, ex_slice))

    data = {}
    for s_name, xc, yc, zc, _ in sphere_centers:
        for h, hl in zip(h_boxes, box_labels):
            i0 = int(round((xc - h - x_lo) / dxg))
            i1 = int(round((xc + h - x_lo) / dxg))
            j0 = int(round((yc - h - y_lo) / dyg))
            j1 = int(round((yc + h - y_lo) / dyg))
            k0 = int(round((zc - h - z_lo) / dzg))
            k1 = int(round((zc + h - z_lo) / dzg))
            Q = gauss_box_charge(Ex, Ey, Ez, (i0, i1, j0, j1, k0, k1),
                                 dxg, dyg, dzg)
            data[f"{s_name}_{hl}"] = Q
    q_data[step] = data

    row = f"  [step {step:4d}]"
    for s_name, *_ in sphere_centers:
        Qs = [data[f"{s_name}_{hl}"] for hl in box_labels]
        row += f"  {s_name}: " + "/".join(f"{Q:+.4e}" for Q in Qs)
    print(row, flush=True)


def capture_snapshot(step):
    """Ez and |E| slices on the y=0 midplane, for the field-snapshot plot
    bracketing the absorption burst. Each E component lives on its own
    staggered (Yee) grid, so |E| is formed on the largest common cropped
    index range -- a diagnostic-only approximation (not used in any
    energy/charge accounting), consistent across arms/steps since the grid
    is identical."""
    Ex, Ey, Ez = _get_fields()
    ex_s = Ex[:, Ex.shape[1] // 2, :]
    ey_s = Ey[:, Ey.shape[1] // 2, :]
    ez_s = Ez[:, Ez.shape[1] // 2, :]
    ni = min(ex_s.shape[0], ey_s.shape[0], ez_s.shape[0])
    nk = min(ex_s.shape[1], ey_s.shape[1], ez_s.shape[1])
    emag = np.sqrt(ex_s[:ni, :nk] ** 2 + ey_s[:ni, :nk] ** 2 + ez_s[:ni, :nk] ** 2)
    field_snapshots[step] = {"Ez": ez_s.copy(), "Emag": emag}
    print(f"  [snapshot step {step}] Ez range [{ez_s.min():.3e}, {ez_s.max():.3e}] "
          f"|E| max {emag.max():.3e}", flush=True)


# ---------------------------------------------------------------------------
# measure_voltages() save-solve-restore energy-neutrality check (v5 parity;
# only meaningful when a correction is actually applied).
# ---------------------------------------------------------------------------
neutrality_check_step = correction_interval * 3 + 1
neutrality_result = {}


def _field_energy_now():
    wx, lib = _warpx()
    mfr = wx.multifab_register()
    D = lib.libwarpx_so.Direction
    return field_energy(mfr, D, lev=0, field="Efield_fp", warpx=wx)


def _check_measure_voltages_neutrality(step):
    W_before = _field_energy_now()
    _ = corrector.measure_voltages()
    W_after = _field_energy_now()
    neutrality_result["step"] = step
    neutrality_result["W_before"] = W_before
    neutrality_result["W_after"] = W_after
    neutrality_result["dW"] = W_after - W_before
    print(
        f"[Neutrality] step {step}: bare measure_voltages() W_E "
        f"before={W_before:.15e} after={W_after:.15e} "
        f"dW={W_after - W_before:+.6e} J", flush=True,
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def after_init():
    print("\n--- Setting up corrector (unit fields + capacitance matrix) ---")
    corrector.setup_after_init()
    if CFG["use_adjoint"]:
        print("\n--- Building adjoint weighting-potential Psi_k ---")
        _build_adjoint()
    print("\n--- Collecting initial frame ---")
    collect(0)
    if 0 in SNAPSHOT_STEPS:
        capture_snapshot(0)


step_counter = [0]

_bias_apply_result = {}


def _do_bias_correction():
    V_before = corrector.measure_voltages().copy()
    V_target = np.array(corrector.v_target)
    dV = V_target - V_before
    corrector._apply_bias(dV)
    V_after = corrector.measure_voltages().copy()
    _bias_apply_result["V_before"] = V_before
    _bias_apply_result["V_after"] = V_after


tracker = ClampWorkTracker(sim, _do_bias_correction, label="ClampWork", verbose=False)

# ---------------------------------------------------------------------------
# W_absorbed / absorbed-charge bookkeeping -- identical to v5
# ---------------------------------------------------------------------------
from pywarpx import particle_containers  # noqa: E402

boundary_names = ["x_lo", "x_hi", "y_lo", "y_hi", "z_lo", "z_hi", "eb"]
species_names = ["pos", "neg"]
species_masses = {"pos": m_species, "neg": m_species}

W_absorbed_by_boundary = {b: 0.0 for b in boundary_names}
W_absorbed_by_species = {sp: 0.0 for sp in species_names}
n_scraped_total = [0]
W_ABSORBED_COMPUTED = True
_particle_buffer = particle_containers.ParticleBoundaryBufferWrapper()


def _kinetic_energy_si(ux, uy, uz, mass):
    inv_c2 = 1.0 / (c_light * c_light)
    u2 = ux * ux + uy * uy + uz * uz
    gamma = np.sqrt(1.0 + u2 * inv_c2)
    return mass * u2 / (1.0 + gamma)


species_charges = {"pos": e, "neg": -e}

Q_absorbed_by_electrode = {"left": 0.0, "right": 0.0}
Q_absorbed_by_species_electrode = {("pos", "left"): 0.0, ("pos", "right"): 0.0,
                                    ("neg", "left"): 0.0, ("neg", "right"): 0.0}

eb_absorbed_records = {"x": [], "y": [], "z": [], "w": [], "species": []}


def _accumulate_absorbed_energy():
    total_n = 0
    for sp in species_names:
        mass = species_masses[sp]
        charge = species_charges[sp]
        for b in boundary_names:
            ux_list = _particle_buffer.get_particle_boundary_buffer(sp, b, "ux", 0)
            uy_list = _particle_buffer.get_particle_boundary_buffer(sp, b, "uy", 0)
            uz_list = _particle_buffer.get_particle_boundary_buffer(sp, b, "uz", 0)
            w_list = _particle_buffer.get_particle_boundary_buffer(sp, b, "w", 0)
            if b == "eb":
                x_list = _particle_buffer.get_particle_boundary_buffer(sp, b, "x", 0)
                y_list = _particle_buffer.get_particle_boundary_buffer(sp, b, "y", 0)
                z_list = _particle_buffer.get_particle_boundary_buffer(sp, b, "z", 0)
            else:
                x_list = y_list = z_list = None
            n_this = 0
            W_this = 0.0
            n_lists = len(w_list)
            iterables = zip(
                ux_list, uy_list, uz_list, w_list,
                x_list if x_list is not None else [None] * n_lists,
                y_list if y_list is not None else [None] * n_lists,
                z_list if z_list is not None else [None] * n_lists,
            )
            for ux, uy, uz, w, x, y, z in iterables:
                ux = np.asarray(ux)
                uy = np.asarray(uy)
                uz = np.asarray(uz)
                w = np.asarray(w)
                n_this += ux.size
                if ux.size:
                    ke = _kinetic_energy_si(ux, uy, uz, mass)
                    W_this += float(np.sum(ke * w))
                    if b == "eb":
                        x = np.asarray(x)
                        y = np.asarray(y)
                        z = np.asarray(z)
                        Q_left = charge * float(np.sum(w[x < 0]))
                        Q_right = charge * float(np.sum(w[x >= 0]))
                        Q_absorbed_by_electrode["left"] += Q_left
                        Q_absorbed_by_electrode["right"] += Q_right
                        Q_absorbed_by_species_electrode[(sp, "left")] += Q_left
                        Q_absorbed_by_species_electrode[(sp, "right")] += Q_right
                        eb_absorbed_records["x"].append(x)
                        eb_absorbed_records["y"].append(y)
                        eb_absorbed_records["z"].append(z)
                        eb_absorbed_records["w"].append(w)
                        eb_absorbed_records["species"].append(
                            np.full(x.shape, 1 if sp == "pos" else -1, dtype=np.int8)
                        )
            W_absorbed_by_boundary[b] += W_this
            W_absorbed_by_species[sp] += W_this
            total_n += n_this
    n_scraped_total[0] = total_n


def after_step():
    step_counter[0] += 1
    step = step_counter[0]

    if step % frame_interval == 0:
        collect(step)

    if step in SNAPSHOT_STEPS:
        capture_snapshot(step)

    did_correction = False
    if DO_CORRECTION and step % correction_interval == 0:
        if step == neutrality_check_step:
            _check_measure_voltages_neutrality(step)
        tracker.tracked_correction()
        V_before = _bias_apply_result["V_before"]
        V_after = _bias_apply_result["V_after"]
        correction_steps.append(step)
        voltage_history.append((step, V_before.copy(), V_after.copy()))
        did_correction = True

        if step % (correction_interval * 5) == 0 or correction_interval >= 50:
            d_w = tracker.history[-1][3]
            print(f"  [corr step {step}] V: [{V_before[0]:+.1f}, {V_before[1]:+.1f}]"
                  f" -> [{V_after[0]:+.1f}, {V_after[1]:+.1f}]  "
                  f"Delta_W_clamp={d_w:+.6e} J  total_W_clamp={tracker.total_W_clamp:+.6e} J",
                  flush=True)

    if CFG["book_absorption"] and step % LEDGER_STRIDE == 0:
        corrector.accumulate_absorption()
        totals = corrector.absorption_totals(reduce=True)
        if totals is not None:
            ledger_records.append(
                (step, totals["booked"].copy(), totals["deficit"].copy(),
                 totals["counts"].copy())
            )

    if step % measure_stride == 0:
        if did_correction:
            v = _bias_apply_result["V_before"]
        else:
            v = corrector.measure_voltages()
        voltage_trace.append((step, np.asarray(v).copy(), did_correction))


installafterInitEsolve(after_init)
installafterstep(after_step)

# ===========================================================================
# Run simulation
# ===========================================================================
print(f"\n{'='*90}")
print(f"Running {total_steps} steps, arm {ARM} ({CFG['label']})")
print(f"{'='*90}", flush=True)

# sim.step(0) forces WarpX's one-time initialization (grid/EB setup,
# corrector.setup_after_init(), the adjoint solve, collect(0)) via the
# afterInitEsolve callback WITHOUT advancing any PIC steps -- picmi.
# Simulation.initialize_warpx() is idempotent (guarded by warpx_initialized),
# so this cleanly separates the one-time setup cost (dominated by the
# adjoint CG solve at nx=80 for arms using it) from the per-step stepping
# cost that the timing-probe extrapolation actually needs.
t_init_start = time.time()
sim.step(0)
t_init_end = time.time()

t_steps_start = time.time()
sim.step(total_steps)
t_steps_end = time.time()

_accumulate_absorbed_energy()
W_absorbed_total = sum(W_absorbed_by_boundary.values())

print(f"\n{'='*90}")
print("W_absorbed (particle-boundary-buffer kinetic-energy sum) -- COMPUTED")
print(f"W_ABSORBED_COMPUTED={W_ABSORBED_COMPUTED}")
print(f"{'='*90}")
print(f"TOTAL W_ABSORBED={W_absorbed_total:.10e} J  (N_scraped_total={n_scraped_total[0]})")
for b in boundary_names:
    print(f"  W_absorbed[{b}]={W_absorbed_by_boundary[b]:.10e} J")
for sp in species_names:
    print(f"  W_absorbed[{sp}]={W_absorbed_by_species[sp]:.10e} J")

print(f"\n{'='*90}")
print("Charge-balance check: absorbed particle charge vs Delta(ChargeOnEB)")
print(f"{'='*90}")
for name, key in (("left", "left"), ("right", "right")):
    Q_abs = Q_absorbed_by_electrode[key]
    fname = f"diags/reducedfiles/Q_eb_{name}.txt"
    try:
        q_eb = np.atleast_2d(np.loadtxt(fname, comments="#"))
        Q_eb_initial = q_eb[0, 2]
        Q_eb_final = q_eb[-1, 2]
        dQ_eb = Q_eb_final - Q_eb_initial
    except OSError:
        Q_eb_initial = dQ_eb = float("nan")
    print(f"  {name.upper()} electrode: Q_absorbed(particles)={Q_abs:.6e} C   "
          f"Delta(Q_eb)={dQ_eb:.6e} C   "
          f"diff={Q_abs - dQ_eb:.6e} C")
for (sp, elec), Q in Q_absorbed_by_species_electrode.items():
    print(f"    Q_absorbed[{sp},{elec}]={Q:.6e} C")

_particle_buffer.clear_buffer()
del _particle_buffer

# ===========================================================================
# Ledger totals (final) -- for arms with book_absorption
# ===========================================================================
if CFG["book_absorption"]:
    final_ledger = corrector.absorption_totals(reduce=True)
else:
    final_ledger = None

# ===========================================================================
# Save data
# ===========================================================================
dt_sim = cfl * dx / (c_light * np.sqrt(3))
steps_sorted = sorted(q_data.keys())

save_dict = dict(
    arm=ARM, arm_label=CFG["label"],
    dx=dx, half=half, nx=nx, nz=nz,
    dt_sim=dt_sim,
    R=R, center_offset=center_offset,
    V_left=V_left, V_right=V_right,
    h_boxes=np.array(h_boxes),
    z_beam_lo=z_beam_lo, vz_drift=vz_drift,
    n_beam=n_beam, beam_hw=beam_hw,
    cells_per_R=cells_per_R,
    correction_interval=correction_interval,
    measure_stride=measure_stride,
    do_correction=DO_CORRECTION,
    book_absorption=CFG["book_absorption"],
    apply_ledger_correction=CFG["apply_ledger_correction"],
    qg_mode=CFG["qg_mode"],
    total_steps=total_steps,
    correction_steps=np.array(correction_steps),
    mode=MODE,
    title=f"Four-arm campaign -- arm {ARM} ({CFG['label']})",
    field_steps=np.array([f[0] for f in field_frames]),
    field_data=np.array([f[1] for f in field_frames]),
    total_W_clamp=tracker.total_W_clamp,
    W_absorbed_total=W_absorbed_total,
    Q_absorbed_left=Q_absorbed_by_electrode["left"],
    Q_absorbed_right=Q_absorbed_by_electrode["right"],
    eb_absorbed_x=(np.concatenate(eb_absorbed_records["x"])
                    if eb_absorbed_records["x"] else np.array([])),
    eb_absorbed_y=(np.concatenate(eb_absorbed_records["y"])
                    if eb_absorbed_records["y"] else np.array([])),
    eb_absorbed_z=(np.concatenate(eb_absorbed_records["z"])
                    if eb_absorbed_records["z"] else np.array([])),
    eb_absorbed_w=(np.concatenate(eb_absorbed_records["w"])
                    if eb_absorbed_records["w"] else np.array([])),
    eb_absorbed_species=(np.concatenate(eb_absorbed_records["species"])
                          if eb_absorbed_records["species"] else np.array([])),
    neutrality_check_step=neutrality_result.get("step", -1),
    neutrality_W_before=neutrality_result.get("W_before", np.nan),
    neutrality_W_after=neutrality_result.get("W_after", np.nan),
    neutrality_dW=neutrality_result.get("dW", np.nan),
    omega_correction=omega_correction,
    # -- timing --
    t_script_start=T_SCRIPT_START,
    total_wall_time_s=time.time() - T_SCRIPT_START,
    init_wall_time_s=t_init_end - t_init_start,
    steps_wall_time_s=t_steps_end - t_steps_start,
    measure_voltages_s=_timing["measure_voltages_s"],
    apply_bias_s=_timing["apply_bias_s"],
    correction_seconds=_timing["measure_voltages_s"] + _timing["apply_bias_s"],
    n_measure_voltages_calls=_timing["n_measure"],
    n_apply_bias_calls=_timing["n_apply"],
)

if voltage_history:
    save_dict["voltage_steps"] = np.array([v[0] for v in voltage_history])
    save_dict["voltage_before"] = np.array([v[1] for v in voltage_history])
    save_dict["voltage_after"] = np.array([v[2] for v in voltage_history])

if voltage_trace:
    save_dict["voltage_trace_steps"] = np.array([v[0] for v in voltage_trace])
    save_dict["voltage_trace_V"] = np.array([v[1] for v in voltage_trace])
    save_dict["voltage_trace_is_correction"] = np.array([v[2] for v in voltage_trace])

if ledger_records:
    save_dict["ledger_steps"] = np.array([r[0] for r in ledger_records])
    save_dict["ledger_booked"] = np.array([r[1] for r in ledger_records])
    save_dict["ledger_deficit"] = np.array([r[2] for r in ledger_records])
    save_dict["ledger_counts"] = np.array([r[3] for r in ledger_records])
if final_ledger is not None:
    save_dict["ledger_booked_final"] = final_ledger["booked"]
    save_dict["ledger_deficit_final"] = final_ledger["deficit"]
    save_dict["ledger_counts_final"] = final_ledger["counts"]

if field_snapshots:
    snap_steps = sorted(field_snapshots.keys())
    save_dict["snapshot_steps"] = np.array(snap_steps)
    save_dict["snapshot_Ez"] = np.array([field_snapshots[s]["Ez"] for s in snap_steps])
    save_dict["snapshot_Emag"] = np.array([field_snapshots[s]["Emag"] for s in snap_steps])

for s in ("left", "right"):
    for hl in box_labels:
        key = f"q_{s}_{hl}"
        save_dict[key] = np.array([q_data[st][f"{s}_{hl}"]
                                   for st in steps_sorted])

np.savez(OUT_NPZ, **save_dict)
print(f"\nSaved {OUT_NPZ} ({len(field_frames)} frames, "
      f"{len(voltage_trace)} voltage-trace points, {len(ledger_records)} ledger points, "
      f"{len(field_snapshots)} field snapshots)", flush=True)

# ---------------------------------------------------------------------------
# Conservation-identity summary (v5 parity)
# ---------------------------------------------------------------------------
wx_final, lib_final = _warpx()
mfr_final = wx_final.multifab_register()
D_final = lib_final.libwarpx_so.Direction
W_E_final = field_energy(mfr_final, D_final, lev=0, field="Efield_fp", warpx=wx_final)

fe = np.atleast_2d(np.loadtxt("diags/reducedfiles/field_energy.txt", comments="#"))
W_E_initial = fe[0, 3]
W_B_initial = fe[0, 4]
W_E_final_diag = fe[-1, 3]
W_B_final_diag = fe[-1, 4]

try:
    pe = np.atleast_2d(np.loadtxt("diags/reducedfiles/particle_energy.txt", comments="#"))
    W_P_initial = pe[0, 2]
    W_P_final = pe[-1, 2]
except OSError:
    W_P_initial = float("nan")
    W_P_final = float("nan")

try:
    pf = np.atleast_2d(np.loadtxt("diags/reducedfiles/poynting_flux.txt", comments="#"))
    W_outgoing_total = float(np.sum(pf[-1, 8:14]))
except OSError:
    W_outgoing_total = float("nan")

raw_drift_pct = (W_E_final_diag / W_E_initial - 1.0) * 100.0
corrected_delta = (W_E_final_diag - W_E_initial) - tracker.total_W_clamp
corrected_drift_pct = corrected_delta / W_E_initial * 100.0
delta_W_particles_E_B = (W_P_final - W_P_initial) + (W_E_final_diag - W_E_initial) + (W_B_final_diag - W_B_initial)
full_residual = delta_W_particles_E_B - tracker.total_W_clamp + W_outgoing_total + W_absorbed_total
full_residual_excl_outgoing_absorbed = delta_W_particles_E_B - tracker.total_W_clamp

print(f"\n{'='*90}")
print(f"Summary -- arm {ARM} ({CFG['label']})")
print(f"{'='*90}")
print(f"STEPS_RUN={total_steps}")
print(f"CORRECTION_INTERVAL={correction_interval}")
print(f"N_CORRECTIONS={len(tracker.history)}")
print(f"W_E_INITIAL={W_E_initial:.10e} J")
print(f"W_E_FINAL_REDUCEDDIAG={W_E_final_diag:.10e} J")
print(f"RAW_DRIFT_PCT(E-field only)={raw_drift_pct:+.6f} %")
print(f"TOTAL_W_CLAMP={tracker.total_W_clamp:.10e} J")
print(f"CORRECTED_DRIFT_PCT(E-field only)={corrected_drift_pct:+.6f} %")
print(f"W_OUTGOING_TOTAL={W_outgoing_total:.10e} J")
print(f"W_ABSORBED_TOTAL={W_absorbed_total:.10e} J")
print(f"FULL_RESIDUAL_EXCL_OUTGOING_ABSORBED={full_residual_excl_outgoing_absorbed:.10e} J")
print(f"FULL_RESIDUAL_INCL_OUTGOING_ABSORBED={full_residual:.10e} J")
if final_ledger is not None:
    print(f"LEDGER_BOOKED_FINAL={final_ledger['booked'].tolist()}")
    print(f"LEDGER_DEFICIT_FINAL={final_ledger['deficit'].tolist()}")
    print(f"LEDGER_COUNTS_FINAL={final_ledger['counts'].tolist()}")
print(f"TOTAL_WALL_TIME_S={time.time() - T_SCRIPT_START:.3f}")
print(f"INIT_WALL_TIME_S={t_init_end - t_init_start:.3f}")
print(f"STEPS_WALL_TIME_S={t_steps_end - t_steps_start:.3f}")
print(f"MEASURE_VOLTAGES_S={_timing['measure_voltages_s']:.3f} "
      f"(n={_timing['n_measure']})")
print(f"APPLY_BIAS_S={_timing['apply_bias_s']:.3f} (n={_timing['n_apply']})")

with open(os.path.join(os.path.dirname(os.path.abspath(OUT_NPZ)),
                        f"arm_{ARM}_summary.txt"), "w") as f:
    f.write(f"ARM={ARM}\n")
    f.write(f"LABEL={CFG['label']}\n")
    f.write(f"STEPS_RUN={total_steps}\n")
    f.write(f"CORRECTION_INTERVAL={correction_interval}\n")
    f.write(f"MEASURE_STRIDE={measure_stride}\n")
    f.write(f"N_CORRECTIONS={len(tracker.history)}\n")
    f.write(f"W_E_INITIAL={W_E_initial:.10e}\n")
    f.write(f"W_E_FINAL_REDUCEDDIAG={W_E_final_diag:.10e}\n")
    f.write(f"RAW_DRIFT_PCT={raw_drift_pct:+.6f}\n")
    f.write(f"TOTAL_W_CLAMP={tracker.total_W_clamp:.10e}\n")
    f.write(f"CORRECTED_DRIFT_PCT={corrected_drift_pct:+.6f}\n")
    f.write(f"W_OUTGOING_TOTAL={W_outgoing_total:.10e}\n")
    f.write(f"W_ABSORBED_TOTAL={W_absorbed_total:.10e}\n")
    f.write(f"FULL_RESIDUAL_EXCL_OUTGOING_ABSORBED={full_residual_excl_outgoing_absorbed:.10e}\n")
    f.write(f"FULL_RESIDUAL_INCL_OUTGOING_ABSORBED={full_residual:.10e}\n")
    f.write(f"TOTAL_WALL_TIME_S={time.time() - T_SCRIPT_START:.3f}\n")
    f.write(f"INIT_WALL_TIME_S={t_init_end - t_init_start:.3f}\n")
    f.write(f"STEPS_WALL_TIME_S={t_steps_end - t_steps_start:.3f}\n")
    f.write(f"MEASURE_VOLTAGES_S={_timing['measure_voltages_s']:.3f}\n")
    f.write(f"APPLY_BIAS_S={_timing['apply_bias_s']:.3f}\n")
    if final_ledger is not None:
        f.write(f"LEDGER_BOOKED_FINAL={final_ledger['booked'].tolist()}\n")
        f.write(f"LEDGER_DEFICIT_FINAL={final_ledger['deficit'].tolist()}\n")
        f.write(f"LEDGER_COUNTS_FINAL={final_ledger['counts'].tolist()}\n")

sys.stdout.flush()
os._exit(0)
