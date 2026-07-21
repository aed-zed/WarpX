#!/usr/bin/env python3
"""3D Gauss-box test: periodic Gauss clean + harmonic bias every 100 steps.

Same geometry as the single-correction test: one passthrough species,
one that impacts the left electrode.  Correction (homogeneous Gauss clean
followed by harmonic bias) is applied every 100 timesteps throughout the
entire simulation.

Usage:
    python inputs_3d_gauss_periodic_100.py
"""

import os
import sys

import numpy as np
from scipy.constants import c as c_light, epsilon_0, e, m_e

from pywarpx import picmi
from pywarpx.callbacks import installafterInitEsolve, installafterstep

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
correction_interval = 100
total_steps = 2000
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

# ---------------------------------------------------------------------------
print("=" * 90)
print(f"3D GAUSS-LAW PERIODIC CORRECTION TEST — every {correction_interval} steps")
print("=" * 90)
print(f"Domain: {L_actual*1e2:.2f} cm cube, {nx}^3 cells, dx = {dx*1e3:.3f} mm")
print(f"Total steps: {total_steps}, correction every {correction_interval} steps")

# ---------------------------------------------------------------------------
# Grid, solver, EB
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
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=cfl)

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
# Asymmetric beams
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
pos_species = picmi.Species(
    name="pos", charge=e, mass=m_species,
    initial_distribution=beam_dist_pos,
)
neg_species = picmi.Species(
    name="neg", charge=-e, mass=m_species,
    initial_distribution=beam_dist_neg,
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

# Reduced diagnostics
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
sim.add_diagnostic(Q_eb_left_diag)
sim.add_diagnostic(Q_eb_right_diag)
sim.add_diagnostic(field_energy_diag)
sim.add_diagnostic(etan_diag)

# ---------------------------------------------------------------------------
# Corrector
# ---------------------------------------------------------------------------
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector  # noqa: E402

corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    correction_interval=999999,
    electrodes=[
        {"name": "left",  "region": "(x<0)", "potential": V_left},
        {"name": "right", "region": "(x>0)", "potential": V_right},
    ],
    enable_gauss_clean=False,
    verbose=True,
)

# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
sphere_centers = [
    ("left",  -center_offset, 0.0, 0.0, V_left),
    ("right", +center_offset, 0.0, 0.0, V_right),
]
q_data = {}
field_frames = []
density_frames = []
correction_log = []


def gauss_box_charge(Ex, Ey, Ez, node_box, dxg, dyg, dzg):
    i0, i1, j0, j1, k0, k1 = node_box
    flux = 0.0
    flux += np.sum(Ex[i1,   j0:j1 + 1, k0:k1 + 1]) * dyg * dzg
    flux -= np.sum(Ex[i0-1, j0:j1 + 1, k0:k1 + 1]) * dyg * dzg
    flux += np.sum(Ey[i0:i1 + 1, j1,   k0:k1 + 1]) * dxg * dzg
    flux -= np.sum(Ey[i0:i1 + 1, j0-1, k0:k1 + 1]) * dxg * dzg
    flux += np.sum(Ez[i0:i1 + 1, j0:j1 + 1, k1  ]) * dxg * dyg
    flux -= np.sum(Ez[i0:i1 + 1, j0:j1 + 1, k0-1]) * dxg * dyg
    return epsilon_0 * flux


def _warpx():
    from pywarpx._libwarpx import libwarpx  # noqa: PLC0415
    return libwarpx.libwarpx_so.get_instance(), libwarpx


def _particle_xz(species_name):
    from pywarpx import particle_containers as pc_mod  # noqa: PLC0415
    pc = pc_mod.ParticleContainerWrapper(species_name)
    x_tiles = pc.get_particle_x()
    y_tiles = pc.get_particle_y()
    z_tiles = pc.get_particle_z()
    w_tiles = pc.get_particle_weight()
    x = np.concatenate(x_tiles) if x_tiles else np.array([])
    y = np.concatenate(y_tiles) if y_tiles else np.array([])
    z = np.concatenate(z_tiles) if z_tiles else np.array([])
    w = np.concatenate(w_tiles) if w_tiles else np.array([])
    return x, y, z, w


def _density_slice(species_name, x_edges, z_edges, y_cut=0.0, y_width=None):
    if y_width is None:
        y_width = 2 * dx
    x, y, z, w = _particle_xz(species_name)
    if len(x) == 0:
        return np.zeros((len(x_edges) - 1, len(z_edges) - 1))
    mask = np.abs(y - y_cut) < y_width
    hist, _, _ = np.histogram2d(
        x[mask], z[mask], bins=[x_edges, z_edges], weights=w[mask],
    )
    cell_vol = (x_edges[1] - x_edges[0]) * (2 * y_width) * (z_edges[1] - z_edges[0])
    return hist / cell_vol


def _get_fields():
    wx, lib = _warpx()
    mfr = wx.multifab_register()
    D = lib.libwarpx_so.Direction
    Ex = np.asarray(mfr.get("Efield_fp", dir=D(0), level=0)[:, :, :])
    Ey = np.asarray(mfr.get("Efield_fp", dir=D(1), level=0)[:, :, :])
    Ez = np.asarray(mfr.get("Efield_fp", dir=D(2), level=0)[:, :, :])
    return Ex, Ey, Ez


def _get_ex_slice():
    Ex, _, _ = _get_fields()
    return Ex[:, Ex.shape[1] // 2, :].copy()


def collect(step):
    Ex, Ey, Ez = _get_fields()
    wx, _ = _warpx()
    geom = wx.Geom(lev=0).data()
    dxg, dyg, dzg = geom.CellSize()
    x_lo, y_lo, z_lo = geom.ProbLo()

    jmid_Ex = Ex.shape[1] // 2
    ex_slice = Ex[:, jmid_Ex, :].copy()
    field_frames.append((step, ex_slice))

    x_edges = np.linspace(-half, half, nx + 1)
    z_edges = np.linspace(-half, half, nz + 1)
    rho_pos = _density_slice("pos", x_edges, z_edges)
    rho_neg = _density_slice("neg", x_edges, z_edges)
    density_frames.append((step, rho_pos.copy(), rho_neg.copy()))

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

    n_pos = sum(len(t) for t in _particle_xz("pos")[0:1])
    n_neg = sum(len(t) for t in _particle_xz("neg")[0:1])

    row = f"  [step {step:4d}] N+={n_pos} N-={n_neg}"
    for s_name, *_ in sphere_centers:
        Qs = [data[f"{s_name}_{hl}"] for hl in box_labels]
        row += f"  {s_name}: " + "/".join(f"{Q:+.4e}" for Q in Qs)
    print(row, flush=True)


def after_init():
    print("\n--- Setting up corrector (unit fields + capacitance matrix) ---")
    corrector.setup_after_init()
    print("\n--- Collecting initial frame ---")
    collect(0)


step_counter = [0]


def after_step():
    step_counter[0] += 1
    step = step_counter[0]

    if step % frame_interval == 0:
        collect(step)

    if step % correction_interval == 0:
        wx_inst, _ = _warpx()
        V_before = corrector.measure_voltages().copy()
        wx_inst.clean_efield_gauss_homogeneous()
        V_target = np.array(corrector.v_target)
        dV = V_target - corrector.measure_voltages()
        corrector._apply_bias(dV)
        V_after = corrector.measure_voltages().copy()
        correction_log.append((step, V_before, V_after))
        if step % (correction_interval * 10) == 0:
            print(f"  Correction at step {step}: "
                  f"V [{V_before[0]:+.1f},{V_before[1]:+.1f}] -> "
                  f"[{V_after[0]:+.1f},{V_after[1]:+.1f}]", flush=True)


installafterInitEsolve(after_init)
installafterstep(after_step)

# ===========================================================================
# Run simulation
# ===========================================================================
print(f"\n{'='*90}")
print(f"Running {total_steps} steps with correction every {correction_interval}")
print(f"{'='*90}", flush=True)

sim.step(total_steps)

# ===========================================================================
# Save data
# ===========================================================================
dt_sim = cfl * dx / (c_light * np.sqrt(3))
steps_sorted = sorted(q_data.keys())

save_dict = dict(
    dx=dx, half=half, nx=nx, nz=nz,
    dt_sim=dt_sim,
    R=R, center_offset=center_offset,
    V_left=V_left, V_right=V_right,
    h_boxes=np.array(h_boxes),
    z_beam_lo=z_beam_lo, vz_drift=vz_drift,
    n_beam=n_beam, beam_hw=beam_hw,
    cells_per_R=cells_per_R,
    title=f"Periodic Correction (every {correction_interval} steps)",
    subtitle=f"Gauss clean + harmonic bias every {correction_interval} steps, "
             f"asymmetric beams (passthrough + impact)",
    beam_desc=f"n = {n_beam:.0e} m^-3, m = m_e (both), "
              f"width = {beam_hw*1e2:.0f} cm; "
              f"pos: vz={vz_drift/c_light:.2f}c (passthrough); "
              f"neg: vx={vx_impact/c_light:.4f}c, vz={vz_drift/c_light:.2f}c "
              f"(impacts left)",
    field_steps=np.array([f[0] for f in field_frames]),
    field_data=np.array([f[1] for f in field_frames]),
    density_pos=np.array([f[1] for f in density_frames]),
    density_neg=np.array([f[2] for f in density_frames]),
    # Periodic correction metadata
    correction_interval=correction_interval,
    correction_steps=np.array([c[0] for c in correction_log]),
    V_before_history=np.array([c[1] for c in correction_log]),
    V_after_history=np.array([c[2] for c in correction_log]),
)

for s in ("left", "right"):
    for hl in box_labels:
        key = f"q_{s}_{hl}"
        save_dict[key] = np.array([q_data[st][f"{s}_{hl}"]
                                   for st in steps_sorted])

np.savez("gauss_data.npz", **save_dict)
print(f"\nSaved gauss_data.npz ({len(field_frames)} frames, "
      f"{len(correction_log)} corrections)", flush=True)

sys.stdout.flush()
os._exit(0)
