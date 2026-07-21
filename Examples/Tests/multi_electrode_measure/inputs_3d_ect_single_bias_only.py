#!/usr/bin/env python3
"""Single-correction test (ECT solver): harmonic bias ONLY (no Gauss clean).

Two-species plasma pulse through spherical electrodes.
Uses ECT Maxwell solver instead of Yee.
After plasma clears (step 1300), applies one harmonic bias correction.
Continues 700 steps to check stability.

Usage:
    python inputs_3d_ect_single_bias_only.py
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
correction_step = 1300
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

MODE = "ect_bias_only"

# ---------------------------------------------------------------------------
print("=" * 90)
print(f"SINGLE-CORRECTION TEST — {MODE}")
print("=" * 90)
print(f"Domain: {L_actual*1e2:.2f} cm cube, {nx}^3 cells, dx = {dx*1e3:.3f} mm")
print(f"Total steps: {total_steps}, correction at step {correction_step}")

# ---------------------------------------------------------------------------
# Grid, solver (ECT), EB
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
# Data collection helpers
# ---------------------------------------------------------------------------
sphere_centers = [
    ("left",  -center_offset, 0.0, 0.0, V_left),
    ("right", +center_offset, 0.0, 0.0, V_right),
]
q_data = {}
field_frames = []
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


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
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

    if step == correction_step:
        V_before = corrector.measure_voltages().copy()

        # --- BIAS ONLY ---
        V_target = np.array(corrector.v_target)
        dV = V_target - V_before
        corrector._apply_bias(dV)

        V_after = corrector.measure_voltages().copy()
        correction_log.append(("bias", step, V_before, V_after))
        print(f"\n  === CORRECTION at step {step} ({MODE}) ===")
        print(f"  V before: [{V_before[0]:+.1f}, {V_before[1]:+.1f}]")
        print(f"  V after:  [{V_after[0]:+.1f}, {V_after[1]:+.1f}]\n", flush=True)
        collect(step)


installafterInitEsolve(after_init)
installafterstep(after_step)

# ===========================================================================
# Run simulation
# ===========================================================================
print(f"\n{'='*90}")
print(f"Running {total_steps} steps, single correction at step {correction_step}")
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
    correction_step=correction_step,
    mode=MODE,
    title=f"Single Correction — {MODE}",
    field_steps=np.array([f[0] for f in field_frames]),
    field_data=np.array([f[1] for f in field_frames]),
)

for s in ("left", "right"):
    for hl in box_labels:
        key = f"q_{s}_{hl}"
        save_dict[key] = np.array([q_data[st][f"{s}_{hl}"]
                                   for st in steps_sorted])

for entry in correction_log:
    tag = entry[0]
    save_dict[f"correction_{tag}_step"] = entry[1]
    save_dict[f"correction_{tag}_V_before"] = entry[2]
    save_dict[f"correction_{tag}_V_after"] = entry[3]

outfile = f"single_correction_{MODE}.npz"
np.savez(outfile, **save_dict)
print(f"\nSaved {outfile} ({len(field_frames)} frames)", flush=True)

sys.stdout.flush()
os._exit(0)
