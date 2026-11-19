#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Voltage clamp with pec_insulator z faces (RZ, Yee, Esirkepov).

An isolated ring electrode, clamped at -1 kV, inside a grounded PEC outer wall.

--face current: both z faces are insulators from the axis to R_INSULATOR and PEC
                beyond, in current mode (B_r = B_theta = 0 prescribed). Electrons
                leave the electrode's lower surface and land on the lower face.
                The landed charge must stay on the face (the clamp adds none),
                and the line integral of E_r from the electrode to the wall must
                hold the target after the electrons have landed.
--face voltage: the lower face has a voltage gap (E_r = K/r, ramped up) between an
                inner PEC disc and the grounded PEC beyond; the upper face is PEC.
                The electrode does not touch the gap. The gap's potential induces
                charge on the electrode, which the clamp must account for: the line
                integral must hold the target.
"""

import argparse
import gc
import sys

import numpy as np

from pywarpx import callbacks, picmi
from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

parser = argparse.ArgumentParser()
parser.add_argument("--face", choices=["current", "voltage"], required=True)
parser.add_argument("--max-grid-size", type=int, default=1024)
args = parser.parse_args()

EPS0, Q_E, C_LIGHT = picmi.constants.ep0, picmi.constants.q_e, picmi.constants.c

# --- Geometry: a ring electrode (a < r < b, z1 < z < z2), away from all boundaries.
# Its geometric surfaces lie halfway between nodes.
NR, NZ, R_WALL, LENGTH = 32, 32, 32.0e-3, 32.0e-3
dr, dz = R_WALL / NR, LENGTH / NZ
A_RING, B_RING, Z1_RING, Z2_RING = 10.5e-3, 16.5e-3, 12.5e-3, 20.5e-3
V_TARGET = -1000.0
rr = "sqrt(x*x+y*y)"
ring = (
    f"min(min({rr}-{A_RING},{B_RING}-{rr}),min(z-{Z1_RING},{Z2_RING}-z))"  # > 0 inside
)
electrodes = [{"name": "ring", "region": "1", "potential": V_TARGET}]
# Current mode: insulator from the axis to R_INSULATOR on both faces, PEC beyond
R_INSULATOR = 24.5e-3
# Voltage mode: gap G_IN < r < G_OUT on the lower face, E_r = K/r ramped over T_RAMP
G_IN, G_OUT, K_GAP = 4.5e-3, 9.5e-3, 300.0

# --- Run control
CFL = 0.99
N_STEPS = {"current": 700, "voltage": 1300}[args.face]
N_INJECT = 40  # current: steps with injection
N_PER_STEP = 8  # current: macroparticles injected per step, spread in radius
WEIGHT = 2.0e3  # current: macroparticle weight
V_ELECTRON = 1.0e7  # current: drift speed toward the lower face [m/s]
N_RAMP = 300  # voltage: ramp length in steps

# --- Tolerances
TARGET_ATOL = 1.0e-9  # observer reading after each correction, 1 kV scale
FACE_RTOL = 1.0e-10  # face charge changes, relative to the landed charge
UNIT_FACE_RTOL = 1.0e-10  # face charge of the unit field, relative to C
CAP_RTOL = 1.0e-12
# Line integral of E_r against the target. The clamp holds its charge coordinate
# exactly; the line integral also sees the induction field (dPhi_B/dt), which does not
# decay in this lossless box. Current mode: the electrons leave about 1e-4 V of it.
# Voltage mode: the ramp of the drive rings a cavity mode (+-0.2 V, with or without the
# clamp); its mean over the plateau (about 12 periods) is within 2e-3 V of the target.
LINE_ATOL = 1.0e-3
LINE_MEAN_ATOL = 1.0e-2

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    n_azimuthal_modes=1,
    lower_bound=[0.0, 0.0],
    upper_bound=[R_WALL, LENGTH],
    # PICMI has no pec_insulator: the z faces are set below
    lower_boundary_conditions=["none", "dirichlet"],
    upper_boundary_conditions=["dirichlet", "dirichlet"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=args.max_grid_size,
)
sim = picmi.Simulation(
    solver=picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=CFL),
    max_steps=10**8,
    particle_shape="linear",
    warpx_embedded_boundary=picmi.EmbeddedBoundary(implicit_function=ring),
    warpx_use_filter=False,
    warpx_current_deposition_algo="esirkepov",
    verbose=0,
)
sim.add_species(
    picmi.Species(
        name="electrons", particle_type="electron", initial_distribution=None
    ),
    layout=None,
)
insulator_inputs = {}
if args.face == "current":
    for side in ("lo", "hi"):
        insulator_inputs[f"area_z_{side}(x,y)"] = f"x<{R_INSULATOR}"
        insulator_inputs[f"Bx_z_{side}(x,y,t)"] = "0"
        insulator_inputs[f"By_z_{side}(x,y,t)"] = "0"
else:
    # sin^2 ramp of the gap field over about N_RAMP steps
    t_ramp = N_RAMP * CFL / (C_LIGHT * np.sqrt(1.0 / dr**2 + 1.0 / dz**2))
    ramp = f"if(t<{t_ramp}, sin({np.pi / 2}*t/{t_ramp})^2, 1)"
    insulator_inputs["area_z_lo(x,y)"] = f"(x>{G_IN})*(x<{G_OUT})"
    insulator_inputs["Ex_z_lo(x,y,t)"] = f"{K_GAP}/x*{ramp}"
    insulator_inputs["Ey_z_lo(x,y,t)"] = "0"
sim.initialize_inputs()
from pywarpx import boundary, particles  # noqa: E402  (after the PICMI inputs)
from pywarpx import warpx as warpx_inputs  # noqa: E402

boundary.field_lo = ["none", "pec_insulator"]
boundary.field_hi = ["pec", "pec_insulator" if args.face == "current" else "pec"]
insulator = warpx_inputs.get_bucket("insulator")
for key, value in insulator_inputs.items():
    setattr(insulator, key, value)
# Particles absorbed at the face deposit their current up to the face
particles.crop_on_PEC_boundary = 1
sim.initialize_warpx()

warpx = sim.extension.warpx
electrons = sim.particles.get("electrons")
is_io_rank = sim.extension.amr.ParallelDescriptor.MyProc() == 0

clamp = StaircaseBiasCorrector(sim, correction_interval=1, electrodes=electrodes)
clamp.setup_after_init()
cap = clamp.capacitance[0, 0]


def field(name, direction=None):
    if direction is None:
        return sim.fields.get(name, level=0)[...]
    return sim.fields.get(name, dir=direction, level=0)[...]


r_node = np.arange(NR + 1) * dr
r_half = (np.arange(NR) + 0.5) * dr
# Node volumes of the divergence (axis: r <= dr/2)
volume = 2.0 * np.pi * r_node * dr * dz
volume[0] = np.pi * dr * dr / 4.0 * dz
# Current-mode face nodes: both neighbouring E_r edges in the insulator
junction = int(np.count_nonzero(r_half < R_INSULATOR))
on_face = np.arange(NR + 1) < junction


def lower_face_charge(name):
    """eps0 V [D_r(E_r) + E_z(dz/2)/dz] on the current-mode nodes of the lower face:
    the face charge that the field update conserves (no flux below the face)."""
    er, ez = field(name, "r")[:, 0], field(name, "z")[:, 0]
    div = np.zeros(NR + 1)
    div[0] = 4.0 * er[0] / dr
    div[1:NR] = (r_half[1:] * er[1:] - r_half[:-1] * er[:-1]) / (r_node[1:NR] * dr)
    div += ez / dz
    return float(np.sum((EPS0 * volume * div)[on_face]))


def first_row_charge():
    """eps0 V div E on the free nodes of the first row above the lower face."""
    er, ez = field("Efield_fp", "r")[:, 1], field("Efield_fp", "z")
    div = np.zeros(NR + 1)
    div[0] = 4.0 * er[0] / dr
    div[1:NR] = (r_half[1:] * er[1:] - r_half[:-1] * er[:-1]) / (r_node[1:NR] * dr)
    div += (ez[:, 1] - ez[:, 0]) / dz
    return float(np.sum((EPS0 * volume * div)[:NR]))  # the wall node is not free


# Electrode's outermost fixed node at mid height, for the line integral to the wall
z_mid = NZ // 2
k_out = int(np.nonzero(field("voltage_clamp_weight_0")[:, z_mid])[0].max())


def line_voltage():
    """phi(electrode) - phi(wall) as the line integral of E_r at mid height."""
    return float(np.sum(field("Efield_fp", "r")[k_out:, z_mid]) * dr)


installed = []


def install(name, func):
    callbacks.installcallback(name, func)
    installed.append((name, func))


worst = {"observer - target": 0.0}


def record():
    voltage = clamp.measure_voltage_state()["voltage"][0]
    worst["observer - target"] = max(
        worst["observer - target"], abs(voltage - V_TARGET)
    )


checks = {"C positive": cap > 0.0}
clamp.initialize_vacuum_bias()

if args.face == "current":
    # The unit potential satisfies the field update's face equation: it puts no charge
    # on the insulator (a multigrid Neumann potential would put a few percent of C there)
    unit_face = abs(lower_face_charge("voltage_clamp_E_0"))
    checks["unit field: no face charge"] = unit_face < UNIT_FACE_RTOL * cap

    # Electrons born between the represented and the geometric lower surface of the
    # ring: their charge starts on the electrode's fixed nodes (see the clamp docs)
    weights = field("voltage_clamp_weight_0")
    rows = np.nonzero(weights[(NR // 2) - 2])[0]  # fixed rows at r = 14 mm
    z_birth = 0.5 * (rows.min() * dz + Z1_RING)
    r_birth = np.linspace(11.5e-3, 15.5e-3, N_PER_STEP)
    injected = [0.0]

    def inject():
        if warpx.getistep(0) < N_INJECT:
            electrons.add_particles(
                x=r_birth, y=np.zeros(N_PER_STEP), z=np.full(N_PER_STEP, z_birth),
                ux=np.zeros(N_PER_STEP), uy=np.zeros(N_PER_STEP),
                uz=np.full(N_PER_STEP, -V_ELECTRON),
                w=np.full(N_PER_STEP, WEIGHT), unique_particles=False,
            )  # fmt: skip
            injected[0] -= Q_E * WEIGHT * N_PER_STEP

    trace = {"jump": 0.0, "landed at": None, "row initial": first_row_charge()}

    def correct():
        """Correct, and record how much face charge the correction changes."""
        before = lower_face_charge("Efield_fp")
        clamp.correct_after_step()
        after = lower_face_charge("Efield_fp")
        trace["jump"] = max(trace["jump"], abs(after - before))
        if trace["landed at"] is None and electrons.number_of_particles(False) == 0:
            trace["landed at"] = after

    install("beforestep", inject)
    install("afterstep", correct)
    install("afterstep", record)
    face_initial = lower_face_charge("Efield_fp")
    sim.step(N_STEPS)
    landed = injected[0]
    face_final = lower_face_charge("Efield_fp")
    checks["all electrons landed"] = trace["landed at"] is not None
    # The clamp adds no charge to the insulator ...
    checks["corrections leave the face charge"] = trace["jump"] < FACE_RTOL * abs(
        landed
    )
    # ... and the landed charge stays there while the clamp keeps acting
    checks["landed charge stays on the face"] = trace["landed at"] is not None and abs(
        face_final - trace["landed at"]
    ) < FACE_RTOL * abs(landed)
    # Without trajectory cropping in Esirkepov deposition (#7435), the absorption step
    # leaves a charge dipole between the face and the first row above it (here about
    # 2 % of the landed charge, with or without the clamp). Together they hold exactly
    # the landed charge: everything landed on the insulator.
    row_initial = trace["row initial"]
    face_and_row = face_final - face_initial + first_row_charge() - row_initial
    checks["face + first row = landed charge"] = abs(
        face_and_row - landed
    ) < FACE_RTOL * abs(landed)
    summary = (
        f"current: C = {cap:.6e} F, unit face charge {unit_face:.2e} C, max face change "
        f"per correction {trace['jump']:.2e} C, landed {landed:.6e} C, face "
        f"{face_final - face_initial:.6e} C"
    )
else:
    plateau = []

    def record_line():
        if warpx.getistep(0) > N_RAMP + 40:  # after the ramp (the cos^2 tail is flat)
            plateau.append(line_voltage())

    install("afterstep", clamp.correct_after_step)
    install("afterstep", record)
    install("afterstep", record_line)
    sim.step(N_STEPS)
    line_mean = float(np.mean(plateau))
    checks["plateau mean of line integral = target"] = (
        abs(line_mean - V_TARGET) < LINE_MEAN_ATOL
    )
    # Charge the gap induces on the grounded electrode, and the voltage error it would
    # cause if it were left out
    drive_error = abs(warpx.voltage_clamp_charge_state(1)[1][0]) / cap
    checks["drive term matters"] = drive_error > 100.0 * LINE_MEAN_ATOL
    summary = (
        f"voltage: C = {cap:.6e} F, drive term {drive_error:.4e} V, plateau mean of the "
        f"line integral {line_mean:.6f} V"
    )

if args.face == "current":
    line = line_voltage()
    checks["line integral = target"] = abs(line - V_TARGET) < LINE_ATOL
    summary += f"; line {line:.6f} V"
checks["observer held target"] = worst["observer - target"] < TARGET_ATOL
summary += f"; max|observer - target| {worst['observer - target']:.2e}"

if is_io_rank:
    print(f"{summary}; {checks}")
for name, func in installed:
    callbacks.uninstallcallback(name, func)
gc.collect()
sim.finalize()
failures = [k for k, ok in checks.items() if not ok]
assert not failures, failures
sys.exit(0)
