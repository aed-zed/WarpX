#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Voltage clamp with pec_insulator z faces (RZ, Yee, Esirkepov).

A clamped electrode inside a grounded PEC outer wall. The line integral of E_r from
the electrode to the wall must hold the target.

--face current: both z faces are insulators from the axis to R_INSULATOR and PEC
                beyond, in current mode (B_r = B_theta = 0 prescribed). The clamp must
                not change the charge on the insulator.
  --case exit:  an isolated ring at -1 kV. Electrons leave its lower surface and land
                on the lower face, where their charge must stay.
  --case land:  the ring at +1 kV. Electron-proton pairs are created below it (no net
                charge); the electrons land on the ring.
  --case rod:   a rod at +1 kV that stands on the lower face (it ends on the
                insulator). Pairs are created beside it, whose electrons land on the
                rod, and near the face, whose electrons land on the face.
  --case two:   the ring at -1 kV and a second ring at +300 V, in vacuum: the
                capacitance matrix must be symmetric with a positive diagonal and
                negative off-diagonal terms, and both rings must hold their targets.
  --case reject: runs this script with setups the clamp must reject (a conductor on a
                driven gap, a PEC plate inside a current-mode area, E and B both
                prescribed) and checks that each aborts with its message.
--face voltage: an isolated ring at -1 kV (with --case two, also the second ring;
                with --gap hi, the gap is on the upper face). The lower face has a voltage gap
                (E_r = K/r, ramped up) between an
                inner PEC disc and the grounded PEC beyond; the upper face is PEC.
                The electrode does not touch the gap. The gap's potential induces
                charge on the electrode, which the clamp must account for: the line
                integral must hold the target.
"""

import argparse
import gc
import os
import re
import subprocess
import sys

import numpy as np

from pywarpx import callbacks, picmi
from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

parser = argparse.ArgumentParser()
parser.add_argument("--face", choices=["current", "voltage"], required=True)
parser.add_argument(
    "--case", choices=["exit", "land", "rod", "two", "reject"], default="exit"
)
parser.add_argument("--gap", choices=["lo", "hi"], default="lo")
parser.add_argument("--reject", choices=["gap", "plate", "both"], default=None)
parser.add_argument("--max-grid-size", type=int, default=1024)
args = parser.parse_args()

if args.case == "reject":
    # Each setup must abort in the clamp setup with its message
    expected = {
        ("voltage", "rod", "gap"): "touches a voltage-mode pec_insulator z face",
        ("current", "exit", "plate"): "a PEC plate inside the insulator area floats",
        ("current", "exit", "both"): "not both or neither",
    }
    # Each run is its own MPI singleton: drop the launcher's variables (mpiexec runs this)
    launcher = ("PMI", "PMIX", "HYDI", "HYDRA", "MPIR", "OMPI", "I_MPI")
    env = {k: v for k, v in os.environ.items() if not k.startswith(launcher)}
    failures = []
    for (face, case, reject), message in expected.items():
        run = subprocess.run(
            [
                sys.executable,
                __file__,
                "--face",
                face,
                "--case",
                case,
                "--reject",
                reject,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        # AMReX wraps error messages over lines that start with "#"
        output = " ".join(re.sub(r"[#\s]+", " ", run.stdout + run.stderr).split())
        aborted = run.returncode != 0 and message in output
        print(f"reject {reject}: return code {run.returncode}, message found {aborted}")
        if not aborted:
            failures.append(reject)
    assert not failures, failures
    sys.exit(0)

EPS0, Q_E, C_LIGHT = picmi.constants.ep0, picmi.constants.q_e, picmi.constants.c

# --- Geometry: a ring electrode (a < r < b, z1 < z < z2) away from all boundaries, or
# a rod (r < R_ROD, z < Z_ROD) standing on the lower face. The geometric surfaces lie
# halfway between nodes. Implicit functions are > 0 inside the conductor.
NR, NZ, R_WALL, LENGTH = 32, 32, 32.0e-3, 32.0e-3
dr, dz = R_WALL / NR, LENGTH / NZ
A_RING, B_RING, Z1_RING, Z2_RING = 10.5e-3, 16.5e-3, 12.5e-3, 20.5e-3
R_ROD, Z_ROD = 4.5e-3, 10.5e-3
rr = "sqrt(x*x+y*y)"
if args.case == "rod":
    conductor = f"min({R_ROD}-{rr},{Z_ROD}-z)"
    Z_LINE = 5  # node row of the line integral, through the rod's side
else:
    conductor = f"min(min({rr}-{A_RING},{B_RING}-{rr}),min(z-{Z1_RING},{Z2_RING}-z))"
    Z_LINE = NZ // 2
V_TARGET = 1000.0 if args.case in ("land", "rod") else -1000.0
electrodes = [{"name": "electrode", "region": "1", "potential": V_TARGET}]
two = args.case == "two"
if two:
    # A second ring, outside the first and clear of the faces and the voltage gap
    ring_b = f"min(min({rr}-19.5e-3,22.5e-3-{rr}),min(z-6.5e-3,14.5e-3-z))"
    conductor = f"max({conductor},{ring_b})"
    electrodes = [
        {"name": "ring", "region": "(x<0.018)", "potential": V_TARGET},
        {"name": "ring_b", "region": "(x>0.018)", "potential": 300.0},
    ]
# Current mode: insulator from the axis to R_INSULATOR on both faces, PEC beyond
R_INSULATOR = 24.5e-3
# Voltage mode: gap G_IN < r < G_OUT on the lower face, E_r = K/r ramped over T_RAMP
G_IN, G_OUT, K_GAP = 4.5e-3, 9.5e-3, 300.0

# --- Run control
CFL = 0.99
N_STEPS = {"current": 700, "voltage": 2560}[args.face]
N_INJECT = 40  # current: steps with injection
N_PER_STEP = 8  # current: macroparticles injected per step and population
# current: macroparticle weight; land: more, since the protons left below the ring
# induce part of the collected charge back
WEIGHT = 6.0e3 if args.case == "land" else 2.0e3
V_ELECTRON = 1.0e7  # exit: drift speed toward the lower face [m/s]
V_TO_FACE = 3.0e7  # rod: initial speed of the electrons sent to the face [m/s]
N_RAMP = 600  # voltage: ramp length in steps
N_TWO = 200  # current, two rings: steps in vacuum

# --- Tolerances
TARGET_ATOL = 1.0e-9  # observer reading after each correction, 1 kV scale
FACE_RTOL = 1.0e-10  # face charge changes, relative to the landed charge
UNIT_FACE_RTOL = 1.0e-10  # face charge of the unit field, relative to C
CAP_RTOL = 1.0e-12  # C symmetry, relative to the largest entry (seen: 2e-14)
# Line integral of E_r against the target. The clamp holds its charge coordinate
# exactly; the line integral also sees the induction field (dPhi_B/dt), which does not
# decay in this lossless box. Current mode: the electrons leave about 1e-4 V of it.
# Voltage mode: the ramp of the drive rings cavity modes (with or without the clamp).
# With this ramp and plateau (about 24 periods) the plateau mean is within 1e-4 V of the
# target for both rings; it changes sign with the plateau length and falls with a
# slower ramp, so it is ringing, not a bias.
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
    warpx_embedded_boundary=picmi.EmbeddedBoundary(implicit_function=conductor),
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
pairs = args.face == "current" and args.case in ("land", "rod")
if pairs:
    sim.add_species(
        picmi.Species(
            name="protons", particle_type="proton", initial_distribution=None
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
    insulator_inputs[f"area_z_{args.gap}(x,y)"] = f"(x>{G_IN})*(x<{G_OUT})"
    insulator_inputs[f"Ex_z_{args.gap}(x,y,t)"] = f"{K_GAP}/x*{ramp}"
    insulator_inputs[f"Ey_z_{args.gap}(x,y,t)"] = "0"
if args.reject == "plate":  # a PEC plate r < G_IN inside the insulator area
    for side in ("lo", "hi"):
        insulator_inputs[f"area_z_{side}(x,y)"] = f"(x>{G_IN})*(x<{R_INSULATOR})"
if args.reject == "both":
    insulator_inputs["Ex_z_lo(x,y,t)"] = "0"
sim.initialize_inputs()
from pywarpx import boundary  # noqa: E402  (after the PICMI inputs)
from pywarpx import warpx as warpx_inputs  # noqa: E402

insulating = {"lo": args.face == "current" or args.gap == "lo"}
insulating["hi"] = args.face == "current" or args.gap == "hi"
boundary.field_lo = ["none", "pec_insulator" if insulating["lo"] else "pec"]
boundary.field_hi = ["pec", "pec_insulator" if insulating["hi"] else "pec"]
insulator = warpx_inputs.get_bucket("insulator")
for key, value in insulator_inputs.items():
    setattr(insulator, key, value)
sim.initialize_warpx()

warpx = sim.extension.warpx
electrons = sim.particles.get("electrons")
protons = sim.particles.get("protons") if pairs else None
is_io_rank = sim.extension.amr.ParallelDescriptor.MyProc() == 0

clamp = StaircaseBiasCorrector(sim, correction_interval=1, electrodes=electrodes)
clamp.setup_after_init()
if args.reject is not None:  # the setup should have aborted
    sim.finalize()
    sys.exit(0)
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
# Free (not fixed, not wall) nodes; on a face these are the insulator nodes that no
# conductor covers
free = field("voltage_clamp_fixed") == 0.0


def radial_divergence(er):
    div = np.zeros(NR + 1)
    div[0] = 4.0 * er[0] / dr
    div[1:NR] = (r_half[1:] * er[1:] - r_half[:-1] * er[:-1]) / (r_node[1:NR] * dr)
    return div


def face_charges(name, side):
    """eps0 V [D_r(E_r) +- E_z(first cell)/dz] on the free nodes of a face, node by node:
    the face charge that the field update conserves (no flux beyond the face)."""
    row, cell, sign = (0, 0, 1.0) if side == "lo" else (NZ, NZ - 1, -1.0)
    er, ez = field(name, "r")[:, row], field(name, "z")[:, cell]
    div = radial_divergence(er) + sign * ez / dz
    return (EPS0 * volume * div)[free[:, row]]


def lower_face_charge(name):
    return float(np.sum(face_charges(name, "lo")))


def all_face_charges(name):
    return np.concatenate([face_charges(name, "lo"), face_charges(name, "hi")])


def first_row_charge():
    """eps0 V div E on the free nodes of the first row above the lower face."""
    er, ez = field("Efield_fp", "r")[:, 1], field("Efield_fp", "z")
    div = radial_divergence(er) + (ez[:, 1] - ez[:, 0]) / dz
    return float(np.sum((EPS0 * volume * div)[free[:, 1]]))


# Electrode's outermost fixed node on row Z_LINE, for the line integral to the wall
k_out = int(np.nonzero(field("voltage_clamp_weight_0")[:, Z_LINE])[0].max())


def line_voltage():
    """phi(electrode) - phi(wall) as the line integral of E_r along row Z_LINE."""
    return float(np.sum(field("Efield_fp", "r")[k_out:, Z_LINE]) * dr)


installed = []


def install(name, func):
    callbacks.installcallback(name, func)
    installed.append((name, func))


worst = {"observer - target": 0.0}


def record():
    voltage = clamp.measure_voltage_state()["voltage"]
    worst["observer - target"] = max(
        worst["observer - target"], float(np.max(np.abs(voltage - clamp.v_target)))
    )


checks = {"C positive": cap > 0.0}
if two:
    C = clamp.capacitance
    checks["C symmetric"] = abs(C[0, 1] - C[1, 0]) < CAP_RTOL * np.max(np.abs(C))
    checks["C diagonal positive"] = bool(np.all(np.diag(C) > 0.0))
    checks["C off-diagonal negative"] = C[0, 1] < 0.0 and C[1, 0] < 0.0
clamp.initialize_vacuum_bias()

if args.face == "current" and two:
    install("afterstep", clamp.correct_after_step)
    install("afterstep", record)
    sim.step(N_TWO)
    summary = f"current two: C = {clamp.capacitance.tolist()} F"
elif args.face == "current":
    # The unit potential satisfies the field update's face equation: it puts no charge
    # on the insulator (a multigrid Neumann potential would put a few percent of C there)
    # (largest node, both faces)
    unit_face = float(np.max(np.abs(all_face_charges("voltage_clamp_E_0"))))
    checks["unit field: no face charge"] = unit_face < UNIT_FACE_RTOL * cap

    injected = {"all": 0.0, "face": 0.0}

    def add(species, r, z, uz, charge):
        n = len(r)
        species.add_particles(
            x=r, y=np.zeros(n), z=np.full(n, z), ux=np.zeros(n), uy=np.zeros(n),
            uz=np.full(n, uz), w=np.full(n, WEIGHT), unique_particles=False,
        )  # fmt: skip
        return charge * WEIGHT * n

    if args.case == "exit":
        # Electrons born between the represented and the geometric lower surface of the
        # ring: their charge starts on the ring's fixed nodes (see the clamp docs)
        rows = np.nonzero(field("voltage_clamp_weight_0")[(NR // 2) - 2])[0]
        z_birth = 0.5 * (rows.min() * dz + Z1_RING)
        r_birth = np.linspace(11.5e-3, 15.5e-3, N_PER_STEP)

        def inject():
            if warpx.getistep(0) < N_INJECT:
                q = add(electrons, r_birth, z_birth, -V_ELECTRON, -Q_E)
                injected["all"] += q
                injected["face"] += q

    else:
        # Electron-proton pairs at rest create no net charge. The protons hardly move.
        # land: below the ring; rod: beside the rod, and near the face, where the
        # electrons are sent to the face fast enough that the rod does not catch them.
        sources = (
            [(np.linspace(11.5e-3, 15.5e-3, N_PER_STEP), 9.5e-3, 0.0, False)]
            if args.case == "land"
            else [
                (np.linspace(7.5e-3, 9.5e-3, N_PER_STEP), 6.5e-3, 0.0, False),
                (np.linspace(18.5e-3, 20.5e-3, N_PER_STEP), 2.5e-3, -V_TO_FACE, True),
            ]
        )

        def inject():
            if warpx.getistep(0) < N_INJECT:
                for r_birth, z_birth, uz, to_face in sources:
                    q = add(electrons, r_birth, z_birth, uz, -Q_E)
                    add(protons, r_birth, z_birth, 0.0, Q_E)
                    injected["all"] += q
                    if to_face:
                        injected["face"] += q

    trace = {"jump": 0.0, "landed at": None, "row initial": first_row_charge()}
    trace["correction"] = 0.0  # sum of the clamp's voltage corrections of the electrode

    def correct():
        """Correct, and record the correction and how much face charge it changes."""
        before = all_face_charges("Efield_fp")
        trace["correction"] += V_TARGET - clamp.measure_voltage_state()["voltage"][0]
        clamp.correct_after_step()
        after = all_face_charges("Efield_fp")
        trace["jump"] = max(trace["jump"], float(np.max(np.abs(after - before))))
        if trace["landed at"] is None and electrons.number_of_particles(False) == 0:
            trace["landed at"] = float(np.sum(after[: np.count_nonzero(free[:, 0])]))

    install("beforestep", inject)
    install("afterstep", correct)
    install("afterstep", record)
    face_initial = lower_face_charge("Efield_fp")
    upper_initial = float(np.sum(face_charges("Efield_fp", "hi")))
    sim.step(N_STEPS)
    landed = injected["face"]  # charge landed on the face
    scale = abs(injected["all"])
    face_final = lower_face_charge("Efield_fp")
    checks["all electrons landed"] = trace["landed at"] is not None
    # The clamp adds no charge to the insulator (largest node, both faces) ...
    checks["corrections leave the face charge"] = trace["jump"] < FACE_RTOL * scale
    upper_change = float(np.sum(face_charges("Efield_fp", "hi"))) - upper_initial
    checks["upper face charge unchanged"] = abs(upper_change) < FACE_RTOL * scale
    # The line-integral check is meaningful: without the clamp the electrode would have
    # drifted by the sum of the corrections, many times the tolerance
    checks["clamp corrected more than 10x the line tolerance"] = (
        abs(trace["correction"]) > 10.0 * LINE_ATOL
    )
    # ... and the landed charge stays there while the clamp keeps acting
    checks["landed charge stays on the face"] = (
        trace["landed at"] is not None
        and abs(face_final - trace["landed at"]) < FACE_RTOL * scale
    )
    # Without trajectory cropping in Esirkepov deposition (#7435), the absorption step
    # leaves a charge dipole between the face and the first row above it (in the exit
    # case about 2 % of the landed charge, with or without the clamp). Together they
    # hold exactly the charge that landed on the face.
    face_and_row = face_final - face_initial + first_row_charge() - trace["row initial"]
    checks["face + first row = charge landed on the face"] = (
        abs(face_and_row - landed) < FACE_RTOL * scale
    )
    summary = (
        f"current {args.case}: C = {cap:.6e} F, unit face charge {unit_face:.2e} C, max "
        f"face change per correction {trace['jump']:.2e} C, injected "
        f"{injected['all']:.6e} C, to the face {landed:.6e} C, face "
        f"{face_final - face_initial:.6e} C, face + first row {face_and_row:.6e} C, "
        f"upper face {upper_change:.2e} C, sum of corrections {trace['correction']:.4e} V"
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
    drive_error = abs(clamp.measure_voltage_state()["grounded_charge"][0]) / cap
    checks["drive term matters"] = drive_error > 100.0 * LINE_MEAN_ATOL
    summary = (
        f"voltage{' two' if two else ''}: C = {clamp.capacitance.tolist()} F, drive term {drive_error:.4e} V, plateau mean of the "
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
