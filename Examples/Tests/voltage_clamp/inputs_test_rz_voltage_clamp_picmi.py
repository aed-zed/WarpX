#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Voltage clamp for embedded-boundary conductors (RZ, Yee, Esirkepov).

A rod (cathode) and a sleeve (anode) inside a grounded PEC wall, periodic z unless
--z-faces is given. With --sleeve joined the sleeve reaches the wall, so it is part
of the ground: it is listed with potential 0 and gets no unit solve.

--z-faces current: pec_insulator z faces in current mode (B_r = B_theta = 0), insulator
                from the axis to 0.95 R_WALL, PEC beyond. Rod and sleeve stand on both
                faces. The insulator conserves its charge node by node with the full
                node volume (no flux beyond the face), so the field stays radial and the
                coax has the effective length LENGTH + dz. Corrections must not change
                the face charge.
--z-faces half: the upper half of the --z-faces current coax. The lower face is a PMC
                symmetry plane (reflecting particles), whose nodes hold half a node
                volume, so the coax has the effective length LENGTH + dz/2.
--z-faces voltage: a lower PEC face and an upper pec_insulator face with a voltage gap
                (E_r = K/r, ramped up), with rod and sleeve shortened away from both
                faces. --case vacuum then checks the drive: the plateau mean of the line
                integral must hold the target.

--case vacuum:  capacitance against the discrete coaxial formula, and the clamped
                vacuum field stays static.
--case diode:   electrons emitted from the rod at the Langmuir-Blodgett current.
                The clamp observer must equal the line integral of E_r at every step,
                the free-node Gauss law must hold, and B stays zero. With --arm on
                the gap stays at 1 kV and the grounded charge -psi^T q must equal a
                grounded Poisson solve; with --arm off the gap discharges.
--case restart: --mode base writes a checkpoint at step N_RESTART; --mode restart
                resumes from it and must reproduce the base run.
--case exit:    (current-mode z faces) neutral electron/ion pairs next to a clamped rod
                that stands on both faces; the electrons land on the lower face and
                their charge stays there. The midplane line integral from the rod to the
                wall must stay at the target.
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from pywarpx import callbacks, picmi
from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

parser = argparse.ArgumentParser()
parser.add_argument(
    "--case", choices=["vacuum", "diode", "restart", "exit"], required=True
)
parser.add_argument("--arm", choices=["on", "off"], default="on")
parser.add_argument("--mode", choices=["base", "restart"], default="base")
parser.add_argument("--sleeve", choices=["separate", "joined"], default="separate")
parser.add_argument(
    "--z-faces", choices=["periodic", "current", "half", "voltage"], default="periodic"
)
parser.add_argument("--max-grid-size", type=int, default=1024)
args = parser.parse_args()
if args.case == "exit" and args.z_faces == "periodic":
    args.z_faces = "current"  # the exit case needs z faces
if args.z_faces == "half" and args.case not in ("vacuum", "diode"):
    parser.error("--z-faces half is tested with --case vacuum and diode")

EPS0, Q_E, M_E = picmi.constants.ep0, picmi.constants.q_e, picmi.constants.m_e

# --- Geometry
r2 = "(x*x+y*y)"
if args.case == "exit":
    # A rod alone inside the grounded outer wall, standing on both z faces
    NR, NZ, R_WALL, LENGTH = 40, 80, 1.2e-2, 2.4e-2
    R_ROD, V_CATHODE = 2.0e-3, -10.0
    implicit = f"{R_ROD}*{R_ROD}-{r2}"
    electrodes = [{"name": "rod", "region": "(x<0.0035)", "potential": V_CATHODE}]
else:
    # Rod (cathode) and sleeve (anode) inside the grounded outer wall
    NR, NZ, R_WALL, LENGTH = 48, 8, 6.0e-2, 4.0e-2
    R_ROD, R_SLEEVE_IN, V_CATHODE = 6.2e-3, 5.03e-2, -1000.0
    # A joined sleeve extends beyond the wall; the gap to the rod is the same
    R_SLEEVE_OUT = 5.53e-2 if args.sleeve == "separate" else 2.0 * R_WALL
    rod = f"({R_ROD}*{R_ROD}-{r2})"
    sleeve = (
        f"(({r2}-{R_SLEEVE_IN}*{R_SLEEVE_IN})*({R_SLEEVE_OUT}*{R_SLEEVE_OUT}-{r2}))"
    )
    implicit = f"-({rod})*({sleeve})"
    # The selector only has to separate the two conductors; x is the radius in RZ
    electrodes = [
        {"name": "cathode", "region": "(x<0.03)", "potential": V_CATHODE},
        {"name": "anode", "region": "(x>0.03)", "potential": 0.0},
    ]
dr, dz = R_WALL / NR, LENGTH / NZ
periodic = args.z_faces == "periodic"
half = args.z_faces == "half"  # PMC symmetry plane at the lower face
current = args.z_faces in ("current", "half")  # current-mode pec_insulator faces
if args.z_faces == "voltage":
    # Shortened away from both faces: a conductor touching the gapped face inward of
    # the gap would be at the gap's potential, and one touching the PEC face grounded
    implicit = f"min({implicit},min(z-{1.5 * dz},{LENGTH - 1.5 * dz}-z))"
# Node rows: the periodic copy of row 0 is dropped; with z faces both face rows count
NZ_NODES = NZ if periodic else NZ + 1
# Current-mode faces conserve charge with the full node volume, so the field lines end
# half a cell beyond each such face; a PMC face node holds half a node volume
LENGTH_EFF = LENGTH + (dz / 2 if half else dz) if current else LENGTH
R_INSULATOR = 0.95 * R_WALL  # current mode: insulator from the axis, PEC beyond
# Voltage mode: gap G_IN < r < G_OUT on the upper face, E_r = K/r ramped over N_RAMP
G_IN, G_OUT, K_GAP = 1.0e-2, 4.0e-2, 300.0

# --- Run control
NO_STEP_LIMIT = 10**8  # max_steps is only an upper bound: the script calls sim.step(n)
N_VACUUM_STEPS = 20
N_RESTART = 30  # checkpoint step; the restart case then runs to 2 * N_RESTART
# CTest runs each test in its own directory; the restart test depends on the base test
BASE_DIR = Path("../test_rz_voltage_clamp_checkpoint_picmi")
CORRECTION_INTERVAL_VACUUM = 5  # vacuum: also tests that skipped steps change nothing
GROUNDED_SOLVE_RTOL, GROUNDED_SOLVE_MAX_ITER = 1.0e-13, 200
DIAGNOSTIC_INTERVAL = 100  # steps between the (costlier) Gauss and B checks
BIRTH_FRACTION = 0.1  # birth radius between the geometric and represented rod surface
BIRTH_U = 1.0e6  # initial radial momentum per mass of emitted electrons [m/s]
RESTART_WEIGHT = 2.0e6  # macroparticle weight in the restart case
# --case exit: M_EXIT electron/ion pairs at rest radially, carrying Q_EXIT in total,
# at R_EXIT and z = 1.5 dz; the electrons move to the lower face at UZ_EXIT
M_EXIT, Q_EXIT, UZ_EXIT = 24, 1.0e-12, -2.0e7
R_EXIT = (3.0e-3, 9.0e-3)
ION_MASS = 1.0e6 * M_E  # heavy ions: they stay put during the run
N_EXIT_STEPS = 1200
EXIT_CHECK_STEPS = (5, 30)  # grounded-solve checks while all pairs are present
N_RAMP, N_PLATEAU = 600, 1920  # voltage faces: ramp and plateau of the drive, in steps
N_TRANSITS = 1.2  # diode run length in space-charge-limited transit times

# --- Tolerances (absolute in V or relative as named)
# The discrete coaxial formula is exact for the staircase; the limit is the solver
CAPACITANCE_RTOL = 1.0e-9
VOLTAGE_ATOL = 1.0e-8  # vacuum readback and line integral, 1 kV scale
FIELD_RTOL = 1.0e-12  # E unchanged by repeated corrections, relative to max |E|
B_RTOL = 1.0e-12  # c |B| relative to max |E|
RESTART_FIELD_RTOL = (
    1.0e-12  # particle order may change on restart: roundoff, not bitwise
)
RESTART_VOLTAGE_ATOL = 1.0e-9
LINE_OBSERVER_ATOL = 1.0e-6  # observer vs independent line integral, over ~2500 steps
TARGET_ATOL = 1.0e-9  # observer reading after each correction
GAP_ATOL = 1.0e-6  # line integral at the end of the diode run
# Roundoff accumulates over ~2500 steps; real violations are 1e-3 or larger
GAUSS_RTOL = 1.0e-10
# B_theta stays zero: a z-uniform radial current is balanced by the displacement current
B_ATOL = 1.0e-12
GROUNDED_RTOL = 1.0e-10  # grounded solve vs reciprocity, relative to the induced charge
# The two deposits of a neutral pair cancel exactly on CPU but only to roundoff on GPU
PAIR_RHO_RTOL = 1.0e-12
# Exit: the induced charge of neutral pairs nearly cancels, so the grounded check is
# relative to its gross scale sum |psi q|
EXIT_GROUNDED_RTOL = 1.0e-9
# Exit: mean line integral after the electrons landed (it also sees the undecayed
# induction field of the departed electrons)
EXIT_LINE_ATOL = 0.01
CAP_SYM_RTOL = 1.0e-12  # z faces: C symmetric, relative to its largest entry
FACE_RTOL = 1.0e-10  # current faces: face charge change, relative to the moved charge
# Voltage faces: the drive's ramp rings cavity modes that do not decay; their mean over
# the plateau leaves this much of the line integral
LINE_MEAN_ATOL = 1.0e-2

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    n_azimuthal_modes=1,
    lower_bound=[0.0, 0.0],
    upper_bound=[R_WALL, LENGTH],
    # PICMI has no pec_insulator: z faces other than periodic are set below
    lower_boundary_conditions=["none", "periodic" if periodic else "dirichlet"],
    upper_boundary_conditions=["dirichlet", "periodic" if periodic else "dirichlet"],
    lower_boundary_conditions_particles=[
        "none",
        "periodic" if periodic else ("reflecting" if half else "absorbing"),
    ],
    upper_boundary_conditions_particles=[
        "absorbing",
        "periodic" if periodic else "absorbing",
    ],
    warpx_blocking_factor=8,
    warpx_max_grid_size=args.max_grid_size,
)
restart = args.case == "restart" and args.mode == "restart"
sim = picmi.Simulation(
    solver=picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9),
    max_steps=NO_STEP_LIMIT,
    particle_shape="linear",
    # An EB potential makes WarpX set up Poisson boundaries, which reject pec_insulator
    warpx_embedded_boundary=picmi.EmbeddedBoundary(
        implicit_function=implicit, potential=0.0 if periodic else None
    ),
    warpx_use_filter=False,
    # The clamp requires charge-conserving deposition
    warpx_current_deposition_algo="esirkepov",
    warpx_amr_restart=str(BASE_DIR / "diags" / f"chk{N_RESTART:06d}")
    if restart
    else None,
    verbose=0,
)
sim.add_species(
    picmi.Species(
        name="electrons", particle_type="electron", initial_distribution=None
    ),
    layout=None,
)
if args.case == "exit":
    sim.add_species(picmi.Species(name="ions", charge=Q_E, mass=ION_MASS), layout=None)
if args.case == "restart" and args.mode == "base":
    sim.add_diagnostic(picmi.Checkpoint(period=N_RESTART, name="chk"))
sim.initialize_inputs()
if not periodic:
    from pywarpx import boundary
    from pywarpx import warpx as warpx_inputs

    insulator = warpx_inputs.get_bucket("insulator")
    boundary.field_lo = [
        "none",
        "pmc" if half else ("pec_insulator" if current else "pec"),
    ]
    boundary.field_hi = ["pec", "pec_insulator"]
    if current:
        for side in ("hi",) if half else ("lo", "hi"):
            setattr(insulator, f"area_z_{side}(x,y)", f"x<{R_INSULATOR}")
            setattr(insulator, f"Bx_z_{side}(x,y,t)", "0")
            setattr(insulator, f"By_z_{side}(x,y,t)", "0")
    else:
        # sin^2 ramp over about N_RAMP steps (dt of the Yee CFL condition)
        t_ramp = N_RAMP * 0.9 / (picmi.constants.c * np.sqrt(1 / dr**2 + 1 / dz**2))
        ramp = f"if(t<{t_ramp},sin({np.pi / 2}*t/{t_ramp})^2,1)"
        setattr(insulator, "area_z_hi(x,y)", f"(x>{G_IN})*(x<{G_OUT})")
        setattr(insulator, "Ex_z_hi(x,y,t)", f"{K_GAP}/x*{ramp}")
        setattr(insulator, "Ey_z_hi(x,y,t)", "0")
sim.initialize_warpx()

warpx = sim.extension.warpx
amr = sim.extension.amr
particles = sim.particles.get("electrons")
is_io_rank = amr.ParallelDescriptor.MyProc() == 0

clamp = StaircaseBiasCorrector(
    sim,
    correction_interval=CORRECTION_INTERVAL_VACUUM
    if args.case == "vacuum"
    and args.z_faces != "voltage"  # the drive changes each step
    else 1,
    electrodes=electrodes,
)
if args.sleeve == "joined":
    # A conductor joined to the grounded wall cannot be held at another voltage, a
    # region must not mix it with a free conductor, and a region must select a
    # conductor; the first listed electrode is checked before any unit solve
    rejected = {
        "joined to a grounded PEC wall": dict(electrodes[1], potential=100.0),
        "list them separately": dict(electrodes[1], region="1"),
        "selects no conductor": dict(electrodes[1], region="0"),
    }
    for message, bad in rejected.items():
        try:
            StaircaseBiasCorrector(sim, 1, [bad, electrodes[0]]).setup_after_init()
            raise AssertionError(f"Electrode {bad} was accepted")
        except ValueError as error:
            assert message in str(error), error
clamp.setup_after_init()
# Number of conductors with a unit solve, for the low-level calls below
n = len(clamp.solved)
assert n == len(electrodes) - (args.sleeve == "joined")


def field(name, direction=None):
    """Valid data of a registered field on all ranks (radius, z), z nodes 0 to
    NZ_NODES-1 (without the duplicate periodic z node). pyAMReX global indexing gathers
    across boxes and ranks."""
    if direction is None:
        mf = sim.fields.get(name, level=0)
    else:
        mf = sim.fields.get(name, dir=direction, level=0)
    return mf[...][:, :NZ_NODES]


RZ_DIRECTIONS = ("r", "theta", "z")


def efield(d):
    return field("Efield_fp", RZ_DIRECTIONS[d])


# Represented conductor surfaces at the midplane: the last cathode node (from the
# clamp's weight field of conductor 0) and the first fixed node beyond it (anode).
z_mid = NZ // 2
cathode = np.nonzero(field("voltage_clamp_weight_0")[:, z_mid])[0]
fixed = field("voltage_clamp_fixed")[:, z_mid]
k_cathode = int(cathode.max())
k_anode = int(np.nonzero(fixed[k_cathode + 1 :])[0][0]) + k_cathode + 1
r_node = np.arange(NR + 1) * dr
r_half = (np.arange(NR) + 0.5) * dr


def gap_voltage():
    """phi(cathode) - phi(anode) as the line integral of E_r at the midplane."""
    return float(np.sum(efield(0)[k_cathode:k_anode, z_mid]) * dr)


def gauss_residual():
    """Largest |eps0 div E - rho| times node volume on the free nodes in the gap.

    div E is computed here with the cylindrical Yee divergence because WarpX's nodal
    ComputeDivE has no Python binding and the divE diagnostic is cell-centered."""
    er, ez = efield(0), efield(2)
    rho = charge_density()
    div = np.zeros((NR + 1, NZ_NODES))
    for i in range(1, NR):
        div[i] = (r_half[i] * er[i] - r_half[i - 1] * er[i - 1]) / (r_node[i] * dr)
    div += axial_divergence(ez)
    residual = ((EPS0 * div - rho) * volume).sum(axis=1)
    return float(np.max(np.abs(residual[k_cathode + 1 : k_anode])))


def grounded_check():
    """Plasma-induced charge: reciprocity -psi^T q against a grounded solve."""
    warpx.voltage_clamp_solve_grounded(GROUNDED_SOLVE_RTOL, GROUNDED_SOLVE_MAX_ITER)
    field_charge, grounded = (
        np.asarray(q) for q in warpx.voltage_clamp_charge_state(n)
    )
    # Live charge on the fixed nodes = Gauss charge of live E - field charge
    on_fixed = np.asarray(warpx.voltage_clamp_gauss_charge(n)) - field_charge
    solved = np.asarray(warpx.voltage_clamp_gauss_charge(n, "voltage_clamp_E_grounded"))
    return solved - on_fixed - grounded


def charge_density():
    """Finalized nodal charge density of all species, as the clamp uses it."""
    mpc = warpx.multi_particle_container()
    return mpc.get_charge_density(lev=0, local=False, finalize=True)[...][:, :NZ_NODES]


def axial_divergence(ez):
    """dE_z/dz on the node rows. On current-mode faces there is no flux beyond the face;
    on a PMC face E_z is odd, so the mirror image doubles the flux."""
    if periodic:
        return (ez - np.roll(ez, 1, axis=1)) / dz
    dez = np.zeros((NR + 1, NZ + 1))
    dez[:, :NZ] += ez
    dez[:, 1:] -= ez
    if half:
        dez[:, 0] += ez[:, 0]
    return dez / dz


# Free (neither fixed nor wall) nodes, and the node volumes of the divergence (half a
# volume on a PMC face)
free = field("voltage_clamp_fixed") == 0.0
volume = 2.0 * np.pi * r_node[:, None] * dr * dz * np.ones(NZ_NODES)
volume[0] = np.pi * dr * dr / 4.0 * dz
if half:
    volume[:, 0] *= 0.5
# Node rows of the current-mode faces
face_rows = [NZ] if half else [0, NZ]


def face_charges(name):
    """Current-mode faces: eps0 V div E on the free nodes of the face rows, node by node,
    the charge that the field update conserves there."""
    er, ez = field(name, "r"), field(name, "z")
    div = np.zeros((NR + 1, NZ + 1))
    div[1:NR] = (r_half[1:, None] * er[1:] - r_half[:-1, None] * er[:-1]) / (
        r_node[1:NR, None] * dr
    )
    div[0] = 4.0 * er[0] / dr
    div += axial_divergence(ez)
    q = EPS0 * volume * div
    return np.concatenate([q[free[:, j], j] for j in face_rows])


def capacitance_checks(cap):
    """z faces: C symmetric with a positive diagonal and negative off-diagonal terms."""
    off = cap - np.diag(np.diag(cap))
    return {
        "C symmetric": np.max(np.abs(cap - cap.T)) < CAP_SYM_RTOL * np.max(np.abs(cap)),
        "C diagonal positive": bool(np.all(np.diag(cap) > 0.0)),
        "C off-diagonal negative": bool(
            np.all(off[~np.eye(len(cap), dtype=bool)] < 0.0)
        ),
    }


face_trace = {"jump": 0.0, "correction": 0.0}


def correct():
    """Correct; on current-mode faces also record how much face charge the correction
    changes (largest node), and the sum of the voltage corrections of the cathode."""
    if not current:
        clamp.correct_after_step()
        return
    before = face_charges("Efield_fp")
    face_trace["correction"] += V_CATHODE - clamp.measure_voltage_state()["voltage"][0]
    clamp.correct_after_step()
    jump = np.max(np.abs(face_charges("Efield_fp") - before))
    face_trace["jump"] = max(face_trace["jump"], float(jump))


installed = []


def install(name, func):
    callbacks.installcallback(name, func)
    installed.append((name, func))


def finish(checks, summary):
    if is_io_rank:
        print(f"{summary}; {checks}")
    # Callbacks and Python-held AMReX objects must be released before finalize
    for name, func in installed:
        callbacks.uninstallcallback(name, func)
    gc.collect()
    sim.finalize()
    failures = [k for k, ok in checks.items() if not ok]
    assert not failures, failures
    sys.exit(0)


if args.case == "exit":
    # Neutral pairs; the electrons move toward the lower face
    r = np.linspace(*R_EXIT, M_EXIT)
    weight = Q_EXIT / Q_E / M_EXIT
    for name, uz in (("electrons", UZ_EXIT), ("ions", 0.0)):
        sim.particles.get(name).add_particles(
            x=r, y=np.zeros(M_EXIT), z=np.full(M_EXIT, 1.5 * dz),
            ux=np.zeros(M_EXIT), uy=np.zeros(M_EXIT), uz=np.full(M_EXIT, uz),
            w=np.full(M_EXIT, weight), unique_particles=False,
        )  # fmt: skip
    rho_species = Q_E * weight / (2.0 * np.pi * r[0] * dr * dz)
    clamp.initialize_vacuum_bias(rho_tolerance=PAIR_RHO_RTOL * rho_species)
    psi = field("voltage_clamp_phi_0")
    trace = {"count": [], "observer": [], "line": []}
    grounded_errors = []

    def lower_face_and_row():
        """Charge on the free nodes of the lower face and of the next row. Without
        trajectory cropping, absorption splits the landed charge between the two."""
        q_face = np.sum(face_charges("Efield_fp")[: np.count_nonzero(free[:, 0])])
        er, ez = efield(0)[:, 1], efield(2)
        div = np.zeros(NR + 1)
        div[1:NR] = (r_half[1:] * er[1:] - r_half[:-1] * er[:-1]) / (r_node[1:NR] * dr)
        div[0] = 4.0 * er[0] / dr
        div += (ez[:, 1] - ez[:, 0]) / dz
        q_row = EPS0 * volume[:, 1] * div - charge_density()[:, 1] * volume[:, 1]
        return float(q_face + np.sum(q_row[free[:, 1]]))

    def record():
        step = warpx.getistep(0)
        count = particles.number_of_particles(False)
        trace["count"].append(count)
        trace["observer"].append(clamp.measure_voltage_state()["voltage"][0])
        # From the rod to the grounded wall
        trace["line"].append(float(np.sum(efield(0)[k_cathode:, z_mid]) * dr))
        if count == M_EXIT and step in EXIT_CHECK_STEPS:
            gross = np.sum(np.abs(psi * charge_density()) * volume)
            grounded_errors.append(float(np.max(np.abs(grounded_check()))) / gross)

    landed_initial = lower_face_and_row()
    install("afterstep", correct)
    install("afterstep", record)
    sim.step(N_EXIT_STEPS)
    t = {key: np.asarray(value) for key, value in trace.items()}
    # Second half of the run, after the last electron left
    after = (t["count"] == 0) & (np.arange(1, N_EXIT_STEPS + 1) >= N_EXIT_STEPS // 2)
    line_offset = float(np.mean(t["line"][after]) - V_CATHODE)
    landed = lower_face_and_row() - landed_initial
    # Charge the landed face charge induces on the rod, which an observer pairing psi
    # with the deposited rho alone would miss
    q_face = face_charges("Efield_fp")[: np.count_nonzero(free[:, 0])]
    missed = -np.sum(psi[free[:, 0], 0] * q_face) / clamp.capacitance[0, 0]
    checks = {
        "grounded solve = reciprocity": len(grounded_errors) == len(EXIT_CHECK_STEPS)
        and max(grounded_errors) < EXIT_GROUNDED_RTOL,
        "electrons landed": t["count"][-1] == 0 and after.any(),
        "observer held target": np.max(np.abs(t["observer"] - V_CATHODE))
        < TARGET_ATOL * abs(V_CATHODE),
        "line integral at target": abs(line_offset) < EXIT_LINE_ATOL,
        "landed charge on the face": abs(landed + Q_EXIT) < FACE_RTOL * Q_EXIT,
        "corrections leave the face charge": face_trace["jump"] < FACE_RTOL * Q_EXIT,
        # The line check is meaningful: that charge is many times its tolerance
        "face charge matters": abs(missed) > 10.0 * EXIT_LINE_ATOL,
    }
    finish(
        checks,
        f"exit: line integral - target = {line_offset:.3e} V, landed {landed:.6e} C, "
        f"max face change per correction {face_trace['jump']:.2e} C, face charge seen "
        f"by the rod {missed:.4e} V",
    )

if restart:
    assert warpx.getistep(0) == N_RESTART
    clamp.resume_from_checkpoint()
else:
    clamp.initialize_vacuum_bias()

if args.case == "vacuum" and args.z_faces == "voltage":
    # The gap's potential induces charge on rod and sleeve, which the clamp reads by
    # reciprocity; the plateau mean of the line integral must hold the target
    plateau = []
    worst = {"observer - target": 0.0}

    def record():
        voltage = clamp.measure_voltage_state()["voltage"]
        worst["observer - target"] = max(
            worst["observer - target"], float(np.max(np.abs(voltage - clamp.v_target)))
        )
        if warpx.getistep(0) > N_RAMP + 40:  # after the ramp (the sin^2 tail is flat)
            plateau.append(gap_voltage())

    install("afterstep", clamp.correct_after_step)
    install("afterstep", record)
    sim.step(N_RAMP + 40 + N_PLATEAU)
    line_mean = float(np.mean(plateau))
    # The voltage error the drive's induced charge would cause if it were left out
    grounded = clamp.measure_voltage_state()["grounded_charge"]
    drive_error = float(np.max(np.abs(np.linalg.solve(clamp.capacitance, grounded))))
    checks = capacitance_checks(clamp.capacitance) | {
        "drive term matters": drive_error > 100.0 * LINE_MEAN_ATOL,
        "plateau mean of line integral = target": abs(line_mean - V_CATHODE)
        < LINE_MEAN_ATOL,
        "observer held target": worst["observer - target"] < TARGET_ATOL,
    }
    finish(
        checks,
        f"vacuum, voltage faces: drive term {drive_error:.4e} V, plateau mean of the "
        f"line integral {line_mean:.6f} V, max|observer - target| "
        f"{worst['observer - target']:.2e} V",
    )

if args.case == "vacuum":
    # Discrete coaxial capacitance between the represented rod and sleeve: the
    # series sum of the radial cell "resistances" dr / r_half across the gap
    c_exact = 2.0 * np.pi * EPS0 * LENGTH_EFF / np.sum(dr / r_half[k_cathode:k_anode])
    cap = clamp.capacitance
    e_initial = [efield(d) for d in range(3)]
    voltage_initial = clamp.measure_voltage_state()["voltage"]
    install("afterstep", clamp.correct_after_step)
    sim.step(N_VACUUM_STEPS)
    e_change = max(np.max(np.abs(efield(d) - e_initial[d])) for d in range(3))
    e_scale = max(np.max(np.abs(e)) for e in e_initial)
    b_max = max(np.max(np.abs(field("Bfield_fp", d))) for d in RZ_DIRECTIONS)
    checks = {
        "C = discrete coax": abs(cap[0, 0] / c_exact - 1.0) < CAPACITANCE_RTOL,
        "C symmetric": np.max(np.abs(cap - cap.T)) < CAPACITANCE_RTOL * abs(cap[0, 0]),
        "initial voltage": np.max(np.abs(voltage_initial - clamp.v_target))
        < VOLTAGE_ATOL,
        "line integral": abs(gap_voltage() - V_CATHODE) < VOLTAGE_ATOL,
        "E static": e_change < FIELD_RTOL * e_scale,
        "B zero": picmi.constants.c * b_max < B_RTOL * e_scale,
    }
    if current:
        # The unit fields put no charge on the insulator (largest node)
        unit_face = max(
            np.max(np.abs(face_charges(f"voltage_clamp_E_{k}"))) for k in range(n)
        )
        checks |= capacitance_checks(cap)
        checks["unit fields: no face charge"] = unit_face < FACE_RTOL * cap[0, 0]
    finish(checks, f"vacuum: C = {cap[0, 0]:.10e} F, discrete coax {c_exact:.10e} F")

# Electrons are born between the geometric rod surface (R_ROD) and the represented
# cathode surface (the last fixed node, a_eff > R_ROD). There, a new particle's charge
# lands on fixed nodes, i.e. on the conductor, so creating it does not break the
# discrete Gauss law in the gap (charge created in free vacuum would have no current).
a_eff, b_eff = k_cathode * dr, k_anode * dr
r_birth = R_ROD + BIRTH_FRACTION * (a_eff - R_ROD)
if periodic:
    z_birth = (np.arange(NZ) + 0.5) * dz
else:
    # Current-mode faces: one particle on each node row, so that every row (face rows
    # have full volumes) gets the same charge; the face particles sit just inside
    z_birth = np.clip(np.arange(NZ + 1) * dz, 1.0e-12 * dz, LENGTH - 1.0e-12 * dz)
N_BIRTH = len(z_birth)
# Relative weights: a PMC face row has half a node volume, so its particle carries half
W_BIRTH = np.ones(N_BIRTH)
if half:
    W_BIRTH[0] = 0.5

if args.case == "restart":

    def inject():
        particles.add_particles(
            x=np.full(NZ, r_birth), y=np.zeros(NZ), z=z_birth,
            ux=np.full(NZ, BIRTH_U), uy=np.zeros(NZ), uz=np.zeros(NZ),
            w=np.full(NZ, RESTART_WEIGHT), unique_particles=False,
        )  # fmt: skip

    trace = {"observer": [], "er": []}

    def record():
        if warpx.getistep(0) > N_RESTART:
            trace["observer"].append(clamp.measure_voltage_state()["voltage"])
            trace["er"].append(efield(0))

    install("beforestep", inject)
    install("afterstep", clamp.correct_after_step)
    install("afterstep", record)
    sim.step(2 * N_RESTART - warpx.getistep(0))
    result = {key: np.asarray(value) for key, value in trace.items()}
    checks = {}
    if args.mode == "base":
        if is_io_rank:
            np.savez("restart_trace.npz", **result)
    else:
        base = np.load(BASE_DIR / "restart_trace.npz")
        scale = np.max(np.abs(base["er"]))
        observer_change = np.max(np.abs(result["observer"] - base["observer"]))
        checks = {
            "E_r trace": np.max(np.abs(result["er"] - base["er"]))
            <= RESTART_FIELD_RTOL * scale,
            "observer trace": observer_change <= RESTART_VOLTAGE_ATOL,
            "target held": np.max(np.abs(result["observer"][:, 0] - V_CATHODE))
            <= RESTART_VOLTAGE_ATOL,
        }
    finish(checks, f"restart {args.mode}")


# --case diode: Langmuir-Blodgett current of the represented gap (a_eff, b_eff).
# beta^2 solves 3 b b'' + b'^2 + 4 b b' + b^2 = 1 in gamma = ln(r/a), with the series
# start b = g - 0.4 g^2 near g = 0 (Langmuir & Blodgett 1923, as reviewed in Zhang et al.,
# Appl. Phys. Rev. 4, 011304 (2017), https://doi.org/10.1063/1.4978231).
def lb_rhs(_, y):
    b, db = y
    return [db, (1.0 - b * b - db * db - 4.0 * b * db) / (3.0 * b)]


GAMMA_START = 1.0e-6  # the ODE is singular at gamma = 0; start on the series
lb = solve_ivp(
    lb_rhs,
    (GAMMA_START, np.log(b_eff / a_eff)),
    [GAMMA_START - 0.4 * GAMMA_START**2, 1.0 - 0.8 * GAMMA_START],
    method="DOP853",
    rtol=1.0e-12,
    atol=1.0e-14,
)
beta_end = lb.y[0, -1]
lb_current = (
    8.0 * np.pi * EPS0 / 9.0 * np.sqrt(2.0 * Q_E / M_E) * abs(V_CATHODE) ** 1.5
    * LENGTH_EFF / (b_eff * beta_end ** 2)
)  # fmt: skip
dt = warpx.getdt(0)
# Space-charge-limited transit time: 3 d / v_final (planar Child-Langmuir)
transit = 3.0 * (b_eff - a_eff) / np.sqrt(2.0 * Q_E * abs(V_CATHODE) / M_E)
n_steps = int(N_TRANSITS * transit / dt)
weight = lb_current * dt / (Q_E * W_BIRTH.sum())  # the births of a step carry I dt
injected = 0.0
n_injected = 0
worst = {"line - observer": 0.0, "observer - target": 0.0, "gauss": 0.0, "B": 0.0}


def inject():
    global injected, n_injected
    particles.add_particles(
        x=np.full(N_BIRTH, r_birth), y=np.zeros(N_BIRTH), z=z_birth,
        ux=np.full(N_BIRTH, BIRTH_U), uy=np.zeros(N_BIRTH), uz=np.zeros(N_BIRTH),
        w=weight * W_BIRTH, unique_particles=False,
    )  # fmt: skip
    injected += Q_E * weight * W_BIRTH.sum()
    n_injected += N_BIRTH


def record():
    voltage = clamp.measure_voltage_state()["voltage"]
    gap = float(voltage[0] - voltage[1])  # the anode reads 0 V when joined to the wall
    worst["line - observer"] = max(worst["line - observer"], abs(gap_voltage() - gap))
    worst["observer - target"] = max(worst["observer - target"], abs(gap - V_CATHODE))
    step = warpx.getistep(0)
    if step % DIAGNOSTIC_INTERVAL == 0 or step == n_steps:
        worst["gauss"] = max(worst["gauss"], gauss_residual())
        b_theta = field("Bfield_fp", "theta")
        worst["B"] = max(worst["B"], float(np.max(np.abs(b_theta))))


install("beforestep", inject)
if args.arm == "on":
    install("afterstep", correct)
install("afterstep", record)
sim.step(n_steps)

final_gap = gap_voltage()
collected = particles.number_of_particles(False) < n_injected
checks = {
    "electrons collected": collected,
    "observer = line integral": worst["line - observer"] < LINE_OBSERVER_ATOL,
    "Gauss law": worst["gauss"] / injected < GAUSS_RTOL,
    "B zero": worst["B"] < B_ATOL,
}
summary = f"diode arm={args.arm} steps={n_steps}: gap {final_gap:.4f} V; " + "; ".join(
    f"max|{k}| = {v:.2e}" for k, v in worst.items()
)
if args.arm == "on":
    grounded_error = np.max(np.abs(grounded_check())) / np.sum(
        np.abs(clamp.measure_voltage_state()["grounded_charge"][clamp.solved])
    )
    checks["observer held target"] = worst["observer - target"] < TARGET_ATOL
    checks["gap held"] = abs(final_gap - V_CATHODE) < GAP_ATOL
    checks["grounded solve = reciprocity"] = grounded_error < GROUNDED_RTOL
    summary += f"; grounded check {grounded_error:.2e}"
    if current:
        checks["corrections leave the face charge"] = (
            face_trace["jump"] < FACE_RTOL * injected
        )
        checks["clamp corrected more than 10x the gap tolerance"] = (
            abs(face_trace["correction"]) > 10.0 * GAP_ATOL
        )
        summary += (
            f"; max face change per correction {face_trace['jump']:.2e} C, sum of "
            f"corrections {face_trace['correction']:.4e} V"
        )
else:
    checks["gap discharged"] = abs(final_gap) < 0.5 * abs(V_CATHODE)
finish(checks, summary)
