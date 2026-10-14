#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Voltage clamp for embedded-boundary conductors (RZ, Yee, Esirkepov).

A rod (cathode) and a sleeve (anode) inside a grounded PEC wall, periodic z unless
--z-faces pmc is given.

--case vacuum:  capacitance against the discrete coaxial formula, and the clamped
                vacuum field stays static. --wall-reference joins the sleeve to the
                wall, so it is part of the ground and only the rod is clamped.
--case diode:   electrons emitted from the rod at the Langmuir-Blodgett current.
                The clamp observer must equal the line integral of E_r at every step,
                the free-node Gauss law must hold, and B stays zero. With --arm on
                the gap stays at 1 kV and the grounded charge -psi^T q must equal a
                grounded Poisson solve; with --arm off the gap discharges.
--case restart: --mode base writes a checkpoint at step N; --mode restart resumes
                from it and must reproduce the base run.
--case exit:    (PMC z faces) neutral electron/ion pairs next to a clamped rod; the
                electrons leave through a PMC face and leave their charge on it. The
                midplane line integral from the rod to the wall must stay at the target.
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

from pywarpx import callbacks, picmi
from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

parser = argparse.ArgumentParser()
parser.add_argument(
    "--case", choices=["vacuum", "diode", "restart", "exit"], required=True
)
parser.add_argument("--arm", choices=["on", "off"], default="on")
parser.add_argument("--mode", choices=["base", "restart"], default="base")
parser.add_argument("--z-faces", choices=["periodic", "pmc"], default="periodic")
parser.add_argument("--wall-reference", action="store_true")
parser.add_argument("--max-grid-size", type=int, default=1024)
args = parser.parse_args()
if args.case == "exit":
    args.z_faces = "pmc"

COMM = MPI.COMM_WORLD
EPS0, Q_E, M_E = picmi.constants.ep0, picmi.constants.q_e, picmi.constants.m_e
N_RESTART = 30
BASE_DIR = Path("../test_rz_voltage_clamp_restart_base_picmi")
r2 = "(x*x+y*y)"
if args.case == "exit":
    NR, NZ, R_WALL, LENGTH = 40, 80, 1.2e-2, 2.4e-2
    R_ROD, V_CATHODE = 2.0e-3, -10.0
    implicit = f"{R_ROD}*{R_ROD}-{r2}"
    electrodes = [{"name": "rod", "region": "(x<0.0035)", "potential": V_CATHODE}]
else:
    NR, NZ, R_WALL, LENGTH = 48, 8, 6.0e-2, 4.0e-2
    R_ROD, R_SLEEVE_IN, V_CATHODE = 6.2e-3, 5.03e-2, -1000.0
    # With the wall reference the sleeve extends into the outer PEC wall
    R_SLEEVE_OUT = R_WALL + 0.01 if args.wall_reference else 5.53e-2
    rod = f"({R_ROD}*{R_ROD}-{r2})"
    sleeve = (
        f"(({r2}-{R_SLEEVE_IN}*{R_SLEEVE_IN})*({R_SLEEVE_OUT}*{R_SLEEVE_OUT}-{r2}))"
    )
    implicit = f"-({rod})*({sleeve})"
    electrodes = [{"name": "cathode", "region": "(x<0.03)", "potential": V_CATHODE}]
    if not args.wall_reference:
        electrodes.append({"name": "anode", "region": "(x>0.03)", "potential": 0.0})
dr, dz = R_WALL / NR, LENGTH / NZ
z_field = "neumann" if args.z_faces == "pmc" else "periodic"
z_particles = "absorbing" if args.z_faces == "pmc" else "periodic"

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    n_azimuthal_modes=1,
    lower_bound=[0.0, 0.0],
    upper_bound=[R_WALL, LENGTH],
    lower_boundary_conditions=["none", z_field],
    upper_boundary_conditions=["dirichlet", z_field],
    lower_boundary_conditions_particles=["none", z_particles],
    upper_boundary_conditions_particles=["absorbing", z_particles],
    warpx_blocking_factor=8,
    warpx_max_grid_size=args.max_grid_size,
)
restart = args.case == "restart" and args.mode == "restart"
sim = picmi.Simulation(
    solver=picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9),
    max_steps=10**8,
    particle_shape="linear",
    warpx_embedded_boundary=picmi.EmbeddedBoundary(
        implicit_function=implicit, potential=0.0
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
    sim.add_species(
        picmi.Species(name="ions", charge=Q_E, mass=1.0e6 * M_E), layout=None
    )
if args.case == "restart" and args.mode == "base":
    sim.add_diagnostic(picmi.Checkpoint(period=N_RESTART, name="chk"))
sim.initialize_inputs()
sim.initialize_warpx()

from pywarpx._libwarpx import libwarpx  # noqa: E402

warpx = libwarpx.libwarpx_so.get_instance()
register = warpx.multifab_register()
direction = libwarpx.libwarpx_so.Direction
particles = sim.particles.get("electrons")

clamp = StaircaseBiasCorrector(
    sim,
    correction_interval=5 if args.case == "vacuum" else 1,
    electrodes=electrodes,
    grounded_wall_reference=args.wall_reference,
)
clamp.setup_after_init()
n = clamp.n


def gather(mf):
    """Valid data of a field on all ranks, as one owned array (radius, z)."""
    shape = (NR + 1, NZ + 1)
    total, count = np.zeros(shape), np.zeros(shape)
    for mfi, arr in zip(mf, mf.to_numpy(copy=True)):
        v, f = mfi.validbox(), mfi.fabbox()
        lo = [v.small_end[d] - f.small_end[d] for d in range(2)]
        hi = [v.big_end[d] - f.small_end[d] + 1 for d in range(2)]
        part = np.squeeze(arr[lo[0] : hi[0], lo[1] : hi[1]])
        cells = (
            slice(v.small_end[0], v.big_end[0] + 1),
            slice(v.small_end[1], v.big_end[1] + 1),
        )
        total[cells] += part
        count[cells] += 1
    total, count = COMM.allreduce(total), COMM.allreduce(count)
    return np.divide(total, count, out=np.zeros_like(total), where=count > 0)[:, :NZ]


def efield(d):
    return gather(register.get("Efield_fp", dir=direction(d), level=0))


# Represented conductor surfaces: last cathode node and first fixed node beyond it
z_mid = NZ // 2
cathode = np.nonzero(gather(register.get("voltage_clamp_weight_0", level=0))[:, z_mid])[
    0
]
fixed = gather(register.get("voltage_clamp_fixed", level=0))[:, z_mid]
k_cathode = int(cathode.max())
k_anode = int(np.nonzero(fixed[k_cathode + 1 :])[0][0]) + k_cathode + 1
r_node = np.arange(NR + 1) * dr
r_half = (np.arange(NR) + 0.5) * dr


def gap_voltage():
    """phi(cathode) - phi(anode) as the line integral of E_r at the midplane."""
    return float(np.sum(efield(0)[k_cathode:k_anode, z_mid]) * dr)


def gauss_residual():
    """Largest |eps0 div E - rho| times node volume on the free nodes in the gap."""
    er, ez = efield(0), efield(2)
    rho = gather(warpx.deposit_scratch_rho(0))
    div = np.zeros((NR + 1, NZ))
    for i in range(1, NR):
        div[i] = (r_half[i] * er[i] - r_half[i - 1] * er[i - 1]) / (r_node[i] * dr)
    div += (ez - np.roll(ez, 1, axis=1)) / dz
    volume = 2.0 * np.pi * r_node * dr * dz
    residual = ((EPS0 * div - rho) * volume[:, None]).sum(axis=1)
    return float(np.max(np.abs(residual[k_cathode + 1 : k_anode])))


def grounded_check():
    """Plasma-induced charge: reciprocity -psi^T q against a grounded solve."""
    warpx.voltage_clamp_solve_grounded(1.0e-13, 200, args.wall_reference)
    field_charge, grounded = (
        np.asarray(q) for q in warpx.voltage_clamp_charge_state(n)
    )
    # Live charge on the fixed nodes = Gauss charge of live E - field charge
    on_fixed = np.asarray(warpx.voltage_clamp_gauss_charge(n)) - field_charge
    solved = np.asarray(warpx.voltage_clamp_gauss_charge(n, "voltage_clamp_E_grounded"))
    return solved - on_fixed - grounded


installed = []


def install(name, func):
    callbacks.installcallback(name, func)
    installed.append((name, func))


def finish(checks, summary):
    if COMM.rank == 0:
        print(f"{summary}; {checks}")
    for name, func in installed:
        callbacks.uninstallcallback(name, func)
    gc.collect()
    sim.finalize()
    failures = [k for k, ok in checks.items() if not ok]
    assert not failures, failures
    sys.exit(0)


if args.case == "exit":
    # Neutral pairs at rest; the electrons move toward the lower PMC face
    m = 24
    r = np.linspace(3.0e-3, 9.0e-3, m)
    weight = 1.0e-12 / Q_E / m
    for name, uz in (("electrons", -2.0e7), ("ions", 0.0)):
        sim.particles.get(name).add_particles(
            x=r, y=np.zeros(m), z=np.full(m, 1.5 * dz),
            ux=np.zeros(m), uy=np.zeros(m), uz=np.full(m, uz),
            w=np.full(m, weight), unique_particles=False,
        )  # fmt: skip
    # The two deposits cancel exactly on CPU but only to roundoff on GPU
    rho_species = Q_E * weight / (2.0 * np.pi * r[0] * dr * dz)
    clamp.initialize_vacuum_bias(rho_tolerance=1.0e-12 * rho_species)
    psi = gather(register.get("voltage_clamp_phi_0", level=0))
    volume = 2.0 * np.pi * r_node[:, None] * dr * dz
    trace = {"count": [], "observer": [], "line": []}
    grounded_errors = []

    def record():
        step = warpx.getistep(0)
        count = particles.number_of_particles(False)
        trace["count"].append(count)
        trace["observer"].append(clamp.measure_voltage_state()["voltage"][0])
        # From the rod to the grounded wall
        trace["line"].append(float(np.sum(efield(0)[k_cathode:, z_mid]) * dr))
        if count == m and step in (5, 30):
            # The induced charge of neutral pairs nearly cancels: use its gross scale
            gross = np.sum(np.abs(psi * gather(warpx.deposit_scratch_rho(0))) * volume)
            grounded_errors.append(float(np.max(np.abs(grounded_check()))) / gross)

    install("afterstep", clamp.correct_after_step)
    install("afterstep", record)
    steps = 1200
    sim.step(steps)
    t = {key: np.asarray(value) for key, value in trace.items()}
    after = (t["count"] == 0) & (np.arange(1, steps + 1) >= steps // 2)
    line_offset = float(np.mean(t["line"][after]) - V_CATHODE)
    checks = {
        "grounded solve = reciprocity": len(grounded_errors) == 2
        and max(grounded_errors) < 1.0e-9,
        "electrons left": t["count"][-1] == 0 and after.any(),
        "observer held target": np.max(np.abs(t["observer"] - V_CATHODE))
        < 1.0e-9 * abs(V_CATHODE),
        # Without the face charge in the grounded charge this is 0.1 V or more
        "line integral at target": abs(line_offset) < 0.01,
    }
    finish(checks, f"exit: line integral - target = {line_offset:.3e} V")

if restart:
    assert warpx.getistep(0) == N_RESTART
    clamp.resume_from_checkpoint()
else:
    clamp.initialize_vacuum_bias()

if args.case == "vacuum":
    # Discrete coaxial capacitance between the represented rod and sleeve
    c_exact = 2.0 * np.pi * EPS0 * LENGTH / np.sum(dr / r_half[k_cathode:k_anode])
    cap = clamp.capacitance
    e_initial = [efield(d) for d in range(3)]
    voltage_initial = clamp.measure_voltage_state()["voltage"]
    install("afterstep", clamp.correct_after_step)
    sim.step(20)
    e_change = max(np.max(np.abs(efield(d) - e_initial[d])) for d in range(3))
    e_scale = max(np.max(np.abs(e)) for e in e_initial)
    b_max = max(
        np.max(np.abs(gather(register.get("Bfield_fp", dir=direction(d), level=0))))
        for d in range(3)
    )
    checks = {
        "C = discrete coax": abs(cap[0, 0] / c_exact - 1.0) < 1.0e-9,
        "C symmetric": np.max(np.abs(cap - cap.T)) < 1.0e-9 * abs(cap[0, 0]),
        "initial voltage": np.max(np.abs(voltage_initial - clamp.v_target)) < 1.0e-8,
        "line integral": abs(gap_voltage() - V_CATHODE) < 1.0e-8,
        "E static": e_change < 1.0e-12 * e_scale,
        "B zero": picmi.constants.c * b_max < 1.0e-12 * e_scale,
    }
    finish(checks, f"vacuum: C = {cap[0, 0]:.10e} F, discrete coax {c_exact:.10e} F")

# Electrons born just outside the represented cathode surface, which lies
# outside the geometric rod; birth between the two keeps the emitted charge
# on fixed nodes, so emission does not break the discrete Gauss law.
a_eff, b_eff = k_cathode * dr, k_anode * dr
r_birth = R_ROD + 0.1 * (a_eff - R_ROD)
z_birth = (np.arange(NZ) + 0.5) * dz

if args.case == "restart":
    weight = 2.0e6

    def inject():
        particles.add_particles(
            x=np.full(NZ, r_birth), y=np.zeros(NZ), z=z_birth,
            ux=np.full(NZ, 1.0e6), uy=np.zeros(NZ), uz=np.zeros(NZ),
            w=np.full(NZ, weight), unique_particles=False,
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
        if COMM.rank == 0:
            np.savez("restart_trace.npz", **result)
    else:
        base = np.load(BASE_DIR / "restart_trace.npz")
        scale = np.max(np.abs(base["er"]))
        observer_change = np.max(np.abs(result["observer"] - base["observer"]))
        # Particle order may change on restart: compare to roundoff, not bitwise
        checks = {
            "E_r trace": np.max(np.abs(result["er"] - base["er"])) <= 1.0e-12 * scale,
            "observer trace": observer_change <= 1.0e-9,
            "target held": np.max(np.abs(result["observer"][:, 0] - V_CATHODE))
            <= 1.0e-9,
        }
    finish(checks, f"restart {args.mode}")

# --case diode: Langmuir-Blodgett current of the represented gap (beta^2 from its ODE)
gamma = np.linspace(1.0e-6, np.log(b_eff / a_eff), 20001)
beta, dbeta = gamma - 0.4 * gamma**2, 1.0 - 0.8 * gamma
for i in range(1, gamma.size):  # midpoint rule on 3 b b'' + b'^2 + 4 b b' + b^2 = 1
    h, b, db = gamma[i] - gamma[i - 1], beta[i - 1], dbeta[i - 1]
    acc = (1.0 - b * b - db * db - 4.0 * b * db) / (3.0 * b)
    bm, dbm = b + 0.5 * h * db, db + 0.5 * h * acc
    accm = (1.0 - bm * bm - dbm * dbm - 4.0 * bm * dbm) / (3.0 * bm)
    beta[i], dbeta[i] = b + h * dbm, db + h * accm
current = (
    8.0 * np.pi * EPS0 / 9.0 * np.sqrt(2.0 * Q_E / M_E) * abs(V_CATHODE) ** 1.5
    * LENGTH / (b_eff * beta[-1] ** 2)
)  # fmt: skip
dt = warpx.getdt(0)
transit = 3.0 * (b_eff - a_eff) / np.sqrt(2.0 * Q_E * abs(V_CATHODE) / M_E)
n_steps = int(1.2 * transit / dt)
weight = current * dt / (Q_E * NZ)
injected = 0.0
worst = {"line - observer": 0.0, "observer - target": 0.0, "gauss": 0.0, "B": 0.0}


def inject():
    global injected
    particles.add_particles(
        x=np.full(NZ, r_birth), y=np.zeros(NZ), z=z_birth,
        ux=np.full(NZ, 1.0e6), uy=np.zeros(NZ), uz=np.zeros(NZ),
        w=np.full(NZ, weight), unique_particles=False,
    )  # fmt: skip
    injected += Q_E * weight * NZ


def record():
    voltage = clamp.measure_voltage_state()["voltage"]
    gap = float(voltage[0] - voltage[1])
    worst["line - observer"] = max(worst["line - observer"], abs(gap_voltage() - gap))
    worst["observer - target"] = max(worst["observer - target"], abs(gap - V_CATHODE))
    step = warpx.getistep(0)
    if step % 100 == 0 or step == n_steps:
        worst["gauss"] = max(worst["gauss"], gauss_residual())
        bt = gather(register.get("Bfield_fp", dir=direction(1), level=0))
        worst["B"] = max(worst["B"], float(np.max(np.abs(bt))))


install("beforestep", inject)
if args.arm == "on":
    install("afterstep", clamp.correct_after_step)
install("afterstep", record)
sim.step(n_steps)

final_gap = gap_voltage()
collected = particles.number_of_particles(False) < round(injected / (Q_E * weight))
checks = {
    "electrons collected": collected,
    "observer = line integral": worst["line - observer"] < 1.0e-6,
    # roundoff accumulates over ~2500 steps; real violations are 1e-3 or larger
    "Gauss law": worst["gauss"] / injected < 1.0e-10,
    "B zero": worst["B"] < 1.0e-12,
}
summary = f"diode arm={args.arm} steps={n_steps}: gap {final_gap:.4f} V; " + "; ".join(
    f"max|{k}| = {v:.2e}" for k, v in worst.items()
)
if args.arm == "on":
    grounded_error = np.max(np.abs(grounded_check())) / np.sum(
        np.abs(clamp.measure_voltage_state()["grounded_charge"])
    )
    checks["observer held target"] = worst["observer - target"] < 1.0e-9
    checks["gap held"] = abs(final_gap - V_CATHODE) < 1.0e-6
    checks["grounded solve = reciprocity"] = grounded_error < 1.0e-10
    summary += f"; grounded check {grounded_error:.2e}"
else:
    checks["gap discharged"] = abs(final_gap) < 0.5 * abs(V_CATHODE)
finish(checks, summary)
