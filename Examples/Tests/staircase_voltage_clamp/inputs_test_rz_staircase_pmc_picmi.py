#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Staircase voltage clamp with PMC (``neumann``) z faces.

--case coax:    z-uniform rod/sleeve/wall coax in vacuum. The unit potentials equal the
                exact discrete radial (log) solution, C equals its analytic value, and
                the clamp holds the target for many steps with B at roundoff.
--case rings:   rod spanning z plus end rings touching the PMC faces (or a PEC/PMC mix,
                or a ring connected to the grounded wall). Native unit potentials and C
                must match an independent dense solve of the same discrete operator.
--case exit:    neutral electron/ion pairs; the electrons leave through a PMC face next to
                a clamped rod. With the field-charge pairing the independent line integral
                stays at the target; with --pairing rho it drifts, which is asserted.
--case restart: --mode base|restart; checkpoint/restart continuity with exiting charge.

--compare-with DIR compares the saved state with an earlier run (domain decomposition).
"""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
from mpi4py import MPI

from pywarpx import boundary as boundary_inputs
from pywarpx import callbacks, picmi
from pywarpx._libwarpx import libwarpx
from pywarpx.particle_containers import ParticleContainerWrapper
from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

EPS0 = picmi.constants.ep0
QE = picmi.constants.q_e
ME = picmi.constants.m_e
COMM = MPI.COMM_WORLD
STATE = "pmc_state.json"

parser = argparse.ArgumentParser()
parser.add_argument(
    "--case", choices=["coax", "rings", "exit", "restart"], required=True
)
parser.add_argument("--max-grid-size", type=int, default=1024)
parser.add_argument("--z-faces", choices=["pmc", "pec_pmc"], default="pmc")
parser.add_argument("--ground-reference", action="store_true")
parser.add_argument("--pairing", choices=["auto", "rho"], default="auto")
parser.add_argument("--mode", choices=["base", "restart"], default="base")
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--compare-with", type=Path, default=None)
args = parser.parse_args()

# The implicit functions are positive inside metal.
# -- geometry per case --------------------------------------------------------------
if args.case == "coax":
    NR, NZ, R_WALL, LENGTH = 48, 8, 6.0e-2, 4.0e-2
    R_ROD, R_SLEEVE_IN, R_SLEEVE_OUT = 6.2e-3, 2.03e-2, 2.53e-2
    STEPS = 2000 if args.steps is None else args.steps
elif args.case == "rings":
    NR, NZ, R_WALL, LENGTH = 24, 32, 3.0e-2, 6.0e-2
    STEPS = 0
elif args.case == "exit":
    NR, NZ, R_WALL, LENGTH = 40, 80, 1.2e-2, 2.4e-2
    STEPS = 1200 if args.steps is None else args.steps
else:
    NR, NZ, R_WALL, LENGTH = 48, 16, 6.0e-2, 4.0e-2
    STEPS = 30
DR, DZ = R_WALL / NR, LENGTH / NZ
r2 = "(x*x+y*y)"

z_lo_field = "dirichlet" if args.z_faces == "pec_pmc" else "neumann"
grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    n_azimuthal_modes=1,
    lower_bound=[0.0, 0.0],
    upper_bound=[R_WALL, LENGTH],
    lower_boundary_conditions=["none", z_lo_field],
    upper_boundary_conditions=["dirichlet", "neumann"],
    lower_boundary_conditions_particles=["none", "absorbing"],
    upper_boundary_conditions_particles=["absorbing", "absorbing"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=args.max_grid_size,
)

electrodes = []
if args.case == "coax":
    rod = f"({R_ROD}*{R_ROD}-{r2})"
    sleeve = (
        f"(({r2}-{R_SLEEVE_IN}*{R_SLEEVE_IN})*({R_SLEEVE_OUT}*{R_SLEEVE_OUT}-{r2}))"
    )
    implicit = f"-({rod})*({sleeve})"
    electrodes = [
        {"name": "rod", "region": "(x<0.012)", "potential": -1.0e4},
        {"name": "sleeve", "region": "(x>0.012)", "potential": -2.5e3},
    ]
elif args.case == "rings":
    # Rod spanning z (or starting above a PEC lower face) and two rings per PMC face.
    z_ring = 4.0e-3
    rod_bottom = 0.25 * LENGTH if args.z_faces == "pec_pmc" else -1.0
    rod = f"min(3.2e-3*3.2e-3-{r2}, z-{rod_bottom})"
    ring_end = f"max(z-{LENGTH - z_ring}, {z_ring}-z)"
    if args.z_faces == "pec_pmc":
        ring_end = f"(z-{LENGTH - z_ring})"
    ring1 = f"min(min({r2}-6.0e-3*6.0e-3, 1.0e-2*1.0e-2-{r2}), {ring_end})"
    outer_r = R_WALL + DR if args.ground_reference else 2.7e-2
    ring2 = f"min(min({r2}-1.4e-2*1.4e-2, {outer_r}*{outer_r}-{r2}), {ring_end})"
    implicit = f"max(max({rod}, {ring1}), {ring2})"
    electrodes = [
        {"name": "rod", "region": "(x<0.0045)", "potential": -1000.0},
        {"name": "ring1", "region": "(x>0.0045)*(x<0.012)", "potential": -600.0},
    ]
    if not args.ground_reference:
        electrodes.append({"name": "ring2", "region": "(x>0.012)", "potential": -200.0})
elif args.case == "exit":
    R_ROD = 2.0e-3
    implicit = f"{R_ROD}*{R_ROD}-{r2}"
    electrodes = [{"name": "rod", "region": "(x<0.0035)", "potential": -10.0}]
else:
    R_ROD = 6.2e-3
    implicit = f"{R_ROD}*{R_ROD}-{r2}"
    electrodes = [{"name": "cathode", "region": "(x<0.03)", "potential": -1000.0}]

restart_dir = Path("../test_rz_staircase_pmc_restart_base_picmi")
sim_kwargs = {}
if args.case == "restart" and args.mode == "restart":
    sim_kwargs["warpx_amr_restart"] = str(restart_dir / "diags" / f"chk{STEPS:06d}")
sim = picmi.Simulation(
    solver=picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9),
    max_steps=max(STEPS, 1) * (2 if args.case == "restart" else 1),
    particle_shape="linear",
    warpx_embedded_boundary=picmi.EmbeddedBoundary(
        implicit_function=implicit, potential=0.0
    ),
    warpx_use_filter=False,
    warpx_current_deposition_algo="esirkepov",
    verbose=0,
    **sim_kwargs,
)
if args.case in ("exit", "restart"):
    sim.add_species(
        picmi.Species(name="electrons", particle_type="electron"), layout=None
    )
if args.case == "exit":
    sim.add_species(picmi.Species(name="ions", charge=QE, mass=1.0e6 * ME), layout=None)
if args.case == "restart" and args.mode == "base":
    sim.add_diagnostic(picmi.Checkpoint(period=STEPS, name="chk"))
sim.initialize_inputs()
if args.z_faces == "pec_pmc":
    # PICMI maps "dirichlet" to PEC; keep the field type explicit.
    boundary_inputs.field_lo = ["none", "pec"]
sim.initialize_warpx()

warpx = libwarpx.libwarpx_so.get_instance()
direction = libwarpx.libwarpx_so.Direction
register = warpx.multifab_register()
efield = [register.get("Efield_fp", dir=direction(k), level=0) for k in range(3)]
bfield = [register.get("Bfield_fp", dir=direction(k), level=0) for k in range(3)]


def gather(mf):
    """Valid region of a single-component level-0 field as one global array."""
    shape = [NR + 1, NZ + 1]
    ix = mf.box_array().ix_type()
    for d in range(2):
        if not ix.node_centered(d):
            shape[d] -= 1
    out = np.zeros(shape)
    for mfi, arr in zip(mf, mf.to_numpy(copy=True)):
        v, f = mfi.validbox(), mfi.fabbox()
        a = np.squeeze(
            arr[
                v.small_end[0] - f.small_end[0] : v.big_end[0] - f.small_end[0] + 1,
                v.small_end[1] - f.small_end[1] : v.big_end[1] - f.small_end[1] + 1,
            ]
        )
        out[v.small_end[0] : v.big_end[0] + 1, v.small_end[1] : v.big_end[1] + 1] += a
    # Shared nodes are owned by several boxes with identical values.
    count = np.zeros(shape)
    for mfi in mf:
        v = mfi.validbox()
        count[v.small_end[0] : v.big_end[0] + 1, v.small_end[1] : v.big_end[1] + 1] += 1
    out = COMM.allreduce(out, op=MPI.SUM)
    count = COMM.allreduce(count, op=MPI.SUM)
    return out / np.maximum(count, 1)


def valid_change(fields, references):
    change = []
    for field, reference in zip(fields, references):
        delta = field.copy()
        delta.saxpy(-1.0, reference, 0, 0, 1, 0)
        change.append(delta.norm0(0, 0, False, False))
    return max(change)


def gauss_volume(half_faces):
    rn = np.arange(NR + 1) * DR
    volume = 2.0 * np.pi * rn[:, None] * DR * DZ * np.ones((1, NZ + 1))
    volume[0] = np.pi * DR * DR * DZ * 0.25
    if half_faces:
        volume[:, [0, -1]] *= 0.5
    return volume


def reference_solution(weights_native):
    """Independent dense solve of the native discrete operator on the native masks."""
    ur = gather(warpx.eb_update_e_flag(0, 0)).astype(bool)
    uz = gather(warpx.eb_update_e_flag(0, 2)).astype(bool)
    metal = np.zeros((NR + 1, NZ + 1), bool)
    metal[:-1, :] |= ~ur
    metal[1:, :] |= ~ur
    metal[:, :-1] |= ~uz
    metal[:, 1:] |= ~uz
    wall = np.zeros_like(metal)
    wall[-1, :] = True
    if args.z_faces == "pec_pmc":
        wall[:, 0] = True
    fixed = metal | wall
    rn = np.arange(NR + 1) * DR
    index = np.arange(fixed.size).reshape(fixed.shape)
    matrix = np.zeros((fixed.size, fixed.size))
    for i in range(NR + 1):
        for j in range(NZ + 1):
            n = index[i, j]
            if fixed[i, j]:
                matrix[n, n] = 1.0
                continue
            if i == 0:
                matrix[n, index[1, j]] += 4.0 / DR**2
                matrix[n, n] -= 4.0 / DR**2
            else:
                rp, rm = rn[i] + 0.5 * DR, rn[i] - 0.5 * DR
                matrix[n, index[i + 1, j]] += rp / (rn[i] * DR**2)
                matrix[n, index[i - 1, j]] += rm / (rn[i] * DR**2)
                matrix[n, n] -= (rp + rm) / (rn[i] * DR**2)
            if 0 < j < NZ:
                matrix[n, index[i, j + 1]] += 1.0 / DZ**2
                matrix[n, index[i, j - 1]] += 1.0 / DZ**2
                matrix[n, n] -= 2.0 / DZ**2
            else:  # homogeneous Neumann (PMC) face: mirror node
                jn = 1 if j == 0 else NZ - 1
                matrix[n, index[i, jn]] += 2.0 / DZ**2
                matrix[n, n] -= 2.0 / DZ**2
    potentials = []
    for weight in weights_native:
        assert np.all(weight[~fixed] == 0.0), "native weight on a free node"
        rhs = np.where(fixed, weight, 0.0).ravel()
        potentials.append(np.linalg.solve(matrix, rhs).reshape(fixed.shape))
    volume = gauss_volume(True)
    columns = []
    for phi in potentials:
        er = -(phi[1:, :] - phi[:-1, :]) / DR
        ez = -(phi[:, 1:] - phi[:, :-1]) / DZ
        div = np.zeros_like(phi)
        flux = (rn[:-1] + 0.5 * DR)[:, None] * er
        div[1:-1] = (flux[1:] - flux[:-1]) / (rn[1:-1, None] * DR)
        div[0] = 4.0 * er[0] / DR
        ezp = np.concatenate([-ez[:, :1], ez, -ez[:, -1:]], axis=1)
        div += (ezp[:, 1:] - ezp[:, :-1]) / DZ
        div[-1] = 0.0  # wall nodes carry no electrode weight
        columns.append([EPS0 * np.sum(div * w * volume) for w in weights_native])
    return potentials, np.asarray(columns).T, fixed


def charge_state(corrector, pairing):
    return np.asarray(
        warpx.staircase_charge_state(
            corrector._psi_names, corrector._weight_names, grounded_pairing=pairing
        ),
        dtype=float,
    )


def reading(corrector, pairing):
    raw, live, grounded = charge_state(corrector, pairing)
    return np.linalg.solve(corrector._capacitance, raw - live - grounded)


corrector = StaircaseBiasCorrector(
    sim,
    correction_interval=1,
    electrodes=electrodes,
    grounded_wall_reference=args.ground_reference,
    grounded_pairing=args.pairing,
)
corrector.setup_after_init()
cap = corrector._capacitance
summary = {"case": args.case, "nprocs": COMM.size, "C": cap.tolist()}
failures = []


def check(name, ok, value=None):
    summary.setdefault("checks", {})[name] = {"ok": bool(ok), "value": value}
    if not ok:
        failures.append(name)


expected_pairing = "rho" if args.pairing == "rho" else "field"
check(
    "pairing",
    corrector.grounded_pairing == expected_pairing,
    corrector.grounded_pairing,
)
check(
    "C symmetric",
    np.max(np.abs(cap - cap.T)) <= 1.0e-12 * np.max(np.abs(cap)),
    float(np.max(np.abs(cap - cap.T)) / np.max(np.abs(cap))),
)
check("C positive definite", bool(np.all(np.linalg.eigvalsh(0.5 * (cap + cap.T)) > 0)))

psi = [gather(register.get(name, level=0)) for name in corrector._psi_names]
weights = [gather(register.get(name, level=0)) for name in corrector._weight_names]
summary["psi_checksum"] = [
    float(np.sum(p * np.arange(p.size).reshape(p.shape))) for p in psi
]

if args.case == "coax":
    # Exact discrete radial solution between the fixed radii of each conductor.
    rod_nodes = int(round(np.sum(weights[0]) / (NZ + 1)))
    sleeve_cols = np.flatnonzero(weights[1][:, NZ // 2])
    m, a, b = rod_nodes - 1, int(sleeve_cols.min()), int(sleeve_cols.max())
    inv = 1.0 / ((np.arange(NR) + 0.5) * DR) * DR  # dr / r_{i+1/2}
    s1, s2 = np.sum(inv[m:a]), np.sum(inv[b:NR])
    exact_rod = np.zeros(NR + 1)
    exact_rod[: m + 1] = 1.0
    exact_rod[m + 1 : a] = 1.0 - np.cumsum(inv[m : a - 1]) / s1
    exact_sleeve = np.zeros(NR + 1)
    exact_sleeve[m + 1 : a] = np.cumsum(inv[m : a - 1]) / s1
    exact_sleeve[a : b + 1] = 1.0
    exact_sleeve[b + 1 : NR] = 1.0 - np.cumsum(inv[b : NR - 1]) / s2
    error = max(
        float(np.max(np.abs(psi[0] - exact_rod[:, None]))),
        float(np.max(np.abs(psi[1] - exact_sleeve[:, None]))),
    )
    check("unit potentials = discrete log solution", error <= 1.0e-10, error)
    # Half face volumes make the summed axial length exactly LENGTH.
    g1, g2 = 2 * np.pi * EPS0 * LENGTH / s1, 2 * np.pi * EPS0 * LENGTH / s2
    exact_cap = np.array([[g1, -g1], [-g1, g1 + g2]])
    cap_error = float(np.max(np.abs(cap / exact_cap - 1.0)))
    check("C = analytic discrete coax", cap_error <= 1.0e-9, cap_error)

    # Refreshing the PMC guards for the observer leaves valid E unchanged.
    corrector.initialize_vacuum_bias()
    saved = [field.copy() for field in efield]
    v_field = reading(corrector, "field")
    v_rho = reading(corrector, "rho")
    check("guard refresh keeps valid E", valid_change(efield, saved) == 0.0)
    check(
        "initial reading",
        np.max(np.abs(v_field - corrector.v_target)) <= 1.0e-8,
        (v_field - corrector.v_target).tolist(),
    )
    # The rho pairing ignores the unit solves' residual divergence, so the two
    # readings agree to the solve accuracy, not to roundoff.
    split = float(np.max(np.abs(v_field - v_rho)) / np.max(np.abs(corrector.v_target)))
    check("field = rho pairing in vacuum", split <= 1.0e-9, split)
    callbacks.installafterstep(corrector.correct_after_step)
    sim.step(STEPS)
    callbacks.uninstallcallback("afterstep", corrector.correct_after_step)
    final = corrector.measure_voltage_state()["voltage"]
    e_scale = max(field.norm0(0, 0, False, False) for field in saved)
    e_change = valid_change(efield, saved) / e_scale
    b_ratio = (
        picmi.constants.c * max(f.norm0(0, 0, False, False) for f in bfield) / e_scale
    )
    check("target held", np.max(np.abs(final - corrector.v_target)) <= 1.0e-8)
    check("E unchanged", e_change <= 1.0e-12, e_change)
    check("B at roundoff", b_ratio <= 1.0e-12, b_ratio)
    saved = None

elif args.case == "rings":
    potentials, reference_cap, fixed = reference_solution(weights)
    psi_error = max(float(np.max(np.abs(p - q))) for p, q in zip(psi, potentials))
    cap_error = float(
        np.max(np.abs(cap - reference_cap)) / np.max(np.abs(reference_cap))
    )
    check("unit potentials = dense reference", psi_error <= 1.0e-8, psi_error)
    check("C = dense reference", cap_error <= 1.0e-8, cap_error)
    if args.z_faces == "pmc":
        face_nodes = sum(int(np.sum(w[:, [0, -1]])) for w in weights)
        check("electrodes on PMC faces", face_nodes > 0, face_nodes)
    if args.ground_reference:
        check("reference ring is fixed and unselected", np.any(fixed[-2, -1]))
    for k in range(len(psi)):
        units = [
            gather(register.get(corrector._unit_names[k], dir=direction(c), level=0))
            for c in (0, 2)
        ]
        er_ref = -(potentials[k][1:, :] - potentials[k][:-1, :]) / DR
        ez_ref = -(potentials[k][:, 1:] - potentials[k][:, :-1]) / DZ
        scale = max(np.max(np.abs(er_ref)), np.max(np.abs(ez_ref)))
        unit_error = (
            max(
                float(np.max(np.abs(units[0] - er_ref))),
                float(np.max(np.abs(units[1] - ez_ref))),
            )
            / scale
        )
        check(f"unit E {k} = dense reference", unit_error <= 1.0e-8, unit_error)
    corrector.initialize_vacuum_bias()
    v0 = corrector.measure_voltage_state()["voltage"]
    check(
        "initial reading",
        np.max(np.abs(v0 - corrector.v_target)) <= 1.0e-8,
        v0.tolist(),
    )
    summary["reading"] = v0.tolist()

elif args.case == "exit":
    n = 24
    r = np.linspace(3.0e-3, 9.0e-3, n)
    z0 = np.full(n, 1.5 * DZ)
    weight = np.full(n, 1.0e-12 / QE / n)
    zeros = np.zeros(n)
    sim.particles.get("electrons").add_particles(
        x=r,
        y=zeros,
        z=z0,
        ux=zeros,
        uy=zeros,
        uz=np.full(n, -2.0e7),
        w=weight,
        unique_particles=False,
    )
    sim.particles.get("ions").add_particles(
        x=r,
        y=zeros,
        z=z0,
        ux=zeros,
        uy=zeros,
        uz=zeros,
        w=weight,
        unique_particles=False,
    )
    corrector.initialize_vacuum_bias()
    electrons = ParticleContainerWrapper("electrons")
    first_live = int(np.argmax(gather(warpx.eb_update_e_flag(0, 0))[:, NZ // 2]))
    trace = {
        "step": [],
        "count": [],
        "observer": [],
        "field": [],
        "rho": [],
        "line": [],
    }
    grounded_checks = []

    def record():
        step = int(warpx.getistep(0))
        er = gather(efield[0])
        trace["step"].append(step)
        trace["count"].append(int(electrons.get_particle_count()))
        trace["observer"].append(float(corrector.measure_voltage_state()["voltage"][0]))
        trace["field"].append(float(reading(corrector, "field")[0]))
        trace["rho"].append(float(reading(corrector, "rho")[0]))
        # Independent instrument: midplane line integral from the rod to the wall.
        trace["line"].append(float(np.sum(er[first_live:, NZ // 2]) * DR))
        has_grounded_solve = hasattr(warpx, "solve_staircase_grounded")
        if has_grounded_solve and trace["count"][-1] == n and step in (5, 30):
            # Step 30 has the electrons inside the face cell. The induced charge of
            # the neutral pairs nearly cancels, so normalize by its gross scale.
            rho = gather(warpx.deposit_scratch_rho(0))
            gross = max(np.sum(np.abs(p * rho) * gauss_volume(False)) for p in psi)
            grounded_checks.append((corrector.compare_grounded_charge(), gross))

    callbacks.installafterstep(corrector.correct_after_step)
    callbacks.installafterstep(record)
    sim.step(STEPS)
    callbacks.uninstallcallback("afterstep", corrector.correct_after_step)
    callbacks.uninstallcallback("afterstep", record)
    t = {k: np.asarray(v) for k, v in trace.items()}
    target = corrector.v_target[0]
    before = t["count"] == n
    after = (t["count"] == 0) & (t["step"] >= STEPS // 2)
    split = np.max(np.abs(t["field"][before] - t["rho"][before]))
    check("pairings agree before exit", split <= 1.0e-9 * abs(target), float(split))
    for entry, gross in grounded_checks:
        diff = float(np.max(np.abs(entry["difference"])) / gross)
        check("grounded solve = reciprocity before exit", diff <= 1.0e-9, diff)
    if hasattr(warpx, "solve_staircase_grounded"):
        check("grounded solve sampled", len(grounded_checks) == 2, len(grounded_checks))
    check("electrons left", t["count"][-1] == 0, int(t["count"][-1]))
    observer_error = float(np.max(np.abs(t["observer"][1:] - target)))
    check("observer at target", observer_error <= 1.0e-9 * abs(target), observer_error)
    line_offset = float(np.mean(t["line"][after]) - target)
    pairing_gap = float(np.mean(t["field"][after] - t["rho"][after]))
    summary.update(line_offset=line_offset, pairing_gap=pairing_gap)
    if args.pairing == "rho":
        # Regression marker: the rho pairing misses the face sheet, so the
        # independent voltage drifts by the sheet's induced charge.
        check("rho pairing drifts", abs(line_offset) >= 0.1, line_offset)
    else:
        check("line integral at target", abs(line_offset) <= 0.01, line_offset)
    check("face charge seen by field pairing", abs(pairing_gap) >= 0.1, pairing_gap)
    summary["trace"] = {k: v.tolist() for k, v in t.items()}

else:  # restart
    if args.mode == "base":
        corrector.initialize_vacuum_bias()
    else:
        check("restart step", warpx.getistep(0) == STEPS, int(warpx.getistep(0)))
        corrector.resume_from_checkpoint()
    electrons = sim.particles.get("electrons")
    z_birth = (np.arange(NZ) + 0.5) * DZ
    r_birth = R_ROD + 0.1 * (5 * DR - R_ROD)
    trace = {"step": [], "observer": []}

    def inject():
        electrons.add_particles(
            x=np.full(NZ, r_birth),
            y=np.zeros(NZ),
            z=z_birth,
            ux=np.full(NZ, 1.0e6),
            uy=np.zeros(NZ),
            uz=np.where(z_birth < 0.5 * LENGTH, -3.0e7, 3.0e7),
            w=np.full(NZ, 2.0e6),
            unique_particles=False,
        )

    def record():
        step = int(warpx.getistep(0))
        if step <= STEPS:
            return
        state = corrector.measure_voltage_state()
        trace["step"].append(step)
        trace["observer"].append(float(state["voltage"][0]))
        if step == 2 * STEPS:
            trace["er_final"] = gather(efield[0])
            # Charge left on the PMC faces separates the two pairings.
            trace["face_gap"] = float(
                np.max(np.abs(reading(corrector, "field") - reading(corrector, "rho")))
            )

    callbacks.installbeforestep(inject)
    callbacks.installafterstep(corrector.correct_after_step)
    callbacks.installafterstep(record)
    sim.step(2 * STEPS - warpx.getistep(0))
    for kind, fn in (
        ("beforestep", inject),
        ("afterstep", corrector.correct_after_step),
        ("afterstep", record),
    ):
        callbacks.uninstallcallback(kind, fn)
    result = {k: np.asarray(v) for k, v in trace.items()}
    if args.mode == "base":
        if COMM.rank == 0:
            np.savez("restart_trace.npz", **result)
        COMM.Barrier()
        check(
            "face charge present",
            float(result["face_gap"]) >= 0.1,
            float(result["face_gap"]),
        )
    else:
        base = np.load(restart_dir / "restart_trace.npz")
        scale = np.max(np.abs(base["er_final"]))
        check("same steps", np.array_equal(base["step"], result["step"]))
        check(
            "final E_r",
            np.max(np.abs(result["er_final"] - base["er_final"])) <= 1.0e-12 * scale,
        )
        check(
            "observer trace",
            np.max(np.abs(result["observer"] - base["observer"])) <= 1.0e-9,
        )
        check("target held", np.max(np.abs(result["observer"] + 1000.0)) <= 1.0e-9)

if args.compare_with is not None:
    other = json.loads((args.compare_with / STATE).read_text())
    cap_other = np.asarray(other["C"])
    rel = float(np.max(np.abs(cap - cap_other)) / np.max(np.abs(cap)))
    check("C matches other decomposition", rel <= 1.0e-12, rel)
    psi_rel = float(
        np.max(
            np.abs(np.subtract(summary["psi_checksum"], other["psi_checksum"]))
            / np.abs(other["psi_checksum"])
        )
    )
    check("psi matches other decomposition", psi_rel <= 1.0e-12, psi_rel)
    if "trace" in other:
        for key in ("observer", "field", "rho", "line"):
            a, b = np.asarray(summary["trace"][key]), np.asarray(other["trace"][key])
            diff = float(np.max(np.abs(a - b)))
            check(f"{key} trace matches other decomposition", diff <= 1.0e-9, diff)
    if "reading" in other:
        diff = float(np.max(np.abs(np.subtract(summary["reading"], other["reading"]))))
        check("reading matches other decomposition", diff <= 1.0e-9, diff)

if COMM.rank == 0:
    Path(STATE).write_text(json.dumps(summary, indent=1))
    for name, entry in summary["checks"].items():
        print(
            f"pmc {args.case}: {'PASS' if entry['ok'] else 'FAIL'} {name}: {entry['value']}"
        )

corrector = electrons = register = efield = bfield = None
gc.collect()
sim.finalize()
assert not failures, f"PMC staircase checks failed: {failures}"
