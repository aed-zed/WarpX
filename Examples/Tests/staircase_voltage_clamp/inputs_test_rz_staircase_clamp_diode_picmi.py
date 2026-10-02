#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Electromagnetic coaxial space-charge diode with and without the staircase clamp.

RZ, Yee, Esirkepov, periodic z. Cathode: EB rod (clamped to -1 kV); anode: EB
sleeve (clamped to 0 V) inside a grounded PEC wall. Electrons are born every step
inside the cathode's fixed (frozen-edge) node shell with a small radial speed, at
the Langmuir-Blodgett current of the represented gap, so the emitted charge leaves
the conductor without breaking the discrete Gauss law.

Without the clamp (--arm off) the EM solver has nothing that maintains the
electrode potentials: the gap discharges within about one transit. The clamp's
observer still runs passively and must equal the independent line integral of
E_r at every step (Shockley-Ramo on the staircase). With the clamp (--arm on) the
gap voltage stays at 1 kV, the free-node Gauss residual stays at roundoff and B
stays zero (radial current in a z-uniform periodic gap).
"""

import argparse
import gc

import numpy as np

from pywarpx import callbacks, picmi
from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

parser = argparse.ArgumentParser()
parser.add_argument("--arm", choices=["on", "off"], required=True)
args = parser.parse_args()

NR, NZ = 48, 8
R_ROD, R_SLEEVE_IN, R_SLEEVE_OUT = 6.2e-3, 5.03e-2, 5.53e-2
R_WALL, LENGTH = 6.0e-2, 4.0e-2
V_CATHODE = -1000.0
V_BIRTH = 1.0e6  # m/s, about 2.8 eV
TRANSITS = 1.2
EPS0 = picmi.constants.ep0
Q_E = picmi.constants.q_e
M_E = picmi.constants.m_e

grid = picmi.CylindricalGrid(
    number_of_cells=[NR, NZ],
    n_azimuthal_modes=1,
    lower_bound=[0.0, 0.0],
    upper_bound=[R_WALL, LENGTH],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    warpx_blocking_factor=8,
    warpx_max_grid_size=1024,
)
solver = picmi.ElectromagneticSolver(grid=grid, method="Yee", cfl=0.9)
r2 = "(x*x+y*y)"
rod = f"({R_ROD}*{R_ROD}-{r2})"
sleeve = f"(({r2}-{R_SLEEVE_IN}*{R_SLEEVE_IN})*({R_SLEEVE_OUT}*{R_SLEEVE_OUT}-{r2}))"
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=f"-({rod})*({sleeve})", potential=0.0
)
sim = picmi.Simulation(
    solver=solver,
    max_steps=10**8,
    particle_shape="linear",
    warpx_embedded_boundary=embedded_boundary,
    warpx_use_filter=False,
    warpx_current_deposition_algo="esirkepov",
    verbose=0,
)
electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=None,
    warpx_save_particles_at_eb=True,
)
sim.add_species(electrons, layout=None)
sim.initialize_inputs()
sim.initialize_warpx()

from pywarpx._libwarpx import libwarpx  # noqa: E402

warpx = libwarpx.libwarpx_so.get_instance()
direction = libwarpx.libwarpx_so.Direction
register = warpx.multifab_register()
particles = sim.particles.get("electrons")
dr, dz = R_WALL / NR, LENGTH / NZ

corrector = StaircaseBiasCorrector(
    sim,
    correction_interval=1,
    electrodes=[
        {"name": "cathode", "region": "(x<0.03)", "potential": V_CATHODE},
        {"name": "anode", "region": "(x>0.03)", "potential": 0.0},
    ],
)
corrector.setup_after_init()
corrector.initialize_vacuum_bias()


def owned(mf):
    """Copy of the single valid box (AMReX-backed views must not outlive finalize)."""
    for mfi, arr in zip(mf, mf.to_numpy(copy=True)):
        v, f = mfi.validbox(), mfi.fabbox()
        a = arr[
            v.small_end[0] - f.small_end[0] : v.big_end[0] - f.small_end[0] + 1,
            v.small_end[1] - f.small_end[1] : v.big_end[1] - f.small_end[1] + 1,
        ]
        return np.array(np.squeeze(a), copy=True)


# Represented conductor radii from WarpX's own frozen-edge mask
frozen = np.where(owned(warpx.eb_update_e_flag(0, 0))[:, 0] == 0)[0]
k_cathode = int(frozen[frozen * dr < 0.03].max()) + 1
k_anode = int(frozen[frozen * dr > 0.03].min())
a_eff, b_eff = k_cathode * dr, k_anode * dr
assert a_eff > R_ROD, (
    "the represented cathode surface must lie outside the geometric rod"
)
r_birth = R_ROD + 0.1 * (a_eff - R_ROD)

# Langmuir-Blodgett current for the represented gap (beta^2 from its ODE)
gamma = np.linspace(1.0e-6, np.log(b_eff / a_eff), 20001)
beta = gamma - 0.4 * gamma**2
dbeta = 1.0 - 0.8 * gamma
for i in range(1, gamma.size):  # explicit midpoint on 3 b b'' + b'^2 + 4 b b' + b^2 = 1
    h = gamma[i] - gamma[i - 1]
    b, db = beta[i - 1], dbeta[i - 1]
    acc = (1.0 - b * b - db * db - 4.0 * b * db) / (3.0 * b)
    bm, dbm = b + 0.5 * h * db, db + 0.5 * h * acc
    accm = (1.0 - bm * bm - dbm * dbm - 4.0 * bm * dbm) / (3.0 * bm)
    beta[i], dbeta[i] = b + h * dbm, db + h * accm
current = (
    8.0
    * np.pi
    * EPS0
    / 9.0
    * np.sqrt(2.0 * Q_E / M_E)
    * abs(V_CATHODE) ** 1.5
    * LENGTH
    / (b_eff * beta[-1] ** 2)
)
dt = warpx.getdt(0)
transit = 3.0 * (b_eff - a_eff) / np.sqrt(2.0 * Q_E * abs(V_CATHODE) / M_E)
n_steps = int(TRANSITS * transit / dt)
z_birth = (np.arange(NZ) + 0.5) * dz
weight = current * dt / (Q_E * NZ)
r_node = np.arange(NR + 1) * dr
ring_volume = 2.0 * np.pi * r_node * dr * dz
injected = 0.0
worst = {"line_vs_observer": 0.0, "observer_vs_target": 0.0, "gauss": 0.0, "b": 0.0}
final = {}


def inject():
    global injected
    particles.add_particles(
        x=np.full(NZ, r_birth),
        y=np.zeros(NZ),
        z=z_birth,
        ux=np.full(NZ, V_BIRTH),
        uy=np.zeros(NZ),
        uz=np.zeros(NZ),
        w=np.full(NZ, weight),
        unique_particles=False,
    )
    injected += Q_E * weight * NZ


def gap_voltage_line_integral():
    er = owned(register.get("Efield_fp", dir=direction(0), level=0))[:, :NZ].mean(
        axis=1
    )
    # E_r = -dphi/dr, so the sum of E_r dr from cathode to anode is phi(cathode) - phi(anode)
    return float(np.sum(er[k_cathode:k_anode]) * dr)


def record():
    state = corrector.measure_voltage_state()
    v = np.asarray(state["voltage"], dtype=float)
    observer_gap = float(v[0] - v[1])
    line = gap_voltage_line_integral()
    worst["line_vs_observer"] = max(worst["line_vs_observer"], abs(line - observer_gap))
    if args.arm == "on":
        worst["observer_vs_target"] = max(
            worst["observer_vs_target"], abs(observer_gap - V_CATHODE)
        )
    step = warpx.getistep(0)
    if step % 100 == 0 or step == n_steps:
        div_e = owned(warpx.compute_div_e(0, "Efield_fp"))[:, :NZ]
        rho = owned(warpx.deposit_scratch_rho(0))[:, :NZ]
        resid = ((EPS0 * div_e - rho) * ring_volume[:, None]).sum(axis=1)
        worst["gauss"] = max(
            worst["gauss"], float(np.max(np.abs(resid[k_cathode + 1 : k_anode])))
        )
        bt = owned(register.get("Bfield_fp", dir=direction(1), level=0))
        worst["b"] = max(worst["b"], float(np.max(np.abs(bt))))
    final["line"] = line


callbacks.installbeforestep(inject)
if args.arm == "on":
    callbacks.installafterstep(corrector.correct_after_step)
callbacks.installafterstep(record)
sim.step(n_steps)

collected = particles.number_of_particles(False) < round(
    injected / (Q_E * weight)
)  # some left the gap
print(
    f"arm={args.arm} steps={n_steps} final gap (line integral) = {final['line']:.4f} V; "
    f"max|line - observer| = {worst['line_vs_observer']:.2e} V; "
    f"max|observer - target| = {worst['observer_vs_target']:.2e} V; "
    f"max free-node Gauss residual / injected = {worst['gauss'] / injected:.2e}; "
    f"max|B| = {worst['b']:.2e} T"
)

callbacks.uninstallcallback("beforestep", inject)
if args.arm == "on":
    callbacks.uninstallcallback("afterstep", corrector.correct_after_step)
callbacks.uninstallcallback("afterstep", record)
corrector = particles = register = None
gc.collect()
sim.finalize()

assert collected, "electrons must reach the anode within the run"
assert worst["line_vs_observer"] < 1.0e-6, (
    "observer and independent line integral disagree"
)
# roundoff accumulates over ~2500 steps; real violations are 1e-3 or larger
assert worst["gauss"] / injected < 1.0e-10, "free-node Gauss law violated"
assert worst["b"] < 1.0e-12, "a z-uniform radial current must not produce B"
if args.arm == "on":
    assert worst["observer_vs_target"] < 1.0e-9, "the clamp did not hold its target"
    assert abs(final["line"] - V_CATHODE) < 1.0e-6, "the gap voltage drifted"
else:
    assert abs(final["line"]) < 0.5 * abs(V_CATHODE), (
        "the unclamped gap should discharge"
    )
