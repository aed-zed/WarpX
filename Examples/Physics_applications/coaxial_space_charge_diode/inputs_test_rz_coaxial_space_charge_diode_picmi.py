#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Electromagnetic coaxial space-charge diode at the Langmuir-Blodgett limit.

RZ, Yee, Esirkepov, linear shape, periodic z. Cathode: EB rod (clamped to
-1 kV). Anode: EB sleeve (clamped to 0 V), inside a grounded PEC outer wall.
Electrons are born every step inside the cathode's fixed (frozen-edge) node
shell with a small radial speed, at the Langmuir-Blodgett current of the
*represented* gap (the radii WarpX's staircase actually exposes to the field
solve, read back from the frozen-edge mask, not the geometric rod/sleeve
radii), so the emitted charge leaves the conductor without breaking the
discrete Gauss law.

Without the clamp (``--arm off``) the electromagnetic solver has nothing that
maintains the electrode potentials: the gap discharges within about one
transit time as the emitted space charge is not replenished. With the clamp
(``--arm on``) the gap voltage is held at 1 kV and a steady Langmuir-Blodgett
profile is reached within a few transits.

This script writes ``diode_result.npz`` with the time-averaged potential
profile over the last transit, the voltage history, and a collected-current
estimate; see ``analysis_coaxial_space_charge_diode.py`` for the comparison
against the Langmuir-Blodgett law and a matched cold-beam oracle.
"""

import argparse
import gc

import numpy as np
from mpi4py import MPI

from pywarpx import callbacks, picmi
from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

parser = argparse.ArgumentParser()
parser.add_argument("--nr", type=int, default=48, help="radial cells")
parser.add_argument(
    "--transits", type=float, default=6.0, help="run length in cold-beam transit times"
)
parser.add_argument(
    "--arm", choices=["on", "off"], default="on", help="clamp the electrodes?"
)
args = parser.parse_args()

NZ = 8
R_ROD, R_SLEEVE_IN, R_SLEEVE_OUT = 6.2e-3, 5.03e-2, 5.53e-2
R_WALL, LENGTH = 6.0e-2, 4.0e-2
V_CATHODE = -1000.0
V_ANODE = 0.0
V_BIRTH = 1.0e6  # m/s, about 2.8 eV
EPS0 = picmi.constants.ep0
Q_E = picmi.constants.q_e
M_E = picmi.constants.m_e

grid = picmi.CylindricalGrid(
    number_of_cells=[args.nr, NZ],
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
from pywarpx.particle_containers import ParticleBoundaryBufferWrapper  # noqa: E402

warpx = libwarpx.libwarpx_so.get_instance()
direction = libwarpx.libwarpx_so.Direction
register = warpx.multifab_register()
particles = sim.particles.get("electrons")
boundary_buffer = ParticleBoundaryBufferWrapper()
dr, dz = R_WALL / args.nr, LENGTH / NZ

corrector = StaircaseBiasCorrector(
    sim,
    correction_interval=1,
    electrodes=[
        {"name": "cathode", "region": "(x<0.03)", "potential": V_CATHODE},
        {"name": "anode", "region": "(x>0.03)", "potential": V_ANODE},
    ],
)
corrector.setup_after_init()
corrector.initialize_vacuum_bias()


def owned(mf):
    """Copy of the single valid box (AMReX-backed views must not outlive finalize).

    This script runs on one rank with the grid in a single box
    (``warpx_max_grid_size=1024`` for ``--nr`` up to a few hundred cells), so
    there is exactly one (mfi, arr) pair to iterate over.
    """
    for mfi, arr in zip(mf, mf.to_numpy(copy=True)):
        v, f = mfi.validbox(), mfi.fabbox()
        a = arr[
            v.small_end[0] - f.small_end[0] : v.big_end[0] - f.small_end[0] + 1,
            v.small_end[1] - f.small_end[1] : v.big_end[1] - f.small_end[1] + 1,
        ]
        return np.array(np.squeeze(a), copy=True)


# Represented conductor radii from WarpX's own frozen-edge mask: the cathode
# and anode surfaces the field solve actually sees are not the geometric rod
# and sleeve radii but the nearest staircased node shells.
frozen = np.where(owned(warpx.eb_update_e_flag(0, 0))[:, 0] == 0)[0]
k_cathode = int(frozen[frozen * dr < 0.03].max()) + 1
k_anode = int(frozen[frozen * dr > 0.03].min())
a_eff, b_eff = k_cathode * dr, k_anode * dr
assert a_eff > R_ROD, (
    "the represented cathode surface must lie outside the geometric rod"
)
r_birth = R_ROD + 0.1 * (a_eff - R_ROD)

# Langmuir-Blodgett current for the represented gap (beta^2 from its ODE:
# 3 b b'' + b'^2 + 4 b b' + b^2 = 1, with b ~ gamma - 0.4 gamma^2 for gamma -> 0).
gamma = np.linspace(1.0e-6, np.log(b_eff / a_eff), 20001)
beta = gamma - 0.4 * gamma**2
dbeta = 1.0 - 0.8 * gamma
for i in range(1, gamma.size):  # explicit midpoint integrator
    h = gamma[i] - gamma[i - 1]
    b, db = beta[i - 1], dbeta[i - 1]
    acc = (1.0 - b * b - db * db - 4.0 * b * db) / (3.0 * b)
    bm, dbm = b + 0.5 * h * db, db + 0.5 * h * acc
    accm = (1.0 - bm * bm - dbm * dbm - 4.0 * bm * dbm) / (3.0 * bm)
    beta[i], dbeta[i] = b + h * dbm, db + h * accm
beta2_b = beta[-1] ** 2
I_LB = (
    8.0
    * np.pi
    * EPS0
    / 9.0
    * np.sqrt(2.0 * Q_E / M_E)
    * abs(V_CATHODE) ** 1.5
    * LENGTH
    / (b_eff * beta2_b)
)
dt = float(warpx.getdt(0))
v_final = np.sqrt(2.0 * Q_E * abs(V_CATHODE) / M_E)
t_transit = 3.0 * (b_eff - a_eff) / v_final
n_steps = int(args.transits * t_transit / dt)
z_birth = (np.arange(NZ) + 0.5) * dz
weight = I_LB * dt / (Q_E * NZ)
r_nodes = np.arange(args.nr + 1) * dr

injected = 0.0
t_hist, v_obs_hist, v_line_hist, q_eb_hist = [], [], [], []
er_accum = np.zeros(args.nr)
n_accum = 0


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


def gap_voltage_line_integral(er_z):
    # E_r = -dphi/dr: summing E_r dr from cathode to anode gives
    # phi(cathode) - phi(anode), which is V_CATHODE when the clamp holds.
    return float(np.sum(er_z[k_cathode:k_anode]) * dr)


def record():
    global er_accum, n_accum
    er = owned(register.get("Efield_fp", dir=direction(0), level=0))
    er_z = er[:, :NZ].mean(axis=1)
    line = gap_voltage_line_integral(er_z)
    state = corrector.measure_voltage_state()
    v = np.asarray(state["voltage"], dtype=float)
    observer_gap = float(v[0] - v[1])
    step = warpx.getistep(0)
    t_hist.append(float(warpx.gett_new(0)))
    v_obs_hist.append(observer_gap)
    v_line_hist.append(line)
    n_eb = int(
        boundary_buffer.get_particle_boundary_buffer_size(
            "electrons", "eb", local=False
        )
    )
    # The EB buffer also counts any electron that happens to return to the
    # cathode shell; in steady Langmuir-Blodgett flow that rate is zero, so
    # this is a valid estimate of the current collected at the anode.
    q_eb_hist.append(n_eb * weight * Q_E)
    if step > n_steps - max(1, int(t_transit / dt)):
        er_accum += er_z
        n_accum += 1


callbacks.installbeforestep(inject)
if args.arm == "on":
    callbacks.installafterstep(corrector.correct_after_step)
callbacks.installafterstep(record)
sim.step(n_steps)

t_arr = np.asarray(t_hist)
v_obs_arr = np.asarray(v_obs_hist)
v_line_arr = np.asarray(v_line_hist)
q_eb_arr = np.asarray(q_eb_hist)

er_tail = er_accum / max(n_accum, 1)
# Potential above the cathode on the represented nodes between cathode and
# anode: psi(r_j) = phi(r_j) - phi(cathode) = -sum_{cathode<=i<j} E_r(i) dr.
r_profile = np.arange(k_cathode, k_anode + 1) * dr
phi_above = np.concatenate(([0.0], -np.cumsum(er_tail[k_cathode:k_anode]) * dr))

# Collected-current estimate over the last transit, from the slope of the
# cumulative EB-boundary charge (monotonic: the buffer is never cleared).
last = t_arr > t_arr[-1] - t_transit
if np.count_nonzero(last) >= 2:
    i_collected = float(np.polyfit(t_arr[last], q_eb_arr[last], 1)[0])
else:
    i_collected = float("nan")

if MPI.COMM_WORLD.rank == 0:
    np.savez(
        "diode_result.npz",
        arm=args.arm,
        nr=args.nr,
        nz=NZ,
        k_cathode=k_cathode,
        k_anode=k_anode,
        dr=dr,
        dt=dt,
        n_steps=n_steps,
        t_transit=t_transit,
        V0=V_CATHODE,
        r_nodes=r_nodes,
        r_profile=r_profile,
        phi_above=phi_above,
        a_eff=a_eff,
        b_eff=b_eff,
        L=LENGTH,
        v_birth=V_BIRTH,
        I_inj=I_LB,
        injected_charge=injected,
        t=t_arr,
        V_obs=v_obs_arr,
        V_line=v_line_arr,
        i_collected=i_collected,
    )

print(
    f"arm={args.arm} nr={args.nr} steps={n_steps} a_eff={a_eff:.4e} b_eff={b_eff:.4e} "
    f"I_LB={I_LB:.4e} A final V_line={v_line_arr[-1]:.3f} V i_collected={i_collected:.4e} A"
)

# Number of macroparticles still in flight, collective across ranks
# (only_local=False); some must have left the gap over the run.
n_live = particles.number_of_particles(False)

# Ordered teardown: Python-owned AMReX objects must die before AMReX
# finalizes, otherwise the interpreter's final garbage collection touches
# freed arenas.
callbacks.uninstallcallback("beforestep", inject)
if args.arm == "on":
    callbacks.uninstallcallback("afterstep", corrector.correct_after_step)
callbacks.uninstallcallback("afterstep", record)
corrector = particles = boundary_buffer = register = None
gc.collect()
sim.finalize()

# Basic sanity only; the physics checks live in the analysis script.
assert np.all(np.isfinite(v_line_arr)), "gap voltage history must stay finite"
assert np.all(np.isfinite(phi_above)), "potential profile must be finite"
assert t_arr.size == n_steps, "one history record expected per step"
assert n_live < round(injected / (Q_E * weight)), (
    "electrons must reach the anode within the run"
)
