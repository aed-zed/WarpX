#!/usr/bin/env python3
#
# --- Test of the external scalar potential of the hybrid-PIC solver in RZ:
# --- a biased embedded-boundary rod on the axis inside a grounded wall
# --- (coaxial geometry, periodic in z) with a cold ion annulus at rest.
# --- Since the ions are cold and at rest, Te = 0 and the initial B is
# --- uniform, the Ohm's-law field vanishes, so after one step the total E
# --- is the superposition of the external fields only:
# ---   vacuum edges:  E = E_Phi + E_A
# ---   dense edges:   E = (1 - shield) * E_Phi
# --- where E_Phi is the vacuum (coaxial) field of the rod and
# --- E_A = -dA/dt the inductive field of an optional ramped A_external.

import argparse
import json

import numpy as np

from pywarpx import callbacks, fields, picmi

parser = argparse.ArgumentParser()
parser.add_argument(
    "--key",
    default="boundaries",
    help="name of the Phi_external entry (any name selects use_boundary_potentials)",
)
parser.add_argument(
    "--shield", type=int, choices=[0, 1], default=1, help="shielding toggle"
)
parser.add_argument(
    "--ramp-A",
    action="store_true",
    help="add a uniform A_z with A_time_external_function = t/T",
)
parser.add_argument(
    "--ramp-phi-time",
    action="store_true",
    help="Phi_time_external_function = 1 + t/T",
)
parser.add_argument(
    "--ramp-eb-potential",
    action="store_true",
    help="warpx.eb_potential = V0*(1 + t/T)",
)
args = parser.parse_args()

constants = picmi.constants

# Physical and numerical parameters
V0 = 100.0  # rod potential (V)
a = 6.7e-3  # rod radius (m), between grid nodes
b = 40.0e-3  # grounded wall radius (m)
Lz = 40.0e-3
nr, nz = 32, 16
n0 = 1.0e20
r_p_in, r_p_out = 15.0e-3, 30.0e-3  # ion annulus
B0 = 0.1
dt = 1.0e-9
T = 10.0 * dt  # time scale of the ramps
A0 = 1.0e-6  # uniform A_z (V s / m), E_A,z = -A0/T

grid = picmi.CylindricalGrid(
    number_of_cells=[nr, nz],
    lower_bound=[0.0, -Lz / 2],
    upper_bound=[b, Lz / 2],
    lower_boundary_conditions=["none", "periodic"],
    upper_boundary_conditions=["dirichlet", "periodic"],
    lower_boundary_conditions_particles=["none", "periodic"],
    upper_boundary_conditions_particles=["absorbing", "periodic"],
    warpx_max_grid_size=32,
)

phi_entry = {"use_boundary_potentials": True}
if args.ramp_phi_time:
    phi_entry["Phi_time_external_function"] = f"1.0 + t/{T!r}"

A_ext = None
if args.ramp_A:
    A_ext = {
        "rampA": {
            "Ax_external_function": "0.0",
            "Ay_external_function": "0.0",
            "Az_external_function": f"{A0!r}",
            "A_time_external_function": f"t/{T!r}",
        }
    }

solver = picmi.HybridPICSolver(
    grid=grid,
    gamma=1.0,
    Te=0.0,
    n0=n0,
    n_floor=0.05 * n0,
    plasma_resistivity=1.0e-4,
    plasma_hyper_resistivity=0.0,
    substeps=10,
    holmstrom_vacuum_region=True,
    Phi_external={args.key: phi_entry},
    A_external=A_ext,
    do_external_diva_cleaning=False if args.ramp_A else None,
    shield_external_E_field_in_dense_plasma=bool(args.shield),
)

potential = f"{V0!r}*(1.0 + t/{T!r})" if args.ramp_eb_potential else f"{V0!r}"
embedded_boundary = picmi.EmbeddedBoundary(
    implicit_function=f"({a!r}**2 - x**2 - y**2)", potential=potential
)

ions = picmi.Species(
    name="ions",
    charge="q_e",
    mass=constants.m_p,
    initial_distribution=picmi.AnalyticDistribution(
        density_expression=(
            f"{n0!r}*(sqrt(x**2+y**2) >= {r_p_in!r})*(sqrt(x**2+y**2) <= {r_p_out!r})"
        ),
    ),
)

field_diag = picmi.FieldDiagnostic(
    name="diag1",
    grid=grid,
    period=1,
    data_list=["E", "B", "rho"],
    write_dir="diags",
    warpx_format="plotfile",
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=dt,
    max_steps=1,
    particle_shape=1,
    warpx_current_deposition_algo="direct",
    verbose=0,
)
sim.embedded_boundary = embedded_boundary
sim.add_applied_field(
    picmi.AnalyticInitialField(Bx_expression=0.0, By_expression=0.0, Bz_expression=B0)
)
sim.add_species(
    ions, layout=picmi.GriddedLayout(n_macroparticle_per_cell=[2, 4, 2], grid=grid)
)
sim.add_diagnostic(field_diag)


def _get(name, idir=None):
    kw = {} if idir is None else {"idir": idir}
    arr = np.asarray(fields.MultiFabWrapper(mf_name=name, level=0, **kw)[...])
    return np.array(arr[..., 0] if arr.ndim == 3 else arr)  # mode m=0


def _record():
    # Edge-centred fields after the step, for the analysis script
    np.savez(
        "fields_step1.npz",
        Er=_get("Efield_fp", 0),
        Ez=_get("Efield_fp", 2),
        rho=_get("rho_fp"),
        Er_phi=_get("hybrid_E_fp_external_phi", 0),
    )


callbacks.installafterstep(_record)

sim.step(1)

with open("params.json", "w") as f:
    json.dump(
        dict(
            V0=V0,
            a=a,
            b=b,
            nr=nr,
            nz=nz,
            n0=n0,
            dt=dt,
            T=T,
            A0=A0,
            q_e=constants.q_e,
            rho_floor=0.05 * n0 * constants.q_e,
            args=vars(args),
        ),
        f,
        indent=1,
    )
