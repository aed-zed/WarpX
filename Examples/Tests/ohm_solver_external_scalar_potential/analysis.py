#!/usr/bin/env python3
#
# Analysis of the hybrid-PIC external scalar potential test (RZ coaxial rod).
# Since the Ohm's-law field vanishes by construction (cold ions at rest,
# Te = 0, uniform B), the total E after one step must be:
#   vacuum edges: E_r = V(t)/(r ln(b/a)),  E_z = E_A
#   dense edges:  E_r = (1 - shield) V(t)/(r ln(b/a)),  E_z = 0
# with V(t) the rod voltage at t = dt, including any time ramp, and
# E_A = -A0/T for a ramped A_external (0 otherwise).

import json

import numpy as np

with open("params.json") as f:
    p = json.load(f)
d = np.load("fields_step1.npz")
args = p["args"]

V0, a, b, nr, dt, T = p["V0"], p["a"], p["b"], p["nr"], p["dt"], p["T"]
dr = b / nr
rho_floor = p["rho_floor"]

ramp = args["ramp_phi_time"] or args["ramp_eb_potential"]
V_t = V0 * (1.0 + dt / T) if ramp else V0
E_A = -p["A0"] / T if args["ramp_A"] else 0.0
shield = args["shield"]

Er, Ez, rho = d["Er"], d["Ez"], d["rho"]
# rho is nodal; Er is cell-centred in r and nodal in z; Ez is nodal in r and
# cell-centred in z. Interpolate rho to the edges as the Ohm's-law solver does.
rho_er = 0.5 * (rho[:-1, : Er.shape[1]] + rho[1:, : Er.shape[1]])
rho_ez = 0.5 * (rho[: Ez.shape[0], :-1] + rho[: Ez.shape[0], 1:])[:, : Ez.shape[1]]
r_er = (np.arange(Er.shape[0]) + 0.5) * dr
r_ez = np.arange(Ez.shape[0]) * dr

E_coax = V_t / (r_er * np.log(b / a))
E_scale = V0 / (a * np.log(b / a))
# leave out the edges next to the EB (cut cells, masked edges)
away_from_eb_er = (r_er > a + 1.5 * dr)[:, None] * np.ones_like(Er, dtype=bool)
away_from_eb_ez = (r_ez > a + 1.5 * dr)[:, None] * np.ones_like(Ez, dtype=bool)

vac_er = away_from_eb_er & (rho_er < 0.5 * rho_floor)
den_er = away_from_eb_er & (rho_er > 2.0 * rho_floor)
vac_ez = away_from_eb_ez & (rho_ez < 0.5 * rho_floor) & (r_ez < b - 0.5 * dr)[:, None]
den_ez = away_from_eb_ez & (rho_ez > 2.0 * rho_floor)
assert vac_er.sum() > 0 and den_er.sum() > 0 and vac_ez.sum() > 0 and den_ez.sum() > 0

E_coax2 = np.broadcast_to(E_coax[:, None], Er.shape)

# 1. vacuum edges: coaxial field of the rod at the voltage V(t)
err_vac = np.max(np.abs(Er[vac_er] - E_coax2[vac_er]) / E_coax2[vac_er])
print(
    f"vacuum E_r vs coax(V(t)={V_t}): max rel err {err_vac:.3e} on {vac_er.sum()} edges"
)
assert err_vac < 3.0e-3

# 2. dense edges: shielded -> 0, unshielded -> coaxial field
if shield:
    err_den = np.max(np.abs(Er[den_er])) / E_scale
    print(
        f"dense E_r (shielded): max |E_r|/E_scale {err_den:.3e} on {den_er.sum()} edges"
    )
    assert err_den < 1.0e-9
else:
    err_den = np.max(np.abs(Er[den_er] - E_coax2[den_er]) / E_coax2[den_er])
    print(
        f"dense E_r (unshielded) vs coax: max rel err {err_den:.3e} on {den_er.sum()} edges"
    )
    assert err_den < 3.0e-3

# 3. E_z: the inductive field of A_ext on vacuum edges (once, not twice),
#    and removed on dense edges regardless of the shielding toggle
err_ez_vac = np.max(np.abs(Ez[vac_ez] - E_A)) / E_scale
print(
    f"vacuum E_z vs E_A={E_A}: max abs err/E_scale {err_ez_vac:.3e} on {vac_ez.sum()} edges"
)
assert err_ez_vac < 1.0e-6
err_ez_den = np.max(np.abs(Ez[den_ez])) / E_scale
print(f"dense E_z: max |E_z|/E_scale {err_ez_den:.3e} on {den_ez.sum()} edges")
assert err_ez_den < 1.0e-9

print("all checks passed")
