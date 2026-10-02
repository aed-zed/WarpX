#!/usr/bin/env python3
#
# This file is part of WarpX.
#
# License: BSD-3-Clause-LBNL

"""Analysis for the coaxial space-charge diode example.

Self-contained: the Langmuir-Blodgett law and a matched cold-beam oracle
(finite, relativistic birth speed, found by shooting on the cathode field)
are both reimplemented inline, from ``diode_result.npz`` alone -- no import
from outside this directory.

With the clamp armed (``--arm on``), this asserts that:

* the time-averaged potential profile in the gap matches the matched oracle
  to within 2e-3 of the 1 kV gap voltage (measured ~6.5e-4 at ``--nr 48``,
  improving with resolution, see ``README.rst``),
* the current collected at the anode (from the EB particle-boundary buffer)
  matches the injected Langmuir-Blodgett current to within 1e-3, and
* the final gap voltage is held to within 1e-6 V of 1 kV.

With the clamp off, the gap is expected to have discharged well below 1 kV;
no quantitative comparison to the LB law is meaningful in that case, so only
a qualitative check and a printed summary are produced.
"""

import sys

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

EPS0 = 8.8541878128e-12
Q_E = 1.602176634e-19
M_E = 9.1093837015e-31
C_L = 299792458.0


# -- Langmuir-Blodgett law (coaxial space-charge-limited current) -----------
#
# I. Langmuir and K. B. Blodgett, Phys. Rev. 22, 347 (1923); see also the
# review P. Zhang, A. Valfells, L. K. Ang, J. W. Luginsland, and Y. Y. Lau,
# "100 years of the physics of diodes", Appl. Phys. Rev. 4, 011304 (2017).
#
# With gamma = ln(r / r_cathode), beta(gamma) solves
#   3 b b'' + b'^2 + 4 b b' + b^2 = 1,  b ~ gamma - 0.4 gamma^2  (gamma -> 0).
# Current per unit length:
#   I / L = (8 pi eps0 / 9) sqrt(2|q|/m) V^{3/2} / (r_anode beta^2(r_anode)).
# Potential at radius r (anode at V, cathode at 0):
#   V(r) = V (r beta^2(r) / (r_anode beta^2(r_anode)))^{2/3}.
def lb_beta_of_gamma(gammas):
    gammas = np.atleast_1d(np.asarray(gammas, dtype=float))
    g0 = 1.0e-4
    b0 = g0 - 0.4 * g0**2 + 11.0 / 120.0 * g0**3
    db0 = 1.0 - 0.8 * g0 + 11.0 / 40.0 * g0**2

    def rhs(_g, y):
        b, db = y
        return [db, (1.0 - b * b - db * db - 4.0 * b * db) / (3.0 * b)]

    gmax = max(float(np.max(gammas)), g0 * 2)
    sol = solve_ivp(
        rhs, (g0, gmax), [b0, db0], rtol=1e-12, atol=1e-14, dense_output=True
    )
    out = np.empty_like(gammas)
    small = gammas <= g0
    out[small] = gammas[small] - 0.4 * gammas[small] ** 2
    out[~small] = sol.sol(gammas[~small])[0]
    return out


def lb_current_per_length(voltage, r_cathode, r_anode):
    b2 = lb_beta_of_gamma(np.log(r_anode / r_cathode))[0] ** 2
    return (
        8.0
        * np.pi
        * EPS0
        / 9.0
        * np.sqrt(2.0 * Q_E / M_E)
        * abs(voltage) ** 1.5
        / (r_anode * b2)
    )


def lb_potential_profile(r, voltage_gap, r_cathode, r_anode):
    """|potential above cathode| at radius r for the given anode-cathode gap voltage."""
    r = np.asarray(r, dtype=float)
    f = r * lb_beta_of_gamma(np.log(np.maximum(r, r_cathode) / r_cathode)) ** 2
    fa = r_anode * lb_beta_of_gamma(np.log(r_anode / r_cathode))[0] ** 2
    return abs(voltage_gap) * (f / fa) ** (2.0 / 3.0)


# -- Matched cold-beam oracle -------------------------------------------------
#
# Steady cold radial beam between coaxial cylinders, matched to the
# simulation's finite birth speed v0 (the Langmuir-Blodgett law above assumes
# v0 = 0). Unknown: psi(r) = phi(r) - phi(cathode) >= 0. Electrons leave
# r = a with speed v0 and current I (magnitude) over length L.
#   gamma(r) = gamma0 + e psi / (m c^2)                         (relativistic)
#   n e = I / (2 pi r L v(r)),  rho = -n e
#   (1/r) d/dr (r dpsi/dr) = -rho/eps0 = I / (2 pi eps0 L r v)
# BCs: psi(a) = 0, psi(b) = V (gap voltage); shoot on E_a = dpsi/dr(a).
def _beam_speed(psi, v0, relativistic):
    if relativistic:
        g0 = 1.0 / np.sqrt(1.0 - (v0 / C_L) ** 2)
        g = np.maximum(g0 + Q_E * psi / (M_E * C_L**2), 1.0 + 1e-30)
        v = C_L * np.sqrt(1.0 - 1.0 / g**2)
    else:
        v = np.sqrt(np.maximum(v0**2 + 2.0 * Q_E * psi / M_E, 0.0))
    return np.maximum(v, 1e-3 * v0)


def _beam_integrate(slope_a, current, length, a, b, v0, relativistic, r_eval=None):
    k = current / (2.0 * np.pi * EPS0 * length)

    def rhs(r, y):
        psi, dpsi = y
        return [dpsi, k / (r * _beam_speed(psi, v0, relativistic)) - dpsi / r]

    return solve_ivp(
        rhs,
        (a, b),
        [0.0, slope_a],
        rtol=1e-11,
        atol=1e-14,
        t_eval=r_eval,
        dense_output=True,
    )


def matched_beam_oracle(voltage, current, length, a, b, v0, relativistic=True):
    """Return psi(r) (potential above cathode) matched to the given birth speed."""

    def miss(s):
        return _beam_integrate(s, current, length, a, b, v0, relativistic).y[
            0, -1
        ] - abs(voltage)

    hi = abs(voltage) / (b - a) * 5.0
    lo = -1e-6 * abs(voltage) / (b - a)
    while miss(lo) > 0:
        lo *= 10.0
    s = brentq(miss, lo, hi, xtol=1e-14 * abs(voltage) / (b - a), maxiter=200)
    sol = _beam_integrate(s, current, length, a, b, v0, relativistic)
    return lambda r: sol.sol(np.asarray(r))[0]


# -- Load the simulation result -----------------------------------------------
d = np.load("diode_result.npz")
arm = str(d["arm"])
nr = int(d["nr"])
a_eff, b_eff, length = float(d["a_eff"]), float(d["b_eff"]), float(d["L"])
v_birth = float(d["v_birth"])
I_inj = float(d["I_inj"])
V0 = float(d["V0"])
r_profile = d["r_profile"]
phi_above = d["phi_above"]
t = d["t"]
v_line = d["V_line"]
v_obs = d["V_obs"]
i_collected = float(d["i_collected"])
injected_charge = float(d["injected_charge"])

print(
    f"== coaxial space-charge diode: arm={arm} nr={nr} a_eff={a_eff * 1e3:.4f} mm "
    f"b_eff={b_eff * 1e3:.4f} mm I_LB={I_inj:.4e} A"
)
print(f"  final gap voltage (line integral) = {v_line[-1]:.4f} V ; target = {V0:.1f} V")
print(
    f"  collected current (EB buffer, last transit) = {i_collected:.4e} A ; "
    f"injected = {I_inj:.4e} A"
)

if arm == "on":
    psi_matched = matched_beam_oracle(
        abs(V0), I_inj, length, a_eff, b_eff, v_birth, relativistic=True
    )
    phi_matched = psi_matched(r_profile)
    phi_lb = lb_potential_profile(r_profile, abs(V0), a_eff, b_eff)

    # exclude the cathode node (psi = 0 there by construction on both sides)
    sel = slice(1, None)
    err_matched = np.max(np.abs(phi_above[sel] - phi_matched[sel])) / 1000.0
    err_lb = np.max(np.abs(phi_above[sel] - phi_lb[sel])) / 1000.0
    err_oracle_vs_lb = np.max(np.abs(phi_matched[sel] - phi_lb[sel])) / 1000.0

    print(f"  max |phi_sim - phi_matched| / 1 kV = {err_matched:.3e}")
    print(f"  max |phi_sim - phi_LB|      / 1 kV = {err_lb:.3e}")
    print(
        f"  max |phi_matched - phi_LB|  / 1 kV = {err_oracle_vs_lb:.3e} "
        f"(birth energy {0.5 * M_E * v_birth**2 / Q_E:.2f} eV shift)"
    )

    print("\n  r [mm]    phi_sim [V]  phi_matched [V]  phi_LB [V]")
    n_rows = min(10, r_profile.size)
    idx = np.linspace(0, r_profile.size - 1, n_rows).round().astype(int)
    for i in idx:
        print(
            f"  {r_profile[i] * 1e3:7.3f}  {phi_above[i]:11.3f}  {phi_matched[i]:14.3f}  "
            f"{phi_lb[i]:9.3f}"
        )

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(r_profile * 1e3, phi_above, "o", ms=3, label="WarpX (time-avg)")
        ax.plot(r_profile * 1e3, phi_matched, "-", label="matched oracle")
        ax.plot(r_profile * 1e3, phi_lb, "--", label="Langmuir-Blodgett")
        ax.set_xlabel("r [mm]")
        ax.set_ylabel("potential above cathode [V]")
        ax.legend()
        fig.tight_layout()
        fig.savefig("diode_profile.png", dpi=150)
    except ImportError:
        pass

    assert err_matched < 2.0e-3, (
        f"gap potential profile does not match the matched oracle: {err_matched:.3e}"
    )
    assert abs(i_collected - I_inj) / I_inj < 1.0e-3, (
        f"collected current does not match the injected Langmuir-Blodgett current: "
        f"collected={i_collected:.4e} A injected={I_inj:.4e} A"
    )
    assert abs(v_line[-1] - V0) < 1.0e-6, (
        f"the clamp did not hold the gap voltage: final={v_line[-1]:.6f} V target={V0:.1f} V"
    )
else:
    print(
        "  clamp off: no quantitative comparison to the Langmuir-Blodgett law is meaningful"
    )
    if abs(v_line[-1]) > 0.5 * abs(V0):
        print(
            "  WARNING: expected the unclamped gap to have discharged well below 1 kV",
            file=sys.stderr,
        )

print("done.")
