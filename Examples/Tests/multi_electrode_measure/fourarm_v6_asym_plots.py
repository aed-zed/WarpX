#!/usr/bin/env python3
"""Plots + summary_asym.json for the asymmetric-sphere two-arm campaign
(arm_b/arm_d.npz, written by inputs_3d_ect_fourarm_v6.py --r_right ...).

This is a standalone sibling of fourarm_v6_plots.py (which hard-requires all
four arms a-d and would break on a b/d-only directory): it only ever looks at
arm b and arm d, and it reads the TRUE right-sphere radius (RR) out of each
npz rather than assuming a fixed value, so the plots stay honest even if the
requested --r_right had to be nudged to dodge a grid-alignment singularity
(see the campaign report for why 0.018 itself is unusable at nx=80 -- an
exact-multiple-of-dx radius on an unequal sphere pair NaNs the capacitance
matrix; the runs here use 0.018001, i.e. +1 micron, physically 1.8000 cm).

Usage: python fourarm_v6_asym_plots.py [--dir fourarm_v6/asym]
"""

import argparse
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

ARM_COLOR = {"b": "#E69F00", "d": "#D55E00"}
ARM_LABEL = {
    "b": "bias-only every 10 (historical)",
    "d": "ledger + every-step (reciprocity)",
}
ARMS = ["b", "d"]

p = argparse.ArgumentParser()
p.add_argument(
    "--dir",
    default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fourarm_v6", "asym"
    ),
)
args = p.parse_args()
D = args.dir
os.makedirs(D, exist_ok=True)

DATA = {}
for arm in ARMS:
    path = os.path.join(D, f"arm_{arm}.npz")
    if os.path.exists(path):
        DATA[arm] = np.load(path, allow_pickle=True)
    else:
        print(f"WARNING: missing {path}, arm {arm} will be skipped")

if not DATA:
    raise SystemExit(f"No arm_{{b,d}}.npz found in {D}")

any_d = next(iter(DATA.values()))
V_LEFT = float(any_d["V_left"])
V_RIGHT = float(any_d["V_right"])
R_LEFT = float(any_d["R"])
# RR: true right-sphere radius as actually used in the EB implicit function
# (may differ infinitesimally from a nominal "1.8 cm" request -- see module
# docstring). Falls back to R_LEFT for any npz written before the --r_right
# plumbing (equal-radii case).
R_RIGHT = float(any_d["RR"]) if "RR" in any_d else R_LEFT
CENTER_OFFSET = float(any_d["center_offset"])
NX = int(any_d["nx"])
DX = float(any_d["dx"])
HALF = float(any_d["half"])
NZ = int(any_d["nz"])

FIXTURE_TITLE = (
    f"asymmetric spheres: R_left={R_LEFT*1e2:.2f} cm, R_right={R_RIGHT*1e2:.4f} cm "
    f"(nominal 1.8 cm) at ±{CENTER_OFFSET*1e2:.1f} cm, "
    f"V=[{V_LEFT:+.0f}, {V_RIGHT:+.0f}] V, ECT solver, {NX}^3 grid"
)

# ---------------------------------------------------------------------------
# pa1: |V_measured - V_target| vs step, log-y, both electrodes, arms b & d
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
for ax, col, label, target in zip(
    axes, (0, 1), ("Left electrode (R=1.5 cm)", "Right electrode (R=1.8 cm)"),
    (V_LEFT, V_RIGHT),
):
    for arm in ARMS:
        if arm not in DATA:
            continue
        d = DATA[arm]
        if "voltage_trace_steps" not in d:
            continue
        steps = d["voltage_trace_steps"]
        V = d["voltage_trace_V"][:, col]
        err = np.clip(np.abs(V - target), 1e-6, None)
        ax.semilogy(steps, err, color=ARM_COLOR[arm], lw=1.4,
                    label=f"arm {arm}: {ARM_LABEL[arm]}")
    ax.set_xlabel("step")
    ax.set_title(label)
    ax.grid(alpha=0.25, which="both")
axes[0].set_ylabel("|V measured - V target| [V]")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, -0.08))
fig.suptitle(
    "Voltage-maintenance error vs step (log scale) -- ASYMMETRIC spheres "
    f"(r_right = 1.8 cm)\n{FIXTURE_TITLE}", fontsize=10,
)
fig.tight_layout(rect=[0, 0.09, 1, 0.92])
fig.savefig(os.path.join(D, "pa1_voltage_error.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote pa1_voltage_error.png")

# ---------------------------------------------------------------------------
# pa2: Ez(step 2000) - Ez(step 0), y=0 midplane, one panel per arm, RdBu_r,
# shared robust color scale (99.5th pct of arm b's bulk), dotted circles at
# the TRUE radii, per-panel bulk RMS in the title.
#
# Coordinate note: snapshot_Ez has shape (n_snap, nx+1, nz) -- axis 1 is
# NODAL in x (nx+1 points at x_i = -half + i*dx, i=0..nx) and axis 2 is
# CELL-CENTERED in z (nz points at z_k = -half + (k+0.5)*dx, k=0..nz-1).
# Using a single shared `extent=[-half,half,-half,half]` for imshow would
# put the dotted TRUE-radius circles half a cell off in z -- use pcolormesh
# with explicit per-axis cell edges instead.
# ---------------------------------------------------------------------------
x_centers = -HALF + np.arange(NX + 1) * DX             # (81,) nodal
x_edges = np.concatenate(([x_centers[0] - DX / 2],
                          (x_centers[:-1] + x_centers[1:]) / 2,
                          [x_centers[-1] + DX / 2]))    # (82,)
z_edges = -HALF + np.arange(NZ + 1) * DX                # (81,) cell edges
z_centers = -HALF + (np.arange(NZ) + 0.5) * DX          # (80,)

Xc, Zc = np.meshgrid(x_centers, z_centers, indexing="ij")  # (81, 80), matches Ez layout

BULK_MARGIN = 0.003  # 0.3 cm


def dist_to(xc):
    return np.sqrt((Xc - xc) ** 2 + Zc ** 2)


dist_left = dist_to(-CENTER_OFFSET)
dist_right = dist_to(CENTER_OFFSET)

bulk_mask = (dist_left >= R_LEFT + BULK_MARGIN) & (dist_right >= R_RIGHT + BULK_MARGIN)
inside_mask = (dist_left < R_LEFT) | (dist_right < R_RIGHT)


def field_diff(d):
    steps = list(d["snapshot_steps"])
    if 0 not in steps or 2000 not in steps:
        return None
    i0 = steps.index(0)
    i1 = steps.index(2000)
    return d["snapshot_Ez"][i1] - d["snapshot_Ez"][i0]  # (81, 80)


diffs = {}
bulk_rms = {}
for arm in ARMS:
    if arm not in DATA:
        continue
    fd = field_diff(DATA[arm])
    if fd is None:
        print(f"WARNING: arm {arm} missing snapshot at step 0 or 2000, skipping pa2 panel")
        continue
    diffs[arm] = fd
    bulk_rms[arm] = float(np.sqrt(np.mean(fd[bulk_mask] ** 2)))

if "b" in diffs:
    vmax = float(np.percentile(np.abs(diffs["b"][bulk_mask]), 99.5))
elif diffs:
    vmax = float(np.percentile(np.abs(next(iter(diffs.values()))[bulk_mask]), 99.5))
else:
    vmax = 1.0
if not np.isfinite(vmax) or vmax <= 0:
    vmax = 1.0

fig, axes = plt.subplots(1, len(ARMS), figsize=(6.2 * len(ARMS), 5.6), squeeze=False)
axes = axes[0]
im_ref = None
for ax, arm in zip(axes, ARMS):
    ax.set_facecolor("white")
    if arm not in diffs:
        ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
        continue
    fd = diffs[arm].copy()
    fd_masked = np.where(inside_mask, np.nan, fd)
    im_ref = ax.pcolormesh(
        x_edges * 1e2, z_edges * 1e2, fd_masked.T,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="flat",
    )
    for xc, rad in ((-CENTER_OFFSET, R_LEFT), (CENTER_OFFSET, R_RIGHT)):
        ax.add_patch(Circle((xc * 1e2, 0.0), rad * 1e2, fill=False,
                            ls=":", lw=1.3, edgecolor="k"))
    ax.set_xlim(-6, 6)
    ax.set_ylim(-6, 6)
    ax.set_aspect("equal")
    ax.set_xlabel("x [cm]")
    ax.set_title(
        f"arm {arm}: {ARM_LABEL[arm]}\nbulk RMS = {bulk_rms[arm]:,.0f} V/m",
        fontsize=10,
    )
axes[0].set_ylabel("z [cm]")
fig.tight_layout(rect=[0, 0, 0.9, 0.86])
if im_ref is not None:
    # Attached to a dedicated axes via add_axes (fixed in figure coordinates,
    # AFTER tight_layout) rather than a shared fig.colorbar(ax=axes,...) --
    # the latter's allocated width is computed before set_aspect("equal")
    # shrinks the equal-aspect image axes, so it ends up overlapping the
    # right panel's title instead of sitting clear of both panels.
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.62])
    cbar = fig.colorbar(im_ref, cax=cax)
    cbar.set_label(r"$\Delta E_z$ [V/m]")
fig.suptitle(
    r"Field change over the run: $E_z$(step 2000) $-$ $E_z$(step 0), y=0 midplane"
    f"\nscale = 99.5th pct of arm b's bulk; dotted circles at TRUE radii "
    f"(left {R_LEFT*1e2:.2f} cm, right {R_RIGHT*1e2:.4f} cm)\n{FIXTURE_TITLE}",
    fontsize=9.5,
)
fig.savefig(os.path.join(D, "pa2_field_difference.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote pa2_field_difference.png")

def _parse_capacitance(log_path):
    """Pull the '[MultiElectrode] capacitance matrix (cond=...):\\n[[a b]\\n
    [c d]]' block that setup_after_init() prints (verbose=True) out of an
    arm's init log, and compute the off-diagonal asymmetry -- the interesting
    number for an unequal-sphere pair (a symmetric pair gives C_01 == C_10
    exactly by construction)."""
    if not os.path.exists(log_path):
        return None
    with open(log_path) as f:
        text = f.read()
    m = re.search(
        r"capacitance matrix \(cond=([\d.eE+-]+)\):\s*\n"
        r"\[\[\s*([\d.eE+-]+)\s+([\d.eE+-]+)\]\s*\n"
        r"\s*\[\s*([\d.eE+-]+)\s+([\d.eE+-]+)\]\]",
        text,
    )
    if not m:
        return None
    cond, c00, c01, c10, c11 = (float(g) for g in m.groups())
    rel_asym = 2.0 * abs(c01 - c10) / (abs(c01) + abs(c10)) if (c01 or c10) else 0.0
    return {
        "cond": cond, "C_00": c00, "C_01": c01, "C_10": c10, "C_11": c11,
        "C_01_minus_C_10": c01 - c10,
        "relative_offdiag_asymmetry_pct": rel_asym * 100.0,
        "C_11_over_C_00": c11 / c00 if c00 else None,
    }


capacitance = {}
for arm in ARMS:
    cap = _parse_capacitance(os.path.join(D, f"arm_{arm}.log"))
    if cap is not None:
        capacitance[arm] = cap

# ---------------------------------------------------------------------------
# summary_asym.json
# ---------------------------------------------------------------------------
summary = {
    "r_right_requested_m": 0.018,
    "r_right_used_m": R_RIGHT,
    "r_left_m": R_LEFT,
    "capacitance_matrix": capacitance,
    "note": (
        "r_right=0.018 (exactly 12*dx at this fixture's dx=1.5e-3 m) NaNs the "
        "capacitance matrix (numpy SVD does not converge on an all-NaN 2x2 "
        "matrix) -- reproduced in both this driver and the unmodified "
        "inputs_3d_t6_booking_exactness.py at nx=80 for ANY unequal radius "
        "landing exactly on a grid line (9,12,13,14 * dx all fail; "
        "non-grid-aligned radii, including a +1 micron nudge off the same "
        "values, all succeed). Ran with r_right=0.018001 m instead "
        "(+1e-6 m, 0.0007*dx) to dodge the singularity -- physically "
        "indistinguishable from 1.8000 cm."
    ),
}

for arm in ARMS:
    if arm not in DATA:
        summary[arm] = {"status": "missing"}
        continue
    d = DATA[arm]
    entry = {"label": ARM_LABEL[arm], "status": "ok"}

    if "voltage_trace_steps" in d:
        steps = d["voltage_trace_steps"]
        V = d["voltage_trace_V"]
        err = np.abs(V - np.array([V_LEFT, V_RIGHT]))
        entry["final_V_error"] = err[-1].tolist()
        burst = (steps >= 350) & (steps <= 800)
        entry["max_abs_V_error_burst_350_800"] = (
            err[burst].max(axis=0).tolist() if burst.any() else [None, None]
        )
        late = steps > 800
        entry["time_avg_abs_V_error_after_800"] = (
            err[late].mean(axis=0).tolist() if late.any() else [None, None]
        )

    entry["total_wall_time_s"] = float(d["total_wall_time_s"])
    entry["init_wall_time_s"] = (
        float(d["init_wall_time_s"]) if "init_wall_time_s" in d else None
    )
    entry["steps_wall_time_s"] = float(d["steps_wall_time_s"])
    entry["correction_seconds"] = float(d["correction_seconds"])
    entry["measure_voltages_s"] = float(d["measure_voltages_s"])
    entry["apply_bias_s"] = float(d["apply_bias_s"])

    if "ledger_counts_final" in d:
        entry["impact_counts_final"] = d["ledger_counts_final"].tolist()
    if "ledger_deficit_final" in d:
        entry["final_cumulative_deficit"] = d["ledger_deficit_final"].tolist()
    if "ledger_booked_final" in d:
        entry["final_cumulative_booked"] = d["ledger_booked_final"].tolist()

    entry["total_W_clamp_J"] = float(d["total_W_clamp"])
    entry["W_absorbed_total_J"] = float(d["W_absorbed_total"])
    entry["Q_absorbed_left_C"] = float(d["Q_absorbed_left"])
    entry["Q_absorbed_right_C"] = float(d["Q_absorbed_right"])

    if arm in bulk_rms:
        entry["delta_Ez_bulk_rms_V_per_m"] = bulk_rms[arm]

    summary[arm] = entry

with open(os.path.join(D, "summary_asym.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("wrote summary_asym.json")
