#!/usr/bin/env python3
"""Plots + summary.json for the four-arm electrode-potential-maintenance
comparison campaign (arm_a/b/c/d.npz, written by inputs_3d_ect_fourarm_v6.py).

Usage: python fourarm_v6_plots.py [--dir fourarm_v6]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# -- fixed categorical palette (Okabe-Ito subset), validated colorblind-safe --
ARM_COLOR = {
    "a": "#0072B2",
    "b": "#E69F00",
    "c": "#009E73",
    "d": "#D55E00",
}
ARM_LABEL = {
    "a": "no correction",
    "b": "bias-only every 10 (historical)",
    "c": "bias-only + ledger every 10",
    "d": "ledger + every-step (reciprocity)",
}
ARMS = ["a", "b", "c", "d"]

p = argparse.ArgumentParser()
p.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)) + "/fourarm_v6")
args = p.parse_args()
D = args.dir

DATA = {}
for arm in ARMS:
    path = os.path.join(D, f"arm_{arm}.npz")
    if os.path.exists(path):
        DATA[arm] = np.load(path, allow_pickle=True)
    else:
        print(f"WARNING: missing {path}, arm {arm} will be skipped in plots")

V_LEFT = float(DATA[ARMS[0]]["V_left"]) if ARMS[0] in DATA else 300.0
V_RIGHT = float(DATA[ARMS[0]]["V_right"]) if ARMS[0] in DATA else -700.0
R = float(next(iter(DATA.values()))["R"])
CENTER_OFFSET = float(next(iter(DATA.values()))["center_offset"])
NX = int(next(iter(DATA.values()))["nx"])
FIXTURE_TITLE = (
    f"clamp-work fixture: spheres R={R*1e2:.2f} cm at ±{CENTER_OFFSET*1e2:.1f} cm, "
    f"V=[{V_LEFT:+.0f}, {V_RIGHT:+.0f}] V, ECT solver, {NX}^3 grid"
)

os.makedirs(D, exist_ok=True)


# ---------------------------------------------------------------------------
# p1: voltage traces, 2 panels (left/right electrode)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
for ax, col, label, target in zip(axes, (0, 1), ("Left electrode", "Right electrode"),
                                  (V_LEFT, V_RIGHT)):
    for arm in ARMS:
        if arm not in DATA:
            continue
        d = DATA[arm]
        if "voltage_trace_steps" not in d:
            continue
        steps = d["voltage_trace_steps"]
        V = d["voltage_trace_V"][:, col]
        ax.plot(steps, V, color=ARM_COLOR[arm], lw=1.4, label=f"arm {arm}: {ARM_LABEL[arm]}")
    ax.axhline(target, color="0.3", ls="--", lw=1.0, label=f"target {target:+.0f} V" if ax is axes[0] else None)
    ax.set_xlabel("step")
    ax.set_title(label)
    ax.grid(alpha=0.25)
axes[0].set_ylabel("measured voltage [V]")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle(f"Measured per-electrode voltage vs step\n{FIXTURE_TITLE}", fontsize=11)
fig.tight_layout(rect=[0, 0.06, 1, 0.94])
fig.savefig(os.path.join(D, "p1_voltage_traces.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote p1_voltage_traces.png")


# ---------------------------------------------------------------------------
# p2: |V_measured - V_target| vs step, log-y, both electrodes
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
for ax, col, label, target in zip(axes, (0, 1), ("Left electrode", "Right electrode"),
                                  (V_LEFT, V_RIGHT)):
    for arm in ARMS:
        if arm not in DATA:
            continue
        d = DATA[arm]
        if "voltage_trace_steps" not in d:
            continue
        steps = d["voltage_trace_steps"]
        V = d["voltage_trace_V"][:, col]
        err = np.abs(V - target)
        err = np.clip(err, 1e-6, None)
        ax.semilogy(steps, err, color=ARM_COLOR[arm], lw=1.4, label=f"arm {arm}: {ARM_LABEL[arm]}")
    ax.set_xlabel("step")
    ax.set_title(label)
    ax.grid(alpha=0.25, which="both")
axes[0].set_ylabel("|V measured - V target| [V]")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))
fig.suptitle(f"Voltage-maintenance error vs step (log scale)\n{FIXTURE_TITLE}", fontsize=11)
fig.tight_layout(rect=[0, 0.09, 1, 0.94])
fig.savefig(os.path.join(D, "p2_voltage_error.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote p2_voltage_error.png")


# ---------------------------------------------------------------------------
# p3: cumulative mis-booked charge vs step -- arm b (implied, uncorrected)
# vs arms c/d (measured+corrected); impact count on a twin axis (explicitly
# requested cross-scale overlay, not a dual y-scale of the same quantity).
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(9, 5.5))
ax2 = ax1.twinx()
for arm in ("b", "c", "d"):
    if arm not in DATA:
        continue
    d = DATA[arm]
    if "ledger_steps" not in d:
        continue
    steps = d["ledger_steps"]
    booked = d["ledger_booked"]          # (T, n_electrodes, n_species)
    counts = d["ledger_counts"]          # (T, n_species)
    total_booked = np.abs(booked).sum(axis=(1, 2))
    total_counts = counts.sum(axis=1)
    ax1.plot(steps, total_booked, color=ARM_COLOR[arm], lw=1.6,
             label=f"arm {arm}: {ARM_LABEL[arm]} -- |booked| charge")
    ax2.plot(steps, total_counts, color=ARM_COLOR[arm], lw=1.0, ls=":", alpha=0.7)
ax1.set_xlabel("step")
ax1.set_ylabel("cumulative |ledger booked charge| [C]  (solid)")
ax2.set_ylabel("cumulative absorbed-impact count  (dotted)")
ax1.grid(alpha=0.25)
ax1.legend(loc="upper left", frameon=False, fontsize=9)
ax1.set_title(f"Absorption ledger: mis-booked charge and impact count\n{FIXTURE_TITLE}", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(D, "p3_ledger.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote p3_ledger.png")


# ---------------------------------------------------------------------------
# p4: grid of Ez midplane slices -- rows = arms (a, b, d), cols = steps
# ---------------------------------------------------------------------------
SNAP_COLS = [300, 600, 750, 2000]
SNAP_ROWS = ["a", "b", "d"]

# shared symmetric color scale from arm a step 600
vmax = 1.0
if "a" in DATA and "snapshot_steps" in DATA["a"]:
    d = DATA["a"]
    ss = list(d["snapshot_steps"])
    if 600 in ss:
        idx = ss.index(600)
        vmax = float(np.nanmax(np.abs(d["snapshot_Ez"][idx])))
if not np.isfinite(vmax) or vmax <= 0:
    vmax = 1.0

fig, axes = plt.subplots(len(SNAP_ROWS), len(SNAP_COLS),
                         figsize=(4 * len(SNAP_COLS), 3.6 * len(SNAP_ROWS)),
                         squeeze=False)
im_ref = None
for r, arm in enumerate(SNAP_ROWS):
    d = DATA.get(arm)
    ss = list(d["snapshot_steps"]) if d is not None and "snapshot_steps" in d else []
    for c, step in enumerate(SNAP_COLS):
        ax = axes[r][c]
        if d is not None and step in ss:
            idx = ss.index(step)
            Ez = d["snapshot_Ez"][idx]
            im_ref = ax.imshow(Ez.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               aspect="auto")
        else:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
        if r == 0:
            ax.set_title(f"step {step}")
        if c == 0:
            ax.set_ylabel(f"arm {arm}\n{ARM_LABEL[arm]}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
if im_ref is not None:
    cbar = fig.colorbar(im_ref, ax=axes, shrink=0.7, pad=0.02)
    cbar.set_label("Ez [V/m] (y=0 midplane)")
fig.suptitle(f"Ez midplane slices bracketing the absorption burst\n{FIXTURE_TITLE}", fontsize=11)
fig.savefig(os.path.join(D, "p4_snapshots.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote p4_snapshots.png")


# ---------------------------------------------------------------------------
# p5: wall time per arm + correction-time share
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
arms_present = [a for a in ARMS if a in DATA]
totals = [float(DATA[a]["total_wall_time_s"]) for a in arms_present]
corr = [float(DATA[a]["correction_seconds"]) for a in arms_present]
colors = [ARM_COLOR[a] for a in arms_present]

axes[0].bar(arms_present, totals, color=colors)
axes[0].set_ylabel("total wall time [s]")
axes[0].set_xlabel("arm")
axes[0].set_title("Total wall time per arm")
for i, t in enumerate(totals):
    axes[0].text(i, t, f"{t:.0f}s", ha="center", va="bottom", fontsize=9)

share = [100.0 * c / t if t > 0 else 0.0 for c, t in zip(corr, totals)]
axes[1].bar(arms_present, share, color=colors)
axes[1].set_ylabel("correction-time share [%]\n(measure_voltages + apply_bias)")
axes[1].set_xlabel("arm")
axes[1].set_title("Correction cost as % of total wall time")
for i, s in enumerate(share):
    axes[1].text(i, s, f"{s:.1f}%", ha="center", va="bottom", fontsize=9)

fig.suptitle(f"Wall-time cost per arm\n{FIXTURE_TITLE}", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(D, "p5_cost.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("wrote p5_cost.png")


# ---------------------------------------------------------------------------
# summary.json
# ---------------------------------------------------------------------------
summary = {}
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
        if burst.any():
            entry["max_abs_V_error_burst_350_800"] = err[burst].max(axis=0).tolist()
        else:
            entry["max_abs_V_error_burst_350_800"] = [None, None]

        late = steps > 800
        if late.any():
            entry["time_avg_abs_V_error_after_800"] = err[late].mean(axis=0).tolist()
        else:
            entry["time_avg_abs_V_error_after_800"] = [None, None]

    entry["total_wall_time_s"] = float(d["total_wall_time_s"])
    entry["init_wall_time_s"] = float(d["init_wall_time_s"]) if "init_wall_time_s" in d else None
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

    summary[arm] = entry

with open(os.path.join(D, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("wrote summary.json")
