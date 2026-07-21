#!/usr/bin/env python3
"""Generate PNG visualizations from Gauss-box test output.

Reads gauss_data.npz (saved by inputs_3d_gauss_movie.py,
inputs_3d_gauss_impact.py, inputs_3d_gauss_correction.py, or
inputs_3d_gauss_comparison.py) and the ChargeOnEB / FieldEnergy reduced
diagnostic files, then saves PNG figures:

  frames/frame_NNNN.png   — Ex + density movie frames
  charge_drift.png        — Q(t) drift plot
  ex_difference.png       — Ex(final) - Ex(initial) with symlog colorscale
  correction_snapshots.png   — (if correction data) Ex before/after variants
  correction_diffs.png       — (if correction data) difference maps (symlog)
  field_energy.png           — (if FieldEnergy data) W_E timeline

Usage:
    python analysis_gauss_visualization.py [data_dir]

    data_dir: directory containing gauss_data.npz and diags/ (default: cwd)
"""

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import SymLogNorm  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
npz_path = os.path.join(data_dir, "gauss_data.npz")
d = np.load(npz_path, allow_pickle=True)

dx = float(d["dx"])
half = float(d["half"])
nx = int(d["nx"])
nz = int(d["nz"])
dz = dx
dt_sim = float(d["dt_sim"])
R = float(d["R"])
center_offset = float(d["center_offset"])
V_left = float(d["V_left"])
V_right = float(d["V_right"])
h_boxes = d["h_boxes"]
z_beam_lo = float(d["z_beam_lo"])
vz_drift = float(d["vz_drift"])
cells_per_R = int(d["cells_per_R"])

title = str(d["title"])
subtitle = str(d["subtitle"])
beam_desc = str(d["beam_desc"])

field_steps = d["field_steps"]
field_data = d["field_data"]
density_pos = d["density_pos"]
density_neg = d["density_neg"]

box_labels = ["close", "mid", "far"]
q = {}
for s in ("left", "right"):
    for hl in box_labels:
        q[f"{s}_{hl}"] = d[f"q_{s}_{hl}"]

has_correction = "correction_step" in d
has_periodic = "correction_interval" in d
corr_format = None  # "two_stage" or "three_variant"

if has_periodic:
    correction_interval = int(d["correction_interval"])
    correction_steps_arr = d["correction_steps"]
    V_before_history = d["V_before_history"]
    V_after_history = d["V_after_history"]
    print(f"  Periodic correction every {correction_interval} steps, "
          f"{len(correction_steps_arr)} corrections applied")

if has_correction:
    correction_step = int(d["correction_step"])

    if "ex_pre" in d:
        corr_format = "three_variant"
        ex_pre = d["ex_pre"]
        V_pre = d["V_pre"]
        W_pre = float(d["W_pre"]) if "W_pre" in d else None
        variants = {}
        for vname in ("full_bias", "clean_bias", "masked_bias"):
            vd = {"ex": d[f"ex_{vname}"], "V": d[f"V_{vname}"]}
            if f"W_{vname}" in d:
                vd["W"] = float(d[f"W_{vname}"])
            variants[vname] = vd
        print(f"  Correction at step {correction_step} (3-variant), "
              f"V_pre={V_pre}")
    else:
        corr_format = "two_stage"
        ex_pre_clean = d["ex_pre_clean"]
        ex_post_clean = d["ex_post_clean"]
        ex_post_bias = d["ex_post_bias"]
        V_pre_clean = d["V_pre_clean"]
        V_post_clean = d["V_post_clean"]
        V_post_bias = d["V_post_bias"]
        corr_q = {}
        corr_eb_q = {}
        for s in ("left", "right"):
            for stage in ("pre_clean", "post_clean", "post_bias"):
                k_q = f"q_{s}_close_{stage}"
                k_eb = f"eb_q_{s}_{stage}"
                if k_q in d:
                    corr_q[f"{s}_{stage}"] = float(d[k_q])
                if k_eb in d:
                    corr_eb_q[f"{s}_{stage}"] = float(d[k_eb])
        print(f"  Correction at step {correction_step} (2-stage), "
              f"V_pre={V_pre_clean}, V_post={V_post_bias}")

n_frames = len(field_steps)

sphere_centers = [
    ("left",  -center_offset, 0.0, 0.0, V_left),
    ("right", +center_offset, 0.0, 0.0, V_right),
]

print(f"Loaded {npz_path}: {n_frames} frames, {nx}^3 grid")

# ---------------------------------------------------------------------------
# Read ChargeOnEB reduced diagnostic files
# ---------------------------------------------------------------------------
eb_charge = {}
for s_name in ("left", "right"):
    fpath = os.path.join(data_dir, f"diags/reducedfiles/Q_eb_{s_name}.txt")
    if os.path.exists(fpath):
        raw = np.loadtxt(fpath, comments="#")
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        eb_charge[s_name] = {"steps": raw[:, 0].astype(int),
                             "time": raw[:, 1], "Q": raw[:, 2]}
        print(f"  ChargeOnEB {s_name}: {len(raw)} rows, "
              f"Q_0 = {raw[0, 2]:.4e}, Q_end = {raw[-1, 2]:.4e}")
    else:
        print(f"  ChargeOnEB {s_name}: not found at {fpath}")

# Read FieldEnergy reduced diagnostic
field_energy_data = None
fe_path = os.path.join(data_dir, "diags/reducedfiles/field_energy.txt")
if os.path.exists(fe_path):
    fe_raw = np.loadtxt(fe_path, comments="#")
    if fe_raw.ndim == 1:
        fe_raw = fe_raw.reshape(1, -1)
    field_energy_data = {
        "steps": fe_raw[:, 0].astype(int),
        "time": fe_raw[:, 1],
        "W_E": fe_raw[:, 2],
    }
    print(f"  FieldEnergy: {len(fe_raw)} rows, "
          f"W_0 = {fe_raw[0, 2]:.6e}, W_end = {fe_raw[-1, 2]:.6e} J")

# ---------------------------------------------------------------------------
# Coordinate arrays
# ---------------------------------------------------------------------------
x_cc = -half + (np.arange(nx) + 0.5) * dx
z_n = -half + np.arange(nz + 1) * dz
x_cell = -half + (np.arange(nx) + 0.5) * dx
z_cell = -half + (np.arange(nz) + 0.5) * dz

# ---------------------------------------------------------------------------
# Color scales
# ---------------------------------------------------------------------------
all_ex = np.concatenate([f.ravel() for f in field_data])
ex_vmax = np.percentile(np.abs(all_ex), 99.5)

all_dens = np.concatenate(
    [f.ravel() for f in density_pos] + [f.ravel() for f in density_neg])
dens_vmax = (np.percentile(all_dens[all_dens > 0], 99)
             if np.any(all_dens > 0) else 1)

box_colors = ["#2b6cb0", "#c07800", "#228b22"]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def _draw_boxes(ax, cm_scale=1e2):
    for _, xc, _, zc, _ in sphere_centers:
        for h_val, hl, col in zip(h_boxes, box_labels, box_colors):
            rect = Rectangle(
                ((zc - h_val) * cm_scale, (xc - h_val) * cm_scale),
                2 * h_val * cm_scale, 2 * h_val * cm_scale,
                linewidth=1.2, edgecolor=col, facecolor="none",
                linestyle="--" if hl == "far" else "-." if hl == "mid" else "-",
                zorder=8, label=hl if xc < 0 else None,
            )
            ax.add_patch(rect)


def _draw_spheres(ax, cm_scale=1e2):
    for _, xc, _, _, V in sphere_centers:
        circle = Circle((0, xc * cm_scale), R * cm_scale, fill=True,
                         facecolor="#888888", edgecolor="k", linewidth=1.5,
                         zorder=10)
        ax.add_patch(circle)
        ax.text(0, xc * cm_scale, f"{V:+.0f}V",
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold", zorder=11)


def _save_fig(fig, name):
    """Save figure to data_dir/<name>.png and close it."""
    path = os.path.join(data_dir, f"{name}.png")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


# ---------------------------------------------------------------------------
# 1. Build field + density frames
# ---------------------------------------------------------------------------
frames_dir = os.path.join(data_dir, "frames")
os.makedirs(frames_dir, exist_ok=True)
n_mid = nx // 2

for i in range(n_frames):
    step = int(field_steps[i])
    Ex_slice = field_data[i]
    rho_pos = density_pos[i]
    rho_neg = density_neg[i]
    t_ns = step * dt_sim * 1e9

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    im1 = ax1.pcolormesh(z_n * 1e2, x_cc * 1e2, Ex_slice,
                         cmap="RdBu_r", vmin=-ex_vmax, vmax=ex_vmax,
                         shading="auto", rasterized=True)
    fig.colorbar(im1, ax=ax1, label=r"$E_x$ [V/m]", shrink=0.85, pad=0.02)
    _draw_spheres(ax1)
    _draw_boxes(ax1)
    ax1.set(xlabel="z [cm]", ylabel="x [cm]", title=r"$E_x$ field (y = 0)",
            xlim=(-half * 1e2, half * 1e2), ylim=(-half * 1e2, half * 1e2),
            aspect="equal")

    composite = np.zeros_like(rho_pos)
    composite[n_mid:, :] = rho_pos[n_mid:, :]
    composite[:n_mid, :] = rho_neg[:n_mid, :]

    im2 = ax2.pcolormesh(z_cell * 1e2, x_cell * 1e2, composite,
                         cmap="inferno", vmin=0, vmax=dens_vmax,
                         shading="auto", rasterized=True)
    fig.colorbar(im2, ax=ax2, label=r"density [m$^{-3}$]", shrink=0.85,
                 pad=0.02)
    _draw_spheres(ax2)
    _draw_boxes(ax2)
    ax2.axhline(0, color="white", linewidth=0.8, alpha=0.6, zorder=9)
    ax2.text(half * 1e2 * 0.92, 0.8, "+ species", color="white",
             fontsize=7, ha="right", va="bottom", zorder=12)
    ax2.text(half * 1e2 * 0.92, -0.8, "- species", color="white",
             fontsize=7, ha="right", va="top", zorder=12)
    ax2.set(xlabel="z [cm]", ylabel="x [cm]",
            title="Particle density (y = 0 slab)",
            xlim=(-half * 1e2, half * 1e2), ylim=(-half * 1e2, half * 1e2),
            aspect="equal")

    fig.suptitle(f"Step {step}  |  t = {t_ns:.2f} ns", fontsize=12,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    frame_path = os.path.join(frames_dir, f"frame_{step:04d}.png")
    fig.savefig(frame_path, dpi=100, bbox_inches="tight")
    plt.close(fig)

print(f"  Saved {n_frames} frames to {frames_dir}/")

# ---------------------------------------------------------------------------
# 2. Q(t) drift plot
# ---------------------------------------------------------------------------
times_ns = field_steps.astype(float) * dt_sim * 1e9

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors_q = {"close": "#2b6cb0", "mid": "#c07800", "far": "#228b22"}

for idx_s, (s_name, _, _, _, V) in enumerate(sphere_centers):
    ax = axes[idx_s]
    Q0 = q[f"{s_name}_close"][0]

    for hl in box_labels:
        Qs = q[f"{s_name}_{hl}"]
        drift_pct = (Qs - Q0) / abs(Q0) * 100
        ax.plot(times_ns, drift_pct, "-o", color=colors_q[hl],
                label=f"box: {hl}", markersize=3, linewidth=1.5)

    if s_name in eb_charge:
        eb = eb_charge[s_name]
        Q0_eb = eb["Q"][0]
        drift_eb_pct = (eb["Q"] - Q0_eb) / abs(Q0_eb) * 100
        ax.plot(eb["time"] * 1e9, drift_eb_pct, "-s", color="#d42054",
                label="EB surface", markersize=4, linewidth=2, zorder=5)

    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time [ns]")
    ax.set_ylabel("(Q - Q_0) / |Q_0|  [%]")
    ax.set_title(f"{s_name} sphere (V = {V:+.0f} V)")
    ax.legend(title="Measurement", fontsize=7, title_fontsize=8)
    ax.grid(True, alpha=0.3)

    t_exit = (half - z_beam_lo) / vz_drift * 1e9
    ax.axvspan(0, t_exit, color="green", alpha=0.06)

    if has_correction:
        t_corr = correction_step * dt_sim * 1e9
        ax.axvline(t_corr, color="#d42054", linewidth=1.5, linestyle="--",
                   zorder=6, label="correction")

    if has_periodic:
        for cs in correction_steps_arr:
            ax.axvline(cs * dt_sim * 1e9, color="#d42054", linewidth=0.3,
                       alpha=0.3, zorder=1)

fig.suptitle("Enclosed charge vs time — Gauss boxes + EB surface integral"
             "   (green = beam in domain)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
_save_fig(fig, "charge_drift")

# ---------------------------------------------------------------------------
# 2b. Voltage tracking plot (periodic correction only)
# ---------------------------------------------------------------------------
if has_periodic:
    corr_times_ns = correction_steps_arr * dt_sim * 1e9

    fig_v, (ax_vl, ax_vr) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, idx, name, V_t in [(ax_vl, 0, "left", V_left),
                                (ax_vr, 1, "right", V_right)]:
        ax.plot(corr_times_ns, V_before_history[:, idx], "-",
                color="#c07800", linewidth=1.0, alpha=0.7, label="before corr.")
        ax.plot(corr_times_ns, V_after_history[:, idx], "-",
                color="#2b6cb0", linewidth=1.0, alpha=0.7, label="after corr.")
        ax.axhline(V_t, color="#228b22", linewidth=1.5, linestyle="--",
                   label=f"target ({V_t:+.0f} V)")
        t_exit = (half - z_beam_lo) / vz_drift * 1e9
        ax.axvspan(0, t_exit, color="green", alpha=0.06)
        ax.set(xlabel="Time [ns]", ylabel="V_eff [V]",
               title=f"{name} electrode")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig_v.suptitle(f"Electrode voltages — correction every "
                   f"{correction_interval} steps", fontsize=11)
    fig_v.tight_layout(rect=[0, 0, 1, 0.94])
    _save_fig(fig_v, "voltage_tracking")

# ---------------------------------------------------------------------------
# 3. E-field difference (final - initial) — symlog colorscale
# ---------------------------------------------------------------------------
Ex_diff = field_data[-1] - field_data[0]
diff_peak = np.max(np.abs(Ex_diff))
diff_vmax = max(np.percentile(np.abs(Ex_diff), 99.5), 1.0)
diff_linthresh = diff_vmax * 0.01

fig_diff, ax_diff = plt.subplots(1, 1, figsize=(8, 6.5))
norm_diff = SymLogNorm(linthresh=diff_linthresh, vmin=-diff_vmax, vmax=diff_vmax)
im_diff = ax_diff.pcolormesh(z_n * 1e2, x_cc * 1e2, Ex_diff,
                             cmap="RdBu_r", norm=norm_diff,
                             shading="auto", rasterized=True)
fig_diff.colorbar(im_diff, ax=ax_diff, label=r"$\Delta E_x$ [V/m] (symlog)",
                  shrink=0.85)
_draw_spheres(ax_diff)
_draw_boxes(ax_diff)
step_i, step_f = int(field_steps[0]), int(field_steps[-1])
ax_diff.set(xlabel="z [cm]", ylabel="x [cm]",
            xlim=(-half * 1e2, half * 1e2), ylim=(-half * 1e2, half * 1e2),
            aspect="equal")
ax_diff.set_title(
    rf"$E_x$(step {step_f}) $-$ $E_x$(step {step_i})   "
    rf"(y = 0)   |   peak $|\Delta E_x|$ = {diff_peak:.2e} V/m",
    fontsize=10)
fig_diff.tight_layout()
_save_fig(fig_diff, "ex_difference")


# ---------------------------------------------------------------------------
# 4. Correction comparison panels (if correction data present)
# ---------------------------------------------------------------------------
def _corr_snapshot_fig(snap_items, suptitle, filename):
    """Render a grid of Ex snapshots and save to file."""
    n = len(snap_items)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows),
                             squeeze=False)
    axes_flat = axes.ravel()
    snap_vmax = max(np.percentile(np.abs(s[0]), 99.5) for s in snap_items)
    for ax, (ex, label, V_arr) in zip(axes_flat, snap_items):
        im = ax.pcolormesh(z_n * 1e2, x_cc * 1e2, ex,
                           cmap="RdBu_r", vmin=-snap_vmax, vmax=snap_vmax,
                           shading="auto", rasterized=True)
        fig.colorbar(im, ax=ax, label=r"$E_x$ [V/m]", shrink=0.85, pad=0.02)
        _draw_spheres(ax)
        _draw_boxes(ax)
        ax.set(xlabel="z [cm]", ylabel="x [cm]",
               xlim=(-half * 1e2, half * 1e2),
               ylim=(-half * 1e2, half * 1e2), aspect="equal")
        v_str = ", ".join(f"{v:+.1f}" for v in V_arr)
        ax.set_title(f"{label}\nV_eff = [{v_str}] V", fontsize=9)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    _save_fig(fig, filename)


def _corr_diff_fig(diffs, suptitle, filename):
    """Render difference maps with symlog and save to file."""
    n = len(diffs)
    diff_corr_vmax = max(
        max(np.percentile(np.abs(dd), 99.5) for dd, _ in diffs), 1e-10)
    norm = SymLogNorm(linthresh=diff_corr_vmax * 0.01,
                      vmin=-diff_corr_vmax, vmax=diff_corr_vmax)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]
    for ax, (dd, label) in zip(axes, diffs):
        im = ax.pcolormesh(z_n * 1e2, x_cc * 1e2, dd,
                           cmap="RdBu_r", norm=norm,
                           shading="auto", rasterized=True)
        fig.colorbar(im, ax=ax, label=r"$\Delta E_x$ [V/m] (symlog)",
                     shrink=0.85, pad=0.02)
        _draw_spheres(ax)
        _draw_boxes(ax)
        ax.set(xlabel="z [cm]", ylabel="x [cm]",
               xlim=(-half * 1e2, half * 1e2),
               ylim=(-half * 1e2, half * 1e2), aspect="equal")
        ax.set_title(f"{label}\npeak |dEx| = {np.max(np.abs(dd)):.2e} V/m",
                     fontsize=9)
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save_fig(fig, filename)


if has_correction and corr_format == "two_stage":
    _corr_snapshot_fig([
        (ex_pre_clean, "Before correction", V_pre_clean),
        (ex_post_clean, "After Gauss clean", V_post_clean),
        (ex_post_bias, "After harmonic bias", V_post_bias),
    ], f"E-field correction at step {correction_step}",
        "correction_snapshots")

    diffs_corr = [
        (ex_post_clean - ex_pre_clean, "Gauss clean effect"),
        (ex_post_bias - ex_post_clean, "Harmonic bias effect"),
    ]
    _corr_diff_fig(diffs_corr,
                   "What each correction step changed (symlog)",
                   "correction_diffs")

    print(f"  Two-stage correction: "
          f"V_pre=[{', '.join(f'{v:+.1f}' for v in V_pre_clean)}], "
          f"V_post=[{', '.join(f'{v:+.1f}' for v in V_post_bias)}], "
          f"target=[{V_left:+.0f}, {V_right:+.0f}]")

elif has_correction and corr_format == "three_variant":
    variant_labels = [
        ("full_bias", "Full bias\n(unmasked, no clean)"),
        ("clean_bias", "Gauss clean\n+ full bias"),
        ("masked_bias", "Masked bias\n(frozen cells preserved)"),
    ]

    _corr_snapshot_fig(
        [(ex_pre, "Before correction", V_pre)] +
        [(variants[k]["ex"], label, variants[k]["V"])
         for k, label in variant_labels],
        f"Correction comparison at step {correction_step}",
        "correction_snapshots")

    diffs_corr = [(variants[k]["ex"] - ex_pre, label)
                  for k, label in variant_labels]
    _corr_diff_fig(diffs_corr,
                   "Correction effect (variant - pre-correction, symlog)",
                   "correction_diffs")

    has_energy = W_pre is not None
    for k, label in variant_labels:
        V_arr = variants[k]["V"]
        err_l = abs(V_arr[0] - V_left)
        err_r = abs(V_arr[1] - V_right)
        line = (f"  {label.split(chr(10))[0]:30s}  "
                f"V=[{', '.join(f'{v:+.1f}' for v in V_arr)}]  "
                f"err=[{err_l:.1f}, {err_r:.1f}] V")
        if has_energy and "W" in variants[k]:
            dW = variants[k]["W"] - W_pre
            dW_pct = dW / W_pre * 100
            line += f"  dW_E={dW:+.4e} ({dW_pct:+.4f}%)"
        print(line)

# ---------------------------------------------------------------------------
# 5. Field energy timeline (if available)
# ---------------------------------------------------------------------------
if field_energy_data is not None:
    fe = field_energy_data
    W0 = fe["W_E"][0]
    dW_pct = (fe["W_E"] - W0) / W0 * 100

    fig_we, ax_we = plt.subplots(1, 1, figsize=(10, 4))
    ax_we.plot(fe["time"] * 1e9, dW_pct, "-", color="#2b6cb0",
               linewidth=1.5, label=r"$\Delta W_E / W_{E,0}$")
    if has_correction:
        t_corr = correction_step * dt_sim * 1e9
        ax_we.axvline(t_corr, color="#d42054", linewidth=1.5,
                      linestyle="--", label="correction")
    t_exit = (half - z_beam_lo) / vz_drift * 1e9
    ax_we.axvspan(0, t_exit, color="green", alpha=0.06)
    ax_we.set(xlabel="Time [ns]",
              ylabel=r"$(W_E - W_{E,0}) / W_{E,0}$ [%]",
              title="Electric field energy drift")
    ax_we.legend(fontsize=8)
    ax_we.grid(True, alpha=0.3)
    fig_we.tight_layout()
    _save_fig(fig_we, "field_energy")

# ---------------------------------------------------------------------------
# 6. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
for s_name, _, _, _, V in sphere_centers:
    Q0_val = q[f"{s_name}_close"][0]
    Q_final = q[f"{s_name}_close"][-1]
    drift = abs(Q_final - Q0_val) / abs(Q0_val) if Q0_val != 0 else 0
    print(f"  {s_name} (V={V:+.0f}V): Q_close drift = {drift:.2e}, "
          f"Q0 = {Q0_val:.4e} C")
    for hl in box_labels:
        Qs = q[f"{s_name}_{hl}"]
        peak_dev = np.max(np.abs(Qs - Q0_val))
        print(f"    {hl:>5s}: peak deviation = {peak_dev:.4e} C "
              f"({peak_dev/abs(Q0_val)*100:.2f}%)")
    if s_name in eb_charge:
        eb = eb_charge[s_name]
        Q0_eb = eb["Q"][0]
        Q_end_eb = eb["Q"][-1]
        drift_eb = abs(Q_end_eb - Q0_eb) / abs(Q0_eb) if Q0_eb != 0 else 0
        print(f"    EB surface: drift = {drift_eb:.2e}, "
              f"Q0 = {Q0_eb:.4e}, Q_end = {Q_end_eb:.4e}")

print(f"\nPeak |dEx| = {diff_peak:.2e} V/m")
print(f"Outputs in {os.path.abspath(data_dir)}/")
