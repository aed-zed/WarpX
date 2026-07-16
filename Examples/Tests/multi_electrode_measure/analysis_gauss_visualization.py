#!/usr/bin/env python3
"""Generate an interactive HTML visualization from Gauss-box test output.

Reads gauss_data.npz (saved by inputs_3d_gauss_movie.py or
inputs_3d_gauss_impact.py) and the ChargeOnEB reduced diagnostic files,
then produces a single self-contained HTML page with:
  1. Animated Ex field + split density movie (slider + play button)
  2. Enclosed charge drift plot (Gauss boxes + EB surface integral)
  3. Static Ex difference panel (final - initial)

Usage:
    python analysis_gauss_visualization.py [data_dir]

    data_dir: directory containing gauss_data.npz and diags/ (default: cwd)

The HTML is written to <data_dir>/gauss_visualization.html.
"""

import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
import base64  # noqa: E402
from io import BytesIO  # noqa: E402

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


# ---------------------------------------------------------------------------
# 1. Build field + density frames
# ---------------------------------------------------------------------------
frame_pngs = []
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

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    frame_pngs.append(base64.b64encode(buf.getvalue()).decode())
    buf.close()

print(f"  Created {len(frame_pngs)} frames")

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

fig.suptitle("Enclosed charge vs time — Gauss boxes + EB surface integral"
             "   (green = beam in domain)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])

drift_buf = BytesIO()
fig.savefig(drift_buf, format="png", dpi=120, bbox_inches="tight")
plt.close(fig)
drift_b64 = base64.b64encode(drift_buf.getvalue()).decode()
drift_buf.close()

# ---------------------------------------------------------------------------
# 3. E-field difference (final - initial)
# ---------------------------------------------------------------------------
Ex_diff = field_data[-1] - field_data[0]
diff_vmax = np.percentile(np.abs(Ex_diff), 99.5)
if diff_vmax < 1e-10:
    diff_vmax = max(np.max(np.abs(Ex_diff)), 1.0)

fig_diff, ax_diff = plt.subplots(1, 1, figsize=(8, 6.5))
im_diff = ax_diff.pcolormesh(z_n * 1e2, x_cc * 1e2, Ex_diff,
                             cmap="RdBu_r", vmin=-diff_vmax, vmax=diff_vmax,
                             shading="auto", rasterized=True)
fig_diff.colorbar(im_diff, ax=ax_diff, label=r"$\Delta E_x$ [V/m]",
                  shrink=0.85)
_draw_spheres(ax_diff)
_draw_boxes(ax_diff)
step_i, step_f = int(field_steps[0]), int(field_steps[-1])
ax_diff.set(xlabel="z [cm]", ylabel="x [cm]",
            xlim=(-half * 1e2, half * 1e2), ylim=(-half * 1e2, half * 1e2),
            aspect="equal")
ax_diff.set_title(
    rf"$E_x$(step {step_f}) $-$ $E_x$(step {step_i})   "
    rf"(y = 0)   |   peak $|\Delta E_x|$ = {np.max(np.abs(Ex_diff)):.2e} V/m",
    fontsize=10)
fig_diff.tight_layout()

diff_buf = BytesIO()
fig_diff.savefig(diff_buf, format="png", dpi=120, bbox_inches="tight")
plt.close(fig_diff)
diff_b64 = base64.b64encode(diff_buf.getvalue()).decode()
diff_buf.close()

# ---------------------------------------------------------------------------
# 4. Write HTML
# ---------------------------------------------------------------------------
frames_js = ",\n".join(f'"{b}"' for b in frame_pngs)
steps_list = [int(s) for s in field_steps]

html = f"""<title>{title}</title>
<style>
  :root {{
    --bg: #f5f5f0; --fg: #1a1a1a; --card-bg: #fff; --border: #d0d0c8;
    --label: #555; --accent: #2b6cb0;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #111318; --fg: #c8cad0; --card-bg: #1a1d26; --border: #2e3140;
      --label: #8890a0; --accent: #5ba3d9;
    }}
  }}
  :root[data-theme="dark"] {{
    --bg: #111318; --fg: #c8cad0; --card-bg: #1a1d26; --border: #2e3140;
    --label: #8890a0; --accent: #5ba3d9;
  }}
  :root[data-theme="light"] {{
    --bg: #f5f5f0; --fg: #1a1a1a; --card-bg: #fff; --border: #d0d0c8;
    --label: #555; --accent: #2b6cb0;
  }}
  body {{
    background: var(--bg); color: var(--fg);
    font-family: -apple-system, 'Segoe UI', system-ui, sans-serif;
    max-width: 960px; margin: 0 auto; padding: 1.5rem 1rem 3rem;
    line-height: 1.55;
  }}
  h1 {{ font-size: 1.35rem; margin: 0 0 0.3rem; letter-spacing: -0.01em; }}
  h2 {{ font-size: 1.05rem; margin: 1.8rem 0 0.6rem; color: var(--label);
        text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }}
  .subtitle {{ color: var(--label); font-size: 0.85rem; margin-bottom: 1.2rem; }}
  .panel {{
    background: var(--card-bg); border: 1px solid var(--border);
    padding: 0.8rem; margin: 0.6rem 0;
  }}
  img {{ max-width: 100%; height: auto; display: block; }}
  .controls {{
    display: flex; align-items: center; gap: 0.8rem; flex-wrap: wrap;
    margin-bottom: 0.5rem;
  }}
  .controls button {{
    padding: 0.35rem 0.9rem; cursor: pointer;
    border: 1px solid var(--border); background: var(--card-bg); color: var(--fg);
    font-family: inherit; font-size: 0.85rem;
  }}
  .controls button:hover {{ border-color: var(--accent); }}
  .controls button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 1px; }}
  input[type=range] {{ flex: 1; min-width: 180px; }}
  .frame-label {{
    font-variant-numeric: tabular-nums; font-size: 0.85rem;
    color: var(--label); min-width: 14ch;
    font-family: 'SF Mono', 'Cascadia Code', 'JetBrains Mono', monospace;
  }}
  .setup {{ font-size: 0.82rem; color: var(--label); }}
  .setup strong {{ color: var(--fg); }}
  .legend-row {{
    display: flex; gap: 1.2rem; flex-wrap: wrap;
    font-size: 0.78rem; color: var(--label); margin-top: 0.4rem;
  }}
  .legend-swatch {{
    display: inline-block; width: 18px; height: 3px;
    vertical-align: middle; margin-right: 4px;
  }}
</style>

<h1>{title}</h1>
<p class="subtitle">{subtitle}</p>

<div class="setup panel">
  <strong>Geometry:</strong> V<sub>left</sub>&nbsp;=&nbsp;{V_left:+.0f}&thinsp;V,
  V<sub>right</sub>&nbsp;=&nbsp;{V_right:+.0f}&thinsp;V,
  R&nbsp;=&nbsp;{R*1e3:.0f}&thinsp;mm, {cells_per_R}&thinsp;cells/R.
  <strong>Beam:</strong> {beam_desc}.
  <strong>BCs:</strong> PEC (grounded) on all faces.
  <br>
  <strong>Boxes</strong> are closed rectangular Gaussian surfaces around each
  sphere.
  <div class="legend-row">
    <span><span class="legend-swatch" style="background:#2b6cb0"></span>close ({h_boxes[0]*1e3:.0f}&thinsp;mm)</span>
    <span><span class="legend-swatch" style="background:#c07800;border-top:1px dashed #c07800"></span>mid ({h_boxes[1]*1e3:.0f}&thinsp;mm)</span>
    <span><span class="legend-swatch" style="background:#228b22;border-top:2px dashed #228b22"></span>far ({h_boxes[2]*1e3:.0f}&thinsp;mm)</span>
    &mdash; half-widths from sphere center.
  </div>
</div>

<h2>Field + Density</h2>
<div class="panel">
  <div class="controls">
    <button id="play-btn" onclick="togglePlay()">&#9654; Play</button>
    <input type="range" id="slider" min="0" max="{n_frames - 1}" value="0"
           oninput="showFrame(this.value)">
    <span class="frame-label" id="frame-label">Step 0</span>
  </div>
  <img id="field-img" src="" alt="Field + density frame">
</div>

<h2>Enclosed Charge Drift</h2>
<div class="panel">
  <img src="data:image/png;base64,{drift_b64}" alt="Q(t) drift plot">
</div>

<h2>E-field Difference (final &minus; initial)</h2>
<div class="panel">
  <img src="data:image/png;base64,{diff_b64}" alt="Ex difference plot">
  <p style="font-size:0.82rem;color:var(--label);margin:0.5rem 0 0;">
    Shows E<sub>x</sub>(step {step_f}) &minus; E<sub>x</sub>(step {step_i}) in
    the y&thinsp;=&thinsp;0 slice. Any permanent drift in the electrode field
    would appear as a residual pattern near the spheres.
  </p>
</div>

<script>
const frames = [{frames_js}];
const steps = {steps_list};
const dt_ns = {dt_sim * 1e9};
let playing = false, timer = null, idx = 0;

function showFrame(i) {{
  idx = parseInt(i);
  document.getElementById("field-img").src = "data:image/png;base64," + frames[idx];
  document.getElementById("slider").value = idx;
  const t = (steps[idx] * dt_ns).toFixed(2);
  document.getElementById("frame-label").textContent =
    "Step " + steps[idx] + " \\u00b7 t = " + t + " ns";
}}

function togglePlay() {{
  playing = !playing;
  document.getElementById("play-btn").textContent = playing ? "\\u23F8 Pause" : "\\u25B6 Play";
  if (playing) {{
    timer = setInterval(() => {{
      idx = (idx + 1) % frames.length;
      showFrame(idx);
    }}, 300);
  }} else clearInterval(timer);
}}

showFrame(0);
</script>
"""

html_path = os.path.join(data_dir, "gauss_visualization.html")
with open(html_path, "w") as f:
    f.write(html)
print(f"  Wrote {html_path}")

# ---------------------------------------------------------------------------
# 5. Summary
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

print(f"\nPeak |dEx| = {np.max(np.abs(Ex_diff)):.2e} V/m")
print(f"Output: {html_path}")
