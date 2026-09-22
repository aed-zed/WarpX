#!/usr/bin/env python3
"""ECT tensor-grid volume diagnostic on a sphere separated from a grounded box.

``ComputeDivE`` for ECT is deliberately the Cartesian Yee divergence, not a
conformal cut-cell Gauss operator.  Its useful invariant is an integral over a
region whose boundary crosses only full-length edges that ECT actively updates.
This test encloses a spherical EB in such a regular cube and checks three
independent consequences:

1. corrector setup restores the live E field exactly;
2. the native volume integral of a unit field equals a direct six-face flux;
3. for an off-grid point charge, the volume adjoint reproduces a fresh grounded
   Poisson solve with a nonzero signal.

The mesh is split into several boxes even on one rank.  The two-rank CTest entry
also exercises ownership of shared nodal points and guard-cell filling.
"""

import numpy as np

from pywarpx import picmi
from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector

ncells = 16
half = 0.5
dx = 2.0 * half / ncells
radius = 0.19
volume_half = 5.0 * dx
volume = f"(abs(x)<={volume_half})*(abs(y)<={volume_half})*(abs(z)<={volume_half})"

assert radius + 1.9 * dx < volume_half
assert volume_half + dx < half

grid = picmi.Cartesian3DGrid(
    number_of_cells=[ncells] * 3,
    lower_bound=[-half] * 3,
    upper_bound=[half] * 3,
    lower_boundary_conditions=["dirichlet"] * 3,
    upper_boundary_conditions=["dirichlet"] * 3,
    lower_boundary_conditions_particles=["absorbing"] * 3,
    upper_boundary_conditions_particles=["absorbing"] * 3,
    warpx_blocking_factor=8,
    warpx_max_grid_size=8,
)
solver = picmi.ElectromagneticSolver(
    grid=grid,
    method="ECT",
    cfl=0.9,
    divE_cleaning=False,
)
eb = picmi.EmbeddedBoundary(
    implicit_function="-(x*x+y*y+z*z-radius*radius)",
    potential=0.0,
    radius=radius,
)

# The complete CIC support is outside both the enclosing volume and the EB,
# while remaining one cell away from the grounded box.  A non-grid-aligned
# position prevents a nodal special case from hiding a transpose error.
position = np.array([0.4013, 0.0237, -0.0411])
assert position[0] - dx > volume_half
assert np.linalg.norm(position) - np.sqrt(3.0) * dx > radius
assert np.all(np.abs(position) + dx < half)
distribution = picmi.ParticleListDistribution(
    x=[position[0]],
    y=[position[1]],
    z=[position[2]],
    ux=[0.0],
    uy=[0.0],
    uz=[0.0],
    weight=[1.0e9],
)
electrons = picmi.Species(
    name="electrons",
    particle_type="electron",
    initial_distribution=distribution,
)

sim = picmi.Simulation(
    solver=solver,
    time_step_size=1.0e-12,
    max_steps=0,
    particle_shape="linear",
    warpx_embedded_boundary=eb,
    warpx_use_filter=False,
    verbose=0,
)
sim.add_species(
    electrons,
    layout=picmi.GriddedLayout(n_macroparticle_per_cell=[0, 0, 0], grid=grid),
)

corrector = MultiElectrodeBiasCorrector(
    sim=sim,
    correction_interval=1,
    electrodes=[
        {
            "name": "sphere",
            "region": "1",
            "volume": volume,
            "potential": 1.0,
        }
    ],
    observer="volume",
    qg_mode="reciprocity",
    actuator_gradient="eb_aware",
    adjoint_tolerance=1.0e-11,
    adjoint_max_iterations=300,
    verbose=True,
)

sim.initialize_inputs()
sim.initialize_warpx()
warpx = corrector._warpx()
mfr = corrector._mfr()
Direction = corrector._Direction

assert warpx.boxArray(0).size > 1, "the fixture must contain multiple grid boxes"


def max_difference(lhs, rhs):
    """Global max norm of the valid-cell difference of two MultiFabs."""
    difference = lhs.copy()
    difference.saxpy(-1.0, rhs, 0, 0, 1, 0)
    return difference.norm0(0, 0, False, False)


# Nonzero sentinels make state restoration discriminate against accidentally
# clearing the fields.  These arrays are never evolved; they are a software
# state-preservation fixture, not a physical PEC initial condition.
for comp in (0, 1, 2):
    live = mfr.get("Efield_fp", dir=Direction(comp), level=0)
    live.set_val(0.017 * (comp + 1))
    assert live.norm0(0, 0, False, False) > 0.0
initial_e = corrector._save_efield(0)
corrector.setup_after_init()

# Setup performs grounded and biased Poisson solves.  Neither is allowed to
# replace the live evolution state.
for comp in (0, 1, 2):
    live = mfr.get("Efield_fp", dir=Direction(comp), level=0)
    error = max_difference(live, initial_e[comp])
    assert error == 0.0, f"setup changed Efield_fp[{comp}] by {error:.16e} V/m"

setup = corrector.setup_state()
assert setup["observer"] == "volume"
assert setup["qg_mode"] == "reciprocity"
assert setup["actuator_gradient"] == "eb_aware"
capacitance = float(np.asarray(setup["capacitance_matrix"])[0, 0])
assert capacitance > 1.0e-13, (
    f"the unit-potential sphere produced negligible capacitance: {capacitance:.16e} F"
)

# The parser selects a contiguous cube of nodal control volumes.  Summing
# DownwardDx over i0..i1 telescopes to Ex[i1]-Ex[i0-1], and cyclically for y,z.
nodes = -half + dx * np.arange(ncells + 1)
selected = np.flatnonzero(np.abs(nodes) <= volume_half + 8.0 * np.finfo(float).eps)
i0, i1 = int(selected[0]), int(selected[-1])
assert np.array_equal(selected, np.arange(i0, i1 + 1))
assert (i0, i1) == (3, 13)


def field_array(name, comp):
    """All-gather one valid field component and remove a singleton component."""
    field = mfr.get(name, dir=Direction(comp), level=0)
    array = np.asarray(field[:, :, :])
    if array.ndim == 4:
        assert array.shape[-1] == 1
        array = array[..., 0]
    assert array.ndim == 3
    return array


def boundary_slices(array, comp):
    """Values on the low and high edge layers crossing the selected nodes."""
    transverse = slice(i0, i1 + 1)
    if comp == 0:
        return array[i0 - 1, transverse, transverse], array[i1, transverse, transverse]
    if comp == 1:
        return array[transverse, i0 - 1, transverse], array[transverse, i1, transverse]
    return array[transverse, transverse, i0 - 1], array[transverse, transverse, i1]


# Verify the hypothesis that makes this ECT diagnostic useful.  This is checked
# from WarpX's native geometry and update masks, rather than inferred from the
# analytic sphere radius.
for comp in (0, 1, 2):
    lengths = np.asarray(mfr.get("edge_lengths", dir=Direction(comp), level=0)[:, :, :])
    update = np.asarray(warpx.eb_update_e_flag(lev=0, dir=comp)[:, :, :])
    if lengths.ndim == 4:
        lengths = lengths[..., 0]
    if update.ndim == 4:
        update = update[..., 0]
    for side, layer in zip(("low", "high"), boundary_slices(lengths, comp)):
        np.testing.assert_allclose(
            layer,
            dx,
            rtol=0.0,
            atol=32.0 * np.finfo(float).eps * dx,
            err_msg=f"component {comp} {side} volume boundary contains cut edges",
        )
    for side, layer in zip(("low", "high"), boundary_slices(update, comp)):
        assert np.all(layer == 1), (
            f"component {comp} {side} volume boundary contains frozen ECT edges"
        )

# Put the stored unit actuator in the live field, because the native divergence
# API intentionally operates on Efield_fp.  The direct flux below reads E
# itself and does not call ComputeDivE or the volume integration implementation.
saved_e = corrector._save_efield(0)
try:
    for comp in (0, 1, 2):
        live = mfr.get("Efield_fp", dir=Direction(comp), level=0)
        unit = mfr.get(corrector._unit_names[0], dir=Direction(comp), level=0)
        live.copymf(unit, 0, 0, 1, 0)

    q_native = float(warpx.div_e_charge_in_regions([volume], 0)[0])
    ex, ey, ez = (field_array("Efield_fp", comp) for comp in (0, 1, 2))
    ex_lo, ex_hi = boundary_slices(ex, 0)
    ey_lo, ey_hi = boundary_slices(ey, 1)
    ez_lo, ez_hi = boundary_slices(ez, 2)
    q_faces = (
        picmi.constants.ep0
        * dx**2
        * (
            np.sum(ex_hi)
            - np.sum(ex_lo)
            + np.sum(ey_hi)
            - np.sum(ey_lo)
            + np.sum(ez_hi)
            - np.sum(ez_lo)
        )
    )
finally:
    corrector._restore_efield(saved_e, 0)

assert abs(q_native) > 1.0e-13, (
    f"the native ECT volume diagnostic produced a negligible signal: {q_native:.16e} C"
)
face_error = abs(q_native - q_faces) / abs(q_native)
calibration_error = abs(q_native - capacitance) / abs(capacitance)
print(
    "ECT unit-field volume charge: "
    f"native={q_native:+.16e} C, faces={q_faces:+.16e} C, "
    f"C={capacitance:+.16e} F, face_error={face_error:.3e}, "
    f"calibration_error={calibration_error:.3e}"
)
assert face_error < 1.0e-12, (
    f"native volume charge does not telescope to the six regular faces: {face_error:.3e}"
)
assert calibration_error < 1.0e-12, (
    f"volume calibration does not equal its unit-field response: {calibration_error:.3e}"
)

# This comparison is independent of the capacitance calibration.  One path
# solves Poisson with the physical point charge and measures the resulting
# regular-box flux; the other pairs freshly deposited rho with the volume
# functional's adjoint weighting potential.
comparison = corrector.compare_grounded_charge()
q_reciprocity = float(comparison["reciprocity_charge"][0])
q_solved = float(comparison["solved_charge"][0])
grounded_error = float(comparison["relative_difference"][0])
print(
    "ECT volume-adjoint grounded charge: "
    f"reciprocity={q_reciprocity:+.16e} C, solved={q_solved:+.16e} C, "
    f"relative_error={grounded_error:.3e}"
)
assert max(abs(q_reciprocity), abs(q_solved)) > 1.0e-13, (
    "the point charge induced too little grounded-sphere charge to test the adjoint"
)
assert np.signbit(q_reciprocity) == np.signbit(q_solved), (
    f"grounded charge signs disagree: {q_reciprocity:+.16e}, {q_solved:+.16e} C"
)
assert grounded_error < 1.0e-8, (
    f"ECT volume functional and grounded solve differ by {grounded_error:.3e}"
)

# Both diagnostic probes above promise to preserve the live state as well.
for comp in (0, 1, 2):
    live = mfr.get("Efield_fp", dir=Direction(comp), level=0)
    error = max_difference(live, initial_e[comp])
    assert error == 0.0, f"diagnostics changed Efield_fp[{comp}] by {error:.16e} V/m"
