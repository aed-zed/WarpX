.. _usage-electrode-voltage-clamp:

Holding embedded electrodes at a prescribed voltage
===================================================

WarpX's electromagnetic update evolves :math:`E` and :math:`B`, but it does not by
itself hold an embedded conductor at a prescribed potential. Over many steps the
effective electrode voltage therefore drifts as the plasma charges the surface.

:class:`pywarpx.multi_electrode_corrector.MultiElectrodeBiasCorrector` adds a
quasi-static correction for several driven electrodes without a Poisson solve per
step.

.. warning::

   These are research interfaces, not general validated conductor models.
   The existing geometric-EB/Yee actuator has known stencil-compatibility
   limitations. The opt-in RZ staircase construction below is a separate
   instrument; do not mix its basis, capacitance, observer or initialization
   with the existing corrector. No default has changed.

Geometry requirement
--------------------

.. important::

   **No conductor surface may coincide with a Dirichlet or PEC domain boundary
   at any point.** Every embedded conductor must have at least a layer of
   vacuum between it and such a boundary.

   The rule is about *conducting* walls. A boundary that is neither a conductor
   nor a charge sink is exempt: the RZ regularity axis at :math:`r = 0`
   (boundary type ``none``) is a symmetry condition, so a body may sit on it,
   and a periodic seam is not a wall at all. Verified in RZ with two spheres
   centred on the axis: the capacitance matrix stayed symmetric to 15 digits
   with a condition number of 1.13.

A conductor touching a conducting domain wall becomes part of that conductor.
It cannot retain an independently assigned, conflicting voltage. This changes
the connectivity and independent voltage unknowns; it does not inherently
remove a grounded reference. The clearance requirement here is an algorithmic
restriction of the current instruments, not a general electrostatic prohibition
on connecting metal to ground. Resolve the numerical stencil support as well
as the physical gap. Matrix symmetry alone cannot validate that geometry.

The domain boundary in the directions transverse to the electrodes should be
Dirichlet. A Dirichlet wall is a conductor that can sink charge, and that is
what supplies the reference the capacitance matrix needs. Neumann or periodic
walls pass no flux and hold no reference, and with those only potential
*differences* are recoverable, never absolute electrode potentials.

How it works
------------

At setup, one unit field :math:`E_{0k} = -\nabla\phi_{0k}` is precomputed per
electrode, with electrode :math:`k` at 1 V and all others grounded, together with
the vacuum capacitance matrix :math:`C_{jk} = \epsilon_0 \oint_j E_{0k}\cdot n\, dS`.

Each correction measures the induced charge :math:`Q_j` of the live field and the
plasma-induced charge :math:`Q_{g,j}` that the same plasma would induce with every
electrode grounded, inverts

.. math::

   V = C^{-1} (Q - Q_g),

and adds :math:`\sum_k \delta V_k E_{0k}` with
:math:`\delta V = \mathrm{relaxation}\,(V_\mathrm{target} - V)`. Because the unit
fields represent harmonic potentials, their intended effect is zero free-space
divergence and zero curl, with nonzero conductor charge reactions. These are
separate discrete requirements: a geometric-EB gradient or post-mask need not
be compatible with the native Maxwell divergence and curl. They must be tested,
not inferred merely from the use of a potential.

:math:`Q_g` is obtained either from a grounded Poisson solve (``qg_mode="grounded"``)
or, by default, from the discrete Shockley--Ramo pairing
:math:`Q_{g,k} = -\sum_a (V_D)_a\rho_a \Psi_k[a]`, which needs no solve; here
:math:`V_D` is the nodal deposition-volume measure. The weighting
potential :math:`\Psi_k` is built from the transpose of WarpX's own discrete EB
operator and charge functional; on a non-symmetric cut-cell operator the plain
unit-voltage basis is not the adjoint of that functional, and using it would
mis-book the charge.

Usage
-----

.. code-block:: python

   from pywarpx.callbacks import (
       installafterEsolve,
       installafterInitatRestart,
       installafterInitEsolve,
   )
   from pywarpx.multi_electrode_corrector import MultiElectrodeBiasCorrector

   corrector = MultiElectrodeBiasCorrector(
       sim=sim,
       correction_interval=10,
       electrodes=[
           {"name": "left", "region": "(x<0)", "potential": +300.0},
           {"name": "right", "region": "(x>0)", "potential": -700.0},
       ],
   )
   # afterInitEsolve does not run on a restart, so register both init hooks
   installafterInitEsolve(corrector.setup_after_init)
   installafterInitatRestart(corrector.setup_after_init)
   installafterEsolve(corrector.correct_field)

``pywarpx.multi_electrode_logger`` provides two optional diagnostics:
``MultiElectrodeClampTelemetry`` writes the per-correction voltages and charges to
a CSV without re-measuring anything, and ``GroundedChargeCrossCheck`` sparsely
compares the adjoint observer against an independent grounded Poisson solve.

Supported envelope
------------------

* 3D and RZ, with embedded boundaries enabled.
* Driven electrodes with prescribed numeric targets. General time-dependent
  circuit driving is not implemented by this constructor.
* Grounded (PEC) outer field boundaries on every wall; in RZ, regularity at
  :math:`r=0`. Other outer boundaries are rejected, because the weighting
  potential is only the adjoint of the charge functional for a grounded reference.
* In RZ, nodal CIC deposition (``particle_shape=1``) without a charge filter.

Floating electrodes, external circuit coupling and dielectric embedded boundaries
are not modelled.

Opt-in RZ Yee staircase research path
------------------------------------

:class:`pywarpx.staircase_bias_corrector.StaircaseBiasCorrector` uses the endpoints
of native frozen Er/Ez edges as fixed-potential conductor nodes. It solves a
harmonic nodal problem with AMReX's existing finite-difference operator, then
takes WarpX's ordinary gradient. This changes no frozen edge, has zero Yee curl,
and adds no free-node charge in the verified cases. Post-masking an arbitrary
gradient does not preserve all three properties.

The matched observer is

.. math::

   Q_{\mathrm{app},k}=\epsilon_0 s_k^T V_G D E-s_k^T q,\qquad
   Q_{g,k}=-\psi_k^T q,\qquad V=C^{-1}(Q_{\mathrm{app}}-Q_g),

where :math:`s_k` selects fixed nodes, :math:`q=V_D\rho` is deposited nodal
charge, and :math:`V_G` is the geometric Gauss volume. The grounded pairing
includes all nodes because apparent charge subtracts fixed-node live charge.
Adding the same collected charge again through a ledger changes this convention.
There is no runtime Poisson solve or SciPy dependency.

The research envelope is CPU, one level, fixed geometry, lab-frame explicit
vacuum-medium Yee, RZ mode zero, particle-only CIC and no charge filtering.
Use a grounded outer radius and grounded or periodic axial boundaries. Moving
windows, fluid species and internal field-zeroing mirrors are rejected. Free-axis
live charge is conservatively rejected with the default RZ deposition correction,
whose axis volume differs from the Gauss measure. ECT is a separate Cartesian
path using its EB-aware gradient, not this staircase construction.

Fresh initialization and correction order are different from the old class:

.. code-block:: python

   from pywarpx import callbacks
   from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

   sim.initialize_inputs()
   sim.initialize_warpx()  # finish all initialization with zero E
   corrector = StaircaseBiasCorrector(
       sim, correction_interval=1,
       electrodes=[
           {"name": "rod", "region": "(x<0.02)", "potential": -1.0e4},
           {"name": "sleeve", "region": "(x>=0.02)", "potential": 0.0},
       ],
   )
   corrector.setup_after_init()  # collective; preserves incoming E
   corrector.initialize_vacuum_bias()  # strict zero E and deposited rho
   callbacks.installafterstep(corrector.correct_after_step)
   sim.step()

These example selectors assume two separate rod/sleeve components on opposite
sides of 20 mm. Select whole frozen-edge components, including grounded embedded
conductors; no manually placed integration surface is required. In RZ selectors
use :math:`x=r,y=0,z=z`. Use one corrector per simulation.

For an initially paired neutral loading, an explicit absolute ``rho_tolerance``
in C/m^3 can allow documented cancellation roundoff. It does not initialize a
nonneutral plasma. Do not install a geometric-EB Poisson bias first, and do not
register initialization on ``afterInitEsolve``: external fields may be added
after that hook. Correction must run on ``afterstep``, after scraping and time
advancement, not ``afterEsolve``.

For a caller-confirmed compatible checkpoint, finish WarpX restart, rebuild the
basis, and call ``resume_from_checkpoint()`` instead of fresh bias initialization.
Register the callback once. A serial collecting-EB checkpoint test reproduces
the uninterrupted trajectory. Source telemetry starts a new segment: retain
the prior source and scraped-charge totals separately. Arbitrary old checkpoints
are not automatically certified as compatible.

``setup_state()``, ``last_correction_state()`` and ``measure_voltage_state()``
provide setup, source-charge and field/charge telemetry. The geometric-EB grounded
solve is not a cross-check for this different operator and is deliberately
unavailable on this class. Exact reported voltage is a same-observer consistency
check; independent field profiles and particle-buffer charge/work checks remain
necessary.

Two embedded periodic-coax conductors recover their independent discrete field
after collection and a fixed-support finite-emission pulse. An absorbing outer
domain wall instead leaves an unresolved last-edge field error. Cold particles
born in the represented zero-field shell may remain trapped: passing a finite
launch test is not proof of a physical surface-emission law. A native one-step,
non-depositing tracer test confirms zero gathered force in some of the physical
vacuum next to the represented surface. With linear particles, energy-conserving
Galerkin gathering uses order zero along the radial electric-field direction;
it samples those frozen values without a cut-surface correction. The tested
particle path uses the default charge-conserving deposition and this gather;
alternative gathering/deposition choices are not certified by these results.
Staircase geometry
and near-metal forces still require refinement. GPU, higher shapes, general
emission, floating circuits and dielectric charging remain outside this path.
