.. _usage-electrode-voltage-clamp:

Holding Embedded Electrodes at a Prescribed Voltage (RZ, Yee)
=============================================================

In an electromagnetic run, WarpX sets embedded-boundary (EB) potentials only in the
initial electrostatic solve. Afterwards nothing maintains them: charge that is emitted
from or collected by an electrode changes its potential, and a current-carrying diode
discharges within about one transit time. ``StaircaseBiasCorrector`` acts as an ideal
voltage source for one or more separate EB conductors. After every step it restores
each conductor to its target voltage without changing ``B`` or any frozen edge.

Method
------

WarpX's explicit Yee solver represents an EB conductor as a *staircase*: E edges whose
stencil touches a cut or covered cell are frozen. The clamp works on exactly this
representation.

* **Actuator.** At setup, one harmonic potential :math:`\psi_k` per conductor is solved
  on the staircase operator (unit potential on conductor *k*, zero on the others and on
  the PEC wall). Its field :math:`U_k = -\nabla_h \psi_k` uses the ordinary Yee
  gradient, so :math:`\nabla_h \times U_k = 0` and adding it changes neither ``B`` nor
  the frozen edges.
* **Observer.** The charge on conductor *k* is read from WarpX's own nodal
  divergence, summed over the conductor's fixed nodes :math:`s_k`, minus the
  deposited charge there: :math:`Q_k = \varepsilon_0\, s_k^T V\, \nabla_h\!\cdot E - s_k^T q`.
* **Plasma contribution.** The charge the plasma induces on the grounded conductors
  follows from the unit potentials (Shockley-Ramo):
  :math:`Q_{g,k} = -\psi_k^T q`. ``compare_grounded_charge()`` checks it against a real
  grounded solve on the same operator.
* **Feedback.** With the capacitance matrix :math:`C_{jk} = s_j^T K \psi_k`, the
  voltages are :math:`V = C^{-1}(Q - Q_g)` and the correction
  :math:`E \mathrel{+}= \sum_k U_k (V^*_k - V_k)` restores the targets exactly.

The identities are exact only if the deposited :math:`\rho` and :math:`J` satisfy the
discrete continuity equation. The clamp therefore requires Esirkepov current deposition
(or Villasenor with linear particles).

Supported configurations
------------------------

* RZ, azimuthal mode 0, one mesh level, explicit Yee solver, laboratory frame.
* Particle shapes 1 to 4 with Esirkepov deposition. No current/charge filter.
* Radial boundary: the axis and a PEC outer wall (the potential reference). Axial
  boundaries: periodic or PEC.
* EB conductors must be separate bodies and must not coincide with a domain boundary.
* If particles are absorbed at a PEC domain wall, set
  ``particles.crop_on_PEC_boundary = 1`` so that their current stops at the wall;
  otherwise the discrete Gauss law is violated next to the wall.
* Emitted particles must be created inside the conductor's fixed node shell (between
  the geometric surface and the represented surface). Particles created in free
  vacuum carry charge without a current, which the field cannot see.

Other configurations abort with a message.

Usage
-----

.. code-block:: python

   from pywarpx import callbacks
   from pywarpx.staircase_bias_corrector import StaircaseBiasCorrector

   sim.initialize_inputs()
   sim.initialize_warpx()

   clamp = StaircaseBiasCorrector(
       sim,
       correction_interval=1,
       electrodes=[
           {"name": "cathode", "region": "(x<0.03)", "potential": -1000.0},
           {"name": "anode", "region": "(x>0.03)", "potential": 0.0},
       ],
   )
   clamp.setup_after_init()        # unit potentials, capacitance matrix
   clamp.initialize_vacuum_bias()  # new run; on restart: clamp.resume_from_checkpoint()
   callbacks.installafterstep(clamp.correct_after_step)  # after particle scraping

   sim.step(n_steps)

A ``region`` is a parser expression in ``(x, z)`` (``x`` is the radius) that selects one
whole staircase component. ``measure_voltage_state()`` returns the observer voltages and charges.

What it does not do
-------------------

The clamp holds the voltages of the *represented* conductors. Next to the metal the
staircase leaves field-free strips in vacuum, so near-surface forces carry the usual
staircase error. The reported voltage is the clamp's own charge coordinate; an
independent check is the line integral of :math:`E_r` between the conductors. There is
no circuit model (no floating electrodes, no series impedance) and no dielectric
charging.

Examples and tests
------------------

``Examples/Tests/staircase_voltage_clamp`` contains the regression tests:

* unit-bias capacitance and voltage hold;
* the grounded-charge cross-check (one and two ranks);
* a coaxial space-charge diode with and without the clamp;
* checkpoint/restart during emission.
