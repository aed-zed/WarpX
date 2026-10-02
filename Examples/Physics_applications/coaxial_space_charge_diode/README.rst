.. _examples-coaxial-space-charge-diode:

Coaxial Space-Charge Diode at the Langmuir-Blodgett Limit
===========================================================

A coaxial space-charge-limited diode: an inner rod cathode at a negative voltage,
surrounded by a grounded sleeve anode, inside a grounded outer wall. Electrons are
emitted radially from the cathode at the current that exactly balances the space charge
in the gap, reproducing the Langmuir-Blodgett law.

An explicit electromagnetic PIC solver has no mechanism of its own to hold an electrode at
a fixed potential: once electrons leave the cathode, nothing replaces their charge, and the
gap voltage collapses within about one transit time of the emitted electrons. This example
uses the staircase voltage clamp (``StaircaseBiasCorrector``, see
:ref:`usage-electrode-voltage-clamp`). After every step it reads each electrode's charge
from WarpX's own discrete divergence on the conductor's fixed nodes and adds curl-free
unit-bias fields that restore the target potentials, acting as an ideal voltage source.
The clamped run therefore reaches a steady Langmuir-Blodgett profile.

Langmuir-Blodgett Law
----------------------

For concentric cylinders of radii :math:`r_c < r_a` with electron current magnitude
:math:`I` per length :math:`L`, the space-charge-limited current and potential profile are
:cite:t:`ex-LangmuirBlodgett1923,ex-Zhang2017`

.. math::
   \frac{I}{L} = \frac{8\pi\varepsilon_0}{9}\sqrt{\frac{2|q|}{m}}\,
                 \frac{|V|^{3/2}}{r\,\beta^2(r)},\quad
   \phi(r) = V\left(\frac{r\,\beta^2(r)}{r_a\,\beta^2(r_a)}\right)^{2/3},

where :math:`\beta(\gamma)`, :math:`\gamma=\ln(r/r_c)`, solves
:math:`3\beta\beta''+\beta'^2+4\beta\beta'+\beta^2=1` with :math:`\beta\sim\gamma` as
:math:`\gamma\to 0`. The injected current uses the *represented* gap: the cathode and
anode radii WarpX's embedded-boundary staircase actually exposes to the field solve (read
back from the frozen-edge mask), not the geometric rod and sleeve radii.

The law assumes electrons start at rest; here they are born with a small radial speed
(about 2.8 eV) to clear the cathode's fixed-charge shell, so the analysis compares against
a matched cold-beam oracle (same ODE, finite relativistic initial speed, found by shooting
on the cathode field) instead -- the finite birth speed shifts the profile by about 0.37%
relative to the zero-velocity curve. Measured convergence of the clamped profile against
that oracle is about 0.065% / 0.020% / 0.006% max relative error at 48 / 96 / 192 radial
cells.

Run with ``python3 inputs_test_rz_coaxial_space_charge_diode_picmi.py --nr 48 --transits 6
--arm on`` (``--arm off`` disables the clamp and lets the gap discharge):

.. literalinclude:: inputs_test_rz_coaxial_space_charge_diode_picmi.py
   :language: python3
   :caption: You can copy this file from ``Examples/Physics_applications/coaxial_space_charge_diode/inputs_test_rz_coaxial_space_charge_diode_picmi.py``.

The analysis script compares the time-averaged gap potential profile against the matched
oracle and Langmuir-Blodgett law, and checks the collected anode current and gap voltage:

.. literalinclude:: analysis_coaxial_space_charge_diode.py
   :language: python3
   :caption: You can copy this file from ``Examples/Physics_applications/coaxial_space_charge_diode/analysis_coaxial_space_charge_diode.py``.
