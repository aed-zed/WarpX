.. _developers-diagnostics-internal:

Exposing Internal Computations for Diagnostics
===============================================

This guide shows how advanced users can expose internal intermediate calculations
(that are normally computed and discarded within kernels) for diagnostic output.

Overview
--------

WarpX computes many intermediate quantities during field solves that are not
normally saved. Examples include:

- Individual terms in Ohm's law (Hall term, pressure gradient, resistivity)
- Intermediate values in implicit solvers
- Terms in the PSATD solver
- Subcycling intermediate states

For debugging or analysis, you may want to output these quantities. This guide
shows the pattern for making them available to diagnostics.

The General Pattern
-------------------

**Step 1: Allocate a persistent MultiFab**

Allocate a named MultiFab from Python that will persist across timesteps and be
available for diagnostic output:

.. code-block:: python

   from pywarpx import callbacks
   
   @callbacks.installafterInitEsolve
   def allocate_diagnostic_fields():
       # Use an existing field as template
       Ex = sim.fields.get("Efield_fp", dir='x', level=0)
       
       # Allocate persistent diagnostic field
       hall_term = sim.fields.alloc_init(
           name="hall_term",
           dir='x',
           level=0,
           ba=Ex.box_array(),
           dm=Ex.dm(),
           ncomp=1,
           ngrow=Ex.n_grow_vect,
           initial_value=0.0,
           redistribute=True,
           redistribute_on_remake=True
       )
       
       # Add to diagnostic output
       sim.extension.warpx.add_field_to_diagnostic("diag1", "hall_term", lev=0)

**Step 2: Copy or compute values into the persistent MultiFab**

This can be done either from Python callbacks or by modifying C++ code.

**Python approach** (recompute in callback):

.. code-block:: python

   @callbacks.callfrombeforediagnostics
   def compute_diagnostic_fields():
       """Runs before diagnostic output - compute derived quantities"""
       import numpy as np

       # Get input fields
       Bx  = sim.fields.get("Bfield_fp", dir='x', level=0)
       Jx  = sim.fields.get("current_fp", dir='x', level=0)
       rho = sim.fields.get("rho_fp",              level=0)

       # Get diagnostic field to write to
       hall_term = sim.fields.get("hall_term", dir='x', level=0)

       # Efficient in-place computation using to_xp():
       # Returns a list of per-FAB device arrays (NumPy on CPU, CuPy on GPU).
       # No device-to-host copy is performed.
       for jx_fab, bx_fab, rho_fab, out_fab in zip(
               Jx.to_xp(copy=False), Bx.to_xp(copy=False),
               rho.to_xp(copy=False), hall_term.to_xp(copy=False)):
           # Each *_fab is a local array of shape (nx, ny, nz, ncomp)
           out_fab[..., 0] = (jx_fab[..., 0] * bx_fab[..., 0]) / rho_fab[..., 0]

       # Alternative: mf[...] = value uses global indexing (allgather over MPI
       # + device-to-host copy) -- simpler to write but significantly slower,
       # especially on GPU or with many MPI ranks.

**C++ approach** (copy during kernel execution):

Modify the relevant C++ kernel to optionally store intermediate values when
the diagnostic field exists. See the complete example below.

**Step 3: Output the field**

The field will automatically be included in diagnostic output since it was
added with ``add_field_to_diagnostic()``.

Complete Example: Hall Term in Ohm's Law
-----------------------------------------

This example shows how to expose the Hall term from the hybrid-PIC Ohm's law solver.

Option A: Pure Python Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Recompute the Hall term in a Python callback:

.. code-block:: python

   from pywarpx import picmi, callbacks
   import numpy as np
   
   sim = picmi.Simulation(
       max_steps=100,
       warpx_hybrid_pic_model=True,
   )
   
   # Create diagnostic
   diag = picmi.FieldDiagnostic(
       name="diag1",
       period=10,
       data_list=["Ex", "Ey", "Ez", "Bx", "By", "Bz"],
   )
   sim.add_diagnostic(diag)
   
   @callbacks.installafterInitEsolve
   def setup_hall_diagnostics():
       """Allocate fields for Hall term diagnostic"""
       Ex = sim.fields.get("Efield_fp", dir='x', level=0)
       
       # Allocate all three components
       for dir_str in ['x', 'y', 'z']:
           sim.fields.alloc_init(
               name="hall_term",
               dir=dir_str,
               level=0,
               ba=Ex.box_array(),
               dm=Ex.dm(),
               ncomp=1,
               ngrow=Ex.n_grow_vect,
               initial_value=0.0,
               redistribute=True,
               redistribute_on_remake=True
           )
       
       # Add to diagnostic output
       sim.extension.warpx.add_field_to_diagnostic("diag1", "hall_term", lev=0)
       print("Hall term diagnostics enabled")
   
   @callbacks.callfrombeforediagnostics
   def compute_hall_term():
       """Compute Hall term: (J - Ji) x B / (ne)

       Uses to_xp() to operate on per-FAB device arrays in-place,
       avoiding any device-to-host data transfer.
       """
       from scipy.constants import elementary_charge as q_e

       level = 0

       Bx  = sim.fields.get("Bfield_fp",               dir='x', level=level)
       By  = sim.fields.get("Bfield_fp",               dir='y', level=level)
       Bz  = sim.fields.get("Bfield_fp",               dir='z', level=level)
       Jx  = sim.fields.get("hybrid_current_fp_plasma", dir='x', level=level)
       Jy  = sim.fields.get("hybrid_current_fp_plasma", dir='y', level=level)
       Jz  = sim.fields.get("hybrid_current_fp_plasma", dir='z', level=level)
       Jix = sim.fields.get("current_fp",              dir='x', level=level)
       Jiy = sim.fields.get("current_fp",              dir='y', level=level)
       Jiz = sim.fields.get("current_fp",              dir='z', level=level)
       rho = sim.fields.get("rho_fp",                           level=level)

       hall_x = sim.fields.get("hall_term", dir='x', level=level)
       hall_y = sim.fields.get("hall_term", dir='y', level=level)
       hall_z = sim.fields.get("hall_term", dir='z', level=level)

       # to_xp() returns per-FAB views (NumPy on CPU, CuPy on GPU).
       # Iterating over FABs avoids device-to-host copies and MPI allgather.
       for i, (bx, by, bz,
               jx, jy, jz,
               jix, jiy, jiz,
               rho_f, hx, hy, hz) in enumerate(zip(
                   Bx.to_xp(copy=False),  By.to_xp(copy=False),  Bz.to_xp(copy=False),
                   Jx.to_xp(copy=False),  Jy.to_xp(copy=False),  Jz.to_xp(copy=False),
                   Jix.to_xp(copy=False), Jiy.to_xp(copy=False), Jiz.to_xp(copy=False),
                   rho.to_xp(copy=False),
                   hall_x.to_xp(copy=False), hall_y.to_xp(copy=False), hall_z.to_xp(copy=False))):

           # Each array has shape (nx+2*ng, ny+2*ng, nz+2*ng, ncomp)
           b0 = bx[...,0]; b1 = by[...,0]; b2 = bz[...,0]
           j0 = jx[...,0]; j1 = jy[...,0]; j2 = jz[...,0]
           i0 = jix[...,0]; i1 = jiy[...,0]; i2 = jiz[...,0]

           # Electron current: Je = J - Ji
           je0 = j0 - i0;  je1 = j1 - i1;  je2 = j2 - i2

           # Cross product: Je x B
           jxb0 = je1 * b2 - je2 * b1
           jxb1 = je2 * b0 - je0 * b2
           jxb2 = je0 * b1 - je1 * b0

           # Electron density (quasi-neutrality: ne = |rho| / q_e)
           xp = np  # replaced by cupy automatically via CuPy's __array_ufunc__
           n_floor = 1e6  # Adjust based on your simulation
           ne = xp.maximum(xp.abs(rho_f[...,0]) / q_e, n_floor)

           hx[...,0] = jxb0 / ne
           hy[...,0] = jxb1 / ne
           hz[...,0] = jxb2 / ne

   sim.step()

**Pros:** No C++ changes needed, flexible, easy to prototype

**Cons:** Recomputes values (slower), requires understanding of field locations

Option B: C++ Modifications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For better performance, modify the C++ kernel to store intermediate values
when a diagnostic field is registered.

**Step 1:** In ``HybridPICSolveE.cpp``, check if diagnostic field exists and store values:

.. code-block:: cpp

   // In Source/FieldSolver/FiniteDifferenceSolver/HybridPICSolveE.cpp
   
   void FiniteDifferenceSolver::HybridPICSolveECartesian (...) {
       // ... existing code ...
       
       // Check if diagnostic field was allocated from Python
       auto& warpx = WarpX::GetInstance();
       bool const store_hall_term = warpx.m_fields.has("hall_term", Direction::x, lev);
       
       Array4<Real> hall_x_diag, hall_y_diag, hall_z_diag;
       if (store_hall_term) {
           hall_x_diag = warpx.m_fields.get("hall_term", Direction::x, lev).array(mfi);
           hall_y_diag = warpx.m_fields.get("hall_term", Direction::y, lev).array(mfi);
           hall_z_diag = warpx.m_fields.get("hall_term", Direction::z, lev).array(mfi);
       }
       
       // In the kernel where Hall term is computed:
       amrex::ParallelFor(tex, tey, tez,
           [=] AMREX_GPU_DEVICE (int i, int j, int k) {
               // ... existing computation of enE (Hall term) ...
               auto const enE_x = (jey - jiy) * bz - (jez - jiz) * by;
               auto const hall_term_x = enE_x / ne;
               
               // Store for diagnostics if field was allocated
               if (store_hall_term) {
                   hall_x_diag(i, j, k) = hall_term_x;
               }
               
               // Continue with E-field update (existing code)
               Ex(i, j, k) = hall_term_x + /* other terms */;
           },
           // ... similar for y and z ...
       );
   }

**Step 2:** From Python, allocate the field as in Option A. The C++ code will
automatically populate it.

**Pros:** Minimal performance overhead, values computed only once

**Cons:** Requires C++ changes and rebuilding WarpX

When to Use Each Approach
--------------------------

**Python-only** (Option A) recommended for:

- Quick debugging and exploration
- Prototyping new diagnostics
- Infrequent output (diagnostic overhead is acceptable)
- When you cannot rebuild WarpX

**C++ modifications** (Option B) recommended for:

- Frequent diagnostic output
- Production runs where performance matters
- Fields computed deep in GPU kernels
- Values that are expensive to recompute

Additional Examples
-------------------

**Example: Copy an internal temporary field**

.. code-block:: python

   @callbacks.installafterInitEsolve  
   def setup_temp_field_diagnostic():
       """Copy a temporary internal field to persistent diagnostic field"""
       
       # Get internal temporary field as template
       temp = sim.fields.get("hybrid_rho_fp_temp", level=0)
       
       # Allocate persistent copy
       diag_copy = sim.fields.alloc_init(
           name="rho_temp_diagnostic",
           level=0,
           ba=temp.box_array(),
           dm=temp.dm(),
           ncomp=1,
           ngrow=temp.n_grow_vect,
           redistribute=True,
           redistribute_on_remake=True
       )
       
       sim.extension.warpx.add_field_to_diagnostic("diag1", "rho_temp_diagnostic")
   
   @callbacks.callfromafterstep
   def copy_temp_field():
       """Copy temporary field data to diagnostic field (device-to-device)"""
       import amrex.space3d as amr
       temp = sim.fields.get("hybrid_rho_fp_temp", level=0)
       diag = sim.fields.get("rho_temp_diagnostic", level=0)

       # amr.copy_mfab stays on the device -- no host copy, no MPI allgather.
       # Use this instead of diag[...] = temp[...] whenever both fields share
       # the same BoxArray and DistributionMapping.
       amr.copy_mfab(dst=diag, src=temp,
                     srccomp=0, dstcomp=0, numcomp=1,
                     nghost=amr.IntVect(0))

**Example: Compute multiple derived fields efficiently**

.. code-block:: python

   @callbacks.callfrombeforediagnostics
   def compute_all_diagnostics():
       """Compute multiple derived fields at once, staying on the device"""

       Ex = sim.fields.get("Efield_fp", dir='x', level=0)
       Ey = sim.fields.get("Efield_fp", dir='y', level=0)
       Ez = sim.fields.get("Efield_fp", dir='z', level=0)
       e_mag = sim.fields.get("E_magnitude", level=0)

       # to_xp() iterates per FAB -- no host copy, works on CPU and GPU.
       for ex_f, ey_f, ez_f, out_f in zip(
               Ex.to_xp(copy=False), Ey.to_xp(copy=False),
               Ez.to_xp(copy=False), e_mag.to_xp(copy=False)):
           xp = type(ex_f)  # numpy or cupy
           out_f[..., 0] = xp.sqrt(ex_f[...,0]**2 + ey_f[...,0]**2 + ez_f[...,0]**2)

       # Compute E parallel to B (requires B field too)
       # ... more derived quantities ...

Best Practices
--------------

1. **Naming convention:** Use descriptive names like ``hall_term``, ``pressure_grad``, ``E_magnitude``

2. **Performance:** Only compute diagnostics when needed:

   - Use ``callfrombeforediagnostics`` callback (not ``callfromafterstep``)
   - This ensures computation only happens when diagnostics are actually written
   - Use ``mf.to_xp(copy=False)`` to operate on per-FAB device arrays (NumPy
     on CPU, CuPy on GPU) without any device-to-host copy or MPI allgather.
     Reserve ``mf[...]`` (global indexing) for quick prototyping or
     post-processing scripts where performance is not critical.

3. **Memory:** Diagnostic fields consume memory - only allocate what you need

4. **Documentation:** Comment your code explaining what each diagnostic represents

5. **Validation:** Compare computed values with known solutions or conservation laws

See Also
--------

- :ref:`usage-python-extend` - Python field access and manipulation
- :ref:`developers-fields` - Internal field structure
- :ref:`developers-diagnostics` - Diagnostic system overview
