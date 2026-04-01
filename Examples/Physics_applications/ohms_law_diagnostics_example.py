#!/usr/bin/env python3
"""
Example: Computing and outputting Ohm's law terms for hybrid-PIC simulations

This script demonstrates how to compute individual terms from Ohm's law:
  E = (J - Ji) x B / (ne) - grad(Pe)/(ne) + eta*J - eta_h*nabla^2(J)
    = [Hall term] + [Pressure term] + [Resistivity] + [Hyper-resistivity]

And add them to diagnostic output using the new dynamic field registration.
"""

from pywarpx import picmi, callbacks
import numpy as np

# ============================================================================
# Simulation setup (simplified - adjust for your actual simulation)
# ============================================================================

sim = picmi.Simulation(
    max_steps=100,
    warpx_hybrid_pic_model=True,  # Enable hybrid-PIC
    # ... other parameters ...
)

# Create field diagnostic that will include Ohm's law terms
diag = picmi.FieldDiagnostic(
    name="diag1",
    period=10,
    data_list=["Bx", "By", "Bz", "Ex", "Ey", "Ez", 
               "Jx", "Jy", "Jz", "rho", "electron_pressure"],
)
sim.add_diagnostic(diag)

# ============================================================================
# Helper function to compute Ohm's law terms
# ============================================================================

def compute_ohm_law_terms():
    """
    Compute and store individual Ohm's law terms.

    This function:
    1. Gets the required fields (B, J, Ji, rho, Pe)
    2. Computes the Hall term: (J - Ji) x B / (ne)
    3. Computes the pressure gradient term: -grad(Pe) / (ne)
    4. (Optional) Resistivity and hyper-resistivity terms
    5. Stores results in custom MultiFabs for diagnostic output

    The computation iterates over per-FAB device arrays obtained via
    ``to_xp()``.  On CPU this returns NumPy views; on GPU it returns CuPy
    views -- in both cases the data stays on the device and no host copy
    is performed.  The result is written back in-place into the diagnostic
    MultiFabs without any device-to-host round-trip.

    Note: This is a simplified example.  In a production run you would need
    to interpolate the Yee-staggered fields to a common nodal staggering
    before computing cross-products.
    """

    level = 0
    from scipy.constants import elementary_charge as q_e

    # Retrieve MultiFab handles (no data transfer yet)
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
    Pe  = sim.fields.get("hybrid_electron_pressure_fp",      level=level)

    hall_x = sim.fields.get("hall_term",     dir='x', level=level)
    hall_y = sim.fields.get("hall_term",     dir='y', level=level)
    hall_z = sim.fields.get("hall_term",     dir='z', level=level)
    pres_x = sim.fields.get("pressure_term", dir='x', level=level)
    pres_y = sim.fields.get("pressure_term", dir='y', level=level)
    pres_z = sim.fields.get("pressure_term", dir='z', level=level)

    # -------------------------------------------------------------------------
    # to_xp() returns a list of per-FAB device arrays (NumPy on CPU, CuPy on
    # GPU) -- one entry per local AMReX Box.  Using copy=False gives a
    # zero-copy view directly into the MultiFab memory.
    # Shape of each array: (nx+2*ng, ny+2*ng, nz+2*ng, ncomp) in Fortran order.
    # -------------------------------------------------------------------------
    bx_fabs  = Bx.to_xp(copy=False);  by_fabs  = By.to_xp(copy=False)
    bz_fabs  = Bz.to_xp(copy=False)
    jx_fabs  = Jx.to_xp(copy=False);  jy_fabs  = Jy.to_xp(copy=False)
    jz_fabs  = Jz.to_xp(copy=False)
    jix_fabs = Jix.to_xp(copy=False); jiy_fabs = Jiy.to_xp(copy=False)
    jiz_fabs = Jiz.to_xp(copy=False)
    rho_fabs = rho.to_xp(copy=False)
    pe_fabs  = Pe.to_xp(copy=False)

    hx_fabs = hall_x.to_xp(copy=False); hy_fabs = hall_y.to_xp(copy=False)
    hz_fabs = hall_z.to_xp(copy=False)
    px_fabs = pres_x.to_xp(copy=False); py_fabs = pres_y.to_xp(copy=False)
    pz_fabs = pres_z.to_xp(copy=False)

    # Iterate over local FABs -- no MPI communication needed
    for i in range(len(hx_fabs)):
        bx = bx_fabs[i][..., 0]; by = by_fabs[i][..., 0]; bz = bz_fabs[i][..., 0]
        jx = jx_fabs[i][..., 0]; jy = jy_fabs[i][..., 0]; jz = jz_fabs[i][..., 0]
        jix = jix_fabs[i][..., 0]; jiy = jiy_fabs[i][..., 0]; jiz = jiz_fabs[i][..., 0]
        rho_arr = rho_fabs[i][..., 0]
        pe_arr  = pe_fabs[i][..., 0]

        # Use the array module of whatever device we are on (numpy or cupy)
        xp = type(bx)
        if hasattr(xp, 'get_array_module'):
            xp = xp.get_array_module(bx)
        else:
            xp = np

        # =====================================================================
        # Hall term: (J - Ji) x B / (ne)
        # =====================================================================
        jex = jx - jix;  jey = jy - jiy;  jez = jz - jiz

        je_x_b_x = jey * bz - jez * by
        je_x_b_y = jez * bx - jex * bz
        je_x_b_z = jex * by - jey * bx

        n_floor = 1e6  # Adjust based on your simulation
        ne = xp.maximum(xp.abs(rho_arr) / q_e, n_floor)

        # Write results in-place into the diagnostic MultiFab FABs
        hx_fabs[i][..., 0] = je_x_b_x / ne
        hy_fabs[i][..., 0] = je_x_b_y / ne
        hz_fabs[i][..., 0] = je_x_b_z / ne

        # =====================================================================
        # Pressure gradient term: -grad(Pe) / (ne)
        # Simplified 2nd-order central differences -- production runs should
        # use the same stencil WarpX applies internally.
        # =====================================================================
        grad_pe_x = xp.gradient(pe_arr, axis=0)
        grad_pe_y = xp.gradient(pe_arr, axis=1)
        grad_pe_z = xp.gradient(pe_arr, axis=2)

        px_fabs[i][..., 0] = -grad_pe_x / ne
        py_fabs[i][..., 0] = -grad_pe_y / ne
        pz_fabs[i][..., 0] = -grad_pe_z / ne

    # =========================================================================
    # Optional: Compute resistivity term (eta * J) and hyper-resistivity
    # =========================================================================
    # These require access to the resistivity parameters from HybridPICModel
    # and additional field operations (Laplacian for hyper-resistivity).

    print(f"Computed Ohm's law terms at step {sim.extension.warpx.getistep(0)}")


# ============================================================================
# Callback setup
# ============================================================================

@callbacks.installafterInitEsolve
def setup_ohm_law_fields():
    """
    Allocate custom MultiFabs for storing Ohm's law terms.
    These will be added to the diagnostic output.
    """
    level = 0
    
    # Use E-field as template for staggering and grid properties
    Ex = sim.fields.get("Efield_fp", dir='x', level=level)
    
    # Allocate Hall term components
    for dir_str in ['x', 'y', 'z']:
        hall = sim.fields.alloc_init(
            name="hall_term",
            dir=dir_str,
            level=level,
            ba=Ex.box_array(),
            dm=Ex.dm(),
            ncomp=1,
            ngrow=Ex.n_grow_vect,
            initial_value=0.0,
            redistribute=True,
            redistribute_on_remake=True
        )
        
        pressure = sim.fields.alloc_init(
            name="pressure_term",
            dir=dir_str,
            level=level,
            ba=Ex.box_array(),
            dm=Ex.dm(),
            ncomp=1,
            ngrow=Ex.n_grow_vect,
            initial_value=0.0,
            redistribute=True,
            redistribute_on_remake=True
        )
    
    # Add these fields to diagnostic output
    sim.extension.warpx.add_field_to_diagnostic("diag1", "hall_term", lev=level)
    sim.extension.warpx.add_field_to_diagnostic("diag1", "pressure_term", lev=level)
    
    print("Allocated Ohm's law diagnostic fields")


@callbacks.callfrombeforediagnostics
def update_ohm_law_terms():
    """
    Callback that runs before each diagnostic output.
    Recomputes the Ohm's law terms so they're up-to-date in the output.
    """
    compute_ohm_law_terms()


# ============================================================================
# Run simulation
# ============================================================================

sim.step()

print("Simulation complete. Ohm's law terms written to diagnostic output.")
print("Output fields include:")
print("  - hall_term_x, hall_term_y, hall_term_z")
print("  - pressure_term_x, pressure_term_y, pressure_term_z")
