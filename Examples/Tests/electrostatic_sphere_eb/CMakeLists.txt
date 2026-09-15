# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_electrostatic_sphere_eb  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_electrostatic_sphere_eb  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1/"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_electrostatic_sphere_eb_mixed_bc  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_electrostatic_sphere_eb_mixed_bc  # inputs
        OFF  # analysis
        "analysis_default_regression.py --path diags/diag1000001"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_electrostatic_sphere_eb_picmi  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_electrostatic_sphere_eb_picmi.py  # inputs
        "analysis.py"  # analysis
        "analysis_default_regression.py --path diags/diag1/"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_adjoint_weighting_decomposed_picmi  # name
        3  # dims
        2  # nprocs (cut, fully regular and fully covered boxes)
        "inputs_test_3d_adjoint_weighting_picmi.py --decomposed"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_adjoint_weighting_picmi  # name
        3  # dims
        2  # nprocs (also exercises shared nodal box interfaces)
        inputs_test_3d_adjoint_weighting_picmi.py  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_adjoint_weighting_ordinary_actuator_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_adjoint_weighting_picmi.py --actuator-gradient ordinary"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_adjoint_weighting_picmi  # name
        RZ  # dims
        2  # nprocs (also exercises shared nodal box interfaces)
        inputs_test_rz_adjoint_weighting_picmi.py  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_adjoint_weighting_ordinary_actuator_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_adjoint_weighting_picmi.py --actuator-gradient ordinary"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

# The clamp's actuator field must be compatible with the Yee Faraday update.
# Both directions are registered: the ordinary gradient must leave B at roundoff,
# and the EB-aware gradient must still grow it, so the two stay distinguishable.
if(WarpX_EB)
    add_warpx_test(
        test_rz_clamp_actuator_faraday_ordinary_picmi  # name
        RZ  # dims
        2  # nprocs (the domain splits in r as well as z)
        "inputs_test_rz_clamp_actuator_faraday_picmi.py --actuator-gradient ordinary"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_clamp_actuator_faraday_eb_aware_picmi  # name
        RZ  # dims
        2  # nprocs (the domain splits in r as well as z)
        "inputs_test_rz_clamp_actuator_faraday_picmi.py --actuator-gradient eb_aware"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

# Two driven electrodes that are SEPARATE BODIES: a cathode rod on the axis and
# an isolated ring, with the grounded vessel as the Dirichlet wall. Asserts the
# properties that make such a geometry a well-posed two-degree-of-freedom
# problem: a reciprocal capacitance matrix, non-zero column sums (so neither
# electrode encloses the other and both actuators have work), and a closing
# loop. Uses the volume charge observer; the EB surface integral gives a 36%
# non-reciprocal matrix on this same geometry.
if(WarpX_EB)
    add_warpx_test(
        test_rz_clamp_coax_ring_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_clamp_coax_ring_picmi.py"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

# Periodic outer boundaries. A periodic direction has no wall, so its domain-edge
# nodes are unknowns rather than constrained rows; charge is placed so that its
# deposition support crosses the seam.
if(WarpX_EB)
    add_warpx_test(
        test_3d_adjoint_weighting_periodic_picmi  # name
        3  # dims
        2  # nprocs
        inputs_test_3d_adjoint_weighting_periodic_picmi.py  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_adjoint_weighting_periodic_picmi  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_adjoint_weighting_periodic_picmi.py  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

# The Cartesian counterpart uses two physically separate conducting spheres, so
# no single body carries two potentials. The registered 3-D adjoint tests do not
# distinguish the actuator representations; these do.
if(WarpX_EB)
    add_warpx_test(
        test_3d_clamp_actuator_faraday_ordinary_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_clamp_actuator_faraday_picmi.py --actuator-gradient ordinary"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_3d_clamp_actuator_faraday_eb_aware_picmi  # name
        3  # dims
        2  # nprocs
        "inputs_test_3d_clamp_actuator_faraday_picmi.py --actuator-gradient eb_aware"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

# Checkpoint/restart pair: afterInitEsolve does not fire on a restart, so the
# clamp must rebuild its calibration from afterInitatRestart, exactly once.
if(WarpX_EB)
    add_warpx_test(
        test_rz_clamp_restart_base_picmi  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_clamp_restart_base_picmi.py  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_clamp_restart_picmi  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_clamp_restart_picmi.py  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        test_rz_clamp_restart_base_picmi  # dependency (reads its checkpoint)
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_bias_decomposed_picmi  # name
        RZ  # dims
        2  # nprocs (six boxes split across both ranks)
        "inputs_test_rz_staircase_bias_picmi.py --max-grid-size 8"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_bias_insulating_endcaps_decomposed_picmi  # name
        RZ  # dims
        2  # nprocs (same six boxes split across both ranks)
        "inputs_test_rz_staircase_bias_picmi.py --max-grid-size 8 --insulating-endcaps"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_bias_insulating_endcaps_picmi  # name
        RZ  # dims
        1  # nprocs (one-box reference)
        "inputs_test_rz_staircase_bias_picmi.py --max-grid-size 1024 --insulating-endcaps"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_bias_picmi  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_staircase_bias_picmi.py --max-grid-size 1024"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_bias_insulator_ring_picmi  # name
        RZ  # dims
        2  # nprocs (radial and axial box interfaces)
        "inputs_test_rz_insulator_ring_picmi.py --max-grid-size 16"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_bias_ground_reference_decomposed_picmi  # name
        RZ  # dims
        2  # nprocs (the reference crosses radial and axial box interfaces)
        "inputs_test_rz_ground_reference_picmi.py --max-grid-size 16"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_bias_ground_reference_picmi  # name
        RZ  # dims
        1  # nprocs (one-box reference)
        "inputs_test_rz_ground_reference_picmi.py --max-grid-size 1024"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_bias_ground_reference_long_coax_picmi  # name
        RZ  # dims
        2  # nprocs (long coax split across axial boxes and ranks)
        "inputs_test_rz_ground_reference_picmi.py --long-coax --max-grid-size 64"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_bias_ground_reference_momentum_gather_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_ground_reference_picmi.py --momentum-gather --max-grid-size 16"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_adjoint_weighting_decomposed_picmi  # name
        RZ  # dims
        2  # nprocs (the domain also splits radially, off the conductor)
        inputs_test_rz_adjoint_weighting_decomposed_picmi.py  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_electrostatic_sphere_eb  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_electrostatic_sphere_eb  # inputs
        "analysis_rz.py diags/diag1000001"  # analysis
        "analysis_default_regression.py --path diags/diag1000001 --skip-particles"  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_electrostatic_sphere_eb_mr  # name
        RZ  # dims
        2  # nprocs
        inputs_test_rz_electrostatic_sphere_eb_mr  # inputs
        "analysis_rz_mr.py diags/diag1/"  # analysis
        "analysis_default_regression.py --path diags/diag1/"  # checksum
        OFF  # dependency
    )
endif()
