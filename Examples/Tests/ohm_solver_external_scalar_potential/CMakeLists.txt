# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
add_warpx_test(
    test_rz_ohm_solver_external_scalar_potential_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_ohm_solver_external_scalar_potential_picmi.py --key boundaries --shield 1"  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_ohm_solver_external_scalar_potential_ramped_eb_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_ohm_solver_external_scalar_potential_picmi.py --key boundary_potentials --shield 1 --ramp-eb-potential"  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_rz_ohm_solver_external_scalar_potential_unshielded_ramped_A_picmi  # name
    RZ  # dims
    1  # nprocs
    "inputs_test_rz_ohm_solver_external_scalar_potential_picmi.py --key electrode --shield 0 --ramp-A --ramp-phi-time"  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000001"  # checksum
    OFF  # dependency
)
endif()
