# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_3d_python_registered_fields_picmi  # name
    3  # dims
    2  # nprocs
    inputs_test_3d_python_registered_fields_picmi.py  # inputs
    "analysis.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_3d_python_registered_fields_picmi_restart  # name
    3  # dims
    2  # nprocs
    "inputs_test_3d_python_registered_fields_picmi.py amr.restart='../test_3d_python_registered_fields_picmi/diags/chk000005'"  # inputs
    "analysis.py diags/diag1000010"  # analysis
    "analysis_default_regression.py --path diags/diag1000010"  # checksum
    test_3d_python_registered_fields_picmi  # dependency
)
