# Add tests (alphabetical order) ##############################################
#

add_warpx_test(
    test_2d_laser_ion_acc  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_laser_ion_acc  # inputs
    "analysis_test_laser_ion.py diags/diagInst/"  # analysis
    "analysis_default_regression.py --path diags/diagInst/"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_laser_ion_acc_restart  # name
    2  # dims
    2  # nprocs
    "inputs_test_2d_laser_ion_acc amr.restart='../test_2d_laser_ion_acc/diags/chk000098'"  # inputs
    "analysis_test_laser_ion.py diags/diagInst/"  # analysis
    OFF  # checksum: compared directly with the original run
    test_2d_laser_ion_acc  # dependency
)

add_warpx_test(
    test_2d_laser_ion_acc_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_laser_ion_acc_picmi.py  # inputs
    "analysis_test_laser_ion.py diags/diagInst/"  # analysis
    "analysis_default_regression.py --path diags/diagInst/"  # checksum
    OFF  # dependency
)
