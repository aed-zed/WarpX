# Custom field diagnostic output (features 1 and 2)
add_warpx_test(
        test_3d_custom_field_diag_picmi  # name
        3  # dims
        1  # nprocs
        inputs_test_custom_field_diag_picmi.py  # inputs
        OFF  # analysis (test validates itself with assertions)
        OFF  # checksum (API test - field values are not physics-specific)
        OFF  # dependency
)

# MultiFabRegister checkpoint flag API
add_warpx_test(
        test_3d_checkpoint_flag_picmi  # name
        3  # dims
        1  # nprocs
        inputs_test_checkpoint_flag_picmi.py  # inputs
        OFF  # analysis (test validates itself with assertions)
        OFF  # checksum (API test - no file output)
        OFF  # dependency
)

# Checkpoint/restart with custom fields - Part 1: Initial run
add_warpx_test(
        test_3d_checkpoint_restart_custom_fields_picmi  # name
        3  # dims
        1  # nprocs
        inputs_test_checkpoint_restart_custom_fields_picmi.py  # inputs
        OFF  # analysis (test validates itself with assertions)
        OFF  # checksum (API test)
        OFF  # dependency
)

# Checkpoint/restart with custom fields - Part 2: Restart and verify
add_warpx_test(
        test_3d_checkpoint_restart_custom_fields_picmi_restart  # name
        3  # dims
        1  # nprocs
        "inputs_test_checkpoint_restart_custom_fields_picmi.py --restart ../test_3d_checkpoint_restart_custom_fields_picmi/diags/checkpoint_restart_test000005"  # inputs
        OFF  # analysis (test validates itself with assertions)
        OFF  # checksum (API test)
        test_3d_checkpoint_restart_custom_fields_picmi  # dependency
)
