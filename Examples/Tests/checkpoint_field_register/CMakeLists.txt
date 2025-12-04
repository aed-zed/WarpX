# Test for PR1: MultiFabRegister checkpoint flag API
add_warpx_test(
        test_3d_checkpoint_flag_picmi  # name
        3  # dims
        1  # nprocs
        inputs_test_checkpoint_flag_picmi.py  # inputs
        OFF  # analysis (test validates itself with assertions)
        OFF  # checksum (API test - no file output)
        OFF  # dependency
)
