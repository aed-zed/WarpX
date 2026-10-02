# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_rz_coaxial_space_charge_diode_picmi  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_coaxial_space_charge_diode_picmi.py --transits 6"  # inputs
        "analysis_coaxial_space_charge_diode.py"  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    label_warpx_test(test_rz_coaxial_space_charge_diode_picmi slow)
endif()
