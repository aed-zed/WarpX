# Add tests (alphabetical order) ##############################################
#
# The inputs contain the numerical assertions, so there are no analysis scripts.

if(WarpX_EB)
    add_warpx_test(
        test_rz_voltage_clamp_checkpoint_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_voltage_clamp_picmi.py --case restart --mode base --max-grid-size 16"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rz_voltage_clamp_checkpoint_picmi_restart  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_voltage_clamp_picmi.py --case restart --mode restart --max-grid-size 16"  # inputs
        OFF  # analysis
        OFF  # checksum
        test_rz_voltage_clamp_checkpoint_picmi  # dependency
    )
    add_warpx_test(
        test_rz_voltage_clamp_diode_off_picmi  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_voltage_clamp_picmi.py --case diode --arm off"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rz_voltage_clamp_diode_picmi  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_voltage_clamp_picmi.py --case diode"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rz_voltage_clamp_diode_wall_ground_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_voltage_clamp_picmi.py --case diode --sleeve joined --max-grid-size 16"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rz_voltage_clamp_insulator_current_exit_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_voltage_clamp_insulator_picmi.py --face current --case exit --max-grid-size 16"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rz_voltage_clamp_insulator_current_land_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_voltage_clamp_insulator_picmi.py --face current --case land --max-grid-size 16"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rz_voltage_clamp_insulator_current_rod_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_voltage_clamp_insulator_picmi.py --face current --case rod --max-grid-size 16"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rz_voltage_clamp_insulator_current_two_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_voltage_clamp_insulator_picmi.py --face current --case two --max-grid-size 16"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rz_voltage_clamp_insulator_voltage_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_voltage_clamp_insulator_picmi.py --face voltage --max-grid-size 16"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rz_voltage_clamp_insulator_voltage_two_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_voltage_clamp_insulator_picmi.py --face voltage --case two --max-grid-size 16"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
    add_warpx_test(
        test_rz_voltage_clamp_vacuum_picmi  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_voltage_clamp_picmi.py --case vacuum"  # inputs
        OFF  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()
