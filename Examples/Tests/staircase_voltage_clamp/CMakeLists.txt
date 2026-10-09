# Add tests (alphabetical order) ##############################################
#

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
        test_rz_staircase_clamp_diode_off_picmi  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_staircase_clamp_diode_picmi.py --arm off"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_clamp_diode_on_picmi  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_staircase_clamp_diode_picmi.py --arm on"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_grounded_charge_decomposed_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_staircase_grounded_charge_picmi.py --max-grid-size 16"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_grounded_charge_picmi  # name
        RZ  # dims
        1  # nprocs
        "inputs_test_rz_staircase_grounded_charge_picmi.py"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_clamp_restart_base_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_staircase_clamp_restart_picmi.py --mode base"  # inputs
        OFF  # analysis (writes the checkpoint and trace used by the restart test)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_clamp_restart_picmi  # name
        RZ  # dims
        2  # nprocs
        "inputs_test_rz_staircase_clamp_restart_picmi.py --mode restart"  # inputs
        OFF  # analysis (the input compares with the base run)
        OFF  # checksum
        test_rz_staircase_clamp_restart_base_picmi  # dependency (reads its checkpoint)
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_pmc_coax_picmi  # name
        RZ  # dims
        1  # nprocs (vacuum coax, exact discrete solution)
        "inputs_test_rz_staircase_pmc_picmi.py --case coax"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_pmc_exit_picmi  # name
        RZ  # dims
        1  # nprocs (charge exits through a PMC face)
        "inputs_test_rz_staircase_pmc_picmi.py --case exit"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_pmc_exit_decomposed_picmi  # name
        RZ  # dims
        2  # nprocs (box boundaries at and away from the faces)
        "inputs_test_rz_staircase_pmc_picmi.py --case exit --max-grid-size 16 --compare-with ../test_rz_staircase_pmc_exit_picmi"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        test_rz_staircase_pmc_exit_picmi  # dependency (reads its output)
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_pmc_exit_rho_pairing_picmi  # name
        RZ  # dims
        1  # nprocs (asserts the rho-pairing drift)
        "inputs_test_rz_staircase_pmc_picmi.py --case exit --pairing rho"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_pmc_restart_base_picmi  # name
        RZ  # dims
        2  # nprocs (writes the checkpoint and trace used by the restart test)
        "inputs_test_rz_staircase_pmc_picmi.py --case restart --mode base --max-grid-size 16"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_pmc_restart_picmi  # name
        RZ  # dims
        2  # nprocs (compares with the base run)
        "inputs_test_rz_staircase_pmc_picmi.py --case restart --mode restart --max-grid-size 16"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        test_rz_staircase_pmc_restart_base_picmi  # dependency (reads its output)
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_pmc_rings_picmi  # name
        RZ  # dims
        1  # nprocs (end rings against a dense reference solve)
        "inputs_test_rz_staircase_pmc_picmi.py --case rings"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_pmc_rings_decomposed_picmi  # name
        RZ  # dims
        2  # nprocs (box boundaries at and away from the faces)
        "inputs_test_rz_staircase_pmc_picmi.py --case rings --max-grid-size 8 --compare-with ../test_rz_staircase_pmc_rings_picmi"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        test_rz_staircase_pmc_rings_picmi  # dependency (reads its output)
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_pmc_rings_ground_reference_picmi  # name
        RZ  # dims
        1  # nprocs (outer ring connected to the grounded wall)
        "inputs_test_rz_staircase_pmc_picmi.py --case rings --ground-reference"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()

if(WarpX_EB)
    add_warpx_test(
        test_rz_staircase_pmc_rings_pec_pmc_picmi  # name
        RZ  # dims
        1  # nprocs (PEC lower face, PMC upper face)
        "inputs_test_rz_staircase_pmc_picmi.py --case rings --z-faces pec_pmc"  # inputs
        OFF  # analysis (the input contains numerical assertions)
        OFF  # checksum
        OFF  # dependency
    )
endif()
