# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    add_warpx_test(
        test_3d_poisson_efield_correction  # name
        3  # dims
        2  # nprocs
        PICMI_inputs_3d.py  # inputs
        analysis.py  # analysis
        OFF  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_3d_poisson_efield_correction_harmonic_bias  # name
        3  # dims
        2  # nprocs
        PICMI_inputs_3d_harmonic_bias.py  # inputs
        analysis_harmonic_bias.py  # analysis
        OFF  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_3d_poisson_efield_correction_combined  # name
        3  # dims
        2  # nprocs
        PICMI_inputs_3d_combined.py  # inputs
        analysis_harmonic_bias.py  # analysis
        OFF  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_3d_poisson_efield_correction_two_electrode  # name
        3  # dims
        2  # nprocs
        PICMI_inputs_3d_two_electrode.py  # inputs
        analysis_two_electrode.py  # analysis
        OFF  # checksum
        OFF  # dependency
    )

    add_warpx_test(
        test_rz_poisson_efield_correction  # name
        RZ  # dims
        2  # nprocs
        PICMI_inputs_rz.py  # inputs
        analysis_rz.py  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()
