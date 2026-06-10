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
        test_rz_poisson_efield_correction  # name
        RZ  # dims
        2  # nprocs
        PICMI_inputs_rz.py  # inputs
        analysis_rz.py  # analysis
        OFF  # checksum
        OFF  # dependency
    )
endif()
