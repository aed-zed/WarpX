# Add tests (alphabetical order) ##############################################
#

if(WarpX_EB)
    foreach(D IN LISTS WarpX_DIMS)
        if(D STREQUAL "3" OR D STREQUAL "RZ")
            warpx_set_suffix_dims(SD ${D})
            set(target test_${SD}_eb_potential_parser_transition)
            add_executable(${target} test_eb_potential_parser_transition.cpp)
            target_link_libraries(${target} PRIVATE WarpX::lib_${SD})
            target_compile_features(${target} PRIVATE cxx_std_20)
            if(NOT BUILD_SHARED_LIBS)
                target_compile_definitions(${target} PRIVATE WARPX_STATIC_DEFINE)
            endif()
            add_test(NAME ${target} COMMAND $<TARGET_FILE:${target}>)
            set_property(TEST ${target} PROPERTY ENVIRONMENT "OMP_NUM_THREADS=1")
        endif()
    endforeach()
endif()

add_warpx_test(
    test_2d_dirichlet_bc  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_dirichlet_bc  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)

add_warpx_test(
    test_2d_dirichlet_bc_picmi  # name
    2  # dims
    2  # nprocs
    inputs_test_2d_dirichlet_bc_picmi.py  # inputs
    "analysis.py"  # analysis
    "analysis_default_regression.py --path diags/diag1000100"  # checksum
    OFF  # dependency
)
