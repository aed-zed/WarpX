# Gauss-law consistency tests for embedded-boundary electrodes.
# These tests measure enclosed charge via discrete surface integrals
# and the ChargeOnEB reduced diagnostic, then generate interactive
# HTML visualizations.

# Pass-through: neutral beam passes between electrodes (no impact)
add_warpx_test(
    test_3d_gauss_passthrough       # name
    3                               # dims
    1                               # nprocs
    inputs_3d_gauss_movie.py        # inputs
    analysis_gauss_visualization.py # analysis
    OFF                             # checksum
    OFF                             # dependency
)
label_warpx_test(test_3d_gauss_passthrough slow)

# Impact: beams aimed at opposite-sign electrodes
add_warpx_test(
    test_3d_gauss_impact            # name
    3                               # dims
    1                               # nprocs
    inputs_3d_gauss_impact.py       # inputs
    analysis_gauss_visualization.py # analysis
    OFF                             # checksum
    OFF                             # dependency
)
label_warpx_test(test_3d_gauss_impact slow)
