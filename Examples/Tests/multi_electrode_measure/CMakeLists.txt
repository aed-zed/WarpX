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

# Correction: asymmetric beams + single Gauss clean + harmonic bias
add_warpx_test(
    test_3d_gauss_correction            # name
    3                                   # dims
    1                                   # nprocs
    inputs_3d_gauss_correction.py       # inputs
    analysis_gauss_visualization.py     # analysis
    OFF                                 # checksum
    OFF                                 # dependency
)
label_warpx_test(test_3d_gauss_correction slow)

# Comparison: three correction variants (full bias, clean+bias, masked bias)
add_warpx_test(
    test_3d_gauss_comparison             # name
    3                                    # dims
    1                                    # nprocs
    inputs_3d_gauss_comparison.py        # inputs
    analysis_gauss_visualization.py      # analysis
    OFF                                  # checksum
    OFF                                  # dependency
)
label_warpx_test(test_3d_gauss_comparison slow)

# Periodic correction: Gauss clean + harmonic bias every 10 steps
add_warpx_test(
    test_3d_gauss_periodic_10            # name
    3                                    # dims
    1                                    # nprocs
    inputs_3d_gauss_periodic_10.py       # inputs
    analysis_gauss_visualization.py      # analysis
    OFF                                  # checksum
    OFF                                  # dependency
)
label_warpx_test(test_3d_gauss_periodic_10 slow)

# Periodic correction: Gauss clean + harmonic bias every 100 steps
add_warpx_test(
    test_3d_gauss_periodic_100           # name
    3                                    # dims
    1                                    # nprocs
    inputs_3d_gauss_periodic_100.py      # inputs
    analysis_gauss_visualization.py      # analysis
    OFF                                  # checksum
    OFF                                  # dependency
)
label_warpx_test(test_3d_gauss_periodic_100 slow)
