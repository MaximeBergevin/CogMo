import pytest
import numpy as np

# Import the functions to be tested
from force_analyses import (
    calculate_mean_force, 
    find_contraction_onset, 
    find_contraction_offset, 
    find_baseline_force
)

# Import the factory for creating mock data
from conftest import create_mock_signal_data

@pytest.mark.parametrize(
    "_test_id, dominant_hand, mvc_val, burst_time_s, max_noise",
    [
        ("Right hand + Short contraction", "right", 400.0, 0.2, 0.5), # Default noise
        ("Left hand + Long contraction", "left", 150.0, 0.5, 0.5), # Default noise
        ("Right hand + Low noise", "right", 400.0, 0.2, 0.01), # Low noise
    ]
)
def test_calculate_mean_force(
    _test_id, dominant_hand, mvc_val, burst_time_s, max_noise,
    mock_signal_data_factory
):
    """
    Tests the calculate_mean_force function.
    This test simulates the app's full dependency chain.
    """
    # --- Generate mock data for the specific test case ---
    mock_df, expected = mock_signal_data_factory(
        total_duration_s=5.0, # Ensure enough time for offset
        dominant_force=dominant_hand,
        mvc=mvc_val,
        burst_time_s=burst_time_s,
        max_noise=max_noise # Pass in the noise level
    )

    # --- Run the prerequisite foundational metrics ---
    onset_time = find_contraction_onset(
        signal_df=mock_df,
        stim_time=expected['stim_time_exact'],
        peak_time=expected['expected_peak_time'],
        response_hand=dominant_hand
    )
    offset_time = find_contraction_offset(
        signal_df=mock_df,
        peak_time=expected['expected_peak_time'],
        peak_value=expected['expected_peak_value'],
        response_hand=dominant_hand
    )
    baseline = find_baseline_force(
        signal_df=mock_df,
        peak_time=expected['expected_peak_time'],
        response_hand=dominant_hand
    )
    
    # --- Call the function under test ---
    results = calculate_mean_force(
        signal_df=mock_df,
        onset_time=onset_time,
        offset_time=offset_time,
        baseline_force=baseline,
        mvc_value=mvc_val,
        response_hand=dominant_hand
    )

    # --- Assertions ---
    assert isinstance(results, dict)
    
    # Check that the baseline-corrected mean force matches the ground truth
    # CORRECTED: Tolerance is now tied to the max_noise parameter
    assert results['mean_force'] == pytest.approx(expected['expected_mean_force'], abs=max_noise)
    
    # Check the normalized value
    # We also need to add a tolerance for the normalized value
    expected_pct_noise = (max_noise / mvc_val) * 100
    assert results['mean_force_percent_mvc'] == pytest.approx(expected['expected_mean_force_pct'], abs=expected_pct_noise)

