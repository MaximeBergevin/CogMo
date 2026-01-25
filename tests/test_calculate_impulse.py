import pytest
import numpy as np

# Import the functions to be tested
from force_analyses import (
    calculate_impulse, 
    find_contraction_onset, 
    find_contraction_offset, 
    find_baseline_force
)

@pytest.mark.parametrize(
    "_test_id, dominant_hand, shift_baseline, motor_cond",
    [
        ("Right hand, high force, clean", "right", 0.05, "high"),
        ("Left hand, high force, shifted", "left", 0.1, "high"),
    ]
)
def test_calculate_impulse(
    _test_id, dominant_hand, shift_baseline, motor_cond,
    mock_signal_data_factory
):
    """
    Tests the calculate_impulse function with baseline correction.
    """
    # --- Generate mock data for the specific test case ---
    mvc_val = 400.0
    mock_df, expected = mock_signal_data_factory(
        total_duration_s = 5.0,
        dominant_force=dominant_hand,
        shift_baseline=shift_baseline,
        mvc=mvc_val,
        motor_condition=motor_cond,
        burst_time_s=1.0,
        max_noise=0.05 
    )

    # --- Run the prerequisite foundational metrics ---
    onset_time = find_contraction_onset(
        signal_df=mock_df,
        stim_time=expected['stim_time_exact'],
        peak_time=expected['expected_peak_time'],
        peak_value=expected['expected_peak_value'],
        response_hand=dominant_hand
    )
    
    offset_time = find_contraction_offset(
        signal_df=mock_df,
        peak_time=expected['expected_peak_time'],
        peak_value=expected['expected_peak_value'],
        response_hand=dominant_hand
    )
    
    baseline_results = find_baseline_force(
        signal_df=mock_df,
        stim_time=expected['stim_time_exact'],
        response_hand=dominant_hand
    )
    
    # Validate prerequisites before calling the test function
    assert baseline_results['mean'] is not None, "Baseline search failed (returned None)"
    assert onset_time is not None, f"Onset detection failed for {_test_id}"
    assert offset_time is not None, f"Offset detection failed for {_test_id}"
    
    # --- Call the function under test ---
    results = calculate_impulse(
        signal_df=mock_df,
        onset_time=onset_time,
        offset_time=offset_time,
        baseline_force=baseline_results['mean'],
        mvc_value=mvc_val,
        response_hand=dominant_hand
    )

    # --- Assertions ---
    # -------------------
    assert isinstance(results, dict), "Result should be a dictionary"
    
    # Check that the baseline-corrected AUC matches the ground truth from the factory
    assert results['impulse_auc'] == pytest.approx(expected['expected_auc'], abs=0.1)
    
    # Check the normalized value (%MVC/s)
    assert results['impulse_auc_percent_mvc'] == pytest.approx(expected['expected_auc_normalized'], abs=0.1)