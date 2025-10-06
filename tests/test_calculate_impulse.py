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
    "_test_id, dominant_hand, shift_baseline",
    [
        ("Right hand, minor shift", "right", 0.1),
        ("Left hand, major shift", "left", 15),
    ]
)
def test_calculate_impulse(
    _test_id, dominant_hand, shift_baseline,
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
        mvc=mvc_val
    )

    # --- Run the prerequisite foundational metrics ---
    #    This simulates the dependency chain in the main app
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
    results = calculate_impulse(
        signal_df=mock_df,
        onset_time=onset_time,
        offset_time=offset_time,
        baseline_force=baseline,
        mvc_value=mvc_val,
        response_hand=dominant_hand
    )

    # --- Assertions ---
    # -------------------
    assert isinstance(results, dict)
    
    # Check that the baseline-corrected AUC matches the ground truth from the factory
    assert results['impulse_auc'] == pytest.approx(expected['expected_auc'], abs=0.05)
    
    # Check the normalized value
    assert results['impulse_auc_percent_mvc'] == pytest.approx(expected['expected_auc_normalized'], abs=0.05)
