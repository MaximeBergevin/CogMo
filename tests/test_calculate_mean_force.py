import pytest
import numpy as np

# Import the functions to be tested
from force_analyses import (
    calculate_mean_force, 
    find_contraction_onset, 
    find_contraction_offset, 
    find_baseline_force
)

@pytest.mark.parametrize(
    "_test_id, dominant_hand, mvc_val, burst_time_s, max_noise",
    [
        ("Right hand + Short contraction", "right", 400.0, 1.0, 0.05), 
        ("Left hand + Long contraction", "left", 150.0, 1.0, 0.05), 
        ("Right hand + Low noise", "right", 400.0, 1.0, 0.05), 
    ]
)
def test_calculate_mean_force(
    _test_id, dominant_hand, mvc_val, burst_time_s, max_noise,
    mock_signal_data_factory
):
    """
    Tests the calculate_mean_force function.
    """
    # --- Generate mock data ---
    mock_df, expected = mock_signal_data_factory(
        total_duration_s=5.0, 
        dominant_force=dominant_hand,
        mvc=mvc_val,
        motor_condition="high",
        burst_time_s=burst_time_s,
        max_noise=max_noise 
    )

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
    
    baseline_res = find_baseline_force(
        signal_df=mock_df,
        stim_time=expected['stim_time_exact'],
        response_hand=dominant_hand
    )
    
    assert onset_time is not None, f"Onset failed (noise={max_noise})"
    assert offset_time is not None, "Offset failed"
    assert baseline_res['mean'] is not None, "Baseline failed"
    
    # --- Call Function ---
    results = calculate_mean_force(
        signal_df=mock_df,
        onset_time=onset_time,
        offset_time=offset_time,
        baseline_force=baseline_res['mean'],
        mvc_value=mvc_val,
        response_hand=dominant_hand
    )

    # --- Assertions ---
    assert isinstance(results, dict)
    # Check mean force matches factory ground truth
    assert results['mean_force'] == pytest.approx(expected['expected_mean_force'], abs=0.2)
    # Check normalized %MVC
    assert results['mean_force_percent_mvc'] == pytest.approx(expected['expected_mean_force_pct'], abs=0.2)