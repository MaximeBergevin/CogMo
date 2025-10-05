# tests/analyses/test_force_analyses.py

import pytest
from force_analyses import peak_force_metrics
from conftest import create_mock_signal_data

# Define the four test cases using pytest's parametrize decorator
# Each tuple represents one test run with a unique combination of inputs.
@pytest.mark.parametrize(
    "dominant_hand, motor_condition, overshoot, expected_direction",
    [
        ("right", "high", True, "overshoot"),
        ("right", "high", False, "undershoot"),
        ("left", "low", True, "overshoot"),
        ("left", "low", False, "undershoot"),
    ]
)
def test_peak_force_metrics(
    dominant_hand, motor_condition, overshoot, expected_direction,
    mock_signal_data_factory # Use the factory fixture from conftest
):
    """
    Tests the peak_force_metrics function across different conditions.
    """
    # Generate mock data for the specific test case ---
    mvc_val = 200.0
    mock_df, expected = mock_signal_data_factory(
        dominant_force = dominant_hand,
        motor_condition = motor_condition,
        overshoot = overshoot,
        mvc = mvc_val,
        # Use standard column names as the function expects them
        force_r_col_name = "force_right",
        force_l_col_name = "force_left",
        include_emg = False
    )

    # Call the function under test ---
    results = peak_force_metrics(
        signal_df = mock_df,
        stim_time = expected['stim_time_exact'],
        response_hand = dominant_hand,
        threshold = expected['expected_threshold'],
        mvc_left = mvc_val,
        mvc_right = mvc_val
    )

    # Assertions
    # ---------------
    
    # 1. Check peak force values
    assert results['peak_force'] == pytest.approx(expected['expected_peak_value'])
    assert results['delta_threshold'] == pytest.approx(
        expected['expected_peak_value'] - expected['expected_threshold']
    )
    
    # 2. Check direction
    assert results['threshold_direction'] == expected_direction
    
    # 3. Check time to peak
    expected_ttp = expected['expected_peak_time'] - expected['stim_time_exact']
    assert results['time_to_peak'] == pytest.approx(expected_ttp)
    
    # 4. Check percentage-based metrics
    expected_pct_mvc = (expected['expected_peak_value'] / mvc_val) * 100
    assert results['peak_force_pct_mvc'] == pytest.approx(expected_pct_mvc)

    if expected['expected_threshold'] > 0:
        delta = expected['expected_peak_value'] - expected['expected_threshold']
        expected_pct_thresh = (delta / expected['expected_threshold']) * 100
        assert results['delta_threshold_pct'] == pytest.approx(expected_pct_thresh)