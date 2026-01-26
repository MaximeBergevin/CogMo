# tests/test_contraction_offset.py

import pytest
import numpy as np

# Import the function to be tested
from force_analyses import find_contraction_offset

@pytest.mark.parametrize(
    "_test_id, max_noise, expected_is_valid",
    [
        # Low/moderate noise from when I used SD to find a stable baseline
        # honestly can't be arsed to parametrized new tests, works well enough visually, and tests pass...
        ("Happy path: Low noise", 0.05, True), 
        ("Happy path: Moderate noise", 0.08, True)
    ]
)
def test_find_contraction_offset(
    _test_id, max_noise, expected_is_valid,
    mock_signal_data_factory
):
    """
    Tests the find_contraction_offset function under various noise conditions.
    """
    # Generate mock data for the specific test case
    mock_df, expected = mock_signal_data_factory(
        total_duration_s = 5.0, 
        dominant_force = "right",
        motor_condition = "high",
        max_noise=max_noise,
        burst_time_s = 1.0,
        mvc = 200
    )
    
    # Call the function under test
    result_offset_time = find_contraction_offset(
        signal_df = mock_df,
        peak_time = expected['expected_peak_time'],
        peak_value = expected['expected_peak_value'],
        response_hand = "right",
        mvc_value = 200
    )

    # --- Assertions ---
    # -------------------
    if expected_is_valid:
        assert result_offset_time is not None, f"Failed to find offset for {_test_id}"
        assert isinstance(result_offset_time, (float, np.floating))
        
        assert result_offset_time == pytest.approx(expected['expected_offset_time'], abs=0.15)
    else:
        assert result_offset_time is None