# tests/test_contraction_offset.py

import pytest
import numpy as np

# Import the function to be tested
from force_analyses import find_contraction_offset

@pytest.mark.parametrize(
    "_test_id, max_noise, expected_is_valid",
    [
        ("Happy path: Low noise", 0.1, True),
        ("Happy path: High noise", 0.9, True),
        ("Bad path: Very high noise", 2.5, False),
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
        burst_time_s = 1.0
    )
    
    # Call the function under test
    result_offset_time = find_contraction_offset(
        signal_df = mock_df,
        peak_time = expected['expected_peak_time'],
        peak_value = expected['expected_peak_value'],
        response_hand = "right"
    )

    # --- Assertions ---
    # -------------------
    if expected_is_valid:
        assert isinstance(result_offset_time, (float, np.floating))
        assert result_offset_time == pytest.approx(expected['expected_offset_time'], abs=0.1)
    else:
        assert result_offset_time is None