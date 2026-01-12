import pytest
import numpy as np

# Import the function to be tested
from force_analyses import find_contraction_onset

# Import the factory for creating mock data
from conftest import create_mock_signal_data

@pytest.mark.parametrize(
    "test_id, max_noise, delay_s, expected_is_valid",
    [
        ("Happy path: Low noise", 0.1, 0.5, True),       # Noise << SD threshold
        ("Happy path: High noise", 0.9, 0.5, True),      # Noise < SD threshold
        ("Bad path: Very high noise", 2.5, 0.5, False),  # Noise > SD threshold
        ("Bad path: False start", 0.5, -0.5, False),     # Burst before stimulus
    ]
)
def test_find_contraction_onset(
    test_id, max_noise, delay_s, expected_is_valid,
    mock_signal_data_factory
):
    """
    Tests the find_contraction_onset function under various noise conditions and edge cases.
    """
    # Generate mock data for the specific test case
    mock_df, expected = mock_signal_data_factory(
        dominant_force = "right",
        motor_condition = "high",
        max_noise = max_noise,
        delay_s = delay_s,
        burst_time_s = 1.0 # Use a longer burst for the 'false start' case
    )
    
    # Call the function under test
    result_onset_time = find_contraction_onset(
        signal_df = mock_df,
        stim_time = expected['stim_time_exact'],
        peak_time = expected['expected_peak_time'],
        response_hand="right"
    )

    # Assertions
    # ------------
    
    # 1. Check that results is approx. the correct float or None based on expected validity
    if expected_is_valid:
        # For happy paths, check that the result is a number and is close to the expected value
        assert isinstance(result_onset_time, (float, np.floating))
        assert result_onset_time == pytest.approx(expected['expected_onset_time'], abs=0.01)
    else:
        # For bad paths, assert that the function correctly returns None
        assert result_onset_time is None