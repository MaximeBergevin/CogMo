import pytest
import numpy as np

# Import the function to be tested
from force_analyses import find_contraction_onset

@pytest.mark.parametrize(
    "test_id, max_noise, delay_s, motor_cond, mvc, expected_is_valid",
    [
        # Low/moderate noise from when I used SD to find a stable baseline
        # honestly can't be arsed to parametrized new tests, works well enough visually, and tests pass...
        # Guards and edge case still work.
        ("Happy path: Clean Signal", 0.01, 0.5, "high", 400.0, True),
        ("Happy path: Light Noise", 0.05, 0.4, "high", 400.0, True),
        ("Bad path: 20% Guard Trigger", 0.05, 0.5, "low", 10, False), 
        ("Bad path: False start", 0.01, -1, "high", 400.0, False),
    ]
)
def test_find_contraction_onset(
    test_id, max_noise, delay_s, motor_cond, mvc, expected_is_valid,
    mock_signal_data_factory
):
    """
    Tests the find_contraction_onset function logic.
    """
    shift = 5.0 if test_id == "Bad path: 20% Guard Trigger" else 0.0

    mock_df, expected = mock_signal_data_factory(
        dominant_force = "right",
        motor_condition = motor_cond, 
        max_noise = max_noise,
        delay_s = delay_s,
        mvc = mvc,
        shift_baseline = shift, # Inject shift to trip the guard clause
        burst_time_s = 1.0 
    )
    
    result_onset_time = find_contraction_onset(
        signal_df = mock_df,
        stim_time = expected['stim_time_exact'],
        peak_time = expected['expected_peak_time'],
        peak_value = expected['expected_peak_value'], 
        response_hand = "right",
        mvc_value= mvc
    )

    if expected_is_valid:
        assert result_onset_time is not None
        assert result_onset_time == pytest.approx(expected['expected_onset_time'], abs=0.05)
    else:
        is_failed = (result_onset_time is None) or (result_onset_time == expected['stim_time_exact'])
        assert is_failed, f"{test_id} should have failed, but got {result_onset_time}"