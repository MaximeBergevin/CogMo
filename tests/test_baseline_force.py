import pytest
import numpy as np

# Import the function to be tested
from force_analyses import find_baseline_force

@pytest.mark.parametrize(
    "_test_id, max_noise, shift_baseline, expected_is_valid",
    [
        ("Happy path: Low noise", 0.1, 0.0, True),
        ("Happy path: High noise", 0.9, 0.0, True),
        ("Happy path: Shifted baseline", 0.5, 5.0, True),
        ("Bad path: Very high noise", 5.0, 0.0, False), # SD of noise will be 5
    ]
)
def test_find_baseline_force(
    _test_id, max_noise, shift_baseline, expected_is_valid,
    mock_signal_data_factory
):
    """
    Tests the find_baseline_force function under various noise conditions.
    """
    # Generate mock data for the specific test case
    mock_df, expected = mock_signal_data_factory(
        dominant_force = "right",
        motor_condition = "high",
        max_noise = max_noise,
        shift_baseline = shift_baseline,
        burst_time_s = 1.0 
    )
    
    result_dict = find_baseline_force(
        signal_df = mock_df,
        stim_time = expected['stim_time_exact'],
        response_hand = "right"
    )

    # --- Assertions ---
    # -------------------
    if expected_is_valid:
        # Check dictionary structure
        assert isinstance(result_dict, dict)
        assert "mean" in result_dict
        assert "sd" in result_dict
        
        # Check Mean: The calculated baseline mean should be very close
        assert result_dict['mean'] == pytest.approx(shift_baseline, abs=max_noise)
        
        # Check SD: For happy paths, SD should be relatively low
        assert result_dict['sd'] <= 1.0
    else:
        assert isinstance(result_dict, dict)
        assert result_dict['sd'] > 1.0