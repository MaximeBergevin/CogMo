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
        ("Bad path: Very high noise", 2.0, 0.0, False), # SD of noise will be > 1.0
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
        dominant_force="right",
        motor_condition="high",
        max_noise=max_noise,
        shift_baseline=shift_baseline,
        burst_time_s=1.0 
    )
    
    # Call the function under test
    result_baseline = find_baseline_force(
        signal_df=mock_df,
        peak_time=expected['expected_peak_time'],
        response_hand="right"
    )

    # --- Assertions ---
    # -------------------
    if expected_is_valid:
        # For happy paths, the result should be a number
        assert isinstance(result_baseline, (float, np.floating))
        # The calculated baseline mean should be very close to the shift we applied
        assert result_baseline == pytest.approx(shift_baseline, abs=max_noise)
    else:
        # For the bad path, the function should fail to find a stable baseline
        # and correctly return None
        assert result_baseline is None