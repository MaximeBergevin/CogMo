import pytest
import numpy as np

# Import the function to be tested
from force_analyses import motor_response_time

# Import the factory for creating mock data
from conftest import create_mock_signal_data

@pytest.mark.parametrize(
    "_test_id, dominant_hand, motor_condition, overshoot, expected_result_type",
    [
        ("Happy path: Low force", "right", "low", True, "int"),
        ("Happy path: High force", "left", "high", True, "int"),
        ("Undershoot returns None", "right", "low", False, "None"),
    ]
)
def test_motor_response_time(
    _test_id, dominant_hand, motor_condition, overshoot, expected_result_type,
    mock_signal_data_factory
):
    """
    Tests the motor_response_time function using its full, original signature.
    """
    # 1. --- Generate mock data to get "ground truth" values ---
    mock_df, expected = mock_signal_data_factory(
        dominant_force=dominant_hand,
        motor_condition=motor_condition,
        overshoot=overshoot
    )

    # 2. --- Call the function under test with the CORRECT, full signature ---
    result = motor_response_time(
        signal_df=mock_df,
        stim_time=expected['stim_time_exact'],
        peak_time=expected['expected_peak_time'],
        peak_force=expected['expected_peak_value'],
        threshold=expected['expected_threshold'],
        response_hand=dominant_hand
    )

    # 3. --- Assertions ---
    if expected_result_type == "int":
        # Calculate the ground truth value
        time_increment = 1 / 500 
        threshold = expected['expected_threshold']
        burst_start_index = int((expected['expected_onset_time'] - mock_df['time'].iloc[0]) / time_increment)
        force_col = f"force_{dominant_hand}"
        burst_vals = mock_df.loc[burst_start_index:, force_col].values
        
        crossing_index_in_burst = np.where(burst_vals >= threshold)[0][0]
        time_at_threshold = mock_df.loc[burst_start_index + crossing_index_in_burst, 'time']
        expected_mrspt_ms = int(round((time_at_threshold - expected['stim_time_exact']) * 1000))
        
        assert isinstance(result, int)
        assert result == pytest.approx(expected_mrspt_ms, abs=6)
    else:
        # Check that the function correctly returned None
        assert result is None

