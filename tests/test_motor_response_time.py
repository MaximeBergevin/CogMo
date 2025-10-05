import pytest
import numpy as np

# Import the functions to be tested
from force_analyses import peak_force_metrics, motor_response_time

# Import the factory for creating mock data
from conftest import create_mock_signal_data

# ==============================================================================
# --- Tests for peak_force_metrics (Parametrized) ---
# ==============================================================================
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
    mock_signal_data_factory
):
    """
    Tests the peak_force_metrics function across different conditions.
    """
    # Generate mock data for the specific test case
    mvc_val = 200.0
    mock_df, expected = mock_signal_data_factory(
        dominant_force = dominant_hand,
        motor_condition = motor_condition,
        overshoot = overshoot,
        mvc = mvc_val,
        force_r_col_name = "force_right",
        force_l_col_name = "force_left",
        include_emg = False
    )

    # Call the function under test
    results = peak_force_metrics(
        signal_df = mock_df,
        stim_time = expected['stim_time_exact'],
        response_hand = dominant_hand,
        threshold = expected['expected_threshold'],
        mvc_left = mvc_val,
        mvc_right = mvc_val
    )

    # Assertions

    # 1. Check absolute force values
    assert results['peak_force'] == pytest.approx(expected['expected_peak_value'])
    assert results['delta_threshold'] == pytest.approx(
        expected['expected_peak_value'] - expected['expected_threshold']
    )

    # 2. Check directionality (overshoot / undershoot)
    assert results['threshold_direction'] == expected_direction

    # 3. Check timing metrics
    expected_ttp = expected['expected_peak_time'] - expected['stim_time_exact']
    assert results['time_to_peak'] == pytest.approx(expected_ttp)
    
    # 4. Check normalized metrics
    expected_pct_mvc = (expected['expected_peak_value'] / mvc_val) * 100
    assert results['peak_force_pct_mvc'] == pytest.approx(expected_pct_mvc)

    if expected['expected_threshold'] > 0:
        delta = expected['expected_peak_value'] - expected['expected_threshold']
        expected_pct_thresh = (delta / expected['expected_threshold']) * 100
        assert results['delta_threshold_pct'] == pytest.approx(expected_pct_thresh)

# ==============================================================================
# --- Tests for motor_response_time ---
# ==============================================================================
def test_motor_response_time_happy_path_low_force(mock_signal_data_factory):
    """
    Tests motor_response_time for a standard right-hand, low-force trial.
    """
    mock_df, expected = mock_signal_data_factory(
        dominant_force = "right",
        motor_condition = "low"
    )

    time_increment = 1 / 500
    threshold = expected['expected_threshold']
    burst_start_index = int((expected['expected_onset_time'] - mock_df['time'].iloc[0]) / time_increment)
    burst_vals = mock_df.loc[burst_start_index:, 'force_right'].values
    
    crossing_index_in_burst = np.where(burst_vals >= threshold)[0][0]
    time_at_threshold = mock_df.loc[burst_start_index + crossing_index_in_burst, 'time']
    expected_mrspt_ms = int(round((time_at_threshold - expected['stim_time_exact']) * 1000))
    
    result = motor_response_time(
        signal_df = mock_df,
        stim_time = expected['stim_time_exact'],
        peak_time = expected['expected_peak_time'],
        peak_force = expected['expected_peak_value'],
        threshold = expected['expected_threshold'],
        response_hand = "right"
    )

    # 1. Check type and value
    assert isinstance(result, int)
    assert result == pytest.approx(expected_mrspt_ms, abs=6)


def test_motor_response_time_happy_path_high_force(mock_signal_data_factory):
    """
    Tests motor_response_time for a standard left-hand, high-force trial.
    """
    mock_df, expected = mock_signal_data_factory(
        dominant_force = "left",
        motor_condition = "high"
    )

    time_increment = 1 / 500
    threshold = expected['expected_threshold']
    burst_start_index = int((expected['expected_onset_time'] - mock_df['time'].iloc[0]) / time_increment)
    burst_vals = mock_df.loc[burst_start_index:, 'force_left'].values

    crossing_index_in_burst = np.where(burst_vals >= threshold)[0][0]
    time_at_threshold = mock_df.loc[burst_start_index + crossing_index_in_burst, 'time']
    expected_mrspt_ms = int(round((time_at_threshold - expected['stim_time_exact']) * 1000))

    result = motor_response_time(
        signal_df = mock_df,
        stim_time = expected['stim_time_exact'],
        peak_time = expected['expected_peak_time'],
        peak_force = expected['expected_peak_value'],
        threshold = expected['expected_threshold'],
        response_hand = "left"
    )
    
    # 1. Check type and value
    assert isinstance(result, int)
    assert result == pytest.approx(expected_mrspt_ms, abs=6)


def test_motor_response_time_undershoot_returns_none(mock_signal_data_factory):
    """
    Tests that motor_response_time returns None when the force never crosses the threshold.
    """
    mock_df, expected = mock_signal_data_factory(
        overshoot = False,
        motor_condition = "low"
    )
    
    result = motor_response_time(
        signal_df = mock_df,
        stim_time = expected['stim_time_exact'],
        peak_time = expected['expected_peak_time'],
        peak_force = expected['expected_peak_value'],
        threshold = expected['expected_threshold'],
        response_hand = "right"
    )
    
    # 1. Check that result is None
    assert result is None