# tests/test_trial_segmentation.py

import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

# Import all the functions we are testing from the module
from trial_segmentation import (
    create_trial_lookup, 
    get_trial_segment, 
    analyze_trial_metrics, 
    get_trial_data_and_metrics
)

# Import the helper functions/fixtures from conftest
from conftest import create_mock_trial_lookup, create_mock_signal_data, create_mock_condition_data


def test_get_trial_data_and_metrics_integration(mock_trial_lookup):
    """
    Integration test for get_trial_data_and_metrics, verifying the full process
    of segmentation and analysis for a single trial.
    """
    #  Define Mock Data & Parameters ---
    stim_time_val = 1.5
    mvc_l_val = 200.0
    mvc_r_val = 210.0
    pre_window_val = 0.125
    post_window_val = 1.25
    
    # Create mock raw data with custom column names and a "left" dominant force
    mock_force_df, expected_metrics = create_mock_signal_data(
        stim_time_within_segment = stim_time_val,
        dominant_force = "left", # To test that response_hand is correctly identified
        motor_condition = "high",
        mvc = mvc_l_val, # MVC for the dominant hand
        force_r_col_name = "OriginalForceR",
        force_l_col_name = "OriginalForceL",
        emg_r_col_name = "OriginalEMGR",
        emg_l_col_name = "OriginalEMGL"
    )
    
    # Define the mapping from standard names to the file's custom names
    channel_map = {
        'time': 'time',
        'force_right': 'OriginalForceR',
        'force_left': 'OriginalForceL',
        'emg_right': 'OriginalEMGR',
        'emg_left': 'OriginalEMGL'
    }

    # Create a custom condition data for this test that has "highForce" for Block 1
    custom_condition_data = create_mock_condition_data(
        block_motor_conditions=["highForce", "lowForce"]
    )

    # Call the function under test ---
    trial_segment_df, trial_metrics = get_trial_data_and_metrics(
        full_df = mock_force_df,
        trial_lookup = mock_trial_lookup,
        condition_data = custom_condition_data, # Use the custom data
        trial_index = 1, # Test the first global trial
        channel_map = channel_map,
        mvc_left = mvc_l_val,
        mvc_right = mvc_r_val,
        pre_window = pre_window_val,
        post_window = post_window_val
    )

    # 1. Check the returned types
    assert isinstance(trial_segment_df, pd.DataFrame)
    assert isinstance(trial_metrics, dict)
    
    # 2. Check metadata values in the metrics dictionary
    assert trial_metrics['participant_id'] == "p01_test"
    assert trial_metrics['global_index'] == 1
    assert trial_metrics['block'] == 1
    assert trial_metrics['stim_time'] == pytest.approx(stim_time_val)
    assert trial_metrics['cognitive_demand'] == "congruent"
    assert trial_metrics['motor_demand'] == "highForce" # Corrected assertion
    assert trial_metrics['response_hand'] == "left"
    
    # 3. Check threshold calculation: motor="high", hand="left"
    expected_threshold = 0.30 * mvc_l_val
    assert trial_metrics['threshold'] == pytest.approx(expected_threshold)

    # 4. Check the sliced signal DataFrame
    assert "force_left" not in trial_segment_df.columns
    assert "OriginalForceL" in trial_segment_df.columns

    # 5. Check dimensions of the sliced signal DataFrame
    time_increment = 1 / 500 # From create_mock_signal_data default
    expected_rows = int((pre_window_val + post_window_val) / time_increment) + 1
    assert len(trial_segment_df) == expected_rows


# ==============================================================================
# Unit test for get_trial_segment function
# ==============================================================================
def test_get_trial_segment_slicing():
    """Unit test to ensure get_trial_segment slices the correct time window."""
    
    # Create a simple time series DataFrame
    full_df = pd.DataFrame({'time': np.arange(0, 10, 0.1)}) # 10s of data at 10Hz
    
    stim_time = 5.0
    pre_window = 1.0
    post_window = 2.0

    result_df = get_trial_segment(full_df, stim_time, 'time', pre_window, post_window)

    # Assertions
    # ------------
    
    # The first time point should be stim_time - pre_window
    assert result_df['time'].min() == pytest.approx(4.0)
    # The last time point should be stim_time + post_window
    assert result_df['time'].max() == pytest.approx(7.0)
    # Check the total number of points
    assert len(result_df) == 31