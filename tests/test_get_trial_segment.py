import pandas as pd
import numpy as np
import pytest
from pandas.testing import assert_frame_equal

# Import all the functions we are testing from the module
from trial_segmentation import (
    create_trial_lookup, 
    get_trial_segment, 
    get_trial_data_and_metrics
)

# Import helper functions/fixtures from conftest
from conftest import create_mock_trial_lookup, create_mock_signal_data, create_mock_condition_data

def test_create_trial_lookup():
    """
    Tests that create_trial_lookup correctly builds the lookup table from raw data.
    """
    n_blocks = 12
    n_trials_per_block = 24
    
    expected_lookup = create_mock_trial_lookup(n_blocks, n_trials_per_block)
    
    raw_data = expected_lookup[['block_number', 'trial_number']].copy()
    raw_data['is_trial_start'] = True
    
    distractor_rows = pd.DataFrame({
        'block_number': [1, 1],
        'trial_number': [1, 2],
        'is_trial_start': [False, False]
    })
    raw_data = pd.concat([raw_data, distractor_rows]).sort_index().reset_index(drop=True)

    result = create_trial_lookup(raw_data)
    
    assert isinstance(result, pd.DataFrame)
    assert_frame_equal(result, expected_lookup)

# ==============================================================================
# Test for the updated get_trial_data_and_metrics function ---
# ==============================================================================
def test_get_trial_data_and_metrics(mock_trial_lookup, mock_signal_data_factory, mock_condition_data):
    """
    Tests that get_trial_data_and_metrics correctly gathers base trial info
    and slices the user-defined view window.
    """
    #  --- Define Mock Data & Parameters ---
    stim_time_val = 1.5
    mvc_l_val = 200.0
    mvc_r_val = 210.0
    pre_window_val = 0.5 
    post_window_val = 1.0 
    
    mock_force_df, _ = mock_signal_data_factory(
        stim_time_within_segment=stim_time_val,
        force_r_col_name="force_right",
        force_l_col_name="force_left"
    )
    
    # Standard internal channel map convention
    channel_map = {
        'time': 'time',
        'force_right': 'force_right',
        'force_left': 'force_left'
    }

    # --- Call function under test ---
    trial_view_df, base_metrics = get_trial_data_and_metrics(
        full_df=mock_force_df,
        trial_lookup=mock_trial_lookup,
        condition_data=mock_condition_data,
        trial_index=1,
        channel_map=channel_map,
        mvc_left=mvc_l_val,
        mvc_right=mvc_r_val,
        pre_window=pre_window_val,
        post_window=post_window_val
    )

    #  Assertions
    # -----------
    
    # 1. Check returned types
    assert isinstance(trial_view_df, pd.DataFrame)
    assert isinstance(base_metrics, dict)
    
    # 2. Check that the base_metrics dictionary contains the correct essential keys.
    expected_keys = ['participant_id', 'global_index', 'block', 'stim_time', 
                     'cognitive_demand', 'motor_demand', 'threshold']
    assert all(key in base_metrics for key in expected_keys)
    assert 'response_hand' not in base_metrics 
    
    # 3. Check specific values in the base_metrics
    assert base_metrics['participant_id'] == "p01_test"
    assert base_metrics['global_index'] == 1
    assert base_metrics['stim_time'] == pytest.approx(stim_time_val)
    assert base_metrics['motor_demand'] == "lowForce"
    # Check that an initial threshold was calculated
    assert base_metrics['threshold'] is not None

    # 4. Check that the returned DataFrame (the user's view) was sliced correctly
    assert trial_view_df['time'].min() == pytest.approx(stim_time_val - pre_window_val)
    assert trial_view_df['time'].max() == pytest.approx(stim_time_val + post_window_val)


# ==============================================================================
# Unit test for get_trial_segment function
# ==============================================================================
def test_get_trial_segment_slicing():
    """Unit test to ensure get_trial_segment slices the correct time window."""
    
    # Create a simple time series DataFrame
    full_df = pd.DataFrame({'time': np.arange(0, 10, 0.1)}) 
    
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