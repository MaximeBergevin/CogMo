# tests/test_trial_segmentation.py

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

# Import the function we are testing
from trial_segmentation import create_trial_lookup

# Import the helper function from conftest to create test data
from conftest import create_mock_trial_lookup

def test_create_trial_lookup():
    """
    Tests that create_trial_lookup correctly builds the lookup table from raw data.
    """
    n_blocks = 12
    n_trials_per_block = 24
    
    # Create expected DataFrame that the function returns
    expected_lookup = create_mock_trial_lookup(n_blocks, n_trials_per_block)

    # Create mock raw_data DataFrame to use as INPUT for the function.
    raw_data = expected_lookup[['block_number', 'trial_number']].copy()
    raw_data['is_trial_start'] = True
    
    # ...and add some extra "distractor" rows that should be ignored by the function.
    distractor_rows = pd.DataFrame({
        'block_number': [1, 1],
        'trial_number': [1, 2],
        'is_trial_start': [False, False] # These rows should be filtered out
    })
    raw_data = pd.concat([raw_data, distractor_rows]).sort_index().reset_index(drop=True)

    result = create_trial_lookup(raw_data)

    # Assertions
    # ------------

    # 1. Output is a DataFrame and is equal to our expected lookup table
    assert isinstance(result, pd.DataFrame)
    assert_frame_equal(result, expected_lookup)