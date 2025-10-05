# Third-Party Dependencies
import pandas as pd
import numpy as np
import pytest
# Local Application Imports
from data_loader import load_signal


def test_load_signal(mock_file_factory):
    """Tests the core functionality of the load_signal function."""

    mock_content = [
        "ChannelTitle=\tforce_right\tforce_left",
        "0\t-4.895415\t-0.8690986#*block_start",
        "0.0005\t-4.895415\t-0.8690986#*stimulus_right",
        "0.001\t-4.895415\t-0.8690986#*response_right",
        "0.0015\t-5.12345\t-0.912345",
        "0.002\t-4.895415\t-0.8690986#*stimulus_left",
        "0.0025\t-6.0\t-1.0#*response_none",
    ]
    
    # Use fixture to create temp file
    mock_filepath = mock_file_factory(mock_content, 'mock_data.txt')
    # Run function under test
    with pytest.warns(UserWarning, match = "Automatically added 'time' column"):
        df, comment_summary = load_signal(filepath = mock_filepath)

    # Assertions
    # ------------
    
    # 1. Check output type
    assert isinstance(df, pd.DataFrame)
    assert isinstance(comment_summary, dict)

    # 2. Check for presence of core columns
    expected_cols = ["time", "force_right", "force_left", "comments", 
                     "is_block_start", "block_number", "is_trial_start", "trial_number"]
    assert set(expected_cols).issubset(df.columns)

    # 3. Check data types
    assert pd.api.types.is_numeric_dtype(df['time'])
    assert pd.api.types.is_numeric_dtype(df['force_right'])
    assert pd.api.types.is_bool_dtype(df['is_trial_start'])
    
    # 4. Check comment summary
    assert comment_summary["block_start"] == 1
    assert comment_summary["stimulus_right"] == 1
    
    # 5. Check dimensions
    assert len(df) == 6
    
    # 6. Check specific values
    assert df.loc[0, 'comments'] == "block_start"
    assert df.loc[0, 'is_block_start']
    assert df.loc[0, 'block_number'] == 1

    assert df.loc[1, 'is_trial_start']
    assert df.loc[1, 'trial_number'] == 1
    
    assert df.loc[3, 'comments'] == ''
    assert df.loc[3, 'trial_number'] == 1 # Value carried forward

    assert df.loc[4, 'trial_number'] == 2 # Incremented by new stimulus
