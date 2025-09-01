# src/load_signal.py
import pandas as pd
from typing import Optional

def get_trial_lookup(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Builds a trial lookup table to map a global trial index to a specific block and trial number.

    Keeps the first row of each trial (marked by `is_trial_start`), preserves
    the original order of appearance, and assigns a `global_index` starting at 1.

    Args:
        data: A pandas DataFrame with at least the following columns:
              'block_number', 'trial_number', and 'is_trial_start' (boolean).

    Returns:
        A new DataFrame with columns:
        'global_index', 'block_number', 'trial_number'.
        Returns None if the 'is_trial_start' column is not found.
    """
    if 'is_trial_start' not in data.columns:
        return None

    # Filter for the start of each trial, then select unique block/trial combinations
    unique_trials = data[data['is_trial_start'] == True].drop_duplicates(
        subset=['block_number', 'trial_number']
    ).copy()

    # Assign a global trial index starting from 1
    unique_trials['global_index'] = range(1, len(unique_trials) + 1)

    # Reorder the columns for the final output
    lookup_table = unique_trials[['global_index', 'block_number', 'trial_number']]

    return lookup_table
