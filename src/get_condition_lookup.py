import pandas as pd
import warnings
from typing import Dict, Any, Optional

def get_condition_lookup(data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Extracts participant ID and summarizes block counts by condition.

    Attempts to flexibly match common column names for participant ID,
    cognitive demand, and motor demand using partial, case-insensitive
    matching.

    Args:
        data: A pandas DataFrame containing at least participant ID,
              cognitive demand, and motor demand columns.

    Returns:
        A dictionary with:
        - 'participant_id': The unique participant ID (with a warning if multiple are found).
        - 'condition_counts': A DataFrame summarizing block counts per condition.
        Returns None if required columns cannot be found.
    """

    # Helper function for flexible column matching using partial strings
    def find_col(possible_substrings: list, df: pd.DataFrame) -> Optional[str]:
        """
        Finds the first column in a DataFrame whose name (case-insensitively)
        contains any of the given substrings.

        Args:
            possible_substrings: A list of strings to search for in column names.
            df: The DataFrame to search.

        Returns:
            The name of the first matching column, or None if no match is found.
        """
        for col in df.columns:
            lower_col = col.lower()
            if any(sub in lower_col for sub in possible_substrings):
                return col
        return None

    # Find the required columns with partial matching
    participant_col = find_col(['part', 'id'], data)
    cognitive_col = find_col(['ment', 'cog'], data)
    motor_col = find_col(['phys', 'mot'], data)

    # Check if all required columns are present
    if not all([participant_col, cognitive_col, motor_col]):
        warnings.warn("Required columns (participant, cognitive, or motor) could not be found.")
        return None

    # Extract participant ID
    participant_ids = data[participant_col].unique()
    if len(participant_ids) > 1:
        warnings.warn("Multiple participant IDs detected, using the first one.")
    participant_id = participant_ids[0]

    # Summarize block count per condition
    summary_df = data.groupby([cognitive_col, motor_col]).size().reset_index(name='n_blocks')

    # Rename the columns to standard names
    summary_df.rename(columns={
        cognitive_col: 'cognitive_demand',
        motor_col: 'motor_demand'
    }, inplace=True)

    # Sort the results for consistency
    summary_df.sort_values(by=['cognitive_demand', 'motor_demand'], inplace=True)

    return {
        'participant_id': participant_id,
        'condition_counts': summary_df
    }
