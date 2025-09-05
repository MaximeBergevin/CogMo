import re
import pandas as pd
from typing import Optional

# ==============================================================================
# --- HELPER FUNCTION ---
# ==============================================================================

def create_trial_lookup(data: pd.DataFrame) -> Optional[pd.DataFrame]:
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


# ==============================================================================
# --- get_trial_segment ---
# ==============================================================================
def get_trial_segment(
    
        full_df: pd.DataFrame,
        trial_lookup: pd.DataFrame,
        condition_data: pd.DataFrame,
        trial_index: int,
        channel_map: dict,
        mvc_left: float,
        mvc_right: float,
        pre_window: float = 0.125,
        post_window: float = 1.25):
    """
        fetches a specific trial segment from the full dataset based on trial index.
            Args:
                full_df: The complete DataFrame containing all recorded data.
                trial_lookup: A DataFrame mapping global trial indices to block and trial numbers.
                trial_index: The global index of the trial to extract (1-based).
                channel_map: A dictionary mapping logical channel names to actual column names in full_df.
                            Expected keys include 'time', 'force_right', 'force_left', 'emg_right', 'emg_left'.
                pre_window: Time in seconds before the stimulus onset to include in the segment.
                post_window: Time in seconds after the stimulus onset to include in the segment.
        """ 
    

    # Fetch trial info using trial navigation system
    # -----------------------------------------------
    trial_info = trial_lookup.query(f'global_index == @trial_index')
    if trial_info.empty:
        raise ValueError(f"Trial index {trial_index} not found in trial lookup table.")

    block_numb = trial_info['block_number'].iloc[0]
    trial_numb = trial_info['trial_number'].iloc[0]

    stim_row_query = f"block_number == {block_numb} & trial_number == {trial_numb} & is_trial_start == True"
    stim_row = full_df.query(stim_row_query) #1-row pd.DataFrame based on stim_row_query


    # Map column names using channel_map
    # -----------------------------------
    time_col= channel_map.get('time')
    force_r_col = channel_map.get('force_right')
    force_l_col = channel_map.get('force_left')
    # Robust, even if non-existent or excluded, e.g., returns None (default)
    emg_r_col = channel_map.get('emg_right')
    emg_l_col = channel_map.get('emg_left')

    
    # Trial segmentation based on stimulus onset
    # -------------------------------------------
    if stim_row.empty:
        raise ValueError(f"Stimulus row could not be uniquely identified for block {block_numb}, trial {trial_numb}.")
    
    stim_time = stim_row[time_col].iloc[0]
    pre_window = 0.125 # in seconds
    post_window = 1.25 # in seconds
    time_mask = (
        (full_df[time_col] >= stim_time - pre_window) & 
        (full_df[time_col] <= stim_time + post_window)
    )
    trial_segment_df = full_df.loc[time_mask].copy()


    # Get actual and expected responses
    # ----------------------------------
    # Small helper to resolve column names case-insensitively
    def resolve_col(df, pattern: str) -> str | None:
        """ Finds the first column in a DataFrame that matches a regex pattern. """
        for col in df.columns:
            # re.search finds the pattern anywhere in the column name, ignoring case
            if re.search(pattern, col, re.IGNORECASE):
                return col
        return None

    peak_force_right = trial_segment_df[force_r_col].max()
    peak_force_left = trial_segment_df[force_l_col].max()
    response_hand = 'right' if peak_force_right > peak_force_left else 'left'

    # For expected response, use regex to find relevant column (e.g., stimulus_left, stimulus_right)
    # Note, for robustness, we exclude columns like 'stimulus_time' using a negative lookahead
    # and lookahead to ensure we capture directional terms
    expected_response_pattern = r"(?=.*(expected|response|direction|hand|stimulus(?!_time)))(?=.*(left|right|l|r))"
    expected_response_col = resolve_col(full_df, expected_response_pattern)
    expected_response = stim_row.get(expected_response_col).iloc[0] if expected_response_col else None


    # Get condition info and compute thresholds based on MVC
    # -------------------------------------------------------
    motor_pattern = 'motor|phys'
    cognitive_pattern = 'cognitive|cog|mental'
    id_pattern = 'part|id'

    motor_col = resolve_col(condition_data, motor_pattern)
    cog_col = resolve_col(condition_data, cognitive_pattern)
    id_col = resolve_col(condition_data, id_pattern)

    condition_row = condition_data.iloc[block_numb - 1]  # Assuming block_number are usually 1-indexed; Python is 0-indexed
    motor_demand = condition_row.get(motor_col) if motor_col else None
    cognitive_demand = condition_row.get(cog_col) if cog_col else None
    participant_id = condition_row.get(id_col) if id_col else None

    threshold = None
    # MVC value based on response_hand. Makes if statements cleaner
    mvc_value = mvc_right if response_hand == 'right' else mvc_left
    if motor_demand is not None and mvc_value is not None:
        try:
            # CASE 1: Treat motor_demand as a direct multiplier (e.g., 0.05 vs .30)
            multiplier = float(motor_demand)
            threshold = multiplier * mvc_value
        except (ValueError, TypeError):
            # CASE 2: motor_demand is a descriptive label (e.g., 'low' vs 'high')
            motor_demand_str = str(motor_demand).lower()
            if 'low' in motor_demand_str:
                threshold = 0.05 * mvc_value
            elif 'high' in motor_demand_str:
                threshold = 0.30 * mvc_value


    # Package all single-value metrics into a dictionary
    # --------------------------------------------------
    trial_metrics = {
        'participant_id': participant_id,
        'global_index': trial_index,
        'block': block_numb,
        'stim_time': stim_time,
        'expected_response': expected_response,
        'response_hand': response_hand,
        'cognitive_demand': cognitive_demand,
        'motor_demand': motor_demand,
        'threshold': threshold,
    }

    return trial_segment_df, trial_metrics

