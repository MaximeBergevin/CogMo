import re
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# ==============================================================================
# --- HELPER & LOOKUP FUNCTIONS ---
# ==============================================================================

def create_trial_lookup(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Builds a trial lookup table to map a global trial index to a block/trial number."""
    if 'is_trial_start' not in data.columns:
        return None
    unique_trials = data[data['is_trial_start'] == True].drop_duplicates(
        subset=['block_number', 'trial_number']
    ).copy()
    unique_trials['global_index'] = range(1, len(unique_trials) + 1)
    return unique_trials[['global_index', 'block_number', 'trial_number']].reset_index(drop=True)

def _resolve_col(df: pd.DataFrame, pattern: str) -> Optional[str]:
    """Finds the first column in a DataFrame that matches a regex pattern, case-insensitively."""
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    return None


def get_trial_segment(
    full_df: pd.DataFrame, 
    stim_time: float, 
    time_col: str, 
    pre_window: float, 
    post_window: float
) -> pd.DataFrame:
    """
    MODULAR FUNCTION 1:
    Slices a segment of raw data from the full DataFrame around the stimulus time.
    Its ONLY job is to slice the data.
    """
    time_mask = (
        (full_df[time_col] >= stim_time - pre_window) & 
        (full_df[time_col] <= stim_time + post_window)
    )
    return full_df.loc[time_mask].copy()

def analyze_trial_metrics(
    trial_segment_df: pd.DataFrame,
    stim_row: pd.DataFrame,
    condition_row: pd.DataFrame,
    trial_info: Dict[str, Any],
    channel_map: Dict[str, str],
    mvc_left: float,
    mvc_right: float
) -> Dict[str, Any]:
    """
    MODULAR FUNCTION 2:
    Calculates all single-value metrics from a given trial segment and related info.
    Its ONLY job is to perform analysis.
    """
    # Map channel names
    force_r_col = channel_map.get('force_right')
    force_l_col = channel_map.get('force_left')

    # Get actual and expected responses
    peak_force_right = trial_segment_df[force_r_col].max()
    peak_force_left = trial_segment_df[force_l_col].max()
    response_hand = 'right' if peak_force_right > peak_force_left else 'left'
    
    expected_pattern = r"(?=.*(expected|response|direction|hand|stimulus(?!_time)))(?=.*(left|right|l|r))"
    expected_col = _resolve_col(stim_row, expected_pattern)
    expected_response = stim_row[expected_col].iloc[0] if expected_col else None

    # Get condition info
    motor_col = _resolve_col(condition_row.to_frame().T, 'motor|phys')
    cog_col = _resolve_col(condition_row.to_frame().T, 'cognitive|cog|mental')
    id_col = _resolve_col(condition_row.to_frame().T, 'part|id')

    motor_demand = condition_row.get(motor_col) if motor_col else None
    cognitive_demand = condition_row.get(cog_col) if cog_col else None
    participant_id = condition_row.get(id_col) if id_col else None

    # Compute threshold
    threshold = None
    mvc_value = mvc_right if response_hand == 'right' else mvc_left
    if motor_demand is not None and mvc_value is not None:
        try:
            multiplier = float(motor_demand)
            threshold = multiplier * mvc_value
        except (ValueError, TypeError):
            motor_demand_str = str(motor_demand).lower()
            if 'low' in motor_demand_str:
                threshold = 0.05 * mvc_value
            elif 'high' in motor_demand_str:
                threshold = 0.30 * mvc_value
    
    # Package metrics into a dictionary
    trial_metrics = {
        'participant_id': participant_id,
        'global_index': trial_info['trial_index'],
        'block': trial_info['block_numb'],
        'stim_time': trial_info['stim_time'],
        'expected_response': expected_response,
        'response_hand': response_hand,
        'cognitive_demand': cognitive_demand,
        'motor_demand': motor_demand,
        'threshold': threshold,
    }
    return trial_metrics

# ==============================================================================
# --- MAIN FUNCTION TO BE CALLED BY THE APP ---
# ==============================================================================

def get_trial_data_and_metrics(
    full_df: pd.DataFrame,
    trial_lookup: pd.DataFrame,
    condition_data: pd.DataFrame,
    trial_index: int,
    channel_map: Dict[str, str],
    mvc_left: float,
    mvc_right: float,
    pre_window: float,
    post_window: float
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    1. Get the slice of data for the USER'S VIEW.
    2. Gather the BASE METRICS (stim_time, threshold, etc.) needed to START the main analysis pipeline.
    """
    # 1. Find the trial and its essential info
    trial_info_row = trial_lookup.query(f'global_index == @trial_index')
    if trial_info_row.empty:
        raise ValueError(f"Trial index {trial_index} not found.")
    
    block_numb = int(trial_info_row['block_number'].iloc[0])
    trial_numb = int(trial_info_row['trial_number'].iloc[0])

    stim_row = full_df.query(f"block_number == {block_numb} & trial_number == {trial_numb} & is_trial_start == True")
    if stim_row.empty:
        raise ValueError(f"Stimulus row not found for block {block_numb}, trial {trial_numb}.")
    
    time_col = channel_map.get('time')
    stim_time = stim_row[time_col].iloc[0]
    
    # 2. Get the user's visualization DataFrame using the view parameters
    trial_view_df = get_trial_segment(full_df, stim_time, time_col, pre_window, post_window)

    # 3. Gather the initial "base_metrics" needed for the main analysis
    condition_row = condition_data.iloc[block_numb - 1]
    
    motor_col = _resolve_col(condition_row.to_frame().T, 'motor|phys')
    cog_col = _resolve_col(condition_row.to_frame().T, 'cognitive|cog|mental')
    id_col = _resolve_col(condition_row.to_frame().T, 'part|id')
    
    motor_demand = condition_row.get(motor_col) if motor_col else None
    
    # This is a naive first guess for response_hand, which will be overwritten by the smart analysis.
    # It's only used here to calculate an initial threshold.
    force_r_col = channel_map.get('force_right')
    force_l_col = channel_map.get('force_left')
    # BUG FIX: Use the sliced trial_view_df, not the full_df, to get a local max.
    initial_response_hand = 'right' if trial_view_df[force_r_col].max() > trial_view_df[force_l_col].max() else 'left'
    
    threshold = None
    mvc_value = mvc_right if initial_response_hand == 'right' else mvc_left
    if motor_demand is not None and mvc_value is not None:
        try:
            multiplier = float(motor_demand)
            threshold = multiplier * mvc_value
        except (ValueError, TypeError):
            motor_demand_str = str(motor_demand).lower()
            if 'low' in motor_demand_str:
                threshold = 0.05 * mvc_value
            elif 'high' in motor_demand_str:
                threshold = 0.30 * mvc_value

    base_metrics = {
        'participant_id': condition_row.get(id_col) if id_col else None,
        'global_index': trial_index,
        'block': block_numb,
        'stim_time': stim_time,
        'cognitive_demand': condition_row.get(cog_col) if cog_col else None,
        'motor_demand': motor_demand,
        'threshold': threshold,
    }

    return trial_view_df, base_metrics