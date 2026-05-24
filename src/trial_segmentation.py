import re
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# ==============================================================================
# --- HELPER & LOOKUP FUNCTIONS ---
# ==============================================================================

def create_trial_lookup(data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Generates a master index mapping every trial to its respective block and number.
    
    This table allows the application to jump to any trial (e.g., Trial 45) by 
    calculating its 'global_index', regardless of which block it belongs to.
    """
    if 'is_trial_start' not in data.columns:
        return None
    # Isolate rows where a new trial begins and drop duplicates to get unique trial IDs
    unique_trials = data[data['is_trial_start'] == True].drop_duplicates(
        subset=['block_number', 'trial_number']
    ).copy()
    # Assign a continuous index (1 to N) for easy navigation
    unique_trials['global_index'] = range(1, len(unique_trials) + 1)
    return unique_trials[['global_index', 'block_number', 'trial_number']].reset_index(drop=True)


def _resolve_col(df: pd.DataFrame, pattern: str) -> Optional[str]:
    """
    Identifies a column name using regular expression (regex) patterns.
    
    This provides flexibility if the input data headers vary slightly (e.g., 
    'ParticipantID' vs 'part_id'). It searches case-insensitively.
    """
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
    Slices a specific time-window of data from the full session.
    
    This function isolates the raw signal surrounding a stimulus event based on 
    user-defined 'Pre-Stim' and 'Post-Stim' parameters.
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
    Extracts foundational metadata and demands for a specific trial.
    
    Logic:
    1. Determines initial responding hand based on peak force in the segment.
    2. Uses regex to find condition columns (Motor vs. Cognitive demand).
    3. Calculates the target force threshold based on MVC and motor demand.
    """
    # Map channel names
    force_r_col = channel_map.get('force_right')
    force_l_col = channel_map.get('force_left')

    # Determine hand based on simple maximum peak in this segment
    peak_force_right = trial_segment_df[force_r_col].max()
    peak_force_left = trial_segment_df[force_l_col].max()
    response_hand = 'right' if peak_force_right > peak_force_left else 'left'
    
    # Locate Expected Response column (e.g., 'expected_hand' or 'stimulus_left')
    expected_pattern = r"(?=.*(expected|response|direction|hand|stimulus(?!_time)))(?=.*(left|right|l|r))"
    expected_col = _resolve_col(stim_row, expected_pattern)
    expected_response = stim_row[expected_col].iloc[0] if expected_col else None

    # Resolve demand columns from the condition metadata
    motor_col = _resolve_col(condition_row.to_frame().T, 'motor|phys')
    cog_col = _resolve_col(condition_row.to_frame().T, 'cognitive|cog|mental')
    id_col = _resolve_col(condition_row.to_frame().T, 'part|id')

    motor_demand = condition_row.get(motor_col) if motor_col else None
    cognitive_demand = condition_row.get(cog_col) if cog_col else None
    participant_id = condition_row.get(id_col) if id_col else None

    # Threshold Calculation: MVC * Demand Percentage
    threshold = None
    mvc_value = mvc_right if response_hand == 'right' else mvc_left
    if motor_demand is not None and mvc_value is not None:
        try:
            # If demand is a direct multiplier (e.g., 0.15 for 15% MVC)
            multiplier = float(motor_demand)
            threshold = multiplier * mvc_value
        except (ValueError, TypeError):
            # Fallback for categorical demand labels
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

    trial_info_row = trial_lookup.query(f'global_index == @trial_index')
    if trial_info_row.empty:
        raise ValueError(f"Trial index {trial_index} not found.")
    
    block_numb = int(trial_info_row['block_number'].iloc[0])
    trial_numb = int(trial_info_row['trial_number'].iloc[0])

    stim_row = full_df.query(f"block_number == {block_numb} & trial_number == {trial_numb} & is_trial_start == True")
    if stim_row.empty:
        raise ValueError(f"Stimulus row not found for block {block_numb}, trial {trial_numb}.")
    
    expected_response = None
    if 'comments' in stim_row.columns:
        comment_text = str(stim_row['comments'].iloc[0]).lower()
        if 'right' in comment_text or ' stimulus_r' in comment_text:
            expected_response = 'right'
        elif 'left' in comment_text or ' stimulus_l' in comment_text:
            expected_response = 'left'
    
    time_col = channel_map.get('time')
    stim_time = stim_row[time_col].iloc[0]
    
    trial_view_df = get_trial_segment(full_df, stim_time, time_col, pre_window, post_window)

    # Resolve metadata from the condition spreadsheet
    condition_row = condition_data.iloc[block_numb - 1]
    
    motor_col = _resolve_col(condition_row.to_frame().T, 'motor|phys')
    cog_col = _resolve_col(condition_row.to_frame().T, 'cognitive|cog|mental')
    id_col = _resolve_col(condition_row.to_frame().T, 'part|id')
    
    motor_demand = condition_row.get(motor_col) if motor_col else None
    
    # Refactor line 180 to this:
    initial_response_hand = 'right' if trial_view_df['force_right'].max() > trial_view_df['force_left'].max() else 'left'   
    
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
        'expected_response': expected_response, 
        'cognitive_demand': condition_row.get(cog_col) if cog_col else None,
        'motor_demand': motor_demand,
        'threshold': threshold,
    }

    return trial_view_df, base_metrics