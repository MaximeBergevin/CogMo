import pandas as pd
import warnings
from typing import Dict, Any, Optional

def get_condition_lookup(data: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Extracts participant ID and summarizes unique conditions from a DataFrame.
    """
    def find_col(possible_substrings: list, df: pd.DataFrame) -> Optional[str]:
        for col in df.columns:
            if any(sub in col.lower() for sub in possible_substrings):
                return col
        return None

    participant_col = find_col(['part', 'id'], data)
    cognitive_col = find_col(['ment', 'cog'], data)
    motor_col = find_col(['phys', 'mot'], data)

    if not all([participant_col, cognitive_col, motor_col]):
        warnings.warn("Required columns (participant, cognitive, or motor) could not be found.")
        return None

    participant_ids = data[participant_col].unique()
    if len(participant_ids) > 1:
        warnings.warn("Multiple participant IDs detected, using the first one.")
    participant_id = participant_ids[0]

    # Get the unique sorted lists of conditions and the total number of blocks.
    cognitive_demand_list = sorted(data[cognitive_col].unique())
    motor_demand_list = sorted(data[motor_col].unique())
    n_blocks = len(data)

    return {
        'participant_id': participant_id,
        'condition_counts': {
            'cognitive_demand': cognitive_demand_list,
            'motor_demand': motor_demand_list,
            'n_blocks': n_blocks
        }
    }
