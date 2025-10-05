# Standard Library Imports
from typing import List, Callable, Tuple, Dict, Any
from pathlib import Path
# Third-Party Dependencies
import numpy as np
import pandas as pd
import pytest




def create_mock_condition_data(
        participant_id: str = "p01_test",
        block_cognitive_conditions: List[str] = ['congruent', 'incongruent'],
        block_motor_conditions: List[str] = ['lowForce', 'highForce'],
) -> pd.DataFrame:
    """
    Creates a mock condition data DataFrame for testing.

    Args:
        participant_id: The participant ID to use.
        block_cognitive_conditions: Cognitive conditions for each block.
        block_motor_conditions: Motor conditions for each block.

    Returns:
        A pandas DataFrame with participant_id, cognitive_demand, and motor_demand.
    """
    n_blocks = len(block_cognitive_conditions)
    if len(block_cognitive_conditions) != n_blocks:
        raise ValueError("block_cognitive_conditions and block_motor_conditions must have the same length.")

    return pd.DataFrame({
        'participant_id': [participant_id] * n_blocks,
        'cognitive_demand': block_cognitive_conditions,
        'motor_demand': block_motor_conditions,
        'block_identifier': list(range(1, n_blocks + 1))
    })

@pytest.fixture
def mock_condition_data() -> pd.DataFrame:
    return create_mock_condition_data()


def create_mock_trial_lookup(
        n_blocks: int = 2,
        trials_per_block: int = 24) -> pd.DataFrame:
    """
    Creates a mock trial lookup table DataFrame for testing.

    Args:
        n_blocks: Total number of blocks.
        trials_per_block: Number of trials within each block.

    Returns:
        A pandas DataFrame with columns: global_index, block_number, trial_number.
    """
    n_trials_total = n_blocks * trials_per_block

    # Generate global trial indices from 1 to n_trials_total
    global_indices = np.arrange(1, n_trials_total + 1)

    # Calculate block numbers
    block_numbers = ((global_indices - 1) // trials_per_block) + 1

    # Calculate trial numbers
    trial_numbers = ((global_indices - 1) % trials_per_block) + 1

    # Create the DataFrame
    return pd.DataFrame({
        'global_index': global_indices,
        'block_number': block_numbers,
        'trial_number': trial_numbers
    })

@pytest.fixture
def mock_trial_lookup() -> pd.DataFrame:
    return create_mock_trial_lookup()


@pytest.fixture
def mock_file_factory(tmp_path: Path) -> Callable:
    """
    A pytest "factory fixture" for creating temporary files with content.
    
    This fixture returns a function that your tests can call to create
    a file in a temporary directory that pytest manages.

    Args:
        tmp_path: Pytest's built-in fixture for a temporary directory path.

    Returns:
        A function that can be called to create a mock file.
    """
    def _create_mock_file(content: List[str], filename: str) -> Path:
        """
        The actual function that creates the file.

        Args:
            content: A list of strings, where each string is a line in the file.
            filename: The name of the file to create (e.g., "test_data.csv").

        Returns:
            A pathlib.Path object pointing to the newly created file.
        """
        file_path = tmp_path / filename
        file_path.write_text("\n".join(content))
        return file_path

    return _create_mock_file


def create_mock_signal_data(
     total_duration_s: float = 3.0,
    sampling_rate_hz: int = 500,
    stim_time_within_segment: float = 1.5,
    block_num_of_segment: int = 1,
    trial_num_at_stim: int = 1,
    include_emg: bool = True,
    dominant_force: str = "right",
    motor_condition: str = "low",
    overshoot: bool = True,
    mvc: float = 400.0,
    force_r_col_name: str = "force_right",
    force_l_col_name: str = "force_left",
    emg_r_col_name: str = "emg_right",
    emg_l_col_name: str = "emg_left",
    max_noise: float = 0.5,
    shift_baseline: float = 0.0,
    delay_s: float = 0.5,
    burst_time_s: float = 0.2
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Generates a DataFrame simulating a segment of force_data for testing.
    Also calculates and returns the "ground truth" metrics for the generated data.
    """   
    # Time vector
    # ------------
    time_increment = 1 / sampling_rate_hz
    n_time_points = int(total_duration_s * time_increment) + 1
    time_points = np.linspace(0, total_duration_s, n_time_points)

    # Base DataFrame with noise
    # --------------------------
    data = {
        "time": time_points,
        force_r_col_name: np.random.uniform(-max_noise, max_noise, n_time_points),
        force_l_col_name: np.random.uniform(-max_noise, max_noise, n_time_points),
        "is_trial_start": False,
        "block_number": block_num_of_segment,
        "trial_number": np.nan,
        "expected_response": None
    }
    df = pd.DataFrame(data)

    if include_emg:
        df[emg_r_col_name] = np.random.uniform(0, 5, n_time_points)
        df[emg_l_col_name] = np.random.uniform(0, 4, n_time_points)

    #  Trial & Stimulus Info
    # -----------------------
    stim_row_index = (df['time'] - stim_time_within_segment).abs().idxmin()
    df.loc[stim_row_index, 'is_trial_start'] = True
    df.loc[stim_row_index, 'trial_number'] = trial_num_at_stim
    stim_time_exact = df.loc[stim_row_index, 'time']

    # Force burst generation
    # -----------------------
    delay_in_points = int(delay_s / time_increment)
    onset_index = min(stim_row_index + delay_in_points, n_time_points - 1)
    burst_in_points = int(burst_time_s / time_increment)
    end_index = min(onset_index + burst_in_points, n_time_points - 1)
    burst_indices = np.arange(onset_index, end_index + 1)

    # Determine threshold and peak force magnitude
    if motor_condition == "low":
        threshold = 0.05 * mvc
        peak_force_magnitude = mvc * 0.20 if overshoot else mvc * 0.03
    elif motor_condition == "high":
        threshold = 0.30 * mvc
        peak_force_magnitude = mvc * 0.50 if overshoot else mvc * 0.25
    else:
        raise ValueError("Invalid motor_condition. Use 'low' or 'high'.")
    
    # Create triangular burst shape
    n_points = len(burst_indices)
    burst_vals = np.array([])
    if n_points > 0:
        half = int(np.ceil(n_points / 2))
        ramp_up = np.linspace(0, peak_force_magnitude, half)
        ramp_down = np.linspace(peak_force_magnitude, 0, n_points - half)
        burst_vals = np.concatenate([ramp_up, ramp_down])

    # Inject the burst into the dominant hand's data
    if dominant_force in ["right", "left"]:
        dominant_col = force_r_col_name if dominant_force == "right" else force_l_col_name
        # Overwrite the noise in the burst region
        df.loc[burst_indices, dominant_col] = burst_vals
        # Add baseline shift
        df[dominant_col] += shift_baseline

    # Calculate 'ground truth' metrics for testing
    # ---------------------------------------------
    expected_metrics = {
        "stim_row_index": stim_row_index,
        "stim_time_exact": stim_time_exact,
        "expected_peak_value": peak_force_magnitude,
        "expected_onset_time": df.loc[onset_index, 'time'] if onset_index < n_time_points else np.nan,
        "expected_peak_time": df.loc[onset_index + half - 1, 'time'] if n_points > 0 else np.nan,
        "expected_threshold": threshold
        # Add any other ground truth metrics you calculate in the R function here...
    }

    return df, expected_metrics

@pytest.fixture
def mock_force_data_factory() -> Callable:
    return create_mock_signal_data
