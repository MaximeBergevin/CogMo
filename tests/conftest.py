# Standard Library Imports
from typing import List, Callable, Tuple, Dict, Any
from pathlib import Path
# Third-Party Dependencies
import numpy as np
import pandas as pd
import pytest
from scipy.signal import butter, filtfilt


def generate_band_limited_emg(fs: int, n_samples: int, snr_db: float, freq_range=(80, 120)) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates background noise and a band-limited stochastic process for sEMG.
    """
    # Background Noise (Gaussian, zero mean, low variance)
    sigma_noise = 0.01 
    noise = np.random.normal(0, sigma_noise, n_samples)
    
    # Muscle Activity (Band-limited stochastic process)
    raw_samples = np.random.normal(0, 1, n_samples)
    nyq = 0.5 * fs
    # Guard against Nyquist frequency for different sampling rates
    low_cut = min(freq_range[0], nyq * 0.8)
    high_cut = min(freq_range[1], nyq * 0.9)
    
    b, a = butter(4, [low_cut / nyq, high_cut / nyq], btype='band')
    emg_stochastic = filtfilt(b, a, raw_samples)
    
    # Scale for SNR: sigma_signal = sqrt(noise_var * 10^(SNR/10))
    sigma_signal = np.sqrt((sigma_noise**2) * (10**(snr_db / 10)))
    if np.std(emg_stochastic) > 0:
        emg_stochastic = (emg_stochastic / np.std(emg_stochastic)) * sigma_signal
    
    return noise, emg_stochastic

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
    global_indices = np.arange(1, n_trials_total + 1)

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
    expected_resp_at_stim: str = "right",
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
    burst_time_s: float = 0.2,
    snr_db: float = 15.0,
    emd_s: float = 0.05 
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Simulates force data and band-limited sEMG with Electromechanical Delay.
    """
    time_increment = 1 / sampling_rate_hz
    n_time_points = int(total_duration_s / time_increment) + 1
    time_points = np.linspace(0, total_duration_s, n_time_points)

    # Base DataFrame
    df = pd.DataFrame({
        "time": time_points,
        force_r_col_name: np.random.uniform(-max_noise, max_noise, n_time_points),
        force_l_col_name: np.random.uniform(-max_noise, max_noise, n_time_points),
        "is_trial_start": False,
        "block_number": block_num_of_segment,
        "trial_number": trial_num_at_stim,
        "expected_response": expected_resp_at_stim
    })

    # Timing logic for Force Burst
    stim_row_index = (df['time'] - stim_time_within_segment).abs().idxmin()
    df.loc[stim_row_index, 'is_trial_start'] = True
    stim_time_exact = df.loc[stim_row_index, 'time']

    onset_index = min(stim_row_index + int(delay_s / time_increment), n_time_points - 1)
    burst_in_points = int(burst_time_s / time_increment)
    end_index = min(onset_index + burst_in_points, n_time_points - 1)
    force_burst_indices = np.arange(onset_index, end_index + 1)

    # sEMG Generation (Stochastic Band-limited Model with EMD)
    if include_emg:
        # EMG burst starts earlier than force by the EMD value
        emd_points = int(emd_s / time_increment)
        emg_burst_indices = np.maximum(0, force_burst_indices - emd_points)

        for col in [emg_r_col_name, emg_l_col_name]:
            noise, emg_stochastic = generate_band_limited_emg(sampling_rate_hz, n_time_points, snr_db)
            
            if (col == emg_r_col_name and dominant_force == "right") or \
               (col == emg_l_col_name and dominant_force == "left"):
                noise[emg_burst_indices] += emg_stochastic[emg_burst_indices]
            
            df[col] = np.abs(noise) 

    # 4. Force Burst (triangular)
    if motor_condition == "low":
        threshold = 0.05 * mvc
        peak_force_magnitude = mvc * 0.20 if overshoot else mvc * 0.03
    else:
        threshold = 0.30 * mvc
        peak_force_magnitude = mvc * 0.50 if overshoot else mvc * 0.25
    
    n_points = len(force_burst_indices)
    burst_vals = np.array([])
    if n_points > 1:
        half = int(np.ceil(n_points / 2))
        ramp_up = np.linspace(0, peak_force_magnitude, half)
        ramp_down = np.linspace(peak_force_magnitude, 0, n_points - half)
        burst_vals = np.concatenate([ramp_up, ramp_down])

        dominant_col = force_r_col_name if dominant_force == "right" else force_l_col_name
        df.loc[force_burst_indices, dominant_col] = burst_vals
        df[dominant_col] += shift_baseline

    # Calculate Metrics
    expected_auc = 0.0
    expected_auc_normalized = 0.0
    if len(burst_vals) > 1:
        # Ground truth AUC of the clean triangular burst
        expected_auc = np.trapezoid(y=burst_vals, dx=time_increment)
        
        # Ground truth Normalized AUC (%MVC/s)
        # Formula: (AUC / Duration) / MVC * 100
        if mvc > 0 and burst_time_s > 0:
            expected_auc_normalized = (expected_auc / burst_time_s) / mvc * 100

    expected_mean_force = np.mean(burst_vals) if n_points > 0 else 0.0
    
    expected_metrics = {
        "stim_row_index": stim_row_index,
        "stim_time_exact": stim_time_exact,
        "expected_peak_value": peak_force_magnitude,
        "expected_onset_time": df.loc[onset_index, 'time'] if onset_index < n_time_points else np.nan,
        "expected_force_onset": df.loc[onset_index, 'time'] if onset_index < n_time_points else np.nan, 
        "expected_peak_time": df.loc[onset_index + half - 1, 'time'] if n_points > 0 else np.nan,
        "expected_offset_time": df.loc[end_index, 'time'] if end_index < n_time_points else np.nan,
        "expected_emg_onset": df.loc[emg_burst_indices[0], 'time'] if include_emg else np.nan,
        "expected_threshold": threshold,
        "expected_auc": expected_auc,
        "expected_auc_normalized": expected_auc_normalized,
        "expected_mean_force": expected_mean_force,
        "expected_mean_force_pct": (expected_mean_force / mvc) * 100 if mvc > 0 else 0
    }

    return df, expected_metrics

@pytest.fixture
def mock_signal_data_factory() -> Callable:
    return create_mock_signal_data
