import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import scipy.signal as signal


def _condition_tkeo(raw_signal: np.ndarray, fs: float, lp_cutoff: float = 50.0) -> np.ndarray:
    """
    Applies the Teager-Kaiser Energy Operator (TKEO) pipeline to condition raw EMG.
    
    The TKEO is used to improve the signal-to-noise ratio by accentuating the 
    instantaneous energy of the signal (both amplitude and frequency). This makes 
    the onset of muscle activity more distinct from background noise.

    Steps:
    1. Bandpass Filter: Isolates typical EMG frequency range (30-300Hz).
    2. TKEO: Calculates energy via x[n]^2 - (x[n-1] * x[n+1]).
    3. Rectification: Takes the absolute value.
    4. Lowpass Filter: Smooths the energy spikes into a usable envelope.
    """
    nyq = 0.5 * fs
    
    # ---  Digital Bandpass ---
    # Removes motion artifacts (<30Hz) and high-frequency noise (>300Hz).
    b_band, a_band = signal.butter(3, [30/nyq, 300/nyq], btype='band')
    filtered = signal.filtfilt(b_band, a_band, raw_signal)
    
    # --- TKEO Operation ---
    # The middle samples are calculated using their neighbors to detect energy shifts.
    tkeo_raw = filtered[1:-1]**2 - (filtered[:-2] * filtered[2:])
    
    # --- Rectification and Padding ---
    # Pad the array to restore the two samples lost during neighbor calculation.
    tkeo_rect = np.abs(tkeo_raw)
    tkeo_env = np.pad(tkeo_rect, (1, 1), mode='edge')
    
    # --- Smoothing (Envelope Generation) ---
    # A lowpass filter creates a smooth curve used for threshold-based detection.
    b_low, a_low = signal.butter(1, lp_cutoff/nyq, btype='low')
    tkeo_env = signal.filtfilt(b_low, a_low, tkeo_env)
    
    return tkeo_env


def calculate_dynamic_threshold(
    full_df: pd.DataFrame,
    channel_map: Dict[str, str],
    response_hand: str,
    duration_sec: float = 0.1,
    h_multiplier: float = 15.0
) -> float:
    """
    Calculates a trial-specific EMG threshold based on local noise levels.
    
    The function identifies the "quietest" window (lowest variance) of a specified 
    duration within the trial. The threshold is then defined as the mean of that 
    quiet window plus a standard deviation multiplier (h).
    """
    emg_col = channel_map.get(f"emg_{response_hand}")
    if emg_col is None or emg_col not in full_df.columns:
        return 999.0
    
    # Remove DC offset (zero-center the signal)
    raw_signal = full_df[emg_col].values.astype(float)
    raw_signal -= np.mean(raw_signal)

    # Determine sampling frequency
    time_col = full_df.columns[0]
    fs = 1.0 / np.mean(np.diff(full_df[time_col].values))
    
    # Generate TKEO envelope for threshold assessment
    envelope = _condition_tkeo(raw_signal, fs, lp_cutoff=50.0)
    
    # Search for the quietest window using rolling variance
    window_samples = int(duration_sec * fs)
    stride = 10 
    
    if window_samples > len(envelope):
        window_samples = len(envelope) // 4

    env_series = pd.Series(envelope)
    rolling_var = env_series.rolling(window_samples, step=stride).var()
    
    # Locate the end index of the window with the minimum variance
    quiet_end_idx = rolling_var.idxmin()
    if pd.isna(quiet_end_idx): 
        quiet_end_idx = window_samples
    
    # Extract the samples from the identified quiet window
    quiet_slice = envelope[int(quiet_end_idx) - window_samples : int(quiet_end_idx)]
    
    # Statistical Threshold: Mean + (h * Standard Deviation)
    mean_val = np.mean(quiet_slice)
    std_val = np.std(quiet_slice)
    calculated_threshold = mean_val + (h_multiplier * std_val)
    
    # Safety Floor: Prevents threshold from dropping below 0.5% of the trial peak
    trial_peak = np.max(envelope)
    final_threshold = max(calculated_threshold, trial_peak * 0.005)
    
    return float(final_threshold)


def find_emg_boundaries(
    signal_df: pd.DataFrame,
    channel_map: Dict[str, str],
    response_hand: str,
    stim_time: float,
    force_onset_time: float,
    force_offset_time: float,
    min_burst_ms: int,
    threshold_on: float, 
    threshold_off: float, 
) -> Tuple[Optional[float], Optional[float], float]:
    """
    Detects the onset and offset of the EMG burst relative to the stimulus and force.
    
    Logic:
    1. Onset: Searches backward from the moment force began to rise to find the 
       earliest preceding EMG activity above the 'on' threshold.
    2. Offset: Searches backward from the moment force ended to find where the 
       EMG energy envelope settles back below the 'off' threshold.
    3. Validation: Ensures the resulting burst meets a minimum duration requirement.
    """

    time = signal_df[signal_df.columns[0]].values
    fs = 1.0 / np.mean(np.diff(time))
    emg_col = channel_map.get(f"emg_{response_hand}")
    
    # Pre-processing and conditioning
    raw_signal = signal_df[emg_col].values.astype(float) - np.mean(signal_df[emg_col].values)
    envelope = _condition_tkeo(raw_signal, fs, lp_cutoff=50.0)

    # Define search boundaries (30ms post-stimulus to force offset)
    search_start = np.searchsorted(time, stim_time + 0.030)
    force_on_idx = np.searchsorted(time, force_onset_time)
    force_off_idx = np.searchsorted(time, force_offset_time)
    
    # Reject if no activity in the search window exceeds the threshold
    peak_idx = search_start + np.argmax(envelope[search_start:force_off_idx])
    if envelope[peak_idx] < threshold_on:
        return None, None, threshold_on

    # --- Onset Detection ---
    # Identifies activity before force onset, allowing for 25 ms gaps (bridges) in signal.
    above_on = (envelope > threshold_on).astype(int)
    gap_limit_on = int(0.025 * fs) # 25ms bridge
    
    pre_force_activity = np.where(above_on[search_start:force_on_idx] == 1)[0]
    
    if len(pre_force_activity) == 0:
        # Fallback: search forward if no activity exists before force onset
        fwd_activity = np.where(above_on[search_start:force_off_idx] == 1)[0]
        if len(fwd_activity) == 0: return None, None, threshold_on
        onset_idx = search_start + fwd_activity[0]
    else:
        # Backwards bridge logic: combine small gaps in activity into a single burst
        current_on = search_start + pre_force_activity[-1]
        for i in range(len(pre_force_activity) - 2, -1, -1):
            earlier = search_start + pre_force_activity[i]
            if (current_on - earlier) <= gap_limit_on:
                current_on = earlier
            else: break
        onset_idx = current_on

    # --- Offset Detection ---
    # Identifies the burst tail by checking for a 20ms window of stable low energy.
    above_off = (envelope > threshold_off).astype(int)
    above_off = (envelope > threshold_off).astype(int)
    win_off = int(0.020 * fs)
    
    final_offset_idx = force_off_idx # Default fallback
    
    # Iterate from force offset back toward the EMG onset
    for i in range(force_off_idx, onset_idx + win_off, -1):
        window = above_off[i - win_off : i]
        # Burst is considered finished when 50% of the window is below threshold
        if np.mean(window) >= 0.5:
            final_offset_idx = i
            break

    # --- Final Burst Validation ---
    duration_ms = (time[final_offset_idx] - time[onset_idx]) * 1000
    if duration_ms < min_burst_ms:
        return None, None, threshold_on

    return time[onset_idx], time[final_offset_idx], threshold_on

def calculate_emg_rms(
    full_df: pd.DataFrame,
    channel_map: Dict[str, str],
    response_hand: str,
    onset_time: float,
    offset_time: float
) -> Optional[float]:
    """
    Computes the Root Mean Square (RMS) of the raw EMG signal.
    
    RMS is a measure of the signal's power. It is calculated over the specific 
    detected burst duration (from onset to offset).
    """
    emg_col = channel_map.get(f"emg_{response_hand}")
    time_col = full_df.columns[0]
    
    if emg_col is None or onset_time is None or offset_time is None:
        return None

    # Isolate the segment of raw EMG between the detected boundaries
    segment = full_df.loc[
        (full_df[time_col] >= onset_time) & (full_df[time_col] <= offset_time),
        emg_col
    ].values.astype(float)

    if len(segment) == 0: return None

    # RMS Formula: sqrt( mean( x^2 ) )
    return float(np.sqrt(np.mean(np.square(segment))))


def premotor_reaction_time(stim_time: float, emg_onset_time: Optional[float]) -> Optional[int]:
    """
    Calculates the latency between stimulus and muscle activation in milliseconds.
    """
    if emg_onset_time is None or emg_onset_time < stim_time:
        return None
    return int(round((emg_onset_time - stim_time) * 1000))