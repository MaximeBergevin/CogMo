import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
import scipy.signal as signal

def _condition_tkeo(raw_signal: np.ndarray, fs: float, lp_cutoff: float = 50.0) -> np.ndarray:
    """
    Conditions EMG signal using the Teager-Kaiser Energy Operator pipeline.
    
    1. Bandpass Filter (30-300Hz): Removes motion artifacts and high-frequency noise.
    2. TKEO: Amplifies energy based on both amplitude and frequency.
    3. Rectification: Ensures all energy values are positive.
    4. Lowpass Filter: Creates a smooth envelope for thresholding.
    """
    nyq = 0.5 * fs
    
    # 1. Digital Bandpass (30-300Hz) - Solnik et al. 6th order Butterworth
    b_band, a_band = signal.butter(3, [30/nyq, 300/nyq], btype='band')
    filtered = signal.filtfilt(b_band, a_band, raw_signal)
    
    # 2. TKEO Calculation: x[n]^2 - x[n-1]*x[n+1]
    tkeo_raw = filtered[1:-1]**2 - (filtered[:-2] * filtered[2:])
    
    # 3. Rectify & Pad
    # TKEO is rectified to ensure a positive energy envelope
    # Padding (1, 1) restores the 2 samples lost during neighbor calculation
    tkeo_rect = np.abs(tkeo_raw)
    tkeo_env = np.pad(tkeo_rect, (1, 1), mode='edge')
    
    # 4. Lowpass Smoothing (Envelope)
    # Using 20Hz instead of 50Hz to bridge gaps in ballistic reaction tasks
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
    Calculates a local TKEO threshold by finding the quietest 100ms window 
    within the trial and applying a user-defined SD multiplier (h).
    """
    emg_col = channel_map.get(f"emg_{response_hand}")
    if emg_col is None or emg_col not in full_df.columns:
        return 999.0
    
    # Pre-processing: Remove DC offset
    raw_signal = full_df[emg_col].values.astype(float)
    raw_signal -= np.mean(raw_signal)
    
    time_col = full_df.columns[0]
    fs = 1.0 / np.mean(np.diff(full_df[time_col].values))
    
    # Process through TKEO pipeline (50Hz for threshold detection)
    envelope = _condition_tkeo(raw_signal, fs, lp_cutoff=50.0)
    
    # Search for Quietest 100ms Window (Rolling Variance)
    window_samples = int(duration_sec * fs)
    stride = 10 
    
    if window_samples > len(envelope):
        window_samples = len(envelope) // 4

    env_series = pd.Series(envelope)
    rolling_var = env_series.rolling(window_samples, step=stride).var()
    
    quiet_end_idx = rolling_var.idxmin()
    if pd.isna(quiet_end_idx): 
        quiet_end_idx = window_samples
    
    quiet_slice = envelope[int(quiet_end_idx) - window_samples : int(quiet_end_idx)]
    
    # Calculate Mean + h*SD using user input
    mean_val = np.mean(quiet_slice)
    std_val = np.std(quiet_slice)
    calculated_threshold = mean_val + (h_multiplier * std_val)
    
    # Safety Floor: Ensure threshold is at least 0.5% of trial peak (should rarely trigger)
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
    time = signal_df[signal_df.columns[0]].values
    fs = 1.0 / np.mean(np.diff(time))
    emg_col = channel_map.get(f"emg_{response_hand}")
    
    # 1. Conditioning
    raw_signal = signal_df[emg_col].values.astype(float) - np.mean(signal_df[emg_col].values)
    envelope = _condition_tkeo(raw_signal, fs, lp_cutoff=50.0)

    # 2. Search Windows
    search_start = np.searchsorted(time, stim_time + 0.030)
    force_on_idx = np.searchsorted(time, force_onset_time)
    force_off_idx = np.searchsorted(time, force_offset_time)
    
    # Global Peak Check (Safety)
    peak_idx = search_start + np.argmax(envelope[search_start:force_off_idx])
    if envelope[peak_idx] < threshold_on:
        return None, None, threshold_on

    # 3. ONSET: Backward Search from Force Onset using THRESHOLD_ON
    above_on = (envelope > threshold_on).astype(int)
    gap_limit_on = int(0.025 * fs) # 25ms bridge
    
    pre_force_activity = np.where(above_on[search_start:force_on_idx] == 1)[0]
    
    if len(pre_force_activity) == 0:
        # Fallback if no activity found before force rise
        fwd_activity = np.where(above_on[search_start:force_off_idx] == 1)[0]
        if len(fwd_activity) == 0: return None, None, threshold_on
        onset_idx = search_start + fwd_activity[0]
    else:
        current_on = search_start + pre_force_activity[-1]
        for i in range(len(pre_force_activity) - 2, -1, -1):
            earlier = search_start + pre_force_activity[i]
            if (current_on - earlier) <= gap_limit_on:
                current_on = earlier
            else: break
        onset_idx = current_on

    # 4. OFFSET: Backward Search from Force Offset using THRESHOLD_OFF
    above_off = (envelope > threshold_off).astype(int)
    win_off = int(0.020 * fs) # 20ms stability window
    
    final_offset_idx = force_off_idx # Default
    
    # Iterate from force_offset back toward the onset
    for i in range(force_off_idx, onset_idx + win_off, -1):
        window = above_off[i - win_off : i]
        # If 50% of the window is above the strict threshold, we've found the burst tail
        if np.mean(window) >= 0.5:
            final_offset_idx = i
            break

    # 5. Validation
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
    """Computes RMS of the raw EMG signal between onset and offset times."""
    emg_col = channel_map.get(f"emg_{response_hand}")
    time_col = full_df.columns[0]
    
    if emg_col is None or onset_time is None or offset_time is None:
        return None

    segment = full_df.loc[
        (full_df[time_col] >= onset_time) & (full_df[time_col] <= offset_time),
        emg_col
    ].values.astype(float)

    if len(segment) == 0: return None
    return float(np.sqrt(np.mean(np.square(segment))))

def premotor_reaction_time(stim_time: float, emg_onset_time: Optional[float]) -> Optional[int]:
    """Calculates Premotor Reaction Time (Stimulus -> EMG Onset) in ms."""
    if emg_onset_time is None or emg_onset_time < stim_time:
        return None
    return int(round((emg_onset_time - stim_time) * 1000))