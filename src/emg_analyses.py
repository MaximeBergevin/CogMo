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
    # We shift the array to compute the operator across the whole signal
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
    duration_sec: float = 0.1,  # Now 100ms instead of 1.0s
    h_multiplier: float = 15.0  # Reset to Solnik standard
) -> float:
    """
    Finds the quietest 100ms window within the current trial to set a local baseline.
    """
    emg_col = channel_map.get(f"emg_{response_hand}")
    if emg_col is None or emg_col not in full_df.columns:
        return 999.0
    
    raw_signal = full_df[emg_col].values.astype(float)
    raw_signal -= np.mean(raw_signal)  # Remove DC offset
    
    time_col = full_df.columns[0]
    fs = 1.0 / np.mean(np.diff(full_df[time_col].values))
    
    # Process through TKEO pipeline (using 50Hz for baseline detection)
    envelope = _condition_tkeo(raw_signal, fs, lp_cutoff=50.0)
    
    # Search for Quietest 100ms Window
    window_samples = int(duration_sec * fs)
    stride = 10  # Smaller stride for higher precision in a local trial
    
    if window_samples > len(envelope):
        window_samples = len(envelope) // 4

    env_series = pd.Series(envelope)
    # Finding the window with the lowest variance ensures we avoid the burst
    rolling_var = env_series.rolling(window_samples, step=stride).var()
    
    quiet_end_idx = rolling_var.idxmin()
    if pd.isna(quiet_end_idx): 
        quiet_end_idx = window_samples
    
    quiet_slice = envelope[int(quiet_end_idx) - window_samples : int(quiet_end_idx)]
    
    # Mean + 15*SD
    mean_val = np.mean(quiet_slice)
    std_val = np.std(quiet_slice)
    calculated_threshold = mean_val + (h_multiplier * std_val)
    
    # Safety Floor: 0.5% of trial peak
    trial_peak = np.max(envelope)
    final_threshold = max(calculated_threshold, trial_peak * 0.005)
    
    return float(final_threshold)

def find_emg_boundaries(
    signal_df: pd.DataFrame,
    channel_map: Dict[str, str],
    response_hand: str,
    stim_time: float,
    force_offset_time: float,
    min_burst_ms: int,
    threshold: float,  # This is the 'final_threshold' (Max of 15SD or 0.5% Peak)
) -> Tuple[Optional[float], Optional[float], float]:
    """
    Detects EMG onset and offset using TKEO with a Force-Anchored Lookahead.
    Bridges mid-burst dips by checking if energy returns to 'threshold' before force ends.
    """
    time = signal_df[signal_df.columns[0]].values
    fs = 1.0 / np.mean(np.diff(time))
    emg_col = channel_map.get(f"emg_{response_hand}")
    
    # 1. Conditioning Pipeline
    raw_signal = signal_df[emg_col].values.astype(float) - np.mean(signal_df[emg_col].values)
    # Using 50Hz low-pass for the envelope to maintain ballistic sharpness
    envelope = _condition_tkeo(raw_signal, fs, lp_cutoff=50.0)

    # 2. Define Search Windows
    win_size = int(0.010 * fs) # 10ms onset window
    search_start = np.searchsorted(time, stim_time + 0.030)
    end_idx = np.searchsorted(time, force_offset_time)
    
    # 3. Detect Onset (80% Density above Threshold)
    above = (envelope > threshold).astype(int)
    check_on = np.convolve(above[search_start:end_idx], np.ones(win_size), mode='valid')
    onsets = np.where(check_on >= (win_size * 0.8))[0]

    if len(onsets) == 0: 
        return None, None, threshold
    
    onset_idx = search_start + onsets[0]
    
    # Backward search to the 'foot' of the rise for precision
    while onset_idx > search_start and envelope[onset_idx] > (threshold * 0.5):
        onset_idx -= 1
        
    # 4. Detect Offset with Dip-Bridging Lookahead
    off_win = int(0.040 * fs) # 40ms silence window
    peak_idx = onset_idx + np.argmax(envelope[onset_idx:end_idx])
    
    # Identify all points below threshold
    below = (envelope < threshold).astype(int)
    check_off = np.convolve(below[peak_idx:end_idx], np.ones(off_win), mode='valid')
    offset_candidates = np.where(check_off >= (off_win * 0.9))[0]
    
    # Default to force offset if no clear EMG offset is found
    final_offset_idx = end_idx 

    for cand in offset_candidates:
        candidate_idx = peak_idx + cand
        
        # BRIDGE LOGIC: Look ahead from this candidate to the Force Offset
        lookahead_zone = envelope[candidate_idx:end_idx]
        
        if len(lookahead_zone) == 0:
            final_offset_idx = candidate_idx
            break
            
        # If the energy NEVER pops back above the threshold before force ends, 
        # then this candidate is the true offset.
        # If it DOES pop back up, we ignore this candidate and keep looking.
        if not np.any(lookahead_zone > threshold):
            final_offset_idx = candidate_idx
            break

    return time[onset_idx], time[final_offset_idx], threshold

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