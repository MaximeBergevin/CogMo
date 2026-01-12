import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict

def _calculate_mcd(series: pd.Series) -> float:
    """Mean Consecutive Difference (MCD) for a pandas Series."""
    if series.empty:
        return 0.0
    return series.diff().abs().mean()


def calculate_emg_rms(
    full_df: pd.DataFrame,
    channel_map: Dict[str, str],
    response_hand: str,
    onset_time: float,
    offset_time: float
) -> Optional[float]:
    """
    Computes RMS of the EMG signal between onset and offset times.

    Parameters
    ----------
    full_df : pd.DataFrame
        Full signal dataframe containing time and EMG columns.
    channel_map : dict
        Channel mapping with 'emg_left'/'emg_right' keys.
    response_hand : str
        'left' or 'right'.
    onset_time : float
        EMG burst onset time (seconds).
    offset_time : float
        EMG burst offset time (seconds).

    Returns
    -------
    float or None
        RMS value of the EMG burst segment.
    """
    emg_col = channel_map.get(f"emg_{response_hand}")
    time_col = full_df.columns[0]
    if emg_col is None or emg_col not in full_df.columns:
        return None
    if onset_time is None or offset_time is None:
        return None

    segment = full_df.loc[
        (full_df[time_col] >= onset_time) & (full_df[time_col] <= offset_time),
        emg_col
    ].abs()

    if segment.empty:
        return None

    rms = np.sqrt(np.mean(np.square(segment)))
    return float(rms)


def get_emg_baseline_window(
    emg: pd.Series,
    time: np.ndarray,
    sampling_rate: float,
    start_time: float,
    end_time: float,
    window_ms: int = 25
) -> tuple[float, float]:
    """
    Finds the quietest (lowest MCD) window between start_time and end_time.

    This is a simplified and general version of the old baseline finder.
    It can be used for any time range, not just pre-stim baselines.

    Parameters
    ----------
    emg : pd.Series
        Rectified EMG signal.
    time : np.ndarray
        Time vector (seconds).
    sampling_rate : float
        Sampling frequency (Hz).
    start_time : float
        Beginning of the search range (s).
    end_time : float
        End of the search range (s).
    window_ms : int, default=25
        Duration of each candidate window in milliseconds.

    Returns
    -------
    (base_start, base_end) : tuple[float, float]
        Start and end times (in seconds) of the quietest window.
    """
    # Convert to samples
    window_samp = int(window_ms * sampling_rate / 1000)
    step_samp = max(1, window_samp // 2)  # half-overlap

    # Ensure valid range
    if end_time <= start_time:
        return time[0], time[0] + window_ms / 1000.0

    # Restrict to search segment
    mask = (time >= start_time) & (time <= end_time)
    emg_seg = emg[mask].reset_index(drop=True)
    time_seg = time[mask].reset_index(drop=True)

    if len(emg_seg) < window_samp:
        return start_time, start_time + window_ms / 1000.0

    # Find lowest MCD segment
    best_mcd = float("inf")
    best_start_idx = 0
    for i in range(0, len(emg_seg) - window_samp, step_samp):
        seg = emg_seg.iloc[i:i + window_samp]
        mcd_val = _calculate_mcd(seg)
        if mcd_val < best_mcd:
            best_mcd = mcd_val
            best_start_idx = i

    base_start = time_seg.iloc[best_start_idx]
    base_end = base_start + window_ms / 1000.0
    return base_start, base_end


def find_emg_burst_boundaries_from_peak(
    full_df: pd.DataFrame,
    channel_map: Dict[str, str],
    response_hand: str,
    stim_time: float,
    peak_time: float,
    threshold_sd_equiv_onset: float,
    threshold_sd_equiv_offset: float,
    min_duration_ms: int,
    rolling_window_ms: int,
    sampling_rate: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Detects EMG burst onset and offset using the MCD method with amplitude-domain thresholds
    and hysteresis (different constants for onset and offset).
    
    The reference window is a short active segment (e.g., 50 ms ending at peak_time),
    used to compute both mean rectified amplitude and MCD.
    """
    time_col = full_df.columns[0]
    emg_col = channel_map.get(f"emg_{response_hand}", None)
    if emg_col is None or emg_col not in full_df.columns:
        return None, None

    time = full_df[time_col].values
    emg = full_df[emg_col].abs()  # rectified EMG

    if sampling_rate is None:
        sampling_rate = 1.0 / np.mean(np.diff(time))

    # --- Rolling MCD of the rectified signal ---
    w = int(rolling_window_ms * sampling_rate / 1000)
    rolling_mcd = emg.rolling(window=w, center=True).apply(_calculate_mcd, raw=False)
    print("rolling window w:", w)


    # --- Reference window: 50 ms ending at peak_time (active EMG) ---
    ref_start = max(time[0], peak_time - 0.050)
    ref_end = peak_time
    ref_seg = emg[(time >= ref_start) & (time <= ref_end)]

    if len(ref_seg) < 3:
        return None, None

    ref_mean = ref_seg.mean()
    ref_mcd = _calculate_mcd(ref_seg)

    # --- Amplitude-domain thresholds (Garvey-style) ---
    MCD_MULTIPLIER_CONSTANT = 0.89
    mcd_multiplier_onset = threshold_sd_equiv_onset * MCD_MULTIPLIER_CONSTANT
    mcd_multiplier_offset = threshold_sd_equiv_offset * MCD_MULTIPLIER_CONSTANT
    lower_onset_threshold = ref_mean - (mcd_multiplier_onset * ref_mcd)
    lower_offset_threshold = ref_mean - (mcd_multiplier_offset * ref_mcd)
    print("Onset Threshold Value:", lower_onset_threshold)
    print("Offset Threshold Value:", lower_offset_threshold)

    # --- Find the index closest to the peak_time ---
    peak_idx = np.argmin(np.abs(time - peak_time))
    min_samples = int(min_duration_ms * sampling_rate / 1000)

    # --- Search left (onset): below threshold for sustained period ---
    left_idx = peak_idx
    below_count = 0
    while left_idx > 0:
        if emg.iloc[left_idx] < lower_onset_threshold:
            below_count += 1
            if below_count >= min_samples:
                break
        else:
            below_count = 0
        left_idx -= 1
    onset_time = time[max(0, left_idx)]

    # --- Search right (offset): below threshold for sustained period ---
    right_idx = peak_idx
    below_count = 0
    while right_idx < len(emg) - 1:
        if emg.iloc[right_idx] < lower_offset_threshold:
            below_count += 1
            if below_count >= min_samples:
                break
        else:
            below_count = 0
        right_idx += 1
    offset_time = time[min(right_idx, len(time) - 1)]

    # Clamp unrealistic pre-stim onsets
    if onset_time < stim_time:
        onset_time = stim_time

    return onset_time, offset_time, lower_onset_threshold

def premotor_reaction_time(
    stim_time: float,
    emg_onset_time: Optional[float]
) -> Optional[int]:
    """
    Calculates Premotor Reaction Time (Stimulus -> EMG Onset) in ms.
    """
    if emg_onset_time is None or emg_onset_time < stim_time:
        return None
    return int(round((emg_onset_time - stim_time) * 1000))