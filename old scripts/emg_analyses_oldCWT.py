import pandas as pd
import numpy as np
import pywt
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


def find_emg_boundaries(
    signal_df: pd.DataFrame,
    channel_map: Dict[str, str],
    response_hand: str,
    stim_time: float,
    force_offset_time: float,
    min_burst_ms: int,
    peak_fraction: int,
) -> Tuple[Optional[float], Optional[float]]:

    # Extract Time & EMG
    # --------------------
    time_col = signal_df.columns[0]
    time = signal_df[time_col].values

    emg_col = channel_map.get(f"emg_{response_hand}", None)
    if emg_col is None or emg_col not in signal_df.columns:
        return None, None

    emg = signal_df[emg_col].values.astype(float)
    emg_abs = emg

    #  Sampling Rate
    # ----------------
    dt = np.mean(np.diff(time))
    if dt <= 0:
        return None, None
    fs = 1.0 / dt

    #  Window for Burst Peak Search
    # ------------------------------
    start_idx = np.searchsorted(time, stim_time)
    end_idx   = np.searchsorted(time, force_offset_time)

    if end_idx <= start_idx:
        return None, None

    window_emg = emg_abs[start_idx:end_idx]
    if len(window_emg) == 0:
        return None, None

    # Peak inside the voluntary-contraction window
    local_peak_offset = np.argmax(window_emg)
    peak_idx = start_idx + local_peak_offset

    # Morlet CWT (Multi-Scale Envelope)
    # -------------------------------------
    # fc = 0.8125 for 'morl' wavelet
    # fc / (scales * dt) gives pseudo-frequencies
    scales = np.array([4, 8, 12, 16, 20, 24]) # 100 Hz - 33 Hz
    coeffs, _ = pywt.cwt(emg_abs, scales, 'morl')
    print(coeffs)
    len(coeffs)
    print(coeffs.shape)
    # Energy across scales (standard EMG envelope from CWT)
    energy = np.sum(coeffs ** 2, axis=0)

    

    #  Smooth Energy (10 ms Moving Average)
    # -------------------------------------
    window = int(0.01 * fs)
    if window > 1:
        kernel = np.ones(window) / window
        energy_smooth = np.convolve(energy, kernel, mode='same')
    else:
        energy_smooth = energy

    # Relative Thresholding
    # ----------------------------------
    peak_value = np.max(energy_smooth[start_idx:end_idx])
    frac = peak_fraction / 100.0

    onset_threshold  = peak_value * frac
    offset_threshold = peak_value * (frac * 0.75)

    # Duration Constraints
    # ---------------------
    onset_min_samples  = int((min_burst_ms / 1000.0) * fs)
    offset_min_samples = max(1, int(onset_min_samples * 0.50))


    #  Detect Onset
    # --------------
    search_start_time = stim_time + 0.03
    search_start_idx  = np.searchsorted(time, search_start_time)

    above = energy_smooth > onset_threshold
    above_search = above[search_start_idx:]

    onset_idx = None
    for i in range(0, len(above_search) - onset_min_samples):
        if np.all(above_search[i : i + onset_min_samples]):
            onset_idx = search_start_idx + i
            break

    if onset_idx is None:
        return None, None

    onset_time = time[onset_idx]

    # Detect Offset 
    # --------------
    offset_idx = None
    upper_bound = np.searchsorted(time, force_offset_time)

    below = energy_smooth < offset_threshold

    # If EMG rises again anywhere after a candidate offset,
    # treat it as a continuation of the same contraction.
    continuation_amp = np.max(window_emg) * 0.20

    for i in range(peak_idx, upper_bound - offset_min_samples):
        # Candidate offset: sustained low energy
        if np.all(below[i : i + offset_min_samples]):
            # Check for additional bursts later in the window
            remaining_emg = emg_abs[i:upper_bound]
            if np.any(remaining_emg > continuation_amp):
                continue  # later burst = keep searching
            offset_idx = i
            break

    # Fallback if no offset was detected
    if offset_idx is None:
        offset_idx = upper_bound

    offset_time = time[offset_idx]

    return onset_time, offset_time


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