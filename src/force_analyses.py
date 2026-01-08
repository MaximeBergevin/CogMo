# src/force_analyses.py
# Standard Library Imports
from typing import Optional, Dict, Any
# Third-party dependencies
import pandas as pd
import numpy as np
from scipy import signal


def calculate_impulse(
    signal_df: pd.DataFrame,
    onset_time: Optional[float],
    offset_time: Optional[float],
    baseline_force: Optional[float],
    mvc_value: Optional[float],
    response_hand: str
) -> Dict[str, Any]:
    """
    Calculates the Force-Time Integral (Impulse) during the contraction.
    
    The impulse is calculated as the area under the baseline-corrected force curve 
    using the trapezoidal rule. It represents the total force generated over time.
    """
    force_col = f"force_{response_hand}"

    # Guard clause: Ensure all necessary time points are valid
    if (onset_time is None or offset_time is None or baseline_force is None or
            offset_time <= onset_time):
        return {'impulse_auc': None, 'impulse_auc_percent_mvc': None}

    # Slice data to the specific contraction window (Onset to Offset)
    contraction_df = signal_df[
        (signal_df['time'] >= onset_time) & (signal_df['time'] <= offset_time)
    ].copy()

    # Rectify the signal by subtracting the pre-calculated baseline
    contraction_df['force_corrected'] = contraction_df[force_col] - baseline_force
    
    # Calculate Area Under the Curve (Impulse) using NumPy's trapezoidal rule
    impulse_auc = np.trapezoid(y=contraction_df['force_corrected'], x=contraction_df['time'])
    
    # Normalization
    contraction_duration = offset_time - onset_time
    impulse_auc_percent_mvc = None
    if mvc_value and mvc_value > 0 and contraction_duration > 0:
        # This normalization represents the mean force over the contraction as a % of MVC
        impulse_auc_percent_mvc = (impulse_auc / contraction_duration) / mvc_value * 100

    return {
        'impulse_auc': impulse_auc,
        'impulse_auc_percent_mvc': impulse_auc_percent_mvc
    }


def calculate_mean_force(
    signal_df: pd.DataFrame,
    onset_time: Optional[float],
    offset_time: Optional[float],
    baseline_force: Optional[float],
    mvc_value: Optional[float],
    response_hand: str
) -> Dict[str, Any]:
    """
    Computes the average baseline-corrected force over the contraction duration.
    """
    force_col = f"force_{response_hand}"

    # Guard clause: Ensure all necessary inputs are valid
    if (onset_time is None or offset_time is None or baseline_force is None or
            offset_time <= onset_time):
        return {'mean_force': None, 'mean_force_percent_mvc': None}

    # Slice the DataFrame to the exact contraction window
    contraction_df = signal_df[
        (signal_df['time'] >= onset_time) & (signal_df['time'] <= offset_time)
    ].copy()

    # Subtract the baseline force
    force_corrected = contraction_df[force_col] - baseline_force
    
    # Calculate the mean of the baseline-corrected force
    mean_force = force_corrected.mean()
    
    # Normalization by MVC
    mean_force_percent_mvc = None
    if mvc_value and mvc_value > 0:
        mean_force_percent_mvc = (mean_force / mvc_value) * 100

    return {
        'mean_force': mean_force,
        'mean_force_percent_mvc': mean_force_percent_mvc
    }


def calculate_rfd(
    signal_df: pd.DataFrame,
    onset_time: Optional[float],
    peak_time: Optional[float],
    baseline_force: Optional[float],
    response_hand: str,
    early_rfd_window_ms: int
) -> Dict[str, Any]:
    """
    Calculates Rate of Force Development (RFD) metrics.
    
    1. Early RFD: The slope of the force rise over a fixed window (e.g., 0-50ms).
    2. Peak RFD: The maximum instantaneous slope found within a 20ms sliding window.
    """
    force_col = f"force_{response_hand}"

    if (onset_time is None or peak_time is None or baseline_force is None or
            peak_time <= onset_time):
        return {'early_rfd': None, 'peak_rfd': None}

    contraction_df = signal_df[
        (signal_df['time'] >= onset_time) & (signal_df['time'] <= peak_time)
    ].copy()

    if len(contraction_df) < 2:
        return {'early_rfd': None, 'peak_rfd': None}

    force_corrected = contraction_df[force_col] - baseline_force
    time_seconds = contraction_df['time']

    # --- 1. Calculate Early RFD (Unchanged) ---
    early_rfd = None
    early_rfd_window_s = early_rfd_window_ms / 1000.0
    end_time = onset_time + early_rfd_window_s
    
    early_df_slice = contraction_df[contraction_df['time'] <= end_time]
    
    if len(early_df_slice) > 1:
        force_at_start = force_corrected.iloc[0]
        force_at_end = force_corrected.loc[early_df_slice.index[-1]]
        time_at_end = early_df_slice['time'].iloc[-1]
        
        delta_time = time_at_end - onset_time
        if delta_time > 0:
            early_rfd = (force_at_end - force_at_start) / delta_time

    # --- 2. Calculate Peak RFD (Late RFD) ---
    #    This is the corrected 20ms sliding window logic, translated from your R code.
    
    force_array = force_corrected.to_numpy()
    time_array = time_seconds.to_numpy()
    
    # Get the median time between samples
    dt = np.median(np.diff(time_array))
    # Calculate how many samples are in a 20ms window
    window_samples = int(round(0.020 / dt))
    
    if window_samples < 1:
        window_samples = 1 # Ensure at least 1 sample difference
    
    if len(force_array) <= window_samples:
        peak_rfd = None # Not enough data to calculate a slope
    else:
        # Calculate the vectorized differences
        n = len(force_array)
        end_vals = force_array[window_samples:]
        start_vals = force_array[:-window_samples]
        
        dt_vals = time_array[window_samples:] - time_array[:-window_samples]
        
        # Calculate all slopes in the window
        slopes = (end_vals - start_vals) / dt_vals
        
        # Find the maximum slope
        peak_rfd = np.max(slopes)

    return {
        'early_rfd': early_rfd,
        'peak_rfd': peak_rfd
    }


def find_baseline_force(
    signal_df: pd.DataFrame,
    peak_time: float,
    response_hand: str
) -> Optional[float]:
    """
    Identifies a stable pre-contraction baseline via iterative window searching.
    
    The algorithm searches 50ms windows before the peak. If the standard deviation 
    is high (> 1.0), it shifts the window to avoid signal drift or early onset 
    interference until a stable mean is found.
    """
    force_col = f"force_{response_hand}"

    # Initialize 50 ms window starting 250ms before the peak
    baseline_start = peak_time - 0.250
    baseline_end = baseline_start + 0.050
    shift_s = 0.050  # 50 ms
    max_iter = 10

    for _ in range(max_iter):
        # Ensure the window does not go before the start of the signal
        if baseline_start < signal_df['time'].min():
            break

        baseline_df = signal_df[
            (signal_df['time'] >= baseline_start) & (signal_df['time'] < baseline_end)
        ]

        if len(baseline_df) < 2: # Need at least 2 points for std dev
            break

        baseline_sd = baseline_df[force_col].std()

        if baseline_sd <= 1.0:
            # Stable baseline found, return its mean
            return baseline_df[force_col].mean()

        # Check for drift and shift the window in the opposite direction
        half_idx = len(baseline_df) // 2
        early_mean = baseline_df[force_col].iloc[:half_idx].mean()
        late_mean = baseline_df[force_col].iloc[half_idx:].mean()

        if late_mean > early_mean:
            # Drift is upward, so the contraction might be starting. Shift earlier.
            baseline_start -= shift_s
            baseline_end -= shift_s
        else:
            # Drift is downward or stable. Shift later to get closer to onset.
            baseline_start += shift_s
            baseline_end += shift_s
            
    # If the loop finishes without finding a stable baseline
    return None



def find_contraction_offset(
    signal_df: pd.DataFrame,
    peak_time: float,
    peak_value: float,
    response_hand: str
) -> Optional[float]:
    """
    Identifies the end of the force contraction (offset).
    
    Finds a stable post-peak baseline and sets a threshold (Mean + 3*SD). The 
    offset is the first point after the peak where force returns below this threshold.
    """
    force_col = f"force_{response_hand}"

    # Guard condition to avoid issues if peak force is very low
    relative_guard = 0.20 * abs(peak_value)
    
    # --- Iteratively search for a stable post-peak baseline window ---
    baseline_start = peak_time + 0.150
    baseline_end = baseline_start + 0.050
    shift_s = 0.050  # 50 ms
    max_iter = 10
    threshold = None

    for _ in range(max_iter):
        if baseline_start > signal_df['time'].max():
            break # Stop if we run out of data

        baseline_df = signal_df[
            (signal_df['time'] >= baseline_start) & (signal_df['time'] < baseline_end)
        ]
        
        if baseline_df.empty:
            # Shift window forward and continue to the next iteration
            baseline_start += shift_s
            baseline_end += shift_s
            continue

        baseline_mean = baseline_df[force_col].mean()
        baseline_sd = baseline_df[force_col].std()
        
        # A baseline is stable if its SD is low AND its mean is low relative to the peak
        if baseline_sd <= 1.0 and baseline_mean <= relative_guard:
            threshold = baseline_mean + (3 * baseline_sd)
            break # Stable baseline found

        # If not stable, just move the window forward for the next iteration
        baseline_start += shift_s
        baseline_end += shift_s
    
    if threshold is None:
        return None # No stable post-peak baseline was found

    # --- Find the first time point after the peak that drops below the threshold ---
    offset_window_df = signal_df[signal_df['time'] >= peak_time]
    offset_candidates = offset_window_df[offset_window_df[force_col] <= threshold]
    
    if offset_candidates.empty:
        return None

    # Return the time of the first occurrence
    offset_time = offset_candidates['time'].iloc[0]

    return offset_time


def find_contraction_onset(
    signal_df: pd.DataFrame,
    stim_time: float,
    peak_time: float,
    response_hand: str
) -> Optional[float]:
    """
    Identifies the start of the force contraction (onset).
    
    Logic follows the same iterative thresholding as offset detection, but 
    scans backward from the peak to find the last point where force was at baseline.
    """
    force_col = f"force_{response_hand}"
    
    # Iteratively search for a stable baseline window
    baseline_start = peak_time - 0.250
    baseline_end = baseline_start + 0.050
    shift_s = 0.050  # 50 ms
    max_iter = 10
    threshold = None

    for _ in range(max_iter):

        baseline_df = signal_df[
            (signal_df['time'] >= baseline_start) & (signal_df['time'] < baseline_end)
        ]
        
        if baseline_df.empty:
            break

        baseline_sd = baseline_df[force_col].std()
        
        if baseline_sd <= 1.0:
            baseline_mean = baseline_df[force_col].mean()
            threshold = baseline_mean + (3 * baseline_sd)
            break # Stable baseline found
        
        # Shift the window based on drift direction
        half_idx = len(baseline_df) // 2
        early_mean = baseline_df[force_col].iloc[:half_idx].mean()
        late_mean = baseline_df[force_col].iloc[half_idx:].mean()
        
        if late_mean > early_mean:
            baseline_start -= shift_s
            baseline_end -= shift_s
        else:
            baseline_start += shift_s
            baseline_end += shift_s

    if threshold is None:
        return None # No stable baseline was found

    # Scan backward from the peak to find the last point at or below the threshold
    onset_window_df = signal_df[
        (signal_df['time'] >= stim_time) & (signal_df['time'] <= peak_time)
    ]
    onset_candidates = onset_window_df[onset_window_df[force_col] <= threshold]
    
    if onset_candidates.empty:
        return None

    return onset_candidates['time'].max()


def find_main_contraction_peak(
    full_df: pd.DataFrame,
    stim_time: float,
    channel_map: Dict[str, str],
    threshold: float,
    min_valid_rt_s: float,
    min_prominence_n: float,
    search_window_pre_s: float,
    search_window_post_s: float
) -> Dict[str, Any]:
    """
    Determines the responding hand and identifies the primary contraction peak.
    
    Uses SciPy's peak detection on both force channels. Selection priority:
    1. The earliest valid peak following the minimum reaction time (RT) boundary.
    2. If no valid peak exists, the latest 'false start' peak before the RT boundary.
    """
    force_r_col = channel_map.get('force_right')
    force_l_col = channel_map.get('force_left')

    # 1. Broad Search for Candidate Peaks on Both Hands
    # --------------------------------------------------
    search_start = stim_time - search_window_pre_s
    search_end = stim_time + search_window_post_s
    search_df = full_df[(full_df['time'] >= search_start) & (full_df['time'] <= search_end)].copy()

    if search_df.empty:
        return {'status': 'omission', 'peak_time': None, 'peak_value': None, 'response_hand': None, 'analysis_df': None}

    # Find all meaningful peaks on the right hand
    r_peak_indices, _ = signal.find_peaks(search_df[force_r_col], height=threshold, prominence=min_prominence_n)
    right_peaks = search_df.iloc[r_peak_indices].copy()
    right_peaks['hand'] = 'right'
    
    # Find all meaningful peaks on the left hand
    l_peak_indices, _ = signal.find_peaks(search_df[force_l_col], height=threshold, prominence=min_prominence_n)
    left_peaks = search_df.iloc[l_peak_indices].copy()
    left_peaks['hand'] = 'left'

    # Combine all found peaks into a single DataFrame
    all_peaks_df = pd.concat([right_peaks, left_peaks]).sort_values(by='time')

    if all_peaks_df.empty:
        return {'status': 'omission', 'peak_time': None, 'peak_value': None, 'response_hand': None, 'analysis_df': None}

    # 2. Apply Prioritized Selection Rule
    # ------------------------------------
    valid_rt_boundary = stim_time + min_valid_rt_s
    
    valid_response_peaks = all_peaks_df[all_peaks_df['time'] >= valid_rt_boundary]
    false_start_peaks = all_peaks_df[all_peaks_df['time'] < valid_rt_boundary]
    
    target_peak = None
    status = 'omission'

    if not valid_response_peaks.empty:
        # The true response is the EARLIEST valid peak after the RT window
        target_peak = valid_response_peaks.iloc[0]
        status = 'valid'
    elif not false_start_peaks.empty:
        # The false start is the LAST peak that occurred before the RT window
        target_peak = false_start_peaks.iloc[-1]
        status = 'false_start'
    
    if target_peak is None:
        return {'status': status, 'peak_time': None, 'peak_value': None, 'response_hand': None, 'analysis_df': None}
    
    peak_time = target_peak['time']
    response_hand = target_peak['hand']
    peak_value = target_peak[channel_map.get(f"force_{response_hand}")] # Get value from the correct column

    # 3. Create the Dynamic Analysis Window
    # --------------------------------------
    analysis_start = peak_time - 1.25
    analysis_end = peak_time + 1.25
    analysis_df = full_df[(full_df['time'] >= analysis_start) & (full_df['time'] <= analysis_end)]

    return {
        'status': status,
        'peak_time': peak_time,
        'peak_value': peak_value,
        'response_hand': response_hand, # Return the determined hand
        'analysis_df': analysis_df
    }


def motor_reaction_time(
    stim_time: float,
    onset_time: Optional[float]
) -> Optional[int]:
    """
    Calculates time from stimulus to force onset in milliseconds.
    """
    # Return None if onset was not detected or occurred before the stimulus
    if onset_time is None or onset_time < stim_time:
        return None
   
    # Calculate motor reaction time in ms
    motor_rt_ms = int(round((onset_time - stim_time) * 1000))
    
    return motor_rt_ms


def motor_response_time(
    signal_df: pd.DataFrame,
    stim_time: float,
    peak_time: float,
    peak_force: float,
    threshold: float,
    response_hand: str
) -> Optional[int]:
    """
    Calculates time from stimulus to the point where force crossed the threshold.
    """
    if response_hand not in ['left', 'right'] or threshold is None:
        return None

    force_col = f"force_{response_hand}"

    # Early exit if the peak force never even reached the threshold
    if peak_force < threshold:
        return None

    # Filter the signal to the window between stimulus and peak force
    response_window_df = signal_df[
        (signal_df['time'] >= stim_time) & (signal_df['time'] <= peak_time)
    ]
    
    # Find all points at or below the threshold within that window
    onset_candidates = response_window_df[response_window_df[force_col] <= threshold]

    if onset_candidates.empty:
        return None # Should not happen if peak > threshold, but a safe check

    # The force onset is the last time point (max time) at or below the threshold
    force_onset_time = onset_candidates['time'].max()
    
    # Calculate duration in milliseconds
    mrspt_ms = int(round((force_onset_time - stim_time) * 1000))

    return mrspt_ms


def peak_force_metrics(
    peak_value: float,
    peak_time: float,
    stim_time: float,
    threshold: float,
    mvc_value: float
) -> Dict[str, Any]:
    """
    Calculates secondary metrics related to peak magnitude and accuracy.
    """
    # 1. Time to Peak
    time_to_peak = peak_time - stim_time

    # 2. Peak Force as %MVC
    peak_force_pct_mvc = (peak_value / mvc_value) * 100 if mvc_value > 0 else 0

    # 3. Overshoot / Undershoot
    delta_threshold = peak_value - threshold
    
    delta_threshold_pct = 0.0
    threshold_direction = "N/A"
    if threshold > 0:
        delta_threshold_pct = (delta_threshold / threshold) * 100
        if abs(delta_threshold_pct) <= 1.0:
            threshold_direction = "on-target"
        elif delta_threshold > 0:
            threshold_direction = "overshoot"
        else:
            threshold_direction = "undershoot"

    metrics = {
        'peak_force': peak_value, # Return the passed-in value for completeness
        'time_to_peak': time_to_peak,
        'peak_force_pct_mvc': peak_force_pct_mvc,
        'delta_threshold': delta_threshold,
        'delta_threshold_pct': delta_threshold_pct,
        'threshold_direction': threshold_direction
    }
    return metrics