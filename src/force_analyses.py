# src/force_analyses.py
# Standard Library Imports
from typing import Optional, Dict, Any
# Third-party dependencies
import pandas as pd
import numpy as np
from scipy import signal

def apply_force_filter(force_series, cutoff=50, order=4):
    """
    Applies a zero-phase Butterworth low-pass filter to a Pandas Series.
    Automatically derives fs from the series index.
    """
    if force_series is None or len(force_series) < 27:
        return force_series

    # Calculate fs from the series index (dt)
    try:
        dt = np.median(np.diff(force_series.index.values))
        if dt <= 0:
            return force_series
        
        fs = 1.0 / dt
    except Exception:
        return force_series

    nyquist = 0.5 * fs
    if cutoff >= nyquist:
        return force_series
        
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    
    # Return as the same type (Series) to maintain indices
    filtered_values = signal.filtfilt(b, a, force_series.values)
    return pd.Series(filtered_values, index=force_series.index, name=force_series.name)


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

    # --- 1. Calculate Early RFD ---
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
    stim_time: float,
    response_hand: str
) -> Dict[str, Optional[float]]:
    """
    Identifies the most stable pre-contraction window and returns 
    both the mean (baseline) and its specific standard deviation (noise).
    """
    force_col = f"force_{response_hand}"
    
    # Define a fixed search zone: From 800ms before stim to 100ms before stim
    # (Adjusted based on your previous 'out of bounds' fix)
    search_start = stim_time
    search_end = stim_time + 0.500
    
    window_size = 0.050
    current_start = search_start
    
    best_mean = None
    best_sd = None
    lowest_sd = float('inf')

    while current_start + window_size <= search_end:
        window_df = signal_df[
            (signal_df['time'] >= current_start) & 
            (signal_df['time'] < current_start + window_size)
        ]
        
        if len(window_df) >= 2:
            current_sd = window_df[force_col].std()
            
            if current_sd < lowest_sd:
                lowest_sd = current_sd
                best_sd = current_sd
                best_mean = window_df[force_col].mean()
        
        current_start += 0.025 
        
    return {'mean': best_mean, 'sd': best_sd}


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
    shift_s = 0.10 
    max_iter = 50
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
        if baseline_sd <= 0.5 and baseline_mean <= relative_guard:
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


def find_contraction_offset(
    signal_df: pd.DataFrame,
    peak_time: float,
    peak_value: float,
    response_hand: str,
    mvc_value: float
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
    shift_s = 0.10 
    max_iter = 50
    threshold = None

    # Track the flattest window found so far
    best_delta = float('inf')

    # --- Dynamic Scaling Parameters ---
    # Dead-zone: 0.5% of MVC. Below this: consider the window 'perfectly flat'
    dz_threshold = mvc_value * 0.005 
    # SD Floor: 0.1% of MVC. Prevents threshold from falling into microscopic noise
    sd_floor = mvc_value * 0.001

    for _ in range(max_iter):
        if baseline_start > signal_df['time'].max():
            break # Stop if we run out of data

        baseline_df = signal_df[
            (signal_df['time'] >= baseline_start) & (signal_df['time'] < baseline_end)
        ]
        
        if baseline_df.empty or len(baseline_df) < 2:
            # Shift window forward and continue to the next iteration
            baseline_start += shift_s
            baseline_end += shift_s
            continue

        baseline_mean = baseline_df[force_col].mean()
        baseline_sd = baseline_df[force_col].std()
        
        # Calculate slope/delta (End - Start) to find 'flatness'
        raw_delta = abs(baseline_df[force_col].iloc[-1] - baseline_df[force_col].iloc[0])

        # --- Dead-zone Logic ---
        # Treat changes less than MVC-scaled threshold as perfectly flat
        current_delta = 0.0 if raw_delta < dz_threshold else raw_delta

        if baseline_mean <= relative_guard:
            # Pick the flattest candidate
            if current_delta <= best_delta:
                best_delta = current_delta
                
                safe_sd = max(baseline_sd, sd_floor)
                threshold = baseline_mean + (3 * safe_sd)
                
                if current_delta == 0.0:
                    break # Stable baseline found

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
    peak_value: float,
    response_hand: str,
    mvc_value: float,
) -> Optional[float]:
    """
    Identifies the onset by scanning backward from the peak.
    Iteratively searches for a stable baseline before the contraction 
    to establish a 3SD threshold.
    """
    force_col = f"force_{response_hand}"

    # Guard: Baseline must be < 20% of peak to avoid the contraction ramp
    relative_guard = 0.20 * abs(peak_value)
    
    # --- Iterative search moving BACKWARD from peak ---
    # Start the search window just before the peak ramp
    baseline_end = peak_time - 0.050 
    baseline_start = baseline_end - 0.025
    shift_s = 0.010 
    max_iter = 50
    threshold = None

    # Track the flattest window found so far
    best_delta = float('inf')

    # --- Dynamic Scaling Parameters ---
    # Dead-zone: 0.5% of MVC. Below this: we consider the window perfectly flat
    dz_threshold = mvc_value * 0.005 
    # SD Floor: 0.1% of MVC. Prevents threshold from falling into microscopic noise
    sd_floor = mvc_value * 0.001

    for _ in range(max_iter):
        # Don't search before the stimulus/recording start
        if baseline_start < signal_df['time'].min():
            break

        baseline_df = signal_df[
            (signal_df['time'] >= baseline_start) & (signal_df['time'] < baseline_end)
        ]
        
        if baseline_df.empty or len(baseline_df) < 2:
            baseline_start -= shift_s
            baseline_end -= shift_s
            continue

        baseline_mean = baseline_df[force_col].mean()
        baseline_sd = baseline_df[force_col].std()
        
        # Calculate slope/delta (End - Start) to find flatness
        raw_delta = abs(baseline_df[force_col].iloc[-1] - baseline_df[force_col].iloc[0])

        # --- Dead-zone Logic ---
        # Treat changes less than MVC-scaled threshold as perfectly flat to ignore drift/noise
        current_delta = 0.0 if raw_delta < dz_threshold else raw_delta

        # Stability logic: Pick the flattest window below the 20% peak guard
        if baseline_mean <= relative_guard:
            if current_delta <= best_delta:
                best_delta = current_delta
                
                # Use a safety floor for SD to prevent hyper-sensitivity
                safe_sd = max(baseline_sd, sd_floor)
                
                threshold = baseline_mean + (3 * safe_sd)
                
                if current_delta == 0.0:
                    break 

        baseline_start -= shift_s
        baseline_end -= shift_s
    
    if threshold is None:
        return None 

    # --- Scan backward from the peak to find the breakaway point ---
    search_window = signal_df[
        (signal_df['time'] >= stim_time) & (signal_df['time'] <= peak_time)
    ]
    
    at_baseline = search_window[search_window[force_col] <= threshold]
    
    if at_baseline.empty:
        return stim_time 

    return at_baseline['time'].max()

def find_main_contraction_peak(
    full_df: pd.DataFrame,
    stim_time: float,
    threshold: float,     
    min_valid_rt_s: float,
    min_prominence_n: float,
    search_window_pre_s: float,
    search_window_post_s: float,
    noise_shield: float = 0.2  #
) -> Dict[str, Any]:
    """
    Identifies the primary contraction peak using a double-threshold approach.
    Prioritizes the earliest crossing of the main 'threshold', falling back 
    to the highest prominence peak that stays above the 'noise_shield'.
    """
    import pandas as pd
    from scipy import signal
    
    force_r_col = 'force_right'
    force_l_col = 'force_left'

    # 1. Define Search Window
    search_start = stim_time - search_window_pre_s
    search_end = stim_time + search_window_post_s
    search_df = full_df[(full_df['time'] >= search_start) & (full_df['time'] <= search_end)].copy()

    if search_df.empty:
        return {'status': 'omission', 'peak_time': None, 'peak_value': None, 'response_hand': None, 'analysis_df': None}

    # 2. Extract Peaks Safely (Decimal prominence allowed)
    # -----------------------------------------------------------------
    def get_peaks(df, col, hand_name):
        idx, props = signal.find_peaks(df[col], prominence=float(min_prominence_n))
        peaks = df.iloc[idx].copy()
        if not peaks.empty:
            peaks['prominence'] = props['prominences']
            peaks['hand'] = hand_name
            peaks['abs_force'] = peaks[col]
            return peaks
        return pd.DataFrame(columns=list(df.columns) + ['prominence', 'hand', 'abs_force'])

    right_peaks = get_peaks(search_df, force_r_col, 'right')
    left_peaks = get_peaks(search_df, force_l_col, 'left')
    
    all_peaks_df = pd.concat([right_peaks, left_peaks]).sort_values(by='time')

    if all_peaks_df.empty:
        return {'status': 'omission', 'peak_time': None, 'peak_value': None, 'response_hand': None, 'analysis_df': None}

    # 3. Competitive Selection with Double Threshold
    # ----------------------------------------------------------
    valid_rt_boundary = stim_time + min_valid_rt_s
    valid_candidates = all_peaks_df[all_peaks_df['time'] >= valid_rt_boundary].copy()
    false_starts = all_peaks_df[all_peaks_df['time'] < valid_rt_boundary]
    
    target_peak = None
    status = 'omission'

    if not valid_candidates.empty:
        # THRESHOLD 1: The Main Target (e.g., 3.0 N)
        above_target = valid_candidates[valid_candidates['abs_force'] >= threshold]

        if not above_target.empty:
            # RULE 1: Earliest peak to hit the target
            target_peak = above_target.sort_values(by='time').iloc[0]
            status = 'valid'
        else:
            # THRESHOLD 2: The Noise Shield (e.g., 0.2 N)
            # This ignores the 'silent' hand noise seen in image_8efb3e.jpg
            real_attempts = valid_candidates[valid_candidates['abs_force'] > noise_shield]
            
            if not real_attempts.empty:
                # RULE 2: Fallback to highest prominence among real attempts
                target_peak = real_attempts.sort_values(by='prominence', ascending=False).iloc[0]
                status = 'low_force_candidate'
            else:
                status = 'omission'

    elif not false_starts.empty:
        # RULE 3: Fallback to latest false start for visual context
        target_peak = false_starts.sort_values(by='time').iloc[-1]
        status = 'false_start'
    
    if target_peak is None:
        return {'status': status, 'peak_time': None, 'peak_value': None, 'response_hand': None, 'analysis_df': None}
    
    peak_time = target_peak['time']
    response_hand = target_peak['hand']
    peak_value = target_peak['abs_force']

    # 4. Create Analysis Window
    analysis_start = stim_time - 0.5
    analysis_end = stim_time + 2.0
    analysis_df = full_df[(full_df['time'] >= analysis_start) & (full_df['time'] <= analysis_end)]

    return {
        'status': status,
        'peak_time': peak_time,
        'peak_value': peak_value,
        'response_hand': response_hand,
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