# Standard Library Imports
# ----------------------------------------------------
import re
import warnings
from pathlib import Path
# Third-Party Dependencies
# ----------------------------------------------------
import pandas as pd

# ==============================================================================
# --- HELPER FUNCTION ---
# ==============================================================================
def _is_data_row(line_text: str, delimiter: str = '\t') -> bool:
    """
    Heuristic to distinguish between metadata/header lines and raw signal data.
    
    Logic:
    Calculates the ratio of numeric fields to total fields. If more than 50% 
    of the columns in a line are numeric, it is classified as a data row.
    """
    fields = line_text.strip().split(delimiter)
    numeric_count = 0
    if len(fields) < 2: return False
    for field in fields:
        try:
            # Handle localized numeric formats (e.g., using commas for decimals)
            float(field.replace(',', '.'))
            numeric_count += 1
        except (ValueError, AttributeError):
            pass
    return (numeric_count / len(fields)) > 0.5

# ==============================================================================
# --- load_signal ---
# ==============================================================================
def load_signal(filepath: Path) -> tuple[pd.DataFrame, dict]:
    """
    Loads signal data from TSV, CSV, or XLSX files with automated marker detection.
    
    This function performs two main tasks:
    1. Parses the raw file into a structured DataFrame.
    2. Analyzes embedded 'comments' to automatically identify Block and Trial starts.
    """
    
    # 1. File Type Validation and Reading
    # ----------------------------------
    file_extension = filepath.suffix.lower()
    
    if file_extension in ['.tsv', '.csv', '.txt']:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        data_frame = _process_text_file(lines, file_extension)
    elif file_extension in ['.xlsx']:
        data_frame = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Please upload a .tsv, .csv, or .xlsx file.")
    
    # --- Marker Identification Logic ---
    # Scientific recording software often embeds comments into the signal stream.
    # Frequency analysis used to distinguish experimental levels.
    comment_summary = {}
    
    if 'comments' in data_frame.columns:
        # Standardize comment column to string type
        data_frame['comments'] = data_frame['comments'].astype(str).replace('<NA>', '') # Ensure comments are strings; error otherwise

        # Filter out empty strings and count occurrences of unique markers
        comment_counts = data_frame['comments'].value_counts().drop(labels=[''], errors='ignore')
        comment_summary = comment_counts.to_dict() # PD series to dictionnary

        if comment_counts.empty:
            warnings.warn("No comments found in the data, cannot create block or trial columns.")
        else:
            # HEURISTIC: Block markers occur least frequently (e.g., once per block).
            block_comment_type = comment_counts.sort_values().index[0]
            
            # Remove the block comment from the pool of candidates
            potential_trial_comments = comment_counts.drop(block_comment_type)

            # HEURISTIC: Trial markers occur most frequently. We group them by base name 
            # to handle direction-specific markers (e.g., 'stim_left' and 'stim_right').
            def get_comment_base(comment_str):
                """Removes a direction suffix to find the base name."""
                base = re.sub(r'[-_\s]?(left|right|l|r)$', '', comment_str, flags=re.IGNORECASE)
                return base.strip()

            # Group remaining comments by their base name
            comment_groups = {}
            for comment in potential_trial_comments.index:
                base = get_comment_base(comment)
                if base not in comment_groups:
                    comment_groups[base] = []
                comment_groups[base].append(comment)
            
            # Identify the group with the highest combined frequency as the 'Trial Start'
            max_count = 0
            stimulus_comment_types = [] 
            for base, comments in comment_groups.items():
                total_count = sum(potential_trial_comments[c] for c in comments)
                if total_count > max_count:
                    max_count = total_count
                    stimulus_comment_types = comments

            # --- Structural Column Generation ---
            # If markers are found, generate boolean flags and cumulative counters.
            if block_comment_type and stimulus_comment_types:
                data_frame['is_block_start'] = (data_frame['comments'] == block_comment_type)
                data_frame['is_trial_start'] = data_frame['comments'].isin(stimulus_comment_types)

                # Generate Block and Trial numbers via cumulative sum
                data_frame['block_number'] = data_frame['is_block_start'].cumsum()
                data_frame['trial_number'] = data_frame.groupby('block_number')['is_trial_start'].cumsum()
            else:
                warnings.warn("Could not robustly determine block and stimulus comments from counts.")
    
    return data_frame, comment_summary

# ==============================================================================
# --- FILE PROCESSING HELPERS ---
# ==============================================================================
def _process_text_file(lines: list[str], file_extension: str) -> pd.DataFrame:
    """
    Parses plain text files by identifying headers and extracting embedded comments.
    """
    delimiter = '\t' if file_extension in ['.tsv', '.txt'] else ','
    
    # --- 1. Locate Data Start ---
    data_start_index = -1
    for i, line in enumerate(lines):
        if _is_data_row(line, delimiter):
            data_start_index = i
            break
            
    # Handle the case where no data rows are found
    if data_start_index == -1:
        warnings.warn("Could not find any headers or numerical data in the file.")
        return pd.DataFrame()
    
    # --- 2. Resolve Column Headers ---
    channel_names = []
    if data_start_index > 0:
        # Search metadata lines for common header keywords
        metadata_lines = lines[:data_start_index]
        best_candidate_index = -1
        header_keywords = ['channel', 'title']
        # Look for the best candidate line containing header keywords
        for i, line in enumerate(metadata_lines):
            if any(keyword in line.lower() for keyword in header_keywords):
                best_candidate_index = i
        # If found, extract channel names from that line
        if best_candidate_index != -1:
            header_line_index = best_candidate_index
            header_text = lines[header_line_index].strip()
            # Handle key-value metadata formats (e.g., "Channels = Time, Force")
            if '=' in header_text:
                header_text = header_text.split('=', 1)[1].strip()
            channel_names = [name.strip() for name in header_text.split(delimiter)]
        # Fallback: Generate generic names if no header line is found
        else:
            warnings.warn("No header keywords found. Using generic column names.")
            num_data_cols = len(lines[data_start_index].strip().split(delimiter))
            channel_names = [f'col_{i+1}' for i in range(num_data_cols)]
    # If no metadata lines, fallback to generic names
    else:
        warnings.warn("No header row detected; using generic column names.")
        num_data_cols = len(lines[data_start_index].strip().split(delimiter))
        channel_names = [f'col_{i+1}' for i in range(num_data_cols)]
    
    # Fix common omission where the 'Time' column is present but not named in header
    num_data_cols = len(lines[data_start_index].strip().split(delimiter))
    if len(channel_names) == num_data_cols - 1: # Softwares often omit naming the time column
        channel_names.insert(0, 'time')
        warnings.warn("Automatically added 'time' column to the header.")

    # --- 3. Parse Data and Embedded Comments ---
    # Scientific formats often append comments using '#*' at the end of data rows.
    final_col_names = [name for name in channel_names if name]
    
    # Check if a comments column should exist. The '#*' is the key here.
    has_comments = any('#*' in line for line in lines[data_start_index:]) # Check if comment exists
            
    if has_comments: 
        final_col_names.append('comments')
    
    data_rows = []
    final_num_cols = len(final_col_names) # Use to truncate/pad rows 
    
    for line in lines[data_start_index:]:
        fields = line.strip().split(delimiter)
        comment_part = None
        
        # Split the line into data and comments based on the '#*' delimiter
        # TODO: Handle a more generic comment delimiter based on regex/non-numeric?
        if '#*' in line:
            parts = line.split('#*')
            line = parts[0]
            comment_part = parts[1].strip()
            fields = line.strip().split(delimiter)
        
        # If comments were found, append them to the fields
        if has_comments:
            fields.append(comment_part)

        # Normalize row length to prevent DataFrame construction errors
        if len(fields) > final_num_cols:
            fields = fields[:final_num_cols]
        while len(fields) < final_num_cols:
            fields.append(None)
            
        data_rows.append(fields)

    # --- 4. DataFrame Finalization ---
    df = pd.DataFrame(data_rows, columns=final_col_names)
    
    # Cast signal columns to numeric and markers to string
    for col in df.columns:
        if col != 'comments':
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if col == 'comments':
            df[col] = df[col].astype('string')
    
    return df


# ==============================================================================
# --- find_best_match ---
# ==============================================================================
def find_best_match(available_channels: list, keyword_patterns: list) -> str | None:
    """
    Heuristic to match raw column names to expected signal types (e.g., Force, EMG).
    
    Matches are found via regex. If multiple matches exist, it selects the 
    shortest string (e.g., 'Force_R' is preferred over 'Force_R_Backup').
    """
    for pattern in keyword_patterns:
        matches = [channel for channel in available_channels if re.match(pattern, channel)]

        # If any matches were found, pick the shortest one
        # e.g., pick "force_r" over "force_r_raw". Robustness comes from UI. This is a heuristic (i.e., design choice).
        if matches:
            best_match = min(matches, key = len)
            return best_match
        
    return None