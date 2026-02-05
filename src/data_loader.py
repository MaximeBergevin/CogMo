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
    """
    file_extension = filepath.suffix.lower()
    
    if file_extension in ['.tsv', '.csv', '.txt']:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        data_frame = _process_text_file(lines, file_extension)
    elif file_extension in ['.xlsx']:
        data_frame = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format.")
    
    comment_summary = {}
    
    if 'comments' in data_frame.columns:
        # Standardize comments: handle missing/NaN values
        data_frame['comments'] = data_frame['comments'].fillna('').astype(str)
        data_frame['comments'] = data_frame['comments'].replace(['nan', 'None', '<NA>'], '')

        # 1. Populate summary with raw counts for UI visibility
        raw_counts = data_frame['comments'].value_counts().drop(labels=[''], errors='ignore')
        comment_summary = raw_counts.to_dict()

        # Inner helper to group markers (e.g., 'stim_left' -> 'stim')
        def get_comment_base(comment_str):
            """Removes direction suffixes to find the base name."""
            if not isinstance(comment_str, str):
                return ""
            base = re.sub(r'[-_\s]?(left|right|none|l|r|n)$', '', comment_str, flags=re.IGNORECASE)
            return base.strip()

        if not raw_counts.empty:
            # 2. Heuristic: Use normalized "Base Names" for frequency decisions
            data_frame['comment_base'] = data_frame['comments'].apply(get_comment_base)
            base_counts = data_frame['comment_base'].value_counts().drop(labels=[''], errors='ignore')

            # Block markers are rare; Trial markers are frequent
            block_base_type = base_counts.sort_values().index[0]
            trial_base_type = base_counts.sort_values(ascending=False).index[0]

            # 3. Create structural boolean flags
            data_frame['is_block_start'] = (data_frame['comment_base'] == block_base_type)
            data_frame['is_trial_start'] = (data_frame['comment_base'] == trial_base_type)
            
            # 4. Generate numbering
            data_frame['block_number'] = data_frame['is_block_start'].fillna(False).cumsum()
            # Grouping by block ensures trial numbers reset to 1 at each new block
            data_frame['is_block_start'] = data_frame['is_block_start'].fillna(False)
            data_frame['is_trial_start'] = data_frame['is_trial_start'].fillna(False)
            data_frame['trial_number'] = data_frame.groupby('block_number')['is_trial_start'].cumsum()
            
            # Clean up temporary processing column
            data_frame.drop(columns=['comment_base'], inplace=True)
    
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
    
    # Check for missing time column and add if necessary
    num_data_cols = len(lines[data_start_index].strip().split(delimiter))
    if len(channel_names) == num_data_cols - 1: # Softwares often omit naming the time column
        channel_names.insert(0, 'time')
        warnings.warn("Automatically added 'time' column to the header.")

    # --- 3. Parse Data and Embedded Comments ---
    final_col_names = [name for name in channel_names if name]
    
    # Check if a column has a comment
    has_comments = False
    for line in lines[data_start_index:]:
        fields = line.strip().split(delimiter)
        for field in fields:
            f_strip = field.strip().replace(',', '.')
            if f_strip == "":
                continue
            try:
                float(f_strip)
            except (ValueError):
            # Found something that isn't a number -> probably a comment
                has_comments = True
                break
        if has_comments: break 

    if has_comments: 
        final_col_names.append('comments')
    
    data_rows = []
    final_num_cols = len(final_col_names) # Use to truncate/pad rows 
    
    for line in lines[data_start_index:]:
        # Split the line into data and comments
        raw_fields = line.strip().split(delimiter)
        numeric_fields = []
        text_fields = []

        for f in raw_fields:
            clean_f = f.strip().replace(',', '.')
            if clean_f == "":
                continue

            # Check if it's a number
            try:
                # If number, keep in data list
                float(clean_f)
                numeric_fields.append(clean_f)
            except ValueError:
                # If not a number, it's a comment
                text_fields.append(f.strip())

        comment_part = " | ".join(text_fields) if text_fields else None

        fields = numeric_fields

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
            df[col] = df[col].astype(str).str.replace(',', '.')  # Handle localized decimal formats
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if col == 'comments':
            df[col] = df[col].astype('string')

    if 'time' in df.columns and len(df) > 1:
        # Detect sampling interval from the first couples rows
        dt = df['time'].iloc[1] - df['time'].iloc[0]
        df['time'] = df.index * dt
    
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