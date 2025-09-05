# Standard Library Imports
# ----------------------------------------------------
import re
import warnings
from pathlib import Path
from typing import Optional
# Third-Party Dependencies
# ----------------------------------------------------
import pandas as pd

# ==============================================================================
# --- HELPER FUNCTION ---
# ==============================================================================
def _is_data_row(line_text: str, delimiter: str = '\t') -> bool:
    """
    Checks if a line of text is a data row by testing if a majority of its
    columns are numeric.

    Args:
        line_text (str): The line of text to check.
        delimiter (str): The column delimiter, typically a tab or a comma.

    Returns:
        bool: True if the line is likely a data row, False otherwise.
    """
    fields = line_text.strip().split(delimiter)
    numeric_count = 0
    if len(fields) < 2: return False
    for field in fields:
        try:
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
    Loads signal data from a TSV, CSV, or XLSX file, handling a metadata header
    and embedded comments.

    Args:
        filepath (Path): The path to the uploaded file.

    Returns:
        tuple[pd.DataFrame, dict]: A tuple containing the processed DataFrame
        and a dictionary of categorized comments and their counts.

    Raises:
        ValueError: If the file format is not supported.
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

    # 2. Automated Comment Identification and Counting
    # ------------------------------------------------
    comment_summary = {}
    
    if 'comments' in data_frame.columns:
        data_frame['comments'] = data_frame['comments'].astype(str).replace('<NA>', '') # Ensure comments are strings; error otherwise

        # Count the frequency of each unique comment, and convert to a dictionary
        comment_counts = data_frame['comments'].value_counts().drop(labels=[''], errors='ignore')
        comment_summary = comment_counts.to_dict() # PD series to dictionnary

        # 3. Create block and trial columns based on comment counts
        # ----------------------------------------------------------
        if comment_counts.empty:
            warnings.warn("No comments found in the data, cannot create block or trial columns.")
        else:
            # --- Heuristic to find block and stimulus markers ---
            # Identify the block comment type (lowest frequency)
            block_comment_type = comment_counts.sort_values().index[0]
            
            # Remove the block comment from the pool of candidates
            potential_trial_comments = comment_counts.drop(block_comment_type)

            # Helper to find a comment's base name
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
            
            # Find the group with the highest total count
            max_count = 0
            stimulus_comment_types = [] # This will be a list of stimulus markers
            for base, comments in comment_groups.items():
                total_count = sum(potential_trial_comments[c] for c in comments)
                if total_count > max_count:
                    max_count = total_count
                    stimulus_comment_types = comments

            # --- Create new columns if markers were found ---
            if block_comment_type and stimulus_comment_types:
                # Create boolean columns for start events
                data_frame['is_block_start'] = (data_frame['comments'] == block_comment_type)
                data_frame['is_trial_start'] = data_frame['comments'].isin(stimulus_comment_types)

                # Create number columns using cumulative sums
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
    A helper function to process TSV and CSV files, which require special handling
    for headers and embedded comments.
    """
    delimiter = '\t' if file_extension in ['.tsv', '.txt'] else ','
    
    # 1. Find Header and Data Start
    # -----------------------------
    # Find the data start index using heuristic
    data_start_index = -1
    for i, line in enumerate(lines):
        if _is_data_row(line, delimiter):
            data_start_index = i
            break
            
    # Handle the case where no data rows are found
    if data_start_index == -1:
        warnings.warn("Could not find any headers or numerical data in the file.")
        return pd.DataFrame()
    
    # 2. Process the Header/Metadata
    # -------------------------------
   # Attempt to find a header row using keywords
    channel_names = []
    if data_start_index > 0:
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
            if '=' in header_text:
                header_text = header_text.split('=', 1)[1].strip()
            channel_names = [name.strip() for name in header_text.split(delimiter)]
        # Else, fallback to generic names
        else:
            warnings.warn("No header keywords found. Using generic column names.")
            num_data_cols = len(lines[data_start_index].strip().split(delimiter))
            channel_names = [f'col_{i+1}' for i in range(num_data_cols)]
    # If no metadata lines, fallback to generic names
    else:
        warnings.warn("No header row detected; using generic column names.")
        num_data_cols = len(lines[data_start_index].strip().split(delimiter))
        channel_names = [f'col_{i+1}' for i in range(num_data_cols)]
    
    # Final check to add the 'time' column if it's missing
    num_data_cols = len(lines[data_start_index].strip().split(delimiter))
    if len(channel_names) == num_data_cols - 1: # Softwares often omit naming the time column
        channel_names.insert(0, 'time')
        warnings.warn("Automatically added 'time' column to the header.")

    # 3. Process the Data
    # -------------------
    final_col_names = [name for name in channel_names if name] # Remove empty names
    
    # Check if a comments column should exist. The '#*' is the key here.
    has_comments = any('#*' in line for line in lines[data_start_index:]) # Check if comment exists
            
    if has_comments: # Prepare a "comments" column
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

        # Truncate or pad the row to match the number of columns
        if len(fields) > final_num_cols:
            fields = fields[:final_num_cols]
        while len(fields) < final_num_cols:
            fields.append(None)
            
        data_rows.append(fields)

    # Create the DataFrame
    df = pd.DataFrame(data_rows, columns=final_col_names)
    
    # Convert all non-comment columns to numeric types
    # Convert comments to string type
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
    Find the best matching columns for force and EMG signals in the DataFrame.
    Args:
    availables_channels (list): List of available column names in the DataFrame.
    keyword_patterns (list): List of regex patterns to match against column names.
    Returns:
        str | None: The best matching column name, or None if no match is found.
    """
    for pattern in keyword_patterns:
        matches = [channel for channel in available_channels if re.match(pattern, channel)]

        # If any matches were found, pick the shortest one
        # e.g., pick "force_r" over "force_r_raw". Robustness comes from UI. This is a heuristic (i.e., design choice).
        if matches:
            best_match = min(matches, key = len)
            return best_match
        
    return None