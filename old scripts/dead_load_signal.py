# src/load_signal.py
from pathlib import Path
import pandas as pd
import re

# --- READ DATA ---
# This part is just a placeholder. You will run this on your machine.
filepath = Path(__file__).parent.parent / "data" / "h021.txt"

try:
    with open(filepath, 'r') as f:
        lines = f.readlines()
    print(f"Successfully read {len(lines)} lines from '{filepath.name}'.")
except FileNotFoundError:
    print(f"Error: File not Found at {filepath}")
    lines = []

# --- CHECK FOR METADATA ---
def is_data_row(line_text, delimiter='\t'):
    """
    A helper function that checks if a line of text is likely a data row
    by testing if a majority of its columns are numeric.
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

# Initialize variables to store our findings.
data_start_index = -1
header_line_index = -1
channel_names = []
warnings = []

if lines:
    for i, line in enumerate(lines):
        if is_data_row(line):
            data_start_index = i
            break

    if data_start_index == -1:
        warnings.append("Could not find any numerical data rows in the file.")
    elif data_start_index == 0:
        warnings.append("No header row detected; using generic column names.")
        num_cols = len(lines[0].strip().split('\t'))
        channel_names = [f'col_{i+1}' for i in range(num_cols)]
    else:
        metadata_lines = lines[:data_start_index]
        num_data_cols = len(lines[data_start_index].strip().split('\t'))
        
        best_candidate_index = -1
        header_keywords = ['channel', 'title']
        
        for i, line in enumerate(metadata_lines):
            if any(keyword in line.lower() for keyword in header_keywords):
                best_candidate_index = i
        
        if best_candidate_index != -1:
            header_line_index = best_candidate_index
            header_text = lines[header_line_index].strip()
            
            if '=' in header_text:
                header_text = header_text.split('=', 1)[1].strip()
            
            channel_names = [name.strip() for name in header_text.split('\t')]
        else:
            warnings.append("No header keywords found. Guessing header is the last line of metadata.")
            header_line_index = data_start_index - 1
            channel_names = [name.strip() for name in lines[header_line_index].strip().split('\t')]

        if len(channel_names) == num_data_cols - 1:
            channel_names.insert(0, 'time')
            print("Note: 'Time' column was automatically added to the header.")

# --- DEBUGGING OUTPUT ---
print(f"Detected header row on line {header_line_index + 1 if header_line_index != -1 else 'N/A'}.")
print(f"   -> Final Channel Names: {channel_names}")
print(f"Detected data start on line {data_start_index + 1 if data_start_index != -1 else 'N/A'}.")
if warnings:
    print("Warnings:", warnings)

# --- STEP 3: CONVERT DATA TO PANDAS DATAFRAME ---
if 'data_start_index' in locals() and data_start_index != -1:
    
    final_col_names = [name for name in channel_names if name]
    
    # Check if a comments column should exist based on the presence of '#*'.
    has_comments = any('#*' in line for line in lines[data_start_index:])
            
    if has_comments:
        final_col_names.append('comments')
    
    data_rows = []
    
    for line in lines[data_start_index:]:
        fields = line.strip().split('\t')
        comment_part = None
        
        # Check if the last field contains the comment delimiter
        if has_comments and fields and '#*' in fields[-1]:
            comment_field = fields[-1]
            fields = fields[:-1]
            
            # Split the comment from the data
            parts = comment_field.split('#*')
            data_part = parts[0].strip()
            comment_part = '#*' + parts[1].strip()
            
            # Re-add the data part if it exists
            if data_part:
                fields.append(data_part)
        
        # Append the comment part, if found
        fields.append(comment_part)
        
        # Truncate or pad as needed
        final_num_cols = len(final_col_names)
        if len(fields) > final_num_cols:
            fields = fields[:final_num_cols]
        while len(fields) < final_num_cols:
            fields.append(None)
            
        data_rows.append(fields)

    raw_df = pd.DataFrame(data_rows, columns=final_col_names)

    print("Successfully created DataFrame from raw data.")
    print("DataFrame shape:", raw_df.shape)
    print("\n--- DataFrame Preview (as strings) ---")
    print(raw_df.head())
    
    # --- STEP 4: ANALYSIS ---
    print("\n--- Analysis of Comments Column ---")
    
    if 'comments' in raw_df.columns:
        unique_comments = raw_df['comments'].unique()
        print(f"Unique comments: {unique_comments}")
    
        block_start_count = raw_df['comments'].astype(str).str.contains('block', case=False).sum()
        print(f"Number of 'block_start' comments: {block_start_count}")
    
        stimulus_count = raw_df['comments'].astype(str).str.contains('stimulus', case=False).sum()
        print(f"Number of 'stimulus' comments (left or right): {stimulus_count}")
    
        response_count = raw_df['comments'].astype(str).str.contains('response', case=False).sum()
        print(f"Number of 'response' comments (left or right): {response_count}")
    else:
        print("No comments column was created in the DataFrame.")
        
else:
    print("Skipping data processing and analysis because data start or header was not found.")