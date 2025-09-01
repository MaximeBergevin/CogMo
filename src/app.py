# ==============================================================================
# CogMo Toolkit - Main Application
# This Dash application provides an interface for uploading and analyzing
# psychophysiological signal data and associated condition order files.
# ==============================================================================

# ==============================================================================
# --- IMPORTS AND APP INITIALIZATION ---
# ==============================================================================
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import io
import base64
from pathlib import Path
import os
import shutil
import threading
import webview
import pandas as pd
from get_condition_lookup import get_condition_lookup
from load_signal import load_signal
import time

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])


# ==============================================================================
# --- APP LAYOUT (UI) ---
# ==============================================================================
app.layout = dbc.Container([
    html.H1("CogMo toolkit", className="my-3"),
    dbc.Tabs([
        # First Tab: Upload Data
        dbc.Tab(
            label="Upload Data",
            children=[
                html.Div([
                    dbc.Row(className="g-2", children=[
                        dbc.Col(
                            html.Div([
                                html.H4("Upload your data file"),
                                dcc.Upload(
                                    id='upload-signal-data',
                                    children=html.Div([
                                        'Signal Data File'
                                    ]),
                                    style={
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                    },
                                    multiple=False
                                ),
                            ]),
                            width=6,
                        ),
                        dbc.Col(
                            html.Div([
                                # A placeholder to align the button with the other one
                                html.H4(html.Br()),
                                dcc.Upload(
                                    id='upload-condition-order',
                                    children=html.Div([
                                        'Condtion Order File'
                                    ]),
                                    style={
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                    },
                                    multiple=False
                                ),
                            ]),
                            width=6,
                        ),
                    ]),
                    # --- Baseline Reference values ---
                    html.H4("Baseline Reference values", className="mt-4"),
                    dbc.Row([
                        dbc.Col(
                            dbc.FormFloating([
                                dbc.Input(type="number", id="input-mvf-left", placeholder="Left"),
                                dbc.Label("Maximum voluntary force (Left)")
                            ]),
                            width=6
                        ),
                        dbc.Col(
                            dbc.FormFloating([
                                dbc.Input(type="number", id="input-mvf-right", placeholder="Right"),
                                dbc.Label("Maximum voluntary force (Right)")
                            ]),
                            width=6
                        ),
                    ]),
                    dbc.Row([
                        dbc.Col(
                            dbc.FormFloating([
                                dbc.Input(type="number", id="input-rfd-left", placeholder="Left"),
                                dbc.Label("Peak rate of force development (Left)")
                            ]),
                            width=6
                        ),
                        dbc.Col(
                            dbc.FormFloating([
                                dbc.Input(type="number", id="input-rfd-right", placeholder="Right"),
                                dbc.Label("Peak rate of force development (Right)")
                            ]),
                            width=6
                        ),
                    ]),
                    # --- Output Message ---
                    dcc.Loading(
                        id="loading-output-message",
                        type="default",
                        children=html.Div(id='upload-output-message', className="mt-3"),
                    )
                ], className="p-3")
            ]
        ),
        # Second Tab: Baseline Noise
        dbc.Tab(
            label="Baseline Noise",
            children=[
                html.Div([
                    html.H4("Content for Baseline Noise Tab"),
                ], className="p-3")
            ]
        ),
        # Third Tab: Analyses Option
        dbc.Tab(
            label="Analyses Option",
            children=[
                html.Div([
                    html.H4("Content for Analyses Option Tab"),
                ], className="p-3")
            ]
        ),
        # Fourth Tab: Trial Viewer
        dbc.Tab(
            label="Trial Viewer",
            children=[
                html.Div([
                    html.H4("Content for Trial Viewer Tab"),
                ], className="p-3")
            ]
        ),
    ])
], fluid=False, style={'width': '1000px', 'overflow-x': 'hidden'})

# Add dcc.Store components to save the identified comment types and reference values
app.layout.children.append(dcc.Store(id='block-comments-store'))
app.layout.children.append(dcc.Store(id='stimulus-comments-store'))
app.layout.children.append(dcc.Store(id='mvf-left-store'))
app.layout.children.append(dcc.Store(id='mvf-right-store'))
app.layout.children.append(dcc.Store(id='rfd-left-store'))
app.layout.children.append(dcc.Store(id='rfd-right-store'))


# ==============================================================================
# --- CALLBACKS (Backend Logic) ---
# ==============================================================================
@app.callback(
    Output('upload-output-message', 'children'),
    Output('block-comments-store', 'data'),
    Output('stimulus-comments-store', 'data'),
    Output('mvf-left-store', 'data'),
    Output('mvf-right-store', 'data'),
    Output('rfd-left-store', 'data'),
    Output('rfd-right-store', 'data'),
    Input('upload-signal-data', 'contents'),
    Input('upload-condition-order', 'contents'),
    Input('input-mvf-left', 'value'),
    Input('input-mvf-right', 'value'),
    Input('input-rfd-left', 'value'),
    Input('input-rfd-right', 'value'),
    State('upload-signal-data', 'filename'),
    State('upload-signal-data', 'last_modified'),
    State('upload-condition-order', 'filename'),
    State('upload-condition-order', 'last_modified')
)
def update_output(signal_contents, condition_contents, mvf_left, mvf_right, rfd_left, rfd_right, signal_filename, signal_last_modified, condition_filename, condition_last_modified):
    ctx = dash.callback_context
    if not ctx.triggered:
        return (
            html.Div("Please upload a file to begin."),
            None,
            None,
            mvf_left,
            mvf_right,
            rfd_left,
            rfd_right
        )
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Initialize return values
    message = html.Div("Please upload a file to begin.")
    block_comments_data = None
    stimulus_comments_data = None

    # 1. Logic for Signal Data Upload
    # -------------------------------
    if trigger_id == 'upload-signal-data' and signal_contents is not None:
        content_type, content_string = signal_contents.split(',')
        
        # Save the uploaded file temporarily
        temp_dir = Path("./temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        temp_filepath = temp_dir / signal_filename
        
        try:
            decoded = base64.b64decode(content_string)
            with open(temp_filepath, 'wb') as f:
                f.write(decoded)
            
            # Load and process the signal data
            df, comment_summary = load_signal(temp_filepath)
            
            # Find the lowest block count and total stimulus count, and capture comment types
            num_blocks = float('inf')
            num_stimulus = 0
            block_comments = []
            stimulus_comments = []

            if comment_summary:
                for comment_type, count in comment_summary.items():
                    comment_lower = comment_type.lower()
                    if 'block' in comment_lower:
                        num_blocks = min(num_blocks, count)
                        if comment_type not in block_comments:
                            block_comments.append(comment_type)
                    elif 'stimulus' in comment_lower:
                        num_stimulus += count
                        if comment_type not in stimulus_comments:
                            stimulus_comments.append(comment_type)
            
            # Store the comment lists
            block_comments_data = block_comments
            stimulus_comments_data = stimulus_comments

            # Adjust the message to reflect the new logic
            blocks_message = f"Lowest block count detected: {num_blocks}" if num_blocks != float('inf') else "No blocks detected."
            stimulus_message = f"Total stimulus trials detected: {num_stimulus}" if num_stimulus != 0 else "No stimulus trials detected."
            block_comments_message = f"Block comments found: {', '.join(block_comments)}" if block_comments else "No block comments found."
            stimulus_comments_message = f"Stimulus comments found: {', '.join(stimulus_comments)}" if stimulus_comments else "No stimulus comments found."

            message = html.Div([
                html.H5("Data successfully uploaded"),
                html.P(blocks_message),
                html.P(stimulus_message),
                html.P(block_comments_message),
                html.P(stimulus_comments_message)
            ])

        except Exception as e:
            message = html.Div(f'There was an error processing this file: {e}', className="text-danger")
        finally:
            # Clean up the temporary file and directory
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return message, block_comments_data, stimulus_comments_data, mvf_left, mvf_right, rfd_left, rfd_right


    # 2. Logic for Condition Order File Upload
    # ----------------------------------------
    elif trigger_id == 'upload-condition-order' and condition_contents is not None:
        content_type, content_string = condition_contents.split(',')
        try:
            decoded = base64.b64decode(content_string)
            
            # Read the file based on its extension
            if condition_filename.endswith('.xlsx'):
                df = pd.read_excel(io.BytesIO(decoded))
            elif condition_filename.endswith('.csv'):
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            else:
                message = html.Div("Unsupported file format. Please upload an .xlsx or .csv file.", className="text-danger")

            # Use the get_condition_lookup function
            lookup_result = get_condition_lookup(df)
            
            if lookup_result:
                participant_id = lookup_result['participant_id']
                condition_counts_df = lookup_result['condition_counts']
                
                # Format the output message with participant ID and a table of condition counts
                message = html.Div([
                    html.H5("Condition file successfully processed"),
                    html.P(f"Participant ID: {participant_id}"),
                    html.P("Condition Counts:"),
                    html.Div(html.Pre(condition_counts_df.to_string()))
                ])
            else:
                message = html.Div("Could not extract condition data from the file. Please check column names.", className="text-danger")

        except Exception as e:
            message = html.Div(f"Error processing condition file: {e}", className="text-danger")
    
    return message, block_comments_data, stimulus_comments_data, mvf_left, mvf_right, rfd_left, rfd_right


# ==============================================================================
# --- MAIN APP EXECUTION ---
# ==============================================================================
def run_app_server():
    # Note: debug=False is important for packaged apps
    app.run(debug=False) 

if __name__ == '__main__':
    # Run the Dash server in a separate thread
    server_thread = threading.Thread(target=run_app_server)
    server_thread.daemon = True
    server_thread.start()

    # Create and start the pywebview window
    webview.create_window(
        'CogMo Toolkit', 
        'http://127.0.0.1:8050/',
        width=1000,
        height=750
    )
    webview.start()
