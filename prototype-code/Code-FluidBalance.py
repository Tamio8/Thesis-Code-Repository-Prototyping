'''
Prototype 3: Fluid Balance Tracker

Purpose:
This prototype visualises fluid balance over time for ICU patients. It supports scenario testing by
allowing users to input a planned fluid change and assess whether it would push a patient beyond a
threshold (e.g., 5% of body weight) to simulate the prediction of fluid overload.

Key Features:
    Interactive Line Graph showing cumulative fluid balance over time in 15-minute intervals.
    User Input Options for planned fluid change and custom fluid overload thresholds.
    Warning Display when projected fluid balance exceeds a set percentage of body weight.
    Patient Selector allowing exploration of multiple pre-filtered patients.
    Normalised Timeline (start aligned to Day 1 for each patient to support comparison).

Data Sources:
    MIMIC-IV inputevents, outputevents, ingredientevents, and chartevents tables.
    Derived fluid balance per 15-min intervals.
    Admission weight used to calculate personalised overload thresholds.

Technology Stack:
    Python, Pandas, Dash (by Plotly), Dash Bootstrap Components, Plotly Express.

Prototype runs locally in a browser.

How to Run:
    Install required packages (dash, dash-bootstrap-components, pandas, plotly).
    Place CSVs for input/output/weight in the same directory.
    Run the script in Python.
    App launches at http://127.0.0.1:8054.
'''

#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go
import threading
import webbrowser
import dash_bootstrap_components as dbc


# patient cohort is filtered after:
# cohort: adult, 24h icu, 1st icu admission, sepsis3, after hadm_id, not immunocompromised, some kind of burn in diagnosis, invasiveVent, IV

df_ingredientevents = pd.read_csv('df_cohort_vent_iv_ingredientevents')
df_inputevents = pd.read_csv('df_cohort_vent_iv_inputevents')
df_outputevents = pd.read_csv('df_cohort_vent_iv_outputevents')
df_items = pd.read_csv('df_d_items')

df_chartevents = pd.read_csv('df_cohort_chartevents')

print(df_items[df_items['unitname'] =='kg'])


df_items.unitname.unique()


# get admission weight
df_weight = df_chartevents[df_chartevents['itemid'] == 226512]


df_ingredientevents.amountuom.unique()
# decided to use only ml
# decided to use starttime


df_inputevents.amountuom.unique()


df_inputevents.totalamountuom.unique()
# delete nan, use only ml
# use starttime

df_outputevents.valueuom.unique()
# use all
# use charttime

input_nan_count = df_inputevents['totalamountuom'].isna().sum()
print(input_nan_count)
# out of 16354, 3212 don't have ml as total amount

# use stay_id, datetime, amount only

df_ingredientevents = df_ingredientevents.loc[df_ingredientevents['amountuom'].str.contains('ml', case=False, na=False)] # only use rows that are ml
df_ingredient = df_ingredientevents[['stay_id', 'starttime', 'amount']].rename(columns={'stay_id': 'id', 'starttime': 'datetime', 'amount': 'input_ml'})
df_ingredient = df_ingredient.dropna(subset=['input_ml']) # delete row with na - 3212

df_inputevents = df_inputevents.loc[df_inputevents['totalamountuom'].str.contains('ml', case=False, na=False)] # only use rows that are ml
df_input = df_inputevents[['stay_id', 'starttime', 'totalamount']].rename(columns={'stay_id': 'id', 'starttime': 'datetime', 'totalamount': 'input_ml'})
df_input = df_input.dropna(subset=['input_ml']) # delete row with na - 3212

df_output = df_outputevents[['stay_id', 'charttime', 'value']].rename(columns={'stay_id': 'id', 'charttime': 'datetime', 'value': 'output_ml'})
df_output = df_output.dropna(subset=['output_ml']) # delete row with na - 3212

# normalise datetime to start at the 01-01-2000 00:00

# find the minimum datetime for each ID
df_output['datetime'] = pd.to_datetime(df_output['datetime'])
min_datetime_output = df_output.groupby('id')['datetime'].transform('min')
# normalize datetime: Shift so that min datetime per ID is "2000-01-01 00:00"
df_output['datetime'] = pd.to_datetime('2000-01-01') + (df_output['datetime'] - min_datetime_output)

# find the minimum datetime for each ID
df_input['datetime'] = pd.to_datetime(df_input['datetime'])
min_datetime_input = df_input.groupby('id')['datetime'].transform('min')
# normalize datetime: Shift so that min datetime per ID is "2000-01-01 00:00"
df_input['datetime'] = pd.to_datetime('2000-01-01') + (df_input['datetime'] - min_datetime_input)

# find the minimum datetime for each ID
df_ingredient['datetime'] = pd.to_datetime(df_ingredient['datetime'])
min_datetime_ingredient = df_ingredient.groupby('id')['datetime'].transform('min')
# normalize datetime: Shift so that min datetime per ID is "2000-01-01 00:00"
df_ingredient['datetime'] = pd.to_datetime('2000-01-01') + (df_ingredient['datetime'] - min_datetime_ingredient)

# merge input and ingredient
df_combined_input = pd.concat([df_input, df_ingredient], ignore_index=True).sort_values(['id', 'datetime'])

# Calculate Fluid Balance
# Resample to hourly intervals to ensure every hour has a value.
# Fill missing input/output with 0 (since no measurement means no change) and more than one value per hour as sum.

df_combined_input['datetime'] = pd.to_datetime(df_combined_input['datetime'])

# Resample function
def resample_to_hourly_input(df):
    return (
        df.set_index('datetime')
        .groupby('id')
        .resample('15T')['input_ml']# Resample to 15-minute intervals
        .sum() # Sum the values if there are multiple measurements in that time bin
        .fillna(0) # Fill missing values with 0
        .reset_index()
    )

def resample_to_hourly_output(df):
    return (
        df.set_index('datetime')
        .groupby('id')
        .resample('15T')['output_ml']
        .sum()
        .fillna(0)
        .reset_index()
    )

# Apply resampling
df_combined_input_hourly = resample_to_hourly_input(df_combined_input)
df_output_hourly = resample_to_hourly_output(df_output)

# display id to check
id = 37189090 # 26200962 hadm_id

df_input_id = df_combined_input_hourly[df_combined_input_hourly['id'] == id]
df_output_id = df_output_hourly[df_output_hourly['id'] == id]

print(df_input_id)
print(df_output_id)

# Merge input and output together

df_merged = pd.merge(df_combined_input_hourly, df_output_hourly, on=['id', 'datetime'], how='outer')
df_merged.fillna(0, inplace=True) # fill missing values with 0

# Calculate fluid balance
# Add a new column for input - output
df_merged['fluid_balance'] = df_merged['input_ml'] - df_merged['output_ml']


# forward fill df_merged so that the last known value persists until the next recorded entry.
df_merged_filled = df_merged.copy()
df_merged_filled['fluid_balance'] = df_merged_filled['fluid_balance'].mask(df_merged_filled['fluid_balance'] == 0.00).ffill()

df_merged_filled_id = df_merged_filled[df_merged_filled['id'] == id]

print(df_merged_filled_id)

# Create a 'day_offset' column (e.g., Day 1, Day 2, etc.) while keeping hours
df_merged_filled_id['day_offset'] = (df_merged_filled_id['datetime'] - df_merged_filled_id['datetime'].min()).dt.days + 1
# Extract hour and minute as a string for time part
df_merged_filled_id['time_part'] = df_merged_filled_id['datetime'].dt.strftime('%H:%M')  # Hours and minutes as string
# Combine day_offset and time_part to create a custom x-axis label
df_merged_filled_id['x_label'] = 'Day ' + df_merged_filled_id['day_offset'].astype(str) + ' ' + df_merged_filled_id['time_part']

# get weight of patient (time of admission)
# fluid overload is an increase in body weight of at least 5-10%,
# or a positive fluid balance of the same magnitude when fluid intake and urine output are measured (doi: 10.3389/fvets.2021.668688)

# Get weight for id
id_weight = df_weight[df_weight.stay_id == id].valuenum.iloc[0]

# Calculate thresholds
threshold_5_percent = id_weight * 0.05 * 1000
threshold_10_percent = id_weight * 0.10 * 1000

# plot like above
df_merged_filled_id['day_offset'] = (df_merged_filled_id['datetime'] - df_merged_filled_id['datetime'].min()).dt.days + 1
# Extract hour and minute as a string for time part
df_merged_filled_id['time_part'] = df_merged_filled_id['datetime'].dt.strftime('%H:%M')  # Hours and minutes as string
# Combine day_offset and time_part to create a custom x-axis label
df_merged_filled_id['x_label'] = 'Day ' + df_merged_filled_id['day_offset'].astype(str) + ' ' + df_merged_filled_id['time_part']


# SET UP DASH

def run_app(app, port):  
    threading.Thread(target=app.run_server, kwargs={'port': port, 'debug': False, 'use_reloader': False}).start() 
    webbrowser.open_new(f'http://127.0.0.1:{port}')  

# Load your full dataset here
# Example: df = pd.read_csv("fluid_balance.csv")
df = df_merged_filled.copy()

# patient weight dictionary
patient_weights = df_weight.set_index('stay_id').to_dict()['valuenum']

# patient names dictionary
patient_names = {id_: f"Patient {id_}" for id_ in df['id'].unique()} 

df_merged['day_offset'] = (df_merged['datetime'] - df_merged['datetime'].min()).dt.days + 1
# Extract hour and minute as a string for time part
df_merged['time_part'] = df_merged['datetime'].dt.strftime('%H:%M')  # Hours and minutes as string
# Combine day_offset and time_part to create a custom x-axis label
df_merged['x_label'] = 'Day ' + df_merged['day_offset'].astype(str) + ' ' + df_merged['time_part']

# Initialize the app
app_fluid = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app_fluid.layout = html.Div([
    html.H1("Fluid Balance"),
    
    html.Label("Select Patient ID:"),
    dcc.Dropdown(
        id='patient-dropdown',
        options=[{'label': str(i), 'value': i} for i in df['id'].unique()],
        value=df['id'].unique()[0]
    ),
    
    html.Div(id='patient-info', style={'font-size': '18px', 'margin': '10px 0'}),  
    
    html.Div([
        html.Div([
            html.Label("Change (ml):"),
            dcc.Input(id='planned-fluid', type='number', value=0, style={'margin-right': '20px'})
        ], style={'display': 'inline-block', 'vertical-align': 'top'}),
        
        html.Div([
            html.Label("Enter Threshold Percentage (% of body weight):"),  
            dcc.Input(id='threshold-percentage', type='number', value=5, step=0.1) 
        ], style={'display': 'inline-block', 'vertical-align': 'top'})
    ], style={'margin-bottom': '30px'}),
    
    html.Div(id='warning-text', style={'color': 'red', 'margin': '10px 0'}),
    
    dcc.Graph(id='fluid-balance-graph')
])

@app_fluid.callback(
    Output('fluid-balance-graph', 'figure'),
    Output('warning-text', 'children'),
    Output('patient-info', 'children'),  
    Input('patient-dropdown', 'value'),
    Input('planned-fluid', 'value'),
    Input('threshold-percentage', 'value')  
)

def update_graph(patient_id, planned_fluid, threshold_percentage):
    patient_df = df[df['id'] == patient_id].copy()
    
    # Sort by time
    patient_df['datetime'] = pd.to_datetime(patient_df['datetime'])
    patient_df = patient_df.sort_values('datetime')
    
    # Get patient weight
    weight_kg = patient_weights.get(patient_id, 70)

    # Get patient name  
    patient_name = patient_names.get(patient_id, "Unknown") 

    # Display patient info  
    patient_info = f"Patient ID: {patient_id} | Name: {patient_name} | Weight: {weight_kg} kg"  
    
    # # Calculate thresholds
    # threshold_5 = weight_kg * 0.05 * 1000
    # threshold_10 = weight_kg * 0.10 * 1000
    # Calculate thresholds
    threshold_custom = weight_kg * (threshold_percentage / 100) * 1000  
    
    # Last known time and balance
    last_time = patient_df['datetime'].max()
    last_balance = patient_df.loc[patient_df['datetime'] == last_time, 'fluid_balance'].values[0]
    
    # Predicted balance after adding fluid
    predicted_balance = last_balance + planned_fluid

    # Build figure
    fig = go.Figure()

    # Actual fluid balance trace
    fig.add_trace(go.Scatter(
        x=patient_df['datetime'],
        y=patient_df['fluid_balance'],
        mode='lines',
        name=f"Patient {patient_id} Fluid Balance",
        line=dict(color='purple')
    ))

    # Threshold lines
    # fig.add_hline(y=threshold_5, line_dash="dot", line_color="orange", annotation_text="5% Threshold")
    # fig.add_hline(y=threshold_10, line_dash="dot", line_color="red", annotation_text="10% Threshold")
    # fig.add_hline(y=0, line_dash="dash", line_color="black")
    fig.add_hline(y=threshold_custom, line_dash="dot", line_color="blue", annotation_text=f"{threshold_percentage}% Threshold")  
    fig.add_hline(y=0, line_dash="dash", line_color="black")

    # Predicted value as an extension
    fig.add_trace(go.Scatter(
        x=[last_time, last_time + pd.Timedelta(hours=1)], # 'predicts' 1h into the future
        y=[last_balance, predicted_balance],
        mode='lines+markers',
        name="Predicted Balance",
        line=dict(color='green')
    ))

    # Layout
    fig.update_layout(
        title=f"Fluid Balance for Patient {patient_id}",
        xaxis_title="Time",
        yaxis_title="Fluid Balance (ml)",
        height=500,
        width=900
    )

    # Check for warning
    warning = ""
    if predicted_balance >= threshold_custom:
        warning = "Warning: Predicted fluid balance exceeds the 5% threshold!"

    return fig, warning, patient_info


# Run the app - calls function from above
run_app(app_fluid, 8054)


# drop rate is not considered here!

