'''
Similarity Prototype for ICU Patient Comparison

Purpose:
This prototype explores methods for identifying similar ICU patients based on their time-series vital signs data.
It offers both quantitative analysis (e.g., similarity scores) and visual comparison (line plots, statistical overlays),
supporting clinician reflection on cases that share physiological trajectories.

Key Features:

    Time-Series Resampling:
        Patient data was resampled into consistent time intervals per vital sign based on median measurement
        frequency (e.g., blood pressure = 60 min, glucose = 236 min). This standardisation allows a simplified
        comparison across patients.
    Data Handling:
        Missing values were linearly interpolated within each patient's trajectory.
        Each patient's time-series was normalised to start at a fixed baseline (2000-01-01) for alignment.
    Similarity Measures:
        Cosine Similarity (primary): Compares the pattern of physiological change over time, rather than magnitude.
        Chosen for its robustness to different baseline levels and its focus on shape similarity.
        MSE (Mean Squared Error): Captures magnitude differences but is sensitive to scale.
        Euclidean Distance: Captures raw difference but is less suitable for misaligned or unevenly scaled data.
        Optional: random and dissimilar patients can be compared.
    User Interface:
        Implemented in Dash.
        Allows exploration of similarities via multiple methods.
        Displays interactive plots comparing target vs. similar or dissimilar patients, including mean and
        standard deviation overlays.

Design Notes & Shortcuts:
    Resampling interval per feature was manually defined from empirical quantiles (e.g., median time difference).
    Temporal alignment was done via binning rather than interpolation to nearest neighbor to preserve time grid
    consistency.
    Standardisation (Z-score) was applied after binning to make features comparable regardless of scale.
    For similarity computation, missing time bins were filled forward, then with zeros as a fallback.
    Cosine similarity was preferred for visual plausibility and stability in exploratory testing.
    The visualisation tool was designed for hypothesis generation, not clinical deployment.

Requirements:
    Python >= 3.8
    dash, pandas, numpy, scikit-learn, plotly, seaborn, matplotlib
'''
# pip install dash

# import libraries
import pandas as pd
import numpy as np
import datetime
import random

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import dash
from dash import dcc, html, Input, Output
import dash_extensions.enrich as dx
from dash_extensions.enrich import DashProxy
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
import threading
import webbrowser

# import datasets

# patient cohort is filtered after:
# df_cohort_stays_septic_immune_burns: adult, 24h icu, 1st icu admission, sepsis3, after hadm_id, not immunocompromised, some kind of burn in diagnosis
# df_cohort_vitalsign: vitalsigns of those patients
# df_cohort_medication: medication of those patients

df_cohort_stays_septic_immune_burns = pd.read_csv('df_cohort_stays_septic_immune_burns')
df_cohort_vitalsign = pd.read_csv('df_cohort_vitalsign')
df_discharge_note = pd.read_csv('df_discharge_note')
df_diagnosis = pd.read_csv('df_diagnosis')

subject_id = df_diagnosis.subject_id.unique().tolist()
print(len(subject_id))

# rename columns to include measuring unit
df_cohort_vitalsign.rename(columns={
    "heart_rate": "Heart Rate (bpm)", 
    "sbp": "Blood Pressure, Systolic (mmHg)", 
    "dbp": "Blood Pressure, Diastolic (mmHg)",
    "mbp": "Blood Pressure, Mean (mmHg)",
    "resp_rate": "Respiratory Rate (min-1)",
    "temperature": "Temperature (celsius)",
    "spo2": "SpO2 (%)",
    "glucose": "Glucose (mmol/L)"}, inplace=True)
# columns of interest: 'Heart Rate (bpm)', 'Blood Pressure, Systolic (mmHg)', 'Blood Pressure, Diastolic (mmHg)', 'Blood Pressure, Mean (mmHg)', 'Respiratory Rate (min-1)', 'Temperature (celsius)', 'SpO2 (%)', 'Glucose (mmol/L)'

# PLOT HR FOR ALL PATIENTS

#create table for heartrate only for all patients
df_cohort_vitalsign_HR = df_cohort_vitalsign[df_cohort_vitalsign['Heart Rate (bpm)'].notna()][['stay_id', 'charttime', 'Heart Rate (bpm)']]

# Group by ID and normalize datetime
df_cohort_vitalsign_HR_normalised = []
for id, group in df_cohort_vitalsign_HR.groupby('stay_id'):
    sorted_group = group.sort_values('charttime')
    normalized = sorted_group.reset_index()
    normalized['stay_id'] = id
    normalized['charttime'] = normalized.index + 1  # Start from day 1
    df_cohort_vitalsign_HR_normalised.append(normalized)

df_cohort_vitalsign_HR_normalised = pd.concat(df_cohort_vitalsign_HR_normalised)

#Reshape Data for Plotting

df_cohort_vitalsign_HR_normalised_melted = df_cohort_vitalsign_HR_normalised.melt(id_vars=['stay_id', 'charttime'], var_name='Heart Rate (bpm)', value_name='datetime_value')

# normalise datetime per ID
df_cohort_vitalsign_HR['charttime'] = pd.to_datetime(df_cohort_vitalsign_HR['charttime'])
df_cohort_vitalsign_HR['relative_time'] = df_cohort_vitalsign_HR.groupby('stay_id')['charttime'].transform(lambda x: (x - x.min()).dt.total_seconds() / 3600)  # Convert to hours

# Plot
plt.figure(figsize=(12, 6))
for key, grp in df_cohort_vitalsign_HR.groupby('stay_id'):
    plt.scatter(grp['relative_time'], grp['Heart Rate (bpm)'], label=f'stay_ID {key}', alpha=0.7)

# Formatting
plt.xlabel("Time since first measurement (hours)")
plt.ylabel("Heartrate (bpm)")
plt.title("Distribution of Heartrate Across Normalized Time")
plt.legend(title="stay_id", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()

# Select data to be plotted and normalise timeseries

# set df_cohort_vitalsign to be plotted
df_cohort_vitalsign['charttime'] = pd.to_datetime(df_cohort_vitalsign['charttime'])

df_cohort_vitalsign_normalised = df_cohort_vitalsign[['stay_id', 'charttime', 'Heart Rate (bpm)', 'Blood Pressure, Systolic (mmHg)', 'Blood Pressure, Diastolic (mmHg)', 'Blood Pressure, Mean (mmHg)', 'Respiratory Rate (min-1)', 'Temperature (celsius)', 'SpO2 (%)', 'Glucose (mmol/L)']]
 
# Normalize datetime (set each ID's first date to "Day 1")
df_cohort_vitalsign_normalised['day_offset'] = df_cohort_vitalsign_normalised.groupby('stay_id')['charttime'].transform(lambda x: (x - x.min()).dt.days + 1)

# Extract time component
df_cohort_vitalsign_normalised['time_part'] = df_cohort_vitalsign_normalised['charttime'].dt.time

# Create normalized datetime starting from a fixed base date
base_date = pd.Timestamp("2000-01-01")
df_cohort_vitalsign_normalised['normalized_datetime'] = df_cohort_vitalsign_normalised['day_offset'].apply(lambda d: base_date + pd.Timedelta(days=d - 1))
df_cohort_vitalsign_normalised['normalized_datetime'] = df_cohort_vitalsign_normalised['normalized_datetime'].astype(str) + ' ' + df_cohort_vitalsign_normalised['time_part'].astype(str)
df_cohort_vitalsign_normalised['normalized_datetime'] = pd.to_datetime(df_cohort_vitalsign_normalised['normalized_datetime'])

# EXPLORE CORRELATION

df_values = df_cohort_vitalsign_normalised[['Heart Rate (bpm)', 'Blood Pressure, Systolic (mmHg)', 'Blood Pressure, Diastolic (mmHg)', 'Blood Pressure, Mean (mmHg)', 'Respiratory Rate (min-1)', 'Temperature (celsius)', 'SpO2 (%)', 'Glucose (mmol/L)']]
# Standardize the data (i.e., scale each column to have mean=0 and std=1)
scaler = StandardScaler()
df_values_scaled = scaler.fit_transform(df_values)

# Convert the scaled data back to a DataFrame
df_scaled = pd.DataFrame(df_values_scaled, columns=df_values.columns)

# Step 3: Calculate the correlation matrix
corr_matrix = df_scaled.corr()

# Step 4: Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.3f', linewidths=0.5)
plt.title("Correlation Heatmap (Standardized Values)")
plt.show()

# Heatmap
# not surprising, BP related, and HR and resp_rate a little bit. and a bit between HR and dbp
# also, correlation between timeseries is tricky

# SET UP DASH FOR A DIFFERENT BROWSER

# Function to run each app in a new thread and open in a new window
def run_app(app, port):
    threading.Thread(target=app.run_server, kwargs={'port': port, 'debug': False, 'use_reloader': False}).start()
    webbrowser.open_new(f'http://127.0.0.1:{port}')

# Function for Visualisation of a dataset that has the columns: 
# id, datetime, heart_rate, sbp, dbp, mbp, resp_rate, temperature, spo2, glucose, day_offset, time_part

def run_vital_sign_visualisation(df, port):
    # Initialize DashProxy app
    app_original = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    
    # Define features
    feature_options = ['Heart Rate (bpm)', 'Blood Pressure, Systolic (mmHg)', 'Blood Pressure, Diastolic (mmHg)', 'Blood Pressure, Mean (mmHg)', 'Respiratory Rate (min-1)', 'Temperature (celsius)', 'SpO2 (%)', 'Glucose (mmol/L)']
    
    # Layout
    df_name = [name for name, value in globals().items() if value is df]
    app_original.layout = html.Div([
        html.H1("Vital Sign Data Visualization - {}".format(df_name[0])),
        
        # Feature selection
        html.Label("Select Feature:"),
        dcc.Dropdown(
            id="feature-selector",
            options=[{"label": f, "value": f} for f in feature_options],
            value=feature_options[0]  # Default to first feature
        ),
    
        # ID selection (Text Input)
        html.Label("Enter ID (comma-separated for multiple):"),
        dcc.Input(
            id="id-selector",
            type="text",
            placeholder="enter id"
        ),
    
        # Number of IDs to Display (Numerical Input)
        html.Label("Number of IDs to Display:"),
        dcc.Input(
            id="id-count-input",
            type="number",
            min=1,
            max=len(df["id"].unique()),
            value=20  # Default 20
        ),
    
        # Time range slider
        html.Label("Select Time Range (Days):"),
        dcc.RangeSlider(
            id="time-range-slider",
            min=df['day_offset'].min(),
            max=df['day_offset'].max(),
            step=3,
            value=[df['day_offset'].min(), df['day_offset'].max()],
            tooltip={"placement": "bottom", "always_visible": True}  # Show exact values on hover
        ),
    
        # Graph output
        dcc.Graph(id="scatter-plot") #(id="time-series-plot")
    ])
    
    
    # Callback to update graph
    @app_original.callback(
        Output("scatter-plot", "figure"),
        Input("feature-selector", "value"),
        Input("id-selector", "value"),
        Input("id-count-input", "value"),
        Input("time-range-slider", "value")
    )
    def update_plot(selected_feature, entered_ids, id_count, time_range):
        try:
            start_day, end_day = time_range
    
            # Compute start and end dates from slider range
            # start_date = min_date + pd.Timedelta(days=time_range[0])
            # end_date = min_date + pd.Timedelta(days=time_range[1])
    
            # Filter data based on selected timeframe
            df_filtered = df[(df['day_offset'] >= start_day) & (df['day_offset'] <= end_day)]
    
            # Convert entered IDs to a list
            selected_ids = [int(i.strip()) for i in entered_ids.split(",") if i.strip().isdigit()] if entered_ids and isinstance(entered_ids, str) else []
    
            # Apply ID filtering
            if selected_ids:
                df_filtered = df_filtered[df_filtered["id"].isin(selected_ids)]
            else:
                # Select the random IDs
                available_ids = df_filtered['id'].unique()
                id_count = min(id_count, len(available_ids))  # Ensure it doesn't exceed available IDs
                selected_ids = np.random.choice(available_ids, size=id_count, replace=False)
                df_filtered = df_filtered[df_filtered['id'].isin(selected_ids)]
    
             # Convert to long format (melt)
            df_melted = df_filtered.melt(id_vars=['id', 'datetime', 'day_offset'],
                                         value_vars=feature_options, var_name='feature', value_name='value')
             # Drop NaN values
            df_melted = df_melted.dropna(subset=['value'])
            
            # Ensure there is data to display
            if df_melted.empty:
                return px.scatter(title="No Data Available for the Selected Criteria")
    
            # Keep only selected feature
            df_melted = df_melted[df_melted['feature'].isin([selected_feature])]
    
            # Convert ID to string for color mapping
            df_melted['_id'] = df_melted['id'].astype(str)

            # Generate scatter plot with color assigned per ID
            fig = px.scatter(df_melted,
                x='datetime',
                y='value',
                color=df_melted['_id'],
                hover_data=['id', 'day_offset', 'feature', 'datetime'],
                title="Vital Signs Distribution (Hourly)",
                category_orders={'_id': sorted(df_melted['_id'].unique())}
            )
            
             # Update axes labels
            fig.update_xaxes(
                title_text="Time (days, hours, minutes)",
                showgrid=True
            )
            
            fig.update_layout(
                xaxis=dict(
                    tickmode='auto',
                    dtick=1000 * 60 * 60 * 6,
                    tickformat="Day %j\n%H:%M"  # Day of year with hour and minute
                )
            )
    
            fig.update_yaxes(title_text=selected_feature, tickformat=",", showgrid=True)
    
            return fig
    
        except Exception as e:
                print(f"ERROR: {str(e)}")
                return px.scatter(title=f"Error in callback: {str(e)}")
    
    # Run the app - calls function from above
    run_app(app_original, port)

# Prepare for visualisation: df_cohort_vitalsign_normalised

#stay_id becomes id, charttime is dropped, normalized_datetime becomes datetime
df_cohort_vitalsign_normalised_plot = df_cohort_vitalsign_normalised.copy()
df_cohort_vitalsign_normalised_plot.rename(columns={'stay_id': 'id', 'normalized_datetime': 'datetime',}, inplace=True)
df_cohort_vitalsign_normalised_plot = df_cohort_vitalsign_normalised_plot.drop('charttime', axis=1)
df_cohort_vitalsign_normalised_plot.head()

# Call function for visualisation on df_cohort_vitalsign_normalised_plot
# DASH APP - dataset, port
run_vital_sign_visualisation(df_cohort_vitalsign_normalised_plot, 8050)

# GET STATISTICS FOR EACH VITAL SIGN

df_grouped = df_cohort_vitalsign_normalised.groupby('stay_id')

#calculate statistics
statistics_heartrate = df_grouped['Heart Rate (bpm)'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
statistics_sbp = df_grouped['Blood Pressure, Systolic (mmHg)'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
statistics_dbp = df_grouped['Blood Pressure, Diastolic (mmHg)'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
statistics_mbp = df_grouped['Blood Pressure, Mean (mmHg)'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
statistics_resp_rate = df_grouped['Respiratory Rate (min-1)'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
statistics_temperature = df_grouped['Temperature (celsius)'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
statistics_spo2 = df_grouped['SpO2 (%)'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()
statistics_glucose = df_grouped['Glucose (mmol/L)'].agg(['mean', 'median', 'min', 'max', 'std']).reset_index()

# get overall min and max for each value
dfs = {
    "Heart Rate": statistics_heartrate,
    "SBP": statistics_sbp,
    "DPB": statistics_dbp,
    "Respiratory Rate": statistics_resp_rate,
    "Temperature": statistics_temperature,
    "SpO2": statistics_spo2,
    "Glucose": statistics_glucose,
}

# Iterate over each dataframe and print the min and max
for name, df in dfs.items():
    min_value = df["min"].min()
    max_value = df["max"].max()
    mean_value = df["mean"].mean()
    print(f"{name} - Min: {min_value}, Max: {max_value}, Mean: {mean_value}")


# df_cohort_vitalsign_normalised.head()
# 'Day 1' is 2000-01-01 - yyyy-mm-dd


# Statistics about the frequency in which each feature is measured per ID
# resulting df: ID, VitalSign, mean, std
# dataset: df_cohort_vitalsign_normalised['stay_id', 'normalized_datetime', 'charttime', 'heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'spo2', 'glucose']

df_cohort_vitalsign_normalised_subset = df_cohort_vitalsign_normalised[['stay_id', 'normalized_datetime', 'Heart Rate (bpm)', 'Blood Pressure, Systolic (mmHg)', 'Blood Pressure, Diastolic (mmHg)', 'Blood Pressure, Mean (mmHg)', 'Respiratory Rate (min-1)', 'Temperature (celsius)', 'SpO2 (%)', 'Glucose (mmol/L)']]


# Reshape to long format
df_cohort_vitalsign_normalised_long = df_cohort_vitalsign_normalised_subset.melt(id_vars=['stay_id', 'normalized_datetime'], 
                                                                          var_name='feature', value_name='value')

# Drop NaNs since each row has only one feature measured
df_cohort_vitalsign_normalised_long = df_cohort_vitalsign_normalised_long.dropna(subset=['value'])

# Sort values by id, feature, and time
df_cohort_vitalsign_normalised_long = df_cohort_vitalsign_normalised_long.sort_values(by=['stay_id', 'feature', 'normalized_datetime'])

# Calculate time differences per ID and feature (in minutes)
df_cohort_vitalsign_normalised_long['time_diff'] = df_cohort_vitalsign_normalised_long.groupby(['stay_id', 'feature'])['normalized_datetime'].diff().dt.total_seconds() / 60  # Convert to minutes

# Compute mean and std of the frequency (time differences)
df_cohort_vitalsign_frequency_statistics = df_cohort_vitalsign_normalised_long.groupby(['stay_id', 'feature'])['time_diff'].agg(['mean', 'std']).reset_index()

df_cohort_vitalsign_frequency_statistics


# take df_cohort_vitalsign_frequency_statistics
# splitting by 'feature', and then for each of them print the list of stay_id sorted by mean.

# Pivot to reshape the dataframe: ID as index, feature names as columns, and mean values as data
df_cohort_vitalsign_frequency_statistics_mean = df_cohort_vitalsign_frequency_statistics.pivot(index="stay_id", columns="feature", values="mean").reset_index()

# Iterate over each feature, sort by mean, and print the sorted list of (ID, mean)
for feature in df_cohort_vitalsign_frequency_statistics_mean.columns[1:]:  # Skip 'ID' column
    df_cohort_vitalsign_frequency_statistics_mean_sorted = df_cohort_vitalsign_frequency_statistics_mean[['stay_id', feature]].dropna().sort_values(by=feature, ascending=False)
    print(f"\nSorted list for feature '{feature}':\n", df_cohort_vitalsign_frequency_statistics_mean_sorted.to_string(index=False))

# plot distribution of time difference per feature

# Create a new column with time_diff converted to hours
df_cohort_vitalsign_normalised_long["time_diff_hours"] = df_cohort_vitalsign_normalised_long["time_diff"] / 60

# Create a FacetGrid with one graph per feature
g = sns.FacetGrid(df_cohort_vitalsign_normalised_long, col="feature", col_wrap=4, sharex=False, sharey=True, height=2)

# Map the histogram using the new time_diff_hours column
g.map(sns.histplot, "time_diff_hours", kde=True, bins=35)

# Add labels and titles
g.set_axis_labels('Time Difference (hours)', 'Frequency')
g.set_titles("{col_name}")

# Manually set y-axis limits
for ax in g.axes.flat:
    ax.set_ylim(0, 2000)  # Adjust the limit to a reasonable range

# Manually set x-axis limits
for ax in g.axes.flat:
    ax.set_xlim(0, 100)  # Adjust the limit to a reasonable range

plt.tight_layout()  # Adjust the layout to make sure titles and labels fit
plt.show()

# SAME PLOT DIFFERENT SCALE
# plot distribution of time difference per feature

# Create a new column with time_diff converted to hours
df_cohort_vitalsign_normalised_long["time_diff_hours"] = df_cohort_vitalsign_normalised_long["time_diff"] / 60

# Create a FacetGrid with one graph per feature
g = sns.FacetGrid(df_cohort_vitalsign_normalised_long, col="feature", col_wrap=4, sharex=False, sharey=True, height=2)

# Map the histogram using the new time_diff_hours column
g.map(sns.histplot, "time_diff_hours", kde=True, bins=35)

# Add labels and titles
g.set_axis_labels('Time Difference (hours)', 'Frequency')
g.set_titles("{col_name}")

# Manually set y-axis limits
for ax in g.axes.flat:
    ax.set_ylim(0, 200)  # Adjust the limit to a reasonable range

# Manually set x-axis limits
for ax in g.axes.flat:
    ax.set_xlim(0, 30)  # Adjust the limit to a reasonable range

# Apply square root transformation to the y-axis
#for ax in g.axes.flat:
    # ax.set_yscale('log', base=2) -> looks really strange

plt.tight_layout()  # Adjust the layout to make sure titles and labels fit
plt.show()

# SAME PLOT LOG SCALE
# plot distribution of time difference per feature

# Create a new column with time_diff converted to hours
df_cohort_vitalsign_normalised_long["time_diff_hours"] = df_cohort_vitalsign_normalised_long["time_diff"] / 60

# Create a FacetGrid with one graph per feature
g = sns.FacetGrid(df_cohort_vitalsign_normalised_long, col="feature", col_wrap=4, sharex=False, sharey=True, height=2)

# Map the histogram using the transformed time_diff_hours column with log scaling
g.map(sns.histplot, "time_diff_hours", kde=True, bins=30, log_scale=(True, True))

# Add labels and titles
g.set_axis_labels('Time Difference (hours, log scale)', 'Frequency')
g.set_titles("{col_name}")

# Apply log scaling
for ax in g.axes.flat:
    #ax.set_xscale("log")  # Log scale for x-axis
    ax.set_yscale("log")  # Log scale for y-axis

    # Dynamically set reasonable log-spaced tick marks
    ax.set_xticks([0.1, 1, 10])  # Adjust based on your data range
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())  # Keep labels readable

    # Automatically determine reasonable log-scaled y-ticks
    y_min, y_max = ax.get_ylim()
    log_yticks = [y for y in [1, 10, 100, 1000, 10000] if y_min < y < y_max]
    ax.set_yticks(log_yticks)
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())  # Keep labels readable

plt.tight_layout()  # Adjust the layout to make sure titles and labels fit
plt.show()

# compute quantiles of time difference
quantiles = df_cohort_vitalsign_normalised_long.groupby("feature")["time_diff"].quantile([0, 0.25, 0.5, 0.75, 1]).unstack()
quantiles.columns = ["Q0", "Q25", "Q50", "Q75", "Q100"]

print(quantiles)

# Calculate summary statistics for each feature
df_cohort_vitalsign_frequency_statistics_summary_big = df_cohort_vitalsign_frequency_statistics.groupby("feature").agg(
    avg_mean=("mean", "mean"),
    avg_std=("std", "mean"),
    min_mean=("mean", "min"),
    max_mean=("mean", "max"),
    min_std=("std", "min"),
    max_std=("std", "max")
).reset_index()

df_cohort_vitalsign_frequency_statistics_summary_big

# Resample dataset based on Quantiles (frequency in minutes)
# Resample: df_cohort_vitalsign_normalised_subset

# bin sizes based on Q50:
feature_bin_sizes = {
    "Blood Pressure, Diastolic (mmHg)": 60.0,
    "Blood Pressure, Mean (mmHg)": 60.0,
    "Blood Pressure, Systolic (mmHg)": 60.0,
    "Glucose (mmol/L)": 236.0,
    "Heart Rate (bpm)": 60.0,
    "Respiratory Rate (min-1)": 60.0,
    "SpO2 (%)": 60.0,
    "Temperature (celsius)": 240.0
}

# Convert dictionary values to timedelta format
feature_bin_sizes = {k: pd.to_timedelta(v, unit="m") for k, v in feature_bin_sizes.items()}

# Prepare an empty list to store processed data
resampled_data = []

# Process each ID and feature separately
for id_value in df_cohort_vitalsign_normalised_subset['stay_id'].unique():
    df_id = df_cohort_vitalsign_normalised_subset[df_cohort_vitalsign_normalised_subset['stay_id'] == id_value].copy()

    for feature, bin_size in feature_bin_sizes.items():  # Use dictionary values for bin size

        # Select feature-specific data (drop NaN)
        df_feature = df_id[['normalized_datetime', feature]].dropna().copy()

        if df_feature.empty:
            continue  # Skip if no data for this feature and ID

        # Convert bin_size to Pandas timedelta before using it in resampling
        resample_freq = f"{int(pd.Timedelta(bin_size).total_seconds() / 60)}T"

        # Resample using the bin size
        df_feature = df_feature.set_index('normalized_datetime').resample(resample_freq).mean()

        # Group by datetime index and take the mean for bins with multiple measurements
        df_feature[feature] = df_feature.groupby(df_feature.index)[feature].transform('mean')

        # Drop the duplicate values: only keep one value per bin
        df_feature = df_feature.loc[~df_feature.index.duplicated(keep='first')]
        
        # Reset index and add ID column
        df_feature = df_feature.reset_index()
        df_feature.insert(0, "stay_id", id_value)

        # Interpolate missing values (if there are bins without any measurement)
        df_feature[feature] = df_feature[feature].interpolate(method="linear", limit_direction="both")

        # Ensure there is exactly one measurement per bin
        df_feature = df_feature.drop_duplicates(subset=['normalized_datetime'])

        resampled_data.append(df_feature)

# Combine all processed data
df_cohort_vitalsign_normalised_subset_resampled = pd.concat(resampled_data, ignore_index=True)

# Prepare for visualisation: resampled dataset: df_cohort_vitalsign_normalised_subset_resampled

df_resampled = df_cohort_vitalsign_normalised_subset_resampled.copy()
df_resampled.rename(columns={'stay_id': 'id', 'normalized_datetime': 'datetime'}, inplace=True) # Rename columns to make it easier
df_resampled["day_offset"] = df_resampled.groupby('id')['datetime'].transform(lambda x: (x - x.min()).dt.days + 1)

# Extract time component
df_resampled['time_part'] = df_resampled['datetime'].dt.time

# Ensure 'datetime' is in datetime64 format
df_resampled['datetime'] = pd.to_datetime(df_resampled['datetime'], errors='coerce')

# Call function for visualisation on df_cohort_vitalsign_normalised_plot

#DASH APP - dataset,port

run_vital_sign_visualisation(df_resampled, 8051)

#-------------------------------------------------------------------------------------------------------------------------------
# SIMPLE SIMILARITY #
#-------------------------------------------------------------------------------------------------------------------------------

# 1. select id to filter
# 2. filter dataset for selected if and compute the MSE of features
# 3. compute the mean for each ID in the dataset
# 4. calculate the MSE between the selected ID and all other ID
# 5. sort and return top n most similar id

# SIMILARITY: MSE with time - compare based on measurement frequency - ALL

def similarity_mse_time(df, target_id, n=5, target_day=3):
    """
    Find the n most similar IDs to a given target_id based on MSE, considering time-based measurement frequencies.

    Parameters:
    df (pd.DataFrame): DataFrame with 'id', 'datetime', and fruit measurement columns.
    target_id (int or str): The ID to compare against.
    feature_time_bin (dict): Dictionary mapping features to their time aggregation level ('hourly', 'daily', 'weekly').
    n (int): Number of similar IDs to return.

    Returns:
    List of tuples (id, mse) sorted by similarity.
    """

    feature_time_bin = {
    "Blood Pressure, Diastolic (mmHg)": 60.0,
    "Blood Pressure, Mean (mmHg)": 60.0,
    "Blood Pressure, Systolic (mmHg)": 60.0,
    "Glucose (mmol/L)": 236.0,
    "Heart Rate (bpm)": 60.0,
    "Respiratory Rate (min-1)": 60.0,
    "SpO2 (%)": 60.0,
    "Temperature (celsius)": 240.0
}
    
    df = df.drop(columns=['day_offset', 'time_part'], errors='ignore')  

    df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure datetime format
    

    # Dictionary to store aggregated data for each feature based on time bins
    aggregated_dfs = []

    for feature, time_bin in feature_time_bin.items():
        temp_df = df[['id', 'datetime', feature]].copy()

        # Round timestamps down to nearest feature-specific time bin -> consistent time intervals per feature
        temp_df['time_bin'] = temp_df['datetime'].dt.floor(f'{int(time_bin)}min')

        # Compute mean values for each (id, time_bin) -> time alignment per feature
        aggregated_df = temp_df.groupby(['id', 'time_bin'])[feature].mean().reset_index() # list of dfs
        aggregated_dfs.append(aggregated_df)

    # Merge all aggregated feature data into a single DataFrame
    merged_df = aggregated_dfs[0]
    for other_df in aggregated_dfs[1:]:
        merged_df = merged_df.merge(other_df, on=['id', 'time_bin'], how='outer')

    # Extract feature columns
    feature_columns = list(feature_time_bin.keys())

    # Compute mean values for each (id, time_bin) in case there is more than one measurement per time_bin
    mean_values = merged_df.groupby(['id', 'time_bin'])[feature_columns].mean()

    # Forward fill missing values within each ID and fill remaining NaNs with 0
    mean_values = mean_values.groupby('id').ffill().fillna(0)

    # Standardize (Z-score normalization) -> ensure all features are on the same scale, each feature has now a mean of 0, a std of 1
    standardized_means = (mean_values - mean_values.mean()) / mean_values.std()

    # Check if target_id exists
    if target_id not in standardized_means.index.get_level_values('id'):
        raise ValueError(f"ID {target_id} not found in the dataset.")

    # Extract target ID's time-aware means
    target_means = standardized_means.loc[target_id] # containing the time-binned, standardized feature values for the target ID

    # Treat each feature as array over time
    mse_values = {}
    for feature in feature_columns:
        # Extract arrays for the feature (time bins) for target_id and other ids
        target_array = standardized_means.loc[target_id, feature].values
        mse_per_id = {}
    
        for id in standardized_means.index.levels[0]:  # Loop through all IDs
            if id == target_id:
                continue  # Skip target_id itself
    
            other_array = standardized_means.loc[id, feature].values
            # Compute MSE for this feature's time-series (array)
                # This computes the difference between each time bin's value for the target ID and the other ID.
                # For example, if you are comparing the apple feature, this calculates how much the values at each time point differ between the two IDs.
                # Squaring the differences ensures that the error values are positive and gives more weight to larger deviations.
                # This operation highlights larger discrepancies between the target ID and the other ID's time series.
                # After squaring the differences, .mean() computes the average of those squared differences.
                # This gives you the Mean Squared Error (MSE) for the feature between the two IDs. It reflects how much the time-series values of one ID 
                # deviate from the target ID for a specific feature.
            
            # Ensure that both arrays have the same length: fill shorter array with 0
            # Ensure target_array and other_array are the same length by filling the shorter one with zeros
            target_len = len(target_array)
            other_len = len(other_array)
            
            # Convert numpy arrays to pandas Series for reindexing
            target_series = pd.Series(target_array)
            other_series = pd.Series(other_array)
            
            # If target_array is shorter, pad with zeros
            if target_len < other_len:
                target_series = target_series.reindex(range(other_len), fill_value=0)
            # If other_array is shorter, pad with zeros
            elif other_len < target_len:
                other_series = other_series.reindex(range(target_len), fill_value=0)
            
            # Convert back to numpy arrays after reindexing
            target_array = target_series.values
            other_array = other_series.values
  
            mse = ((target_array - other_array) ** 2).mean()  # MSE per feature
    
            mse_per_id[id] = mse  # Store the MSE for each ID
    
        mse_values[feature] = mse_per_id  # Store per-feature MSE values
    
    # Example: Find the most similar ID based on minimum MSE across all features
    final_similarity = {}
    for feature, mse_per_id in mse_values.items():
        for id, mse in mse_per_id.items():
            if id not in final_similarity:
                final_similarity[id] = []
            final_similarity[id].append(mse)
    
    # Combine the per-feature MSEs to get final similarity
    final_similarity = {id: sum(mse_list) for id, mse_list in final_similarity.items()}

    
    # Remove the target ID itself from comparison by using a dictionary comprehension
    final_similarity = {id: mse_sum for id, mse_sum in final_similarity.items() if id != target_id}
    
    # Get the top n most similar IDs based on the sum of MSEs
    similar_ids = sorted(final_similarity.items(), key=lambda x: x[1])[:n]
    not_similar_ids = sorted(final_similarity.items(), key=lambda x: x[1], reverse=True)[:n]
    
    return similar_ids, not_similar_ids  # Returns list of (id, mse) tuples


# Cosine Similarity - ALL

# Description: Measures the cosine of the angle between two time series, treating them as vectors. It calculates how similar the direction of the time series is, irrespective of their magnitude.
# Use case: When you care about the orientation or pattern of the time series rather than their absolute values.
# Advantage: Robust to different magnitudes and shifts in the time series.
# Disadvantage: Ignores absolute differences in values, which could be important for some applications.


def similarity_cosine_time(df, target_id, n=5, target_day=3):
    """
    Find the n most similar IDs to a given target_id based on cosine similarity, considering time-based measurement frequencies.

    Parameters:
    df (pd.DataFrame): DataFrame with 'id', 'datetime', and fruit measurement columns.
    target_id (int or str): The ID to compare against.
    n (int): Number of similar IDs to return.

    Returns:
    List of tuples (id, similarity) sorted by similarity (higher is more similar).
    """
    feature_time_bin = {
        "Blood Pressure, Diastolic (mmHg)": 60.0,
        "Blood Pressure, Mean (mmHg)": 60.0,
        "Blood Pressure, Systolic (mmHg)": 60.0,
        "Glucose (mmol/L)": 236.0,
        "Heart Rate (bpm)": 60.0,
        "Respiratory Rate (min-1)": 60.0,
        "SpO2 (%)": 60.0,
        "Temperature (celsius)": 240.0
    }
    
    df = df.drop(columns=['day_offset', 'time_part'], errors='ignore')  
    df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure datetime format

    aggregated_dfs = []
    # Different features have different measurement frequencies.
    # This standardizes time intervals for all features before further analysis.
    # It ensures that comparisons (MSE or cosine similarity) are fair by aligning the data points properly.
    for feature, time_bin in feature_time_bin.items():
        temp_df = df[['id', 'datetime', feature]].copy()
        temp_df['time_bin'] = temp_df['datetime'].dt.floor(f'{int(time_bin)}min')
        aggregated_df = temp_df.groupby(['id', 'time_bin'])[feature].mean().reset_index()
        aggregated_dfs.append(aggregated_df)

    merged_df = aggregated_dfs[0]
    for other_df in aggregated_dfs[1:]:
        merged_df = merged_df.merge(other_df, on=['id', 'time_bin'], how='outer')

    feature_columns = list(feature_time_bin.keys())
    mean_values = merged_df.groupby(['id', 'time_bin'])[feature_columns].mean()
    
    # Forward fill missing values within each ID and fill remaining NaNs with 0
    mean_values = mean_values.groupby('id').ffill().fillna(0)
    
    standardized_means = (mean_values - mean_values.mean()) / mean_values.std()

    if target_id not in standardized_means.index.get_level_values('id'):
        raise ValueError(f"ID {target_id} not found in the dataset.")

    target_vector = standardized_means.loc[target_id].values.flatten()
    similarity_scores = {}
    
    for other_id in standardized_means.index.get_level_values('id').unique():
        if other_id == target_id:
            continue
        
        other_vector = standardized_means.loc[other_id].values.flatten()
        
        # Ensure vectors have the same length
        target_len, other_len = len(target_vector), len(other_vector)
        if target_len < other_len:
            target_vector = np.pad(target_vector, (0, other_len - target_len), 'constant')
        elif other_len < target_len:
            other_vector = np.pad(other_vector, (0, target_len - other_len), 'constant')
        
        # Compute cosine similarity
        similarity = cosine_similarity([target_vector], [other_vector])[0][0]
        similarity_scores[other_id] = similarity
    
    similar_ids = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:n]
    not_similar_ids = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=False)[:n]
    
    return similar_ids, not_similar_ids

# Euclidean Distance - ALL

# Description: Measures the straight-line distance between two time series. It's straightforward and fast to compute.
# Use case: When you expect the time series to be aligned and have similar length.
# Advantage: Simple and intuitive.
# Disadvantage: Sensitive to differences in the overall magnitude of the time series, and does not handle time shifts or scaling well.

def similarity_euclidean_time(df, target_id, n=5, target_day=3):
    """
    Find the n most similar IDs to a given target_id based on Euclidean distance, 
    considering time-based measurement frequencies.

    Parameters:
    df (pd.DataFrame): DataFrame with 'id', 'datetime', and fruit measurement columns.
    target_id (int or str): The ID to compare against.
    n (int): Number of similar IDs to return.

    Returns:
    List of tuples (id, similarity) sorted by similarity (lower is more similar).
    """
    feature_time_bin = {
        "Blood Pressure, Diastolic (mmHg)": 60.0,
        "Blood Pressure, Mean (mmHg)": 60.0,
        "Blood Pressure, Systolic (mmHg)": 60.0,
        "Glucose (mmol/L)": 236.0,
        "Heart Rate (bpm)": 60.0,
        "Respiratory Rate (min-1)": 60.0,
        "SpO2 (%)": 60.0,
        "Temperature (celsius)": 240.0
    }
    
    df = df.drop(columns=['day_offset', 'time_part'], errors='ignore')  
    df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure datetime format

    aggregated_dfs = []
    for feature, time_bin in feature_time_bin.items():
        temp_df = df[['id', 'datetime', feature]].copy()
        temp_df['time_bin'] = temp_df['datetime'].dt.floor(f'{int(time_bin)}min')
        aggregated_df = temp_df.groupby(['id', 'time_bin'])[feature].mean().reset_index()
        aggregated_dfs.append(aggregated_df)

    merged_df = aggregated_dfs[0]
    for other_df in aggregated_dfs[1:]:
        merged_df = merged_df.merge(other_df, on=['id', 'time_bin'], how='outer')

    feature_columns = list(feature_time_bin.keys())
    mean_values = merged_df.groupby(['id', 'time_bin'])[feature_columns].mean()
    # Forward fill missing values within each ID and fill remaining NaNs with 0
    mean_values = mean_values.groupby('id').ffill().fillna(0)
    standardized_means = (mean_values - mean_values.mean()) / mean_values.std()

    if target_id not in standardized_means.index.get_level_values('id'):
        raise ValueError(f"ID {target_id} not found in the dataset.")

    target_vector = standardized_means.loc[target_id].values.flatten()
    distance_scores = {}

    for other_id in standardized_means.index.get_level_values('id').unique():
        if other_id == target_id:
            continue

        other_vector = standardized_means.loc[other_id].values.flatten()

        # Ensure vectors have the same length
        target_len, other_len = len(target_vector), len(other_vector)
        if target_len < other_len:
            target_vector = np.pad(target_vector, (0, other_len - target_len), 'constant')
        elif other_len < target_len:
            other_vector = np.pad(other_vector, (0, target_len - other_len), 'constant')

        # Compute Euclidean distance
        distance = euclidean(target_vector, other_vector)
        distance_scores[other_id] = distance

    # Sort by smallest distance (lower = more similar)
    similar_ids = sorted(distance_scores.items(), key=lambda x: x[1])[:n]
    not_similar_ids = sorted(distance_scores.items(), key=lambda x: x[1], reverse = True)[:n]
    return similar_ids, not_similar_ids

# Prepare for similarity

# EXAMPLE
id = 34253386 # target ID for similarity
target_day = 3 # number of days that target ID has data (delete the rest from df_resampled)

# delete days past target_day from target_id before plotting so that it does not interfere with the comparison
target_data = df_resampled[df_resampled['id'] == id] #get the target id's data
# filter out rows where the 'daycount' exceeds 'target_day'
filtered_target_data = target_data[target_data['day_offset'] <= target_day]
# merge the filtered data back with the rest of the dataframe
df_resampled_target = pd.concat([df_resampled[df_resampled['id'] != id], filtered_target_data], ignore_index=True)

# Generate 2 fake similar ID - salting the data that you know

# Define the amount of noise (small standard deviation)
std_dev_percentage1 = 0.02  # 2% of the feature_value
std_dev_percentage2 = 0.015  # 1.5% of the feature_value

df_noise1 = filtered_target_data.copy()
df_noise2 = filtered_target_data.copy()

noisy_columns = ['Blood Pressure, Diastolic (mmHg)', 'Blood Pressure, Mean (mmHg)', 'Blood Pressure, Systolic (mmHg)', 
                          'Glucose (mmol/L)', 'Heart Rate (bpm)', 'Respiratory Rate (min-1)', 'SpO2 (%)', 'Temperature (celsius)']

# Generate noise and add it to the columns
for col in noisy_columns:
    noise1 = np.random.normal(loc=0, scale=df_noise1[col] * std_dev_percentage1)
    df_noise1[col] = df_noise1[col] + noise1
    noise2 = np.random.normal(loc=0, scale=df_noise2[col] * std_dev_percentage2)
    df_noise2[col] = df_noise2[col] + noise2

# replace id
df_noise1['id'] = 99999
df_noise2['id'] = 88888

# Append it back to the original dataframe
df_resampled_target = pd.concat([df_resampled_target, df_noise1], ignore_index=True)
df_resampled_target = pd.concat([df_resampled_target, df_noise2], ignore_index=True)

# Check what fake ID look like
run_vital_sign_visualisation(df_resampled_target, 8052)

# Compute Dis_Similarity

similar_ids_list_mse, not_similar_ids_list_mse = similarity_mse(df_resampled_target, target_id=id, n=5)
print("Similarity MSE:", similar_ids_list_mse)
print("NOT Similarity MSE:", not_similar_ids_list_mse)

similar_ids_list_mse_time, not_similar_ids_list_mse_time = similarity_mse_time(df_resampled_target, target_id=id, n=5)
print("Similarity MSE+Time:", similar_ids_list_mse_time)
print("NOT Similarity MSE+Time:", not_similar_ids_list_mse_time)

similar_ids_list_cosine_time, not_similar_ids_list_cosine_time = similarity_cosine_time(df_resampled_target, target_id=id, n=5)
all_similar, all_not_similar = similarity_cosine_time(df_resampled_target, target_id=id, n=100)
print("Similarity Cosine+Time:", similar_ids_list_cosine_time)
print("NOT Similarity Cosine+Time:", not_similar_ids_list_cosine_time)
print("All Id sorted from similar -> dissimilar:", all_similar)

similar_ids_list_euclidean_time, not_similar_ids_list_euclidean_time = similarity_euclidean_time(df_resampled_target, target_id=id, n=5)
print("Similarity Euclidean+Time:", similar_ids_list_euclidean_time)
print("NOT Similarity Euclidean+Time:", not_similar_ids_list_euclidean_time)

# similar_ids_list_granger_causality = similarity_granger_causality(df_resampled_target, target_id=id, n=5)
# print("Similarity Granger Causality:", similar_ids_list_granger_causality)

# Display (not) similar as a dataframe

# Convert each list into a dataframe
MSE = pd.DataFrame(similar_ids_list_mse, columns=['id', 'MSE']) # lower = closer
MSE_time = pd.DataFrame(similar_ids_list_mse_time, columns=['id', 'MSE_time'])
Cosine = pd.DataFrame(similar_ids_list_cosine_time, columns=['id', 'Cosine']) # -1 (dissimilar) to 1 (similar)
Euclidean = pd.DataFrame(similar_ids_list_euclidean_time, columns=['id', 'Euclidean']) # lower score - closer together

# Convert each list into a dataframe
not_MSE = pd.DataFrame(not_similar_ids_list_mse, columns=['id', 'not_MSE'])
not_MSE_time = pd.DataFrame(not_similar_ids_list_mse_time, columns=['id', 'not_MSE_time'])
not_Cosine = pd.DataFrame(not_similar_ids_list_cosine_time, columns=['id', 'not_Cosine'])
not_Euclidean = pd.DataFrame(not_similar_ids_list_euclidean_time, columns=['id', 'not_Euclidean'])

# Merge all dataframes on the 'id' column, using outer join to keep all ids
Similarity_Results = MSE.merge(MSE_time, on='id', how='outer')\
        .merge(Cosine, on='id', how='outer')\
        .merge(Euclidean, on='id', how='outer')\
        .merge(not_MSE, on='id', how='outer')\
        .merge(not_MSE_time, on='id', how='outer')\
        .merge(not_Cosine, on='id', how='outer')\
        .merge(not_Euclidean, on='id', how='outer')\

# Display the final dataframe
print("Most and Least (_not) Similar IDs")
display(Similarity_Results)

# visualise dis_similar_ids

def plot_dis_similar_ids(df, target_id, similar_ids, s):
    """
    Visualizes the time series of fruit measurements for the target ID and similar IDs.
    
    Parameters:
    df (pd.DataFrame): Original dataset containing 'id', 'datetime', and fruit measurements.
    target_id (int or str): The ID of interest.
    similar_ids (list): List of IDs similar to the target ID.

    Returns:
    A Plotly figure with subplots.
    """
    feature_columns = df.columns.difference(['id', 'datetime', 'day_offset', 'time_part'])  # Extract fruit measurement columns
    df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure datetime is in proper format

    # Format 'time_part' to string if it's a datetime.time type
    df['time_part'] = df['time_part'].apply(lambda x: x.strftime('%H:%M') if isinstance(x, datetime.time) else str(x))

    # Create subplot grid
    num_features = len(feature_columns)
    fig = make_subplots(rows=num_features, cols=1, shared_xaxes=True,
                        subplot_titles=feature_columns, vertical_spacing=0.1)

    # Plot each category as a separate line graph
    for i, feature in enumerate(feature_columns, start=1):
        for sim_id in similar_ids:
            sim_data = df[df['id'] == sim_id]
            fig.add_trace(go.Scatter(x=sim_data['datetime'], y=sim_data[feature],
                                     mode='lines', name=f"ID {sim_id}",
                                     line=dict(color='blue', width=1, dash='dash')),
                          row=i, col=1)

        # Plot target ID in purple
        target_data = df[df['id'] == target_id]
        fig.add_trace(go.Scatter(x=target_data['datetime'], y=target_data[feature],
                                 mode='lines', name=f"Target ID {target_id}",
                                 line=dict(color='purple', width=2)),
                      row=i, col=1)

        # Update axes labels
        fig.update_xaxes(
                title_text="Time (days, hours)",
                showgrid=True,
                row=i, col=1,
                tickmode='auto',
                dtick=1000 * 60 * 60 * 6,
                tickformat="Day %j\n%H:%M",
                showticklabels=True
            )  

    # Layout settings
    if s == 0:
        title = f"Comparison of Target ID {target_id} vs Similar IDs"
    if s == 1:
        title = f"Comparison of Target ID {target_id} vs NOT Similar IDs"

    # Update layout for the figure
    fig.update_layout(
        height=300 * num_features,
        width=900,
        title_text=title,
        showlegend=True,  
    )

    return fig

# visualise dis_similar_ids: avg and std

def plot_dis_similar_ids_std(df, target_id, similar_ids, feature_columns= ["ALL"] , s=0):
    """
    Visualizes the time series of fruit measurements for the target ID and the averaged similar IDs.
    
    Parameters:
    df (pd.DataFrame): Original dataset containing 'id', 'datetime', and fruit measurements.
    target_id (int or str): The ID of interest.
    similar_ids (list): List of IDs similar to the target ID.

    Returns:
    A Plotly figure with subplots.
    """

    if feature_columns == ["ALL"]:
        feature_columns = ["Blood Pressure, Diastolic (mmHg)", "Blood Pressure, Mean (mmHg)", "Blood Pressure, Systolic (mmHg)", "Glucose (mmol/L)",
                           "Heart Rate (bpm)", "Respiratory Rate (min-1)", "SpO2 (%)", "Temperature (celsius)"]
    else:
        feature_columns = feature_columns

    
    # feature_columns = df.columns.difference(['id', 'datetime', 'day_offset', 'time_part'])  # get's passed as argument
    df['datetime'] = pd.to_datetime(df['datetime'])  

    # Ensure 'time_part' is in string format
    #df['time_part'] = df['time_part'].apply(lambda x: x.strftime('%H:%M') if isinstance(x, datetime.time) else str(x))
    df['time_part'] = df['time_part'].apply(lambda x: str(x))

    # Create subplot grid
    num_features = len(feature_columns)
    fig = make_subplots(rows=num_features, cols=1, shared_xaxes=True,
                        subplot_titles=feature_columns, vertical_spacing=0.1)

    for i, feature in enumerate(feature_columns, start=1):
        # Compute mean & std for similar_ids group
        similar_data = df[df['id'].isin(similar_ids)].groupby('datetime')[feature].agg(['mean', 'std']).reset_index()
        # Convert 'datetime'
        similar_data['datetime'] = similar_data['datetime'].dt.to_pydatetime()
        # Problem with Glucose and Temperature here, because they are measured less frequently than hourly - nan at timestamps.

        # Plot continuous shaded region
        fig.add_trace(go.Scatter(
            x=similar_data['datetime'].tolist() + similar_data['datetime'][::-1].tolist(),
            y=(similar_data['mean'] + similar_data['std']).tolist() + (similar_data['mean'] - similar_data['std'])[::-1].tolist(),
            fill='toself', fillcolor='rgba(150, 150, 150, 0.6)',  # Continuous grey shading
            line=dict(color='rgba(255,255,255,0)'),
            name=f"+1 Std Dev ({feature})",
            connectgaps=True, # this connects the line over NaN values (Glucose and Temperature)
            showlegend=False
        ), row=i, col=1)
        
        # Plot mean of similar IDs on top of the shaded areaplot_dis_similar_ids_std
        valid_similar_data = similar_data.dropna(subset=['mean']) # Drop NaN values for plotting the mean line and solve the problem for Glucose and Temperature
        fig.add_trace(go.Scatter(x=valid_similar_data['datetime'], y=valid_similar_data['mean'],
                                 mode='lines', name=f"Similar IDs Avg ({feature})",
                                 line=dict(color='blue', width=2)),
                      row=i, col=1)

        # Plot target ID in purple
        target_data = df[df['id'] == target_id]
        fig.add_trace(go.Scatter(x=target_data['datetime'], y=target_data[feature],
                                 mode='lines', name=f"Target ID {target_id}",
                                line=dict(color='purple', width=2)),
                      row=i, col=1)
        
        # Update x-axis settings
        fig.update_xaxes(
            title_text="Time (days, hours)",
            showgrid=True,
            row=i, col=1,
            tickmode='auto',
            dtick=1000 * 60 * 60 * 6,
            tickformat="Day %j\n%H:%M",
            showticklabels=True  
        )  

    # Title setting
        group_labels = {
        0: "Similar ID Group",
        1: "NOT Similar ID Group",
        2: "Random ID Group"
    }
    label = group_labels.get(s, "Unknown Group")
    title = f"Comparison of Target ID {target_id} vs {label}"

    # Update layout
    fig.update_layout(
        height=300 * num_features,
        width=900,
        title_text=title,
        showlegend=True,  
    )

    return fig

# Example: id = 34253386

similar_ids_list = similar_ids_list_cosine_time # set whichever list to display
dissimilar_ids_list = not_similar_ids_list_cosine_time

# target_day = 5 #make sure it is the same as above

# # delete days past target_day from df_resampled for plotting - DON'T NEED THIS
# target_data = df_resampled[df_resampled['id'] == id] #get the target id's data from id set earlier
# filtered_target_data = target_data[target_data['day_offset'] <= target_day]
# # merge the filtered data back with the rest of the dataframe
# target_df = pd.concat([df_resampled[df_resampled['id'] != id], filtered_target_data], ignore_index=True)

# plot SIMILAR
similar_ids = [item[0] for item in similar_ids_list]
fig = plot_dis_similar_ids_std(df_resampled_target, target_id=id, similar_ids=similar_ids, feature_columns=["ALL"], s=0) #id set earlier
fig.show()

# issue: when ID asked has been there for a long time - the longer there, the less similar ID?

# plot NOT SIMILAR
NOT_similar_ids = [item[0] for item in dissimilar_ids_list]
fig = plot_dis_similar_ids_std(df_resampled_target, target_id=id, similar_ids=NOT_similar_ids, feature_columns=["ALL"], s=1) #id set earlier
fig.show()

# plot RANDOM from all_similar

def select_random_tuples(arr, num_parts=10):
    # Calculate the size of each chunk
    chunk_size = len(arr) // num_parts
    selected = []
    
    for i in range(num_parts):
        # Determine the start and end indices for each chunk
        start = i * chunk_size
        # Ensure the last chunk includes any remaining elements
        end = (i + 1) * chunk_size if i < num_parts - 1 else len(arr)
        
        chunk = arr[start:end]
        
        if chunk:  # Ensure the chunk is not empty
            selected.append(random.choice(chunk))
    
    return selected

# Example usage


random_similar_ids = select_random_tuples(all_similar)
random_ids = [item[0] for item in random_similar_ids]
fig = plot_dis_similar_ids_std(df_resampled_target, target_id=id, similar_ids=random_ids, feature_columns=["ALL"], s=0) #id set earlier
fig.show()

# --------------------------------------------
# Compare Selected Features Only
# --------------------------------------------

        # "Blood Pressure, Diastolic (mmHg)"
        # "Blood Pressure, Mean (mmHg)"
        # "Blood Pressure, Systolic (mmHg)"
        # "Glucose (mmol/L)"
        # "Heart Rate (bpm)"
        # "Respiratory Rate (min-1)"
        # "SpO2 (%)"
        # "Temperature (celsius)"

# Cosine Similarity - Selected Features

# needed to select random ids
def select_random_tuples(arr, num_parts=10):
    # Calculate the size of each chunk
    chunk_size = len(arr) // num_parts
    selected = []
    
    for i in range(num_parts):
        # Determine the start and end indices for each chunk
        start = i * chunk_size
        # Ensure the last chunk includes any remaining elements
        end = (i + 1) * chunk_size if i < num_parts - 1 else len(arr)
        
        chunk = arr[start:end]
        
        if chunk:  # Ensure the chunk is not empty
            selected.append(random.choice(chunk))
    
    return selected

def similarity_cosine_time_selected(df, target_id, selected_features, n=5):
    """
    Find the n most similar and dissimilar IDs to a given target_id based on cosine similarity, 
    considering user-selected features and time-based measurement frequencies.

    Parameters:
    df (pd.DataFrame): DataFrame with 'id', 'datetime', and measurement columns. (!) needs to shorten days for target id before passing (!)
    target_id (int or str): The ID to compare against.
    selected_features (list of str): List of feature column names to include in the comparison.
    n (int): Number of similar and dissimilar IDs to return.

    Returns:
    Tuple:
        - List of tuples (id, similarity) sorted by similarity (higher is more similar).
        - List of tuples (id, similarity) sorted by dissimilarity (lower is less similar).
        - List of tuples (id, similarity) randomly selected from similarity.
    """

    feature_time_bin = {
        "Blood Pressure, Diastolic (mmHg)": 60.0,
        "Blood Pressure, Mean (mmHg)": 60.0,
        "Blood Pressure, Systolic (mmHg)": 60.0,
        "Glucose (mmol/L)": 236.0,
        "Heart Rate (bpm)": 60.0,
        "Respiratory Rate (min-1)": 60.0,
        "SpO2 (%)": 60.0,
        "Temperature (celsius)": 240.0
    }

    # Validate selected features
    available_features = set(feature_time_bin.keys())
    if not set(selected_features).issubset(available_features):
        raise ValueError(f"Invalid features selected. Available options: {available_features}")

    df = df.drop(columns=['day_offset', 'time_part'], errors='ignore')  
    df['datetime'] = pd.to_datetime(df['datetime'])  # Ensure datetime format

    aggregated_dfs = []
    for feature in selected_features:
        time_bin = feature_time_bin[feature]
        temp_df = df[['id', 'datetime', feature]].copy()
        temp_df['time_bin'] = temp_df['datetime'].dt.floor(f'{int(time_bin)}min')
        aggregated_df = temp_df.groupby(['id', 'time_bin'])[feature].mean().reset_index()
        aggregated_dfs.append(aggregated_df)

    merged_df = aggregated_dfs[0]
    for other_df in aggregated_dfs[1:]:
        merged_df = merged_df.merge(other_df, on=['id', 'time_bin'], how='outer')

    mean_values = merged_df.groupby(['id', 'time_bin'])[selected_features].mean()
    
    # Forward fill missing values within each ID and fill remaining NaNs with 0
    mean_values = mean_values.groupby('id').ffill().fillna(0)
    
    standardized_means = (mean_values - mean_values.mean()) / mean_values.std()

    if target_id not in standardized_means.index.get_level_values('id'):
        raise ValueError(f"ID {target_id} not found in the dataset.")

    target_vector = standardized_means.loc[target_id].values.flatten()
    similarity_scores = {}
    
    for other_id in standardized_means.index.get_level_values('id').unique():
        if other_id == target_id:
            continue
        
        other_vector = standardized_means.loc[other_id].values.flatten()
        
        # Ensure vectors have the same length
        target_len, other_len = len(target_vector), len(other_vector)
        if target_len < other_len:
            target_vector = np.pad(target_vector, (0, other_len - target_len), 'constant')
        elif other_len < target_len:
            other_vector = np.pad(other_vector, (0, target_len - other_len), 'constant')
        
        # Compute cosine similarity
        similarity = cosine_similarity([target_vector], [other_vector])[0][0]
        similarity_scores[other_id] = similarity

    #results
    all_similar = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:100] # all 100 similar (ignoring 4 least simmilar)
    similar_ids = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)[:n] # n similar
    not_similar_ids = sorted(similarity_scores.items(), key=lambda x: x[1])[:n] # n dissimilar
    random_ids = select_random_tuples(all_similar) # random - calls function above
    
    return similar_ids, not_similar_ids, random_ids

# Similarity for heart rate only
similar_ids_list_cosine_time_HR, not_similar_ids_list_cosine_time_HR,random_ids_list_cosine_time_HR = similarity_cosine_time_selected(df_resampled_target, id, selected_features=["Heart Rate (bpm)"], n=10)
print("Similarity Cosine+Time_HR:", similar_ids_list_cosine_time_HR)
print("NOT Similarity Cosine+Time_HR:", not_similar_ids_list_cosine_time_HR)
print("Random Cosine+Time_HR:", random_ids_list_cosine_time_HR)

# Compare with previous cosine similarity of all

# Convert each list into a dataframe

Cosine_All = pd.DataFrame(similar_ids_list_cosine_time, columns=['id', 'Cosine_All']) 
Cosine_HR = pd.DataFrame(similar_ids_list_cosine_time_HR, columns=['id', 'Cosine_HR'])

# Convert each list into a dataframe
not_Cosine_All = pd.DataFrame(not_similar_ids_list_cosine_time, columns=['id', 'not_Cosine_All'])
not_Cosine_HR = pd.DataFrame(not_similar_ids_list_cosine_time_HR, columns=['id', 'not_Cosine_HR'])

# Merge all dataframes on the 'id' column, using outer join to keep all ids
Similarity_Results_HR_All = Cosine_All.merge(Cosine_HR, on='id', how='outer')\
        .merge(not_Cosine_All, on='id', how='outer')\
        .merge(not_Cosine_HR, on='id', how='outer')\


# Display the final dataframe
print("Most and Least (_not) Similar IDs")
display(Similarity_Results_HR_All)


# In[62]:


# plot HR Similarity with std
HR_similar_ids = [item[0] for item in similar_ids_list_cosine_time_HR]
fig = plot_dis_similar_ids_std(df_resampled_target, target_id=id, similar_ids=HR_similar_ids, feature_columns=["Heart Rate (bpm)"], s=0) #id set earlier
fig.show()


# for comparing via age: I don't think my dataset is big enough to allow for that

# Similarity Glucose only - checking if it works
similar_ids_list_cosine_GL, not_similar_ids_list_cosine_GL,random_ids_list_cosine_GL = similarity_cosine_time_selected(df_resampled_target, id, selected_features=["Glucose (mmol/L)"], n=10)
print("Similarity Cosine+Glucose:", similar_ids_list_cosine_GL)
print("NOT Similarity Cosine+Glucose:", not_similar_ids_list_cosine_GL)
print("Random Cosine+Glucose:", random_ids_list_cosine_GL)

# Glucose has similarity values

# plot Glucose Similarity with std
GL_similar_ids = [item[0] for item in similar_ids_list_cosine_GL]
fig = plot_dis_similar_ids_std(df_resampled_target, target_id=id, similar_ids=GL_similar_ids, feature_columns=["Glucose (mmol/L)"], s=0) #id set earlier
fig.show()

# plot does not show similar id!

# ------------------------------------------------------------
# Combine similarity and Plotting beased on selected features

        # "Blood Pressure, Diastolic (mmHg)"
        # "Blood Pressure, Mean (mmHg)"
        # "Blood Pressure, Systolic (mmHg)"
        # "Glucose (mmol/L)"
        # "Heart Rate (bpm)"
        # "Respiratory Rate (min-1)"
        # "SpO2 (%)"
        # "Temperature (celsius)"
# ------------------------------------------------------------

def plot_ids(df, target_id, feature_columns=["ALL"], s=0, n=5, target_day=None):
    
    # Convert 'datetime' to pandas datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    # Ensure 'time_part' is in string format
    df['time_part'] = df['time_part'].apply(lambda x: str(x))
    #df['time_part'] = df['time_part'].apply(lambda x: x.strftime('%H:%M') if isinstance(x, datetime.time) else str(x))

    # Filter target data into main and continued parts based on target_day
    target_data_full = df[df['id'] == target_id]
    if target_day is not None:
        target_data_main = target_data_full[target_data_full['day_offset'] <= target_day]
        target_data_continued = target_data_full[target_data_full['day_offset'] > target_day]
        # Exclude future data from similarity calculations
        df_target = pd.concat([df[df['id'] != target_id], target_data_main], ignore_index=True)
    else:
        target_data_main = target_data_full
        target_data_continued = pd.DataFrame()
        df_target = df

    # Determine feature columns
    if feature_columns == ["ALL"]:
        feature_columns = ["Blood Pressure, Diastolic (mmHg)", "Blood Pressure, Mean (mmHg)",
                           "Blood Pressure, Systolic (mmHg)", "Glucose (mmol/L)", "Heart Rate (bpm)",
                           "Respiratory Rate (min-1)", "SpO2 (%)", "Temperature (celsius)"]

    # Obtain similar, dissimilar, and random IDs
    similar_ids, not_similar_ids, random_ids = similarity_cosine_time_selected(df_target, target_id, feature_columns, n)
    group_mapping = {0: similar_ids, 1: not_similar_ids, 2: random_ids}
    selected_group = [item[0] for item in group_mapping.get(s, [])]

    transition_time = target_data_main['datetime'].max().to_pydatetime() # when main stops and continue begins (used for mean and std)

    # Create subplots: two per feature (main and continued)
    num_features = len(feature_columns)
    
    fig = make_subplots(rows=num_features, cols=1, shared_xaxes=False,
                    subplot_titles=feature_columns, vertical_spacing=0.15)

    # ranges for y axis of each feature
    feature_ranges = {"Blood Pressure, Diastolic (mmHg)": [20,120], 
                    "Blood Pressure, Mean (mmHg)": [40,140],
                    "Blood Pressure, Systolic (mmHg)": [20,180],
                    "Glucose (mmol/L)": [80,300], 
                    "Heart Rate (bpm)": [50,200],
                    "Respiratory Rate (min-1)": [0,50], 
                    "SpO2 (%)": [80,110], 
                    "Temperature (celsius)": [34,42]
    }

    for i, feature in enumerate(feature_columns):
        row_main = i + 1

        # Compute mean, std, and count of unique IDs contributing at each timestamp
        similar_data = df_target[df_target['id'].isin(selected_group)].groupby('datetime').agg(
            mean=(feature, 'mean'),
            std=(feature, 'std'),
            count=('id', 'nunique')  # Count unique IDs
        ).reset_index()
        similar_data['datetime'] = similar_data['datetime'].dt.to_pydatetime()

        # Filter out time points where only 1 ID remains
        valid_similar_data = similar_data[similar_data['count'] > 1].dropna(subset=['mean'])  # Ensure at least 2 IDs contribute to the mean

        # Ensure mean and std end at the same time by filtering them together
        valid_std_data = valid_similar_data.dropna(subset=['std'])
        valid_std_data_main = valid_std_data[valid_std_data['datetime'] <= transition_time]
        valid_std_data_continued = valid_std_data[valid_std_data['datetime'] > transition_time]

        # Std up to transition point
        fig.add_trace(go.Scatter(
            x=valid_std_data_main['datetime'].tolist() + valid_std_data_main['datetime'][::-1].tolist(),
            y=(valid_std_data_main['mean'] + valid_std_data_main['std']).tolist() + 
              (valid_std_data_main['mean'] - valid_std_data_main['std'])[::-1].tolist(),
            fill='toself', fillcolor='rgba(150, 150, 150, 0.6)',
            line=dict(color='rgba(255,255,255,0)'), connectgaps=True,
            showlegend=True, name=f"Std (Main) - {feature}", visible='legendonly'),
            row=row_main, col=1)
        
        # Std after transition
        fig.add_trace(go.Scatter(
            x=valid_std_data_continued['datetime'].tolist() + valid_std_data_continued['datetime'][::-1].tolist(),
            y=(valid_std_data_continued['mean'] + valid_std_data_continued['std']).tolist() + 
              (valid_std_data_continued['mean'] - valid_std_data_continued['std'])[::-1].tolist(),
            fill='toself', fillcolor='rgba(150, 150, 150, 0.3)',
            line=dict(color='rgba(255,255,255,0)'), connectgaps=True,
            showlegend=True, name=f"Std (Continued) - {feature}", visible='legendonly'),
            row=row_main, col=1)

        # Ensure mean line also stops when std stops
        if not valid_std_data.empty:
                        # Mean up to transition point
            fig.add_trace(go.Scatter(x=valid_std_data_main['datetime'], y=valid_std_data_main['mean'],
                                     mode='lines', name=f"Mean (Main) - {feature}",
                                     line=dict(color='blue', width=2), visible='legendonly'),
                          row=row_main, col=1)
            
            # Mean after transition
            fig.add_trace(go.Scatter(x=valid_std_data_continued['datetime'], y=valid_std_data_continued['mean'],
                                     mode='lines', name=f"Mean (Continued) - {feature}",
                                     line=dict(color='#4A90E2', width=2), visible='legendonly'),
                          row=row_main, col=1)
            
        #Ensure target continue does not go longer than std continue
        # Determine last datetime of valid std continued
        if not valid_std_data_continued.empty:  
            max_std_datetime = valid_std_data_continued['datetime'].max()
            target_data_continued_filtered = target_data_continued[target_data_continued['datetime'] <= max_std_datetime] 
        else:
            target_data_continued_filtered = target_data_continued.iloc[0:0]  # Empty DataFrame if no std continued  
        
        fig.add_trace(go.Scatter(x=target_data_main['datetime'], y=target_data_main[feature],
                                 mode='lines', name=f"Target ID {target_id} Main", line=dict(color='purple', width=2), visible='legendonly'),
                      row=row_main, col=1)

               
        # Add filtered purple continued line
        fig.add_trace(go.Scatter(x=target_data_continued_filtered['datetime'], y=target_data_continued_filtered[feature], 
                                 mode='lines', name=f"Target ID {target_id} Continued", 
                                 line=dict(color='rgba(128, 0, 128, 0.5)', width=2), visible='legendonly'),  
                      row=row_main, col=1)

        y_min, y_max = feature_ranges[feature]
        fig.add_trace(go.Scatter(
            x=[transition_time, transition_time],
            y=[y_min, y_max],  # Use unpacked min/max values
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name=f"Transition - {feature}",
             visible='legendonly'
        ), row=row_main, col=1)

        fig.update_yaxes(range=[y_min, y_max], row=row_main, col=1)
        
        # Maintain original x-axis tick formatting
        fig.update_xaxes(
            title_text="Time (days, hours)",
            showgrid=True,
            row=row_main, col=1,
            tickmode='auto',
            dtick=1000 * 60 * 60 * 6,
            tickformat="Day %j\n%H:%M",
            showticklabels=True
        )

    # Set the main title
    fig.update_layout(
        height=600 * num_features,
        width=900,
        title_text=f"Comparison of Target ID {target_id} with {['Similar', 'Dissimilar', 'Random'][s]} ID Group",
        showlegend=True,
        legend=dict(
        yanchor="top",  # Align legends to the top
        y=1.05,  # Move legends slightly above the top row
        xanchor="left",
        x=1.02,  # Push legends to the right column
        tracegroupgap=520,  # Keep spacing
        traceorder="grouped")
    )

    return fig

# INTERACTIVE DASH APP

# Dash App Setup
app_similarity = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app_similarity.layout = html.Div([
    html.H1("Patient Data Comparison"),
    html.Div([
        html.Label("Select Target ID:"),
        dcc.Dropdown(
            id='target_id_dropdown',
            options=[{'label': id, 'value': id} for id in df_resampled['id'].unique()],
            value=df_resampled['id'].unique()[0]
        )
    ], style={'padding': '10px'}),

    html.Div([
        html.Label("Select Features:"),
        dcc.Checklist(
            id='feature_columns_checklist',
            options=[{'label': feature, 'value': feature} for feature in
                     ["Blood Pressure, Diastolic (mmHg)", "Blood Pressure, Mean (mmHg)",
                      "Blood Pressure, Systolic (mmHg)", "Glucose (mmol/L)", "Heart Rate (bpm)",
                      "Respiratory Rate (min-1)", "SpO2 (%)", "Temperature (celsius)"]],
            value=["Blood Pressure, Diastolic (mmHg)", "Heart Rate (bpm)"],
            inline=False,
            style={'padding': '5px'}
        )
    ], style={'padding': '20px'}),

    html.Div([
        html.Label("Select Sub Group:"),
        dcc.RadioItems(
            id='sub_group_radio',
            options=[
                {'label': 'Similar', 'value': 0},
                {'label': 'Dissimilar', 'value': 1},
                {'label': 'Random', 'value': 2},
            ],
            value=0
        )
    ], style={'padding': '10px'}),

    html.Div([
        html.Label("Select the number of days that data is available:"),
        dcc.Input(id='day_input', type='number', value=3, min=1, max=20)
    ], style={'padding': '10px'}),

    html.Div([
        html.Label("Select the number of similar/dissimilar/random patients for comparison:"),
        dcc.Input(id='n_input', type='number', value=5, min=1, max=30)
    ], style={'padding': '10px'}),

    # dcc.Graph(id='plot', style={'padding': '20px'})
    dcc.Graph(id='plot', style={'height': '100vh', 'width': '100vw'})
])

# Callback to update the graph
@app_similarity.callback(
    Output('plot', 'figure'),
    [
        Input('target_id_dropdown', 'value'),
        Input('feature_columns_checklist', 'value'),
        Input('sub_group_radio', 'value'),
        Input('day_input', 'value'),
        Input('n_input', 'value'),
    ]
)
def update_plot(target_id, feature_columns, s, target_day, n):
    return plot_ids(df_resampled, target_id, feature_columns, s, n, target_day)
    
# Run the app - calls function from above
run_app(app_similarity, 8053)

# SAME but with dummy drop downs

# Dummy data for dropdown selections
age_options = [{'label': str(age), 'value': age} for age in range(18, 90, 5)]
diagnosis_options = [{'label': d, 'value': d} for d in ["Pneumonia", "Sepsis", "Stroke", "Myocardial Infarction", "..."]]
past_medical_history_options = [{'label': pmh, 'value': pmh} for pmh in ["Hypertension", "Sleep Apnea", "Drug Use", "..."]]
comorbidities_options = [{'label': c, 'value': c} for c in ["Asthma", "Type I Diabetes Mellitus", "Type II Diabetes Mellitus", "..."]]
procedure_options = [{'label': p, 'value': p} for p in ["Surgery", "Blood Transfusion", "Dialysis", "..."]]
gender_options = [{'label': g, 'value': g} for g in ["Biological Male", "Biological Female", "Other"]]
medication_options = [{'label': m, 'value': m} for m in ["Antibiotics", "Anticoagulants", "Painkillers", "..."]]
allergy_options = [{'label': a, 'value': a} for a in ["Penicillin", "Morphine", "Peanuts", "Latex", "..."]]

# Dash App Setup
app_similarity_dummy = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout of the app
app_similarity_dummy.layout = dbc.Container([
    html.H1("Patient Data Comparison", className="text-center"),

    dbc.Row([
        # Left Column - Main Controls
        dbc.Col([
            html.Div([
                html.Label("Select Target ID:"),
                dcc.Dropdown(
                    id='target_id_dropdown',
                    options=[{'label': id, 'value': id} for id in df_resampled['id'].unique()],
                    value=df_resampled['id'].unique()[0]
                )
            ], style={'padding': '10px'}),

            html.Div([
                html.Label("Select Features:"),
                dcc.Checklist(
                    id='feature_columns_checklist',
                    options=[{'label': feature, 'value': feature} for feature in
                             ["Blood Pressure, Diastolic (mmHg)", "Blood Pressure, Mean (mmHg)",
                              "Blood Pressure, Systolic (mmHg)", "Glucose (mmol/L)", "Heart Rate (bpm)",
                              "Respiratory Rate (min-1)", "SpO2 (%)", "Temperature (celsius)"]],
                    value=["Blood Pressure, Diastolic (mmHg)", "Heart Rate (bpm)"],
                    inline=False
                )
            ], style={'padding': '10px'}),

            html.Div([
                html.Label("Select Sub Group:"),
                dcc.RadioItems(
                    id='sub_group_radio',
                    options=[
                        {'label': 'Similar', 'value': 0},
                        {'label': 'Dissimilar', 'value': 1},
                        {'label': 'Random', 'value': 2},
                    ],
                    value=0
                )
            ], style={'padding': '10px'}),

            html.Div([
                html.Label("Select the number of days that data is available:"),
                dcc.Input(id='day_input', type='number', value=3, min=1, max=20)
            ], style={'padding': '10px'}),

            html.Div([
                html.Label("Select the number of similar/dissimilar/random patients for comparison:"),
                dcc.Input(id='n_input', type='number', value=5, min=1, max=30)
            ], style={'padding': '10px'}),
        ], width=6),  # Left Column takes 6 out of 12 grid width

        # Right Column - Dummy Dropdowns
        dbc.Col([
            html.H4("Filter similar patient data via:"),

            html.Div([
                html.Label("Age:"),
                dcc.Dropdown(id='age_dropdown', options=age_options, value=None)
            ], style={'padding': '10px'}),

             html.Div([
                html.Label("Gender:"),
                dcc.Dropdown(id='gender_dropdown', options=gender_options, value=None)
            ], style={'padding': '10px'}),

            html.Div([
                html.Label("Allergies:"),
                dcc.Dropdown(id='allergy_dropdown', options=allergy_options, value=None, multi=True)
            ], style={'padding': '10px'}),

            html.Div([
                html.Label("Reason for Admission:"),
                dcc.Dropdown(id='diagnosis_dropdown', options=diagnosis_options, value=None)
            ], style={'padding': '10px'}),

            html.Div([
                html.Label("Comorbidities:"),
                dcc.Dropdown(id='comorbidities_dropdown', options=comorbidities_options, value=None, multi=True)
            ], style={'padding': '10px'}),

            html.Div([
                html.Label("Past Medical History:"),
                dcc.Dropdown(id='pmh_dropdown', options=past_medical_history_options, value=None, multi=True)
            ], style={'padding': '10px'}),

            html.Div([
                html.Label("Procedure:"),
                dcc.Dropdown(id='procedure_dropdown', options=procedure_options, value=None)
            ], style={'padding': '10px'}),

            html.Div([
                html.Label("Medication:"),
                dcc.Dropdown(id='medication_dropdown', options=medication_options, value=None, multi=True)
            ], style={'padding': '10px'}),

        ], width=6)  # Right Column takes 6 out of 12 grid width

    ], align="start"),  # Aligns content to the top

    # Full-Width Graph
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='plot', style={'height': '80vh', 'width': '100%'})
        ])
    ])
], fluid=True)  # Enables full-width layout

# Callback to update the graph
@app_similarity_dummy.callback(
    Output('plot', 'figure'),
    [
        Input('target_id_dropdown', 'value'),
        Input('feature_columns_checklist', 'value'),
        Input('sub_group_radio', 'value'),
        Input('day_input', 'value'),
        Input('n_input', 'value'),
    ]
)
def update_plot(target_id, feature_columns, s, target_day, n):
    return plot_ids(df_resampled, target_id, feature_columns, s, n, target_day)
    
# Run the app - calls function from above
run_app(app_similarity_dummy, 8056)

