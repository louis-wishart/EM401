import pandas as pd
import matplotlib.pyplot as plt
import os

# Files
INPUT_FILE = "200_2024.parquet"
TEXT_OUT = "sample_summary.txt"
PLOT_OUT = "feeder_plot.png"


def count_individual_homes(dataframe, col_name='aggregated_device_count_active'):
    
    if col_name in dataframe.columns:
        # Get the highest recorded number of homes for each unique feeder
        homes_per_feeder = dataframe.groupby('lv_feeder_unique_id')[col_name].max()
        return int(homes_per_feeder.sum())
   

print(f"Reading {INPUT_FILE}:")
df = pd.read_parquet(INPUT_FILE)

# Counts
total_rows = len(df)
unique_days = df['data_collection_log_timestamp'].dt.date.nunique()

# Count Homes
total_homes = count_individual_homes(df, 'aggregated_device_count_active')

# Feeder Quality
# 17520 ideal, 17000 is acceptable
counts = df['lv_feeder_unique_id'].value_counts()

good_feeders = sum(counts > 17000)
partial_feeders = sum(counts <= 17000)

# Gap Analysis
expected_range = pd.date_range("2024-03-01", "2025-03-01", freq="30min")
theoretical_max = (len(expected_range) - 96) * 200

missing_rows = theoretical_max - total_rows
complete_per = (total_rows / theoretical_max) * 100

# Outage Check
time_activity = df.groupby("data_collection_log_timestamp").size()
system_gaps = len(time_activity[time_activity < 10])

# Report
report = f"""
File: {INPUT_FILE}
Total Rows: {total_rows:,}
Unique Days: {unique_days}
Total Homes Analysed: {total_homes}

FEEDERS:
Full Year Feeders (>17k rows): {good_feeders}
Partial Feeders: {partial_feeders}

DATA:
Theoretical Max Rows: {theoretical_max:,}
Actual Rows: {total_rows:,}
Missing: {missing_rows:,}
% Complete: {complete_per:.2f}%

OUTAGES:
Timestamps with <10 active feeders: {system_gaps}

"""

print(report)

with open(TEXT_OUT, "w") as f:
    f.write(report)
print(f"Report: {TEXT_OUT}")


# Plot
plt.figure(figsize=(12, 6))

plt.plot(time_activity.index, time_activity.values, label='Active Feeders', alpha=0.8, lw=0.5)
plt.axhline(y=200, color='red', linestyle='--', label='Target (200)')

plt.title(f"Feeder Count Overview (SSEN)\nCompleteness: {complete_per:.1f}%")
plt.xlabel("Date")
plt.ylabel("Count of Active Feeders")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(PLOT_OUT, dpi=200)
print(f"Plot saved to {PLOT_OUT}")