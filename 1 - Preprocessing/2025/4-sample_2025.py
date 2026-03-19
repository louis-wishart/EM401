import pandas as pd
import os

# Find sample Feeder IDs
df_train = pd.read_parquet("200_2024.parquet")
target_feeders = df_train['lv_feeder_unique_id'].unique().tolist()
print(f"Found {len(target_feeders)} feeders")

# Take sample Feeders 2025 data
print("Start Date: 2025-03-01 00:00:00+00:00")

df_test = pd.read_parquet(
    "s3://weave.energy/smart-meter",
    filters=[
        ("lv_feeder_unique_id", "in", target_feeders),
        ("data_collection_log_timestamp", ">=", pd.Timestamp("2025-03-01", tz="UTC"))
    ]
)

# Data cleaning - extreme values & clocks
df_test = df_test[df_test["total_consumption_active_import"] < 20000]
df_test = df_test[~df_test["data_collection_log_timestamp"].dt.date.isin(
    [pd.Timestamp("2025-03-30").date(), pd.Timestamp("2025-10-26").date()]
)]

df_test = df_test[(df_test["aggregated_device_count_active"] > 10) & 
                  (df_test["aggregated_device_count_active"] < 50)]

# Take useful columns 
cols = ["lv_feeder_unique_id", "data_collection_log_timestamp",
        "total_consumption_active_import", "aggregated_device_count_active"]
df_test = df_test[cols]

# End date 
last_date = df_test['data_collection_log_timestamp'].max()
print(f"End Date: {last_date}")

# 4. Save
df_test.to_parquet("200_2025.parquet")
print(f"Saved to 200_2025.parquet")