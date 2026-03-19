import pandas as pd
import numpy as np


FILE_2024 = "200_2024.parquet"
FILE_2025 = "200_2025.parquet"


COLUMN_POWER  = "total_consumption_active_import"
COLUMN_TIME   = "data_collection_log_timestamp"
COLUMN_FEEDER = "lv_feeder_unique_id"


df24 = pd.read_parquet(FILE_2024)
df25 = pd.read_parquet(FILE_2025)


df24[COLUMN_TIME] = pd.to_datetime(df24[COLUMN_TIME])
df25[COLUMN_TIME] = pd.to_datetime(df25[COLUMN_TIME])


# Calculates the overall baseline growth in total energy volume
mean_24 = df24[COLUMN_POWER].mean()
mean_25 = df25[COLUMN_POWER].mean()
volume_drift = (mean_25 - mean_24) / mean_24


# Group by individual feeder and specific day, find the maximum, then average those peaks
peak_24 = df24.groupby([COLUMN_FEEDER, df24[COLUMN_TIME].dt.date])[COLUMN_POWER].max().mean()
peak_25 = df25.groupby([COLUMN_FEEDER, df25[COLUMN_TIME].dt.date])[COLUMN_POWER].max().mean()
peak_drift = (peak_25 - peak_24) / peak_24



print(f"{'Metric':<25} | {'2024 Value':<12} | {'2025 Value':<12} | {'Growth (Drift)':<12}")
print("-" * 75)
print(f"{'Load Volume':<25} | {mean_24:<12.3f} | {mean_25:<12.3f} | {volume_drift:>12.2%}")
print(f"{'Avg Daily Peak':<25} | {peak_24:<12.3f} | {peak_25:<12.3f} | {peak_drift:>12.2%}")
print(f"\nDrift = {volume_drift:.2%}")