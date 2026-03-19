import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os

# Files
FILE_CLUSTERS_2024 = "clusters_2024_k2.parquet"
FILE_RAW_2025 = "200_2025.parquet"
FILE_CENTROIDS = "centroids_k2.npy"

OUTPUT_2025 = "clusters_2025_k2.parquet"
OUTPUT_RATIO_2024 = "pysindy_ratio_2024.csv"
OUTPUT_RATIO_2025 = "pysindy_ratio_2025.csv"

# Define Seasons matching SINDy script
WINTER_MONTHS = [10, 11, 12, 1, 2, 3]

def calc_ratio_and_compare(df, filename, year_label):
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['day_id'].str.split('_').str[0])
    
    # Calculate initial daily ratio
    daily = df.groupby('date')['cluster'].agg(['count', 'sum']).reset_index()
    daily['high_user_ratio'] = daily['sum'] / daily['count']
    
    # Split Context
    daily['month'] = daily['date'].dt.month
    daily['is_weekend'] = daily['date'].dt.dayofweek >= 5
    daily['is_winter'] = daily['month'].isin(WINTER_MONTHS)
    
    # Create a unique context string for grouping (e.g., "True_False" = Winter Weekday)
    daily['context'] = daily['is_winter'].astype(str) + "_" + daily['is_weekend'].astype(str)
    
    # Sort chronologically to ensure rolling works correctly
    daily = daily.sort_values(['context', 'date'])
    
    # Apply average to context
    daily['high_user_ratio_smooth'] = daily.groupby('context')['high_user_ratio'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    
    # Calculate derivatives 
    daily['dx_raw'] = daily.groupby('context')['high_user_ratio'].diff()
    daily['dx_smooth'] = daily.groupby('context')['high_user_ratio_smooth'].diff()
    
    # raw vs smooth 
    print(f"{year_label} ")
    
    for context, group in daily.groupby('context'):
        is_win, is_wknd = context.split('_')
        ctx_name = ("Winter" if is_win == "True" else "Summer") + " " + \
                   ("Weekend" if is_wknd == "True" else "Weekday")
        
        # Calculate standard deviation 
        std_raw = group['high_user_ratio'].std()
        std_smooth = group['high_user_ratio_smooth'].std()
        
        # Calculate Max dx/dt 
        max_vel_raw = group['dx_raw'].abs().max()
        max_vel_smooth = group['dx_smooth'].abs().max()
        
        print(f"[{ctx_name}]")
        print(f"  Raw Data      -> Std Dev (σ): {std_raw:.5f} | Max (dx/dt): {max_vel_raw:.5f}")
        print(f"  Smoothed Data -> Std Dev (σ): {std_smooth:.5f} | Max (dx/dt): {max_vel_smooth:.5f}")
        
        # Avoid division by zero
        if max_vel_raw > 0:
            reduction = (1 - (max_vel_smooth / max_vel_raw)) * 100
            print(f"  Velocity Spike Reduction: {reduction:.1f}%\n")
        else:
            print("\n")

    daily = daily.drop(columns=['dx_raw', 'dx_smooth'])
    daily.to_csv(filename, index=False)


## 2024 Data 
df_2024 = pd.read_parquet(FILE_CLUSTERS_2024)
calc_ratio_and_compare(df_2024, OUTPUT_RATIO_2024, "2024 TRAINING DATA")



## 2025 Data 

df_2025 = pd.read_parquet(FILE_RAW_2025)
df_2025['timestamp'] = pd.to_datetime(df_2025['data_collection_log_timestamp'])

# Data cleaning - extreme values & clocks
df_2025 = df_2025[df_2025['total_consumption_active_import'] < 20000] 
bad_dates = [pd.Timestamp("2025-03-30").date(), pd.Timestamp("2025-10-26").date()]
df_2025 = df_2025[~df_2025['timestamp'].dt.date.isin(bad_dates)]

# Convert to Matrix
df_2025['step'] = df_2025['timestamp'].dt.hour * 2 + (df_2025['timestamp'].dt.minute // 30)
df_2025['day_id'] = df_2025['timestamp'].dt.date.astype(str) + "_" + df_2025['lv_feeder_unique_id']

# Discard incomplete days
df_matrix = df_2025.pivot(index='day_id', columns='step', values='total_consumption_active_import').dropna()
data_2025 = df_matrix.values


centroids = np.load(FILE_CENTROIDS)
CLUSTERS = len(centroids)

# Calculate Distances (Canberra)
dists = np.zeros((len(data_2025), CLUSTERS))
for r in range(len(data_2025)):
    for c in range(CLUSTERS):
        dists[r, c] = distance.canberra(data_2025[r], centroids[c])
        
labels = np.argmin(dists, axis=1)

# 2025 Transitions
df_res = pd.DataFrame({'day_id': df_matrix.index, 'cluster': labels})
df_res.to_parquet(OUTPUT_2025)


# Get ratios for 2025
calc_ratio_and_compare(df_res, OUTPUT_RATIO_2025, "2025 TESTING DATA")
