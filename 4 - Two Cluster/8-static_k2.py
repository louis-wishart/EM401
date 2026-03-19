import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial import distance
from scipy.spatial.distance import cdist
import sys
import os

FILENAME = "200_2024.parquet"
OUTPUT_FILE = "clusters_2024_k2.parquet"
OUTPUT_PLOT = "cluster_plot_k2.png"

CLUSTERS = 2
RESTARTS = 10

# Function 1 
def get_data():
    df = pd.read_parquet(FILENAME)
    df['timestamp'] = pd.to_datetime(df['data_collection_log_timestamp'])
    df = df[(df['timestamp'] >= '2024-03-01') & (df['timestamp'] < '2025-03-01')].copy()
    
    df['step'] = df['timestamp'].dt.hour * 2 + (df['timestamp'].dt.minute // 30)
    df['day_id'] = df['timestamp'].dt.date.astype(str) + "_" + df['lv_feeder_unique_id']
    
    return df.pivot(index='day_id', columns='step', values='total_consumption_active_import').dropna()

df_matrix = get_data()
data = df_matrix.values

# Downsample 
if len(data) > 3000:
    idx = np.random.choice(len(data), 3000, replace=False)
    train_data = data[idx]
else:
    train_data = data

best_error = float('inf')
best_centroids = None

# Function 2 
for run in range(RESTARTS):
    print(f"  Run {run+1}/10")
    
    # Random start
    centroids = train_data[np.random.choice(len(train_data), CLUSTERS, replace=False)]
    
    for i in range(100):
        # Canberra
        dists = cdist(train_data, centroids, metric='canberra')
        
        # Assign clusters
        labels = np.argmin(dists, axis=1)
        
        # Average
        new_centroids = np.zeros_like(centroids)
        for k in range(CLUSTERS):
            if np.sum(labels == k) > 0:
                new_centroids[k] = np.mean(train_data[labels == k], axis=0)
            else:
                new_centroids[k] = train_data[np.random.choice(len(train_data))]
        
        # Convergence
        if np.sum(np.abs(centroids - new_centroids)) < 0.001:
            break
        centroids = new_centroids

    # Error Calc
    final_dists = cdist(train_data, centroids, metric='canberra')
        
    error = np.sum(np.min(final_dists, axis=1))
    print(f"  Error for Run {run+1}: {error:.2f}")
    
    if error < best_error:
        best_error = error
        best_centroids = centroids
print(f"Best Error: {best_error:.2f}")

# Cluster 0 = low
if np.sum(best_centroids[0]) > np.sum(best_centroids[1]):
    best_centroids = best_centroids[::-1]

np.save("centroids_k2.npy", best_centroids)

# Apple to full dataset 
full_dists = cdist(data, best_centroids, metric='canberra')
final_labels = np.argmin(full_dists, axis=1)

# Results
pd.DataFrame({'day_id': df_matrix.index, 'cluster': final_labels}).to_parquet(OUTPUT_FILE)
print(f"Results: {OUTPUT_FILE}")

# Plot 
plt.figure(figsize=(12, 6))
ax = plt.gca()
timestamps = pd.date_range(start='2024-01-01 00:00', periods=48, freq='30min')

plt.plot(timestamps, best_centroids[0], label='Cluster 0 (Low Profile)', 
         color='#4682B4', linewidth=2.5, linestyle='-')
plt.plot(timestamps, best_centroids[1], label='Cluster 1 (High Profile)', 
         color='#B22222', linewidth=2.5, linestyle='-')

ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
ax.set_xlim(timestamps[0], timestamps[-1])
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)

plt.grid(True, which='major', linestyle=':', alpha=0.6, color='gray')

plt.xlabel("Time of Day (UTC)", fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel("Consumption (Wh)", fontsize=12, fontweight='bold', labelpad=10)
plt.title("Cluster Centroids: Daily Load Profiles", fontsize=14, fontweight='bold', pad=15)

plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=True, fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
plt.close()

print(f"Plot: {OUTPUT_PLOT}")