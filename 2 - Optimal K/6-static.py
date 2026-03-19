import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial import distance
import sys
import os
 
FILENAME = "200_2024.parquet"
OUTPUT_CENTROIDS = "centroids.npy"
OUTPUT_SEQUENCE = "clusters_2024.parquet"
OUTPUT_ELBOW = "elbow_plot.png"
OUTPUT_CLUSTERS = "cluster_plot.png"
 
MAX_K = 15
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
raw_data = df_matrix.values
 
# Normalise
row_sums = raw_data.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
data = raw_data / row_sums
 
# Downsample 
if len(data) > 3000:
    idx = np.random.choice(len(data), 3000, replace=False)
    train_data = data[idx]
else:
    train_data = data
 
# Function 2
history_k = []
history_error = []
best_models = {} 
 
for k in range(2, MAX_K + 1):
    print(f"K={k} ->", end=" ")
    
    min_inertia = float('inf')
    best_centroids_k = None
    
    for run in range(RESTARTS):
        
        centroids = train_data[np.random.choice(len(train_data), k, replace=False)]
        
        for i in range(30):
            # Canberra
            dists = distance.cdist(train_data, centroids, metric='canberra')
            
            # Assign clusters
            labels = np.argmin(dists, axis=1)
            
            # Average
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                if np.sum(labels == c) > 0:
                    new_centroids[c] = np.mean(train_data[labels == c], axis=0)
                else:
                    new_centroids[c] = train_data[np.random.choice(len(train_data))]
            
            # Convergence 
            if np.sum(np.abs(centroids - new_centroids)) < 0.001:
                break
            centroids = new_centroids
            
        # Error Calc 
        final_dists = distance.cdist(train_data, centroids, metric='canberra')
        curr_inertia = np.sum(np.min(final_dists, axis=1))
        
        if curr_inertia < min_inertia:
            min_inertia = curr_inertia
            best_centroids_k = centroids
 
    avg_err = min_inertia / len(train_data)
    print(f"Error: {avg_err:.4f}")
    
    history_k.append(k)
    history_error.append(avg_err)
    best_models[k] = best_centroids_k


# Geometric Elbow
k_list = np.array(history_k)
error_list = np.array(history_error)
k_norm = (k_list - k_list.min()) / (k_list.max() - k_list.min())
err_norm = (error_list - error_list.min()) / (error_list.max() - error_list.min())
 
p1 = np.array([k_norm[0], err_norm[0]])
p2 = np.array([k_norm[-1], err_norm[-1]])
 
dists = []
for i in range(len(k_list)):
    p_curr = np.array([k_norm[i], err_norm[i]])
    d = np.abs(np.cross(p2-p1, p1-p_curr)) / np.linalg.norm(p2-p1)
    dists.append(d)
 
optimal_k = k_list[np.argmax(dists)]
print(f"\n Optimal K: {optimal_k}")
 
final_centroids = best_models[optimal_k]
np.save(OUTPUT_CENTROIDS, final_centroids)
print(f"Centroids: {OUTPUT_CENTROIDS}")
 
# Full Dataset
full_dists = distance.cdist(data, final_centroids, metric='canberra')
        
final_labels = np.argmin(full_dists, axis=1)
 
# Save Logbook
results = pd.DataFrame({
    'day_id': df_matrix.index,
    'cluster': final_labels
})
results[['date_str', 'feeder']] = results['day_id'].str.split('_', n=1, expand=True)
results['date'] = pd.to_datetime(results['date_str'])
results.to_parquet(OUTPUT_SEQUENCE)
print(f"Transitions: {OUTPUT_SEQUENCE}")
 
# Elbow Plot
plt.figure(figsize=(10, 6))
plt.plot(history_k, history_error, marker='o', color='#4682B4', linewidth=2)
plt.axvline(x=optimal_k, color='#B22222', linestyle='--', label=f'Optimal K={optimal_k}')
plt.xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
plt.ylabel('Inertia (Error)', fontsize=12, fontweight='bold')
plt.title('Elbow Analysis', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(OUTPUT_ELBOW)
print(f"Saved {OUTPUT_ELBOW}")
 
# Cluster Plot
cols = 3
rows = (optimal_k + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5*rows))
axes = axes.flatten()
 
timestamps = pd.date_range(start='2024-01-01 00:00', periods=48, freq='30min')
 
for i in range(optimal_k):
    ax = axes[i]
    centroid = final_centroids[i]
    
    ax.plot(timestamps, centroid, color=f'C{i}', linewidth=2.5)
    
    peak_idx = np.argmax(centroid)
    peak_time = timestamps[peak_idx]
    peak_val = centroid[peak_idx]
    peak_str = peak_time.strftime('%H:%M')
    
    ax.text(0.5, 0.9, f"Peak: {peak_str}\n({peak_val:.1%})", 
            transform=ax.transAxes, ha='center', va='top', fontsize=9, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round'))
    
    ax.set_title(f'Cluster {i+1}', fontweight='bold')
    ax.set_ylim(0, np.max(centroid)*1.25)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    
    if i % cols == 0: 
        ax.set_ylabel("Intensity (Share of Day)", fontsize=9)
    if i >= optimal_k - cols:
        ax.set_xlabel("Time of Day", fontsize=9)
 
for i in range(optimal_k, len(axes)):
    axes[i].axis('off')
 
plt.suptitle(f'Static Behavioral Clusters (K={optimal_k})', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_CLUSTERS, bbox_inches='tight')
print(f"Cluster Plot: {OUTPUT_CLUSTERS}")