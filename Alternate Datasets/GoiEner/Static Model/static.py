import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial import distance
import glob

SPRING_START = '2019-03-01'
SPRING_END   = '2019-06-30'

MAX_K    = 20
RESTARTS = 10

np.random.seed(42)

# Find parquet
files = glob.glob("*.parquet") + glob.glob("../*.parquet") + glob.glob("Preprocessing/*.parquet")
if len(files) == 1:
    data_file = files[0]
else:
    for i, f in enumerate(files): print(f"{i+1}: {f}")
    data_file = files[int(input("Select file: ")) - 1]

def get_data(filepath, start, end):
    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.strip()

    if 'data' not in df.columns:
        for col in df.columns:
            if any(k in col.lower() for k in ['kwh', 'value', 'data']):
                df.rename(columns={col: 'data'}, inplace=True)
                break

    df = df[(df['date'] >= start) & (df['date'] <= end)].copy()

    df['date_only'] = df['date'].dt.date
    ppd = df.groupby(['meter', 'date_only']).size().mode()[0]

    if ppd == 24:
        df['step'] = df['date'].dt.hour
    elif ppd == 48:
        df['step'] = df['date'].dt.hour * 2 + (df['date'].dt.minute // 30)
    else:
        df['step'] = df.groupby(['meter', 'date_only']).cumcount()

    df['day_id'] = df['date'].dt.date.astype(str) + "_" + df['meter']
    return df.pivot(index='day_id', columns='step', values='data').dropna(), ppd

df_matrix, ppd = get_data(data_file, SPRING_START, SPRING_END)
print(f"Resolution: {ppd} points/day  |  {len(df_matrix)} training vectors")

raw_data = df_matrix.values

# Normalise
row_sums = raw_data.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
data = raw_data / row_sums

# Downsample
if len(data) > 2000:
    idx = np.random.choice(len(data), 2000, replace=False)
    train_data = data[idx]
else:
    train_data = data

# Elbow search
history_k     = []
history_error = []
best_models   = {}

for k in range(2, MAX_K + 1):
    print(f"K={k} ->", end=" ")
    min_inertia      = float('inf')
    best_centroids_k = None

    for run in range(RESTARTS):
        centroids = train_data[np.random.choice(len(train_data), k, replace=False)]

        for i in range(30):
            dists  = distance.cdist(train_data, centroids, metric='canberra')
            labels = np.argmin(dists, axis=1)

            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                if np.sum(labels == c) > 0:
                    new_centroids[c] = np.mean(train_data[labels == c], axis=0)
                else:
                    new_centroids[c] = train_data[np.random.choice(len(train_data))]

            if np.sum(np.abs(centroids - new_centroids)) < 0.001:
                break
            centroids = new_centroids

        final_dists  = distance.cdist(train_data, centroids, metric='canberra')
        curr_inertia = np.sum(np.min(final_dists, axis=1))

        if curr_inertia < min_inertia:
            min_inertia      = curr_inertia
            best_centroids_k = centroids

    avg_err = min_inertia / len(train_data)
    print(f"Error: {avg_err:.4f}")

    history_k.append(k)
    history_error.append(avg_err)
    best_models[k] = best_centroids_k

# Geometric elbow
k_list   = np.array(history_k)
err_list = np.array(history_error)
k_norm   = (k_list - k_list.min()) / (k_list.max() - k_list.min())
err_norm = (err_list - err_list.min()) / (err_list.max() - err_list.min())

p1 = np.array([k_norm[0],  err_norm[0]])
p2 = np.array([k_norm[-1], err_norm[-1]])

dists = []
for i in range(len(k_list)):
    p_curr = np.array([k_norm[i], err_norm[i]])
    d = np.abs(np.cross(p2-p1, p1-p_curr)) / np.linalg.norm(p2-p1)
    dists.append(d)

optimal_k       = k_list[np.argmax(dists)]
final_centroids = best_models[optimal_k]
print(f"\nOptimal K: {optimal_k}")

np.save('centroids.npy', final_centroids)

# Elbow plot
plt.figure(figsize=(10, 6))
plt.plot(history_k, history_error, marker='o', color='#4682B4', linewidth=2)
plt.axvline(x=optimal_k, color='#B22222', linestyle='--', label=f'Optimal K={optimal_k}')
plt.scatter([optimal_k], [history_error[history_k.index(optimal_k)]], color='#B22222', s=100, zorder=5)
plt.xlabel('Number of Clusters (K)', fontsize=12, fontweight='bold')
plt.ylabel('Inertia (Error)', fontsize=12, fontweight='bold')
plt.title('Elbow Analysis', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('graph0_elbow_curve.png')
print("Saved graph0_elbow_curve.png")

# Cluster plot
timestamps = pd.date_range(start='2024-01-01 00:00', periods=ppd, freq='30min' if ppd == 48 else '1h')
cols = 3
rows = (optimal_k + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(15, 3.5*rows))
axes = axes.flatten()

for i in range(optimal_k):
    ax       = axes[i]
    centroid = final_centroids[i]

    ax.plot(timestamps, centroid, color=f'C{i}', linewidth=2.5)

    peak_idx = np.argmax(centroid)
    peak_str = timestamps[peak_idx].strftime('%H:%M')
    peak_val = centroid[peak_idx]

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

plt.suptitle(f'Static Behavioural Clusters (K={optimal_k})', fontsize=16, y=1.02)
plt.tight_layout()
plt.savefig('graph1_static_clusters.png', bbox_inches='tight')
print("Saved graph1_static_clusters.png")