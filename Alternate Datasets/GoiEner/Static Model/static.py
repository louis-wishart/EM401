import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
import random
import glob
import sys
import os

# --- CONFIGURATION ---
SPRING_START = '2019-03-01'
SPRING_END = '2019-06-30'
SUMMER_START = '2019-07-01'
SUMMER_END = '2019-10-30'

# Algorithm Parameters
MAX_K = 20              # Max clusters to test (to see the full curve)
RND_M = 10              # Inner Loop: Number of random restarts

# --- HELPER CLASS: Custom K-Means with Canberra Distance ---
class KMeansCanberra:
    def __init__(self, k, max_iters=100, tol=1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.inertia = None 

    def _canberra_dist(self, u, v):
        return distance.canberra(u, v)

    def fit(self, X):
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]
        
        for i in range(self.max_iters):
            distances = np.zeros((n_samples, self.k))
            for idx, centroid in enumerate(self.centroids):
                for sample_i in range(n_samples):
                    distances[sample_i, idx] = self._canberra_dist(X[sample_i], centroid)
            
            self.labels = np.argmin(distances, axis=1)
            new_centroids = np.zeros_like(self.centroids)
            for cluster_idx in range(self.k):
                cluster_points = X[self.labels == cluster_idx]
                if len(cluster_points) > 0:
                    new_centroids[cluster_idx] = cluster_points.mean(axis=0)
                else:
                    new_centroids[cluster_idx] = X[np.random.choice(n_samples)]
            
            shift = np.linalg.norm(self.centroids - new_centroids)
            self.centroids = new_centroids
            if shift < self.tol:
                break
        
        self.inertia = 0
        for idx, centroid in enumerate(self.centroids):
            cluster_points = X[self.labels == idx]
            for point in cluster_points:
                self.inertia += self._canberra_dist(point, centroid)
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.k))
        for idx, centroid in enumerate(self.centroids):
            for sample_i in range(n_samples):
                distances[sample_i, idx] = self._canberra_dist(X[sample_i], centroid)
        return np.argmin(distances, axis=1)

# --- HELPER: Normalization (L1 / "Share of Day") ---
def normalize_data(X):
    """
    Normalizes each row by its SUM (L1 Norm).
    Result: Each hour represents the % of that day's total energy used.
    This preserves 'Flatness' vs 'Peakiness'.
    """
    # Sum across the row (axis=1)
    row_sum = X.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sum[row_sum == 0] = 1 
    return X / row_sum

# --- HELPER: Find Elbow Point ---
def find_optimal_k(k_values, error_values):
    """
    Finds the 'Elbow' K using the Maximum Distance method.
    It draws a line from the first to the last point and finds the K
    that is furthest from that line (the point of max curvature).
    """
    # Normalize values to 0-1 range to handle different scales of K and Error
    k_norm = (k_values - np.min(k_values)) / (np.max(k_values) - np.min(k_values))
    err_norm = (error_values - np.min(error_values)) / (np.max(error_values) - np.min(error_values))
    
    # Coordinates of the line (start to end)
    # Point 1: (0, 1) -> normalized first point (Low K, High Error)
    # Point 2: (1, 0) -> normalized last point (High K, Low Error)
    # Note: Error usually decreases, so first point is max error (1.0), last is min (0.0)
    
    # Vector of the line
    line_vec = np.array([1, -1]) # rough direction from (0,1) to (1,0)
    # More precise: (k_norm[-1] - k_norm[0], err_norm[-1] - err_norm[0])
    p1 = np.array([k_norm[0], err_norm[0]])
    p2 = np.array([k_norm[-1], err_norm[-1]])
    
    distances = []
    for i in range(len(k_values)):
        p_curr = np.array([k_norm[i], err_norm[i]])
        
        # Perpendicular distance formula from point to line defined by p1, p2
        # d = |cross_product((p2-p1), (p1-p0))| / |p2-p1|
        dist = np.abs(np.cross(p2-p1, p1-p_curr)) / np.linalg.norm(p2-p1)
        distances.append(dist)
    
    best_idx = np.argmax(distances)
    return k_values[best_idx]

# --- MAIN PIPELINE ---

def find_data_file():
    search_paths = ["*.parquet", "../*.parquet", "Preprocessing/*.parquet"]
    found_files = []
    
    for path in search_paths:
        found_files.extend(glob.glob(path))
    
    if not found_files:
        print("ERROR: No .parquet files found.")
        print("Please run 'interactive_convert.py' to generate your data.")
        sys.exit()
        
    if len(found_files) == 1:
        return found_files[0]
    
    print("\nMultiple data files found. Please select one:")
    for i, f in enumerate(found_files):
        print(f"  {i+1}: {f}")
        
    while True:
        try:
            sel = int(input("Enter number: ")) - 1
            if 0 <= sel < len(found_files):
                return found_files[sel]
        except ValueError:
            pass
        print("Invalid selection.")

def run_pipeline():
    # Set seed for reproducibility
    np.random.seed(42)
    random.seed(42)

    print("--- Phase 1: Data Engineering ---")
    
    data_file = find_data_file()
    print(f"Loading {data_file}...")
    df = pd.read_parquet(data_file)
    
    # --- ROBUST COLUMN FIXING ---
    df.columns = df.columns.str.strip()
    
    if 'data' not in df.columns:
        found_col = None
        for col in df.columns:
            c_lower = col.lower()
            if 'kwh' in c_lower or 'value' in c_lower or 'data' in c_lower:
                found_col = col
                break
        
        if found_col:
            print(f"Note: Automatically renaming '{found_col}' column to 'data'...")
            df.rename(columns={found_col: 'data'}, inplace=True)
        else:
            print("ERROR: Could not find a consumption column.")
            sys.exit()
    
    # 2. Filter for Spring
    print(f"Isolating Spring data ({SPRING_START} to {SPRING_END})...")
    mask_spring = (df['date'] >= SPRING_START) & (df['date'] <= SPRING_END)
    df_spring = df.loc[mask_spring].copy()
    
    if df_spring.empty:
        print("ERROR: No data found in Spring range. Check your dataset dates.")
        return

    # --- AUTO-DETECT TIME RESOLUTION ---
    df_spring['date_only'] = df_spring['date'].dt.date
    sample_counts = df_spring.groupby(['meter', 'date_only']).size()
    
    if sample_counts.empty:
        print("ERROR: Grouping failed.")
        return

    points_per_day = sample_counts.mode()[0]
    print(f"Detected resolution: {points_per_day} points per day.")
    
    if points_per_day == 24:
        print("Using Hourly resolution (0-23).")
        df_spring['step_index'] = df_spring['date'].dt.hour
    elif points_per_day == 48:
        print("Using Half-Hourly resolution (0-47).")
        df_spring['step_index'] = df_spring['date'].dt.hour * 2 + (df_spring['date'].dt.minute // 30)
    else:
        print(f"WARNING: Unusual resolution ({points_per_day}). Defaulting to enumeration.")
        df_spring['step_index'] = df_spring.groupby(['meter', 'date_only']).cumcount()

    # 3. Reshape
    print("Reshaping to daily vectors...")
    df_spring['day_id'] = df_spring['date'].dt.date.astype(str) + "_" + df_spring['meter']
    
    X_train_df = df_spring.pivot(index='day_id', columns='step_index', values='data')
    
    # 4. Clean
    original_count = len(X_train_df)
    X_train_df = X_train_df.dropna()
    clean_count = len(X_train_df)
    print(f"Cleaned training vectors: {clean_count}")
    
    # 5. NORMALIZE (The Fix: Share of Day)
    print("Normalizing profiles (Share of Day) to capture BEHAVIOR...")
    X_train = X_train_df.values
    X_train_norm = normalize_data(X_train)
    
    # Downsample
    if len(X_train_norm) > 2000:
        print("NOTE: Downsampling training set to 2000 random days for Pilot Speed...")
        indices = np.random.choice(len(X_train_norm), 2000, replace=False)
        X_train_sample = X_train_norm[indices]
    else:
        X_train_sample = X_train_norm

    print("\n--- Phase 2: The Static Model (Optimal K Search) ---")
    
    history_k = []
    history_error = []
    models_cache = {} # Store models so we don't have to retrain
    
    for k in range(2, MAX_K + 1):
        print(f"Testing K={k}...", end=" ")
        
        current_k_best_inertia = float('inf')
        current_k_best_model = None
        
        for r in range(RND_M):
            model = KMeansCanberra(k=k, max_iters=20)
            model.fit(X_train_sample)
            
            if model.inertia < current_k_best_inertia:
                current_k_best_inertia = model.inertia
                current_k_best_model = model
        
        mean_error = current_k_best_inertia / len(X_train_sample)
        print(f"Mean Error: {mean_error:.4f}")
        
        history_k.append(k)
        history_error.append(mean_error)
        models_cache[k] = current_k_best_model

    # --- AUTOMATIC ELBOW DETECTION ---
    print("\nCalculating Optimal K (Elbow Method)...")
    best_k = find_optimal_k(np.array(history_k), np.array(history_error))
    print(f"-> Optimal K detected: {best_k}")
    
    best_model = models_cache[best_k]
    static_centroids = best_model.centroids
    
    print("\n--- Phase 3: Dynamic Tracking ---")
    
    print(f"Loading Summer data ({SUMMER_START} to {SUMMER_END})...")
    mask_summer = (df['date'] >= SUMMER_START) & (df['date'] <= SUMMER_END)
    df_summer = df.loc[mask_summer].copy()
    
    if df_summer.empty:
        print("WARNING: No Summer data found.")
    else:
        if points_per_day == 24:
            df_summer['step_index'] = df_summer['date'].dt.hour
        elif points_per_day == 48:
            df_summer['step_index'] = df_summer['date'].dt.hour * 2 + (df_summer['date'].dt.minute // 30)
        else:
            df_summer['date_only'] = df_summer['date'].dt.date
            df_summer['step_index'] = df_summer.groupby(['meter', 'date_only']).cumcount()

        df_summer['day_str'] = df_summer['date'].dt.date.astype(str)
        df_summer['day_id'] = df_summer['day_str'] + "_" + df_summer['meter']
        
        X_summer_df = df_summer.pivot(index='day_id', columns='step_index', values='data').dropna()
        
        # NORMALIZE SUMMER DATA TOO
        print("Normalizing Summer data...")
        X_summer_norm = normalize_data(X_summer_df.values)
        
        print("Classifying Summer days...")
        summer_labels = best_model.predict(X_summer_norm)
        
        results_df = pd.DataFrame({
            'day_id': X_summer_df.index,
            'cluster': summer_labels
        })
        results_df[['date_str', 'meter']] = results_df['day_id'].str.split('_', expand=True)
        results_df['date'] = pd.to_datetime(results_df['date_str'])
        
        heatmap_data = results_df.pivot(index='meter', columns='date', values='cluster')
        plt.figure(figsize=(15, 10))
        first_day_col = heatmap_data.columns[0]
        heatmap_data = heatmap_data.sort_values(by=first_day_col)
        cmap = plt.get_cmap('tab10', best_k)
        sns.heatmap(heatmap_data, cmap=cmap, cbar_kws={'ticks': range(best_k)})
        plt.title('Dynamic Tracking: Customer Behavior Clusters through Summer')
        plt.xlabel('Date')
        plt.ylabel('Meters (Sorted)')
        plt.tight_layout()
        plt.savefig('graph2_dynamic_tracking.png')
        print("Saved 'graph2_dynamic_tracking.png'")
    
    print("\n--- Phase 4: Results & Visualization ---")
    
    # Graph 0: Elbow Curve with Indicator
    plt.figure(figsize=(10, 6))
    plt.plot(history_k, history_error, marker='o', label='Mean Error')
    # Add red line for selected K
    error_at_best_k = history_error[history_k.index(best_k)]
    plt.axvline(x=best_k, color='r', linestyle='--', label=f'Optimal K={best_k}')
    plt.scatter([best_k], [error_at_best_k], color='red', s=100, zorder=5)
    
    plt.title('Elbow Curve: Mean Canberra Error vs K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Mean Error')
    plt.legend()
    plt.grid(True)
    plt.savefig('graph0_elbow_curve.png')
    print("Saved 'graph0_elbow_curve.png'")

    n_clusters = len(static_centroids)
    cols = 3
    rows = (n_clusters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()
    
    for i in range(n_clusters):
        ax = axes[i]
        ax.plot(static_centroids[i], color=f'C{i}', linewidth=2)
        ax.set_title(f'Cluster {i+1}')
        
        # Adjust Y-Axis for Percentage
        # Max of centroids is usually around 0.10 (10%) or 0.20 (20%)
        y_max = np.max(static_centroids) * 1.2
        ax.set_ylim(0, y_max) 
        
        ax.grid(True, alpha=0.3)
        if i >= n_clusters - cols:
            ax.set_xlabel('Time Step (Hour)')
        
        # Add peak text
        peak_hour = np.argmax(static_centroids[i])
        peak_val = np.max(static_centroids[i])
        ax.text(0.5, 0.9, f"Peak: Hour {peak_hour} ({peak_val:.1%})", transform=ax.transAxes, 
                ha='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
            
    for i in range(n_clusters, len(axes)):
        axes[i].axis('off')
        
    plt.suptitle(f'Behavioral Profiles (% of Daily Usage) (Spring Training, K={best_k})')
    plt.tight_layout()
    plt.savefig('graph1_static_clusters.png')
    print("Saved 'graph1_static_clusters.png'")

if __name__ == "__main__":
    run_pipeline()