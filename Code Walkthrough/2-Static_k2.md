# Static K=2 Code Walkthrough - static_k2.py 

## Methodology Switch 
* unlike the variable static model, K is preset to exactly 2 clusters
* designed to explicitly isolate high vs low magnitude users, intentionally skipping the normalisation step

## Data Preprocessing 
* loads parquet file for 2024 training period
* converts raw 30-minute timestamped data into a daily 48-vector matrix

## Downsampling 
* randomly samples 3000 days from the matrix if dataset is too large
* preserves enough variance to identify centroids without crashing processing limits

## Custom K-Means Logic
* Restarts: outer loop runs the full clustering process 10 times
* Distance Calculation: uses Canberra Distance to assign points to the nearest of the two centroids
* Averaging: moves centroid to the mathematical mean of all assigned points
* Convergence: inner loop breaks if centroids move less than 0.001
* Error: logs the total distance error, retains the centroid positions from the restart with the lowest overall error

## Clustering Balance 
* sums the total energy of each finalised centroid
* forces the lower-usage centroid to always be indexed as Cluster 0 and higher-usage as Cluster 1 for downstream consistency

## Full Dataset Execution
* runs the 2-cluster model across the entire dataset to capture full yearly behavior

## Outputs
* "centroids_k2.npy" 
* "clusters_2024_k2.parquet" 
* "cluster_plot_k2.png" 