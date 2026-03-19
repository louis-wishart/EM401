# Static Code Walkthrough - static.py

## Methodology Switch
* unlike the fixed K model, K is variable (searches 2 to 15)
* prioritises shape over magnitude to better analyze specific usage behaviors (requires normalisation)

## Data Preprocessing
* loads raw parquet file containing 2024 dataset
* filters data for predetermined training period (1/3/24 - 1/3/25)
* converts timestamped row data into a matrix where rows are individual days and columns are the 48 half-hourly readings

## Normalisation & Downsampling
* divides every row by its total sum to convert absolute energy into relative shape (percentage of daily usage)
* checks if dataset has more than 3000 days
* takes random sample of 3000 days if threshold is exceeded to reduce computational load for centroid identification

## Custom K-Means Loop
* K-Loop: iterates through cluster counts from 2 up to 15
* Restarts: runs the nested logic 10 times for each K to avoid luck-based clustering traps
* Distance Calculation: calculates the Canberra Distance between every data point and the current K centroids
* Convergence: stops early if centroids move less than 0.001 between iterations
* Inertia: records the lowest average error across restarts to build the elbow curve

## Geometric Elbow
* calculates the point of maximum curvature on the error plot mathematically rather than relying on manual visual selection
* selects optimal K where adding more clusters yields diminishing returns

## Full Dataset Execution
* applies the optimal K centroids to the entire unfiltered dataset (beyond the 3000-day sample)
* uses Canberra distance to assign every real day to its closest centroid

## Outputs
* saves "centroids.npy" holding the optimal shape coordinates
* saves "clusters_2024.parquet" logging the cluster ID for every feeder day
* saves "elbow_plot.png" showing error vs number of clusters
* saves "cluster_plot.png" visualising the final centroid shapes found