# EM401 - Using Population Dynamics to Model Large Scale Residential Electrical Demand 
# Code Overview 

* This explanatory file details the use of every script within the project as well Input/Output for each
* Numbered folders represent each phase of work to be ran chronologically 
* Each folder contains; input(s), script, output(s)
* Further breakdown of more complex scripts can be found in 'Code Walkthrough' folder
* 'Alternate Datasets' contains work carried out on GoiEner and Slovak Datasets

# Required Libraries 
* pandas
* geopandas
* matplotlib
* random
* s3fs
* pyarrow
* contextily
* fsspec
* boto3
* numpy
* scipy
* seaborn
* re
* pysindy
* json

# Workflow

## 2024 Preprocessing 

### sample_2024.py 
* initial script to gather sample data from SSEN (200 Virtual Feeders)
* cleans data and filters for feeders with aggregation 40>N<50
* OUTPUT: "200_2024.parquet"

### sample_checker_2024.py 
* reads parquet file and conducts health check to look for missing readings and gaps (outages)  
* creates summary and plot of active feeders by day 
* INPUT: "200_2024.parquet"
* OUTPUT: "sample_summary.txt" , "feeder_plot.png"


## Static - Optimal K

### static.py
* applies custom made K Means clustering with canberra distancing to find optimal number of clusters prioritising shape over magnitude 
* INPUT: "200_2024.parquet"
* OUTPUT: "centroids.npy" , "cluster_plot.png" , "clusters_2024.parquet" , "elbow_plot.png" 


## Static - K=2 


### static_k2.py    
* applies custom made K Means clustering with canberra distancing 
* INPUT: "200_2024.parquet"
* OUTPUT: "clusters_2024_k2.parquet" , "cluster_plot_k2.png" , "centroids_k2.npy"

## 2025 Preprocessing 

### sample_2025.py 
* takes 2025 data for sample feeders 
* cleans data and saves as parquet file 
* OUTPUT: "200_2025.parquet"

### sample_checker_2025.py 
* reads parquet file and conducts health check to find date range, missing data and complete days
* prints summary in terminal 
* INPUT: "200_2025.parquet"



## Simple Dynamic

### markov_training.py 
* extracts transition matrices for agent switching behavior by tracking day-to-day cluster changes
* splits data by season (winter/summer) and weekday/weekend
* INPUT: "clusters_2024_k2.parquet"
* OUTPUT: Contextual CSV , PNG

### preprocessing.py 
* calculates daily high user ratios and groups them by context string
* applies 3-day rolling average to smooth extreme spikes before taking derivatives
* INPUT: "clusters_2024_k2.parquet", "200_2025.parquet", "centroids_k2.npy"
* OUTPUT: "clusters_2025_k2.parquet", "pysindy_ratio_2024.csv", "pysindy_ratio_2025.csv"

### boundary.py 
* simulates 14-day agent transition using Markov probability regimes
* extracts true median demand shapes from raw data, filtering out massive anomalies
* INPUT: "200_2024.parquet"
* OUTPUT: "boundary.png"

### pysindy_training.py
* uses Sparse Identification of Nonlinear Dynamics (SINDy) to find governing equations
* outputs mathematical formulas representing physical growth rates
* INPUT: "pysindy_ratio_2024.csv"
* OUTPUT: "training_equations.txt", "training_plot.png"

### compare.py 
* iterates year-long simulation of 2025 using both Markov rules and PySINDy differential equations
* blends seasonal coefficients based on date logic to simulate continuous transitions
* INPUT: "pysindy_ratio_2025.csv", seasonal CSV matrices
* OUTPUT: "comparison.png"

### error.py
* deploys 30-run Monte Carlo simulation to evaluate Markov accuracy
* calculates rolling RMSE and absolute errors broken down by seasonal context
* INPUT: "pysindy_ratio_2025.csv", seasonal CSV matrices
* OUTPUT: "error_analysis.png"


## Lotka-Volterra

### extract.py 
* isolates and formats predator/prey ratios from smoothed dataset
* tags dates with appropriate contextual flags
* INPUT: "pysindy_ratio_2025.csv"
* OUTPUT: "lv_dataset.csv"

### equations.py 
* enforces strict Lotka-Volterra parameters (no bias, interaction only) on PySINDy
* filters for continuous context strings to extract true state transitions
* INPUT: "lv_dataset.csv"
* OUTPUT: "coefficients.json"

### maths.py 
* calculates theoretical roots, stationary points, and mathematical stability
* builds Jacobian matrices and calculates Eigenvalues to classify system nature
* INPUT: "coefficients.json"
* OUTPUT: terminal prints

### graph.py 
* builds 20x20 visual vector fields (gravity) from extracted coefficients
* plots physical constraints and highlights stable vs unstable nodes
* INPUT: "coefficients.json"
* OUTPUT: "phase_plot.png"

### drift.py 
* calculates macro-level baseline volume and peak demand growth from 2024 to 2025
* INPUT: "200_2024.parquet", "200_2025.parquet"
* OUTPUT: terminal prints


## Peak Demand

### demand.py 
* blends mathematical roots with physical grid centroids to forecast coincident system peak
* calculates physical fragility margins in kW between safe states and tipping points
* INPUT: "centroids_k2.npy", "coefficients.json"
* OUTPUT: "demand_[context].png"

### bifurcation.py 
* runs continuous mathematical stress sweep to find exact parameters for topological collapse
* converts abstract stability history back into observable physical kW limits
* INPUT: "centroids_k2.npy", "coefficients.json"
* OUTPUT: "bifurcation_scenario_1.png", "bifurcation_scenario_2.png"