# Simple Dynamic Modeling

## Methodology
* deploys a dual-track dynamic strategy to evaluate system behavior
* Track A: First-Order Markov Chain (discrete) to capture random switching probabilities
* Track B: PySINDy (continuous) to derive governing differential equations
* models are trained on 2024 data and simulated against 2025 data to measure accuracy and drift
* splits analysis into Weekend and Weekday contexts to mitigate non-homogeneity 

## preprocessing.py
* calculates daily high-user ratios based on K=2 cluster classifications
* categorises dates into specific contexts (e.g., Winter Weekday, Summer Weekend)
* applies a 3-day rolling average to smooth extreme behavioral spikes before calculating derivatives
* Outputs: "pysindy_ratio_2024.csv", "pysindy_ratio_2025.csv", "clusters_2025_k2.parquet"

## markov_training.py
* extracts transition matrices by tracking how agents switch between Cluster 0 and Cluster 1 day-to-day
* normalizes counts into distinct probability regimes based on season and day of the week
* Outputs: seasonal CSV matrices , PNG heatmaps of transition matrices

## pysindy_training.py
* leverages Sparse Identification of Nonlinear Dynamics (SINDy) on smoothed 2024 ratios
* applies strict boundaries to prevent data leaking (e.g., skipping Friday-to-Saturday transitions)
* extracts polynomial formulas representing the physical growth rate of demand
* Outputs: "training_equations.txt", "training_plot.png"

## boundary.py
* sets a physical baseline by running a 14-day Markov agent simulation to track population drift
* extracts true median demand shapes from raw data, actively filtering out massive anomalies (>20,000)
* Outputs: "boundary.png", terminal prints of volatility metrics

## compare.py
* executes a year-long simulation of 2025 using both discrete Markov rules and continuous PySINDy equations
* dynamically blends seasonal coefficients (using scaling weights for transition months like April/September)
* Outputs: "comparison.png", terminal prints of RMSE comparisons

## error.py
* runs a 30-iteration Monte Carlo simulation of the Markov model to eliminate random luck
* extracts absolute error margins between model predictions and actual 2025 data
* calculates global RMSE/MAE and tracks 30-day rolling RMSE to identify chronological degradation
* Outputs: "error_analysis.png"