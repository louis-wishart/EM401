# Error Analysis - error.py

## Methodology
* deploys 30-run Monte Carlo simulation to eliminate randomness from Markov evaluation
* compares physical simulation (PySindy) and probabilistic simulation (Markov) against real 2025 data

## Seasonal Weighting
* applies dynamic blending multiplier to transition months (April, September)
* assigns pure weights (1.0 or 0.0) to hard Winter and Summer months

## PySindy Simulation
* extracts dynamic seasonal coefficients
* iterates day by day, blending seasonal physics to calculate daily rate of change
* bounds result strictly between 0 and 1

## Markov Simulation
* repeats simulation 30 times with 5,000 agents to find median behavior
* blends probability matrices based on seasonal weighting
* rolls random probabilities for state transitions and takes 50th percentile across all runs to eliminate luck

## Error Calculation
* extracts absolute error margin between model predictions and actual data
* calculates global RMSE and MAE
* tracks 30-day rolling RMSE to identify chronological degradation

## Results
* saves four-panel error_analysis.png (Bias, Rolling RMSE, Histogram, Boxplots)
* prints overall RMSE/MAE and best-performing model per context