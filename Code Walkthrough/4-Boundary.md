# Boundary Justification - boundary.py

## Methodology
* simulates 14-day agent transition using Markov probabilities
* extracts median physical demand shapes, filtering out massive anomalies for clean baseline

## Markov Simulation
* initialises 5,000 agents with predefined starting ratio (47.9%)
* iterates through 14-day calendar assigning weekday or weekend probability regimes
* generates random rolls for each individual agent to determine state transition (up or down)
* calculates daily mean of predator ratio to track population drift

## Physical Demand Profiling
* loads raw 2024 dataset and removes extreme outliers (>20,000)
* isolates hour of day and weekday/weekend tags
* calculates median kW demand per half-hour to find true typical shape, ignoring extremes

## Results
* plots two-panel graph: Predator Ratio drift and Weekend vs Weekday Median Demand
* saves plot 
* prints peak, minimum and volatility metrics to console