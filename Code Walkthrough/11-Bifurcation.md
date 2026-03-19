# Tipping Point Sweeps - bifurcation.py

## Methodology
* applies stress multiplier (μ) to baseline coefficients to simulate systemic degradation
* calculates exact mathematical collapse point

## Bifurcation Sweep
* iterates stress parameter from 1.0 to 1.5 across 3000 resolution steps
* dynamically alters prey birth rate or interaction rate
* solves quadratic discriminant at every step
* stops generating roots when discriminant drops below zero (Topological Collapse)

## kW Conversion
* converts surviving stable and unstable mathematical roots into physical kW peak demand values

## Results
* creates plots showing stable attractor convergence with unstable saddle
* saves bifurcation_scenario_1.png and bifurcation_scenario_2.png