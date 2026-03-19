# Peak Demand Forecasting - demand.py

## Methodology
* bridges abstract mathematical equilibrium with physical grid behavior
* blends static K-Means centroids based on mathematically derived population ratios

## Equilibrium Blending
* calculates stable and unstable mathematical roots
* weights original low and high centroid curves by the derived physical roots
* sums the weighted curves to generate expected peak kW/hh shape

## Margins
* calculates the delta between stable safe state and unstable tipping point
* plots hardware limit (8.075 kW) to show physical transformer safety headroom

## Results
* prints stable kW, unstable kW, and fragility margin to terminal
* saves individual demand_[context].png plots for each valid context