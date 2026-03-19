# Lotka-Volterra & System Stability

## Methodology
* forces PySINDy outputs into strict Lotka-Volterra population dynamics (x' = Ax + By + Cxy)
* bypasses probability to mathematically analyse the grid as a predator-prey ecosystem
* calculates physical stability, tipping points and maximum safety margins

## extract.py
* isolates and formats predator/prey ratios from the smoothed 2025 dataset
* tags precise contextual flags to prepare for mathematical extraction
* Outputs: "lv_dataset.csv"

## equations.py
* configures PySINDy with bias removed and interaction-only terms strictly enforced
* loops through daily transitions, skipping non-continuous days to ensure pure derivatives
* Outputs: "coefficients.json" containing pure A, B, and C vectors

## maths.py
* calculates theoretical equilibrium points (roots) using the quadratic formula
* discards complex and non-physical roots (outside 0 to 1 bounds)
* builds Jacobian matrices and computes Eigenvalues to classify system stability (Stable Node vs Unstable Saddle)
* Outputs: terminal prints of roots, matrices, and classifications

## graph.py
* translates derived mathematics into a 20x20 visual vector field (gravity)
* plots the physical system constraint line (x + y = 1)
* dynamically maps stable attractors and unstable edges based on Eigenvalue polarity
* Outputs: "phase_plot.png"

## drift.py
* calculates the macro-level systemic drift between 2024 and 2025
* calculates pure volume baseline growth and average coincident peak growth
* Outputs: terminal prints of drift percentages

## demand.py
* bridges abstract mathematical equilibrium with actual K=2 physical grid centroids
* blends physical curves based on the mathematically derived stable/unstable population ratios
* calculates the "fragility margin" (delta between stable safe state and unstable tipping point)
* plots scenarios against the physical 8.075 kW hardware limit
* Outputs: "demand_[context].png" for each valid season

## bifurcation.py
* applies a mathematical stress multiplier (μ) to baseline coefficients, simulating continuous degradation
* solves the quadratic discriminant iteratively until the system hits Topological Collapse (discriminant < 0)
* converts surviving mathematical roots into physical peak kW predictions to track the narrowing operational margin
* Outputs: "bifurcation_scenario_1.png", "bifurcation_scenario_2.png"