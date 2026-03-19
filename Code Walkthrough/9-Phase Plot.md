# Phase Portraits - graph.py

## Methodology
* translates mathematical stability into visual vector fields (gravity)
* plots physical limits constraint line (x+y=1)

## Vector Field
* builds 20x20 grid mapping potential population states
* applies Lotka-Volterra coefficients to find vector direction
* normalizes vectors to plot clean speed and direction arrows

## Dynamic Nodes
* calculates real physical roots mathematically
* tests Eigenvalues of the Jacobian to plot Safe States (green) or Tipping Edges (red)
* dynamically tags complex roots as transient states

## Results
* generates 2x2 grid plot of phase portraits
* saves phase_plot.png