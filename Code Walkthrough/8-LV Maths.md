# Mathematical Stability Analysis - maths.py

## Methodology
* bypasses simulation to analyze system mathematically
* calculates theoretical limits, stationary points, and stability classifications

## Roots Calculation
* takes A, B, C coefficients and formulates quadratic equation
* solves for system equilibrium roots
* discards complex roots and roots outwith physical bounds (0 to 1)

## Jacobian & Stability
* identifies theoretical stability points
* builds Jacobian matrix using partial derivatives of the system
* computes Eigenvalues to determine stability nature
* classifies point as Stable, Unstable, or Neutrally Stable based on Eigenvalue polarity

## Outputs
* prints system roots, stationary points, Jacobian matrix, Eigenvalues, and classification to terminal