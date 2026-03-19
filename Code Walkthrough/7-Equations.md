# Lotka-Volterra Extraction - equations.py

## Methodology
* uses custom PySINDy configuration to isolate specific interaction dynamics
* removes bias and squared terms to force Lotka-Volterra shape (x' = Ax + By + Cxy)

## Data continuity
* iterates through day pairs (today vs tomorrow)
* skips transitions where days are missing or context shifts abruptly (e.g., Weekday to Weekend)
* stores state variables (x, y) and rate of change (dx) into distinct context buckets

## SINDy Training
* checks if context bucket has sufficient data (>5 days)
* fits custom SINDy model to extract core coefficients (A, B, C)

## Outputs
* prints derived physical equations to console
* saves exact coefficients to coefficients.json for mathematical modelling