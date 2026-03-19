import numpy as np
import json

# x' = Ax + By + Cxy
with open("coefficients.json", 'r') as f:
    contexts = json.load(f)

# Maths Engine 
for name, coeffs in contexts.items():
    A = coeffs["A"]
    B = coeffs["B"]
    C = coeffs["C"]
    
    print(f"[{name}]")
    
    ## Stationary Points (x*, y*)
    q_a = -C
    q_b = A - B + C
    q_c = B
    
    # Prevent numpy error if quadratic term C is exactly 0
    if abs(q_a) < 1e-9:
        roots = [-q_c / q_b] if abs(q_b) > 1e-9 else []
    else:
        roots = np.roots([q_a, q_b, q_c])
        
    print(f" Roots: {np.round(roots, 3)}")
    
    # Complex Check
    is_complex = any(np.iscomplex(r) and abs(r.imag) > 1e-6 for r in roots)
    
    if is_complex:
        print("  -> Nature: Complex\n")
        continue
    
    # Filter [Real 0-1]
    physical_roots = [r.real for r in roots if 0.0 <= r.real <= 1.0]
    
    if not physical_roots:
        print("  -> Nature: Real, Outwith Bounds (0-1)")
        continue
        
    for i, x_star in enumerate(physical_roots):
        x_star = x_star.real
        y_star = 1.0 - x_star
        
        print(f"  -> Stability Point {i+1}: Prey (x*) = {x_star:.3f}, Predator (y*) = {y_star:.3f}")
        
        ## Jacobian Matrix
        df_dx = A + C * y_star
        df_dy = B + C * x_star
        dg_dx = -df_dx
        dg_dy = -df_dy
        
        J = np.array([
            [df_dx, df_dy],
            [dg_dx, dg_dy]
        ])

        print(f"  -> Jacobian Matrix:\n       [[{df_dx:.3f}, {df_dy:.3f}],\n        [{dg_dx:.3f}, {dg_dy:.3f}]]")
        
        ## Eigenvalues &
        #  Stability 
        eigenvalues = np.linalg.eigvals(J)
        non_zero_eig = sum(eigenvalues)
        
        print(f"  -> Eigenvalues: \u03BB1 = {eigenvalues[0]:.3f}, \u03BB2 = {eigenvalues[1]:.3f}")
        
        if non_zero_eig < -1e-6:
            classification = "Stable"
        elif non_zero_eig > 1e-6:
            classification = "Unstable"
        else:
            classification = "Neutrally Stable"
            
        print(f"  -> Classification: {classification}")
        print()