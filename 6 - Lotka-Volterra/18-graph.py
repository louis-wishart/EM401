import numpy as np
import matplotlib.pyplot as plt
import json


with open("coefficients.json", 'r') as f:
    contexts = json.load(f)

# Plot Config
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

Y, X = np.mgrid[-0.1:1.1:20j, -0.1:1.1:20j]

for idx, (name, coeffs) in enumerate(contexts.items()):
    if idx >= 4: break # Safety limit for subplots
    
    ax = axes[idx]
    A, B, C = coeffs["A"], coeffs["B"], coeffs["C"]
    
    # Vector Field (Gravity)
    U = A*X + B*Y + C*X*Y
    V = -U  
    
    # Normalise vectors 
    speed = np.sqrt(U**2 + V**2)
    U_norm = U / (speed + 1e-6)
    V_norm = V / (speed + 1e-6)
    
    # Plot Phase
    ax.quiver(X, Y, U_norm, V_norm, speed, cmap='coolwarm', alpha=0.6)
    
    # Plot Constraint (x + y = 1)
    x_line = np.linspace(0, 1, 100)
    y_line = 1.0 - x_line
    ax.plot(x_line, y_line, color='black', linewidth=3, label='Physical Grid Constraint (x+y=1)')
    
    # Plot Stage 3 Results 
    q_a = -C
    q_b = A - B + C
    q_c = B
    
    roots = []
    

    if abs(q_a) > 1e-9: 
        # Quadratic calculation for Summer contexts (C != 0)
        roots = np.roots([q_a, q_b, q_c])
    elif abs(q_b) > 1e-9: 
        # Linear calculation for Winter Weekday (C = 0)
        roots = [-q_c / q_b]
    # If both are 0 (Winter Weekend), roots remains empty []
        
    if len(roots) > 0:
        is_complex = any(np.iscomplex(r) and abs(r.imag) > 1e-6 for r in roots)
        
        if is_complex:
            # Complex
            ax.text(0.5, 0.5, 'Complex Roots\n(Transient)', 
                    horizontalalignment='center', verticalalignment='center',
                    fontsize=12, fontweight='bold', color='purple', alpha=0.7,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        else:
            physical_roots = [r.real for r in roots if 0.0 <= r.real <= 1.0]
            
            for r in physical_roots:
                x_star = r
                y_star = 1.0 - x_star
                
                # Dynamic Stability calc
                df_dx = A + C * y_star
                df_dy = B + C * x_star
                J = np.array([[df_dx, df_dy], [-df_dx, -df_dy]])
                non_zero_eig = sum(np.linalg.eigvals(J))
                
                if non_zero_eig < -1e-6:
                    # Stable Node
                    ax.plot(x_star, y_star, 'go', markersize=15, markeredgecolor='white', markeredgewidth=2, label='Stable Node (Safe State)')
                elif non_zero_eig > 1e-6:
                    # Unstable Threshold
                    ax.plot(x_star, y_star, 'ro', markersize=15, markeredgecolor='white', markeredgewidth=2, label='Unstable Point (Tipping Edge)')

    # Plot
    ax.set_title(f"Phase Portrait: {name}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Prey Population (x)", fontsize=12)
    ax.set_ylabel("Predator Population (y)", fontsize=12)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Plot handles correctly to avoid duplicates on dynamic loops
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.tight_layout()
plt.savefig("phase_plot.png", dpi=300)