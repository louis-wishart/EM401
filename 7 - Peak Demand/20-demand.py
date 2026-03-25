import numpy as np
import matplotlib.pyplot as plt
import json

FILE_CENTROIDS = "centroids_k2.npy" 
COEFFS_FILE = "coefficients.json"
TRANSFORMER_LIMIT_KW = 8075  

# Load PySINDy coefficients
with open(COEFFS_FILE, 'r') as f:
    contexts = json.load(f)

# K-Means centroids (Aggregated Feeder Data)
raw_data = np.load(FILE_CENTROIDS)
centroids = raw_data.T if raw_data.shape == (48, 2) else raw_data

# Removed the / 1000.0 division so it stays in total feeder scale
if np.max(centroids[0]) > np.max(centroids[1]):
    predator_centroid, prey_centroid = centroids[0], centroids[1]
else:
    prey_centroid, predator_centroid = centroids[0], centroids[1]

# Process each extracted context
for ctx_name, coeffs in contexts.items():
    print(f"\n{ctx_name}")
    
    A, B, C = coeffs["A"], coeffs["B"], coeffs["C"]
    q_a, q_b, q_c = -C, A - B + C, B

    # Solve for equilibrium roots
    if abs(q_a) < 1e-9:
        roots = [-q_c / q_b] if abs(q_b) > 1e-9 else []
    else:
        roots = np.roots([q_a, q_b, q_c])

    valid_roots = [r.real for r in roots if 0.0 <= r.real <= 1.0 and not np.iscomplex(r)]
    
    if not valid_roots:
        print("No valid physical roots")
        continue

    valid_roots.sort(reverse=True) 
    
    stable_node = {"prey": valid_roots[0], "predator": 1.0 - valid_roots[0]}
    unstable_node = {"prey": valid_roots[1], "predator": 1.0 - valid_roots[1]} if len(valid_roots) == 2 else None

    # Reconstruct physical peak demand profiles for the whole feeder
    stable_profile = (stable_node["prey"] * prey_centroid) + (stable_node["predator"] * predator_centroid)
    print(f"Stable Peak Demand:   {np.max(stable_profile):.0f} kW")

    delta_margin = 0
    if unstable_node:
        unstable_profile = (unstable_node["prey"] * prey_centroid) + (unstable_node["predator"] * predator_centroid)
        delta_margin = np.max(unstable_profile) - np.max(stable_profile)
        print(f"Unstable Peak Demand: {np.max(unstable_profile):.0f} kW")
        print(f"Delta Margin:         {delta_margin:.0f} kW headroom")

    # Setup plotting limits and labels
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    fig, ax = plt.subplots(figsize=(10, 6))
    hours = np.linspace(0, 24, 48, endpoint=False)
    ticks = range(0, 25, 4)
    tick_labels = [f"{h:02d}:00" for h in ticks]

    # Plot Equilibrium Forecasts
    ax.plot(hours, stable_profile, color='#264653', linewidth=4, label=f'Stable Attractor (y={stable_node["predator"]*100:.1f}%)')

    if unstable_node:
        ax.plot(hours, unstable_profile, color='darkred', linewidth=3, linestyle='--', label=f'Tipping Point (y={unstable_node["predator"]*100:.1f}%)')
        ax.fill_between(hours, stable_profile, unstable_profile, color='red', alpha=0.1, label=f'Fragility Margin ({delta_margin:.0f} kW)')

    ax.set_title(f"Feeder Peak Demand Profile ({ctx_name})", fontsize=14, fontweight='bold')
    ax.set_xlabel("Hour of the Day", fontsize=12, fontweight='bold')
    ax.set_ylabel("Total Feeder Demand (kW)", fontsize=12, fontweight='bold')
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # The Hardware Limit
    ax.axhline(TRANSFORMER_LIMIT_KW, color='black', linestyle='-.', linewidth=2, label=f'Hardware Limit ({TRANSFORMER_LIMIT_KW} kW)')
    ax.legend(loc='upper left', fontsize=11)

    # Clean up the borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    filename = f"demand_{ctx_name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.close(fig)