import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json
import warnings
warnings.filterwarnings("ignore")

TARGET_CONTEXT = "SUMMER WD"
TRANSFORMER_LIMIT_KW = 8.075 


with open("coefficients.json", 'r') as f:
    coeffs = json.load(f)[TARGET_CONTEXT]
A_BASE, B_BASE, C_BASE = coeffs["A"], coeffs["B"], coeffs["C"]

raw_data = np.load("centroids_k2.npy")
centroids = raw_data.T if raw_data.shape == (48, 2) else raw_data

# Ensure Predator is the higher peak (converted to kW)
if np.max(centroids[1]) > np.max(centroids[0]):
    prey_centroid, predator_centroid = centroids[0]/1000.0, centroids[1]/1000.0
else:
    prey_centroid, predator_centroid = centroids[1]/1000.0, centroids[0]/1000.0

def exact_peak_kw(y_star):
    x_star = 1.0 - y_star
    blended_profile = (x_star * prey_centroid) + (y_star * predator_centroid)
    return np.max(blended_profile)

# Bifurctaion Sweep

mu_values = np.linspace(1.0, 1.50, 3000)

def run_sweep(stress_target):
    stable_kw, unstable_kw = [], []
    bifurcation_mu, bifurcation_kw = None, None
    
    for mu in mu_values:
        # Apply stress to A or shrink C 
        A_current = A_BASE * mu if stress_target == 'A' else A_BASE
        C_current = C_BASE / mu if stress_target == 'C' else C_BASE
        
        q_a = -C_current
        q_b = A_current - B_BASE + C_current
        q_c = B_BASE
        
        discriminant = (q_b**2) - (4 * q_a * q_c)
        
        if discriminant >= 0:
            x1 = (-q_b + np.sqrt(discriminant)) / (2 * q_a)
            x2 = (-q_b - np.sqrt(discriminant)) / (2 * q_a)
            y1, y2 = 1.0 - x1, 1.0 - x2
            
            y_stable = min(y1, y2)
            y_unstable = max(y1, y2)
            
            stable_kw.append(exact_peak_kw(y_stable))
            unstable_kw.append(exact_peak_kw(y_unstable))
        else:
            if bifurcation_mu is None:
                bifurcation_mu = mu
                bifurcation_kw = stable_kw[-1]
                
            stable_kw.append(np.nan)
            unstable_kw.append(np.nan)
            
    return stable_kw, unstable_kw, bifurcation_mu, bifurcation_kw

stable_A, unstable_A, b_mu_A, b_kw_A = run_sweep('A')
stable_C, unstable_C, b_mu_C, b_kw_C = run_sweep('C')

# Plot
COLOR_STABLE = '#0A2463'    
COLOR_UNSTABLE = '#D8315B'  
COLOR_FILL = '#E0E7FA'      

def create_bifurcation_plot(stable_history, unstable_history, b_mu, b_kw, title, filename):
    plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
    fig, ax = plt.subplots(figsize=(10, 6.5))

    ax.plot(mu_values, stable_history, color=COLOR_STABLE, linewidth=3.5, label='Stable Attractor')
    ax.plot(mu_values, unstable_history, color=COLOR_UNSTABLE, linewidth=2.5, linestyle='--', label='Unstable Saddle')
    ax.fill_between(mu_values, stable_history, unstable_history, color=COLOR_FILL, alpha=0.6, label='Operational Margin')
    
    # Hardware Limit Line
    ax.axhline(TRANSFORMER_LIMIT_KW, color='black', linestyle='-.', linewidth=2.5, label=f'Hardware Limit ({TRANSFORMER_LIMIT_KW} kW)')

    # Dynamic X-Axis limit calculation
    x_max_limit = b_mu + 0.05 if b_mu else max(mu_values)

    if b_mu:
        ax.scatter([b_mu], [b_kw], color=COLOR_UNSTABLE, s=120, zorder=5, edgecolor='white', linewidth=1.5)
        ax.axhline(b_kw, color='black', linestyle=':', alpha=0.3, linewidth=1.5)
        

        ax.axvspan(b_mu, x_max_limit, color='#FAECEF', alpha=0.7)
      
        text_x_pos = b_mu + ((x_max_limit - b_mu) / 2)
        ax.text(text_x_pos, b_kw, 'Topological\nCollapse', 
                ha='center', va='center', color='#8A1C36', fontweight='bold', alpha=0.9, fontsize=12)
        
        ax.annotate(f'Bifurcation Point\n({b_kw:.2f} kW / hh)', 
                     xy=(b_mu, b_kw), 
                     xytext=(b_mu - 0.02, b_kw + 0.25),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=6, headlength=8, edgecolor='none'),
                     ha='right', va='bottom', fontweight='bold', color='black', fontsize=11)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel(r"Stress Parameter ($\mu$)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Coincident Peak Demand (kW / Household)", fontsize=12, fontweight='bold', labelpad=10)

    def pct_formatter(x, pos): return "Baseline" if x == 1.0 else f"+{(x-1.0)*100:.1f}%"
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(pct_formatter))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f kW'))

    ax.grid(True, linestyle=':', alpha=0.7, color='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    ax.legend(loc='upper left', framealpha=0.95, fontsize=10)

    min_kw = np.nanmin(stable_history)
    max_kw = np.nanmax(unstable_history)
    padding = (max_kw - min_kw) * 0.25
    
    ax.set_xlim(1.0, x_max_limit)
    ax.set_ylim(min_kw - padding, max(max_kw + padding, TRANSFORMER_LIMIT_KW + 0.5))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close(fig)

create_bifurcation_plot(stable_A, unstable_A, b_mu_A, b_kw_A, "Scenario 1: Baseline Prey Increase", "bifurcation_scenario_1.png")
create_bifurcation_plot(stable_C, unstable_C, b_mu_C, b_kw_C, "Scenario 2: Predator Synchronisation", "bifurcation_scenario_2.png")