import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error


FILE_INPUT_RATIO = "pysindy_ratio_2025.csv"
FILE_WIN_WD = "winter_weekday.csv"
FILE_WIN_WE = "winter_weekend.csv"
FILE_SUM_WD = "summer_weekday.csv"
FILE_SUM_WE = "summer_weekend.csv"

# PySindy
COEFFS_WIN_WD = [-0.00941, 0.10672, -0.19965]
COEFFS_WIN_WE = [0.13553, -0.67164, 0.82987]
COEFFS_SUM_WD = [0.01875, -0.15834, 0.28131]
COEFFS_SUM_WE = [0.00441, 0.09181, -0.31577]


N_AGENTS = 5000
MONTE_CARLO_RUNS = 30  # Eliminate randomness (mean)



# Extract Data
df_actual = pd.read_csv(FILE_INPUT_RATIO)

# Safely extract dates
if 'date' in df_actual.columns:
    df_actual['date'] = pd.to_datetime(df_actual['date'])
    df_actual = df_actual.sort_values('date')
    dates = df_actual['date'].reset_index(drop=True)

actual_ratios = df_actual['high_user_ratio_smooth'].values
days_count = len(actual_ratios)

# Seasonal Weighting 
def get_seasonal_factor(date):
    m = date.month
    d = date.day
    if m in [10, 11, 12, 1, 2, 3]: return 1.0 
    if m in [5, 6, 7, 8]:          return 0.0 
    if m == 4: return 1.0 - (d / 30.0)        
    if m == 9: return (d / 30.0)              
    return 0.0

# Boxplot Contexts
day_contexts = []
for d in dates:
    w = get_seasonal_factor(d)
    is_weekend = (d.dayofweek >= 5)
    
    if w > 0.5:
        day_contexts.append("Winter WE" if is_weekend else "Winter WD")
    elif w < 0.5:
        day_contexts.append("Summer WE" if is_weekend else "Summer WD")
    else:
        day_contexts.append("Transition")



## PySindy Sim
ratio_pysindy = []
current_x = actual_ratios[0] 

for i in range(days_count):
    ratio_pysindy.append(current_x)
    
    date = dates[i]
    is_weekend = (date.dayofweek >= 5)
    w = get_seasonal_factor(date) 
    
    
    if is_weekend:
        dx_winter = COEFFS_WIN_WE[0] + (COEFFS_WIN_WE[1] * current_x) + (COEFFS_WIN_WE[2] * (current_x**2))
        dx_summer = COEFFS_SUM_WE[0] + (COEFFS_SUM_WE[1] * current_x) + (COEFFS_SUM_WE[2] * (current_x**2))
    else:
        dx_winter = COEFFS_WIN_WD[0] + (COEFFS_WIN_WD[1] * current_x) + (COEFFS_WIN_WD[2] * (current_x**2))
        dx_summer = COEFFS_SUM_WD[0] + (COEFFS_SUM_WD[1] * current_x) + (COEFFS_SUM_WD[2] * (current_x**2))
    
    # Blend seasons and update
    dx_total = (w * dx_winter) + ((1 - w) * dx_summer)
    current_x = np.clip(current_x + dx_total, 0.0, 1.0)



## Markov Sim

M_win_wd = pd.read_csv(FILE_WIN_WD, index_col=0).values
M_win_we = pd.read_csv(FILE_WIN_WE, index_col=0).values
M_sum_wd = pd.read_csv(FILE_SUM_WD, index_col=0).values
M_sum_we = pd.read_csv(FILE_SUM_WE, index_col=0).values

markov_all_runs = np.zeros((MONTE_CARLO_RUNS, days_count))

for run in range(MONTE_CARLO_RUNS):
  
    current_agents = np.zeros(N_AGENTS)
    num_high = int(actual_ratios[0] * N_AGENTS)
    current_agents[:num_high] = 1 
    np.random.shuffle(current_agents)
    
    for i in range(days_count):
        markov_all_runs[run, i] = np.mean(current_agents)
        
        date = dates[i]
        is_weekend = (date.dayofweek >= 5)
        w = get_seasonal_factor(date) 
        
        # Unpacked Matrix Blending
        if is_weekend:
            M_today = (M_win_we * w) + (M_sum_we * (1 - w))
        else:
            M_today = (M_win_wd * w) + (M_sum_wd * (1 - w))
            
        p_0_to_1 = M_today[0, 1] / (M_today[0, 0] + M_today[0, 1])
        p_1_to_0 = M_today[1, 0] / (M_today[1, 0] + M_today[1, 1])
        
        # Roll dice 
        rolls = np.random.random(N_AGENTS)
        switching_up = (current_agents == 0) & (rolls < p_0_to_1)
        switching_down = (current_agents == 1) & (rolls < p_1_to_0)
        
        current_agents[switching_up] = 1
        current_agents[switching_down] = 0
# Median
markov_median = np.percentile(markov_all_runs, 50, axis=0)



## Error Analysis

# e = actual - predicted
error_pysindy = actual_ratios - ratio_pysindy
error_markov = actual_ratios - markov_median

# Global Metrics
rmse_py = np.sqrt(mean_squared_error(actual_ratios, ratio_pysindy))
mae_py = mean_absolute_error(actual_ratios, ratio_pysindy)

rmse_mk = np.sqrt(mean_squared_error(actual_ratios, markov_median))
mae_mk = mean_absolute_error(actual_ratios, markov_median)

# 30 Day Rolling RMSE
rolling_rmse_py = np.sqrt(pd.Series(error_pysindy**2).rolling(window=30, min_periods=1).mean())
rolling_rmse_mk = np.sqrt(pd.Series(error_markov**2).rolling(window=30, min_periods=1).mean())

# Boxplot DataFrame
df_errors = pd.DataFrame({
    'Context': day_contexts * 2,
    'Absolute Error': list(np.abs(error_pysindy)) + list(np.abs(error_markov)),
    'Model': ['PySindy'] * days_count + ['Markov'] * days_count
})
df_errors = df_errors[df_errors['Context'] != 'Transition'] 



## Plotting


plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 1.2, 1], hspace=0.35)

# Plot 1: 14 Day Bias Error 
ax1 = fig.add_subplot(gs[0, :])
ax1.axhline(0, color='black', linewidth=1.5, linestyle='--')
max_err = max(np.max(error_pysindy), np.max(error_markov)) * 1.1
min_err = min(np.min(error_pysindy), np.min(error_markov)) * 1.1

ax1.fill_between(dates, 0, max_err, color='#ffcccc', alpha=0.4, label='High Zone: Underpredicting')
ax1.fill_between(dates, min_err, 0, color='#cce5ff', alpha=0.4, label='Low Zone: Overpredicting')

ax1.plot(dates, error_pysindy, color='#2ca02c', linewidth=1, alpha=0.5, label='PySindy Daily Error')
ax1.plot(dates, error_markov, color='#ff7f0e', linewidth=1, alpha=0.5, label='Markov Daily Error')
ax1.plot(dates, pd.Series(error_pysindy).rolling(14).mean(), color='darkgreen', linewidth=2.5, label='PySindy 14 Day Bias')
ax1.plot(dates, pd.Series(error_markov).rolling(14).mean(), color='darkorange', linewidth=2.5, label='Markov 14 Day Bias')

ax1.set_title("Daily Error with 14 Day Rolling Average", fontweight='bold')
ax1.set_ylabel("Error Margin")
ax1.set_ylim(min_err, max_err)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper left', ncol=2)

# Plot 2: 30 Day Rolling RMSE
ax2 = fig.add_subplot(gs[1, :])
ax2.plot(dates, rolling_rmse_py, color='#2ca02c', linewidth=2.5, label=f'PySindy 30 Day RMSE')
ax2.plot(dates, rolling_rmse_mk, color='#ff7f0e', linewidth=2.5, label=f'Markov 30 Day RMSE')
ax2.fill_between(dates, 0, rolling_rmse_py, color='#2ca02c', alpha=0.1)
ax2.fill_between(dates, 0, rolling_rmse_mk, color='#ff7f0e', alpha=0.1)

ax2.set_title("30 Day Rolling RMSE", fontweight='bold')
ax2.set_ylabel("RMSE")
ax2.set_ylim(0, max(np.max(rolling_rmse_py), np.max(rolling_rmse_mk)) * 1.2)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')

# Plot 3: Histogram 
ax3 = fig.add_subplot(gs[2, 0])
ax3.hist(error_pysindy, bins=40, alpha=0.6, color='#2ca02c', label='PySindy', density=True)
ax3.hist(error_markov, bins=40, alpha=0.6, color='#ff7f0e', label='Markov', density=True)
ax3.axvline(0, color='k', linestyle='--', linewidth=1.5)
ax3.set_title("Histogram of Error Distribution", fontweight='bold')
ax3.set_xlabel("Error Value")
ax3.set_ylabel("Probability Density")
ax3.grid(True, alpha=0.3)
ax3.legend()

# Plot 4: Boxplots
ax4 = fig.add_subplot(gs[2, 1])
sns.boxplot(x='Context', y='Absolute Error', hue='Model', data=df_errors, 
            palette={'PySindy': '#2ca02c', 'Markov': '#ff7f0e'}, ax=ax4, fliersize=3)
ax4.set_title("Seasonal Error Breakdown by Context", fontweight='bold')
ax4.set_xlabel("")
ax4.set_ylabel("Absolute Error Magnitude")
ax4.grid(True, alpha=0.3, axis='y')
ax4.legend(title='')

plt.savefig("error_analysis.png", dpi=300, bbox_inches='tight')


# RMSE and MAE 

print("\n[1] PySindy")
print(f"    RMSE: {rmse_py:.4f}")
print(f"    MAE:  {mae_py:.4f}")

print("\n[2] Markov")
print(f"    RMSE: {rmse_mk:.4f}")
print(f"    MAE:  {mae_mk:.4f}")

print("\n[3] Contextual Accuracy:")
grouped = df_errors.groupby(['Context', 'Model'])['Absolute Error'].mean().unstack()
for context in grouped.index:
    best_model = "PySindy" if grouped.loc[context, 'PySindy'] < grouped.loc[context, 'Markov'] else "Markov"
    print(f"{context:<12} =     {best_model}")

