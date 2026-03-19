import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# Files
FILE_RAW_2024 = "200_2024.parquet"
PARQUET_DATE_COL = "data_collection_log_timestamp"  
PARQUET_DEMAND_COL = "total_consumption_active_import"       

# Markov Matrix (Ensure you replace these with your Winter CSV values for best results!)
REGIME_WEEKDAY = {"P_01": 0.0910, "P_10": 0.0987} 
REGIME_WEEKEND = {"P_01": 0.0919, "P_10": 0.1061} 

# Markov Simulation 
SIMULATION_AGENTS = 5000
DAYS = 14
agents = np.zeros(SIMULATION_AGENTS, dtype=int)

agents[:int(SIMULATION_AGENTS * 0.479)] = 1

history_y = [np.mean(agents)]
day_labels = []

calendar = [
    ('Tue', REGIME_WEEKDAY), ('Wed', REGIME_WEEKDAY), ('Thu', REGIME_WEEKDAY), ('Fri', REGIME_WEEKDAY),
    ('Sat', REGIME_WEEKEND), ('Sun', REGIME_WEEKEND),
    ('Mon', REGIME_WEEKDAY), ('Tue', REGIME_WEEKDAY), ('Wed', REGIME_WEEKDAY), ('Thu', REGIME_WEEKDAY), ('Fri', REGIME_WEEKDAY),
    ('Sat', REGIME_WEEKEND), ('Sun', REGIME_WEEKEND),
    ('Mon', REGIME_WEEKDAY)
]

for day_name, regime in calendar:
    day_labels.append(day_name)
    rolls = np.random.random(SIMULATION_AGENTS)
    
    mask_up = (agents == 0) & (rolls < regime["P_01"])
    mask_down = (agents == 1) & (rolls < regime["P_10"])
    
    agents[mask_up] = 1
    agents[mask_down] = 0
    
    history_y.append(np.mean(agents))

# Plot B - 

df_raw = pd.read_parquet(FILE_RAW_2024)

# --- THE FIX: FILTER OUTLIERS ---
# Remove massive anomalous data spikes that corrupt the shape
df_raw = df_raw[df_raw[PARQUET_DEMAND_COL] < 20000] 
# --------------------------------

df_raw['timestamp'] = pd.to_datetime(df_raw[PARQUET_DATE_COL], utc=True).dt.tz_localize(None)

df_raw['hour'] = df_raw['timestamp'].dt.hour + (df_raw['timestamp'].dt.minute / 60.0)
df_raw['day_of_week'] = df_raw['timestamp'].dt.dayofweek  # 0=Mon, 6=Sun

df_raw['is_weekend'] = df_raw['day_of_week'].isin([5, 6])

df_raw['kw'] = df_raw[PARQUET_DEMAND_COL] / 1000.0

# --- THE FIX: USE MEDIAN ---
# Median finds the true 'typical' shape of the grid, immune to extreme high-users
profile_pivot = df_raw.groupby(['hour', 'is_weekend'])['kw'].median().unstack('is_weekend')
# --------------------------------

hours = profile_pivot.index
weekday_kw = profile_pivot[False]
weekend_kw = profile_pivot[True]

# Plot 

plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), gridspec_kw={'height_ratios': [1, 1.2]})

# Plot A - Markov
ax1.plot(range(DAYS + 1), history_y, color='#264653', linewidth=3, marker='o', markersize=8, label='Predator Ratio')

for i in range(DAYS):
    if calendar[i][0] in ['Sat', 'Sun']:
        ax1.axvspan(i, i+1, color='#E76F51', alpha=0.15, label='Weekend' if i==4 else "")

ax1.axhline(y=0.479, color='#2A9D8F', linestyle='--', linewidth=2, label='Weekday Equilibrium (~47.9%)')
ax1.axhline(y=0.464, color='#E76F51', linestyle=':', linewidth=2, label='Weekend Equilibrium (~46.4%)')

ax1.set_title("Population Ratio by Day", fontsize=14, fontweight='bold')
ax1.set_ylabel("Predator Population Ratio (y)", fontsize=11, fontweight='bold')
ax1.set_xticks(range(1, DAYS + 1))
ax1.set_xticklabels(day_labels)
ax1.set_ylim(0.44, 0.51)
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.legend(loc='upper right')

# Plot B - Demand Profiles
ax2.plot(hours, weekday_kw, color='#264653', linewidth=3, label='Weekday Profile')
ax2.plot(hours, weekend_kw, color='#E9C46A', linewidth=3.5, label='Weekend Profile')

ax2.fill_between(hours, weekend_kw, weekday_kw, 
                 where=(weekend_kw > weekday_kw), 
                 interpolate=True, color='#E9C46A', alpha=0.3, label='Excess Demand')

ax2.annotate('The Resolution:\nDemand is lower at peak, but sustained all day.', 
             xy=(12, 0.6), xytext=(6, 0.9),
             arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.5),
             fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

ax2.set_title("Weekend vs Weekday Demand Distribution", fontsize=14, fontweight='bold')
ax2.set_xlabel("Hour of the Day (00:00 - 24:00)", fontsize=11, fontweight='bold')
ax2.set_ylabel("Median System Demand (kW)", fontsize=11, fontweight='bold')
ax2.set_xticks(range(0, 25, 2))
ax2.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)])
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.legend(loc='upper left')

plt.tight_layout(pad=3.0)
output_file = "boundary.png"
plt.savefig(output_file, dpi=300)

# Metrics 
print(f"Weekday Peak: {weekday_kw.max():.2f} kW | Weekday Minimum: {weekday_kw.min():.2f} kW")
print(f"Weekend Peak: {weekend_kw.max():.2f} kW | Weekend Minimum: {weekend_kw.min():.2f} kW")

weekday_volatility = weekday_kw.max() - weekday_kw.min()
weekend_volatility = weekend_kw.max() - weekend_kw.min()

print(f"\nDemand Volatility (Peak minus Trough):")
print(f"  -> Weekday: {weekday_volatility:.2f} kW swing (Highly Volatile)")
print(f"  -> Weekend: {weekend_volatility:.2f} kW swing (More Consistent)")
if weekend_volatility < weekday_volatility:
    print(f"\n[CONCLUSION] Weekends are {(1 - weekend_volatility/weekday_volatility)*100:.1f}% more consistent than weekdays.")