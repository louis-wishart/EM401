import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


INPUT_FILE = "clusters_2024_k2.parquet"

# Define Seasons 
WINTER_MONTHS = [10, 11, 12, 1, 2, 3]       # Oct - Mar
SUMMER_MONTHS = [4, 5, 6, 7, 8, 9]          # Apr - Sept


df = pd.read_parquet(INPUT_FILE)

# Extract data, find transitions
split_data = df['day_id'].str.split('_', expand=True)
df['date'] = pd.to_datetime(split_data[0])
df['feeder'] = split_data[1]  
df = df.sort_values(['feeder', 'date'])

# Shift columns 
df['next_cluster'] = df.groupby('feeder')['cluster'].shift(-1)
df['next_date'] = df.groupby('feeder')['date'].shift(-1)

# Calculate gap 
df['gap'] = (df['next_date'] - df['date']).dt.days
valid = df[df['gap'] == 1].copy()

# Contexts
valid['month'] = valid['date'].dt.month
valid['is_weekend'] = valid['date'].dt.dayofweek >= 5
valid['is_winter'] = valid['month'].isin(WINTER_MONTHS)

winter_wd = valid[(valid['is_winter'] == True)  & (valid['is_weekend'] == False)]
winter_we = valid[(valid['is_winter'] == True)  & (valid['is_weekend'] == True)]
summer_wd = valid[(valid['is_winter'] == False) & (valid['is_weekend'] == False)]
summer_we = valid[(valid['is_winter'] == False) & (valid['is_weekend'] == True)]

print(f"  Winter Weekday: {len(winter_wd)} transitions")
print(f"  Winter Weekend: {len(winter_we)} transitions")
print(f"  Summer Weekday: {len(summer_wd)} transitions")
print(f"  Summer Weekend: {len(summer_we)} transitions")

datasets = [
    ("Winter_Weekday", winter_wd),
    ("Winter_Weekend", winter_we),
    ("Summer_Weekday", summer_wd),
    ("Summer_Weekend", summer_we)
]

# Processing Loop
for name, data in datasets:

    # Count transitions and normalise
    matrix = pd.crosstab(data['cluster'], data['next_cluster'], normalize='index')
    
    filename = f"{name.lower()}.csv"
    matrix.to_csv(filename)
    
    # Plot
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1)
    plt.title(f"Transition Matrix: {name}")
    plt.ylabel("Today's State")
    plt.xlabel("Tomorrow's State")
    plt.tight_layout()
    plt.savefig(f"{name.lower()}.png")
    plt.close() 

