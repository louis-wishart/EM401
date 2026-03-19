import pandas as pd
import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt

INPUT_FILE = "pysindy_ratio_2024.csv"
OUTPUT_FILE = "training_equations.txt"
OUTPUT_PLOT = "training_plot.png"

POLY_DEGREE = 2    
THRESHOLD = 0.0001 

# Define Seasons
WINTER_MONTHS = [10, 11, 12, 1, 2, 3]       # Oct - Mar
SUMMER_MONTHS = [4, 5, 6, 7, 8, 9]          # Apr - Sept

# Load Data
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Calculate Derivatives and Gaps
# Note: dx is calculated globally first, meaning Friday dx includes Saturday's state
df['next_val'] = df['high_user_ratio_smooth'].shift(-1)
df['next_date'] = df['date'].shift(-1)

df['dx'] = df['next_val'] - df['high_user_ratio_smooth']
df['dt'] = (df['next_date'] - df['date']).dt.days
valid = df[df['dt'] == 1].copy()

# Contexts
valid['month'] = valid['date'].dt.month
valid['next_month'] = valid['next_date'].dt.month # Added to check seasonal boundaries
valid['is_weekend'] = valid['date'].dt.dayofweek >= 5
valid['is_winter'] = valid['month'].isin(WINTER_MONTHS)
valid['next_is_winter'] = valid['next_month'].isin(WINTER_MONTHS)

win_wd = valid[(valid['is_winter'] == True)  & (valid['is_weekend'] == False)]
win_we = valid[(valid['is_winter'] == True)  & (valid['is_weekend'] == True)]
sum_wd = valid[(valid['is_winter'] == False) & (valid['is_weekend'] == False)]
sum_we = valid[(valid['is_winter'] == False) & (valid['is_weekend'] == True)]



# Drop Fridays from Weekdays to prevent leaking Saturday's state
win_wd = win_wd[win_wd['date'].dt.dayofweek != 4]
sum_wd = sum_wd[sum_wd['date'].dt.dayofweek != 4]

# Drop Sundays from Weekends to prevent leaking Monday's state
win_we = win_we[win_we['date'].dt.dayofweek != 6]
sum_we = sum_we[sum_we['date'].dt.dayofweek != 6]


# Drop the specific days where the season changes overnight 
win_wd = win_wd[win_wd['is_winter'] == win_wd['next_is_winter']]
win_we = win_we[win_we['is_winter'] == win_we['next_is_winter']]
sum_wd = sum_wd[sum_wd['is_winter'] == sum_wd['next_is_winter']]
sum_we = sum_we[sum_we['is_winter'] == sum_we['next_is_winter']]



datasets = [
    ("Winter_Weekday", win_wd),
    ("Winter_Weekend", win_we),
    ("Summer_Weekday", sum_wd),
    ("Summer_Weekend", sum_we)
]

# Training Loop 
plt.figure(figsize=(12, 10)) # Big canvas for 4 plots

f = open(OUTPUT_FILE, "w")
f.write("PySindy Training Equations (2024)\n")

for i, (name, data) in enumerate(datasets):
    
    # Reshape to column vectors
    X = data['high_user_ratio_smooth'].values.reshape(-1, 1)
    X_dot = data['dx'].values.reshape(-1, 1)
    
    # Setup Model 
    model = ps.SINDy(
        feature_library=ps.PolynomialLibrary(degree=POLY_DEGREE), 
        optimizer=ps.STLSQ(threshold=THRESHOLD)
    )
    # Fit Model (Note: passing x_dot overrides the t=1 parameter, which is perfect)
    model.fit(X, x_dot=X_dot, t=1)
    
    # Coefficients
    coeffs = model.coefficients()[0]
    # c[0]=const, c[1]=x, c[2]=x^2
    eq = f"dx/dt = {coeffs[0]:.5f} + {coeffs[1]:.5f}x + {coeffs[2]:.5f}x^2"
    
    print(f"{name}: {eq}")
    f.write(f"{name}: {eq}\n")

    # Plot
    plt.subplot(2, 2, i+1) 
    plt.scatter(X, X_dot, alpha=0.3, color='blue', label='Real Data')
    
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_pred = model.predict(x_range)
    plt.plot(x_range, y_pred, color='red', linewidth=2, label='PySINDy Fit')
    
    plt.title(name)
    plt.xlabel("Population Ratio (x)")
    plt.ylabel("Growth Rate (dx/dt)")
    plt.legend()
    plt.grid(True, alpha=0.3)

f.close()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)