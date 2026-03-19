import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re 


INPUT_FILE = "pysindy_ratio_2025.csv"
OUTPUT_FILE = "comparison.png"

# Load Data and Sort
df = pd.read_csv(INPUT_FILE)
df['date'] = pd.to_datetime(df['date'])
dates = df.sort_values('date')['date']
real_vals = df.sort_values('date')['high_user_ratio_smooth'].values

win_wd = pd.read_csv("winter_weekday.csv", index_col=0).values
win_we = pd.read_csv("winter_weekend.csv", index_col=0).values
sum_wd = pd.read_csv("summer_weekday.csv", index_col=0).values
sum_we = pd.read_csv("summer_weekend.csv", index_col=0).values


# Coeefficients Function
def get_coeffs(name):
    print(f"\n{name} Equation:")
    raw = input("> ") 
    
    nums = re.findall(r"[-+]?\d*\.\d+", raw)
    return [float(x) for x in nums]

c_win_wd = get_coeffs("Winter Weekday")
c_win_we = get_coeffs("Winter Weekend")
c_sum_wd = get_coeffs("Summer Weekday")
c_sum_we = get_coeffs("Summer Weekend")



## Simulation Loop

agents = 5000
pop = np.zeros(agents)
# Start with same ratio as real data
start_count = int(agents * real_vals[0])
pop[:start_count] = 1 
np.random.shuffle(pop)

x_phys = real_vals[0]

# History lists (Start with Day 0)
markov_res = [np.mean(pop)]
pysindy_res = [x_phys]

# Iterate Year 
for i in range(1, len(dates)):
    date = dates.iloc[i-1]
    m = date.month
    d = date.day
    
    # Season Weighting
    w = 0.0
    if m in [10, 11, 12, 1, 2, 3]: 
        w = 1.0 # Winter
    elif m == 9: 
        w = (d / 31.0) # September Blend
    elif m == 4: 
        w = 1.0 - (d / 30.0) # April Blend
    
    
    is_weekend = date.dayofweek >= 5


    # Markov Update
    if is_weekend:
        P = win_we * w + sum_we * (1-w)
    else:
        P = win_wd * w + sum_wd * (1-w)
        
    # Transition Probabilities
    p_0_to_1 = P[0, 1] / (P[0, 0] + P[0, 1])
    p_1_to_0 = P[1, 0] / (P[1, 0] + P[1, 1])
    
    # Roll Dice and Apply
    rolls = np.random.random(agents)
    next_pop = pop.copy()
    
    mask_up = (pop == 0) & (rolls < p_0_to_1)
    next_pop[mask_up] = 1
    
    mask_down = (pop == 1) & (rolls < p_1_to_0)
    next_pop[mask_down] = 0
    
    pop = next_pop
    markov_res.append(np.mean(pop))


    
    # PySindy Update
    if is_weekend:
        c1, c2 = c_win_we, c_sum_we
    else:
        c1, c2 = c_win_wd, c_sum_wd
        
    # Blend Coefficients
    c = [
        c1[0]*w + c2[0]*(1-w),
        c1[1]*w + c2[1]*(1-w),
        c1[2]*w + c2[2]*(1-w)
    ]
    
    # c + ax + bx^2
    dx = c[0] + c[1]*x_phys + c[2]*(x_phys**2)
    x_phys += dx
    
    # 0-1
    if x_phys < 0: x_phys = 0
    if x_phys > 1: x_phys = 1
    
    pysindy_res.append(x_phys)



# Results
rmse_m = np.sqrt(np.mean((real_vals - markov_res)**2))
rmse_p = np.sqrt(np.mean((real_vals - pysindy_res)**2))

print(f"\nMarkov RMSE:  {rmse_m:.4f}")
print(f"PySINDy RMSE: {rmse_p:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(dates, real_vals, color='grey', alpha=0.5, linewidth=4, label='Real 2025')
plt.plot(dates, markov_res, label=f'Markov (RMSE {rmse_m:.2f})')
plt.plot(dates, pysindy_res, linestyle='--', label=f'PySINDy (RMSE {rmse_p:.2f})')

plt.title("Comparison: Agents vs Physics")
plt.ylabel("High User Ratio")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(OUTPUT_FILE)
