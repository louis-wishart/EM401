import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

FILE_CENTROIDS = "centroids.npy"

hours = np.linspace(0, 24, 48, endpoint=False)

k_data = np.load(FILE_CENTROIDS)
    # Ensure correct orientation (k clusters, 48 time steps)
if k_data.shape[0] == 48:
        k_data = k_data.T

num_clusters = k_data.shape[0]

cluster_names = [f"Cluster {i+1}" for i in range(num_clusters)]
df_profiles = pd.DataFrame(k_data.T, columns=cluster_names)

# Pearson Correlation 
correlation_matrix = df_profiles.corr()
plt.rcParams.update({'font.size': 11, 'font.family': 'sans-serif'})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), gridspec_kw={'width_ratios': [1, 1.2]})

# Plot A: The Profiles
colors = sns.color_palette("husl", num_clusters)
for i in range(num_clusters):
    ax1.plot(hours, k_data[i], color=colors[i], linewidth=2.5, label=f'C{i+1}')

ax1.set_title(f"K={num_clusters} Complete Profile Plot", fontsize=13, fontweight='bold')
ax1.set_ylabel("Intensity (Share of Day)", fontsize=11, fontweight='bold')
ax1.set_xlabel("Time of Day", fontsize=11, fontweight='bold')
ax1.set_xticks(range(0, 25, 4))
ax1.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 4)])
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend(loc='upper left', ncol=2, fontsize=9)

# Plot B: Correlation Heatmap
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", 
            vmin=-1, vmax=1, square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax2)

ax2.set_title(f"Pairwise Profile Correlation (Pearson $r$)\n(k={num_clusters})", fontsize=13, fontweight='bold')
ax2.set_xlabel("Cluster", fontsize=11, fontweight='bold')
ax2.set_ylabel("Cluster", fontsize=11, fontweight='bold')

# save
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("pairwise.png", dpi=300)

# Output
high_corr_pairs = []
for i in range(num_clusters):
    for j in range(i):
        r_val = correlation_matrix.iloc[i, j]
        if r_val > 0.80:
            high_corr_pairs.append((cluster_names[i], cluster_names[j], r_val))

print("Redundant Pairs (r > 0.80):")
for c1, c2, r in high_corr_pairs:
    print(f" -> {c1} & {c2}: r = {r:.2f}")

with open("redundant_pairs.txt", "w") as text_file:
    text_file.write("Redundant Pairs (r > 0.80):\n")
    for c1, c2, r in high_corr_pairs:
        text_file.write(f" -> {c1} & {c2}: r = {r:.2f}\n")