import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx


FILE_INPUT = "200_2024.parquet"
FILE_OUTPUT = "feeder_map.png"

gdf = gpd.read_parquet(FILE_INPUT)

# Isolate feeder locations 
unique_feeders = gdf.drop_duplicates(subset=['lv_feeder_unique_id']).copy()
num_feeders = len(unique_feeders)


# Coordinates
if unique_feeders.crs is None:
    unique_feeders.set_crs(epsg=4326, inplace=True)

unique_feeders = unique_feeders.to_crs(epsg=3857)



# Plot
fig, ax = plt.subplots(figsize=(10, 12))

unique_feeders.plot(
    ax=ax, 
    color='crimson', 
    markersize=30, 
    alpha=0.8, 
    edgecolor='black',
    label=f'LV Feeders (n={num_feeders})'
)

ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)


ax.set_title("SSEN Feeder Locations", fontsize=14, pad=15, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)

ax.set_xticks([])
ax.set_yticks([])

plt.tight_layout()
plt.savefig(FILE_OUTPUT, dpi=300, bbox_inches='tight')
print(f"-> Map saved to {FILE_OUTPUT}")

plt.show()