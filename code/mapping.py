# -*- coding: utf-8 -*-
"""
Created on Wed May 22 19:15:44 2024

@author: user
"""

##### MAPPING THE CLUSTERS #####
## mapping.py

### Import libraries and data
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



## Import cluster dataset
df = pd.read_csv('data/european clusters.csv')

## Import European country borders shapefile
world = gpd.read_file('data/borders/ne_10m_admin_0_sovereignty.shp')[['NAME_LONG','CONTINENT','geometry']]

# Filter for European countries
europe = world[world['CONTINENT'] == 'Europe']

# Rename countries to align with names in df
name_mapping = {
    'United Kingdom': 'UK',
    'Russian Federation': 'Russia',
    'Czech Republic' : 'Czechia'
}
europe['NAME_LONG'] = europe['NAME_LONG'].replace(name_mapping)

# Merge the cluster data with the European countries GeoDataFrame
europe_clusters = europe.merge(df, how='left', left_on='NAME_LONG', right_on='ShortName')

### Creating maps
## Maps of the clusters
# Define the number of categories for each map
num_categories_list = [2, 4, 6]

# Define the base colormap
base_cmap = plt.get_cmap('Accent')

# Create custom colormaps for each number of categories
custom_cmaps = [mcolors.ListedColormap(base_cmap.colors[:num_categories]) for num_categories in num_categories_list]

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(60, 20))
plt.style.use("fivethirtyeight")

# Plot each cluster column in its respective subplot
columns = ['k2', 'k4', 'k6']
titles = ['k = 2', 'k = 4', 'k = 6']

for ax, column, title, cmap in zip(axes, columns, titles, custom_cmaps):
    europe_clusters.boundary.plot(ax=ax, linewidth=1, edgecolor='#171717')
    europe_clusters.plot(column=column, ax=ax, legend=False, categorical=True, cmap=cmap,
                         missing_kwds={"color": "lightgrey"})
    bbox = (-20, 33, 50, 72)  # Limit the bounding box of the plot to Europe (minx, miny, maxx, maxy)
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_title(title, fontsize=50, fontweight='semibold', color = '#171717')
    ax.axis('off')
    ax.grid(False)

# Customize the overall plot
plt.suptitle('Clustering European countries along human development and inequality',
             fontsize=80, fontweight='semibold', x=0.02, ha='left', color = '#171717', y= 0.99)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('viz/cluster_maps.png')  # Save the figure
plt.show()

## Maps of pre-existing definitions
# Define the number of categories for each map
num_categories_list = [2, 4, 4, 6]

# Define the base colormap
base_cmap = plt.get_cmap('Accent')

# Create custom colormaps for each number of categories
custom_cmaps = [mcolors.ListedColormap(base_cmap.colors[:num_categories]) for num_categories in num_categories_list]

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(21, 20))
axes = axes.flatten()

plt.style.use("fivethirtyeight")

# Plot each cluster column in its respective subplot
columns = ['Cold_War','EuroVoc','UN_Geoscheme','Personal']
titles = ['Cold War (k=2)', 'EuroVoc (k=4)', 'UN Geoscheme (k=4)', 'Personal classification (k=6)']

for ax, column, title, cmap in zip(axes, columns, titles, custom_cmaps):
    europe_clusters.boundary.plot(ax=ax, linewidth=1, edgecolor='#171717')
    europe_clusters.plot(column=column, ax=ax, legend=False, categorical=True, cmap=cmap,
                         missing_kwds={"color": "lightgrey"})
    bbox = (-20, 33, 50, 72)  # Limit the bounding box of the plot to Europe (minx, miny, maxx, maxy)
    ax.set_xlim(bbox[0], bbox[2])
    ax.set_ylim(bbox[1], bbox[3])
    ax.set_title(title, fontsize=30, fontweight='semibold', color = '#171717')
    ax.axis('off')
    ax.grid(False)

# Customize the overall plot
plt.suptitle('Pre-existing definitions of European regional divisions',
             fontsize=50, fontweight='semibold', x=0.02, ha='left', color = '#171717', y = 0.98)
plt.tight_layout()
plt.subplots_adjust(top=0.95)
plt.savefig('viz/division_maps.png')  # Save the figure
plt.show()
