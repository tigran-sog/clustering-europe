# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:25:36 2024

@author: user
"""

######### CODING ATKINSON INDEX IN PYTHON ##########

import numpy as np
import pandas as pd
import seaborn as sns


#### Generating income distributions


#### Downloading existing income distributions
#/ From Milanovic et al 2013, annual income per decile 1998-2008

df = pd.read_stata('data/milanovic.dta')

# Countries I want to look at
countries = ['Nigeria','Brazil','Italy','United States','Sweden','Armenia','Czech Republic']

df_subset = df[df['country'].isin(countries)]

## Filter for only the largest 'year' values for each 'country'
# This gets me a bunch of years around 2008
df_filtered = df_subset[df_subset['year'] == df_subset.groupby('country')['year'].transform('max')]


## Plotting
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Function to format the y-axis labels
def yaxis_format(y, pos):
    return f'{int(y):,}'

# Use the 'fivethirtyeight' style for the plot
plt.style.use("fivethirtyeight")

# Set the size of the figure
plt.figure(figsize=(40, 32))

# Create the line plot
scatter = sns.lineplot(
    x='group', y='RRinc', 
    hue='country', 
    data=df_filtered,
    linewidth=2,
    legend='full'
)

# Set the y-axis to a logarithmic scale
plt.yscale('log')

# Use FuncFormatter to format the y-axis labels
plt.gca().yaxis.set_major_formatter(FuncFormatter(yaxis_format))

# Adjust font size of plot elements
plt.tick_params(axis='both', which='major', labelsize=40)  # Adjust font size for x and y axes
plt.legend(title='Country', title_fontsize=40, fontsize=30, markerscale=3,
           frameon=True, facecolor='#FCFCFC')
plt.xlabel('Income decile', fontsize=50, labelpad=30)
plt.ylabel('Real annual income ($)', fontsize=50, labelpad=30)
plt.title('Income per decile for selected countries in 2008', fontsize=60, pad=50, loc='left')

# Show the plot
plt.show()
