# -*- coding: utf-8 -*-
"""
Created on Tue May  7 20:30:21 2024

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
import random
from scipy.stats import pearsonr

import seaborn as sns
import matplotlib.pyplot as plt

import os

import re



# Import and clean the data
df = (
    pd.read_csv('IHDI 2022 dataset Europe.csv')   # Import dataframe
    .query("`Inequality-loss` != '..'")          # Filter out rows where 'Inequality-loss' is '..'
    .query("`PopBelow100k` != 'y'")              # Further filter out rows where 'PopBelow100k' is 'y'
)



## View
df.head()
print(df.dtypes) # We see that "Inequality-loss" column needs to be converted into float

df['Inequality-loss'] = df['Inequality-loss'].astype(float)

# EDA
## Summary stats
print(df.iloc[:,[3,5]].describe())

## Correlation between the two variables

## Scatter plots 

schemes = {
    "Cold_War" : 'Cold War regional classification (k = 2)',
    "UN_Geoscheme" : 'UN Geoscheme regional classification (k = 4)' ,
    "EuroVoc" : 'EuroVoc regional classification (k = 4)',
    "Personal" : 'My own regional classification (k = 6)'
    }


# Preparation for visualisation

## Create a new column for jittered y-values (for visualisation)
np.random.seed(35)
df['JitteredY'] = df['Inequality-loss'] + np.random.uniform(-.2, .2, size=len(df))

## Set the font family globally
plt.rcParams['font.family'] = 'Segoe UI'  # Examples of font families: 'serif', 'sans-serif', 'monospace'

## Ensure a directory exists to save plot images to
directory = f'./ihdi loss viz/'
if not os.path.exists(directory):
    os.makedirs(directory)


## Function to plot European countries by the two variables, coloured by specified cluster set
def cluster_plot(df, cluster_set):
    
    # Is the cluster set referring to a pre-existing definition or a K-means produced one?

    # K-means generated cluster set
    if bool(re.match(r'^k\d*$',cluster_set)):
        k_value = len(df[cluster_set].unique()) # Extract number of clusters (k) from each classification
        title = f'Clustering of European countries by human development and inequality for k = {k_value}'

    # Pre-existing definition cluster set    
    else:
        k_value = len(df[cluster_set].unique()) # Extract number of clusters (k) from each classification
        title = schemes[cluster_set]

    
    # Choose a color palette with good contrast
    palette = sns.color_palette("colorblind", n_colors=k_value)

    # Create the plot
    plt.style.use("fivethirtyeight")
    
    plt.figure(figsize=(40, 32))
    scatter = sns.scatterplot(
        x='HDI', y='JitteredY', 
        hue=df[cluster_set], 
        palette=palette,  # Assign distinct colors
        data=df,
        edgecolor='w', 
        linewidth=0.5,
        s = 450,
        alpha=0.6,
        legend='full'
    )
    plt.title(title, fontsize = 60, pad = 50, loc = 'left')
    plt.xlabel('HDI', fontsize = 50, labelpad = 30)
    plt.ylabel('IHDI Overall loss (%)', fontsize = 50, labelpad = 30)
    
    # Adjust font size of the tick labels
    plt.tick_params(axis='both', which='major', labelsize=40)  # Adjust font size for x and y axes


    # Annotate each point with its country label if the column exists
    for i, point in df.iterrows():
        # Apply jitter function to the y-coordinate
        plt.text(x = point['HDI'] + 0.002, y = point['JitteredY'], s = str(point['ShortName']),
                 fontsize=40)  # Adjust label size for better readability
            
    plt.legend(title='Cluster', title_fontsize = 40, fontsize = 30, markerscale = 3,
               frameon=True, facecolor='#FDF6E3')
    
    # Define the path to save the file, creating directories if necessary
    plt.savefig(f'{directory}plot {cluster_set}.png')  # Save the figure

    plt.show()


# Plot the pre-existing definitions
for s in schemes:
    cluster_plot(df, cluster_set = s)

# Notes:
"""
As you can see, the Cold War division does split the countries quite cleanly along HDI. However, this misses
out large variation along inequality (many post-communist countries are more equal than several countries which 
                                      did not have a communist history.)
"""


# Clustering
## Standardise data
# Importing packages
from sklearn.preprocessing import StandardScaler #for scaling the features
from sklearn.cluster import KMeans # For running the Kmeans algorithm
import matplotlib.pyplot as plt
from kneed import KneeLocator # identify elbow points automatically
from sklearn.metrics import silhouette_score # For finding optimum amount of clusters

# Scaling: Standardise data to m= 0, sd = 1
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df.iloc[:,[3,5]])
df.insert(6,'HDI_scaled',scaled_features[:,0])
df.insert(7,'Inequality-loss_scaled',scaled_features[:,1])


## Choose K (by measuring SSE and silhouette scores per k)
# Create the KMeans estimator objcet
kmeans = KMeans (
    init = "random",
    n_clusters = 3,
    n_init = 10,
    max_iter = 300,
    random_state = 42
)

kmeans_kwargs = { #setting the fixed parameters for the k-means tests
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42
    }


# Initialising arrays to hold evaluation scores
sse = [] # list to hold SSE values for each k (elbow plot)
sl_scores = [] # List holding silhouette coefficient for each cluster (average of score of datapoints in each cluster)



# Running the K-means tests
for k in range(1,11):
    kmeans = KMeans(n_clusters = k, **kmeans_kwargs)
    kmeans.fit(df.iloc[:,6:8])
    # Collecting SSE
    sse.append(kmeans.inertia_)
    if k > 1: # Only collect silhouette scores from 2 clusters onwards...
        # Collecting silhouette scores
        score = silhouette_score(df.iloc[:,6:8], kmeans.labels_)
        sl_scores.append(score)
    else: # If k is 1 then do not collect silhouette score
        continue
   

# Finding elbow point programmatically
kl = KneeLocator(
    range(1,11),
    sse, # input the see list we generated already
    curve ="convex", 
    direction = "decreasing")

print('Elbow point from KneeLocator:',kl.elbow)

# Locating K with highest silhouette score
sl_max = sl_scores.index(max(sl_scores)) + 2 # Add 2 to reflect index location and 
print("K with highest silhouette score: ", sl_max) 


# Elbow plot
#plt.style.use("fivethirtyeight")
plt.figure(figsize = (40,32))
scatter = sns.lineplot(
        x=range(1,11), y=sse, 
        linewidth=10)

#plt.plot(range(1,11), sse, linewidth = 5)
plt.xticks(range(1,11))
plt.title('Elbow plot for K-means clustering of European countries', fontsize = 60, pad = 50)
plt.xlabel('Number of clusters', fontsize = 50, labelpad = 30)
plt.ylabel('SSE', fontsize = 50, labelpad = 30)
plt.axvline(x = kl.elbow, color = "orange", linestyle = "dotted", linewidth = 10) # adding line where elbow point creases
# Adjust font size of the tick labels
plt.tick_params(axis='both', which='major', labelsize=30)  # Adjust font size for x and y axes
plt.savefig(f'{directory}elbow plot.png')  # Save the figure
plt.show()


# Plotting the silhouette scores
plt.figure(figsize = (40,32))
scatter = sns.lineplot(
        x=range(2,11), y=sl_scores, 
        linewidth=10)
plt.xticks(range(2,11))
plt.title('Silhouette scores for K-means clustering of European countries', fontsize = 60, pad = 50)
plt.xlabel('Number of clusters', fontsize = 50, labelpad = 30)
plt.ylabel('Silhouette Coefficient', fontsize = 50, labelpad = 30)
plt.axvline(x = sl_max, color = "orange", linestyle = "dotted", linewidth = 10) # adding line where elbow point creases
# Adjust font size of the tick labels
plt.tick_params(axis='both', which='major', labelsize=30)  # Adjust font size for x and y axes
plt.savefig(f'{directory}silhouette score plot.png')  # Save the figure
plt.show()


## Compute clusters for each K of: 2, 4, 6

# Running the K-means tests
for k in [2,4,6]:
    kmeans = KMeans(n_clusters = k, **kmeans_kwargs)
    kmeans.fit(df.iloc[:,6:8])

    # Dynamically create a column name based on value of k
    column_name = f'k{k}'
    df[column_name] = kmeans.labels_  # Append labels as a new column
    cluster_plot(df, column_name)
