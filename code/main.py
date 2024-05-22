# -*- coding: utf-8 -*-
"""
Created on Tue May  7 20:30:21 2024

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from IPython.display import display




# Import and clean the data
df = (
    pd.read_csv('data/IHDI 2022 dataset Europe.csv')   # Import dataframe
    .query("`Inequality-loss` != '..'")          # Filter out rows where 'Inequality-loss' is '..'
    .query("`PopBelow100k` != 'y'")              # Further filter out rows where 'PopBelow100k' is 'y'
)



## View
df.head()
print(df.dtypes) # We see that "Inequality-loss" column needs to be converted into float
df['Inequality-loss'] = df['Inequality-loss'].astype(float) # Convert to float

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
    palette = sns.color_palette("tab10", n_colors=k_value)

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
    plt.title(title, fontsize = 60, pad = 50, loc = 'left', fontweight = 'Semibold')
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
               frameon=True, facecolor='#FCFCFC')
    
    # Define the path to save the file, creating directories if necessary
    plt.savefig('viz/plot {cluster_set}.png')  # Save the figure

    plt.show()


# Plot the pre-existing definitions
for s in schemes:
    cluster_plot(df, cluster_set = s)



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
plt.savefig('viz/elbow plot.png')  # Save the figure
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
plt.savefig('viz/silhouette score plot.png')  # Save the figure
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
    
# Export European countries table with clustering results
df.to_csv('data/european clusters.csv', index=False)


#### Compare K-means clusters to pre-existing regional divsions
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score

### Remap pre-exisitng divisions to cluster labels
ColdWar_remap = {'Western': 1,
            'Eastern': 0}

UN_Geoscheme_remap = {'Northern': 2,
                      'Western': 1,
                      'Eastern': 0,
                      'Southern': 3}

EuroVoc_remap = {'Northern': 3,
                 'Western': 1,
                 'Central and Eastern': 0,
                 'Southern': 2}

Personal_remap = {'Western': 1,
                  'Southern': 5,
                  'South-East': 4,
                  'Northern': 3,
                  'North-East': 0,
                  'Central-East': 2}

remap_list = [UN_Geoscheme_remap, EuroVoc_remap, ColdWar_remap, Personal_remap]

df_remap = df

for i in range(0,4):
    print(i)
    df_remap.iloc[:,i+8] = np.array([remap_list[i][label] for label in df_remap.iloc[:,i+8]])
    

### Comparing each K-means cluster set to pre-existing definitions

# Creating a dictionary to correspond the pre-existing definitions with the cluster sets
comparison_dict = {
    'UN_Geoscheme': 'k4',
    'EuroVoc': 'k4',
    'Cold_War': 'k2',
    'Personal': 'k6'
    }

# Initialise a list to store results (to convert later into table)
comparison_results = []

# Calculate metrics for each definition-cluster set pairing
for i in comparison_dict:
    definition = df_remap[comparison_dict[i]]
    cluster_set = df_remap[i]
    ari = adjusted_rand_score(definition,cluster_set) # Calculate the Adjusted Rand Index for the pairing
    homogeneity = homogeneity_score(definition,cluster_set) # Measures if each cluster contains only members of a single class.
    completeness = completeness_score(definition,cluster_set) # Measures if all members of a given class are assigned to the same cluster.
    v_measure = v_measure_score(definition,cluster_set) # Harmonic mean of homogeneity and completeness.
    
    comparison_results.append({
        'definition': i, 'cluster_set' : comparison_dict[i],
        'ari':ari, 'homogeneity': homogeneity, 'completeness': completeness, 'v_measure': v_measure})
    
comparison_results_df = pd.DataFrame(comparison_results)
display(comparison_results_df)


