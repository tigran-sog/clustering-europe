# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:00:52 2024

@author: user
"""

import numpy as np
import pandas as pd
import seaborn as sns
import math
import matplotlib.pyplot as plt
from scipy.stats import gmean
from scipy.stats import spearmanr


#### Generating income distributions


#### Downloading existing income distributions
#/ From Milanovic et al 2013, annual income per decile 1998-2008

df = pd.read_stata('data/milanovic.dta')


## Filter for only the largest 'year' values for each 'country'
# This gets me a bunch of years around 2008
df = df[df['year'] == df.groupby('country')['year'].transform('max')]

# Remove any countries with at least one NAN in their income distributions
df = df.groupby('country').filter(lambda x: x['RRinc'].notna().all())


# Countries I want to look at
countries = ['Nigeria','Brazil','Italy','United States','Sweden','Armenia']
df_subset = df[df['country'].isin(countries)]


######### CODING ATKINSON INDEX IN PYTHON ##########
income_dist = df[df['country'] == 'Armenia'].iloc[:,10]


def atkinson_index(income_dist, aversion_parameter):    
    try:
        E = float(aversion_parameter)
    except ValueError:
        print("The input is not a valid number.")
    
    if E == 1:
        geometric_mean = gmean(income_dist)
        arithmetic_mean = np.mean(income_dist)
        atkinson = 1 - geometric_mean / arithmetic_mean
        return atkinson
    
    
    elif E < 0:
        print('Please specify a value above 0.')
        return atkinson_index(income_dist) # Restarts the function
    
    else: #if E is 0 <= and not 1
        atkinson_exponent = 1 - E
        
        generalised_mean = math.pow(np.sum(np.power(income_dist, atkinson_exponent)) / len(income_dist), 1 / atkinson_exponent)
        arithmetic_mean = np.mean(income_dist)
        atkinson = 1 - generalised_mean / arithmetic_mean
        
        return atkinson
    
atkinson = atkinson_index(income_dist,1)
print(atkinson)


######### CODING GINI COEFFICIENT IN PYTHON ##########

def gini_coefficient(income_dist):
    # Create an array of order rankings (1-based index)
    order_ranking = np.arange(1, len(income_dist) + 1)
    
    # Multiply each element by its order ranking
    result = income_dist * order_ranking
    
    # Calculate Gini coefficent
    gini = (2*np.sum(result)) / (len(income_dist)*np.sum(income_dist)) - ((len(income_dist) + 1) / len(income_dist))
    
    return gini

gini = gini_coefficient(income_dist)
print(gini)


######### COMPARING INEQUALITY METRICS ###########
###### Atkinson vs. Gini #####
#### Correlation between Atkinson and Gini values for all countries, varying Atkinson by aversion parameter

atkinson_results = []
atkinson_parameter_range = np.arange(0,2.01,.01)

for country in df['country'].unique():
    income_distribution = df[df['country'] == country]['RRinc'].values
    # Calculate Atkinson indices
    for parameter in atkinson_parameter_range:  # Vary parameter from 0 to 2 by 0.01 steps
        inequality = atkinson_index(income_distribution, parameter)
        atkinson_results.append({'country': country, 'metric': 'Atkinson', 'inequality': inequality, 'parameter': parameter})
        
gini_results = []
for country in df['country'].unique():       
    income_distribution = df[df['country'] == country]['RRinc'].values
    # Calculate Gini coefficient
    gini = gini_coefficient(income_distribution)
    gini_results.append({'country': country, 'metric': 'Gini', 'inequality': gini})



# Convert results to DataFrame and display
atkinson_results_df = pd.DataFrame(atkinson_results)
gini_results_df = pd.DataFrame(gini_results)

corr_results = []

for parameter in atkinson_parameter_range:
    corr = spearmanr(atkinson_results_df[atkinson_results_df['parameter'] == parameter].iloc[:,2],
              gini_results_df.iloc[:,2])[0]
    corr_results.append({'coefficient':corr})
corr_results_df = pd.DataFrame(corr_results)

# Plot correlation
plt.figure(figsize=(15, 5))
plt.plot(atkinson_parameter_range[1:], corr_results_df['coefficient'][1:], label='Spearman coefficient')
plt.title('Correlation between Atkinson and Gini')
plt.xlabel('Aversion parameter')
plt.ylabel('Correlation between Atkinson and Gini')
plt.grid(True)
plt.legend()
plt.show()
