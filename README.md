# clustering-europe

![Image: Maps of European countries coloured by my k-means clustering results](viz/cluster_maps.png)

Categorising Europe into regional divisions is as much determined by economic history as it is by geography. Is Greece, a country further east than the Czech Republic, in Eastern Europe? Popular conceptions rooted in Cold War history would say it is in Western Europe.

How far can social and economic factors go in producing a coherent regional classification of European countries?

**This project use K-means clustering to categorise European countries based on the distribution of two variables, human development and inequality.** 

We compare these results to pre-existing definitions of European regional divisions to see how closely our geographically-agnostic unsupervised clustering method aligns.

## Background
The variables used here, human development and inequality, are derived from the UN's inequality-adjusted Human Development Index (IHDI). "Human development" is calculated as an average of three factors: income per capita, life expectancy and years of schooling. Whereas "inequality" uses a form of the Atkinson index to measure inequalities across these three outcomes as a "percentage loss", which is removed from the country's human development score to yield the IHDI.

The Atkinson index is distinguished from other inequality metrics (like the Gini coefficient) in that it assumes a diminising marginal utility. Have a look at **this repository** to explore how the Atkinson index works in more detail. 


## Results
### Determining *k*
![Image: Elbow plot and Silhouette score plot](viz/determining%20k.png)
### European clusters according to optimal *k* = 4
![Image: Scatter plot of European countries by human development and inequality, coloured by k-means clustering results](viz/plot%20k4.png)

### Comparing to pre-existing regional divisions of Europe 
![Image: Maps of European countires coloured by pre-existing regional divisions of Europe](viz/division_maps.png)

| definition   | cluster_set   |   ari |   homogeneity |   completeness |   v_measure |
|:-------------|:--------------|------:|--------------:|---------------:|------------:|
| Cold_War     | k = 2            |  0.17 |          0.14 |           0.14 |        0.14 |
| UN_Geoscheme | k = 4            |  0.18 |          0.32 |           0.3  |        0.31 |
| EuroVoc      | k = 4            |  0.16 |          0.38 |           0.39 |        0.39 |
| Personal     | k = 6            |  0.35 |          0.61 |           0.6  |        0.6  |

## Repository
### /code/
- **main.py**  - Code for *k*-means clustering and visualisations 
- **mapping.py** - Mapping of *k*-means clustering results
### /data/
- **IHDI dataset 2022 Europe.csv** - Input dataset containing European countries according to pre-existing regional definitions
- **european clusters.csv** - Output dataset containing results of *k*-means clustering
- **/borders/** - Shape files for global sovereign states (for use in `mapping.py`)