# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 15:35:50 2023
K-means
dataset: Iris dataset (built in sklearn), driver details
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn import datasets
from sklearn.model_selection import ParameterGrid

"""
-----------------------------------
K-means on Iris Dataset
-----------------------------------
"""


data = datasets.load_iris()

df = pd.DataFrame(data=data.data,columns=data.feature_names )
df["target"] = data.target


#EDA
print(df.head())
print(df.describe())

plt.figure()
df["target"].value_counts().plot(kind='bar')
plt.xlabel("Class")
plt.ylabel("Count")

plt.figure()
sns.pairplot(df.drop(["target"],axis=1))


#cluster
mod = KMeans(n_clusters=3)
mod.fit(df.drop(["target"],axis=1))

df["label"] = mod.labels_

#re plot with clusters
plt.figure()
sns.pairplot(df.drop(["target"],axis=1),hue="label")




"""
-----------------------------------
DBSCAN/K-means clustering hyperparameter tuning
-----------------------------------
"""


df2 = pd.read_csv(r"driver_Details.csv")

print(df2.head())
#drop ID field (not actual data)
df2data = df2.drop(["Driver_ID"],axis=1)


#visualize data (only 2 features)

plt.figure()
sns.scatterplot(df2data,x="Distance_Feature",y="Speeding_Feature")
plt.title("Distance vs Speed for Driver")
plt.xlabel("Distance")
plt.ylabel("Speed")



#use kmeans
k = [1,2,3,4,5,6,10,20]
wcss = []
silscore = [np.nan]
for i in k:
    #cluster
    mod = KMeans(n_clusters=i)
    mod.fit(df2data)
    wcss.append(mod.inertia_)
    if i != 1:
        silscore.append(silhouette_score(df2data,mod.labels_))

#use elbow method for k means
plt.figure()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

sns.lineplot(x=k,y=wcss,ax=ax1)
ax1.set_xlabel("Clusters")
ax1.set_ylabel("WCSS")
plt.title("Errors vs Number of Clusters for K-means")
plt.xticks(k)
ax2.set_ylabel("Silhouette Score",color="g")
sns.lineplot(x=k,y=silscore,ax=ax2,color="g")

#Elbow method suggests using 2 cluster
mod = KMeans(n_clusters=2)
mod.fit(df2data)
df2data["Label"] = mod.labels_

plt.figure()
sns.scatterplot(df2data,x="Distance_Feature",y="Speeding_Feature",hue="Label")
plt.title("Distance vs Speed for Driver using K-Means")
plt.xlabel("Distance")
plt.ylabel("Speed")

df2data.drop(["Label"],axis=1,inplace=True)

#use DBSCAN

#hyperparams
param_grid = {'eps': [0.5,1,5,10,20], 'min_samples': [5, 10,20]}
grd = list(ParameterGrid(param_grid))
silscore = -1

for i in grd:   
    print(i)
    mod = DBSCAN(eps=i["eps"], min_samples=i["min_samples"]).fit(df2data)    
    if silscore < silhouette_score(df2data,mod.labels_):
        silcore = silhouette_score(df2data,mod.labels_)
        bestparam = i
        

#get best model        
mod = DBSCAN(eps=bestparam["eps"], min_samples=bestparam["min_samples"]).fit(df2data)   

df2data["Label"] = mod.labels_
    
plt.figure()
sns.scatterplot(df2data,x="Distance_Feature",y="Speeding_Feature",hue="Label")
plt.title("Distance vs Speed for Driver Using DBSCAN eps = " + str(bestparam["eps"]) + ", Min Sample = " + str(bestparam["min_samples"]))
plt.xlabel("Distance")
plt.ylabel("Speed")
