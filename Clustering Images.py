# -*- coding: utf-8 -*-
"""
@author: kholm
Dataset: MNIST digits
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from sklearn.cluster import KMeans




df = pd.read_csv(r"train.csv")

print(df.head())

plt.figure()
plt.imshow(np.array(df.iloc[10,1:]).reshape(28,28),cmap="gray")


dfdata_x = df.drop(["label"],axis=1)

# we know there are 10 numbers
mod = KMeans(n_clusters=10,max_iter=100).fit(dfdata_x)


#look at centers
fig, ax = plt.subplots()
for cen in range(0,len(mod.cluster_centers_)):
    plt.subplot(2,5,cen+1)
    plt.imshow(mod.cluster_centers_[cen].reshape(28,28),cmap="gray")