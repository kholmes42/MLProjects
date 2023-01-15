# -*- coding: utf-8 -*-
"""
Outlier/Novelty Detection
dataset: course data
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""
OUTLIER SECTION (data in training set)
"""

df = pd.read_csv(r"student_performance.csv")
df.drop("Outliers",axis=1,inplace=True)
print(df.head())
print(df.info())
print(df.describe())
sns.set_palette("bright")

fig,ax = plt.subplots(1,4,figsize=(20,8))
sns.scatterplot(data=df,x="Hours Studied",y="Score Obtained",ax=ax[0])
ax[0].set_title("Score vs Study Time Data")


palette = ["red", "blue"]

"""
-------------------------
Use Local Outlier Detection (LOD) (K nearest neighbors global dist vs nearest dist)
-------------------------
"""

from sklearn.neighbors import LocalOutlierFactor



lod = LocalOutlierFactor(n_neighbors=20,contamination=0.2)
lod.fit_predict(df)

outlies = lod.fit_predict(df)

sns.scatterplot(x=df["Hours Studied"],y=df["Score Obtained"],hue=outlies,ax=ax[1],palette=palette)
ax[1].set_title("Outliers using Local Outlier Factor")



"""
-------------------------
Use Isolation Forest (less splits mean data point is more likley to be outlier)
-------------------------
"""

from sklearn.ensemble import IsolationForest

isfore = IsolationForest(contamination=0.19)

outlies = isfore.fit_predict(df)
sns.scatterplot(x=df["Hours Studied"],y=df["Score Obtained"],hue=outlies,ax=ax[2] ,palette=palette)
ax[2].set_title("Outliers using Isolation Forest")


"""
-------------------------
Use Elliptic Envelope (assume data Gaussian distributed)
-------------------------
"""
from sklearn.covariance import EllipticEnvelope

ee = EllipticEnvelope(support_fraction=1,contamination = 0.19)
outlies = ee.fit_predict(df)
sns.scatterplot(x=df["Hours Studied"],y=df["Score Obtained"],hue=outlies,ax=ax[3],palette=palette)
ax[3].set_title("Outliers using Eliliptic Envelope")




"""
Head brain data
"""

df = pd.read_csv(r"headbrain.csv")
print(df.head())

ee = EllipticEnvelope(support_fraction=1,contamination = 0.19)

y_pred = ee.fit_predict(df)

plt.figure()
sns.scatterplot(data=df,x="Head Size(cm^3)",y="Brain Weight(grams)",hue=y_pred,palette=palette)
plt.title("Data w Outliers from Elliptic Envelope")








"""
NOVELTY SECTION (data in testing/production set)
"""

df = pd.read_csv(r"student_performance_modified.csv")
print(df.head())

dftrain = df[df["Training"] == 1].drop(["Outliers","Training","Test"],axis=1)
dftest = df[df["Test"] == 1].drop(["Outliers","Training","Test"],axis=1)



fig,ax = plt.subplots(1,2)
sns.scatterplot(data=dftrain,x="Hours Studied",y="Score Obtained",ax=ax[0])
ax[0].set_title("Training Data")

# sns.scatterplot(data=dftest,x="Hours Studied",y="Score Obtained",ax=ax[1])
# ax[1].set_title("Testing Data")


lod = LocalOutlierFactor(n_neighbors=5,novelty=True,contamination=0.1)
lod.fit(dftrain)
y_pred = lod.predict(dftest)

sns.scatterplot(data=dftest,x="Hours Studied",y="Score Obtained",ax=ax[1],hue=y_pred,palette=palette)
ax[1].set_title("Testing Data w Outliers from LOF")

