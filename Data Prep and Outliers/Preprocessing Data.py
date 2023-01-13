# -*- coding: utf-8 -*-
"""
Preprocessing Data
dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database, artificial dataset
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv(r"diabetes.csv")

print(df.head())
print(df.info())

print(df.describe())

print(df.columns)

plt.figure()
df.boxplot(column=["Glucose","Age","BMI"])



"""
-------------------------------------
standardization
------------------------------------
"""

from sklearn.preprocessing import StandardScaler,RobustScaler

y = df["Outcome"]
x = df.drop("Outcome",axis=1)


scaler = StandardScaler()

x_scaled = pd.DataFrame(data=scaler.fit_transform(x),columns=df.columns[:-1])

#scaled boxplot
plt.figure()
x_scaled.boxplot(column=["Glucose","Age","BMI"])


scaler_rob = RobustScaler()

x_scaled = pd.DataFrame(data=scaler_rob.fit_transform(x),columns=df.columns[:-1])

#robust scaled boxplot
plt.figure()
x_scaled.boxplot(column=["Glucose","Age","BMI"])



"""
-------------------------------------
Quantile Transformation
------------------------------------
"""

from sklearn.preprocessing import QuantileTransformer

df = pd.read_csv(r"store_visits.csv")
print(df.head())

#bimodal data
fig,ax = plt.subplots(2,2,figsize=(20,12))
fig.suptitle("Revenue and Visits", fontsize=30)

ax[0][0].hist(df["Visits"],bins=40,edgecolor="red")
ax[0][0].set_xlabel("Visits")
ax[0][0].set_ylabel("Count")
ax[0][0].set_title("Raw")

ax[0][1].hist(df["Revenue"],bins=40,edgecolor="red")
ax[0][1].set_xlabel("Revenue")
ax[0][1].set_ylabel("Count")
ax[0][1].set_title("Raw")


trans = QuantileTransformer(output_distribution="normal",n_quantiles=200)
x_trans = pd.DataFrame(data=trans.fit_transform(df[["Visits","Revenue"]]),columns=["visit_tran","rev_tran"])

ax[1][0].hist(x_trans["visit_tran"],bins=40,edgecolor="red")
ax[1][0].set_xlabel("Visits")
ax[1][0].set_ylabel("Count")
ax[1][0].set_title("Transformed")

ax[1][1].hist(x_trans["rev_tran"],bins=40,edgecolor="red")
ax[1][1].set_xlabel("Revenue")
ax[1][1].set_ylabel("Count")
ax[1][1].set_title("Transformed")


